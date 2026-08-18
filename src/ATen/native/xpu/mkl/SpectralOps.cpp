/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Portions of this file are derived from PyTorch
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <ATen/native/Resize.h>
#include <ATen/native/SpectralOpsUtils.h>
#include <ATen/native/xpu/mkl/SpectralOps.h>
#include <ATen/native/xpu/sycl/FFTKernelFunctor.h>
#include <ATen/ops/complex.h>
#include <ATen/ops/empty_strided.h>
#include <ATen/ops/imag.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/real.h>
#include <ATen/ops/zeros_like.h>
#include <comm/SYCLContext.h>
#include <comm/TensorInfo.h>
#include <oneapi/mkl.hpp>

using namespace oneapi::mkl::dft;

namespace at::native::xpu {

namespace impl {

constexpr int64_t mkl_max_ndim = 3;

template <precision prec, domain signal_type, typename scalar_t>
void _mkl_dft(
    const Tensor& input,
    Tensor& output,
    int64_t signal_ndim,
    bool complex_input,
    bool complex_output,
    bool inverse,
    IntArrayRef checked_signal_sizes,
    bool onesided) {
  auto& queue = at::xpu::getCurrentSYCLQueue();
  int64_t batch = checked_signal_sizes[0];
  std::vector<int64_t> mkl_signal_sizes(
      checked_signal_sizes.begin() + 1, checked_signal_sizes.end());

  auto istrides = input.strides();
  auto ostrides = output.strides();

  int64_t idist = istrides[0];
  int64_t odist = ostrides[0];

  std::vector<int64_t> input_strides(
      istrides.cbegin(), istrides.cbegin() + signal_ndim + 1),
      output_strides(ostrides.cbegin(), ostrides.cbegin() + signal_ndim + 1);
  input_strides[0] = 0;
  output_strides[0] = 0;

  auto desc = descriptor<prec, signal_type>(mkl_signal_sizes);
  desc.set_value(config_param::PLACEMENT, config_value::NOT_INPLACE);
  desc.set_value(config_param::NUMBER_OF_TRANSFORMS, batch);

  if (!inverse) {
    desc.set_value(config_param::FWD_DISTANCE, idist);
    desc.set_value(config_param::BWD_DISTANCE, odist);

    desc.set_value(config_param::FWD_STRIDES, input_strides);
    desc.set_value(config_param::BWD_STRIDES, output_strides);
  } else {
    desc.set_value(config_param::FWD_DISTANCE, odist);
    desc.set_value(config_param::BWD_DISTANCE, idist);

    desc.set_value(config_param::FWD_STRIDES, output_strides);
    desc.set_value(config_param::BWD_STRIDES, input_strides);
  }

  desc.set_value(
      oneapi::mkl::dft::config_param::WORKSPACE,
      oneapi::mkl::dft::config_value::WORKSPACE_EXTERNAL);
  desc.commit(queue);

  // Obtain the size of workspace required after commit.
  int64_t workspaceSizeBytes = 0;
  desc.get_value(
      oneapi::mkl::dft::config_param::WORKSPACE_BYTES, &workspaceSizeBytes);

  // Allocate USM workspace and provide it to the descriptor.
  Tensor workspaceBuf = at::empty(
      {(long)(workspaceSizeBytes / sizeof(double))},
      input.options().dtype(at::kDouble),
      std::nullopt);
  desc.set_workspace((double*)workspaceBuf.mutable_data_ptr());

  auto in_data = (scalar_t*)input.const_data_ptr();
  auto out_data = (scalar_t*)output.mutable_data_ptr();

  sycl::event event;
  if (!inverse) {
    event = compute_forward(desc, in_data, out_data);
  } else {
    event = compute_backward(desc, in_data, out_data);
  }
  queue.throw_asynchronous();
}

void _fft_with_size(
    Tensor& output,
    const Tensor& self,
    int64_t signal_ndim,
    bool complex_input,
    bool complex_output,
    bool inverse,
    IntArrayRef checked_signal_sizes,
    bool onesided) {
  Tensor input_ = self;
  // real/imag dimension must aligned when viewed as of complex type

  if (complex_input) {
    const auto strides = input_.strides();
    bool need_contiguous = strides.back() != 1;
    for (int64_t i = 0; !need_contiguous && i <= signal_ndim; i++) {
      need_contiguous |= (strides[i] % 2 != 0);
    }

    if (need_contiguous) {
      input_ = input_.contiguous();
    }
  }

  bool complex_type = inverse ? complex_output : complex_input;

  void (*dft_func)(
      const class at::Tensor&,
      class at::Tensor&,
      int64_t,
      bool,
      bool,
      bool,
      class c10::ArrayRef<int64_t>,
      bool);
  Tensor input = input_;

  if (input.scalar_type() == ScalarType::Float ||
      input.scalar_type() == ScalarType::ComplexFloat) {
    dft_func = complex_type
        ? _mkl_dft<precision::SINGLE, domain::COMPLEX, float>
        : _mkl_dft<precision::SINGLE, domain::REAL, float>;
  } else if (
      input.scalar_type() == ScalarType::Double ||
      input.scalar_type() == ScalarType::ComplexDouble) {
    dft_func = complex_type
        ? _mkl_dft<precision::DOUBLE, domain::COMPLEX, double>
        : _mkl_dft<precision::DOUBLE, domain::REAL, double>;
  } else {
    TORCH_CHECK(false, "MKL FFT doesn't support tensor of type");
  }

  dft_func(
      input,
      output,
      signal_ndim,
      complex_input,
      complex_output,
      inverse,
      checked_signal_sizes,
      onesided);
}

// Execute a general fft operation (can be c2c, onesided r2c or onesided c2r)
Tensor& _exec_fft(
    Tensor& out,
    Tensor self,
    IntArrayRef out_sizes,
    IntArrayRef dim,
    bool onesided,
    bool forward) {
  // oneMKL DFT rejects zero-element batches (NUMBER_OF_TRANSFORMS = 0). An
  // empty batch has nothing to transform, so skip planning and execution and
  // return the output in its expected (empty) shape. Matches CPU MKL and CUDA
  // (pytorch/pytorch#190483).
  if (out.numel() == 0) {
    out.resize_(out_sizes, MemoryFormat::Contiguous);
    return out;
  }
  const auto ndim = self.dim();
  const int64_t signal_ndim = dim.size();
  const auto batch_dims = ndim - signal_ndim;

  // Permute dimensions so batch dimensions come first, and in stride order
  // This maximizes data locality when collapsing to a single batch dimension
  DimVector dim_permute(ndim);
  std::iota(dim_permute.begin(), dim_permute.end(), int64_t{0});

  c10::SmallVector<bool, kDimVectorStaticSize> is_transformed_dim(ndim);
  for (const auto& d : dim) {
    is_transformed_dim[d] = true;
  }

  auto batch_end =
      std::partition(dim_permute.begin(), dim_permute.end(), [&](int64_t d) {
        return !is_transformed_dim[d];
      });

  auto self_strides = self.strides();
  std::sort(dim_permute.begin(), batch_end, [&](int64_t a, int64_t b) {
    return self_strides[a] > self_strides[b];
  });
  std::copy(dim.cbegin(), dim.cend(), batch_end);

  auto input = self.permute(dim_permute);

  // Collapse batch dimensions into a single dimension
  DimVector batched_sizes(signal_ndim + 1);
  batched_sizes[0] = -1;
  std::copy(
      input.sizes().cbegin() + batch_dims,
      input.sizes().cend(),
      batched_sizes.begin() + 1);
  input = input.reshape(batched_sizes);

  const auto in_sizes = input.sizes();
  const auto batch_size = in_sizes[0];
  DimVector signal_size(signal_ndim + 1);
  signal_size[0] = batch_size;

  for (const auto i : c10::irange(signal_ndim)) {
    auto in_size = in_sizes[i + 1];
    auto out_size = out_sizes[dim[i]];
    signal_size[i + 1] = std::max(in_size, out_size);
    TORCH_INTERNAL_ASSERT(
        in_size == signal_size[i + 1] ||
        in_size == (signal_size[i + 1] / 2) + 1);
    TORCH_INTERNAL_ASSERT(
        out_size == signal_size[i + 1] ||
        out_size == (signal_size[i + 1] / 2) + 1);
  }

  batched_sizes[0] = batch_size;
  DimVector batched_out_sizes(batched_sizes.begin(), batched_sizes.end());

  for (const auto i : c10::irange(dim.size())) {
    batched_out_sizes[i + 1] = out_sizes[dim[i]];
  }

  out.resize_(batched_out_sizes, MemoryFormat::Contiguous);

  // run the FFT
  _fft_with_size(
      out,
      input,
      signal_ndim,
      input.is_complex(),
      out.is_complex(),
      !forward,
      signal_size,
      onesided);

  // Inplace reshaping to original batch shape and inverting the dimension
  // permutation
  DimVector out_strides(ndim);
  int64_t batch_numel = 1;

  for (int64_t i = batch_dims - 1; i >= 0; --i) {
    out_strides[dim_permute[i]] = batch_numel * out.stride(0);
    batch_numel *= out_sizes[dim_permute[i]];
  }

  for (const auto i : c10::irange(batch_dims, ndim)) {
    out_strides[dim_permute[i]] = out.stride(1 + (i - batch_dims));
  }

  out.as_strided_(out_sizes, out_strides, out.storage_offset());

  return out;
}

// _sort_dims, _dft_scale, _fft_apply_normalization, and promote_fft_input are
// defined in sycl/FFTKernelFunctor.cpp and declared in sycl/FFTKernelFunctor.h

// _exec_fft rewrites the destination metadata via resize_/as_strided_. The
// layout it leaves behind only matches a contiguous destination when the
// batch/signal permutation it computes is the identity, so a caller-provided
// output may only be used directly when this holds.
static bool _exec_fft_preserves_layout(const Tensor& self, IntArrayRef dim) {
  const auto ndim = self.dim();
  const auto signal_ndim = static_cast<int64_t>(dim.size());
  const auto batch_dims = ndim - signal_ndim;

  for (const auto i : c10::irange(signal_ndim)) {
    if (dim[i] != batch_dims + i) {
      return false;
    }
  }

  auto self_strides = self.strides();
  for (const auto i : c10::irange(int64_t{1}, batch_dims)) {
    if (self_strides[i - 1] <= self_strides[i]) {
      return false;
    }
  }
  return true;
}

// Result type of promote_fft_input, without materializing the promoted tensor.
ScalarType promote_fft_dtype(ScalarType dtype) {
  if (dtype == ScalarType::Half || dtype == ScalarType::BFloat16)
    return ScalarType::Float;
  if (dtype == ScalarType::ComplexHalf)
    return ScalarType::ComplexFloat;
  return dtype;
}

} // namespace impl

static void _fft_c2c_mkl_out_impl(
    const Tensor& orig_self,
    IntArrayRef dim,
    int64_t normalization,
    bool forward,
    Tensor& out,
    bool preserve_out_layout) {
  auto self = impl::promote_fft_input(orig_self);

  auto sorted_dims = impl::_sort_dims(self, dim);
  auto out_sizes = self.sizes();
  auto input_sizes = self.sizes();

  const auto pass_count =
      (sorted_dims.size() + impl::mkl_max_ndim - 1) / impl::mkl_max_ndim;
  const bool needs_type_conversion = self.scalar_type() != out.scalar_type();
  // A multi-pass transform hands its last pass the outermost dims, so the
  // permutation is no longer the identity even when the full dim set is.
  const bool can_write_out = !needs_type_conversion && out.is_contiguous() &&
      (!preserve_out_layout ||
       (pass_count == 1 &&
        impl::_exec_fft_preserves_layout(self, sorted_dims)));
  Tensor fft_out = can_write_out ? out : at::empty(out_sizes, self.options());

  Tensor scratch;
  if (pass_count > 1) {
    scratch = at::empty(out_sizes, self.options());
  }

  auto working_tensor = self;
  size_t pass = 0;

  while (!sorted_dims.empty()) {
    const auto max_dims =
        std::min(static_cast<size_t>(impl::mkl_max_ndim), sorted_dims.size());
    auto fft_dims =
        IntArrayRef(sorted_dims).slice(sorted_dims.size() - max_dims, max_dims);
    const auto remaining_passes = pass_count - pass;
    Tensor& pass_out = remaining_passes % 2 == 1 ? fft_out : scratch;

    impl::_exec_fft(
        pass_out,
        working_tensor,
        out_sizes,
        fft_dims,
        /*onesided=*/false,
        forward);

    working_tensor = pass_out;
    sorted_dims.resize(sorted_dims.size() - max_dims);
    ++pass;
    if (!sorted_dims.empty()) {
      sorted_dims = impl::_sort_dims(self, sorted_dims);
    }
  }

  impl::_fft_apply_normalization(fft_out, normalization, input_sizes, dim);
  if (!fft_out.is_same(out)) {
    out.copy_(fft_out);
  }
}

Tensor _fft_c2c_mkl(
    const Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    bool forward) {
  if (dim.empty()) {
    return self.clone();
  }

  // Allocate at the transform's own precision so the impl can write straight
  // into out; converting afterwards preserves the layout _exec_fft left behind
  // and keeps the conversion a linear pass.
  const auto fft_dtype = impl::promote_fft_dtype(self.scalar_type());
  auto out = at::empty(self.sizes(), self.options().dtype(fft_dtype));
  _fft_c2c_mkl_out_impl(
      self, dim, normalization, forward, out, /*preserve_out_layout=*/false);
  return out.to(self.scalar_type());
}

Tensor& _fft_c2c_mkl_out(
    const Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    bool forward,
    Tensor& out) {
  if (dim.empty() || out.is_alias_of(self)) {
    auto result = _fft_c2c_mkl(self, dim, normalization, forward);
    at::native::resize_output(out, result.sizes());
    out.copy_(result);
    return out;
  }

  at::native::resize_output(out, self.sizes());
  _fft_c2c_mkl_out_impl(
      self, dim, normalization, forward, out, /*preserve_out_layout=*/true);
  return out;
}

void HermitSymmImpl(Tensor& input, int64_t dim, int pos) {
  std::vector<at::indexing::TensorIndex> indices(
      input.dim(), at::indexing::Slice());

  indices[dim] = pos;

  Tensor values = at::complex(
      at::real(input.index(indices)),
      at::zeros_like(at::imag(input.index(indices))));

  input.index_put_(indices, values);
}

void HermitSymm(Tensor& input, int64_t dim, int64_t out_size) {
  HermitSymmImpl(input, dim, 0);

  if (out_size % 2 == 0)
    HermitSymmImpl(input, dim, -1);
}

static DimVector _fft_c2r_out_sizes(
    const Tensor& self,
    IntArrayRef dim,
    int64_t last_dim_size) {
  DimVector out_sizes(self.sizes().begin(), self.sizes().end());
  out_sizes[dim.back()] = last_dim_size;
  return out_sizes;
}

static void _fft_c2r_mkl_out_impl(
    const Tensor& orig_self,
    IntArrayRef dim,
    int64_t normalization,
    int64_t last_dim_size,
    Tensor& out,
    bool preserve_out_layout) {
  auto self = impl::promote_fft_input(orig_self);

  auto input = self;

  if (dim.size() > 1) {
    auto c2c_dims = dim.slice(0, dim.size() - 1);
    input = _fft_c2c_mkl(
        self,
        c2c_dims,
        static_cast<int64_t>(fft_norm_mode::none),
        /*forward=*/false);
  }

  // HermitSymm mutates in-place; avoid mutating user-visible input when
  // aliased.
  if (input.is_same(self)) {
    auto input_copy =
        at::empty_strided(input.sizes(), input.strides(), input.options());
    input_copy.copy_(input);
    input = input_copy;
  }

  auto in_sizes = input.sizes();
  DimVector out_sizes(in_sizes.begin(), in_sizes.end());
  out_sizes[dim.back()] = last_dim_size;
  const bool needs_type_conversion =
      c10::toRealValueType(self.scalar_type()) != out.scalar_type();
  const bool can_write_out = !needs_type_conversion && out.is_contiguous() &&
      (!preserve_out_layout ||
       impl::_exec_fft_preserves_layout(input, dim.back()));
  Tensor fft_out = can_write_out
      ? out
      : at::empty(
            out_sizes,
            self.options().dtype(c10::toRealValueType(self.scalar_type())));

  HermitSymm(input, dim.back(), out_sizes[dim.back()]);

  impl::_exec_fft(
      fft_out,
      input,
      out_sizes,
      dim.back(),
      /*onesided=*/true,
      /*forward=*/false);

  impl::_fft_apply_normalization(fft_out, normalization, out_sizes, dim);
  if (!fft_out.is_same(out)) {
    out.copy_(fft_out);
  }
}

Tensor _fft_c2r_mkl(
    const Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    int64_t last_dim_size) {
  if (dim.empty()) {
    return self.clone();
  }

  auto out_sizes = _fft_c2r_out_sizes(self, dim, last_dim_size);
  const auto fft_dtype =
      c10::toRealValueType(impl::promote_fft_dtype(self.scalar_type()));
  auto out = at::empty(out_sizes, self.options().dtype(fft_dtype));
  _fft_c2r_mkl_out_impl(
      self,
      dim,
      normalization,
      last_dim_size,
      out,
      /*preserve_out_layout=*/false);
  return out.to(c10::toRealValueType(self.scalar_type()));
}

Tensor& _fft_c2r_mkl_out(
    const Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    int64_t last_dim_size,
    Tensor& out) {
  if (dim.empty() || out.is_alias_of(self)) {
    auto result = _fft_c2r_mkl(self, dim, normalization, last_dim_size);
    at::native::resize_output(out, result.sizes());
    out.copy_(result);
    return out;
  }

  auto out_sizes = _fft_c2r_out_sizes(self, dim, last_dim_size);
  at::native::resize_output(out, out_sizes);
  _fft_c2r_mkl_out_impl(
      self,
      dim,
      normalization,
      last_dim_size,
      out,
      /*preserve_out_layout=*/true);
  return out;
}

REGISTER_XPU_DISPATCH(
    fft_fill_with_conjugate_symmetry_stub,
    &_fft_fill_with_conjugate_symmetry_xpu);

static DimVector _fft_r2c_out_sizes(
    const Tensor& self,
    IntArrayRef dim,
    bool onesided) {
  DimVector out_sizes(self.sizes().begin(), self.sizes().end());
  if (onesided) {
    auto last_dim = dim.back();
    out_sizes[last_dim] = self.size(last_dim) / 2 + 1;
  }
  return out_sizes;
}

static void _fft_r2c_mkl_out_impl(
    const Tensor& orig_self,
    IntArrayRef dim,
    int64_t normalization,
    bool onesided,
    Tensor& out,
    bool preserve_out_layout) {
  TORCH_INTERNAL_ASSERT(!dim.empty());
  auto self = impl::promote_fft_input(orig_self);

  auto input_sizes = self.sizes();
  auto last_dim = dim.back();
  auto last_dim_halfsize = (input_sizes[last_dim]) / 2 + 1;
  auto out_sizes = _fft_r2c_out_sizes(self, dim, onesided);
  const auto fft_dtype = c10::toComplexType(self.scalar_type());

  const auto c2c_pass_count =
      (dim.size() - 1 + impl::mkl_max_ndim - 1) / impl::mkl_max_ndim;
  const auto pass_count = 1 + c2c_pass_count;

  const bool needs_type_conversion = fft_dtype != out.scalar_type();
  // The transform runs on a contiguous copy of self, so _exec_fft leaves a
  // contiguous layout only when the single pass acts on the innermost dim.
  const bool can_write_out = !needs_type_conversion && out.is_contiguous() &&
      (!preserve_out_layout || (pass_count == 1 && last_dim == self.dim() - 1));
  Tensor fft_out = can_write_out
      ? out
      : at::empty(out_sizes, self.options().dtype(fft_dtype));

  Tensor scratch;
  if (pass_count > 1) {
    scratch = at::empty(out_sizes, self.options().dtype(fft_dtype));
  }

  auto working_tensor = self.contiguous();

  // First do the R2C transform on the last dimension
  Tensor& first_pass_out = pass_count % 2 == 1 ? fft_out : scratch;
  impl::_exec_fft(
      first_pass_out,
      working_tensor,
      out_sizes,
      last_dim,
      onesided,
      /*forward=*/true);
  working_tensor = first_pass_out;

  DimVector sorted_dims(dim.begin(), dim.end() - 1);
  size_t pass = 1;

  while (!sorted_dims.empty()) {
    sorted_dims = impl::_sort_dims(self, sorted_dims);

    const auto max_dims =
        std::min(static_cast<size_t>(impl::mkl_max_ndim), sorted_dims.size());
    auto fft_dims =
        IntArrayRef(sorted_dims).slice(sorted_dims.size() - max_dims, max_dims);
    const auto remaining_passes = pass_count - pass;
    Tensor& pass_out = remaining_passes % 2 == 1 ? fft_out : scratch;
    impl::_exec_fft(
        pass_out,
        working_tensor,
        out_sizes,
        fft_dims,
        onesided,
        /*forward=*/true);
    working_tensor = pass_out;
    sorted_dims.resize(sorted_dims.size() - max_dims);
    ++pass;
  }

  // Only need to normalize the onesided slice since data in the other half is
  // overwritten
  auto out_slice = fft_out.slice(last_dim, 0, last_dim_halfsize);
  impl::_fft_apply_normalization(out_slice, normalization, input_sizes, dim);

  if (!onesided) {
    if (fft_out.sizes()[last_dim] != out_sizes[last_dim]) {
      auto full_out = at::empty(out_sizes, self.options().dtype(fft_dtype));
      full_out.slice(last_dim, 0, last_dim_halfsize).copy_(fft_out);
      fft_out = std::move(full_out);
    }
    at::native::_fft_fill_with_conjugate_symmetry_(fft_out, dim);
  }

  if (!fft_out.is_same(out)) {
    out.copy_(fft_out);
  }
}

Tensor _fft_r2c_mkl(
    const Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    bool onesided) {
  if (dim.empty()) {
    return self.clone();
  }

  auto out_sizes = _fft_r2c_out_sizes(self, dim, onesided);
  const auto fft_dtype =
      c10::toComplexType(impl::promote_fft_dtype(self.scalar_type()));
  // Not toComplexType(self.scalar_type()): that maps bfloat16 to complex
  // bfloat16, while the transform runs at float and reports complex float.
  auto out_dtype = self.scalar_type() == ScalarType::Half
      ? ScalarType::ComplexHalf
      : fft_dtype;
  auto out = at::empty(out_sizes, self.options().dtype(fft_dtype));
  _fft_r2c_mkl_out_impl(
      self, dim, normalization, onesided, out, /*preserve_out_layout=*/false);
  return out.to(out_dtype);
}

Tensor& _fft_r2c_mkl_out(
    const Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    bool onesided,
    Tensor& out) {
  if (dim.empty() || out.is_alias_of(self)) {
    auto result = _fft_r2c_mkl(self, dim, normalization, onesided);
    at::native::resize_output(out, result.sizes());
    out.copy_(result);
    return out;
  }

  auto out_sizes = _fft_r2c_out_sizes(self, dim, onesided);
  at::native::resize_output(out, out_sizes);
  _fft_r2c_mkl_out_impl(
      self, dim, normalization, onesided, out, /*preserve_out_layout=*/true);
  return out;
}

} // namespace at::native::xpu
