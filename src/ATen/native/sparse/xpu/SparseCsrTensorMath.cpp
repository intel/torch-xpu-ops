/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/ExpandUtils.h>
#include <ATen/SparseCsrTensorUtils.h>
#include <ATen/TensorOperators.h>
#include <ATen/native/Resize.h>
#include <ATen/native/sparse/SparseStubs.h>
#include <ATen/native/sparse/xpu/SparseCsrTensorMath.h>
#include <ATen/native/sparse/xpu/sycl/SparseCsrTensorMathKernels.h>
#include <ATen/ops/_convert_indices_from_coo_to_csr_native.h>
#include <ATen/ops/_convert_indices_from_csr_to_coo_native.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/add.h>
#include <ATen/ops/addmm.h>
#include <ATen/ops/addmv.h>
#include <ATen/ops/baddbmm.h>
#include <ATen/ops/copy_native.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/scalar_tensor_native.h>
#endif

namespace at::native {

using namespace at::sparse;
using namespace at::sparse_csr;

TORCH_IMPL_FUNC(_convert_indices_from_coo_to_csr_structured_xpu)
(const Tensor& input,
 const int64_t size,
 const bool out_int32,
 const Tensor& result) {
  xpu::convert_indices_from_coo_to_csr_structured_kernel(
      input, size, out_int32, result);
};

TORCH_IMPL_FUNC(_convert_indices_from_csr_to_coo_structured_xpu)
(const Tensor& crow_indices,
 const Tensor& col_indices,
 const bool out_int32,
 const bool transpose,
 const Tensor& result) {
  xpu::convert_indices_from_csr_to_coo_structured_kernel(
      crow_indices, col_indices, out_int32, transpose, result);
};

Tensor _sparse_csr_sum_xpu(
    const Tensor& input,
    IntArrayRef dims_to_sum,
    bool keepdim,
    std::optional<ScalarType> dtype) {
  return xpu::_sparse_csr_sum_xpu_kernel(input, dims_to_sum, keepdim, dtype);
}

Tensor _sparse_csr_prod_xpu(
    const Tensor& input,
    IntArrayRef dims_to_reduce,
    bool keepdim,
    std::optional<ScalarType> dtype) {
  return xpu::_sparse_csr_prod_xpu_kernel(
      input, dims_to_reduce, keepdim, dtype);
}

void addmm_out_sparse_csr(
    const Tensor& input,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& result) {
  TORCH_INTERNAL_ASSERT(
      !((mat1.layout() == kStrided) && (mat2.layout() == kStrided) &&
        (result.layout() == kStrided)),
      "Expected at least one sparse input");

  // Layout checks are nested mat1, mat2, result
  // Conditions are ordered strided, csr, csc, bsr, bsc.
  // Valid combinations terminate in a return
  // Invalid combinations are omitted and will fall though to the TORCH check
  // generating an informative error message

  // mm functions that copy input to result when needed (e.g. mm
  // triton kernels do not require result being initialized with
  // input):
  if (mat1.layout() == kSparseBsr) {
    if (mat2.layout() == kStrided) {
      if (result.layout() == kStrided) {
        at::addmm_out(result, input, mat1.to_dense(), mat2, beta, alpha);
        return;
      }
    }
  }

  if (mat1.layout() == kStrided) {
    if (mat2.layout() == kSparseBsc) {
      if (result.layout() == kStrided) {
        at::addmm_out(result, input, mat1, mat2.to_dense(), beta, alpha);
        return;
      }
    }
  }

  // copy input to result:
  if (beta.toComplexDouble() != 0. && !result.is_same(input)) {
    result.copy_(input);
  }

  // mm functions that assume that result contains input:
  if (mat1.layout() == kStrided) {
    if (mat2.layout() == kSparseCsr) {
      if (result.layout() == kStrided) {
        at::addmm_out(result, input, mat1, mat2.to_dense(), beta, alpha);
        return;
      }
    }
    if (mat2.layout() == kSparseCsc) {
      if (result.layout() == kStrided) {
        at::addmm_out(result, input, mat1, mat2.to_dense(), beta, alpha);
        return;
      }
    }
  }
  if (mat1.layout() == kSparseCsr) {
    if (mat2.layout() == kStrided) {
      if (result.layout() == kStrided) {
        at::addmm_out(result, input, mat1.to_dense(), mat2, beta, alpha);
        return;
      }
    }
    if (mat2.layout() == kSparseCsr) {
      if (result.layout() == kSparseCsr) {
        Tensor result_dense =
            at::addmm(input, mat1.to_dense(), mat2.to_dense(), beta, alpha);
        result = result_dense.to_sparse_csr();
        return;
      }
    }
    if (mat2.layout() == kSparseCsc) {
      if (result.layout() == kSparseCsr) {
        Tensor result_dense =
            at::addmm(input, mat1.to_dense(), mat2.to_dense(), beta, alpha);
        result = result_dense.to_sparse_csr();
        return;
      }
    }
  }
  if (mat1.layout() == kSparseCsc) {
    if (mat2.layout() == kStrided) {
      if (result.layout() == kStrided) {
        at::addmm_out(result, input, mat1.to_dense(), mat2, beta, alpha);
        return;
      }
    }
    if (mat2.layout() == kSparseCsr) {
      if (result.layout() == kSparseCsr) {
        Tensor result_dense =
            at::addmm(input, mat1.to_dense(), mat2.to_dense(), beta, alpha);
        result = result_dense.to_sparse_csr();
        return;
      }
    }
    if (mat2.layout() == kSparseCsc) {
      if (result.layout() == kSparseCsr) {
        Tensor result_dense =
            at::addmm(input, mat1.to_dense(), mat2.to_dense(), beta, alpha);
        result = result_dense.to_sparse_csr();
        return;
      }
      if (result.layout() == kSparseCsc) {
        Tensor result_dense =
            at::addmm(input, mat1.to_dense(), mat2.to_dense(), beta, alpha);
        result = result_dense.to_sparse_csc();
        return;
      }
    }
  }
  TORCH_CHECK(
      false,
      "addmm: computation on XPU is not implemented for ",
      result.layout(),
      " + ",
      mat1.layout(),
      " @ ",
      mat2.layout());
}

// result = beta * self + alpha * (mat1 @ mat2)
Tensor& addmm_out_sparse_compressed_xpu(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& result) {
  sparse::impl::_check_is_xpu(self, "self");
  sparse::impl::_check_is_xpu(mat1, "mat1");
  sparse::impl::_check_is_xpu(mat2, "mat2");
  sparse::impl::_check_is_xpu(result, "result");

  // Same checks as in TORCH_META_FUNC(addmm) at
  // aten/src/ATen/native/LinearAlgebra.cpp
  sparse::impl::_check_dim(mat1, 2, "mat1");
  sparse::impl::_check_dim(mat2, 2, "mat2");

  TORCH_CHECK(
      mat1.size(1) == mat2.size(0),
      "mat1 and mat2 shapes cannot be multiplied (",
      mat1.size(0),
      "x",
      mat1.size(1),
      " and ",
      mat2.sizes()[0],
      "x",
      mat2.sizes()[1],
      ")");

  c10::MaybeOwned<at::Tensor> self_;
  // Don't expand self if this is an in-place operation
  if (&result == &self) {
    self_ = c10::MaybeOwned<Tensor>::borrowed(self);
  } else {
    self_ = expand_size(self, {mat1.size(0), mat2.size(1)}, "addmm");
  }

  sparse::impl::_check_dim(*self_, 2, "self");
  TORCH_CHECK(
      ((self_->dim() == 2) && (self_->size(0) == mat1.size(0)) &&
       (self_->size(1) == mat2.size(1))),
      "The input tensor must be a matrix with size ",
      mat1.size(0),
      "x",
      mat2.size(1),
      ", but got a ",
      self_->dim(),
      "-D tensor with size ",
      self_->size(0),
      "x",
      self_->size(1));

  if (!result.is_same(self)) {
    if (result.layout() == kStrided) {
      at::native::resize_output(result, self_->sizes());
    } else {
      result.resize_as_sparse_(*self_);
    }
  }

  if (result.numel() == 0) {
    return result;
  }

  if (sparse::impl::_is_sparse_and_zero(mat1) ||
      sparse::impl::_is_sparse_and_zero(mat2)) {
    // According to docs, when beta==0 values in self should be ignored.
    // nans and infs should not propagate
    const auto beta_val = beta.toComplexDouble();
    if (beta_val == 0.) {
      result.zero_();
    } else {
      if (!result.is_same(self)) {
        result.copy_(*self_);
      }
      if (beta_val != 1.) {
        result.mul_(beta);
      }
    }
    return result;
  }

  addmm_out_sparse_csr(*self_, mat1, mat2, beta, alpha, result);
  return result;
}

Tensor& addmv_out_sparse_compressed_xpu(
    const Tensor& self,
    const Tensor& mat,
    const Tensor& vec,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& result) {
  if (mat.layout() == kSparseCsc) {
    return addmv_out_sparse_compressed_xpu(
        self, mat.to_sparse_csr(), vec, beta, alpha, result);
  }
  TORCH_CHECK(
      mat.layout() != kSparseBsc,
      "addmm_out_sparse_csr_xpu currently does not support layout SparseBsc for input mat.");

  TORCH_CHECK(mat.dim() == 2, "addmv: Expected mat to be 2-D");
  TORCH_CHECK(vec.dim() == 1, "addmv: Expected vec to be 1-D");

  c10::MaybeOwned<Tensor> self_ = expand_size(self, {mat.size(0)});
  auto betaval = beta.toComplexDouble();

  if (&result != &self) {
    at::native::resize_output(result, self_->sizes());
    if (betaval != 0.0) {
      at::native::copy_(result, *self_);
    }
  }

  if (mat._nnz() == 0) {
    // shortcut for an empty matrix
    // By definition, when beta==0, values in self should be ignored. nans and
    // infs should not propagate
    if (betaval == 0.0) {
      return result.zero_();
    } else {
      return at::mul_out(
          result,
          self,
          at::native::scalar_tensor(
              beta,
              self.scalar_type(),
              std::nullopt /* layout */,
              at::kCPU,
              std::nullopt /* pin_memory */));
    }
  }

  at::addmv_out(result, self, mat.to_dense(), vec, beta, alpha);

  return result;
}

void expand_batch_if_necessary(const Tensor& mat) {
  auto indice_batch_ndim = sparse_csr::numBatchDimensions(mat);
  auto [compressed_indices, plain_indices] =
      sparse_csr::getCompressedPlainIndices(mat);
  auto values = mat.values();
  auto batch_diff_size = mat.sizes().vec();
  auto real_batch_ndim = mat.sizes().size() - 2;
  if (indice_batch_ndim < real_batch_ndim) {
    batch_diff_size.erase(
        batch_diff_size.begin() + (real_batch_ndim - indice_batch_ndim),
        batch_diff_size.end());
    auto reshaped_compressed_indices_shape = compressed_indices.sizes().vec();
    reshaped_compressed_indices_shape.insert(
        std::begin(reshaped_compressed_indices_shape),
        std::begin(batch_diff_size),
        std::end(batch_diff_size));
    compressed_indices =
        compressed_indices.expand(reshaped_compressed_indices_shape);
    auto reshaped_plain_indices_shape = plain_indices.sizes().vec();
    reshaped_plain_indices_shape.insert(
        reshaped_plain_indices_shape.begin(),
        batch_diff_size.begin(),
        batch_diff_size.end());
    plain_indices = plain_indices.expand(reshaped_plain_indices_shape);
    auto reshaped_values_indices_shape = values.sizes().vec();
    reshaped_values_indices_shape.insert(
        reshaped_values_indices_shape.begin(),
        batch_diff_size.begin(),
        batch_diff_size.end());
    values = values.expand(reshaped_values_indices_shape);
  }
  get_sparse_csr_impl(mat)->set_member_tensors(
      compressed_indices, plain_indices, values, mat.sizes());
  return;
}

Tensor& baddbmm_out_sparse_csr_xpu(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& result) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(mat1.is_sparse_csr());

  TORCH_CHECK(
      self.layout() == kStrided,
      "torch.baddbmm: Expected self to be strided, but got layout ",
      self.layout());
  TORCH_CHECK(
      mat2.layout() == kStrided,
      "torch.baddbmm: Expect mat2 to be strided, but got ",
      mat2.layout());
  TORCH_CHECK(
      result.layout() == kStrided,
      "torch.baddbmm: Expect result to be strided, but got ",
      result.layout());

  if (!result.is_same(self)) {
    at::native::resize_output(result, self.sizes());
  }

  if (mat1._nnz() == 0) {
    // According to docs, when beta==0 values in self should be ignored
    // nans and infs should not propagate
    if (beta.toComplexDouble() == 0.) {
      result.zero_();
    } else {
      if (!result.is_same(self)) {
        result.copy_(self);
      }
      if (beta.toComplexDouble() != 1.) {
        result.mul_(beta);
      }
    }
    return result;
  }

  // broadcast batch of sparse indices and values if not compatible with sizes
  // before to_dense() to_dense issue:
  // https://github.com/intel/torch-xpu-ops/issues/2801
  expand_batch_if_necessary(mat1);

  at::baddbmm_out(result, self, mat1.to_dense(), mat2, beta, alpha);
  return result;
}

Tensor& bmm_out_sparse_csr_xpu(
    const Tensor& mat1,
    const Tensor& mat2,
    Tensor& result) {
  Scalar beta(0.0);
  Scalar alpha(1.0);
  return at::native::baddbmm_out_sparse_csr_xpu(
      result, mat1, mat2, beta, alpha, result);
}

Tensor& add_out_sparse_compressed_xpu(
    const Tensor& self,
    const SparseCsrTensor& other,
    const Scalar& alpha,
    SparseCsrTensor& out) {
  if (self.layout() == kStrided) {
    at::add_out(out, self, other.to_dense(), alpha);
    return out;
  } else if (other.layout() == kStrided) {
    at::add_out(out, other, self.to_dense(), alpha);
    return out;
  } else {
    TORCH_CHECK(
        self.sizes().equals(other.sizes()),
        "torch.add: Expected input tensors to have the same shape, but got tensor `self` with shape ",
        self.sizes(),
        " and tensor `other` with shape ",
        other.sizes());
    TORCH_CHECK(
        self.is_xpu(),
        "add: expected 'self' to be XPU tensor, but got tensor on device: ",
        self.device());
    TORCH_CHECK(
        other.is_xpu(),
        "add: expected 'other' to be XPU tensor, but got tensor on device: ",
        other.device());
    TORCH_CHECK(
        out.is_xpu(),
        "add: expected 'out' to be XPU tensor, but got tensor on device: ",
        out.device());

    if (only_sparse_compressed_add_trivial_cases(self, other, alpha, out)) {
      return out;
    }

    Tensor out_dense = at::add(self.to_dense(), other.to_dense(), alpha);
    out = out_dense.to_sparse_csr();
  }
  return out;
}

} // namespace at::native
