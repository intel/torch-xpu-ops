/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

// #define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch_v2.h>
#include <ATen/WrapDimUtilsMulti.h>
#include <ATen/native/xpu/sycl/MemoryAccess.h>
#include <ATen/native/xpu/sycl/OffsetCalculator.h>
#include <comm/SYCLContext.h>
#include <comm/xpu_aten.h>

#include <ATen/native/xpu/sycl/TensorTransformationsKernels.h>

namespace at::native::xpu {

template <int N>
struct alignas(N) OpaqueType {
  char data[N];
};

template <typename func_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void elementwise_sub_kernel(
    int loops,
    int total_n_elems,
    func_t f,
    int total_work_items) {
  auto item = syclext::this_work_item::get_nd_item<1>();
  int idx = item.get_global_linear_id();

  for (int i = 0; i < loops; ++i) {
    if (idx < total_n_elems) {
      f(idx);
      idx += total_work_items;
    }
  }
}

template <typename func_t>
void elementwise_kernel(int total_n_elems, func_t f) {
  constexpr auto kfn = elementwise_sub_kernel<func_t>;

  auto& queue = getCurrentSYCLQueue();
  int64_t max_wg_size = syclMaxWorkGroupSize<kfn>();
  const auto target_global_size = syclMaxWorkItemsPerTile();
  int work_group_size =
      total_n_elems > max_wg_size ? max_wg_size : total_n_elems;
  const int max_work_group_num = target_global_size / work_group_size;
  int total_group_num = (total_n_elems + work_group_size - 1) / work_group_size;
  int work_group_num = total_group_num < max_work_group_num
      ? total_group_num
      : max_work_group_num;
  // work item in each work group calculates loops' elements
  int loops = total_group_num / work_group_num + 1;

  int total_work_items = work_group_size * work_group_num;

  sycl_kernel_submit<kfn>(
      sycl::range<1>(total_work_items),
      sycl::range<1>(work_group_size),
      queue,
      0,
      loops,
      total_n_elems,
      f,
      total_work_items);
}

template <typename func_t>
static void launch_kernel(int total_n_elems, func_t f) {
  TORCH_INTERNAL_ASSERT(
      total_n_elems >= 0 &&
      total_n_elems <= std::numeric_limits<int32_t>::max());
  elementwise_kernel<func_t>(total_n_elems, f);
}

template <typename scalar_t, typename offset_calc_t>
struct FlipKernelImplLoopFunctor {
  void operator()(const int i) const {
    const auto offsets = offset_calc.get(i);
    // offsets can be negative here, but it's fine
    scalar_t* const RESTRICT out_data =
        reinterpret_cast<scalar_t*>(out_ptr + offsets[0]);
    const scalar_t* const RESTRICT in_data =
        reinterpret_cast<const scalar_t*>(in_ptr + offsets[1]);
    *out_data = *in_data;
  }

  FlipKernelImplLoopFunctor(
      char* const RESTRICT out_ptr,
      const char* const RESTRICT in_ptr,
      const offset_calc_t offset_calc)
      : out_ptr(out_ptr), in_ptr(in_ptr), offset_calc(offset_calc) {}

  FlipKernelImplLoopFunctor& operator=(const FlipKernelImplLoopFunctor&) =
      delete;

 private:
  char* const RESTRICT out_ptr;
  const char* const RESTRICT in_ptr;
  const offset_calc_t offset_calc;
};

template <typename scalar_t>
void flip_kernel_impl(TensorIterator& iter) {
  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      flip_kernel_impl<scalar_t>(sub_iter);
    }
    return;
  }

  char* const RESTRICT out_ptr = reinterpret_cast<char*>(iter.data_ptr(0));
  const char* const RESTRICT in_ptr =
      reinterpret_cast<const char*>(iter.data_ptr(1));

  const auto offset_calc =
      make_offset_calculator<2, /*signed_strides=*/true>(iter);

  FlipKernelImplLoopFunctor<scalar_t, decltype(offset_calc)> loop(
      out_ptr, in_ptr, offset_calc);
  launch_kernel(iter.numel(), loop);
}

void flip_kernel(TensorIterator& iter, bool quantized) {
  if (quantized) {
    TORCH_CHECK(false, "XPU current does not flip for quantized tensor");
  }
  AT_DISPATCH_V2(
      iter.dtype(),
      "flip_xpu",
      AT_WRAP([&] {
        using dtype = OpaqueType<sizeof(scalar_t)>;
        flip_kernel_impl<dtype>(iter);
      }),
      AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX),
      AT_EXPAND(AT_FLOAT8_TYPES),
      AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES),
      kComplexHalf,
      kBComplex32,
      kHalf,
      kBool,
      kBFloat16);
}

template <typename scalar_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void roll_kernel(
    const scalar_t* in_data,
    scalar_t* out_data,
    int val_of_work_item,
    int64_t N,
    int64_t total_offset,
    int64_t stride,
    int64_t shift,
    int64_t offset,
    int64_t start_offset,
    int global_range) {
  auto item = syclext::this_work_item::get_nd_item<1>();
  int64_t linear_index = item.get_global_id(0);
  for (int i = 0; i < val_of_work_item; i++) {
    if (linear_index < N) {
      // roll dim idx is the index of linear_index along the rolling
      // dimension.
      int64_t roll_dim_idx = linear_index % (total_offset) / stride;
      // index into the source data to find appropriate value.
      int64_t source_idx = 0;
      source_idx = roll_dim_idx >= shift ? linear_index - offset
                                         : linear_index + start_offset;
      out_data[linear_index] = in_data[source_idx];
      linear_index += global_range;
    }
  }
}

template <typename scalar_t>
void roll_template(
    const Tensor& in_tensor,
    Tensor& out_tensor,
    int64_t N,
    int64_t roll_dim,
    int64_t start,
    int64_t size,
    int64_t stride,
    int64_t total_dims) {
  constexpr auto kfn = roll_kernel<scalar_t>;

  auto shift = size - start;
  auto offset = shift * stride;
  auto start_offset = start * stride;
  auto total_offset = size * stride;

  auto local_range = syclMaxWorkGroupSize<kfn>();
  const auto target_global_range =
      syclMaxWorkItemsPerTile() / local_range * local_range;
  int global_range = (N + local_range - 1) / local_range * local_range;
  auto val_of_work_item =
      (global_range + target_global_range - 1) / target_global_range;
  global_range =
      global_range < target_global_range ? global_range : target_global_range;

  auto in_data = in_tensor.const_data_ptr<scalar_t>();
  auto out_data = out_tensor.data_ptr<scalar_t>();

  sycl_kernel_submit<kfn>(
      sycl::range<1>(global_range),
      sycl::range<1>(local_range),
      getCurrentSYCLQueue(),
      0,
      in_data,
      out_data,
      val_of_work_item,
      N,
      total_offset,
      stride,
      shift,
      offset,
      start_offset,
      global_range);
}

void roll_kernel(
    const Tensor& input,
    Tensor& output,
    IntArrayRef shifts,
    IntArrayRef dims) {
  const int64_t N = input.numel();
  const int64_t dim = dims[0];
  const int64_t size = input.size(dim);
  int64_t start = (size - shifts[0]) % size;
  if (start < 0)
    start += size;

  auto total_dims = input.dim();
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
      at::ScalarType::Half,
      at::ScalarType::Bool,
      at::ScalarType::BFloat16,
      at::ScalarType::ComplexHalf,
      input.scalar_type(),
      "roll_xpu",
      [&] {
        roll_template<scalar_t>(
            input, output, N, dim, start, size, input.stride(dim), total_dims);
      });
}

} // namespace at::native::xpu
