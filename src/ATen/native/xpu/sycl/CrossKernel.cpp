/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/ATen.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/xpu/sycl/OffsetCalculator.h>
#include <c10/core/WrapDimMinimal.h>
#include <comm/SYCLContext.h>

#include <ATen/native/xpu/sycl/CrossKernel.h>

namespace at::native::xpu {
template <typename scalar_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void cross_ff_kernel(
    int64_t ostride_,
    int64_t xstride_,
    int64_t ystride_,
    const int64_t N_,
    int64_t work_group_size_,
    int64_t work_group_num_,
    scalar_t* out_,
    const scalar_t* x_,
    const scalar_t* y_,
    OffsetCalculator<3> offset_calculator_) {
  auto item = syclext::this_work_item::get_nd_item<1>();
  int64_t linear_index = item.get_global_id(0);
  for (int64_t i = linear_index; i < N_;
       i += work_group_num_ * work_group_size_) {
    const auto offsets = offset_calculator_.get(i);
    auto* out_row = out_ + offsets[0];
    const auto* x_row = x_ + offsets[1];
    const auto* y_row = y_ + offsets[2];

    const scalar_t val0 =
        (x_row[1 * xstride_] * y_row[2 * ystride_] -
         x_row[2 * xstride_] * y_row[1 * ystride_]);

    const scalar_t val1 =
        (x_row[2 * xstride_] * y_row[0 * ystride_] -
         x_row[0 * xstride_] * y_row[2 * ystride_]);

    const scalar_t val2 =
        (x_row[0 * xstride_] * y_row[1 * ystride_] -
         x_row[1 * xstride_] * y_row[0 * ystride_]);

    out_row[0 * ostride_] = val0;
    out_row[1 * ostride_] = val1;
    out_row[2 * ostride_] = val2;
  }
}

void launch_cross_kernel(
    const TensorIteratorBase& iter,
    int64_t ostride,
    int64_t x1_stride,
    int64_t x2_stride) {
  const auto N = iter.numel();
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      N > 0 && N <= std::numeric_limits<int32_t>::max());

  auto offset_calculator = make_element_offset_calculator<3>(iter);
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      iter.common_dtype(),
      "cross_xpu",
      [&]() {
        int64_t work_group_size =
            syclMaxWorkGroupSize<cross_ff_kernel<scalar_t>>();
        int64_t work_group_num = (N + work_group_size - 1) / work_group_size;
        auto out = static_cast<scalar_t*>(iter.data_ptr(0));
        auto x1 = static_cast<const scalar_t*>(iter.data_ptr(1));
        auto x2 = static_cast<const scalar_t*>(iter.data_ptr(2));

        sycl_kernel_submit<cross_ff_kernel<scalar_t>>(
            work_group_num * work_group_size,
            work_group_size,
            getCurrentSYCLQueue(),
            0,
            ostride,
            x1_stride,
            x2_stride,
            N,
            work_group_size,
            work_group_num,
            out,
            x1,
            x2,
            offset_calculator);
      });
}

void linalg_cross_kernel(
    const Tensor& result,
    const Tensor& x1,
    const Tensor& x2,
    int64_t dim) {
  const int64_t ostride = result.stride(dim);
  const int64_t x1_stride = x1.stride(dim);
  const int64_t x2_stride = x2.stride(dim);
  auto iter = TensorIteratorConfig()
                  .add_output(result)
                  .add_const_input(x1)
                  .add_const_input(x2)
                  .resize_outputs(false)
                  .declare_static_shape(result.sizes(), /*squash_dims=*/dim)
                  .build();

  if (iter.numel() == 0) {
    return;
  }

  if (iter.can_use_32bit_indexing()) {
    launch_cross_kernel(iter, ostride, x1_stride, x2_stride);
  } else {
    for (auto&& sub_iter : iter.with_32bit_indexing()) {
      launch_cross_kernel(sub_iter, ostride, x1_stride, x2_stride);
    }
  }
}

} // namespace at::native::xpu
