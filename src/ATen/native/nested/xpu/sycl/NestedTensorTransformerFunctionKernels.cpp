/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/core/TensorAccessor.h>
#include <ATen/native/StridedRandomAccessor.h>
#include <ATen/native/nested/xpu/sycl/NestedTensorTransformerFunctionKernels.h>
#include <comm/SYCLContext.h>

// keep align with cuda, global range0 is set to output_batch_size, global_range
// for dim1 is set to 16,
#define GRID_DIM_Y 16
#define BLOCK_DIM 256

namespace at::native::xpu {

template <typename T>
struct RemovePaddingFunctor {
  void operator()(sycl::nd_item<2> item) const {
    const int batch_id = item.get_group(1);
    const int grid_id = item.get_group(0);
    const int tid = item.get_local_id(1) + grid_id * BLOCK_DIM;
    const int grainsize = GRID_DIM_Y * BLOCK_DIM;
    const int offset = offsets_[batch_id];
    const int* sizes_i = output_sizes_ + batch_id * output_dim_;
    const int numel_i = sizes_i[0] * sizes_i[1] * sizes_i[2];
    int input_offset =
        batch_id * input_sizes_[1] * input_sizes_[2] * input_sizes_[3];
    for (int ii = 0; ii < (numel_i / grainsize); ii++) {
      const int i = ii * grainsize + tid;
      const int i0 = i / (sizes_i[1] * sizes_i[2]);
      const int i1 = (i % (sizes_i[1] * sizes_i[2])) / sizes_i[2];
      const int i2 = i % sizes_i[2];
      const int i0_offset = i0 * input_sizes_[2] * input_sizes_[3];
      const int i1_offset = i1 * input_sizes_[3];
      output_[offset + i] = input_[input_offset + i0_offset + i1_offset + i2];
    }
    const int i = (numel_i / grainsize) * grainsize + tid;
    if (i < numel_i) {
      const int i0 = i / (sizes_i[1] * sizes_i[2]);
      const int i1 = (i % (sizes_i[1] * sizes_i[2])) / sizes_i[2];
      const int i2 = i % sizes_i[2];
      const int i0_offset = i0 * input_sizes_[2] * input_sizes_[3];
      const int i1_offset = i1 * input_sizes_[3];
      output_[offset + i] = input_[input_offset + i0_offset + i1_offset + i2];
    }
  }

  RemovePaddingFunctor(
      const T* input,
      T* output,
      const int* offsets,
      const int* input_sizes,
      const int* output_sizes,
      int output_dim,
      const int batch_size)
      : input_(input),
        output_(output),
        offsets_(offsets),
        input_sizes_(input_sizes),
        output_sizes_(output_sizes),
        output_dim_(output_dim),
        batch_size_(batch_size) {}

 private:
  const T* input_;
  T* output_;
  const int* offsets_;
  const int* input_sizes_;
  const int* output_sizes_;
  int output_dim_;
  const int batch_size_;
};

template <typename T>
struct RemovePadding2Functor {
  void operator()(sycl::nd_item<2> item) const {
    const int batch_id = item.get_group(1);
    const int grid_id = item.get_group(0);
    const int tid = item.get_local_id(1) + grid_id * BLOCK_DIM;
    const int grainsize = GRID_DIM_Y * BLOCK_DIM;
    const int offset = offsets_[batch_id];
    const int* sizes_i = output_sizes_ + batch_id * output_dim_;
    const int numel_i = sizes_i[0] * sizes_i[1];
    int input_offset = batch_id * input_sizes_[1] * input_sizes_[2];
    for (int ii = 0; ii < (numel_i / grainsize); ii++) {
      const int i = ii * grainsize + tid;
      const int i0 = i / sizes_i[1];
      const int i1 = i % sizes_i[1];
      const int i0_offset = i0 * input_sizes_[2];
      output_[offset + i] = input_[input_offset + i0_offset + i1];
    }
    const int i = (numel_i / grainsize) * grainsize + tid;
    if (i < numel_i) {
      const int i0 = i / sizes_i[1];
      const int i1 = i % sizes_i[1];
      const int i0_offset = i0 * input_sizes_[2];
      output_[offset + i] = input_[input_offset + i0_offset + i1];
    }
  }

  RemovePadding2Functor(
      const T* input,
      T* output,
      const int* offsets,
      const int* input_sizes,
      const int* output_sizes,
      int output_dim,
      const int batch_size)
      : input_(input),
        output_(output),
        offsets_(offsets),
        input_sizes_(input_sizes),
        output_sizes_(output_sizes),
        output_dim_(output_dim),
        batch_size_(batch_size) {}

  const T* input_;
  T* output_;
  const int* offsets_;
  const int* input_sizes_;
  const int* output_sizes_;
  int output_dim_;
  const int batch_size_;
};

template <typename T>
struct RemovePaddingTransform0213Functor {
  void operator()(sycl::nd_item<2> item) const {
    const int batch_id = item.get_group(1);
    const int grid_id = item.get_group(0);
    const int tid = item.get_local_id(1) + grid_id * BLOCK_DIM;
    const int grainsize = GRID_DIM_Y * BLOCK_DIM;
    const int offset = offsets_[batch_id];
    const int* sizes_i = output_sizes_ + batch_id * output_dim_;
    const int numel_i = sizes_i[0] * sizes_i[1];
    int input_offset =
        batch_id * input_sizes_[1] * input_sizes_[2] * input_sizes_[3];
    for (int ii = 0; ii < (numel_i / grainsize); ii++) {
      const int i = ii * grainsize + tid;
      const int i2 = i / sizes_i[1];
      const int i13 = i % sizes_i[1];
      const int i1 = i13 / (sizes_i[1] / input_sizes_[1]);
      const int i3 = i13 % (sizes_i[1] / input_sizes_[1]);

      output_[offset + i] = input_
          [input_offset + i1 * input_sizes_[2] * input_sizes_[3] +
           i2 * input_sizes_[3] + i3];
    }
    const int i = (numel_i / grainsize) * grainsize + tid;
    if (i < numel_i) {
      const int i2 = i / sizes_i[1];
      const int i13 = i % sizes_i[1];
      const int i1 = i13 / (sizes_i[1] / input_sizes_[1]);
      const int i3 = i13 % (sizes_i[1] / input_sizes_[1]);
      output_[offset + i] = input_
          [input_offset + i1 * input_sizes_[2] * input_sizes_[3] +
           i2 * input_sizes_[3] + i3];
    }
  }

  RemovePaddingTransform0213Functor(
      const T* input,
      T* output,
      const int* offsets,
      const int* input_sizes,
      const int* output_sizes,
      int output_dim,
      const int batch_size)
      : input_(input),
        output_(output),
        offsets_(offsets),
        input_sizes_(input_sizes),
        output_sizes_(output_sizes),
        output_dim_(output_dim),
        batch_size_(batch_size) {}

  const T* input_;
  T* output_;
  const int* offsets_;
  const int* input_sizes_;
  const int* output_sizes_;
  int output_dim_;
  const int batch_size_;
};

template <typename T>
void remove_padding_kernel(
    const T* input,
    T* output,
    const int* offsets,
    const int* input_sizes,
    const int* output_sizes,
    int64_t output_dim,
    const int64_t batch_size) {
  auto queue = getCurrentSYCLQueue();
  if (output_dim == 2) {
    auto kfn = RemovePadding2Functor<T>(
        input,
        output,
        offsets,
        input_sizes,
        output_sizes,
        output_dim,
        batch_size);
    int64_t max_wg_size = syclMaxWorkGroupSize(kfn);
    sycl::range<2> global_range(GRID_DIM_Y, batch_size * max_wg_size);
    sycl::range<2> local_range(1, max_wg_size);
    sycl_kernel_submit(global_range, local_range, queue, kfn);
  } else {
    auto kfn = RemovePaddingFunctor<T>(
        input,
        output,
        offsets,
        input_sizes,
        output_sizes,
        output_dim,
        batch_size);
    int64_t max_wg_size = syclMaxWorkGroupSize(kfn);
    sycl::range<2> global_range(GRID_DIM_Y, batch_size * max_wg_size);
    sycl::range<2> local_range(1, max_wg_size);
    sycl_kernel_submit(global_range, local_range, queue, kfn);
  }
}

template <typename T>
void remove_padding_transform0213_kernel(
    const T* input,
    T* output,
    const int* offsets,
    const int* input_sizes,
    const int* output_sizes,
    int64_t output_dim,
    const int64_t batch_size) {
  TORCH_CHECK(
      output_dim == 2,
      "remove padding transform0213 only support output dim == 2");

  auto queue = getCurrentSYCLQueue();
  auto kfn = RemovePaddingTransform0213Functor<T>(
      input,
      output,
      offsets,
      input_sizes,
      output_sizes,
      output_dim,
      batch_size);

  int64_t max_wg_size = syclMaxWorkGroupSize(kfn);
  sycl::range<2> global_range(GRID_DIM_Y, batch_size * max_wg_size);
  sycl::range<2> local_range(1, max_wg_size);

  sycl_kernel_submit(global_range, local_range, queue, kfn);
}

void remove_padding_kernel_float(
    const float* input,
    float* output,
    const int* offsets,
    const int* input_sizes,
    const int* output_sizes,
    int64_t output_dim,
    const int64_t batch_size) {
  remove_padding_kernel<float>(
      input,
      output,
      offsets,
      input_sizes,
      output_sizes,
      output_dim,
      batch_size);
}

void remove_padding_kernel_half(
    const c10::Half* input,
    c10::Half* output,
    const int* offsets,
    const int* input_sizes,
    const int* output_sizes,
    int64_t output_dim,
    const int64_t batch_size) {
  remove_padding_kernel<c10::Half>(
      input,
      output,
      offsets,
      input_sizes,
      output_sizes,
      output_dim,
      batch_size);
}

void remove_padding_transform0213_kernel_float(
    const float* input,
    float* output,
    const int* offsets,
    const int* input_sizes,
    const int* output_sizes,
    int64_t output_dim,
    const int64_t batch_size) {
  remove_padding_transform0213_kernel<float>(
      input,
      output,
      offsets,
      input_sizes,
      output_sizes,
      output_dim,
      batch_size);
}

void remove_padding_transform0213_kernel_half(
    const c10::Half* input,
    c10::Half* output,
    const int* offsets,
    const int* input_sizes,
    const int* output_sizes,
    int64_t output_dim,
    const int64_t batch_size) {
  remove_padding_transform0213_kernel<c10::Half>(
      input,
      output,
      offsets,
      input_sizes,
      output_sizes,
      output_dim,
      batch_size);
}

template <typename T>
struct AddPadding1Functor {
  void operator()(sycl::nd_item<2> item) const {
    const int batch_id = item.get_group(1);
    const int grid_id = item.get_group(0);
    const int tid = item.get_local_id(1) + grid_id * BLOCK_DIM;
    const int grainsize = GRID_DIM_Y * BLOCK_DIM;
    const int* sizes_i = input_sizes_ + batch_id * input_dim_;
    const int batch_output_offset = batch_id * output_sizes_1_;
    for (int ii = 0; ii < (output_sizes_1_ / grainsize); ii++) {
      const int i = ii * grainsize + tid;
      const int output_offset = batch_output_offset + i;
      if (batch_id < batch_size_ && i < sizes_i[0]) {
        const int batch_input_offset = offsets_[batch_id];
        output_[output_offset] = input_[batch_input_offset + i];
      } else {
        output_[output_offset] = padding_value_;
      }
    }
    const int i = (output_sizes_1_ / grainsize) * grainsize + tid;
    if (i < output_sizes_1_) {
      const int output_offset = batch_output_offset + i;
      if (batch_id < batch_size_ && (i < sizes_i[0])) {
        const int batch_input_offset = offsets_[batch_id];
        output_[output_offset] = input_[batch_input_offset + i];
      } else {
        output_[output_offset] = padding_value_;
      }
    }
  }
  AddPadding1Functor(
      const T* input,
      T* output,
      T padding_value,
      const int* offsets,
      const int* input_sizes,
      int input_dim,
      int output_sizes_1,
      const int batch_size)
      : input_(input),
        output_(output),
        padding_value_(padding_value),
        offsets_(offsets),
        input_sizes_(input_sizes),
        input_dim_(input_dim),
        output_sizes_1_(output_sizes_1),
        batch_size_(batch_size) {}

 private:
  const T* input_;
  T* output_;
  T padding_value_;
  const int* offsets_;
  const int* input_sizes_;
  int input_dim_;
  int output_sizes_1_;
  const int batch_size_;
};

template <typename T>
struct AddPadding2Functor {
  void operator()(sycl::nd_item<2> item) const {
    const int batch_id = item.get_group(1);
    const int grid_id = item.get_group(0);
    const int tid = item.get_local_id(1) + grid_id * BLOCK_DIM;
    const int grainsize = GRID_DIM_Y * BLOCK_DIM;
    const int* sizes_i = input_sizes_ + batch_id * input_dim_;
    const int output_offset = batch_id * output_sizes_1_ * output_sizes_2_;
    const int output_numel = output_sizes_1_ * output_sizes_2_;
    for (int ii = 0; ii < (output_numel / grainsize); ii++) {
      const int i = ii * grainsize + tid;
      const int i0 = i / (output_sizes_2_);
      const int i1 = i - i0 * output_sizes_2_;
      if (batch_id < batch_size_ && i0 < sizes_i[0] && i1 < sizes_i[1]) {
        const int offset = offsets_[batch_id];
        const int input_offset = offset + i0 * sizes_i[1] + i1;
        output_[output_offset + i] = input_[input_offset];
      } else {
        output_[output_offset + i] = padding_value_;
      }
    }
    const int i = (output_numel / grainsize) * grainsize + tid;
    if (i < output_numel) {
      const int i0 = i / (output_sizes_2_);
      const int i1 = i - i0 * output_sizes_2_;
      if (batch_id < batch_size_ && i0 < sizes_i[0] && i1 < sizes_i[1]) {
        const int offset = offsets_[batch_id];
        const int input_offset = offset + i0 * sizes_i[1] + i1;
        output_[output_offset + i] = input_[input_offset];
      } else {
        output_[output_offset + i] = padding_value_;
      }
    }
  }
  AddPadding2Functor(
      const T* input,
      T* output,
      T padding_value,
      const int* offsets,
      const int* input_sizes,
      int input_dim,
      int output_sizes_1,
      int output_sizes_2,
      const int batch_size)
      : input_(input),
        output_(output),
        padding_value_(padding_value),
        offsets_(offsets),
        input_sizes_(input_sizes),
        input_dim_(input_dim),
        output_sizes_1_(output_sizes_1),
        output_sizes_2_(output_sizes_2),
        batch_size_(batch_size) {}

 private:
  const T* input_;
  T* output_;
  T padding_value_;
  const int* offsets_;
  const int* input_sizes_;
  int input_dim_;
  int output_sizes_1_;
  int output_sizes_2_;
  const int batch_size_;
};

template <typename T>
struct AddPadding3Functor {
  void operator()(sycl::nd_item<2> item) const {
    const int batch_id = item.get_group(1);
    const int grid_id = item.get_group(0);
    const int tid = item.get_local_id(1) + grid_id * BLOCK_DIM;
    const int grainsize = GRID_DIM_Y * BLOCK_DIM;
    const int* sizes_i = input_sizes_ + batch_id * input_dim_;
    const int output_offset =
        batch_id * output_sizes_1_ * output_sizes_2_ * output_sizes_3_;
    const int output_numel =
        output_sizes_1_ * output_sizes_2_ * output_sizes_3_;
    for (int ii = 0; ii < (output_numel / grainsize); ii++) {
      const int i = ii * grainsize + tid;
      const int i0 = i / (output_sizes_2_ * output_sizes_3_);
      const int i1 =
          (i % (output_sizes_2_ * output_sizes_3_)) / output_sizes_3_;
      const int i2 = i % output_sizes_3_;
      if (batch_id < batch_size_ && i0 < sizes_i[0] && i1 < sizes_i[1] &&
          i2 < sizes_i[2]) {
        const int offset = offsets_[batch_id];
        const int input_offset =
            offset + i0 * (sizes_i[1] * sizes_i[2]) + i1 * sizes_i[2] + i2;
        output_[output_offset + i] = input_[input_offset];
      } else {
        output_[output_offset + i] = padding_value_;
      }
    }
    const int i = (output_numel / grainsize) * grainsize + tid;
    if (i < output_numel) {
      const int i0 = i / (output_sizes_2_ * output_sizes_3_);
      const int i1 =
          (i % (output_sizes_2_ * output_sizes_3_)) / output_sizes_3_;
      const int i2 = i % output_sizes_3_;
      if (batch_id < batch_size_ && i0 < sizes_i[0] && i1 < sizes_i[1] &&
          i2 < sizes_i[2]) {
        const int offset = offsets_[batch_id];
        const int input_offset =
            offset + i0 * (sizes_i[1] * sizes_i[2]) + i1 * sizes_i[2] + i2;
        output_[output_offset + i] = input_[input_offset];
      } else {
        output_[output_offset + i] = padding_value_;
      }
    }
  }
  AddPadding3Functor(
      const T* input,
      T* output,
      T padding_value,
      const int* offsets,
      const int* input_sizes,
      int input_dim,
      int output_sizes_1,
      int output_sizes_2,
      int output_sizes_3,
      const int batch_size)
      : input_(input),
        output_(output),
        padding_value_(padding_value),
        offsets_(offsets),
        input_sizes_(input_sizes),
        input_dim_(input_dim),
        output_sizes_1_(output_sizes_1),
        output_sizes_2_(output_sizes_2),
        output_sizes_3_(output_sizes_3),
        batch_size_(batch_size) {}

 private:
  const T* input_;
  T* output_;
  T padding_value_;
  const int* offsets_;
  const int* input_sizes_;
  int input_dim_;
  int output_sizes_1_;
  int output_sizes_2_;
  int output_sizes_3_;
  const int batch_size_;
};

template <typename T>
void add_padding_kernel_impl(
    T* input, // [batch_size x None]
    T* output, // [batch_size x max(input.nested_size(1)) x inner_size]
    T padding_value,
    const int* offsets,
    const int* input_sizes,
    int input_dim,
    const std::vector<int64_t>& output_sizes,
    const int batch_size,
    const int output_batch_size) {
  auto queue = getCurrentSYCLQueue();
  if (input_dim == 1) {
    auto kfn = AddPadding1Functor<T>(
        input,
        output,
        padding_value,
        offsets,
        input_sizes,
        input_dim,
        output_sizes[1],
        batch_size);
    int64_t max_wg_size = syclMaxWorkGroupSize(kfn);
    sycl::range<2> global_range(GRID_DIM_Y, output_batch_size * max_wg_size);
    sycl::range<2> local_range(1, max_wg_size);
    sycl_kernel_submit(global_range, local_range, queue, kfn);
  }
  if (input_dim == 2) {
    auto kfn = AddPadding2Functor<T>(
        input,
        output,
        padding_value,
        offsets,
        input_sizes,
        input_dim,
        output_sizes[1],
        output_sizes[2],
        batch_size);
    int64_t max_wg_size = syclMaxWorkGroupSize(kfn);
    sycl::range<2> global_range(GRID_DIM_Y, output_batch_size * max_wg_size);
    sycl::range<2> local_range(1, max_wg_size);
    sycl_kernel_submit(global_range, local_range, queue, kfn);
  }
  if (input_dim == 3) {
    auto kfn = AddPadding3Functor<T>(
        input,
        output,
        padding_value,
        offsets,
        input_sizes,
        input_dim,
        output_sizes[1],
        output_sizes[2],
        output_sizes[3],
        batch_size);
    int64_t max_wg_size = syclMaxWorkGroupSize(kfn);
    sycl::range<2> global_range(GRID_DIM_Y, output_batch_size * max_wg_size);
    sycl::range<2> local_range(1, max_wg_size);
    sycl_kernel_submit(global_range, local_range, queue, kfn);
  }
}

void add_padding_kernel(
    at::Tensor input,
    at::Tensor output,
    double padding,
    const at::Tensor offsets,
    const at::Tensor nt_sizes,
    int input_dim,
    const std::vector<int64_t>& new_size,
    const int batch_size,
    const int output_batch_size) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "NestedTensor_to_padded_tensor_xpu", [&]() {
        add_padding_kernel_impl<scalar_t>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            (scalar_t)(padding),
            offsets.data_ptr<int>(),
            nt_sizes.data_ptr<int>(),
            input_dim,
            new_size,
            batch_size,
            output_batch_size);
      });
}

#define JAGGED_TENSOR_DISPATCH_DIMS()                                         \
  AT_DISPATCH_INDEX_TYPES(x_offsets[0].scalar_type(), "jagged_indices", [=] { \
    switch (num_jagged_dim) {                                                 \
      case 1:                                                                 \
        INVOKE_KERNEL_WITH_DIM(1);                                            \
        break;                                                                \
      case 2:                                                                 \
        INVOKE_KERNEL_WITH_DIM(2);                                            \
        break;                                                                \
      case 3:                                                                 \
        INVOKE_KERNEL_WITH_DIM(3);                                            \
        break;                                                                \
      case 4:                                                                 \
        INVOKE_KERNEL_WITH_DIM(4);                                            \
        break;                                                                \
      case 5:                                                                 \
        INVOKE_KERNEL_WITH_DIM(5);                                            \
        break;                                                                \
      default:                                                                \
        TORCH_CHECK(                                                          \
            false, "unsupported number of jagged dim ", num_jagged_dim);      \
    }                                                                         \
  });

inline std::string torch_tensor_device_name(const at::Tensor& ten) {
  return c10::DeviceTypeName(ten.device().type());
}

inline std::string torch_tensor_device_name(
    const std::optional<at::Tensor>& ten) {
  if (ten.has_value()) {
    return torch_tensor_device_name(ten.value());
  } else {
    return "N/A";
  }
}

inline bool torch_tensor_on_xpu_gpu_check(const at::Tensor& ten) {
  return ten.is_xpu();
}

inline bool torch_tensor_on_xpu_gpu_check(
    const std::optional<at::Tensor>& ten) {
  return !ten.has_value() || torch_tensor_on_xpu_gpu_check(ten.value());
}

#define TENSOR_ON_XPU_GPU(x)                                  \
  TORCH_CHECK(                                                \
      torch_tensor_on_xpu_gpu_check(x),                       \
      #x " must be a XPU tensor; it is currently on device ", \
      torch_tensor_device_name(x))

// A wrapper class for passing dynamically sized dimension information (e.g.
// tensor.dims()) from the host to device.
constexpr size_t kStackArrayMaxDims = 5;

template <typename T>
struct StackArray {
  T vals[kStackArrayMaxDims];
  size_t ndim;
};

template <typename scalar_t>
struct PaddingValueFuncutor {
  scalar_t operator()(scalar_t x, scalar_t /*unused*/) const {
    return x;
  }
};

// Subgroup size
static constexpr int32_t kSubgroupSize = 32;
// Max thread num in one thread workgroup
static constexpr int32_t kMaxThreads = 1024;

inline int32_t div_round_up(int32_t a, int32_t b) {
  return (a + b - 1) / b;
}

inline int32_t round_down(int32_t a, int32_t b) {
  return a / b * b;
}

inline std::tuple<sycl::range<2>, sycl::range<2>, StackArray<int64_t>>
check_shape_and_partition_(
    const Tensor& values,
    const std::vector<Tensor>& offsets,
    const Tensor& dense_tensor) {
  const int outer_dense_size = dense_tensor.size(0);
  TORCH_CHECK(
      outer_dense_size == offsets[0].numel() - 1,
      "outer_dense_size, ",
      outer_dense_size,
      " != offsets[0].numel() - 1, ",
      offsets[0].numel() - 1);
  const int inner_dense_size = dense_tensor.size(-1);
  TORCH_CHECK(
      inner_dense_size == values.size(-1),
      "inner_dense_size, ",
      inner_dense_size,
      " != values.size(-1), ",
      values.size(-1));
  const int jagged_folded_size =
      dense_tensor.numel() / (outer_dense_size * inner_dense_size);

  const int wg_size_x =
      inner_dense_size >= kSubgroupSize / 2 ? kSubgroupSize : inner_dense_size;
  const int wg_size_y = kMaxThreads / kSubgroupSize;
  const int num_group =
      div_round_up(outer_dense_size * jagged_folded_size, wg_size_y);

  StackArray<int64_t> jagged_dims_tensor{};
  const int num_jagged_dim = dense_tensor.dim() - 2;
  TORCH_CHECK(num_jagged_dim <= static_cast<int>(kStackArrayMaxDims));
  jagged_dims_tensor.ndim = num_jagged_dim;
  std::memcpy(
      &(jagged_dims_tensor.vals[0]),
      dense_tensor.sizes().data() + 1,
      num_jagged_dim * sizeof(int64_t));
  return {
      sycl::range<2>(wg_size_x, wg_size_y),
      sycl::range<2>(num_group * wg_size_x, wg_size_y),
      jagged_dims_tensor};
}

template <int NUM_JAGGED_DIM, typename index_t>
inline bool walk_down_tensor_storage_tree_(
    int& offset,
    const int flattened_jagged_idx,
    const StackArray<int64_t>& jagged_dims,
    const StackArray<index_t*>& x_offsets) {
  // compute coorindates
  int jagged_coords[NUM_JAGGED_DIM];
  int j_temp = flattened_jagged_idx;
#pragma unroll
  for (int d = NUM_JAGGED_DIM - 1; d >= 0; --d) {
    const int jagged_size = jagged_dims.vals[d];
    jagged_coords[d] = j_temp % jagged_size;
    j_temp /= jagged_size;
  }

  // walk down the tree
  bool is_zero = false;
#pragma unroll
  for (int d = 0; d < NUM_JAGGED_DIM; ++d) {
    const int begin = x_offsets.vals[d][offset];
    const int end = x_offsets.vals[d][offset + 1];
    if (jagged_coords[d] >= end - begin) {
      is_zero = true;
      break;
    }
    offset = begin + jagged_coords[d];
  }
  return is_zero;
}

template <int NUM_JAGGED_DIM, typename index_t, typename scalar_t, typename F>
struct JaggedDenseElementwiseDenseFunctor {
  void operator()(sycl::nd_item<2> item) const {
    const int outer_dense_size = y_.size(0);
    const int jagged_folded_size = y_.size(1);
    const int inner_dense_size = y_.size(2);
    auto output = output_;
    const int outer_begin =
        item.get_group(0) * item.get_local_range(1) + item.get_local_id(1);
    const int outer_stride = item.get_group_range(0) * item.get_local_range(1);
    for (int outer = outer_begin; outer < outer_dense_size * jagged_folded_size;
         outer += outer_stride) {
      const int oidx = outer / jagged_folded_size;
      const int jidx = outer % jagged_folded_size;

      int offset = oidx;
      const bool is_zero = walk_down_tensor_storage_tree_<NUM_JAGGED_DIM>(
          offset, jidx, jagged_dims_, x_offsets_);

      if (is_zero) {
        int iidx;
        for (iidx = item.get_local_id(0); iidx * 2 + 1 < inner_dense_size;
             iidx += item.get_local_range(0)) {
          output[oidx][jidx][2 * iidx] =
              f_(padding_value_, y_[oidx][jidx][2 * iidx]);
          output[oidx][jidx][2 * iidx + 1] =
              f_(padding_value_, y_[oidx][jidx][2 * iidx + 1]);
        }
        if (iidx * 2 + 1 == inner_dense_size) {
          output[oidx][jidx][2 * iidx] =
              f_(padding_value_, y_[oidx][jidx][2 * iidx]);
        }
      } else {
        int iidx;
        for (iidx = item.get_local_id(0); iidx * 2 + 1 < inner_dense_size;
             iidx += item.get_local_range(0)) {
          output[oidx][jidx][2 * iidx] =
              f_(x_values_[offset][2 * iidx], y_[oidx][jidx][2 * iidx]);
          output[oidx][jidx][2 * iidx + 1] =
              f_(x_values_[offset][2 * iidx + 1], y_[oidx][jidx][2 * iidx + 1]);
        }
        if (iidx * 2 + 1 == inner_dense_size) {
          output[oidx][jidx][2 * iidx] =
              f_(x_values_[offset][2 * iidx], y_[oidx][jidx][2 * iidx]);
        }
      }
    }
  }
  JaggedDenseElementwiseDenseFunctor(
      const at::PackedTensorAccessor32<scalar_t, 2, RestrictPtrTraits> x_values,
      StackArray<index_t*> x_offsets,
      const at::PackedTensorAccessor32<scalar_t, 3, RestrictPtrTraits> y,
      at::PackedTensorAccessor32<scalar_t, 3, RestrictPtrTraits> output,
      StackArray<int64_t> jagged_dims,
      F f,
      const scalar_t padding_value)
      : x_values_(x_values),
        x_offsets_(x_offsets),
        y_(y),
        output_(output),
        jagged_dims_(jagged_dims),
        f_(f),
        padding_value_(padding_value) {}

 private:
  const at::PackedTensorAccessor32<scalar_t, 2, RestrictPtrTraits> x_values_;
  StackArray<index_t*> x_offsets_;
  const at::PackedTensorAccessor32<scalar_t, 3, RestrictPtrTraits> y_;
  at::PackedTensorAccessor32<scalar_t, 3, RestrictPtrTraits> output_;
  StackArray<int64_t> jagged_dims_;
  F f_;
  const scalar_t padding_value_;
};

template <typename scalar_t, typename F>
void jagged_dense_elementwise_dense_template(
    const Tensor& x_values,
    const std::vector<Tensor>& x_offsets,
    const Tensor& y,
    const Tensor& output,
    F f,
    const scalar_t padding_value = static_cast<scalar_t>(0)) {
  TENSOR_ON_XPU_GPU(x_values);
  for (auto& x_offset : x_offsets) {
    TENSOR_ON_XPU_GPU(x_offset);
  }

  const int num_jagged_dim = y.dim() - 2;
  TORCH_CHECK(
      x_offsets.size() == static_cast<size_t>(num_jagged_dim),
      "x_offsets.size(), ",
      x_offsets.size(),
      " != num_jagged_dim ",
      num_jagged_dim);

  if (y.numel() == 0) {
    return;
  }

  sycl::range<2> global_range, local_range;
  StackArray<int64_t> jagged_dims_tensor;
  std::tie(local_range, global_range, jagged_dims_tensor) =
      check_shape_and_partition_(x_values, x_offsets, y);

  // Canonicalize y and output to 3D, collapsing jagged dimensions.
  const Tensor y_reshaped = y.view({y.size(0), -1, y.size(-1)});
  Tensor output_reshaped = output.view(y_reshaped.sizes());

#define INVOKE_KERNEL_WITH_DIM(NUM_JAGGED_DIM)                                 \
  {                                                                            \
    std::vector<Tensor> x_offsets_contig;                                      \
    x_offsets_contig.resize(num_jagged_dim);                                   \
    StackArray<index_t*> x_offset_ptrs;                                        \
    x_offset_ptrs.ndim = num_jagged_dim;                                       \
    for (int d = 0; d < num_jagged_dim; ++d) {                                 \
      x_offsets_contig[d] = x_offsets[d].contiguous();                         \
      x_offset_ptrs.vals[d] =                                                  \
          x_offsets_contig[d].template data_ptr<index_t>();                    \
    }                                                                          \
    auto kfn = JaggedDenseElementwiseDenseFunctor<                             \
        NUM_JAGGED_DIM,                                                        \
        index_t,                                                               \
        scalar_t,                                                              \
        F>(                                                                    \
        x_values.packed_accessor32<scalar_t, 2, RestrictPtrTraits>(),          \
        x_offset_ptrs,                                                         \
        y_reshaped.packed_accessor32<scalar_t, 3, RestrictPtrTraits>(),        \
        output_reshaped.packed_accessor32<scalar_t, 3, RestrictPtrTraits>(),   \
        jagged_dims_tensor,                                                    \
        f,                                                                     \
        padding_value);                                                        \
    sycl_kernel_submit(global_range, local_range, getCurrentSYCLQueue(), kfn); \
  }

  JAGGED_TENSOR_DISPATCH_DIMS();

#undef INVOKE_KERNEL_WITH_DIM
}

at::Tensor _fbgemm_jagged_to_padded_dense_forward_kernel(
    const Tensor& values,
    TensorList offsets,
    c10::IntArrayRef max_lengths,
    const double padding_value) {
  const Tensor values_canonicalized = values.view(
      {values.size(0),
       std::accumulate(
           values.sizes().begin() + 1,
           values.sizes().end(),
           1,
           std::multiplies<size_t>())});
  at::SymDimVector padded_values_shape({at::SymInt(offsets[0].size(0) - 1)});
  padded_values_shape.insert(
      padded_values_shape.end(), max_lengths.begin(), max_lengths.end());

  // Canonicalize padded_values by unsqueeze the last dim if the inner dense
  // dimension is 1 and folded.
  const bool D_folded = values.dim() == 1;
  if (!D_folded) {
    padded_values_shape.push_back(values.size(-1));
  }
  Tensor padded_values =
      at::empty_symint(padded_values_shape, values.options());
  Tensor padded_values_view =
      D_folded ? padded_values.unsqueeze(-1) : padded_values;

  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      values.scalar_type(),
      "jagged_to_padded_dense_xpu",
      [&] {
        jagged_dense_elementwise_dense_template<scalar_t>(
            values_canonicalized,
            offsets.vec(),
            padded_values_view, // dummy not used in the lambda function
            padded_values_view,
            PaddingValueFuncutor<scalar_t>(),
            static_cast<scalar_t>(padding_value));
      });

  return padded_values;
}

} // namespace at::native::xpu
