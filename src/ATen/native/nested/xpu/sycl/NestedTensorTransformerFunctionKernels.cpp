#include <ATen/ATen.h>
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

} // namespace at::native::xpu
