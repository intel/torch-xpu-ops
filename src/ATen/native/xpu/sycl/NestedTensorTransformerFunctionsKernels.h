#include <comm/SYCLContext.h>

// keep align with cuda, global range0 is set to output_batch_size, global_range
// for dim1 is set to 16,
#define GRID_DIM_Y 16
#define BLOCK_DIM 1024

namespace at::native::xpu {

template <typename T>
struct remove_padding_functor {
  void operator()(sycl::nd_item<2> item) const {
    const int batch_id = item.get_group(0);
    const int grid_id = item.get_group(1);
    const int tid = item.get_local_id()[0] + grid_id * BLOCK_DIM;
    const int grainsize = GRID_DIM_Y * BLOCK_DIM;
    const int offset = offsets[batch_id];

    const int* sizes_i = output_sizes + batch_id * output_dim;
    const int numel_i = sizes_i[0] * sizes_i[1] * sizes_i[2];
    int input_offset =
        batch_id * input_sizes[1] * input_sizes[2] * input_sizes[3];
    for (int ii = 0; ii < (numel_i / grainsize); ii++) {
      const int i = ii * grainsize + tid;
      const int i0 = i / (sizes_i[1] * sizes_i[2]);
      const int i1 = (i % (sizes_i[1] * sizes_i[2])) / sizes_i[2];
      const int i2 = i % sizes_i[2];
      const int i0_offset = i0 * input_sizes[2] * input_sizes[3];
      const int i1_offset = i1 * input_sizes[3];
      output[offset + i] = input[input_offset + i0_offset + i1_offset + i2];
    }
    const int i = (numel_i / grainsize) * grainsize + tid;
    if (i < numel_i) {
      const int i0 = i / (sizes_i[1] * sizes_i[2]);
      const int i1 = (i % (sizes_i[1] * sizes_i[2])) / sizes_i[2];
      const int i2 = i % sizes_i[2];
      const int i0_offset = i0 * input_sizes[2] * input_sizes[3];
      const int i1_offset = i1 * input_sizes[3];
      output[offset + i] = input[input_offset + i0_offset + i1_offset + i2];
    }
  }

  remove_padding_functor(
      const T* input_,
      T* output_,
      const int* offsets_,
      const int* input_sizes_,
      const int* output_sizes_,
      int output_dim_,
      const int batch_size_)
      : input(input_),
        output(output_),
        offsets(offsets_),
        input_sizes(input_sizes_),
        output_sizes(output_sizes_),
        output_dim(output_dim_),
        batch_size(batch_size_) {}

 private:
  const T* input;
  T* output;
  const int* offsets;
  const int* input_sizes;
  const int* output_sizes;
  int output_dim;
  const int batch_size;
};

template <typename T>
struct remove_padding_2_functor {
  void operator()(sycl::nd_item<2> item) const {
    const int batch_id = item.get_group(0);
    const int grid_id = item.get_group(1);
    const int tid = item.get_local_id()[0] + grid_id * BLOCK_DIM;
    const int grainsize = GRID_DIM_Y * BLOCK_DIM;
    const int offset = offsets[batch_id];
    const int* sizes_i = output_sizes + batch_id * output_dim;
    const int numel_i = sizes_i[0] * sizes_i[1];
    int input_offset = batch_id * input_sizes[1] * input_sizes[2];
    for (int ii = 0; ii < (numel_i / grainsize); ii++) {
      const int i = ii * grainsize + tid;
      const int i0 = i / sizes_i[1];
      const int i1 = i % sizes_i[1];
      const int i0_offset = i0 * input_sizes[2];
      output[offset + i] = input[input_offset + i0_offset + i1];
    }
    const int i = (numel_i / grainsize) * grainsize + tid;
    if (i < numel_i) {
      const int i0 = i / sizes_i[1];
      const int i1 = i % sizes_i[1];
      const int i0_offset = i0 * input_sizes[2];
      output[offset + i] = input[input_offset + i0_offset + i1];
    }
  }

  remove_padding_2_functor(
      const T* input_,
      T* output_,
      const int* offsets_,
      const int* input_sizes_,
      const int* output_sizes_,
      int output_dim_,
      const int batch_size_)
      : input(input_),
        output(output_),
        offsets(offsets_),
        input_sizes(input_sizes_),
        output_sizes(output_sizes_),
        output_dim(output_dim_),
        batch_size(batch_size_) {}

  const T* input;
  T* output;
  const int* offsets;
  const int* input_sizes;
  const int* output_sizes;
  int output_dim;
  const int batch_size;
};

template <typename T>
struct remove_padding_transform0213_functor {
  void operator()(sycl::nd_item<2> item) const {
    const int batch_id = item.get_group(0);
    const int grid_id = item.get_group(1);
    const int tid = item.get_local_id()[0] + grid_id * BLOCK_DIM;
    const int grainsize = GRID_DIM_Y * BLOCK_DIM;
    const int offset = offsets[batch_id];
    const int* sizes_i = output_sizes + batch_id * output_dim;
    const int numel_i = sizes_i[0] * sizes_i[1];
    int input_offset =
        batch_id * input_sizes[1] * input_sizes[2] * input_sizes[3];
    for (int ii = 0; ii < (numel_i / grainsize); ii++) {
      const int i = ii * grainsize + tid;
      const int i2 = i / sizes_i[1];
      const int i13 = i % sizes_i[1];
      const int i1 = i13 / (sizes_i[1] / input_sizes[1]);
      const int i3 = i13 % (sizes_i[1] / input_sizes[1]);

      output[offset + i] = input
          [input_offset + i1 * input_sizes[2] * input_sizes[3] +
           i2 * input_sizes[3] + i3];
    }
    const int i = (numel_i / grainsize) * grainsize + tid;
    if (i < numel_i) {
      const int i2 = i / sizes_i[1];
      const int i13 = i % sizes_i[1];
      const int i1 = i13 / (sizes_i[1] / input_sizes[1]);
      const int i3 = i13 % (sizes_i[1] / input_sizes[1]);
      output[offset + i] = input
          [input_offset + i1 * input_sizes[2] * input_sizes[3] +
           i2 * input_sizes[3] + i3];
    }
  }

  remove_padding_transform0213_functor(
      const T* input_,
      T* output_,
      const int* offsets_,
      const int* input_sizes_,
      const int* output_sizes_,
      int output_dim_,
      const int batch_size_)
      : input(input_),
        output(output_),
        offsets(offsets_),
        input_sizes(input_sizes_),
        output_sizes(output_sizes_),
        output_dim(output_dim_),
        batch_size(batch_size_) {}

  const T* input;
  T* output;
  const int* offsets;
  const int* input_sizes;
  const int* output_sizes;
  int output_dim;
  const int batch_size;
};

template <typename T>
void remove_padding_transform0213_kernelLauncher(
    const T* input,
    T* output,
    const int* offsets,
    const int* input_sizes,
    const int* output_sizes,
    int output_dim,
    const int batch_size) {
  TORCH_CHECK(
      output_dim == 2,
      "remove padding transform0213 only support output dim == 2");

  auto queue = getCurrentSYCLQueue();
  auto kfn = remove_padding_transform0213_functor<T>(
      input,
      output,
      offsets,
      input_sizes,
      output_sizes,
      output_dim,
      batch_size);
  int64_t max_wg_size = syclMaxWorkGroupSize(kfn);
  sycl::range<2> global_range{(size_t)batch_size * max_wg_size, GRID_DIM_Y};
  sycl::range<2> local_range{(size_t)max_wg_size, 1};

  sycl_kernel_submit(global_range, local_range, queue, kfn);
}

template <typename T>
void remove_padding_kernelLauncher(
    const T* input,
    T* output,
    const int* offsets,
    const int* input_sizes,
    const int* output_sizes,
    int output_dim,
    const int batch_size) {
  auto queue = getCurrentSYCLQueue();

  if (output_dim == 2) {
    auto kfn = remove_padding_2_functor<T>(
        input,
        output,
        offsets,
        input_sizes,
        output_sizes,
        output_dim,
        batch_size);
    int64_t max_wg_size = syclMaxWorkGroupSize(kfn);
    sycl::range<2> global_range{(size_t)batch_size * max_wg_size, GRID_DIM_Y};
    sycl::range<2> local_range{(size_t)max_wg_size, 1};

    sycl_kernel_submit(global_range, local_range, queue, kfn);
  } else {
    auto kfn = remove_padding_functor<T>(
        input,
        output,
        offsets,
        input_sizes,
        output_sizes,
        output_dim,
        batch_size);

    int64_t max_wg_size = syclMaxWorkGroupSize(kfn);
    sycl::range<2> global_range{(size_t)batch_size * max_wg_size, GRID_DIM_Y};
    sycl::range<2> local_range{(size_t)max_wg_size, 1};

    sycl_kernel_submit(global_range, local_range, queue, kfn);
  }
}

} // namespace at::native::xpu