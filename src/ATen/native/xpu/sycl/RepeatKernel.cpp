#include <ATen/ATen.h>
#include <ATen/native/Repeat.h>
#include <ATen/native/xpu/sycl/RepeatKernel.h>
#include <comm/SYCLContext.h>

#include <ATen/native/xpu/sycl/RepeatKernel.h>

namespace at::native::xpu {
template <typename index_t>
struct RepeatInterleaveKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    auto rep_ptr = rep_data_;
    auto cum_ptr = cum_data_;
    auto res_ptr = res_data_;

    for (int64_t i = item.get_global_id(0); i < size_;
         i += item.get_global_range()[0]) {
      int64_t end = cum_ptr[i];
      int64_t repeat = rep_ptr[i];
      int64_t start = end - repeat;
      for (int64_t j = start; j < end; j++) {
        res_ptr[j] = i;
      }
    }
  }
  RepeatInterleaveKernelFunctor(
      const index_t* rep_data,
      const int64_t* cum_data,
      index_t* res_data,
      int64_t size,
      int64_t result_size)
      : rep_data_(rep_data),
        cum_data_(cum_data),
        res_data_(res_data),
        size_(size),
        result_size_(result_size) {}

 private:
  const index_t* rep_data_;
  const int64_t* cum_data_;
  index_t* res_data_;
  int64_t size_;
  int64_t result_size_;
};

template <typename index_t>
static void compute_xpu(
    const index_t* repeat_ptr,
    const int64_t* cumsum_ptr,
    index_t* result_ptr,
    int64_t size,
    int64_t result_size) {
  if (size == 0)
    return;

  auto kfn = RepeatInterleaveKernelFunctor<index_t>(
      repeat_ptr, cumsum_ptr, result_ptr, size, result_size);

  int64_t wg_size = syclMaxWorkGroupSize(kfn);
  int64_t local_range = size < wg_size ? size : wg_size;
  int64_t global_range = ((size + local_range - 1) / local_range) * local_range;

  auto queue = getCurrentSYCLQueue();
  sycl_kernel_submit(global_range, local_range, queue, kfn);
}

Tensor repeat_interleave_kernel(
    const Tensor& repeat,
    std::optional<int64_t> output_size) {
  Tensor output;

  AT_DISPATCH_INDEX_TYPES(repeat.scalar_type(), "repeat_interleave_xpu", [&] {
    output = repeat_interleave_common<index_t, compute_xpu<index_t>>(
        repeat, output_size);
  });
  return output;
}
} // namespace at::native::xpu
