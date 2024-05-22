#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/WrapDimUtilsMulti.h>
#include <aten/sycl/MemoryAccess.h>
#include <aten/sycl/OffsetCalculator.h>
#include <comm/SYCLContext.h>

#if (defined(_WIN32))
#define RESTRICT __restrict
#else
#define RESTRICT __restrict__
#endif


namespace at::native::xpu {

template <int N>
struct alignas(N) OpaqueType {
  char data[N];
};

template <typename func_t>
struct ElementwiseKernelFunctor {
  void operator()(sycl::nd_item<1> itemId) const {
    int idx = itemId.get_global_linear_id();

    for (int i = 0; i < loops; ++i) {
      if (idx < total_n_elems) {
        f(idx);
        idx += total_work_items;
      }
    }
  }
  ElementwiseKernelFunctor(
      int loops_,
      int total_n_elems_,
      func_t f_,
      int total_work_items_)
      : loops(loops_),
        total_n_elems(total_n_elems_),
        f(f_),
        total_work_items(total_work_items_) {}

 private:
  int loops;
  int total_n_elems;
  func_t f;
  int total_work_items;
};

template <typename func_t>
void elementwise_kernel(int total_n_elems, func_t f) {
  auto& queue = getCurrentSYCLQueue();
  auto dev_id = getDeviceIndexOfCurrentQueue();
  int64_t max_wg_size = syclMaxWorkGroupSize(dev_id);
  const auto target_global_size = syclMaxWorkItemsPerTile(dev_id);
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

  ElementwiseKernelFunctor<func_t> kfn(
      loops, total_n_elems, f, total_work_items);

  sycl_kernel_submit(
      sycl::range<1>(total_work_items),
      sycl::range<1>(work_group_size),
      queue,
      kfn);
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

void flip_kernel(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      at::ScalarType::Half,
      at::ScalarType::Bool,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "flip_xpu",
      [&] {
        using dtype = OpaqueType<sizeof(scalar_t)>;
        flip_kernel_impl<dtype>(iter);
      });
}
} // namespace at::native::xpu