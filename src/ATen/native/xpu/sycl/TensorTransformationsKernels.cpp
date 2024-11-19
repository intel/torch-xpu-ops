// #define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
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
  using KernelClass = ElementwiseKernelFunctor<func_t>;

  auto& queue = getCurrentSYCLQueue();
  int64_t max_wg_size = syclMaxWorkGroupSize<KernelClass>();
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

  KernelClass kfn(loops, total_n_elems, f, total_work_items);

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

void flip_kernel(TensorIterator& iter, bool quantized) {
  if (quantized) {
    TORCH_CHECK(false, "XPU current does not flip for quantized tensor");
  }
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

template <typename scalar_t>
struct RollKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    int64_t linear_index = item.get_global_id(0);
    for (int i = 0; i < val_of_work_item_; i++) {
      if (linear_index < N_) {
        // roll dim idx is the index of linear_index along the rolling
        // dimension.
        int64_t roll_dim_idx = linear_index % (total_offset_) / stride_;
        // index into the source data to find appropriate value.
        int64_t source_idx = 0;
        source_idx = roll_dim_idx >= shift_ ? linear_index - offset_
                                            : linear_index + start_offset_;
        out_data_[linear_index] = in_data_[source_idx];
        linear_index += global_range_;
      }
    }
  }
  RollKernelFunctor(
      const scalar_t* in_data,
      scalar_t* out_data,
      int val_of_work_item,
      int64_t N,
      int64_t total_offset,
      int64_t stride,
      int64_t shift,
      int64_t offset,
      int64_t start_offset,
      int global_range)
      : in_data_(in_data),
        out_data_(out_data),
        val_of_work_item_(val_of_work_item),
        N_(N),
        total_offset_(total_offset),
        stride_(stride),
        shift_(shift),
        offset_(offset),
        start_offset_(start_offset),
        global_range_(global_range) {}

 private:
  const scalar_t* in_data_;
  scalar_t* out_data_;
  int val_of_work_item_;
  int64_t N_;
  int64_t total_offset_;
  int64_t stride_;
  int64_t shift_;
  int64_t offset_;
  int64_t start_offset_;
  int global_range_;
};

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
  using KernelClass = RollKernelFunctor<scalar_t>;

  auto shift = size - start;
  auto offset = shift * stride;
  auto start_offset = start * stride;
  auto total_offset = size * stride;

  auto local_range = syclMaxWorkGroupSize<KernelClass>();
  const auto target_global_range =
      syclMaxWorkItemsPerTile() / local_range * local_range;
  int global_range = (N + local_range - 1) / local_range * local_range;
  auto val_of_work_item =
      (global_range + target_global_range - 1) / target_global_range;
  global_range =
      global_range < target_global_range ? global_range : target_global_range;

  auto in_data = in_tensor.const_data_ptr<scalar_t>();
  auto out_data = out_tensor.data_ptr<scalar_t>();
  KernelClass kfn(
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

  sycl_kernel_submit(
      sycl::range<1>(global_range),
      sycl::range<1>(local_range),
      getCurrentSYCLQueue(),
      kfn);
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
