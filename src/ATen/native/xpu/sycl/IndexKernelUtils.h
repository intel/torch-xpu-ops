#include <ATen/ceil_div.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/MemoryAccess.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <comm/SYCLContext.h>
#include <cstdint>
namespace at::native::xpu {

template <int alignment>
inline bool fast_gather_kernel_eligible(
    const TensorIterator& iter,
    char* const out_ptr,
    char* const in_ptr,
    const size_t index_stride_bytes,
    const size_t element_size) {
  using at::native::memory::get_alignment;
  const auto index_element_size = iter.element_size(2);
  // TensorIterator strides and sizes are ordered fastest moving to slowest
  // moving, in contrast to regular sizes
  // we need contiguous source and dst slices and aligned pointers and strides
  // and slice size to do vectorized loads also we need idx to be expanded in
  // the last dimension so we can copy entire slices and we need the src tensor
  // to keep 0 stride from restriding (it could have been deleted by dimension
  // collapse, in this case iterator would still be 2d but we cannot use fast
  // path)

  return iter.ndim() == 2 && iter.strides(2)[0] == 0 &&
      iter.strides(2)[1] == index_element_size &&
      static_cast<size_t>(iter.strides(0)[0]) == element_size &&
      static_cast<size_t>(iter.strides(1)[0]) == element_size &&
      static_cast<size_t>(iter.strides(1)[1] == 0) &&
      get_alignment(out_ptr) == alignment &&
      get_alignment(in_ptr) == alignment &&
      get_alignment(static_cast<size_t>(iter.shape()[0] * element_size)) ==
      alignment &&
      get_alignment(static_cast<size_t>(index_stride_bytes)) == alignment &&
      get_alignment(static_cast<size_t>(iter.strides(0)[1])) == alignment;
}

#define SIMD 32

template <int Alignment, typename index_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<2>))
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::sub_group_size<SIMD>)) void vectorized_gather_kernel(
    char* out_,
    char* inp_,
    index_t* idx_,
    int num_ind_,
    int64_t slice_size_,
    int64_t ind_dim_size_,
    int64_t inp_stride_,
    int64_t out_stride_,
    bool allow_neg_indices_) {
  auto item = syclext::this_work_item::get_nd_item<2>();
  int64_t ind = idx_[item.get_group(1)];
  if (allow_neg_indices_) {
    ind = (ind < 0) ? ind + ind_dim_size_ : ind;
  }
  SYCL_KERNEL_ASSERT(
      ind >= 0 && ind < ind_dim_size_ &&
      "vectorized gather kernel index out of bounds");
  int32_t off =
      (item.get_local_range(1) * item.get_group(0) + item.get_local_id(1)) *
      Alignment; // off is guaranteed to be within int32 limits
  if (off >= slice_size_)
    return;
  auto vec =
      at::native::memory::ld_vec<Alignment>(inp_ + ind * inp_stride_ + off);
  at::native::memory::st_vec<Alignment>(
      out_ + item.get_group(1) * (int32_t)out_stride_ + off,
      vec); // out offset is guaranteed to be within int32 limits
}

template <int64_t Alignment, typename index_t>
void vectorized_gather_kernel_launch(
    char* out,
    char* inp,
    index_t* idx,
    int num_ind,
    int64_t slice_size_in_bytes,
    int64_t ind_dim_size,
    int64_t inp_stride_bytes,
    int64_t out_stride_bytes,
    bool allow_neg_indices = false) {
  int64_t max_num_threads = syclMaxWorkItemsPerSubSlice();
  auto num_threads = at::round_up(
      at::ceil_div(slice_size_in_bytes, Alignment), static_cast<int64_t>(SIMD));
  auto wg_size = std::min(max_num_threads, num_threads);
  sycl::range<2> local_range(1, wg_size);
  sycl::range<2> global_range(
      static_cast<uint32_t>(
          at::ceil_div(slice_size_in_bytes, max_num_threads * Alignment)),
      static_cast<uint32_t>(num_ind) * wg_size);

  sycl_kernel_submit<vectorized_gather_kernel<Alignment, index_t>, 2>(
      global_range,
      local_range,
      at::xpu::getCurrentSYCLQueue(),
      0,
      out,
      inp,
      idx,
      num_ind,
      slice_size_in_bytes,
      ind_dim_size,
      inp_stride_bytes,
      out_stride_bytes,
      allow_neg_indices);
}

// explicit template instantiation
template void vectorized_gather_kernel_launch<16, int64_t>(
    char* out,
    char* inp,
    int64_t* idx,
    int num_ind,
    int64_t slice_size_in_bytes,
    int64_t ind_dim_size,
    int64_t inp_stride_bytes,
    int64_t out_stride_bytes,
    bool allow_neg_indices);

// explicit template instantiation
template void vectorized_gather_kernel_launch<16, int32_t>(
    char* out,
    char* inp,
    int32_t* idx,
    int num_ind,
    int64_t slice_size_in_bytes,
    int64_t ind_dim_size,
    int64_t inp_stride_bytes,
    int64_t out_stride_bytes,
    bool allow_neg_indices);

} // namespace at::native::xpu
