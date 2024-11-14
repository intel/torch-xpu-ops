#include <ATen/ATen.h>
#include <ATen/native/BucketizationUtils.h>
#include <comm/SYCLContext.h>

#include <ATen/native/xpu/sycl/BucketizationKernels.h>

namespace at::native::xpu {

// customized lower_bound func to ensure the low bound of 'nan', 'inf' etc. be
// the end of boundary and we can properly handle a sorter argument
// std::lower_bound can not be used here since its customized comparator need
// strict weak ordering and the customized comparators require both arguments to
// have the same type, which wouldn't happen when comparing val of input_t to an
// indexer value from sorter of int64
template <typename input_t>
int64_t cus_lower_bound(
    int64_t start,
    int64_t end,
    const input_t val,
    const input_t* bd,
    const int64_t* sort) {
  // sorter gives relative ordering for ND tensors, so we need to save and add
  // the non-updated start as an offset i.e. the second row of a 3x3 tensors
  // starts at element 3 but sorter's second row only contains 0, 1, or 2
  const int64_t orig_start = start;
  while (start < end) {
    const int64_t mid = start + ((end - start) >> 1);
    const input_t mid_val = sort ? bd[sort[mid] + orig_start] : bd[mid];
    if (!(mid_val >= val)) {
      start = mid + 1;
    } else {
      end = mid;
    }
  }
  return start;
}

// customized upper_bound func to ensure we can properly handle a sorter
// argument std::upper_bound can not be used here since its customized
// comparator requires both arguments to have the same type, which wouldn't
// happen when comparing val of input_t to an indexer value from sorter of int64
template <typename input_t>
int64_t cus_upper_bound(
    int64_t start,
    int64_t end,
    const input_t val,
    const input_t* bd,
    const int64_t* sort) {
  // sorter gives relative ordering for ND tensors, so we need to save and add
  // the non-updated start as an offset i.e. the second row of a 3x3 tensors
  // starts at element 3 but sorter's second row only contains 0, 1, or 2
  const int64_t orig_start = start;
  while (start < end) {
    const int64_t mid = start + ((end - start) >> 1);
    const input_t mid_val = sort ? bd[sort[mid] + orig_start] : bd[mid];
    if (!(mid_val > val)) {
      start = mid + 1;
    } else {
      end = mid;
    }
  }
  return start;
}

template <typename input_t, typename output_t>
struct SearchsortedKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    for (int64_t i = item.get_global_id(0); i < numel_in_;
         i += item.get_global_range()[0]) {
      // If boundaries tensor is 1d, we always search the entire boundary
      // tensor
      int64_t start_bd = is_1d_boundaries_ ? 0 : i / idim_in_ * idim_bd_;
      int64_t end_bd = start_bd + idim_bd_;

      int64_t pos = !right_
          ? cus_lower_bound(
                start_bd, end_bd, data_in_data_[i], data_bd_data_, data_st_) -
              start_bd
          : cus_upper_bound(
                start_bd, end_bd, data_in_data_[i], data_bd_data_, data_st_) -
              start_bd;

      // type conversion might happen here
      data_out_data_[i] = pos;
    }
  }

  SearchsortedKernelFunctor(
      const bool right,
      int64_t numel_in,
      int64_t idim_in,
      int64_t idim_bd,
      const int64_t* data_st,
      bool is_1d_boundaries,
      const input_t* data_in_data,
      const input_t* data_bd_data,
      output_t* data_out_data)
      : right_(right),
        numel_in_(numel_in),
        idim_in_(idim_in),
        idim_bd_(idim_bd),
        data_st_(data_st),
        is_1d_boundaries_(is_1d_boundaries),
        data_in_data_(data_in_data),
        data_bd_data_(data_bd_data),
        data_out_data_(data_out_data) {}

 private:
  const bool right_;
  int64_t numel_in_;
  int64_t idim_in_;
  int64_t idim_bd_;
  const int64_t* data_st_;
  bool is_1d_boundaries_;
  const input_t* data_in_data_;
  const input_t* data_bd_data_;
  output_t* data_out_data_;
};
template <typename input_t, typename output_t>
void searchsorted_template(
    Tensor& result,
    const Tensor& input,
    const Tensor& boundaries,
    const bool& right,
    const Tensor& sorter) {
  int64_t numel_in = input.numel();
  bool is_scalar_input = input.dim() == 0 && numel_in == 1;
  // inner most dim size of input and boundaries
  int64_t idim_in = is_scalar_input ? 1 : input.sizes().back();
  int64_t idim_bd = boundaries.sizes().back();

  const int64_t* data_st =
      sorter.defined() ? sorter.const_data_ptr<int64_t>() : nullptr;

  bool is_1d_boundaries = boundaries.dim() == 1;
  auto data_in_data = input.const_data_ptr<input_t>();
  auto data_bd_data = boundaries.const_data_ptr<input_t>();
  auto data_out_data = result.mutable_data_ptr<output_t>();
  SearchsortedKernelFunctor<input_t, output_t> kfn(
      right,
      numel_in,
      idim_in,
      idim_bd,
      data_st,
      is_1d_boundaries,
      data_in_data,
      data_bd_data,
      data_out_data);

  int64_t rng, grng, tile_size;
  tile_size = syclMaxWorkGroupSize(kfn);
  rng = numel_in;
  if (rng == 0) {
    rng = static_cast<int64_t>(1);
  }

  grng = rng;
  if (tile_size > grng) {
    tile_size = grng;
  } else if (grng > tile_size) {
    int64_t xMode = static_cast<int64_t>(grng % tile_size);
    if (xMode != 0) {
      grng += static_cast<int64_t>(tile_size - xMode);
    }
  }

  sycl_kernel_submit(grng, tile_size, getCurrentSYCLQueue(), kfn);
}

void searchsorted_dispatch(
    Tensor& result,
    const Tensor& input,
    const Tensor& boundaries,
    bool out_int32,
    bool right,
    const Tensor& sorter) {
  if (!out_int32) {
    AT_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input.scalar_type(),
        "searchsorted_xpu",
        [&] {
          searchsorted_template<scalar_t, int64_t>(
              result, input, boundaries, right, sorter);
        });
  } else {
    AT_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input.scalar_type(),
        "searchsorted_xpu",
        [&] {
          searchsorted_template<scalar_t, int>(
              result, input, boundaries, right, sorter);
        });
  }
}

void searchsorted_kernel(
    Tensor& result,
    const Tensor& input,
    const Tensor& sorted_sequence,
    bool out_int32,
    bool right,
    const Tensor& sorter) {
  // for non-contiguous result tensors, we write the output to a contiguous copy
  // so we can later copy back, maintaining the original result tensor
  Tensor out = result;
  if (!result.is_contiguous()) {
    out = result.contiguous();
  }
  if (sorted_sequence.is_contiguous() && input.is_contiguous() &&
      sorted_sequence.dtype() == input.dtype() && sorter.is_contiguous()) {
    searchsorted_dispatch(
        out, input, sorted_sequence, out_int32, right, sorter);
  } else {
    Tensor trimmed_input;
    Tensor trimmed_boundaries;
    Tensor trimmed_sorter;
    at::native::searchsorted_maybe_trim_input_tensors(
        trimmed_input,
        trimmed_boundaries,
        trimmed_sorter,
        input,
        sorted_sequence,
        sorter);
    const Tensor& final_input = trimmed_input.defined() ? trimmed_input : input;
    const Tensor& final_boundaries =
        trimmed_boundaries.defined() ? trimmed_boundaries : sorted_sequence;
    const Tensor& final_sorter =
        trimmed_sorter.defined() ? trimmed_sorter : sorter;
    searchsorted_dispatch(
        out, final_input, final_boundaries, out_int32, right, final_sorter);
  }

  // if result is non-contiguous, we wrote the answer to a copied version, so we
  // copy back to the original result tensor
  if (!result.is_contiguous()) {
    result.copy_(out);
  }
}
} // namespace at::native::xpu
