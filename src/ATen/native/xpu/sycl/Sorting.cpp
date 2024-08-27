#pragma clang diagnostic push
#pragma GCC diagnostic push
// Avoid SYCL compiler return-type error
#pragma clang diagnostic ignored "-Wreturn-type"
#pragma GCC diagnostic ignored "-Wreturn-type"
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/NumericUtils.h>
#include <ATen/core/TensorBase.h>
#include <ATen/native/CanUse32BitIndexMath.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/SortingCommon.h>
#include <ATen/native/xpu/sycl/SortingKernels.h>
#include <ATen/native/xpu/sycl/SortingRadixSelect.h>
#include <c10/macros/Macros.h>

namespace at::native::xpu {

template <typename key_t, typename value_t, typename func_t>
inline void host_kvsort(
    key_t* kbegin,
    key_t* kend,
    value_t* vbegin,
    const func_t& fn) {
  for (auto kit = kbegin, vit = vbegin; kit != kend; kit++, vit++) {
    for (auto kit_ = kit, vit_ = vit; kit_ != kend; kit_++, vit_++) {
      if (!fn(*kit, *kit_)) {
        std::swap(*kit, *kit_);
        std::swap(*vit, *vit_);
      }
    }
  }
}

std::vector<int64_t> infer_dense_strides_dim_last(
    const Tensor& self,
    int64_t dim) {
  int64_t ndim = self.dim();
  // sort the strides in descending order according to its value,
  // keeping dim the last.
  std::vector<int64_t> strides = self.strides().vec();
  strides[dim] = -1;
  std::vector<int64_t> original_dim(ndim);
  for (int64_t i = 0; i < ndim; i++) {
    original_dim[i] = i;
  }
  host_kvsort(
      strides.data(),
      strides.data() + ndim,
      original_dim.data(),
      std::greater<int64_t>());
  // generate contiguous strides on permuted dims
  std::vector<int64_t> new_strides(ndim);
  std::vector<int64_t> new_strides_unsort(ndim);
  int64_t cumprod = 1;
  for (int64_t i = 0; i < ndim; i++) {
    new_strides[ndim - 1 - i] = cumprod;
    cumprod *= self.sizes()[original_dim[ndim - 1 - i]];
  }
  // unsort new strides
  for (int64_t i = 0; i < ndim; i++) {
    new_strides_unsort[original_dim[i]] = new_strides[i];
  }
  return new_strides_unsort;
}

std::tuple<Tensor&, Tensor&> sort_stable_kernel(
    const Tensor& self,
    c10::optional<bool> stable,
    Tensor& values,
    Tensor& indices,
    int dim,
    bool descending) {
  TORCH_INTERNAL_ASSERT(
      stable.has_value(),
      "sort_out(): c10::optional<bool> for stable has to have value.");

  bool is_non_overlapping_and_dense = self.is_non_overlapping_and_dense();
  int64_t numel = self.numel();
  int64_t ndim = self.dim();
  dim = maybe_wrap_dim(dim, ndim);
  int64_t nsort = self.sizes()[dim];

  TORCH_CHECK(
      nsort <= std::numeric_limits<int>::max(),
      "The dimension being sorted can not have more than INT_MAX elements.");
  const auto self_dtype = self.dtype();
  TORCH_CHECK(
      self_dtype != ScalarType::ComplexFloat &&
          self_dtype != ScalarType::ComplexDouble,
      "Sort currently does not support complex dtypes on XPU.");

  if (ndim == 0) {
    if (!values.defined()) {
      values = self.clone();
    } else {
      values.resize_as_(self);
      values.copy_(self);
    }
    if (!indices.defined()) {
      indices = at::zeros({}, self.options().dtype(kLong));
    } else {
      indices.resize_as_(self);
      indices.zero_();
    }
    return std::forward_as_tuple(values, indices);
  }

  Tensor self_;
  bool newself = false;
  if (is_non_overlapping_and_dense && self.stride(dim) == 1) {
    self_ = self;
  } else {
    auto new_strides_unsort = infer_dense_strides_dim_last(self, dim);
    self_ = at::empty_strided(self.sizes(), new_strides_unsort, self.options());
    self_.copy_(self);
    newself = true;
  }

  Tensor values_tmp, indices_tmp;
  void* values_ptr_;
  int64_t* indices_ptr;
  if (!values.defined()) {
    if (is_non_overlapping_and_dense) {
      values = at::empty_strided(self.sizes(), self.strides(), self.options());
    } else {
      auto strides = at::infer_dense_strides(self.sizes(), self.strides());
      values = at::empty_strided(self.sizes(), strides, self.options());
    }
  } else {
    TORCH_CHECK(
        self_.scalar_type() == values.scalar_type(),
        "Unexpected dtype for values, expect ",
        self_.scalar_type(),
        ", got ",
        values.scalar_type());
    values.resize_as_(self);
  }

  if (values.strides() == self_.strides() &&
      (newself || get_overlap_status(self, values) == MemOverlapStatus::No)) {
    values_ptr_ = values.data_ptr();
  } else {
    values_tmp =
        at::empty_strided(self_.sizes(), self_.strides(), self_.options());
    values_ptr_ = values_tmp.data_ptr();
  }

  if (!indices.defined()) {
    if (is_non_overlapping_and_dense) {
      indices = at::empty_strided(
          self.sizes(), self.strides(), self.options().dtype(kLong));
    } else {
      auto strides = at::infer_dense_strides(self.sizes(), self.strides());
      indices =
          at::empty_strided(self.sizes(), strides, self.options().dtype(kLong));
    }
  } else {
    TORCH_CHECK(
        kLong == indices.scalar_type(),
        "Unexpected dtype for values, expect torch.long, got ",
        indices.scalar_type());
    indices.resize_as_(self);
  }

  if (indices.strides() != self_.strides()) {
    indices_tmp = at::empty_strided(
        self_.sizes(), self_.strides(), self_.options().dtype(kLong));
    indices_ptr = indices_tmp.data_ptr<int64_t>();
  } else {
    indices_ptr = indices.data_ptr<int64_t>();
  }

  if (numel == 0) {
    return std::forward_as_tuple(values, indices);
  }

  AT_DISPATCH_ALL_TYPES_AND3(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Bool,
      self_.scalar_type(),
      "sort_stable_kernel",
      [&]() {
        scalar_t* self_ptr = self_.data_ptr<scalar_t>();
        int nsegments = numel / nsort;
        segmented_sort_pairs<scalar_t, int64_t>(
            self_ptr,
            (scalar_t*)values_ptr_,
            nullptr,
            (int64_t*)indices_ptr,
            nsegments,
            nsort,
            descending);
      });

  if (values_tmp.defined()) {
    values.copy_(values_tmp);
  }
  if (indices_tmp.defined()) {
    indices.copy_(indices_tmp);
  }
  return std::forward_as_tuple(values, indices);
}

template <typename scalar_t, typename index_t, int Dim>
struct GatherMedianKernelFunctor : public __SYCL_KER_CONFIG_CONVENTION__ {
  void operator()(sycl::nd_item<1> item) const {
    index_t slice = item.get_group_linear_id();

    // Finds the start offset for our slice
    index_t valuesSliceStartIndex =
        IndexToOffset<scalar_t, index_t>::get(slice, values_);
    index_t indicesSliceStartIndex =
        IndexToOffset<int64_t, index_t>::get(slice, indices_);
    index_t inputSliceStartIndex =
        IndexToOffset<scalar_t, index_t>::get(slice, input_);

    scalar_t* valuesSliceStart = values_data_ + valuesSliceStartIndex;
    int64_t* indicesSliceStart = indices_data_ + indicesSliceStartIndex;
    scalar_t* inputSliceStart = in_data_ + inputSliceStartIndex;

    index_t nan_count = 0;
    for (index_t i = item.get_local_id(0); i < inputSliceSize_;
         i += item.get_local_range(0)) {
      scalar_t val = inputSliceStart[i * inputWithinSliceStride_];
      nan_count += at::_isnan(val) ? 1 : 0;
    }

    // Counts number of nan values
    // This code performs a parallel sum reduction
    if (item.get_local_id(0) == 0) {
      num_nan_[0] = 0;
    }

    item.barrier(sycl_local_fence);
    if (nan_count > 0) {
      atomicAdd(
          (sycl_local_ptr<index_t>)(num_nan_
                                        .template get_multi_ptr<
                                            sycl::access::decorated::no>()
                                        .get()),
          nan_count);
    }
    item.barrier(sycl_local_fence);

    // For torch.median, if we found nan set k to last index so the computed
    // value is nan, otherwise set k to the middle element of the non-nan
    // values
    index_t k = (!ignore_nan_ && num_nan_[0] > 0)
        ? inputSliceSize_ - 1
        : (inputSliceSize_ - num_nan_[0] - 1) / 2;

    // Find the median
    scalar_t median = static_cast<scalar_t>(0);
    radixSelect<
        scalar_t,
        typename TopKTypeConfig<scalar_t>::RadixType,
        index_t,
        false>(
        (sycl_global_ptr<scalar_t>)inputSliceStart,
        k + 1,
        inputSliceSize_,
        inputWithinSliceStride_,
        smem_,
        &median,
        item);

    valuesSliceStart[0] = median;

    // Find the index of the median value in the slice
    for (index_t i = item.get_local_id(0); i < inputSliceSize_;
         i += item.get_local_range(0)) {
      scalar_t val = inputSliceStart[i * inputWithinSliceStride_];
      if (val == median || (at::_isnan(val) && at::_isnan(median))) {
        indicesSliceStart[0] = i;
        break;
      }
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    smem_ = sycl_local_acc_t<int>(32, cgh);
    num_nan_ = sycl_local_acc_t<index_t>(1, cgh);
  }

  GatherMedianKernelFunctor(
      TensorInfo<scalar_t, index_t> values,
      TensorInfo<int64_t, index_t> indices,
      TensorInfo<scalar_t, index_t> input,
      index_t inputSliceSize,
      index_t numInputSlices,
      index_t inputWithinSliceStride,
      bool ignore_nan,
      scalar_t* in_data,
      scalar_t* values_data,
      int64_t* indices_data)
      : values_(values),
        indices_(indices),
        input_(input),
        inputSliceSize_(inputSliceSize),
        numInputSlices_(numInputSlices),
        inputWithinSliceStride_(inputWithinSliceStride),
        ignore_nan_(ignore_nan),
        in_data_(in_data),
        values_data_(values_data),
        indices_data_(indices_data) {}

 private:
  TensorInfo<scalar_t, index_t> values_;
  TensorInfo<int64_t, index_t> indices_;
  TensorInfo<scalar_t, index_t> input_;
  index_t inputSliceSize_;
  index_t numInputSlices_;
  index_t inputWithinSliceStride_;
  bool ignore_nan_;
  scalar_t* in_data_;
  scalar_t* values_data_;
  int64_t* indices_data_;
  sycl_local_acc_t<int> smem_;
  sycl_local_acc_t<index_t> num_nan_;
};

// kernel to find the median, and its index, of the values along dimension dim
template <typename scalar_t, typename index_t, int Dim>
void gatherMedian(
    TensorInfo<scalar_t, index_t> values,
    TensorInfo<int64_t, index_t> indices,
    TensorInfo<scalar_t, index_t> input,
    index_t inputSliceSize,
    index_t numInputSlices,
    index_t inputWithinSliceStride,
    bool ignore_nan) {
  // Shared memory for the subroutine RadixSelect. Note that RadixSelect
  // converts the floating point type to int with the same relative ordering.

  auto values_data = values.data;
  auto indices_data = indices.data;
  auto in_data = input.data;

  GatherMedianKernelFunctor<scalar_t, index_t, Dim> kfn(
      values,
      indices,
      input,
      inputSliceSize,
      numInputSlices,
      inputWithinSliceStride,
      ignore_nan,
      in_data,
      values_data,
      indices_data);
  int64_t local_size = syclMaxWorkGroupSize(kfn);
  sycl_kernel_submit(
      numInputSlices * local_size, local_size, getCurrentSYCLQueue(), kfn);
}

struct MedianLauncher {
  bool ignore_nan;

  MedianLauncher(bool ignore_nan) : ignore_nan(ignore_nan) {}

  template <typename scalar_t, typename index_t, int all_dims>
  inline void launch(
      TensorInfo<scalar_t, index_t> values_info,
      int collapse_values_dim,
      TensorInfo<int64_t, index_t> indices_info,
      int collapse_indices_dim,
      TensorInfo<scalar_t, index_t> self_info,
      int collapse_self_dim,
      int64_t num_slices,
      int64_t slice_size) {
    gatherMedian<scalar_t, index_t, all_dims>(
        values_info,
        indices_info,
        self_info,
        slice_size,
        num_slices,
        self_info.strides[collapse_self_dim],
        ignore_nan);
  }
};

void launch_median_kernel(
    const TensorBase& vals,
    const TensorBase& inds,
    const TensorBase& self,
    int64_t dim,
    bool ignore_nan) {
  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "median_out_impl_xpu",
      [&] {
        if (canUse32BitIndexMath(vals) && canUse32BitIndexMath(inds) &&
            canUse32BitIndexMath(self)) {
          run_launcher<scalar_t, uint32_t>(
              vals, inds, self, dim, MedianLauncher(ignore_nan));
        } else {
          run_launcher<scalar_t, uint64_t>(
              vals, inds, self, dim, MedianLauncher(ignore_nan));
        }
      });
}

} // namespace at::native::xpu

#pragma GCC diagnostic pop
#pragma clang diagnostic pop
