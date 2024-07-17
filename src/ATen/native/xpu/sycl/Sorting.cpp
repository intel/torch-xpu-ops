#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/NumericUtils.h>
#include <ATen/core/TensorBase.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/SortingCommon.h>
#include <ATen/native/xpu/sycl/SortingKernels.h>
#include <c10/macros/Macros.h>

namespace at {
namespace native {
namespace xpu {

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
struct GatherMedianKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    index_t slice = item.get_group_linear_id();

    // Finds the start offset for our slice
    index_t valuesSliceStartIndex =
        IndexToOffset<scalar_t, index_t>::get(slice, values);
    index_t indicesSliceStartIndex =
        IndexToOffset<int64_t, index_t>::get(slice, indices);
    index_t inputSliceStartIndex =
        IndexToOffset<scalar_t, index_t>::get(slice, input);

    scalar_t* valuesSliceStart = values_data + valuesSliceStartIndex;
    int64_t* indicesSliceStart = indices_data + indicesSliceStartIndex;
    scalar_t* inputSliceStart = in_data + inputSliceStartIndex;

    index_t nan_count = 0;
    for (index_t i = item.get_local_id(0); i < inputSliceSize;
         i += item.get_local_range(0)) {
      scalar_t val = inputSliceStart[i * inputWithinSliceStride];
      nan_count += Numerics<scalar_t>::isnan(val) ? 1 : 0;
    }

    // Counts number of nan values
    // This code performs a parallel sum reduction
    if (item.get_local_id(0) == 0) {
      num_nan[0] = 0;
    }

    item.barrier(dpcpp_local_fence);
    if (nan_count > 0) {
      atomicAdd(
          (dpcpp_local_ptr_pt<index_t>)(num_nan
                                            .template get_multi_ptr<
                                                sycl::access::decorated::no>()
                                            .get()),
          nan_count);
    }
    item.barrier(dpcpp_local_fence);

    // For torch.median, if we found nan set k to last index so the computed
    // value is nan, otherwise set k to the middle element of the non-nan
    // values
    index_t k = (!ignore_nan && num_nan[0] > 0)
        ? inputSliceSize - 1
        : (inputSliceSize - num_nan[0] - 1) / 2;

    // Find the median
    scalar_t median = static_cast<scalar_t>(0);
    radixSelect<
        scalar_t,
        typename TopKTypeConfig<scalar_t>::RadixType,
        index_t,
        false>(
        (dpcpp_global_ptr_pt<scalar_t>)inputSliceStart,
        k + 1,
        inputSliceSize,
        inputWithinSliceStride,
        smem,
        &median,
        item);

    valuesSliceStart[0] = median;

    // Find the index of the median value in the slice
    for (index_t i = item.get_local_id(0); i < inputSliceSize;
         i += item.get_local_range(0)) {
      scalar_t val = inputSliceStart[i * inputWithinSliceStride];
      if (Numerics<scalar_t>::eq(val, median) ||
          (Numerics<scalar_t>::isnan(val) &&
           Numerics<scalar_t>::isnan(median))) {
        indicesSliceStart[0] = i;
        break;
      }
    }
  }
  GatherMedianKernelFunctor(
      TensorInfo<scalar_t, index_t> values_,
      TensorInfo<int64_t, index_t> indices_,
      TensorInfo<scalar_t, index_t> input_,
      index_t inputSliceSize_,
      index_t numInputSlices_,
      index_t inputWithinSliceStride_,
      bool ignore_nan_,
      scalar_t* in_data_,
      scalar_t* values_data_,
      int64_t* indices_data_,
      dpcpp_local_acc_t<int> smem_,
      dpcpp_local_acc_t<index_t> num_nan_)
      : values(values_),
        indices(indices_),
        input(input_),
        inputSliceSize(inputSliceSize_),
        numInputSlices(numInputSlices_),
        inputWithinSliceStride(inputWithinSliceStride_),
        ignore_nan(ignore_nan_),
        in_data(in_data_),
        values_data(values_data_),
        indices_data(indices_data_),
        smem(smem_),
        num_nan(num_nan_) {}

 private:
  TensorInfo<scalar_t, index_t> values;
  TensorInfo<int64_t, index_t> indices;
  TensorInfo<scalar_t, index_t> input;
  index_t inputSliceSize;
  index_t numInputSlices;
  index_t inputWithinSliceStride;
  bool ignore_nan;
  scalar_t* in_data;
  scalar_t* values_data;
  int64_t* indices_data;
  dpcpp_local_acc_t<int> smem;
  dpcpp_local_acc_t<index_t> num_nan;
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

  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int64_t local_size = dpcppMaxWorkGroupSize(dev_id);

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto values_data = values.data;
    auto indices_data = indices.data;
    auto in_data = input.data;

    auto smem = dpcpp_local_acc_t<int>(32, cgh);
    auto num_nan = dpcpp_local_acc_t<index_t>(1, cgh);

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
        indices_data,
        smem,
        num_nan);

    cgh.parallel_for<decltype(kfn)>(
        sycl::nd_range<1>(
            sycl::range<1>(numInputSlices * local_size),
            sycl::range<1>(local_size)),
        kfn);
  };

  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
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

} // namespace xpu
} // namespace native
} // namespace at
