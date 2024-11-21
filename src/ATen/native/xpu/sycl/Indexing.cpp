#pragma clang diagnostic push
#pragma GCC diagnostic push
// Avoid SYCL compiler return-type error
#pragma clang diagnostic ignored "-Wreturn-type"
#pragma GCC diagnostic ignored "-Wreturn-type"

#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/Atomics.h>
#include <ATen/native/xpu/sycl/Indexing.h>
#include <ATen/native/xpu/sycl/IndexingUtils.h>
#include <ATen/native/xpu/sycl/Loops.h>
#include <ATen/native/xpu/sycl/pstl/PSTLFunctions.h>

#include <comm/SYCLContext.h>
#include <comm/TensorInfo.h>

#include <ATen/native/xpu/sycl/IndexingKernels.h>

using namespace at::xpu::detail;
using namespace at::xpu;

namespace at::native::xpu {

template <typename dtype>
struct IndexFunctor {
  void operator()(char* out_data, char* in_data, int64_t offset) const {
    *(dtype*)out_data = *(dtype*)(in_data + offset);
  }
};

void index_kernel(
    TensorIteratorBase& iter,
    IntArrayRef index_size,
    IntArrayRef index_stride) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
      at::ScalarType::ComplexHalf,
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      at::ScalarType::Bool,
      iter.dtype(),
      "index_xpu",
      [&] {
        using dtype = OpaqueType<sizeof(scalar_t)>;
        IndexFunctor<dtype> f;
        _index_kernel(
            iter, index_size, index_stride, IntArrayRef{}, IntArrayRef{}, f);
      });
}

template <typename ValType>
class IndexSelectScalarFunctor {
 public:
  void operator()(
      ValType* dst,
      const ValType* src,
      int64_t dst_off,
      int64_t src_off,
      int64_t idx,
      ValType alpha) const {
    dst[dst_off] = src[src_off];
  }
};

template <
    class SrcInfo,
    class DstInfo,
    class IdxInfo,
    bool TrivialOffCal = false>
static inline void _index_select_kernel(
    SrcInfo& src_info,
    DstInfo& dst_info,
    IdxInfo& index_info,
    int64_t dim) {
  using scalar_t = typename DstInfo::scalar_t;
  using IdxConfig = IndexKernelConfig<
      SrcInfo,
      DstInfo,
      IdxInfo,
      IndexSelectScalarFunctor<scalar_t>>;

  using IndexKnownProblemInnerKernel =
      IndexKernel<IdxConfig, TrivialOffCal, true>;
  auto IndexKnownProblemInnerKernel_cfg =
      IdxConfig::template make_config<IndexKnownProblemInnerKernel>(
          src_info,
          dst_info,
          index_info,
          static_cast<scalar_t>(0),
          dim,
          false,
          IndexSelectScalarFunctor<scalar_t>());

  using IndexUnknownProblemInnerKernel =
      IndexKernel<IdxConfig, TrivialOffCal, false>;
  auto IndexUnknownProblemInnerKernel_cfg =
      IdxConfig::template make_config<IndexUnknownProblemInnerKernel>(
          src_info,
          dst_info,
          index_info,
          static_cast<scalar_t>(0),
          dim,
          false,
          IndexSelectScalarFunctor<scalar_t>());

  if (IndexKnownProblemInnerKernel_cfg.problem_inner_) {
    launch_index_kernel<IdxConfig, TrivialOffCal, true>(
        IndexKnownProblemInnerKernel_cfg);
  } else {
    launch_index_kernel<IdxConfig, TrivialOffCal, false>(
        IndexUnknownProblemInnerKernel_cfg);
  }
}

void index_select_kernel(
    const Tensor& src,
    int64_t dim,
    const Tensor& indices,
    const Tensor& dst) {
  at::assert_no_internal_overlap(dst);
  at::assert_no_overlap(dst, src);
  at::assert_no_overlap(dst, indices);

  dim = at::maybe_wrap_dim(dim, src.dim());
  int srcDims = src.dim() == 0 ? 1 : src.dim();
  int dstDims = dst.dim();
  int idxDims = indices.dim();

  TORCH_CHECK(
      srcDims <= XPU_MAX_TENSORINFO_DIMS,
      "src tensor dim should be < ",
      XPU_MAX_TENSORINFO_DIMS);
  TORCH_CHECK(
      dstDims <= XPU_MAX_TENSORINFO_DIMS,
      "dst tensor dim should be < ",
      XPU_MAX_TENSORINFO_DIMS);
  TORCH_CHECK(
      idxDims <= XPU_MAX_TENSORINFO_DIMS,
      "index tensor dim should be < ",
      XPU_MAX_TENSORINFO_DIMS);
  TORCH_CHECK(
      idxDims <= 1, "Index is supposed to be an empty tensor or a vector");
  TORCH_CHECK(
      dim >= -1 && dim < srcDims,
      "Indexing dim should be >= -1 and < dims - 1");
  TORCH_CHECK(srcDims > 0, "Source tensor is empty");
  TORCH_CHECK(
      indices.scalar_type() == ScalarType::Long ||
          indices.scalar_type() == ScalarType::Int,
      "index_select(): Expected dtype int32 or int64 for index but got: ",
      indices.scalar_type());
  TORCH_CHECK(
      src.scalar_type() == dst.scalar_type(),
      "index_select(): Source and result must have the same scalar type");

  AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "index_select", [&] {
    TensorInfo<const index_t, int64_t> index_info =
        tensorInfoIfScalar(getTensorInfo<const index_t, int64_t>(indices));
    index_info.collapseDims();

    auto new_size = src.sizes().vec();

    if (src.dim() > 0) {
      new_size[dim] = indices.numel();
    }

    at::native::resize_output(dst, new_size);

    ptrdiff_t dst_num_elem = dst.numel();
    if (dst_num_elem == 0) {
      return;
    }

    AT_DISPATCH_V2(
        dst.scalar_type(),
        "index_select_xpu",
        AT_WRAP([&] {
          TensorInfo<scalar_t, int64_t> dst_info =
              tensorInfoIfScalar(getTensorInfo<scalar_t, int64_t>(dst));
          TensorInfo<const scalar_t, int64_t> src_info = tensorInfoIfScalar(
              getTensorInfo<const scalar_t, int64_t>(src.contiguous()));
          int new_indexing_dim = src_info.collapseDims(dim);

          using SrcInfo = TensorInfo<const scalar_t, int64_t>;
          using DstInfo = TensorInfo<scalar_t, int64_t>;
          using IdxInfo = TensorInfo<const index_t, int64_t>;

          // Improve efficiency of generated native instructions for contiguous.
          // See comm/TensorInfo.h
          if (dst.is_contiguous() && indices.is_contiguous())
            _index_select_kernel<
                SrcInfo,
                DstInfo,
                IdxInfo,
                /* TrivialOffCal */ true>(
                src_info, dst_info, index_info, new_indexing_dim);
          else
            _index_select_kernel<
                SrcInfo,
                DstInfo,
                IdxInfo,
                /* TrivialOffCal */ false>(
                src_info, dst_info, index_info, new_indexing_dim);
        }),
        AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX),
        AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES),
	AT_EXPAND(AT_FLOAT8_TYPES),
        kComplexHalf,
        kHalf,
        kBool,
        kBFloat16);
  });
  return;
}

template <typename scalar_t>
struct MaskedFillFunctor {
  scalar_t operator()(scalar_t self, bool mask) const {
    if (mask) {
      return value_;
    }
    return self;
  }
  MaskedFillFunctor(scalar_t value) : value_(value) {}

 private:
  scalar_t value_;
};

void masked_fill_kernel(TensorIterator& iter, const Scalar& value) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
      kBool,
      kHalf,
      kBFloat16,
      kComplexHalf,
      iter.common_dtype(),
      "masked_fill__xpu",
      [&]() {
        const auto value_ = value.to<scalar_t>();
        gpu_kernel(iter, MaskedFillFunctor<scalar_t>(value_));
      });
}

// Check tensor dimensions for index operations, and return the slice size.
static ptrdiff_t getSliceSize(
    const Tensor& dst,
    int dim,
    const Tensor& index,
    const Tensor& src) {
  const auto dstDims = dst.dim();
  const auto srcDims = src.dim();

  TORCH_CHECK(index.dim() <= 1, "Index must be vector or scalar");

  ptrdiff_t dstSliceSize = 1;
  TORCH_CHECK(
      dim >= 0 && dim < dstDims, "Indexing dim ", dim, " is out of bounds");
  for (const auto d : c10::irange(dstDims)) {
    if (d != dim) {
      dstSliceSize *= dst.size(d);
    }
  }

  TORCH_CHECK(dim < srcDims, "Indexing dim ", dim, " is out of bounds");
  TORCH_CHECK(
      index.numel() == src.size(dim),
      "length of src.size[dim] is not equal to length of indices");

  ptrdiff_t srcSliceSize = 1;
  bool mismatch = false;

  if (dstDims != srcDims)
    mismatch = true;

  for (const auto d : c10::irange(srcDims)) {
    if (d != dim) {
      srcSliceSize *= src.size(d);
      if (!mismatch && dst.size(d) != src.size(d))
        mismatch = true;
    }
  }

  TORCH_CHECK(
      dstSliceSize == srcSliceSize,
      "Source/destination tensor have different slice sizes (%ld vs %ld)",
      dstSliceSize,
      srcSliceSize);

  if (mismatch) {
    TORCH_WARN_ONCE(
        "Warning: source/destination slices have same size but different "
        "shape for an index operation.  This behavior is deprecated.\n");
  }

  return dstSliceSize;
}

template <typename ValType>
struct IndexAddScalarFunctor {
  void operator()(
      ValType* dst,
      const ValType* src,
      int64_t dst_off,
      int64_t src_off,
      int64_t idx,
      ValType alpha) const {
    atomicAdd((sycl_global_ptr<ValType>)(dst + dst_off), src[src_off] * alpha);
  }
};

template <>
struct IndexAddScalarFunctor<bool> {
  void operator()(
      bool* dst,
      const bool* src,
      int64_t dst_off,
      int64_t src_off,
      int64_t idx,
      bool alpha) const {
    atomicAdd((sycl_global_ptr<bool>)(dst + dst_off), src[src_off] && alpha);
  }
};

template <typename scalar_t>
struct IndexReduceMultiplyFunctor {
  void operator()(
      scalar_t* dst,
      const scalar_t* src,
      int64_t dst_off,
      int64_t src_off,
      int64_t idx) const{
    atomicMul((sycl_global_ptr<scalar_t>)(dst + dst_off), src[src_off]);
  }
};


template <typename scalar_t>
struct IndexReduceMaxFunctor {
  void operator()(
      scalar_t* dst,
      const scalar_t* src,
      int64_t dst_off,
      int64_t src_off,
      int64_t idx) const{
    atomicMax((sycl_global_ptr<scalar_t>)(dst + dst_off), src[src_off]);
  }
};


template <typename scalar_t>
struct IndexReduceMinFunctor {
  void operator()(
      scalar_t* dst,
      const scalar_t* src,
      int64_t dst_off,
      int64_t src_off,
      int64_t idx) const{
    atomicMin((sycl_global_ptr<scalar_t>)(dst + dst_off), src[src_off]);
  }
};




void index_add_kernel(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& source,
    const Scalar& alpha,
    const Tensor& result) {
  if (!result.is_same(self)) {
    result.copy_(self);
  }

  auto numel = index.numel();
  if (result.dim() > 1) {
    if (numel == 0 || self.numel() == 0) {
      return;
    }
  }

  // Scalars are treated as 1-d tensor
  const Tensor self_ = (result.dim() == 0) ? result.view(1) : result;
  const Tensor source_ = (source.dim() == 0) ? source.view(1) : source;

  TORCH_CHECK(
      result.dim() <= XPU_MAX_TENSORINFO_DIMS,
      "tensor has too many (>",
      XPU_MAX_TENSORINFO_DIMS,
      ") dims");
  TORCH_CHECK(
      source.dim() <= XPU_MAX_TENSORINFO_DIMS,
      "tensor has too many (>",
      XPU_MAX_TENSORINFO_DIMS,
      ") dims");
  TORCH_CHECK(
      index.dim() <= XPU_MAX_TENSORINFO_DIMS,
      "tensor has too many (>",
      XPU_MAX_TENSORINFO_DIMS,
      ") dims");

  if (globalContext().deterministicAlgorithms()) {
    torch::List<c10::optional<Tensor>> indices;
    indices.reserve(dim + 1);
    for (int i = 0; i < dim; i++) {
      indices.emplace_back();
    }
    indices.emplace_back(index.to(at::kLong));
    result.index_put_(indices, source * alpha, true);
    return;
  }

  // The `source` is partitioned into two parts:
  // -the size of each slice we are indexing, which is the
  // total size of the tensor ignoring dimension `dim`;
  // -the number of index we are choosing, which is the total size
  // of the tensor `index`.
  const ptrdiff_t sliceSize = getSliceSize(self_, dim, index, source_);

  if (sliceSize == 0) {
    return;
  }

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
      at::ScalarType::Bool,
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::ComplexHalf,
      source_.scalar_type(),
      "index_add_xpu",
      [&] {
        AT_DISPATCH_INDEX_TYPES(index.scalar_type(), "index_add_xpu", [&]() {
          TensorInfo<const index_t, int64_t> index_info =
              getTensorInfo<const index_t, int64_t>(index);
          index_info.collapseDims();

          TensorInfo<const scalar_t, int64_t> src_info =
              getTensorInfo<const scalar_t, int64_t>(source_);

          TensorInfo<scalar_t, int64_t> dst_info =
              getTensorInfo<scalar_t, int64_t>(self_);
          int new_indexing_dim = dst_info.collapseDims(dim);

          using IdxConfig = IndexKernelConfig<
              decltype(src_info),
              decltype(dst_info),
              decltype(index_info),
              IndexAddScalarFunctor<scalar_t>>;
          using KernelClass = IndexKernel<IdxConfig, false, false>;

          auto cfg = IdxConfig::template make_config<KernelClass>(
              src_info,
              dst_info,
              index_info,
              alpha.to<scalar_t>(),
              new_indexing_dim,
              true,
              IndexAddScalarFunctor<scalar_t>());
          launch_index_kernel(cfg);
        });
      });
}

template <typename ValType>
struct IndexFillScalarFunctor {
  void operator()(
      ValType* dst,
      const ValType* src,
      int64_t dst_off,
      int64_t src_off,
      int64_t idx,
      ValType alpha) const {
    dst[dst_off] = alpha;
  }
};

void index_fill_kernel(
    TensorIterator& iter,
    const int64_t dim,
    const int64_t self_dim_size,
    const int64_t self_dim_stride,
    const Scalar& source) {
  Tensor self = iter.tensor(0);
  Tensor index = iter.tensor(1);

  // index_fill operator generates TensorIterator as kernel input,
  // self tensor is restrided to meet TensorIterator broadcast requirements. But
  // xpu kernel doesn't support such restrided shape, so we restride self tensor
  // back here.
  auto self_sizes = self.sizes().vec();
  auto self_strides = self.strides().vec();
  self_sizes[dim] = self_dim_size;
  self_strides[dim] = self_dim_stride;
  auto self_restrided = self.as_strided(self_sizes, self_strides);

  if (self_restrided.numel() == 0 || index.numel() == 0) {
    return;
  }

  TORCH_CHECK(
      self_restrided.dim() <= XPU_MAX_TENSORINFO_DIMS,
      "self has too many (>",
      XPU_MAX_TENSORINFO_DIMS,
      ") dims");
  TORCH_CHECK(
      index.dim() <= XPU_MAX_TENSORINFO_DIMS,
      "index has too many (>",
      XPU_MAX_TENSORINFO_DIMS,
      ") dims");

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
      at::ScalarType::Bool,
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::ComplexHalf,
      self_restrided.scalar_type(),
      "index_fill_xpu",
      [&] {
        TensorInfo<const int64_t, int64_t> index_info =
            getTensorInfo<const int64_t, int64_t>(index);
        index_info.collapseDims();

        TensorInfo<scalar_t, int64_t> dst_info =
            getTensorInfo<scalar_t, int64_t>(self_restrided);
        int new_indexing_dim = dst_info.collapseDims(dim);

        // No used in index kernel frame for index_fill.
        auto src_info = TensorInfo<const scalar_t, int64_t>();

        using IdxConfig = IndexKernelConfig<
            decltype(src_info),
            decltype(dst_info),
            decltype(index_info),
            IndexFillScalarFunctor<scalar_t>>;
        using KernelClass = IndexKernel<IdxConfig, false, false>;

        auto cfg = IdxConfig::template make_config<KernelClass>(
            src_info,
            dst_info,
            index_info,
            source.to<scalar_t>(),
            new_indexing_dim,
            true,
            IndexFillScalarFunctor<scalar_t>());
        launch_index_kernel(cfg);
      });
}

template <typename scalar_t>
struct IndexPutAccumulateFunctor {
  void operator()(
      char* const out_data,
      const char* const in_data,
      int64_t offset) const {
    sycl_global_ptr<scalar_t> out_ptr = sycl_global_ptr<scalar_t>(
        reinterpret_cast<scalar_t*>(out_data + offset));
    auto in = *reinterpret_cast<const scalar_t*>(in_data);
    atomicAdd(out_ptr, in);
  }
};

template <typename scalar_t>
struct IndexPutFunctor {
  void operator()(
      char* const out_data,
      const char* const in_data,
      int64_t offset) const {
    *reinterpret_cast<scalar_t*>(out_data + offset) =
        *reinterpret_cast<const scalar_t*>(in_data);
  }
};

void index_put_kernel(
    TensorIterator& iter,
    IntArrayRef index_size,
    IntArrayRef index_stride,
    bool accumulate) {
  if (accumulate) {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
        at::ScalarType::ComplexHalf,
        at::ScalarType::BFloat16,
        at::ScalarType::Half,
        at::ScalarType::Bool,
        iter.dtype(),
        "index_put_xpu",
        [&] {
          IndexPutAccumulateFunctor<scalar_t> f;
          _index_kernel(
              iter, index_size, index_stride, IntArrayRef{}, IntArrayRef{}, f);
        });
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
        at::ScalarType::ComplexHalf,
        at::ScalarType::BFloat16,
        at::ScalarType::Half,
        at::ScalarType::Bool,
        iter.dtype(),
        "index_put_xpu",
        [&] {
          using dtype = OpaqueType<sizeof(scalar_t)>;
          IndexPutFunctor<dtype> f;
          _index_kernel(
              iter, index_size, index_stride, IntArrayRef{}, IntArrayRef{}, f);
        });
  }
}

void index_put_deterministic_kernel(
    Tensor& self,
    const c10::List<c10::optional<Tensor>>& indices,
    const Tensor& value,
    bool accumulate,
    bool unsafe) {
  TORCH_CHECK(
      !indices.empty() || is_expandable_to(value.sizes(), self.sizes()),
      "shape mismatch: value tensor of shape ",
      value.sizes(),
      " cannot be broadcast to indexing result of shape ",
      self.sizes());
  if (indices.size() > (size_t)self.dim()) {
    TORCH_CHECK_INDEX(
        false,
        "too many indices for tensor of dimension ",
        self.dim(),
        " (got ",
        indices.size(),
        ")");
  }
  bool self_contiguous = self.is_contiguous();
  auto self_ = self_contiguous ? self : self.contiguous();
  Tensor linearIndex, src, expandedValue = value;
  int64_t nElemBefore, strideBefore, sliceSize;
  std::vector<int64_t> inversePerm;
  std::tie(
      linearIndex, src, nElemBefore, strideBefore, sliceSize, inversePerm) =
      makeLinearIndex(self_, indices, !unsafe);
  int64_t num_indices = linearIndex.numel();

  if (expandedValue.numel() < num_indices * nElemBefore * sliceSize) {
    auto expanded_size = at::DimVector(expandedValue.sizes());

    auto size1 = expandedValue.sizes();
    auto size2 = linearIndex.sizes();
    if (are_expandable(size1, size2)) {
      expanded_size = infer_size_dimvector(size1, size2);
    }
    if (nElemBefore > 1) {
      expanded_size.insert(expanded_size.begin(), nElemBefore);
    }
    if (sliceSize > 1) {
      expanded_size.insert(expanded_size.end(), sliceSize);
    }
    expandedValue = expandedValue.expand(expanded_size);
  }
  expandedValue = expandedValue.contiguous();

  if (num_indices > 0 && sliceSize > 0) {
    const bool permuted = !src.is_contiguous();
    auto src_ = permuted ? src.contiguous() : src;
    linearIndex = linearIndex.reshape(-1);
    auto sorted_indices =
        at::empty_like(linearIndex, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    auto orig_indices =
        at::empty_like(linearIndex, LEGACY_CONTIGUOUS_MEMORY_FORMAT);

    linearIndex.divide_(sliceSize, "trunc");

    sorted_indices.copy_(linearIndex);
    pstl::itoa(
        orig_indices.data_ptr<int64_t>(),
        orig_indices.data_ptr<int64_t>() + linearIndex.numel(),
        (int64_t)0);
    pstl::sort<int64_t, int64_t>(
        linearIndex.const_data_ptr<int64_t>(),
        sorted_indices.data_ptr<int64_t>(),
        orig_indices.data_ptr<int64_t>(),
        linearIndex.numel(),
        false);
    TORCH_INTERNAL_ASSERT(
        linearIndex.numel() * sliceSize * nElemBefore == expandedValue.numel(),
        "number of flattened indices did not match number of elements in the value tensor: ",
        linearIndex.numel() * sliceSize * nElemBefore,
        " vs ",
        expandedValue.numel());
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
        at::ScalarType::ComplexHalf,
        at::ScalarType::BFloat16,
        at::ScalarType::Half,
        at::ScalarType::Bool,
        expandedValue.scalar_type(),
        "index_put_deterministic_kernel",
        [&] {
          launch_index_put_deterministic_kernel<scalar_t>(
              sorted_indices.mutable_data_ptr<int64_t>(),
              orig_indices.mutable_data_ptr<int64_t>(),
              expandedValue.const_data_ptr<scalar_t>(),
              src_.mutable_data_ptr<scalar_t>(),
              num_indices,
              sliceSize,
              strideBefore,
              nElemBefore,
              accumulate);
        });
    if (permuted)
      self.copy_(src_.permute(inversePerm));
    else if (!self_contiguous) {
      self.copy_(self_);
    }
  }
}

template <typename scalar_t>
struct MaskedScatterElementwiseFunctor {
  scalar_t operator()(
      const scalar_t a,
      const bool mask,
      const int64_t maskPrefixSum) const {
    if (mask) {
      return source_ptr_[maskPrefixSum];
    }
    return a;
  }
  MaskedScatterElementwiseFunctor(const scalar_t* source_ptr)
      : source_ptr_(source_ptr) {}

 private:
  const scalar_t* source_ptr_;
};

struct MaskedScatterSizeCheckFunctor {
  void operator()(sycl::nd_item<1> item) const {
    const auto totalElements = *mask_exclusive_sum_ + *mask_;
    SYCL_KERNEL_ASSERT(totalElements <= srcSize_);
  }
  MaskedScatterSizeCheckFunctor(
      const int64_t* const mask_exclusive_sum,
      const bool* const mask,
      const int64_t srcSize)
      : mask_exclusive_sum_(mask_exclusive_sum),
        mask_(mask),
        srcSize_(srcSize) {}

 private:
  const int64_t* const mask_exclusive_sum_;
  const bool* const mask_;
  const int64_t srcSize_;
};

void masked_scatter_kernel(
    const TensorBase& self,
    const TensorBase& mask,
    const TensorBase& maskPrefixSum,
    const TensorBase& source) {
  const auto srcSize = source.numel();
  const auto mask_cont = mask.contiguous();
  const auto mask_numel = mask.numel();

  // Use a prefix sum to determine the output locations of the masked elements
  auto maskPrefixSum_data = maskPrefixSum.mutable_data_ptr<int64_t>();
  auto mask_data = mask_cont.const_data_ptr<bool>();

  pstl::exclusive_scan(
      mask_data,
      mask_data + mask_numel,
      maskPrefixSum_data,
      static_cast<int64_t>(0));

  // Asynchronously check that the number of `1` elements present in the mask
  // must be <= the number of elements available in `src`.
  auto caller = MaskedScatterSizeCheckFunctor(
      &maskPrefixSum_data[mask_numel - 1], &mask_data[mask_numel - 1], srcSize);
  sycl_kernel_submit((size_t)1, (size_t)1, getCurrentSYCLQueue(), caller);

  // We are getting elements from `src` based on an offset from
  // `maskPrefixSum`, so that should be made contiguous too
  auto source_contig = source.contiguous();

  auto iter = TensorIteratorConfig()
                  .set_check_mem_overlap(false)
                  .check_all_same_dtype(false)
                  .resize_outputs(false)
                  .add_output(self)
                  .add_input(self)
                  .add_const_input(mask_cont)
                  .add_input(maskPrefixSum)
                  .build();

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      ScalarType::Bool,
      ScalarType::BFloat16,
      ScalarType::Half,
      self.scalar_type(),
      "masked_scatter_",
      [&]() {
        auto source_ptr = source_contig.const_data_ptr<scalar_t>();
        gpu_kernel(iter, MaskedScatterElementwiseFunctor<scalar_t>(source_ptr));
      });
}

template <typename ValType>
class IndexCopyScalarFunctor {
 public:
  void operator()(
      ValType* dst,
      const ValType* src,
      int64_t dst_off,
      int64_t src_off,
      int64_t idx,
      ValType alpha) const {
    dst[dst_off] = src[src_off];
  }
};

template <class SrcInfo, class DstInfo, class IdxInfo>
static inline void _index_copy_kernel(
    SrcInfo& src_info,
    DstInfo& dst_info,
    IdxInfo& index_info,
    int64_t dim) {
  using scalar_t = typename DstInfo::scalar_t;
  using IdxConfig = IndexKernelConfig<
      SrcInfo,
      DstInfo,
      IdxInfo,
      IndexCopyScalarFunctor<scalar_t>>;
  using KernelClass = IndexKernel<IdxConfig, false, false>;
  auto cfg = IdxConfig::template make_config<KernelClass>(
      src_info,
      dst_info,
      index_info,
      scalar_t{},
      dim,
      true,
      IndexCopyScalarFunctor<scalar_t>());
  launch_index_kernel(cfg);
}

template <typename scalar_t>
static inline void index_copy_impl(
    Tensor& dst,
    int64_t dim,
    const Tensor& indices,
    const Tensor& source) {
  static constexpr string_view DIM_WARNING =
      "Tensor too large or too many (> 12) dimensions";

  TORCH_CHECK(dst.dim() <= XPU_MAX_TENSORINFO_DIMS, DIM_WARNING);
  TORCH_CHECK(indices.dim() <= XPU_MAX_TENSORINFO_DIMS, DIM_WARNING);
  TORCH_CHECK(source.dim() <= XPU_MAX_TENSORINFO_DIMS, DIM_WARNING);

  // The `src` is partitioned into two parts:
  // -the size of each slice we are indexing, which is the
  // total size of the tensor ignoring dimension `dim`;
  // -the number of indices we are choosing, which is the total size
  // of the tensor `indices`.
  int64_t dstDims = dst.dim() == 0 ? 1 : dst.dim();

  TORCH_CHECK(dim >= 0 && dim < dstDims, "Indexing dim is out of bounds");

  ptrdiff_t sliceSize = 1;
  for (int d = 0; d < dstDims; d++) {
    if (d != dim) {
      sliceSize *= dst.dim() == 0 ? 1 : dst.size(d);
    }
  }
  if (sliceSize == 0) {
    return;
  }

  TensorInfo<const int64_t, int64_t> indices_info =
      getTensorInfo<const int64_t, int64_t>(indices);
  indices_info.collapseDims();

  TensorInfo<const scalar_t, int64_t> src_info =
      getTensorInfo<const scalar_t, int64_t>(source);

  TensorInfo<scalar_t, int64_t> dst_info =
      getTensorInfo<scalar_t, int64_t>(dst);
  auto collapse_dim = (dst.dim() == 0) ? -1 : dim;
  int new_indexing_dim = dst_info.collapseDims(collapse_dim);
  _index_copy_kernel(src_info, dst_info, indices_info, new_indexing_dim);
}

void index_copy_kernel(
    int64_t dim,
    const Tensor& index,
    const Tensor& source,
    Tensor& out) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND6(
      at::ScalarType::ComplexHalf,
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Bool,
      at::ScalarType::Float8_e4m3fn,
      at::ScalarType::Float8_e5m2,
      out.scalar_type(),
      "index_copy_xpu",
      [&]() { index_copy_impl<scalar_t>(out, dim, index, source); });
}

template <typename scalar_t, typename offset_cal_t>
struct IndexCopyLoopFunctor {
  void operator()(int i) const {
    const auto offsets = offset_calc_.get(i);
    auto self_data = reinterpret_cast<scalar_t*>(self_ptr_ + offsets[0]);
    auto idx = *reinterpret_cast<int64_t*>(idx_ptr_ + offsets[1]);
    auto source_data = reinterpret_cast<scalar_t*>(source_ptr_ + offsets[2]);
    SYCL_KERNEL_ASSERT(idx >= 0 && idx < self_dim_size_);
    self_data[idx * self_dim_stride_] = *source_data;
  }
  IndexCopyLoopFunctor(
      offset_cal_t offset_calc,
      char* self_ptr,
      char* idx_ptr,
      char* source_ptr,
      const int64_t self_dim_size,
      const int64_t self_dim_stride)
      : offset_calc_(offset_calc),
        self_ptr_(self_ptr),
        idx_ptr_(idx_ptr),
        source_ptr_(source_ptr),
        self_dim_size_(self_dim_size),
        self_dim_stride_(self_dim_stride) {}

 private:
  offset_cal_t offset_calc_;
  char* self_ptr_;
  char* idx_ptr_;
  char* source_ptr_;
  const int64_t self_dim_size_;
  const int64_t self_dim_stride_;
};

template <typename scalar_t>
void index_copy_kernel_impl(
    TensorIterator& iter,
    const int64_t dim,
    const int64_t self_dim_size,
    const int64_t self_dim_stride) {
  if (iter.numel() == 0) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      index_copy_kernel_impl<scalar_t>(
          sub_iter, dim, self_dim_size, self_dim_stride);
    }
    return;
  }

  char* self_ptr = reinterpret_cast<char*>(iter.data_ptr(0));
  char* idx_ptr = reinterpret_cast<char*>(iter.data_ptr(1));
  char* source_ptr = reinterpret_cast<char*>(iter.data_ptr(2));

  const auto offset_calc = make_offset_calculator<3>(iter);

  auto fn = IndexCopyLoopFunctor<scalar_t, decltype(offset_calc)>(
      offset_calc,
      self_ptr,
      idx_ptr,
      source_ptr,
      self_dim_size,
      self_dim_stride);
  launch_index_group_stride_kernel<4, decltype(fn)>(iter.numel(), fn);
}

void index_copy_kernel(
    TensorIterator& iter,
    const int64_t dim,
    const int64_t self_dim_size,
    const int64_t self_dim_stride) {
  // See note [Writing Nondeterministic Operations]
  // Nondeterministic when index contains duplicate entries
  // this kernel will not be called when
  // torch.use_deterministic_algorithms(True)
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
      at::ScalarType::Half,
      at::ScalarType::Bool,
      at::ScalarType::BFloat16,
      kComplexHalf,
      iter.dtype(),
      "index_copy_xpu",
      [&] {
        using dtype = OpaqueType<sizeof(scalar_t)>;
        index_copy_kernel_impl<dtype>(
            iter, dim, self_dim_size, self_dim_stride);
      });
}

template <
    typename scalar_t,
    typename index_t,
    typename func_t,
    typename offset_cal_t,
    typename idx_offset_cal_t>
struct TakePutLoopFunctor {
  void operator()(int i) const {
    const auto offsets = offset_calc_.get(i);
    auto& iterated = *reinterpret_cast<scalar_t*>(iterated_ptr_ + offsets[0]);
    const auto idx = *reinterpret_cast<int64_t*>(idx_ptr_ + offsets[1]);
    SYCL_KERNEL_ASSERT(
        idx < numel_ && idx >= -numel_ &&
        "take_put_kernel_template() index out of bounds");
    index_t offset = static_cast<index_t>(idx);
    if (offset < 0) {
      offset += numel_;
    }
    if (!is_contiguous_) {
      offset = offset_indexed_.get(offset)[0];
    }
    f_(iterated, offset);
  }

  TakePutLoopFunctor(
      offset_cal_t offset_calc,
      char* iterated_ptr,
      char* idx_ptr,
      int64_t numel,
      bool is_contiguous,
      idx_offset_cal_t offset_indexed,
      func_t f)
      : offset_calc_(offset_calc),
        iterated_ptr_(iterated_ptr),
        idx_ptr_(idx_ptr),
        numel_(numel),
        is_contiguous_(is_contiguous),
        offset_indexed_(offset_indexed),
        f_(f) {}

 private:
  offset_cal_t offset_calc_;
  char* iterated_ptr_;
  char* idx_ptr_;
  int64_t numel_;
  bool is_contiguous_;
  idx_offset_cal_t offset_indexed_;
  func_t f_;
};

template <typename scalar_t, typename index_t, typename func_t>
void take_put_kernel_template(
    TensorIterator& iter,
    const TensorBase& indexed,
    const func_t& f) {
  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      take_put_kernel_template<scalar_t, index_t>(sub_iter, indexed, f);
    }
    return;
  }

  const auto numel = indexed.numel();
  const bool is_contiguous = indexed.is_contiguous();

  auto iterated_ptr = reinterpret_cast<char*>(iter.data_ptr(0));
  auto idx_ptr = reinterpret_cast<char*>(iter.data_ptr(1));

  const auto offset_calc = make_offset_calculator<2>(iter);
  using uindex_t = std::make_unsigned_t<index_t>;

  // OffsetCalculator needs the sizes and strides reveresed
  const auto indexed_sizes =
      std::vector<int64_t>(indexed.sizes().rbegin(), indexed.sizes().rend());
  const auto indexed_strides = std::vector<int64_t>(
      indexed.strides().rbegin(), indexed.strides().rend());
  const auto* indexed_strides_data = indexed_strides.data();
  const auto offset_indexed = OffsetCalculator<1, uindex_t>(
      indexed.dim(), indexed_sizes.data(), &indexed_strides_data);

  auto N = iter.numel();
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }

  auto loop_fn = TakePutLoopFunctor<
      scalar_t,
      index_t,
      func_t,
      decltype(offset_calc),
      decltype(offset_indexed)>(
      offset_calc,
      iterated_ptr,
      idx_ptr,
      numel,
      is_contiguous,
      offset_indexed,
      f);
  auto caller =
      TakePutKernelFunctor<TAKE_PUT_UNROLL_SZIE, decltype(loop_fn)>(N, loop_fn);
  auto wg_sz = syclMaxWorkItemsPerEU();
  auto num_wg =
      (N + wg_sz * TAKE_PUT_UNROLL_SZIE - 1) / (wg_sz * TAKE_PUT_UNROLL_SZIE);
  sycl_kernel_submit(num_wg * wg_sz, wg_sz, getCurrentSYCLQueue(), caller);
}

template <typename scalar_t, typename index_t>
struct TakeFunctor {
  void operator()(scalar_t& iterated, const index_t offset) const {
    iterated = indexed_ptr_[offset];
  }
  TakeFunctor(const scalar_t* indexed_ptr) : indexed_ptr_(indexed_ptr) {}

 private:
  const scalar_t* indexed_ptr_;
};

void take_kernel(TensorIterator& iter, const TensorBase& input) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      at::ScalarType::Bool,
      iter.dtype(),
      "take_xpu",
      [&]() {
        AT_DISPATCH_INDEX_TYPES(
            canUse32BitIndexMath(input) ? ScalarType::Int : ScalarType::Long,
            "take_xpu_index",
            [&] {
              const scalar_t* indexed_ptr = input.template const_data_ptr<scalar_t>();
              TakeFunctor<scalar_t, index_t> f(indexed_ptr);
              take_put_kernel_template<scalar_t, index_t>(iter, input, f);
            });
      });
}

template <typename scalar_t, typename index_t>
struct PutFunctor {
  void operator()(scalar_t& iterated, const index_t offset) const {
    indexed_ptr_[offset] = iterated;
  }
  PutFunctor(scalar_t* indexed_ptr) : indexed_ptr_(indexed_ptr) {}

 private:
  scalar_t* indexed_ptr_;
};

template <typename scalar_t, typename index_t>
struct PutAccumulateFunctor {
  void operator()(scalar_t& iterated, const index_t offset) const {
    atomicAdd(sycl_global_ptr<scalar_t>(indexed_ptr_ + offset), iterated);
  }
  PutAccumulateFunctor(scalar_t* indexed_ptr) : indexed_ptr_(indexed_ptr) {}

 private:
  scalar_t* indexed_ptr_;
};

void put_kernel(
    TensorIterator& iter,
    const TensorBase& output,
    const bool accumulate) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      at::ScalarType::Bool,
      iter.dtype(),
      "put_xpu",
      [&] {
        AT_DISPATCH_INDEX_TYPES(
            canUse32BitIndexMath(output) ? ScalarType::Int : ScalarType::Long,
            "put_xpu_index",
            [&] {
              scalar_t* indexed_ptr = output.template data_ptr<scalar_t>();
              if (accumulate) {
                PutAccumulateFunctor<scalar_t, index_t> f(indexed_ptr);
                take_put_kernel_template<scalar_t, index_t>(iter, output, f);
              } else {
                PutFunctor<scalar_t, index_t> f(indexed_ptr);
                take_put_kernel_template<scalar_t, index_t>(iter, output, f);
              }
            });
      });
}

void index_reduce_kernel(
  const Tensor& self,
  int64_t dim,
  const Tensor& index,
  const Tensor& source,
  bool include_self,
  const ReductionType& reduce,
  const Tensor& result) {

  if (!result.is_same(self)) result.copy_(self);

  // Scalars are treated as 1-d tensor
  Tensor self_ = (result.dim() == 0) ? result.view(1) : result;
  Tensor source_ = (source.dim() == 0) ? source.view(1) : source;

  //Perform checkings
  int srcDims = source.dim() == 0 ? 1 : src.dim();
  int dstDims = result.dim();
  int idxDims = index.dim();

  TORCH_CHECK(
      srcDims <= XPU_MAX_TENSORINFO_DIMS,
      "source tensor dim should be < ",
      XPU_MAX_TENSORINFO_DIMS);
  TORCH_CHECK(
      dstDims <= XPU_MAX_TENSORINFO_DIMS,
      "result tensor dim should be < ",
      XPU_MAX_TENSORINFO_DIMS);
  TORCH_CHECK(
      idxDims <= XPU_MAX_TENSORINFO_DIMS,
      "index tensor dim should be < ",
      XPU_MAX_TENSORINFO_DIMS);

  if (!include_self) {            
    AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      self.scalar_type(), "index_reduce_func_xpu_exclude_input_init", [&] {
      scalar_t init_val;
      switch (reduce) {
        case ReductionType::PROD:
          init_val = (scalar_t) 1;
          using reduceFunctor = IndexReduceMultiplyFunctor<scalar_t>;
          break;
        case ReductionType::MAX:
          init_val = std::numeric_limits<scalar_t>::has_infinity ? 
                     -std::numeric_limits<scalar_t>::infinity()
                     : std::numeric_limits<scalar_t>::lowest();
          using reduceFunctor = IndexReduceMaxFunctor<scalar_t>;
          break;
        case ReductionType::MIN:
          init_val = std::numeric_limits<scalar_t>::has_infinity ? 
                     std::numeric_limits<scalar_t>::infinity()
                     : std::numeric_limits<scalar_t>::max();
          using reduceFunctor = IndexReduceMinFunctor<scalar_t>;
          break;
        default:
          init_val = (scalar_t) 0;
          break;
      }
      // index_fill_ requires index to be a LongTensor
      self_.index_fill_(dim, index.to(at::ScalarType::Long), init_val);
    });
  }

  ptrdiff_t sliceSize = getSliceSize(self_, dim, index, source_);
  // uint64_t sourceTotalSize = source.numel();
  // uint64_t selfReduceDimSize = self_.size(dim);
  // uint64_t numIndex = index.numel();
  // uint64_t selfNumel = self_.numel();

  //early exit
  if (sliceSize == 0) {return;}

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
      at::ScalarType::Bool,
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::ComplexHalf,
      source_.scalar_type(),
      "index_reduce_xpu",
      [&] {
        AT_DISPATCH_INDEX_TYPES(index.scalar_type(), "index_reduce_xpu", [&]() {
          TensorInfo<const index_t, int64_t> index_info =
              getTensorInfo<const index_t, int64_t>(index);
          index_info.collapseDims();

          TensorInfo<const scalar_t, int64_t> src_info =
              getTensorInfo<const scalar_t, int64_t>(source_);

          TensorInfo<scalar_t, int64_t> dst_info =
              getTensorInfo<scalar_t, int64_t>(self_);
          int new_indexing_dim = dst_info.collapseDims(dim);

          //What is the meaning of this?
          //Create a config / abstraction to contain metadata for tensors + hardware configurations (workgroup, policy etc)
          //for calculating the grid/problems. It inherits the BatchKernelConfig.
          using IdxConfig = IndexKernelConfig<
              decltype(src_info),
              decltype(dst_info),
              decltype(index_info),
              reduceFunctor>;
          using KernelClass = IndexKernel<IdxConfig, false, false>;

          auto cfg = IdxConfig::template make_config<KernelClass>(
              src_info,
              dst_info,
              index_info,
              alpha.to<scalar_t>(),
              new_indexing_dim,
              true,
              reduceFunctor());
          launch_index_kernel(cfg);
        });
      });  
}

} // namespace at::native::xpu

#pragma GCC diagnostic pop
#pragma clang diagnostic pop
