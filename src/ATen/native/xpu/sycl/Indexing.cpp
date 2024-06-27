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
      ValType* src,
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
  using scalar_t = typename SrcInfo::scalar_t;
  auto cfg = IndexKernelConfig<
      SrcInfo,
      DstInfo,
      IdxInfo,
      IndexSelectScalarFunctor<scalar_t>>::
      make_config(
          src_info,
          dst_info,
          index_info,
          static_cast<scalar_t>(0),
          dim,
          false,
          IndexSelectScalarFunctor<scalar_t>());
  if (cfg.problem_inner_) {
    launch_index_kernel<decltype(cfg), TrivialOffCal, true>(cfg);
  } else {
    launch_index_kernel<decltype(cfg), TrivialOffCal, false>(cfg);
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
    TensorInfo<index_t, int64_t> index_info =
        tensorInfoIfScalar(getTensorInfo<index_t, int64_t>(indices));
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
          TensorInfo<scalar_t, int64_t> src_info = tensorInfoIfScalar(
              getTensorInfo<scalar_t, int64_t>(src.contiguous()));
          int new_indexing_dim = src_info.collapseDims(dim);

          // Improve efficiency of generated native instructions for contiguous.
          // See comm/TensorInfo.h
          if (dst.is_contiguous() && indices.is_contiguous())
            _index_select_kernel<
                decltype(src_info),
                decltype(dst_info),
                decltype(index_info),
                /* TrivialOffCal */ true>(
                src_info, dst_info, index_info, new_indexing_dim);
          else
            _index_select_kernel<
                decltype(src_info),
                decltype(dst_info),
                decltype(index_info),
                /* TrivialOffCal */ false>(
                src_info, dst_info, index_info, new_indexing_dim);
        }),
        AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX),
        AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES),
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
      ValType* src,
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
      bool* src,
      int64_t dst_off,
      int64_t src_off,
      int64_t idx,
      bool alpha) const {
    atomicAdd((sycl_global_ptr<bool>)(dst + dst_off), src[src_off] && alpha);
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
      "index_add_kernel",
      [&] {
        TensorInfo<int64_t, int64_t> index_info =
            getTensorInfo<int64_t, int64_t>(index);
        index_info.collapseDims();

        TensorInfo<scalar_t, int64_t> src_info =
            getTensorInfo<scalar_t, int64_t>(source_);

        TensorInfo<scalar_t, int64_t> dst_info =
            getTensorInfo<scalar_t, int64_t>(self_);
        int new_indexing_dim = dst_info.collapseDims(dim);

        auto cfg = IndexKernelConfig<
            decltype(src_info),
            decltype(dst_info),
            decltype(index_info),
            IndexAddScalarFunctor<scalar_t>>::
            make_config(
                src_info,
                dst_info,
                index_info,
                alpha.to<scalar_t>(),
                new_indexing_dim,
                true,
                IndexAddScalarFunctor<scalar_t>());
        launch_index_kernel(cfg);
      });
}

template <typename scalar_t>
struct IndexPutAccumulateFunctor {
  void operator()(char* out_data, char* in_data, int64_t offset) const {
    sycl_global_ptr<scalar_t> out_ptr =
        sycl_global_ptr<scalar_t>((scalar_t*)(out_data + offset));
    auto in = *(scalar_t*)in_data;
    atomicAdd(out_ptr, in);
  }
};

template <typename scalar_t>
struct IndexPutFunctor {
  void operator()(char* out_data, char* in_data, int64_t offset) const {
    *(scalar_t*)(out_data + offset) = *(scalar_t*)in_data;
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
        linearIndex.data_ptr<int64_t>(),
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
              sorted_indices.data_ptr<int64_t>(),
              orig_indices.data_ptr<int64_t>(),
              expandedValue.data_ptr<scalar_t>(),
              src_.data_ptr<scalar_t>(),
              num_indices,
              sliceSize,
              strideBefore,
              nElemBefore,
              accumulate);
        });
    if (permuted)
      self.copy_(src_.permute(inversePerm));
  }
}

} // namespace at::native::xpu
#pragma GCC diagnostic pop
#pragma clang diagnostic pop
