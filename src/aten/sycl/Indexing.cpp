#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/CanUse32BitIndexMath.h>
#include <ATen/native/TensorIterator.h>
#include <aten/sycl/Loops.h>
#include <aten/sycl/TensorInfo.h>
#include <comm/SYCLContext.h>

namespace at {
namespace native {
namespace xpu {

using namespace at::xpu::detail;

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

// We prefer this kernel to avoid reloading index points if the number
// of indices is a small number.
// This kernel in fact works for all choices of problem size, but if
// the number of indices chosen is large, then the
// indexFuncLargeIndex kernel is a better choice to increase
// parallelism.
template <
    typename T,
    typename IndicesType,
    typename IndexType,
    int DstDim,
    int SrcDim,
    int IdxDim,
    typename func_t>
struct IndexFuncSmallIndexFunctor {
  void operator()(sycl::nd_item<1> item) const {
    // In order to avoid reloading the index that we are copying, load
    // it once to handle all of the points that are being selected, so
    // it can be reused as much as possible. This kernel is chosen when
    // this is a good choice (small number of chosen indices), since
    // re-accessing indices in addition to src elements can be slow.
    for (IndexType srcIndex = 0; srcIndex < indices.sizes[0]; ++srcIndex) {
      // Lua indices begin at 1
      IndexType dstIndex =
          indices.data[IndexToOffset<const IndicesType, IndexType, IdxDim>::get(
              srcIndex, indices)];

      // We stride over the output ignoring the indexed dimension
      // (innerSize), whose offset calculation is handled differently
      for (IndexType linearIndex = item.get_global_linear_id();
           linearIndex < innerSize;
           linearIndex += item.get_group_range(0) * item.get_local_range(0)) {
        IndexType dstOffset =
            IndexToOffset<T, IndexType, DstDim>::get(linearIndex, dst);
        dstOffset += dstIndex * dst.strides[dstAddDim];

        IndexType srcOffset =
            IndexToOffset<const T, IndexType, SrcDim>::get(linearIndex, src);
        srcOffset += srcIndex * src.strides[srcAddDim];

        T val;
        if constexpr (std::is_same<T, bool>::value) {
          val = src.data[srcOffset] && alpha;
        } else {
          val = src.data[srcOffset] * alpha;
        }
        op(dst.data, dstOffset, dstNumel, &val);
      }
    }
  }

  IndexFuncSmallIndexFunctor(
      TensorInfo<T, IndexType> dst,
      TensorInfo<const T, IndexType> src,
      TensorInfo<const IndicesType, IndexType> indices,
      int dstAddDim,
      int srcAddDim,
      IndexType innerSize,
      int64_t dstAddDimSize,
      int64_t dstNumel,
      func_t op,
      T alpha)
      : dst(dst),
        src(src),
        indices(indices),
        dstAddDim(dstAddDim),
        srcAddDim(srcAddDim),
        innerSize(innerSize),
        dstAddDimSize(dstAddDimSize),
        dstNumel(dstNumel),
        op(op),
        alpha(alpha) {}

 private:
  TensorInfo<T, IndexType> dst;
  TensorInfo<const T, IndexType> src;
  TensorInfo<const IndicesType, IndexType> indices;
  int dstAddDim;
  int srcAddDim;
  IndexType innerSize;
  int64_t dstAddDimSize;
  int64_t dstNumel;
  func_t op;
  T alpha;
};

// We prefer this kernel to balance parallelism across index points,
// if there are a large number of indices.
// This kernel in fact works for all choices of problem size, but if
// the number of indices chosen is small, then the
// indexFuncSmallIndex kernel is a better choice to reduce memory
// accesses.
template <
    typename T,
    typename IndicesType,
    typename IndexType,
    int DstDim,
    int SrcDim,
    int IdxDim,
    bool IndexIsMajor,
    typename func_t>
struct IndexFuncLargeIndexFunctor {
  void operator()(sycl::nd_item<1> item) const {
    // We stride over the output including the indexed dimension
    // (totalSize), and calculate the destination index point based on that
    for (IndexType linearIndex = item.get_global_linear_id();
         linearIndex < totalSize;
         linearIndex += item.get_group_range(0) * item.get_local_range(0)) {
      IndexType srcIndex, elementInSlice;
      if (IndexIsMajor) {
        srcIndex = linearIndex / innerSize;
        elementInSlice = linearIndex % innerSize;
      } else {
        elementInSlice = linearIndex / innerSize;
        srcIndex = linearIndex % innerSize;
      }

      // Lua indices begin at 1
      IndexType dstIndex =
          indices.data[IndexToOffset<const IndicesType, IndexType, IdxDim>::get(
              srcIndex, indices)];

      IndexType dstOffset =
          IndexToOffset<T, IndexType, DstDim>::get(elementInSlice, dst);
      dstOffset += dstIndex * dst.strides[dstAddDim];

      IndexType srcOffset =
          IndexToOffset<const T, IndexType, SrcDim>::get(elementInSlice, src);
      srcOffset += srcIndex * src.strides[srcAddDim];

      T val;
      if constexpr (std::is_same<T, bool>::value) {
        val = src.data[srcOffset] && alpha;
      } else {
        val = src.data[srcOffset] * alpha;
      }
      op(dst.data, dstOffset, dstNumel, &val);
    }
  }

  IndexFuncLargeIndexFunctor(
      TensorInfo<T, IndexType> dst,
      TensorInfo<const T, IndexType> src,
      TensorInfo<const IndicesType, IndexType> indices,
      int dstAddDim,
      int srcAddDim,
      IndexType totalSize,
      IndexType innerSize,
      int64_t dstAddDimSize,
      int64_t dstNumel,
      func_t op,
      T alpha)
      : dst(dst),
        src(src),
        indices(indices),
        dstAddDim(dstAddDim),
        srcAddDim(srcAddDim),
        totalSize(totalSize),
        innerSize(innerSize),
        dstAddDimSize(dstAddDimSize),
        dstNumel(dstNumel),
        op(op),
        alpha(alpha) {}

 private:
  TensorInfo<T, IndexType> dst;
  TensorInfo<const T, IndexType> src;
  TensorInfo<const IndicesType, IndexType> indices;
  int dstAddDim;
  int srcAddDim;
  IndexType totalSize;
  IndexType innerSize;
  int64_t dstAddDimSize;
  int64_t dstNumel;
  func_t op;
  T alpha;
};

template <typename scalar_t>
bool indexShouldBeMajor(
    TensorInfo<scalar_t, unsigned int>& info,
    int sliceDim) {
  // The stride between adjacent slices (e.g., between element #0 of slice #100
  // and element #0 of slice #101).
  unsigned int sliceStride = info.strides[sliceDim];

  for (const auto i : c10::irange(info.dims)) {
    if (i != sliceDim && info.sizes[i] > 1 && info.strides[i] < sliceStride) {
      return true;
    }
  }

  return false;
}

template <typename scalar_t>
struct ReduceAdd {
  inline void operator()(
      scalar_t* self_data_start,
      int64_t index,
      int64_t numel,
      const scalar_t* src_data) const {
    // TODO: enable fast atomic add
    sycl_global_ptr<scalar_t> out_ptr = {self_data_start + index};
    auto in = *src_data;
    atomicAdd(out_ptr, in);
  }
};

void index_add_impl(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& source,
    const Scalar& alpha,
    const Tensor& result) {
  if (!result.is_same(self)) {
    result.copy_(self);
  }

  // Scalars are treated as 1-d tensor
  const Tensor self_ = (result.dim() == 0) ? result.view(1) : result;
  const Tensor source_ = (source.dim() == 0) ? source.view(1) : source;

  TORCH_CHECK(
      result.dim() <= MAX_TENSORINFO_DIMS,
      "tensor has too many (>",
      MAX_TENSORINFO_DIMS,
      ") dims");
  TORCH_CHECK(
      source.dim() <= MAX_TENSORINFO_DIMS,
      "tensor has too many (>",
      MAX_TENSORINFO_DIMS,
      ") dims");
  TORCH_CHECK(
      index.dim() <= MAX_TENSORINFO_DIMS,
      "tensor has too many (>",
      MAX_TENSORINFO_DIMS,
      ") dims");

  if (globalContext().deterministicAlgorithms()) {
    torch::List<c10::optional<Tensor>> indices;
    indices.reserve(dim + 1);
    // for (auto i : c10::irange(dim)) {
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
  const ptrdiff_t sourceTotalSize = source.numel();
  const int64_t selfAddDimSize = self_.size(dim);
  const ptrdiff_t numIndex = index.numel();
  const int64_t selfNumel = self_.numel();

  if (sliceSize == 0) {
    return;
  }

  bool indContig = index.is_contiguous();
  const int ssc = syclMaxDSSNum();

#define SMALL_INDEX(                                                \
    TENSOR_TYPE, INDICES_TYPE, TYPE, SELF_DIM, SOURCE_DIM, IDX_DIM) \
  {                                                                 \
    auto caller = IndexFuncSmallIndexFunctor<                       \
        TENSOR_TYPE,                                                \
        INDICES_TYPE,                                               \
        TYPE,                                                       \
        SELF_DIM,                                                   \
        SOURCE_DIM,                                                 \
        IDX_DIM,                                                    \
        ReduceAdd<scalar_t>>(                                       \
        selfInfo,                                                   \
        sourceInfo,                                                 \
        indexInfo,                                                  \
        selfAddDim,                                                 \
        sourceAddDim,                                               \
        sliceSize,                                                  \
        selfAddDimSize,                                             \
        selfNumel,                                                  \
        ReduceAdd<scalar_t>(),                                      \
        alpha_value);                                               \
    sycl_kernel_submit(                                             \
        small_index_num_groups* small_index_group_size,             \
        small_index_group_size,                                     \
        getCurrentSYCLQueue(),                                      \
        caller);                                                    \
  }

#define LARGE_INDEX(                                    \
    TENSOR_TYPE,                                        \
    INDICES_TYPE,                                       \
    TYPE,                                               \
    SELF_DIM,                                           \
    SOURCE_DIM,                                         \
    IDX_DIM,                                            \
    IDX_IS_MAJOR)                                       \
  {                                                     \
    auto caller = IndexFuncLargeIndexFunctor<           \
        TENSOR_TYPE,                                    \
        INDICES_TYPE,                                   \
        TYPE,                                           \
        SELF_DIM,                                       \
        SOURCE_DIM,                                     \
        IDX_DIM,                                        \
        IDX_IS_MAJOR,                                   \
        ReduceAdd<scalar_t>>(                           \
        selfInfo,                                       \
        sourceInfo,                                     \
        indexInfo,                                      \
        selfAddDim,                                     \
        sourceAddDim,                                   \
        sourceTotalSize,                                \
        (IDX_IS_MAJOR) ? sliceSize : numIndex,          \
        selfAddDimSize,                                 \
        selfNumel,                                      \
        ReduceAdd<scalar_t>(),                          \
        alpha_value);                                   \
    sycl_kernel_submit(                                 \
        large_index_num_groups* large_index_group_size, \
        large_index_group_size,                         \
        getCurrentSYCLQueue(),                          \
        caller);                                        \
  }

  auto small_index_num_groups =
      std::min(ceil_div(sliceSize, (ptrdiff_t)256), (ptrdiff_t)(ssc * 8));
  auto small_index_group_size = std::min(sliceSize, (ptrdiff_t)256);
  auto large_index_num_groups =
      std::min(ceil_div(sourceTotalSize, (ptrdiff_t)256), (ptrdiff_t)(ssc * 8));
  auto large_index_group_size = std::min(sourceTotalSize, (ptrdiff_t)256);

  if (canUse32BitIndexMath(result) && canUse32BitIndexMath(source) &&
      canUse32BitIndexMath(index)) {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
        at::ScalarType::Bool,
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        at::ScalarType::ComplexHalf,
        result.scalar_type(),
        "index_add",
        [&] {
          TensorInfo<scalar_t, unsigned int> selfInfo =
              getTensorInfo<scalar_t, unsigned int>(self_);
          const int selfAddDim = selfInfo.collapseDims(dim);
          selfInfo.reduceDim(selfAddDim);
          const auto alpha_value = alpha.to<scalar_t>();
          AT_DISPATCH_INDEX_TYPES(index.scalar_type(), "index_add_xpu_", [&]() {
            auto sourceInfo =
                getTensorInfo<const scalar_t, unsigned int>(source_);
            const int sourceAddDim = sourceInfo.collapseDims(dim);
            sourceInfo.reduceDim(sourceAddDim);

            auto indexInfo = getTensorInfo<const index_t, unsigned int>(index);
            indexInfo.collapseDims();

            // A reasonable choice for when to have each thread iterate over
            // index to choose
            if (numIndex <= 16) {
              if (selfInfo.dims == 1 && sourceInfo.dims == 1 && indContig) {
                SMALL_INDEX(scalar_t, index_t, unsigned int, 1, 1, -2);
              } else if (
                  selfInfo.dims == 2 && sourceInfo.dims == 2 && indContig) {
                SMALL_INDEX(scalar_t, index_t, unsigned int, 2, 2, -2);
              } else if (
                  selfInfo.dims == 3 && sourceInfo.dims == 3 && indContig) {
                SMALL_INDEX(scalar_t, index_t, unsigned int, 3, 3, -2);
              } else {
                SMALL_INDEX(scalar_t, index_t, unsigned int, -1, -1, -1);
              }
            } else {
              const bool indexIsMajor =
                  indexShouldBeMajor(selfInfo, selfAddDim);

              if (selfInfo.dims == 1 && sourceInfo.dims == 1 && indContig) {
                LARGE_INDEX(scalar_t, index_t, unsigned int, 1, 1, -2, true);
              } else if (
                  selfInfo.dims == 2 && sourceInfo.dims == 2 && indContig) {
                if (indexIsMajor) {
                  LARGE_INDEX(scalar_t, index_t, unsigned int, 2, 2, -2, true);
                } else {
                  LARGE_INDEX(scalar_t, index_t, unsigned int, 2, 2, -2, false);
                }
              } else if (
                  selfInfo.dims == 3 && sourceInfo.dims == 3 && indContig) {
                if (indexIsMajor) {
                  LARGE_INDEX(scalar_t, index_t, unsigned int, 3, 3, -2, true);
                } else {
                  LARGE_INDEX(scalar_t, index_t, unsigned int, 3, 3, -2, false);
                }
              } else {
                LARGE_INDEX(scalar_t, index_t, unsigned int, -1, -1, -1, true);
              }
            }
          });
        });
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
        at::ScalarType::Bool,
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        self.scalar_type(),
        "index_add",
        [&] {
          TensorInfo<scalar_t, uint64_t> selfInfo =
              getTensorInfo<scalar_t, uint64_t>(self_);
          const int selfAddDim = selfInfo.collapseDims(dim);
          selfInfo.reduceDim(selfAddDim);
          const auto alpha_value = alpha.to<scalar_t>();

          TensorInfo<const scalar_t, uint64_t> sourceInfo =
              getTensorInfo<const scalar_t, uint64_t>(source_);
          const int sourceAddDim = sourceInfo.collapseDims(dim);
          sourceInfo.reduceDim(sourceAddDim);

          AT_DISPATCH_INDEX_TYPES(index.scalar_type(), "index_add_xpu_", [&]() {
            TensorInfo<const index_t, uint64_t> indexInfo =
                getTensorInfo<const index_t, uint64_t>(index);
            indexInfo.collapseDims();

            LARGE_INDEX(scalar_t, index_t, uint64_t, -1, -1, -1, true);
          });
        });
  }

#undef SMALL_INDEX
#undef LARGE_INDEX
}

} // namespace xpu
} // namespace native
} // namespace at
