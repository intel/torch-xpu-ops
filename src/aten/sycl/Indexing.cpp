#pragma clang diagnostic push
#pragma GCC diagnostic push
// Avoid SYCL compiler return-type error
#pragma clang diagnostic ignored "-Wreturn-type"
#pragma GCC diagnostic ignored "-Wreturn-type"

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

namespace impl {
// Pretend that the scalar tensor is in fact a one-element vector.
template <typename T, typename IndexType>
TensorInfo<T, IndexType> tensorInfoIfScalar(TensorInfo<T, IndexType> ti) {
  if (ti.dims == 0) {
    ti.dims = 1;
    ti.sizes[0] = 1;
    ti.strides[0] = 1;
  }
  return ti;
}

template <class SrcInfo, class DstInfo, class IdxInfo, class FuncType>
class IndexKernelConfig : public BatchKernelConfig {
 public:
  using ValType = typename SrcInfo::scalar_t;
  using IdxType = typename IdxInfo::scalar_t;

  IndexKernelConfig() = delete;
  IndexKernelConfig(
      SrcInfo& sinfo,
      DstInfo& dinfo,
      IdxInfo& iinfo,
      ValType alpha,
      int64_t index_num,
      int64_t indexing_dimension_size,
      bool indexing_dst,
      bool problem_inner,
      FuncType func,
      int64_t batch,
      int64_t problem,
      int64_t stride,
      int64_t problem_batch,
      bool problem_along_x)
      : BatchKernelConfig(
            batch,
            problem,
            stride,
            problem_batch,
            problem_along_x,
            Policy::pSegment,
            syclMaxWorkItemsPerEU()),
        sinfo_(sinfo),
        dinfo_(dinfo),
        iinfo_(iinfo),
        alpha_(alpha),
        index_num_(index_num),
        indexing_dimension_size_(indexing_dimension_size),
        indexing_dst_(indexing_dst),
        problem_inner_(problem_inner),
        func_(func) {}

  template <class TarInfo>
  static inline void indexing_problem_mapping(
      TarInfo& tinfo,
      IdxInfo& iinfo,
      int dim,
      int64_t index_num,
      int64_t indexing_dimension_size,
      int64_t& batch,
      int64_t& problem,
      int64_t& stride,
      int64_t& problem_batch,
      bool& problem_along_x,
      bool& problem_inner) {
    int64_t outer = tinfo.outerSize(dim);
    int64_t inner = tinfo.innerSize(dim);

    if (inner == 1) {
      problem = outer;
      stride = indexing_dimension_size;
      batch = 1;
      problem_batch = index_num;
      problem_along_x = tinfo.strides[dim] == 1 ? false : true;
      problem_inner = false;
    } else if (outer == 1) {
      problem = inner;
      stride = 1;
      batch = indexing_dimension_size;
      problem_batch = index_num;
      problem_along_x = tinfo.strides[tinfo.dims - 1] == 1 ? true : false;
      problem_inner = true;
    } else {
      problem = inner;
      stride = 1;
      batch = outer * indexing_dimension_size;
      problem_batch = outer * index_num;
      problem_along_x = tinfo.strides[tinfo.dims - 1] == 1 ? true : false;
      problem_inner = true;
    }
    return;
  }

  static IndexKernelConfig<SrcInfo, DstInfo, IdxInfo, FuncType> make_config(
      SrcInfo& src_info,
      DstInfo& dst_info,
      IdxInfo& index_info,
      ValType alpha,
      int64_t dim,
      bool indexing_dst,
      FuncType func) {
    int64_t index_num = index_info.sizes[0];
    int64_t indexing_dimension_size;

    bool problem_along_x, problem_inner;
    int64_t batch, problem, stride, problem_batch;

    TORCH_CHECK(
        indexing_dst || src_info.data != nullptr,
        "Indexing kernel backbone does not support null src ...");

    if (indexing_dst) {
      indexing_dimension_size = dst_info.sizes[dim];
      indexing_problem_mapping(
          dst_info,
          index_info,
          dim,
          index_num,
          indexing_dimension_size,
          batch,
          problem,
          stride,
          problem_batch,
          problem_along_x,
          problem_inner);
    } else {
      indexing_dimension_size = src_info.sizes[dim];
      indexing_problem_mapping(
          src_info,
          index_info,
          dim,
          index_num,
          indexing_dimension_size,
          batch,
          problem,
          stride,
          problem_batch,
          problem_along_x,
          problem_inner);
    }

    return {
        src_info,
        dst_info,
        index_info,
        alpha,
        index_num,
        indexing_dimension_size,
        indexing_dst,
        problem_inner,
        func,
        batch,
        problem,
        stride,
        problem_batch,
        problem_along_x};
  }

 public:
  SrcInfo sinfo_; // sinfo_.data could be nullptr, while indexing along dst.
  DstInfo dinfo_;
  IdxInfo iinfo_;
  ValType alpha_;
  int64_t index_num_;
  int64_t indexing_dimension_size_;
  bool indexing_dst_;
  bool problem_inner_;
  FuncType func_;
};

template <
    class IdxConfig,
    bool TrivialOffCal = false,
    bool known_problem_inner = false>
class IndexKernel {
 public:
  using ValType = typename IdxConfig::ValType;
  using IdxType = typename IdxConfig::IdxType;

  IndexKernel() = delete;
  IndexKernel(IdxConfig& cfg) : cfg_(cfg) {}

  void init_global_batch_info(
      BatchKernelConfig::ItemDesc& id,
      int64_t& idx_logical_off,
      int64_t& glb_batch_group,
      int64_t& glb_batch_group_loc_off) const {
    idx_logical_off = id.glb_batch % cfg_.index_num_;
    int64_t idx_off;
    if constexpr (TrivialOffCal) {
      idx_off = idx_logical_off;
    } else {
      idx_off = IndexToOffset<IdxType, int64_t>::get(
          idx_logical_off,
          cfg_.iinfo_,
          IndexToOffset<IdxType, int64_t>::NON_STRICT_CONTIGUOUS);
    }
    glb_batch_group = id.glb_batch / cfg_.index_num_;
    glb_batch_group_loc_off = cfg_.iinfo_.data[idx_off];
    glb_batch_group_loc_off = glb_batch_group_loc_off >= 0
        ? glb_batch_group_loc_off
        : cfg_.indexing_dimension_size_ + glb_batch_group_loc_off;
  }

  int64_t inline indexing_logical_off(
      BatchKernelConfig::ItemDesc& id,
      int64_t glb_batch_group,
      int64_t glb_batch_group_loc_off) const {
    int64_t si, pi, bi;
    int64_t glb_batch_group_glb_off =
        glb_batch_group * cfg_.indexing_dimension_size_ +
        glb_batch_group_loc_off;
    auto stride = cfg_.stride_;
    if constexpr (known_problem_inner) {
      si = 0;
      pi = id.glb_problem;
      bi = glb_batch_group_glb_off;
      return (pi + bi * cfg_.problem_) * stride;
    } else {
      if (cfg_.problem_inner_) {
        si = 0;
        pi = id.glb_problem;
        bi = glb_batch_group_glb_off;
        return (pi + bi * cfg_.problem_) * stride;
      } else {
        si = glb_batch_group_glb_off;
        pi = id.glb_problem;
        bi = 0;
        return si + pi * stride;
      }
    }
  }

  int64_t inline fixing_logical_off(
      BatchKernelConfig::ItemDesc& id,
      int64_t glb_batch_group,
      int64_t idx_logical_off) const {
    int64_t si, pi, bi, stride;
    int64_t glb_batch_group_glb_off =
        glb_batch_group * cfg_.index_num_ + idx_logical_off;
    if constexpr (known_problem_inner) {
      si = 0;
      stride = 1;
      pi = id.glb_problem;
      bi = glb_batch_group_glb_off;
      return pi + bi * cfg_.problem_;
    } else {
      if (cfg_.problem_inner_) {
        si = 0;
        stride = 1;
        pi = id.glb_problem;
        bi = glb_batch_group_glb_off;
        return pi + bi * cfg_.problem_;
      } else {
        bi = 0;
        si = glb_batch_group_glb_off;
        pi = id.glb_problem;
        stride = cfg_.index_num_;
        return si + pi * stride;
      }
    }
  }

  void operator()(sycl::nd_item<2> item) const {
    auto id = cfg_.get_item_desc(item);

    if (id.glb_problem >= cfg_.problem_ ||
        id.glb_batch >= cfg_.problem_batch_) {
      return;
    }

    // index kernel has three operands,
    // 1. index operand
    // 2. operand indexing on
    // 3. operand has fixing size as index (optional)
    int64_t indexing_si, indexing_pi, indexing_bi;
    int64_t fixing_si, fixing_pi, fixing_bi;
    int64_t idx_logical_off, glb_batch_group, glb_batch_group_loc_off;
    int64_t glb_indexing_logical_off, glb_fixing_logical_off;
    int64_t glb_indexing_off, glb_fixing_off;
    int64_t dst_off, src_off;

    init_global_batch_info(
        id, idx_logical_off, glb_batch_group, glb_batch_group_loc_off);

    glb_indexing_logical_off =
        indexing_logical_off(id, glb_batch_group, glb_batch_group_loc_off);

    if (cfg_.sinfo_.data != nullptr && cfg_.dinfo_.data != nullptr) {
      glb_fixing_logical_off =
          fixing_logical_off(id, glb_batch_group, idx_logical_off);
    }

    if constexpr (TrivialOffCal) {
      if (cfg_.indexing_dst_) {
        dst_off = glb_indexing_logical_off;
        if (cfg_.sinfo_.data != nullptr) {
          src_off = glb_fixing_logical_off;
        }
      } else {
        src_off = glb_indexing_logical_off;
        dst_off = glb_fixing_logical_off;
      }
    } else {
      if (cfg_.indexing_dst_) {
        // index_copy, index_add, index_fill
        dst_off = IndexToOffset<ValType, int64_t>::get(
            glb_indexing_logical_off,
            cfg_.dinfo_,
            IndexToOffset<ValType, int64_t>::NON_STRICT_CONTIGUOUS);
        if (cfg_.sinfo_.data != nullptr) {
          src_off = IndexToOffset<ValType, int64_t>::get(
              glb_fixing_logical_off,
              cfg_.sinfo_,
              IndexToOffset<ValType, int64_t>::NON_STRICT_CONTIGUOUS);
        }
      } else {
        // index_select
        src_off = IndexToOffset<ValType, int64_t>::get(
            glb_indexing_logical_off,
            cfg_.sinfo_,
            IndexToOffset<ValType, int64_t>::NON_STRICT_CONTIGUOUS);
        dst_off = IndexToOffset<ValType, int64_t>::get(
            glb_fixing_logical_off,
            cfg_.dinfo_,
            IndexToOffset<ValType, int64_t>::NON_STRICT_CONTIGUOUS);
      }
    }
    cfg_.func_(
        cfg_.dinfo_.data,
        cfg_.sinfo_.data,
        dst_off,
        src_off,
        glb_batch_group_loc_off,
        cfg_.alpha_);
  }

 private:
  IdxConfig cfg_;
};

template <
    class IdxConfig,
    bool TrivialOffCal = false,
    bool known_problem_inner = false>
static inline void launch_index_kernel(IdxConfig& cfg) {
  auto& queue = getCurrentSYCLQueue();
  IndexKernel<IdxConfig, TrivialOffCal, known_problem_inner> idx_ker(cfg);
  sycl_kernel_submit(cfg.global_size(), cfg.group_size(), queue, idx_ker);
}

template <typename ValType>
class IndexSelectOperator {
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
      IndexSelectOperator<scalar_t>>::
      make_config(
          src_info,
          dst_info,
          index_info,
          static_cast<scalar_t>(0),
          dim,
          false,
          IndexSelectOperator<scalar_t>());
  if (cfg.problem_inner_) {
    launch_index_kernel<decltype(cfg), TrivialOffCal, true>(cfg);
  } else {
    launch_index_kernel<decltype(cfg), TrivialOffCal, false>(cfg);
  }
}

template <typename scalar_t>
void IndexSelect(
    const Tensor& dst,
    const Tensor& src,
    int dim,
    const Tensor& indices) {
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
    TensorInfo<index_t, int64_t> indices_info =
        tensorInfoIfScalar(getTensorInfo<index_t, int64_t>(indices));
    indices_info.collapseDims();

    auto new_size = src.sizes().vec();

    if (src.dim() > 0) {
      new_size[dim] = indices.numel();
    }

    at::native::resize_output(dst, new_size);

    ptrdiff_t dst_num_elem = dst.numel();
    if (dst_num_elem == 0) {
      return;
    }

    TensorInfo<scalar_t, int64_t> dst_info =
        tensorInfoIfScalar(getTensorInfo<scalar_t, int64_t>(dst));
    TensorInfo<scalar_t, int64_t> src_info =
        tensorInfoIfScalar(getTensorInfo<scalar_t, int64_t>(src.contiguous()));
    int new_indexing_dim = src_info.collapseDims(dim);

    if (dst.is_contiguous() && indices.is_contiguous())
      _index_select_kernel<
          decltype(src_info),
          decltype(dst_info),
          decltype(indices_info),
          /* TrivialOffCal */ true>(
          src_info, dst_info, indices_info, new_indexing_dim);
    else
      _index_select_kernel<
          decltype(src_info),
          decltype(dst_info),
          decltype(indices_info),
          /* TrivialOffCal */ false>(
          src_info, dst_info, indices_info, new_indexing_dim);
  });
  return;
}

} // namespace impl

Tensor& index_select_out_kernel(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    Tensor& out) {
  AT_DISPATCH_V2(
      self.scalar_type(),
      "index_select",
      AT_WRAP([=]() { impl::IndexSelect<scalar_t>(out, self, dim, index); }),
      AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX),
      AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES),
      at::ScalarType::ComplexHalf,
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Bool);
  return out;
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
    for (IndexType srcIndex = 0; srcIndex < indices_.sizes[0]; ++srcIndex) {
      // Lua indices begin at 1
      IndexType dstIndex =
          indices_
              .data[IndexToOffset<const IndicesType, IndexType, IdxDim>::get(
                  srcIndex, indices_)];

      // We stride over the output ignoring the indexed dimension
      // (innerSize), whose offset calculation is handled differently
      for (IndexType linearIndex = item.get_global_linear_id();
           linearIndex < innerSize_;
           linearIndex += item.get_group_range(0) * item.get_local_range(0)) {
        IndexType dstOffset =
            IndexToOffset<T, IndexType, DstDim>::get(linearIndex, dst_);
        dstOffset += dstIndex * dst_.strides[dstAddDim_];

        IndexType srcOffset =
            IndexToOffset<const T, IndexType, SrcDim>::get(linearIndex, src_);
        srcOffset += srcIndex * src_.strides[srcAddDim_];

        T val;
        if constexpr (std::is_same<T, bool>::value) {
          val = src_.data[srcOffset] && alpha_;
        } else {
          val = src_.data[srcOffset] * alpha_;
        }
        op_(dst_.data, dstOffset, dstNumel_, &val);
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
      : dst_(dst),
        src_(src),
        indices_(indices),
        dstAddDim_(dstAddDim),
        srcAddDim_(srcAddDim),
        innerSize_(innerSize),
        dstAddDimSize_(dstAddDimSize),
        dstNumel_(dstNumel),
        op_(op),
        alpha_(alpha) {}

 private:
  TensorInfo<T, IndexType> dst_;
  TensorInfo<const T, IndexType> src_;
  TensorInfo<const IndicesType, IndexType> indices_;
  int dstAddDim_;
  int srcAddDim_;
  IndexType innerSize_;
  int64_t dstAddDimSize_;
  int64_t dstNumel_;
  func_t op_;
  T alpha_;
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
         linearIndex < totalSize_;
         linearIndex += item.get_group_range(0) * item.get_local_range(0)) {
      IndexType srcIndex, elementInSlice;
      if (IndexIsMajor) {
        srcIndex = linearIndex / innerSize_;
        elementInSlice = linearIndex % innerSize_;
      } else {
        elementInSlice = linearIndex / innerSize_;
        srcIndex = linearIndex % innerSize_;
      }

      // Lua indices begin at 1
      IndexType dstIndex =
          indices_
              .data[IndexToOffset<const IndicesType, IndexType, IdxDim>::get(
                  srcIndex, indices_)];

      IndexType dstOffset =
          IndexToOffset<T, IndexType, DstDim>::get(elementInSlice, dst_);
      dstOffset += dstIndex * dst_.strides[dstAddDim_];

      IndexType srcOffset =
          IndexToOffset<const T, IndexType, SrcDim>::get(elementInSlice, src_);
      srcOffset += srcIndex * src_.strides[srcAddDim_];

      T val;
      if constexpr (std::is_same<T, bool>::value) {
        val = src_.data[srcOffset] && alpha_;
      } else {
        val = src_.data[srcOffset] * alpha_;
      }
      op_(dst_.data, dstOffset, dstNumel_, &val);
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
      : dst_(dst),
        src_(src),
        indices_(indices),
        dstAddDim_(dstAddDim),
        srcAddDim_(srcAddDim),
        totalSize_(totalSize),
        innerSize_(innerSize),
        dstAddDimSize_(dstAddDimSize),
        dstNumel_(dstNumel),
        op_(op),
        alpha_(alpha) {}

 private:
  TensorInfo<T, IndexType> dst_;
  TensorInfo<const T, IndexType> src_;
  TensorInfo<const IndicesType, IndexType> indices_;
  int dstAddDim_;
  int srcAddDim_;
  IndexType totalSize_;
  IndexType innerSize_;
  int64_t dstAddDimSize_;
  int64_t dstNumel_;
  func_t op_;
  T alpha_;
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

#pragma GCC diagnostic pop
#pragma clang diagnostic pop
