#pragma clang diagnostic push
#pragma GCC diagnostic push
// Avoid SYCL compiler return-type error
#pragma clang diagnostic ignored "-Wreturn-type"
#pragma GCC diagnostic ignored "-Wreturn-type"

#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/ceil_div.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/Atomics.h>
#include <ATen/native/xpu/sycl/Indexing.h>
#include <ATen/native/xpu/sycl/IndexingUtils.h>
#include <ATen/native/xpu/sycl/Loops.h>
#include <ATen/native/xpu/sycl/SortingKernels.h>
#include <ATen/native/xpu/sycl/pstl/PSTLFunctions.h>
#include <ATen/ops/_sparse_coo_tensor_with_dims_and_tensors.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/gather.h>
#include <ATen/ops/ones_like.h>
#include <ATen/ops/zeros_like.h>

#include <comm/SYCLContext.h>
#include <comm/TensorInfo.h>
#include <sycl/sycl.hpp>

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
  AT_DISPATCH_V2(
      iter.dtype(),
      "index_xpu",
      AT_WRAP([&] {
        using dtype = OpaqueType<sizeof(scalar_t)>;
        IndexFunctor<dtype> f;
        _index_kernel(
            iter,
            index_size,
            index_stride,
            IntArrayRef{},
            IntArrayRef{},
            f,
            true);
      }),
      AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX),
      AT_EXPAND(AT_FLOAT8_TYPES),
      kComplexHalf,
      kHalf,
      kBool,
      kBFloat16);
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
              iter,
              index_size,
              index_stride,
              IntArrayRef{},
              IntArrayRef{},
              f,
              false);
        });
  } else {
    AT_DISPATCH_V2(
        iter.dtype(),
        "index_put_xpu",
        AT_WRAP([&] {
          using dtype = OpaqueType<sizeof(scalar_t)>;
          IndexPutFunctor<dtype> f;
          _index_kernel(
              iter,
              index_size,
              index_stride,
              IntArrayRef{},
              IntArrayRef{},
              f,
              false);
        }),
        AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX),
        AT_EXPAND(AT_FLOAT8_TYPES),
        kComplexHalf,
        kHalf,
        kBool,
        kBFloat16);
  }
}

DimVector valsShape(
    IntArrayRef self_sizes,
    int64_t dims_before,
    int64_t dims_indexed,
    IntArrayRef replacement_shape) {
  auto shape = DimVector(self_sizes);
  int64_t end = dims_before + dims_indexed;
  shape.erase(shape.begin() + dims_before, shape.begin() + end);
  shape.insert(
      shape.begin() + dims_before,
      replacement_shape.begin(),
      replacement_shape.end());
  return shape;
}

void index_put_deterministic_kernel(
    Tensor& self,
    const c10::List<std::optional<Tensor>>& indices,
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
  int64_t nElemBefore, strideBefore, sliceSize, dims_before, dims_indexed;
  std::vector<int64_t> inversePerm;
  std::tie(
      linearIndex,
      src,
      nElemBefore,
      strideBefore,
      sliceSize,
      inversePerm,
      dims_before,
      dims_indexed) = makeLinearIndex(self_, indices, !unsafe);
  auto vals_shape =
      valsShape(src.sizes(), dims_before, dims_indexed, linearIndex.sizes());
  int64_t num_indices = linearIndex.numel();
  expandedValue = expandedValue.expand(vals_shape).contiguous();

  if (num_indices > 0 && sliceSize > 0) {
    const bool permuted = !src.is_contiguous();
    auto src_ = permuted ? src.contiguous() : src;
    linearIndex = linearIndex.reshape(-1);
    auto sorted_indices =
        at::empty_like(linearIndex, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    auto orig_indices =
        at::empty_like(linearIndex, LEGACY_CONTIGUOUS_MEMORY_FORMAT);

    linearIndex.divide_(sliceSize, "trunc");

    auto range = at::arange(num_indices, linearIndex.options());
    sort_pairs<int64_t, int64_t>(
        linearIndex.const_data_ptr<int64_t>(),
        sorted_indices.mutable_data_ptr<int64_t>(),
        range.const_data_ptr<int64_t>(),
        orig_indices.mutable_data_ptr<int64_t>(),
        num_indices,
        false);

    TORCH_INTERNAL_ASSERT(
        linearIndex.numel() * sliceSize * nElemBefore == expandedValue.numel(),
        "number of flattened indices did not match number of elements in the value tensor: ",
        linearIndex.numel() * sliceSize * nElemBefore,
        " vs ",
        expandedValue.numel());
    if (sliceSize > SIMD) {
      AT_DISPATCH_V2(
          expandedValue.scalar_type(),
          "index_put_deterministic_kernel",
          AT_WRAP([&] {
            launch_index_put_deterministic_kernel<scalar_t, scalar_t>(
                sorted_indices.mutable_data_ptr<int64_t>(),
                orig_indices.mutable_data_ptr<int64_t>(),
                expandedValue.const_data_ptr<scalar_t>(),
                src_.mutable_data_ptr<scalar_t>(),
                num_indices,
                sliceSize,
                strideBefore,
                nElemBefore,
                accumulate);
          }),
          AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX),
          // TODO: Enable AT_FLOAT8_DTYPES after accumulation behavior is
          // cleared for float8 dtypes.
          kFloat8_e4m3fn,
          kFloat8_e5m2,
          kFloat8_e4m3fnuz,
          kFloat8_e5m2fnuz,
          kComplexHalf,
          kHalf,
          kBool,
          kBFloat16);
    } else {
      // Align acc type with CUDA
      AT_DISPATCH_V2(
          expandedValue.scalar_type(),
          "index_put_deterministic_kernel",
          AT_WRAP([&] {
            using accscalar_t = at::opmath_type<scalar_t>;
            launch_index_put_deterministic_kernel<scalar_t, accscalar_t>(
                sorted_indices.mutable_data_ptr<int64_t>(),
                orig_indices.mutable_data_ptr<int64_t>(),
                expandedValue.const_data_ptr<scalar_t>(),
                src_.mutable_data_ptr<scalar_t>(),
                num_indices,
                sliceSize,
                strideBefore,
                nElemBefore,
                accumulate);
          }),
          AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX),
          // TODO: Enable AT_FLOAT8_DTYPES after accumulation behavior is
          // cleared for float8 dtypes.
          kFloat8_e4m3fn,
          kFloat8_e5m2,
          kFloat8_e4m3fnuz,
          kFloat8_e5m2fnuz,
          kComplexHalf,
          kHalf,
          kBool,
          kBFloat16);
    }

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
  static constexpr std::string_view DIM_WARNING =
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
  auto wg_sz = syclMaxWorkItemsPerSubSlice();
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
              const scalar_t* indexed_ptr =
                  input.template const_data_ptr<scalar_t>();
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
  // Nondeterministic when index contains duplicate entries and we do not
  // accumulate If we accumulate on GPU, we use atomicGPUAdd, which is
  // non-deterministic
  if (!accumulate ||
      (accumulate && iter.tensor(1).device().type() == DeviceType::XPU)) {
    at::globalContext().alertNotDeterministic("put_");
  }

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
      SYCL_KERNEL_ASSERT(dstIndex < dstAddDimSize_);

      // We stride over the output ignoring the indexed dimension
      // (innerSize), whose offset calculation is handled differently
      for (IndexType linearIndex = item.get_group(0) * item.get_local_range(0) +
               item.get_local_id(0);
           linearIndex < innerSize_;
           linearIndex += item.get_group_range(0) * item.get_local_range(0)) {
        IndexType dstOffset =
            IndexToOffset<T, IndexType, DstDim>::get(linearIndex, dst_);
        dstOffset += dstIndex * dst_.strides[dstAddDim_];

        IndexType srcOffset =
            IndexToOffset<const T, IndexType, SrcDim>::get(linearIndex, src_);
        srcOffset += srcIndex * src_.strides[srcAddDim_];

        T val = src_.data[srcOffset] * alpha_;
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

#define SMEM_SIZE 4096

template <
    typename T,
    typename IndicesType,
    typename IndexType,
    int DstDim,
    int SrcDim,
    int IdxDim,
    bool IndexIsMajor,
    typename func_t>
struct IndexFuncLargeIndexFunctor : public __SYCL_KER_CONFIG_CONVENTION__ {
  [[intel::reqd_sub_group_size(SIMD)]] void operator()(
      sycl::nd_item<1> item) const {
    auto local_range = item.get_local_range(0);
    T identity = (T)0;

    for (int i = item.get_local_id(0); i < SMEM_SIZE; i += local_range) {
      smem_offsets[i] = (IndexType)-1;
      smem_values[i] = identity;
    }

    item.barrier(sycl_local_fence);

    for (IndexType linearIndex =
             item.get_group(0) * local_range + item.get_local_id(0);
         linearIndex < totalSize_;
         linearIndex += item.get_group_range(0) * local_range) {
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
      SYCL_KERNEL_ASSERT(dstIndex < dstAddDimSize_);

      IndexType dstOffset =
          IndexToOffset<T, IndexType, DstDim>::get(elementInSlice, dst_);
      dstOffset += dstIndex * dst_.strides[dstAddDim_];

      IndexType srcOffset =
          IndexToOffset<const T, IndexType, SrcDim>::get(elementInSlice, src_);
      srcOffset += srcIndex * src_.strides[srcAddDim_];

      T val = src_.data[srcOffset] * alpha_;
      const int smem_idx = (dstOffset / sizeof(T)) & (SMEM_SIZE - 1);
      IndexType current_offset = smem_offsets[smem_idx];

      if (current_offset == dstOffset) {
        atomicAddLocal(
            static_cast<sycl_local_ptr<T>>(&smem_values[smem_idx]), val);
      } else if (current_offset == (IndexType)-1) {
        IndexType expected = (IndexType)-1;
        if (atomicCAS(&smem_offsets[smem_idx], expected, dstOffset) ==
            expected) {
          atomicAddLocal(
              static_cast<sycl_local_ptr<T>>(&smem_values[smem_idx]), val);
        } else {
          op_(dst_.data, dstOffset, dstNumel_, &val);
        }
      } else {
        op_(dst_.data, dstOffset, dstNumel_, &val);
      }
    }

    item.barrier(sycl_local_fence);

    if (item.get_local_id(0) < SMEM_SIZE) {
      IndexType final_dstOffset = smem_offsets[item.get_local_id(0)];

      if (final_dstOffset != -1) {
        T final_val = smem_values[item.get_local_id(0)];

        op_(dst_.data, final_dstOffset, dstNumel_, &final_val);
      }
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    smem_offsets = sycl_local_acc_t<IndexType>(SMEM_SIZE, cgh);
    smem_values = sycl_local_acc_t<T>(SMEM_SIZE, cgh);
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
  sycl_local_acc_t<IndexType> smem_offsets;
  sycl_local_acc_t<T> smem_values;
};

struct IndexReduceAddFunctor {
  template <typename scalar_t>
  void operator()(
      scalar_t* self_data_start,
      int64_t index,
      int64_t numel,
      const scalar_t* src_data) const {
    atomicAdd((sycl_global_ptr<scalar_t>)(self_data_start + index), *src_data);
  }
};
static IndexReduceAddFunctor reduce_add;

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

template <typename func_t>
void index_reduce_add_xpu_template(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& source,
    const Scalar& alpha,
    const Tensor& result,
    const func_t& func) {
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
    torch::List<std::optional<Tensor>> indices;
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
  const uint64_t sliceSize = getSliceSize(self_, dim, index, source_);
  const uint64_t sourceTotalSize = source.numel();
  const uint64_t selfAddDimSize = self_.size(dim);
  const uint64_t numIndex = index.numel();
  const uint64_t selfNumel = self_.numel();

  if (sliceSize == 0) {
    return;
  }

  const bool indContig = index.is_contiguous();
  int ssc = syclMaxDSSNum();

#define SMALL_INDEX(                                                        \
    TENSOR_TYPE, INDICES_TYPE, TYPE, SELF_DIM, SOURCE_DIM, IDX_DIM, FUNC_T) \
  IndexFuncSmallIndexFunctor<                                               \
      TENSOR_TYPE,                                                          \
      INDICES_TYPE,                                                         \
      TYPE,                                                                 \
      SELF_DIM,                                                             \
      SOURCE_DIM,                                                           \
      IDX_DIM,                                                              \
      FUNC_T>(                                                              \
      selfInfo,                                                             \
      sourceInfo,                                                           \
      indexInfo,                                                            \
      selfAddDim,                                                           \
      sourceAddDim,                                                         \
      sliceSize,                                                            \
      selfAddDimSize,                                                       \
      selfNumel,                                                            \
      reduce_add,                                                           \
      alpha_value);

#define LARGE_INDEX(                         \
    TENSOR_TYPE,                             \
    INDICES_TYPE,                            \
    TYPE,                                    \
    SELF_DIM,                                \
    SOURCE_DIM,                              \
    IDX_DIM,                                 \
    IDX_IS_MAJOR,                            \
    FUNC_T)                                  \
  IndexFuncLargeIndexFunctor<                \
      TENSOR_TYPE,                           \
      INDICES_TYPE,                          \
      TYPE,                                  \
      SELF_DIM,                              \
      SOURCE_DIM,                            \
      IDX_DIM,                               \
      IDX_IS_MAJOR,                          \
      FUNC_T>(                               \
      selfInfo,                              \
      sourceInfo,                            \
      indexInfo,                             \
      selfAddDim,                            \
      sourceAddDim,                          \
      sourceTotalSize,                       \
      (IDX_IS_MAJOR) ? sliceSize : numIndex, \
      selfAddDimSize,                        \
      selfNumel,                             \
      reduce_add,                            \
      alpha_value);

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
              size_t num_wg = std::min(
                  ceil_div(sliceSize, (uint64_t)128), (uint64_t)(ssc * 8));
              size_t wg_size = std::min(sliceSize, (uint64_t)128);
              if (selfInfo.dims == 1 && sourceInfo.dims == 1 && indContig) {
                auto caller = SMALL_INDEX(
                    scalar_t, index_t, unsigned int, 1, 1, -2, func_t);
                sycl_kernel_submit(
                    num_wg * wg_size, wg_size, getCurrentSYCLQueue(), caller);
              } else if (
                  selfInfo.dims == 2 && sourceInfo.dims == 2 && indContig) {
                auto caller = SMALL_INDEX(
                    scalar_t, index_t, unsigned int, 2, 2, -2, func_t);
                sycl_kernel_submit(
                    num_wg * wg_size, wg_size, getCurrentSYCLQueue(), caller);
              } else if (
                  selfInfo.dims == 3 && sourceInfo.dims == 3 && indContig) {
                auto caller = SMALL_INDEX(
                    scalar_t, index_t, unsigned int, 3, 3, -2, func_t);
                sycl_kernel_submit(
                    num_wg * wg_size, wg_size, getCurrentSYCLQueue(), caller);
              } else {
                auto caller = SMALL_INDEX(
                    scalar_t, index_t, unsigned int, -1, -1, -1, func_t);
                sycl_kernel_submit(
                    num_wg * wg_size, wg_size, getCurrentSYCLQueue(), caller);
              }
            } else {
              const bool indexIsMajor =
                  indexShouldBeMajor(selfInfo, selfAddDim);
              uint64_t defaultMaxGroupThreads = syclDeviceMaxWorkGroupSize();
              size_t num_wg = std::min(
                  ceil_div(sourceTotalSize, (uint64_t)128),
                  (uint64_t)(ssc * 8));
              size_t wg_size = (sourceTotalSize < defaultMaxGroupThreads)
                  ? sourceTotalSize
                  : defaultMaxGroupThreads;
              if (selfInfo.dims == 1 && sourceInfo.dims == 1 && indContig) {
                auto caller = LARGE_INDEX(
                    scalar_t, index_t, unsigned int, 1, 1, -2, true, func_t);
                sycl_kernel_submit(
                    num_wg * wg_size, wg_size, getCurrentSYCLQueue(), caller);
              } else if (
                  selfInfo.dims == 2 && sourceInfo.dims == 2 && indContig) {
                if (indexIsMajor) {
                  auto caller = LARGE_INDEX(
                      scalar_t, index_t, unsigned int, 2, 2, -2, true, func_t);
                  sycl_kernel_submit(
                      num_wg * wg_size, wg_size, getCurrentSYCLQueue(), caller);
                } else {
                  auto caller = LARGE_INDEX(
                      scalar_t, index_t, unsigned int, 2, 2, -2, false, func_t);
                  sycl_kernel_submit(
                      num_wg * wg_size, wg_size, getCurrentSYCLQueue(), caller);
                }
              } else if (
                  selfInfo.dims == 3 && sourceInfo.dims == 3 && indContig) {
                if (indexIsMajor) {
                  auto caller = LARGE_INDEX(
                      scalar_t, index_t, unsigned int, 3, 3, -2, true, func_t);
                  sycl_kernel_submit(
                      num_wg * wg_size, wg_size, getCurrentSYCLQueue(), caller);
                } else {
                  auto caller = LARGE_INDEX(
                      scalar_t, index_t, unsigned int, 3, 3, -2, false, func_t);
                  sycl_kernel_submit(
                      num_wg * wg_size, wg_size, getCurrentSYCLQueue(), caller);
                }
              } else {
                auto caller = LARGE_INDEX(
                    scalar_t, index_t, unsigned int, -1, -1, -1, true, func_t);
                sycl_kernel_submit(
                    num_wg * wg_size, wg_size, getCurrentSYCLQueue(), caller);
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

            auto caller = LARGE_INDEX(
                scalar_t, index_t, uint64_t, -1, -1, -1, true, func_t);
            // uint64_t defaultMaxGroupThreads = syclMaxWorkGroupSize(caller);
            uint64_t defaultMaxGroupThreads = syclDeviceMaxWorkGroupSize();
            size_t num_wg = std::min(
                ceil_div(sourceTotalSize, (uint64_t)128), (uint64_t)(ssc * 8));
            size_t wg_size = (sourceTotalSize < defaultMaxGroupThreads)
                ? sourceTotalSize
                : defaultMaxGroupThreads;
            sycl_kernel_submit(
                num_wg * wg_size, wg_size, getCurrentSYCLQueue(), caller);
          });
        });
  }

#undef SMALL_INDEX
#undef LARGE_INDEX
}

void index_add_kernel(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& source,
    const Scalar& alpha,
    const Tensor& result) {
  index_reduce_add_xpu_template(
      self, dim, index, source, alpha, result, reduce_add);
}

template <typename func_t>
void index_reduce_func_xpu_template(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& source,
    bool include_self,
    const ReductionType& reduce,
    const func_t& reduce_func,
    const Tensor& result) {
  globalContext().alertNotDeterministic("index_reduce_xpu");

  if (!result.is_same(self))
    result.copy_(self);

  // Scalars are treated as 1-d tensor
  Tensor self_ = (result.dim() == 0) ? result.view(1) : result;
  Tensor source_ = (source.dim() == 0) ? source.view(1) : source;

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

  if (!include_self) {
    AT_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        self.scalar_type(),
        "index_reduce_func_xpu_exclude_input_init",
        [&] {
          scalar_t init_val;
          switch (reduce) {
            case ReductionType::PROD:
              init_val = (scalar_t)1;
              break;
            case ReductionType::MAX:
              init_val = std::numeric_limits<scalar_t>::has_infinity
                  ? -std::numeric_limits<scalar_t>::infinity()
                  : std::numeric_limits<scalar_t>::lowest();
              break;
            case ReductionType::MIN:
              init_val = std::numeric_limits<scalar_t>::has_infinity
                  ? std::numeric_limits<scalar_t>::infinity()
                  : std::numeric_limits<scalar_t>::max();
              break;
            default:
              init_val = (scalar_t)0;
              break;
          }
          // index_fill_ requires index to be a LongTensor
          self_.index_fill_(dim, index.to(at::ScalarType::Long), init_val);
        });
  }

  uint64_t sliceSize = getSliceSize(self_, dim, index, source_);
  uint64_t sourceTotalSize = source.numel();
  uint64_t selfReduceDimSize = self_.size(dim);
  uint64_t numIndex = index.numel();
  uint64_t selfNumel = self_.numel();
  if (sliceSize == 0) {
    return;
  }
  bool indContig = index.is_contiguous();

#define SMALL_INDEX(                                                        \
    TENSOR_TYPE, INDICES_TYPE, TYPE, SELF_DIM, SOURCE_DIM, IDX_DIM, FUNC_T) \
  IndexFuncSmallIndexFunctor<                                               \
      TENSOR_TYPE,                                                          \
      INDICES_TYPE,                                                         \
      TYPE,                                                                 \
      SELF_DIM,                                                             \
      SOURCE_DIM,                                                           \
      IDX_DIM,                                                              \
      FUNC_T>(                                                              \
      selfInfo,                                                             \
      sourceInfo,                                                           \
      indexInfo,                                                            \
      selfReduceDim,                                                        \
      sourceReduceDim,                                                      \
      sliceSize,                                                            \
      selfReduceDimSize,                                                    \
      selfNumel,                                                            \
      reduce_func,                                                          \
      alpha_value);

#define LARGE_INDEX(                         \
    TENSOR_TYPE,                             \
    INDICES_TYPE,                            \
    TYPE,                                    \
    SELF_DIM,                                \
    SOURCE_DIM,                              \
    IDX_DIM,                                 \
    IDX_IS_MAJOR,                            \
    FUNC_T)                                  \
  IndexFuncLargeIndexFunctor<                \
      TENSOR_TYPE,                           \
      INDICES_TYPE,                          \
      TYPE,                                  \
      SELF_DIM,                              \
      SOURCE_DIM,                            \
      IDX_DIM,                               \
      IDX_IS_MAJOR,                          \
      FUNC_T>(                               \
      selfInfo,                              \
      sourceInfo,                            \
      indexInfo,                             \
      selfReduceDim,                         \
      sourceReduceDim,                       \
      sourceTotalSize,                       \
      (IDX_IS_MAJOR) ? sliceSize : numIndex, \
      selfReduceDimSize,                     \
      selfNumel,                             \
      reduce_func,                           \
      alpha_value);

  int ssc = syclMaxDSSNum();

  if (canUse32BitIndexMath(result) && canUse32BitIndexMath(source) &&
      canUse32BitIndexMath(index)) {
    AT_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        result.scalar_type(),
        "index_reduce",
        [&] {
          TensorInfo<scalar_t, unsigned int> selfInfo =
              getTensorInfo<scalar_t, unsigned int>(self_);
          int selfReduceDim = selfInfo.collapseDims(dim);
          selfInfo.reduceDim(selfReduceDim);
          auto alpha_value = (scalar_t)1;
          AT_DISPATCH_INDEX_TYPES(
              index.scalar_type(), "index_reduce_xpu", [&]() {
                auto sourceInfo =
                    getTensorInfo<const scalar_t, unsigned int>(source_);
                int sourceReduceDim = sourceInfo.collapseDims(dim);
                sourceInfo.reduceDim(sourceReduceDim);

                auto indexInfo =
                    getTensorInfo<const index_t, unsigned int>(index);
                indexInfo.collapseDims();

                // A reasonable choice for when to have each thread iterate
                // over index to choose
                if (numIndex <= 16) {
                  if (selfInfo.dims == 1 && sourceInfo.dims == 1 && indContig) {
                    auto caller = SMALL_INDEX(
                        scalar_t, index_t, unsigned int, 1, 1, -2, func_t);
                    size_t num_wg = std::min(
                        ceil_div(sliceSize, (uint64_t)128),
                        (uint64_t)(ssc * 8));
                    size_t wg_size = std::min(sliceSize, (uint64_t)128);
                    sycl_kernel_submit(
                        num_wg * wg_size,
                        wg_size,
                        getCurrentSYCLQueue(),
                        caller);
                  } else if (
                      selfInfo.dims == 2 && sourceInfo.dims == 2 && indContig) {
                    auto caller = SMALL_INDEX(
                        scalar_t, index_t, unsigned int, 2, 2, -2, func_t);
                    size_t num_wg = std::min(
                        ceil_div(sliceSize, (uint64_t)128),
                        (uint64_t)(ssc * 8));
                    size_t wg_size = std::min(sliceSize, (uint64_t)128);
                    sycl_kernel_submit(
                        num_wg * wg_size,
                        wg_size,
                        getCurrentSYCLQueue(),
                        caller);
                  } else if (
                      selfInfo.dims == 3 && sourceInfo.dims == 3 && indContig) {
                    auto caller = SMALL_INDEX(
                        scalar_t, index_t, unsigned int, 3, 3, -2, func_t);
                    size_t num_wg = std::min(
                        ceil_div(sliceSize, (uint64_t)128),
                        (uint64_t)(ssc * 8));
                    size_t wg_size = std::min(sliceSize, (uint64_t)128);
                    sycl_kernel_submit(
                        num_wg * wg_size,
                        wg_size,
                        getCurrentSYCLQueue(),
                        caller);
                  } else {
                    auto caller = SMALL_INDEX(
                        scalar_t, index_t, unsigned int, -1, -1, -1, func_t);
                    size_t num_wg = std::min(
                        ceil_div(sliceSize, (uint64_t)128),
                        (uint64_t)(ssc * 8));
                    size_t wg_size = std::min(sliceSize, (uint64_t)128);
                    sycl_kernel_submit(
                        num_wg * wg_size,
                        wg_size,
                        getCurrentSYCLQueue(),
                        caller);
                  }
                } else {
                  bool indexIsMajor =
                      indexShouldBeMajor(selfInfo, selfReduceDim);

                  if (selfInfo.dims == 1 && sourceInfo.dims == 1 && indContig) {
                    auto caller = LARGE_INDEX(
                        scalar_t,
                        index_t,
                        unsigned int,
                        1,
                        1,
                        -2,
                        true,
                        func_t);
                    int defaultMaxGroupThreads = syclMaxWorkGroupSize(caller);
                    size_t num_wg = std::min(
                        ceil_div(sourceTotalSize, (uint64_t)128),
                        (uint64_t)(ssc * 8));
                    size_t wg_size = (sourceTotalSize < defaultMaxGroupThreads)
                        ? sourceTotalSize
                        : defaultMaxGroupThreads;
                    sycl_kernel_submit(
                        num_wg * wg_size,
                        wg_size,
                        getCurrentSYCLQueue(),
                        caller);
                  } else if (
                      selfInfo.dims == 2 && sourceInfo.dims == 2 && indContig) {
                    if (indexIsMajor) {
                      auto caller = LARGE_INDEX(
                          scalar_t,
                          index_t,
                          unsigned int,
                          2,
                          2,
                          -2,
                          true,
                          func_t);
                      int defaultMaxGroupThreads = syclMaxWorkGroupSize(caller);
                      size_t num_wg = std::min(
                          ceil_div(sourceTotalSize, (uint64_t)128),
                          (uint64_t)(ssc * 8));
                      size_t wg_size =
                          (sourceTotalSize < defaultMaxGroupThreads)
                          ? sourceTotalSize
                          : defaultMaxGroupThreads;
                      sycl_kernel_submit(
                          num_wg * wg_size,
                          wg_size,
                          getCurrentSYCLQueue(),
                          caller);
                    } else {
                      auto caller = LARGE_INDEX(
                          scalar_t,
                          index_t,
                          unsigned int,
                          2,
                          2,
                          -2,
                          false,
                          func_t);
                      int defaultMaxGroupThreads = syclMaxWorkGroupSize(caller);
                      size_t num_wg = std::min(
                          ceil_div(sourceTotalSize, (uint64_t)128),
                          (uint64_t)(ssc * 8));
                      size_t wg_size =
                          (sourceTotalSize < defaultMaxGroupThreads)
                          ? sourceTotalSize
                          : defaultMaxGroupThreads;
                      sycl_kernel_submit(
                          num_wg * wg_size,
                          wg_size,
                          getCurrentSYCLQueue(),
                          caller);
                    }
                  } else if (
                      selfInfo.dims == 3 && sourceInfo.dims == 3 && indContig) {
                    if (indexIsMajor) {
                      auto caller = LARGE_INDEX(
                          scalar_t,
                          index_t,
                          unsigned int,
                          3,
                          3,
                          -2,
                          true,
                          func_t);
                      int defaultMaxGroupThreads = syclMaxWorkGroupSize(caller);
                      size_t num_wg = std::min(
                          ceil_div(sourceTotalSize, (uint64_t)128),
                          (uint64_t)(ssc * 8));
                      size_t wg_size =
                          (sourceTotalSize < defaultMaxGroupThreads)
                          ? sourceTotalSize
                          : defaultMaxGroupThreads;
                      sycl_kernel_submit(
                          num_wg * wg_size,
                          wg_size,
                          getCurrentSYCLQueue(),
                          caller);
                    } else {
                      auto caller = LARGE_INDEX(
                          scalar_t,
                          index_t,
                          unsigned int,
                          3,
                          3,
                          -2,
                          false,
                          func_t);
                      int defaultMaxGroupThreads = syclMaxWorkGroupSize(caller);
                      size_t num_wg = std::min(
                          ceil_div(sourceTotalSize, (uint64_t)128),
                          (uint64_t)(ssc * 8));
                      size_t wg_size =
                          (sourceTotalSize < defaultMaxGroupThreads)
                          ? sourceTotalSize
                          : defaultMaxGroupThreads;
                      sycl_kernel_submit(
                          num_wg * wg_size,
                          wg_size,
                          getCurrentSYCLQueue(),
                          caller);
                    }
                  } else {
                    auto caller = LARGE_INDEX(
                        scalar_t,
                        index_t,
                        unsigned int,
                        -1,
                        -1,
                        -1,
                        true,
                        func_t);
                    int defaultMaxGroupThreads = syclMaxWorkGroupSize(caller);
                    size_t num_wg = std::min(
                        ceil_div(sourceTotalSize, (uint64_t)128),
                        (uint64_t)(ssc * 8));
                    size_t wg_size = (sourceTotalSize < defaultMaxGroupThreads)
                        ? sourceTotalSize
                        : defaultMaxGroupThreads;
                    sycl_kernel_submit(
                        num_wg * wg_size,
                        wg_size,
                        getCurrentSYCLQueue(),
                        caller);
                  }
                }
              });
        });
  } else {
    AT_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        self.scalar_type(),
        "index_reduce",
        [&] {
          TensorInfo<scalar_t, uint64_t> selfInfo =
              getTensorInfo<scalar_t, uint64_t>(self_);
          int selfReduceDim = selfInfo.collapseDims(dim);
          selfInfo.reduceDim(selfReduceDim);
          auto alpha_value = (scalar_t)1;

          TensorInfo<const scalar_t, uint64_t> sourceInfo =
              getTensorInfo<const scalar_t, uint64_t>(source_);
          int sourceReduceDim = sourceInfo.collapseDims(dim);
          sourceInfo.reduceDim(sourceReduceDim);

          AT_DISPATCH_INDEX_TYPES(
              index.scalar_type(), "index_reduce_xpu", [&]() {
                TensorInfo<const index_t, uint64_t> indexInfo =
                    getTensorInfo<const index_t, uint64_t>(index);
                indexInfo.collapseDims();
                auto caller = LARGE_INDEX(
                    scalar_t, index_t, uint64_t, -1, -1, -1, true, func_t);
                int defaultMaxGroupThreads = syclMaxWorkGroupSize(caller);
                size_t num_wg = std::min(
                    ceil_div(sourceTotalSize, (uint64_t)128),
                    (uint64_t)(ssc * 8));
                size_t wg_size = (sourceTotalSize < defaultMaxGroupThreads)
                    ? sourceTotalSize
                    : defaultMaxGroupThreads;
                sycl_kernel_submit(
                    num_wg * wg_size, wg_size, getCurrentSYCLQueue(), caller);
              });
        });
  }

#undef SMALL_INDEX
#undef LARGE_INDEX
}

struct IndexReduceMultiplyFunctor {
  template <typename scalar_t>
  void operator()(
      scalar_t* self_data_start,
      int64_t index,
      int64_t numel,
      const scalar_t* src_data) const {
    (void)numel; // suppress unused warning
    atomicMul((sycl_global_ptr<scalar_t>)(self_data_start + index), *src_data);
  }
};
static IndexReduceMultiplyFunctor index_reduce_multiply;

struct IndexReduceMeanFunctor {
  template <typename scalar_t>
  void operator()(
      scalar_t* self_data_start,
      int64_t index,
      int64_t numel,
      const scalar_t* src_data) const {
    atomicAdd((sycl_global_ptr<scalar_t>)(self_data_start + index), *src_data);
  }
};
static IndexReduceMeanFunctor index_reduce_mean;

struct IndexReduceMaxFunctor {
  template <typename scalar_t>
  void operator()(
      scalar_t* self_data_start,
      int64_t index,
      int64_t numel,
      const scalar_t* src_data) const {
    (void)numel; // suppress unused warning
    atomicMax((sycl_global_ptr<scalar_t>)(self_data_start + index), *src_data);
  }
};
static IndexReduceMaxFunctor index_reduce_max;

struct IndexReduceMinFunctor {
  template <typename scalar_t>
  void operator()(
      scalar_t* self_data_start,
      int64_t index,
      int64_t numel,
      const scalar_t* src_data) const {
    (void)numel; // suppress unused warning
    atomicMin((sycl_global_ptr<scalar_t>)(self_data_start + index), *src_data);
  }
};
static IndexReduceMinFunctor index_reduce_min;

void index_reduce_prod_kernel(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& source,
    bool include_self,
    const ReductionType& reduce,
    const Tensor& result) {
  index_reduce_func_xpu_template(
      self,
      dim,
      index,
      source,
      include_self,
      reduce,
      index_reduce_multiply,
      result);
}

void index_reduce_mean_kernel(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& source,
    bool include_self,
    const ReductionType& reduce,
    const Tensor& result) {
  index_reduce_func_xpu_template(
      self,
      dim,
      index,
      source,
      include_self,
      reduce,
      index_reduce_mean,
      result);
}

void index_reduce_amax_kernel(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& source,
    bool include_self,
    const ReductionType& reduce,
    const Tensor& result) {
  index_reduce_func_xpu_template(
      self, dim, index, source, include_self, reduce, index_reduce_max, result);
}

void index_reduce_amin_kernel(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& source,
    bool include_self,
    const ReductionType& reduce,
    const Tensor& result) {
  index_reduce_func_xpu_template(
      self, dim, index, source, include_self, reduce, index_reduce_min, result);
}

// ForwardIt: only legacy random access iterator is supported.
template <class ForwardIt, class T, bool is_lower = true>
static inline ForwardIt find_bound(
    ForwardIt first,
    ForwardIt last,
    const T& value) {
  ForwardIt it;
  typename std::iterator_traits<ForwardIt>::difference_type count, step;
  // NOTE: std::distance(first, last) compiles but produces wrong results
  // here, so only legacy random access iterators are safe in this code.
  count = last - first;

  while (count > 0) {
    it = first;
    step = count / 2;
    // avoiding std::advance(it, step),
    // although it does work unlike std::distance
    it += step;
    if (is_lower ? *it < value : value >= *it) {
      first = ++it;
      count -= step + 1;
    } else {
      count = step;
    }
  }
  return first;
}

template <
    typename T,
    typename IndicesType,
    typename IndexType,
    int DstDim,
    int SrcDim,
    int IdxDim>
struct IndexSelectSmallIndexFunctor {
  void operator()(sycl::nd_item<1> item) const {
    // In order to avoid reloading the index that we are copying, load
    // it once to handle all of the points that are being selected, so
    // it can be reused as much as possible. This kernel is chosen when
    // this is a good choice (small number of chosen indices), since
    // re-accessing indices in addition to src elements can be slow.
    for (IndexType dstIndex = 0; dstIndex < indices.sizes[0]; ++dstIndex) {
      IndexType srcIndex =
          indices.data[IndexToOffset<const IndicesType, IndexType, IdxDim>::get(
              dstIndex, indices)];
      SYCL_KERNEL_ASSERT(srcIndex < srcSelectDimSize);

      // We stride over the output ignoring the indexed dimension
      // (innerSize), whose offset calculation is handled differently
      for (IndexType linearIndex = item.get_group(0) * item.get_local_range(0) +
               item.get_local_id(0);
           linearIndex < innerSize;
           linearIndex += item.get_group_range(0) * item.get_local_range(0)) {
        IndexType dstOffset =
            IndexToOffset<T, IndexType, DstDim>::get(linearIndex, dst);
        dstOffset += dstIndex * dst.strides[dstSelectDim];

        IndexType srcOffset =
            IndexToOffset<const T, IndexType, SrcDim>::get(linearIndex, src);
        srcOffset += srcIndex * src.strides[srcSelectDim];

        dst.data[dstOffset] = src.data[srcOffset];
      }
    }
  }

  IndexSelectSmallIndexFunctor(
      TensorInfo<T, IndexType> dst,
      TensorInfo<const T, IndexType> src,
      TensorInfo<const IndicesType, IndexType> indices,
      int dstSelectDim,
      int srcSelectDim,
      IndexType innerSize,
      int64_t srcSelectDimSize)
      : dst(dst),
        src(src),
        indices(indices),
        dstSelectDim(dstSelectDim),
        srcSelectDim(srcSelectDim),
        innerSize(innerSize),
        srcSelectDimSize(srcSelectDimSize) {}

 private:
  TensorInfo<T, IndexType> dst;
  TensorInfo<const T, IndexType> src;
  TensorInfo<const IndicesType, IndexType> indices;
  int dstSelectDim;
  int srcSelectDim;
  IndexType innerSize;
  int64_t srcSelectDimSize;
};

// When using a 0-dim scalar tensor, we need the legacy (THC) semantics of
// TensorInfo: Pretend that the scalar tensor is in fact a one-element vector.
template <typename T, typename IndexType>
TensorInfo<T, IndexType> tensorInfoLegacyIfScalar(TensorInfo<T, IndexType> ti) {
  if (ti.dims == 0) {
    ti.dims = 1;
    ti.sizes[0] = 1;
    ti.strides[0] = 1;
  }
  return ti;
}

template <typename scalar_t>
void index_select_out_impl(
    Tensor& out,
    const Tensor& self,
    int64_t dim,
    const Tensor& index) {
  uint64_t numIndices = index.numel();
  auto selfDims = self.dim() == 0 ? 1 : self.dim();

  TORCH_CHECK(
      index.dim() <= 1, "Index is supposed to be an empty tensor or a vector");
  TORCH_CHECK(
      !(self.dim() == 0 && numIndices != 1),
      "index_select(): Index to scalar can have only 1 value, got ",
      numIndices,
      " value(s)");
  TORCH_CHECK(dim < selfDims, "Indexing dim is out of bounds");

  std::vector<int64_t> newSize = self.sizes().vec();
  if (self.dim() > 0) {
    newSize[dim] = numIndices;
  }

  at::native::resize_output(out, newSize);

  uint64_t outTotalSize = out.numel();
  if (outTotalSize == 0) {
    return;
  }

  bool indContig = index.is_contiguous();

  // The `self` is partitioned into two parts:
  // -the size of each slice we are indexing, which is the
  // total size of the tensor ignoring dimension `dim`;
  // -the number of indices we are choosing, which is the total size
  // of the tensor `indices`.
  uint64_t selfSelectDimSize = self.dim() == 0 ? 1 : self.size(dim);
  uint64_t sliceSize = outTotalSize / numIndices;

  int ssc = syclMaxDSSNum();

#define SMALL_INDEX(                                            \
    TENSOR_TYPE, INDICES_TYPE, TYPE, DST_DIM, SRC_DIM, IDX_DIM) \
  IndexSelectSmallIndexFunctor<                                 \
      TENSOR_TYPE,                                              \
      INDICES_TYPE,                                             \
      TYPE,                                                     \
      DST_DIM,                                                  \
      SRC_DIM,                                                  \
      IDX_DIM>(                                                 \
      outInfo,                                                  \
      selfInfo,                                                 \
      indicesInfo,                                              \
      outSelectDim,                                             \
      selfSelectDim,                                            \
      static_cast<TYPE>(sliceSize),                             \
      selfSelectDimSize);

  // SmallIndexKernel is more performant when the number of indices is
  // small, and pre-loading the index reduces memory accesses. When the
  // number of indices is large, we avoid that and increase parallellism by
  // calling gather_out which is a generalization of index_select
  if (canUse32BitIndexMath(out) && canUse32BitIndexMath(self) &&
      canUse32BitIndexMath(index) && numIndices <= 16) {
    auto outInfo =
        tensorInfoLegacyIfScalar(getTensorInfo<scalar_t, unsigned int>(out));
    int outSelectDim = outInfo.collapseDims(dim);
    outInfo.reduceDim(outSelectDim);

    auto selfInfo = tensorInfoLegacyIfScalar(
        getTensorInfo<const scalar_t, unsigned int>(self));
    int selfSelectDim = selfInfo.collapseDims(dim);
    selfInfo.reduceDim(selfSelectDim);

    AT_DISPATCH_INDEX_TYPES(
        index.scalar_type(), "index_select_out_impl", [&]() {
          auto indicesInfo = tensorInfoLegacyIfScalar(
              getTensorInfo<const index_t, unsigned int>(index));
          indicesInfo.collapseDims();

          uint64_t defaultMaxGroupThreads = syclDeviceMaxWorkGroupSize() / 2;
          size_t num_wg = std::min(
              ceil_div(sliceSize, defaultMaxGroupThreads), (uint64_t)(ssc * 8));
          size_t wg_size = std::min(sliceSize, defaultMaxGroupThreads);

          // A reasonable choice for when to have each thread iterate over
          // indices to choose
          if (outInfo.dims == 1 && selfInfo.dims == 1 && indContig) {
            auto caller =
                SMALL_INDEX(scalar_t, index_t, unsigned int, 1, 1, -2);
            sycl_kernel_submit(
                num_wg * wg_size, wg_size, getCurrentSYCLQueue(), caller);
          } else if (outInfo.dims == 2 && selfInfo.dims == 2 && indContig) {
            auto caller =
                SMALL_INDEX(scalar_t, index_t, unsigned int, 2, 2, -2);
            sycl_kernel_submit(
                num_wg * wg_size, wg_size, getCurrentSYCLQueue(), caller);
          } else if (outInfo.dims == 3 && selfInfo.dims == 3 && indContig) {
            auto caller =
                SMALL_INDEX(scalar_t, index_t, unsigned int, 3, 3, -2);
            sycl_kernel_submit(
                num_wg * wg_size, wg_size, getCurrentSYCLQueue(), caller);
          } else {
            auto caller =
                SMALL_INDEX(scalar_t, index_t, unsigned int, -1, -1, -1);
            sycl_kernel_submit(
                num_wg * wg_size, wg_size, getCurrentSYCLQueue(), caller);
          }
        });
  } else {
    std::vector<int64_t> tmpSize(newSize.size(), 1);
    if (self.dim() > 0) {
      tmpSize[dim] = numIndices;
    }
    at::gather_out(out, self, dim, index.view(tmpSize).expand(newSize));
    return;
  }
#undef SMALL_INDEX
}

Tensor& index_select_kernel(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    Tensor& out) {
  static constexpr std::string_view DIM_WARNING =
      "Tensor too large or too many (> 25) dimensions";
  at::assert_no_internal_overlap(out);
  at::assert_no_overlap(out, self);
  at::assert_no_overlap(out, index);

  dim = at::maybe_wrap_dim(dim, self);
  TORCH_CHECK(self.dim() <= XPU_MAX_TENSORINFO_DIMS, DIM_WARNING);
  TORCH_CHECK(index.dim() <= XPU_MAX_TENSORINFO_DIMS, DIM_WARNING);

  AT_DISPATCH_V2(
      out.scalar_type(),
      "index_select_xpu",
      AT_WRAP([&] { index_select_out_impl<scalar_t>(out, self, dim, index); }),
      AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX),
      AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES),
      AT_EXPAND(AT_FLOAT8_TYPES),
      kComplexHalf,
      kHalf,
      kBool,
      kBFloat16);

  return out;
}

template <typename index_t>
struct IndexSelectSparse1Functor {
  index_t operator()(index_t idx) const {
    SYCL_KERNEL_ASSERT(
        idx >= -size_ && idx < size_ && "index_select(): index out of bounds");
    return idx < 0 ? idx + size_ : idx;
  }
  IndexSelectSparse1Functor(index_t size) : size_(size) {}

 private:
  index_t size_;
};

template <typename index_t>
struct IndexSelectSparse2Functor {
  index_t operator()(index_t idx_val, index_t idx_idx) const {
    auto* lb = find_bound<const index_t*, index_t, true>(
        ptr_sorted_dim_indices_, ptr_sorted_dim_indices_ + nnz_, idx_val);
    auto* ub = find_bound<const index_t*, index_t, false>(
        ptr_sorted_dim_indices_, ptr_sorted_dim_indices_ + nnz_, idx_val);
    const auto idx_count = ub - lb;
    ptr_intrsc_counts_nneg_index_[idx_idx] = idx_count;

    return lb - ptr_sorted_dim_indices_;
  }

  IndexSelectSparse2Functor(
      index_t* ptr_intrsc_counts_nneg_index,
      const index_t* ptr_sorted_dim_indices,
      int64_t nnz)
      : ptr_intrsc_counts_nneg_index_(ptr_intrsc_counts_nneg_index),
        ptr_sorted_dim_indices_(ptr_sorted_dim_indices),
        nnz_(nnz) {}

 private:
  index_t* ptr_intrsc_counts_nneg_index_;
  const index_t* ptr_sorted_dim_indices_;
  int64_t nnz_;
};

template <typename index_t>
struct IndexSelectSparse3Functor {
  index_t operator()(
      index_t idx_idx,
      index_t count,
      index_t offset,
      index_t first_match) const {
    index_t* RESTRICT ptr_res_dim_indices_out = ptr_res_dim_indices_ + offset;
    const index_t* RESTRICT ptr_argsort_dim_indices_in =
        ptr_argsort_dim_indices_ + first_match;
    index_t* RESTRICT ptr_selected_dim_indices_out =
        ptr_selected_dim_indices_ + offset;
    for (index_t i = 0; i < count; ++i) {
      *ptr_res_dim_indices_out++ = idx_idx;
      *ptr_selected_dim_indices_out++ = *ptr_argsort_dim_indices_in++;
    }
    // A dummy return scalar for a dummy output
    return static_cast<index_t>(1);
  }
  IndexSelectSparse3Functor(
      index_t* ptr_res_dim_indices,
      index_t* ptr_selected_dim_indices,
      const index_t* ptr_argsort_dim_indices)
      : ptr_res_dim_indices_(ptr_res_dim_indices),
        ptr_selected_dim_indices_(ptr_selected_dim_indices),
        ptr_argsort_dim_indices_(ptr_argsort_dim_indices) {}

 private:
  index_t* ptr_res_dim_indices_;
  index_t* ptr_selected_dim_indices_;
  const index_t* ptr_argsort_dim_indices_;
};

Tensor index_select_sparse_kernel(
    const Tensor& self,
    int64_t dim,
    const Tensor& index) {
  const auto ndim = self.dim();
  TORCH_CHECK_INDEX(
      ndim, "index_select() cannot be applied to a 0-dim tensor.");
  TORCH_CHECK_INDEX(
      index.dim() == 1 && index.dtype() == at::kLong &&
          index.options().layout() == at::kStrided,
      "index_select() argument index must be 1-D strided (non-sparse) long-tensor.");

  dim = maybe_wrap_dim(dim, ndim);
  const auto size = self.size(dim);
  const auto sparse_dim = self.sparse_dim();
  const auto dense_dim = self.dense_dim();
  const auto indices = self._indices();
  const auto values = self._values();
  const auto nnz = values.size(0);
  const auto index_len = index.size(0);
  auto res_sizes = self.sizes().vec();
  res_sizes[dim] = index_len;

  // If indexing into sparse dimensions
  if (dim < sparse_dim) {
    const auto make_output =
        [dim, sparse_dim, dense_dim, res_sizes, &self, &indices, &values](
            const Tensor& selected_dim_indices,
            const Tensor& res_dim_indices) -> Tensor {
      auto res_indices = indices.index_select(1, selected_dim_indices);
      res_indices[dim] = res_dim_indices;
      const auto res_values = values.index_select(0, selected_dim_indices);

      return at::_sparse_coo_tensor_with_dims_and_tensors(
          sparse_dim,
          dense_dim,
          res_sizes,
          res_indices,
          res_values,
          self.options());
    };

    // short-circuit if index is empty
    if (!index_len) {
      return make_output(index, index);
    }

    const auto nneg_index = [&index, size]() -> Tensor {
      auto nneg_index = at::empty_like(index, at::MemoryFormat::Contiguous);

      auto iter = TensorIteratorConfig()
                      .add_output(nneg_index)
                      .add_input(index)
                      .build();

      AT_DISPATCH_INDEX_TYPES(
          index.scalar_type(), "index_select_sparse_xpu", [&]() {
            gpu_kernel(iter, IndexSelectSparse1Functor<index_t>(size));
          });
      return nneg_index;
    }();

    const auto dim_indices = indices[dim].contiguous();
    const auto idx_nneg_index = at::arange(index_len, nneg_index.options());
    const auto idx_dim_indices = at::arange(nnz, dim_indices.options());

    Tensor sorted_dim_indices, argsort_dim_indices;
    std::tie(sorted_dim_indices, argsort_dim_indices) =
        [&]() -> std::tuple<Tensor, Tensor> {
      if (dim == 0 && self.is_coalesced()) {
        return std::make_tuple(dim_indices, idx_dim_indices);
      } else {
        return dim_indices.sort();
      }
    }();

    Tensor intrsc_counts_nneg_index;
    Tensor intrsc_first_match_nneg_index;
    std::tie(intrsc_counts_nneg_index, intrsc_first_match_nneg_index) =
        [&]() -> std::tuple<Tensor, Tensor> {
      auto intrsc_counts_nneg_index = at::zeros_like(nneg_index);
      auto intrsc_first_match_nneg_index = at::zeros_like(nneg_index);

      auto iter = TensorIteratorConfig()
                      .add_output(intrsc_first_match_nneg_index)
                      .add_input(nneg_index)
                      .add_input(idx_nneg_index)
                      .build();

      AT_DISPATCH_INDEX_TYPES(
          nneg_index.scalar_type(), "index_select_sparse_xpu", [&]() {
            index_t* ptr_intrsc_counts_nneg_index =
                intrsc_counts_nneg_index.mutable_data_ptr<index_t>();
            const index_t* ptr_sorted_dim_indices =
                sorted_dim_indices.const_data_ptr<index_t>();
            gpu_kernel(
                iter,
                IndexSelectSparse2Functor<index_t>(
                    ptr_intrsc_counts_nneg_index, ptr_sorted_dim_indices, nnz));
          });

      return std::make_tuple(
          intrsc_counts_nneg_index, intrsc_first_match_nneg_index);
    }();

    // Unavoidable sync since the shape of the result is not known in advance
    auto res_len = intrsc_counts_nneg_index.sum().item<int64_t>();
    // Short-circuit if empty intersection
    if (!res_len) {
      auto empty_idx = at::empty({0}, nneg_index.options());
      return make_output(empty_idx, empty_idx);
    }

    auto [selected_dim_indices, res_dim_indices] =
        [&]() -> std::tuple<Tensor, Tensor> {
      auto res_dim_indices = at::empty({res_len}, nneg_index.options());
      auto selected_dim_indices = at::empty_like(res_dim_indices);
      auto selected_dim_indices_offsets =
          intrsc_counts_nneg_index.cumsum(0).sub_(intrsc_counts_nneg_index);

      // Need to have output as TensorIterator does not allow having void
      // lambdas.
      auto dummy_output = at::empty({1}, dim_indices.options())
                              .expand(IntArrayRef({index_len}));
      auto iter = TensorIteratorConfig()
                      .add_output(dummy_output)
                      // All iterations map to a single element in dummy_output
                      // by design, hence removed output memory overlap check.
                      .set_check_mem_overlap(false)
                      .add_input(idx_nneg_index)
                      .add_input(intrsc_counts_nneg_index)
                      .add_input(selected_dim_indices_offsets)
                      .add_input(intrsc_first_match_nneg_index)
                      .build();

      AT_DISPATCH_INDEX_TYPES(
          nneg_index.scalar_type(), "index_select_sparse_xpu", [&]() {
            index_t* ptr_res_dim_indices =
                res_dim_indices.mutable_data_ptr<index_t>();
            index_t* ptr_selected_dim_indices =
                selected_dim_indices.mutable_data_ptr<index_t>();
            const index_t* ptr_argsort_dim_indices =
                argsort_dim_indices.const_data_ptr<index_t>();
            gpu_kernel(
                iter,
                IndexSelectSparse3Functor<index_t>(
                    ptr_res_dim_indices,
                    ptr_selected_dim_indices,
                    ptr_argsort_dim_indices));
          });

      return std::make_tuple(selected_dim_indices, res_dim_indices);
    }();

    return make_output(selected_dim_indices, res_dim_indices);
  }
  // If indexing into dense dimensions
  else {
    // It is sufficient to just perform `index_select` on values
    // if `dim` refers to dense dimensions.
    const auto res_values = values.index_select(dim - sparse_dim + 1, index);

    return _sparse_coo_tensor_with_dims_and_tensors(
        sparse_dim, dense_dim, res_sizes, indices, res_values, self.options());
  }
}

} // namespace at::native::xpu

#pragma GCC diagnostic pop
#pragma clang diagnostic pop
