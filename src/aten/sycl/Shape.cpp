#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/NumericUtils.h>
#include <ATen/native/CanUse32BitIndexMath.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/TensorShape.h>
#include <aten/sycl/Loops.h>
#include <comm/SYCLContext.h>

namespace at::native::xpu {

template <unsigned N>
struct alignas(N) OpaqueType {
  char data[N];
};

constexpr int CAT_ARRAY_BATCH_SIZE = 64;
constexpr int CAT_ARRAY_MAX_INPUT_DIMS = 4;
constexpr int ALIGNED_VEC_LOAD_BYTES = 16;

// Similar to any other IndexToOffset calculation for copying along a given
// dimension.
template <typename IndexType, int Dims>
struct CatArrIndexToOffset {
  static inline IndexType compute(
      const IndexType tensorSize[Dims],
      const IndexType tensorStride[Dims],
      const IndexType dimSize,
      const unsigned int concatDim,
      IndexType linearIndex) {
    // linearIndex is not really linear index, but instead the offset in
    // input tensor. If the input tensor is contiguous, then this offset
    // is the linear index, but if the input tensor is channels last, then
    // it is the linear index of the permuted contiguous tensor
    IndexType offset = 0;

#pragma unroll
    for (int i = Dims - 1; i >= 1; --i) {
      IndexType curDimSize =
          (unsigned int)i == concatDim ? dimSize : tensorSize[i];
      IndexType nextDimIndex = linearIndex / curDimSize;
      IndexType curDimIndex = linearIndex - curDimSize * nextDimIndex;
      IndexType curDimOffset = curDimIndex * tensorStride[i];
      offset += curDimOffset;
      linearIndex = nextDimIndex;
    }

    return offset + linearIndex * tensorStride[0];
  }
};

template <typename IndexType, unsigned int MaxDims>
struct TensorSizeStride {
  IndexType tensorSize[MaxDims];
  IndexType tensorStride[MaxDims];
};

// pass meta data directly through kernel argument instead of pin memory
// In contiguous case, we will not need stride_size, setting it as 1 as
// placeholder to pass compile.
template <typename T, typename IndexType, int n, int stride_size>
struct CatArrInputTensorMetadata {
  const T* input[n];
  IndexType offset[n];
  IndexType dimSize[n];
  IndexType nElements[n];
  bool isContiguous[n];
  TensorSizeStride<IndexType, CAT_ARRAY_MAX_INPUT_DIMS>
      tensorStride[stride_size];
};

template <
    typename T,
    typename IndexType,
    int Dims,
    int batch_size,
    int stride_size>
struct CatArrayBatchedCopyFunctor {
  void operator()(sycl::nd_item<2> item) const {
    IndexType tid =
        item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
    IndexType bid = item.get_group(1);

    IndexType nElements = inputs_.nElements[bid];
    TensorSizeStride<IndexType, CAT_ARRAY_MAX_INPUT_DIMS> ins =
        stride_size > 1 ? inputs_.tensorStride[bid] : inputs_.tensorStride[0];
    bool isContig = inputs_.isContiguous[bid];

    if (tid >= nElements)
      return;

    const T* data = inputs_.input[bid];
    IndexType offset = inputs_.offset[bid];
    IndexType dimSize = inputs_.dimSize[bid];
    IndexType dataOffset = offset * dimStride_;

    IndexType stride = item.get_group_range(0) * item.get_local_range(0);

    while (tid < nElements) {
      IndexType elementOffset = CatArrIndexToOffset<IndexType, Dims>::compute(
          os_.tensorSize, os_.tensorStride, dimSize, concatDim_, tid);
      if (isContig) {
        output_[dataOffset + elementOffset] = data[tid];
      } else {
        IndexType inElementOffset =
            CatArrIndexToOffset<IndexType, Dims>::compute(
                ins.tensorSize, ins.tensorStride, dimSize, concatDim_, tid);
        output_[dataOffset + elementOffset] = data[inElementOffset];
      }
      tid += stride;
    }
  }
  CatArrayBatchedCopyFunctor(
      T* output,
      CatArrInputTensorMetadata<T, IndexType, batch_size, stride_size> inputs,
      TensorSizeStride<IndexType, CAT_ARRAY_MAX_INPUT_DIMS> os,
      const int concatDim,
      IndexType dimStride)
      : output_(output),
        inputs_(inputs),
        os_(os),
        concatDim_(concatDim),
        dimStride_(dimStride) {}

 private:
  T* output_;
  CatArrInputTensorMetadata<T, IndexType, batch_size, stride_size> inputs_;
  TensorSizeStride<IndexType, CAT_ARRAY_MAX_INPUT_DIMS> os_;
  const int concatDim_;
  IndexType dimStride_;
};

template <
    typename T,
    typename IndexType,
    int Dims,
    int batch_size,
    int stride_size>
struct CatArrayBatchedCopyContigFunctor {
  void operator()(sycl::nd_item<2> item) const {
    IndexType tid =
        item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
    IndexType bid = item.get_group(1);
    IndexType nElements = inputs_.nElements[bid];

    if (tid >= nElements)
      return;

    const T* data = inputs_.input[bid];
    IndexType offset = inputs_.offset[bid];
    IndexType dimSize = inputs_.dimSize[bid];
    IndexType dataOffset = offset * dimStride_;

    IndexType stride = item.get_group_range(0) * item.get_local_range(0);

    while (tid < nElements) {
      IndexType elementOffset = CatArrIndexToOffset<IndexType, Dims>::compute(
          os_.tensorSize, os_.tensorStride, dimSize, concatDim_, tid);

      output_[dataOffset + elementOffset] = data[tid];
      tid += stride;
    }
  }
  CatArrayBatchedCopyContigFunctor(
      T* output,
      CatArrInputTensorMetadata<T, IndexType, batch_size, stride_size> inputs,
      TensorSizeStride<IndexType, CAT_ARRAY_MAX_INPUT_DIMS> os,
      const int concatDim,
      IndexType dimStride)
      : output_(output),
        inputs_(inputs),
        os_(os),
        concatDim_(concatDim),
        dimStride_(dimStride) {}

 private:
  T* output_;
  CatArrInputTensorMetadata<T, IndexType, batch_size, stride_size> inputs_;
  TensorSizeStride<IndexType, CAT_ARRAY_MAX_INPUT_DIMS> os_;
  const int concatDim_;
  IndexType dimStride_;
};

inline std::tuple<unsigned int, unsigned int> get_cat_range(
    ptrdiff_t nTensors) {
  const int sum_ss = ::syclMaxDSSNum();
  return std::make_tuple(2LL * sum_ss, (long long)nTensors);
}

template <typename T>
inline std::tuple<unsigned int, unsigned int, unsigned int> get_cat_range_contig(
    unsigned int max_elements_per_tensor,
    ptrdiff_t nTensors) {
  constexpr unsigned int wi_per_group = 256;
  constexpr unsigned int min_aligned_vec_per_wi = 1;
  constexpr unsigned int max_group_per_ss = 32;

  unsigned int elements_per_wi =
      ALIGNED_VEC_LOAD_BYTES / sizeof(T) * min_aligned_vec_per_wi;
  unsigned int max_wi = ceil_div(max_elements_per_tensor, elements_per_wi);
  unsigned int ngroups = ceil_div(max_wi, wi_per_group);

  // Limit the number of thread blocks to prevent too many threads to load the
  // metadata if they operate on very small tensors.

  const unsigned int num_ss = ::syclMaxDSSNum();

  ngroups = std::min(num_ss * max_group_per_ss, ngroups);

  return std::make_tuple(ngroups, (long long)nTensors, wi_per_group);
}

template <typename scalar_t, int batch_size, int stride_size>
void parallel_cat(
    const Tensor& out,
    const MaterializedITensorListRef& inputs,
    int64_t dimension,
    int nDims,
    c10::MemoryFormat memory_format) {
  // First, let's set up our kernel parameters. We start with a raw pointer to
  // the storage for the output Tensor.
  scalar_t* data = (scalar_t*)(out.mutable_data_ptr());
  CatArrInputTensorMetadata<scalar_t, unsigned int, batch_size, stride_size>
      catMetaData;
  TensorSizeStride<unsigned int, CAT_ARRAY_MAX_INPUT_DIMS> outputParam;

  // Next, let's initialize the size, stride arrays for the output Tensor.
  if (memory_format == c10::MemoryFormat::Contiguous) {
    for (int i = 0; i < nDims; ++i) {
      outputParam.tensorSize[i] = out.size(i);
      outputParam.tensorStride[i] = out.stride(i);
    }
  } else if (
      memory_format == c10::MemoryFormat::ChannelsLast ||
      memory_format == c10::MemoryFormat::ChannelsLast3d) {
    // permute the semantics of dims from NCHW to NHWC so that the input
    // tensor is now contiguous
    outputParam.tensorSize[0] = out.size(0);
    outputParam.tensorStride[0] = out.stride(0);
    for (int i = 1; i < nDims - 1; ++i) {
      outputParam.tensorSize[i] = out.size(i + 1);
      outputParam.tensorStride[i] = out.stride(i + 1);
    }
    outputParam.tensorSize[nDims - 1] = out.size(1);
    outputParam.tensorStride[nDims - 1] = out.stride(1);
  } else {
    TORCH_CHECK(false, "unsupported memory format:", memory_format);
  }

  auto queue = at::xpu::getCurrentSYCLQueue();

  // If all batches are contiguous we can call a specialized implementation
  // which requires the input tensor addresses to be aligned to a
  // 16 Byte boundary.

  bool isContig = true;
  // bool isAligned = true;
  unsigned int max_elements_per_tensor = 0;

  // Now we loop
  int batchCounter = 0;
  int64_t offset = 0;
  for (unsigned i = 0; i < inputs.size(); i += batch_size) {
    for (batchCounter = 0;
         batchCounter < batch_size && (i + batchCounter) < inputs.size();
         ++batchCounter) {
      int64_t dimSize = 0;
      // There is a legacy case where a 1-D empty tensor can be concat with
      // high-dimensional tensor
      if (inputs[i + batchCounter].get().numel() > 0) {
        dimSize = inputs[i + batchCounter].get().size(dimension);
      }

      catMetaData.input[batchCounter] =
          (scalar_t*)(inputs[i + batchCounter].get().const_data_ptr());
      catMetaData.offset[batchCounter] = offset;
      catMetaData.dimSize[batchCounter] = dimSize;
      catMetaData.nElements[batchCounter] =
          inputs[i + batchCounter].get().numel();

      // If at least one of the inputs is not aligned, we can't call the
      //   isAligned &= is_aligned_vec4(catMetaData.input[batchCounter]);

      if (stride_size > 1) {
        auto strides = inputs[i + batchCounter].get().strides();
        auto sizes = inputs[i + batchCounter].get().sizes();
        for (int j = 0; j < nDims; j++) {
          catMetaData.tensorStride[batchCounter].tensorSize[j] = sizes[j];
          catMetaData.tensorStride[batchCounter].tensorStride[j] = strides[j];
        }
        catMetaData.isContiguous[batchCounter] = false;
        isContig = false;
      } else {
        catMetaData.isContiguous[batchCounter] = true;
      }

      // Update offset
      offset += dimSize;

      // We need max elements per tensor to compute parameters
      max_elements_per_tensor = std::max(
          max_elements_per_tensor, catMetaData.nElements[batchCounter]);
    }

    // Skip if the tensor is empty. Otherwise, the global range dim is invalid
    if (max_elements_per_tensor == 0)
      continue;

    unsigned int group_range_x, group_range_y, local_range;

    if (isContig && sizeof(scalar_t) > 2) {
      std::tuple<unsigned int, unsigned int, unsigned int> launchParams =
          get_cat_range_contig<scalar_t>(max_elements_per_tensor, batchCounter);
      group_range_x = std::get<0>(launchParams);
      group_range_y = std::get<1>(launchParams);
      local_range = std::get<2>(launchParams);
    } else {
      local_range = 32 * 32;
      auto launchParams = get_cat_range(batchCounter);
      group_range_x = std::get<0>(launchParams);
      group_range_y = std::get<1>(launchParams);
    }
    auto global_range_ =
        sycl::range<2>(group_range_x * local_range, group_range_y);
    auto local_range_ = sycl::range<2>(local_range, 1);

    if (memory_format != c10::MemoryFormat::Contiguous) {
      switch (dimension) {
        case 0:
          break;
        case 1:
          dimension = nDims - dimension;
          break;
        default:
          dimension--;
      }
    }
    // Template Declarations for dim = 1, 2, 3, 4
#define HANDLE_CASE(DIMS)                                      \
  if (isContig) {                                              \
    auto f = CatArrayBatchedCopyContigFunctor<                 \
        scalar_t,                                              \
        unsigned int,                                          \
        DIMS,                                                  \
        batch_size,                                            \
        stride_size>(                                          \
        data,                                                  \
        catMetaData,                                           \
        outputParam,                                           \
        dimension,                                             \
        outputParam.tensorStride[dimension]);                  \
    sycl_kernel_submit(global_range_, local_range_, queue, f); \
  } else {                                                     \
    auto f = CatArrayBatchedCopyFunctor<                       \
        scalar_t,                                              \
        unsigned int,                                          \
        DIMS,                                                  \
        batch_size,                                            \
        stride_size>(                                          \
        data,                                                  \
        catMetaData,                                           \
        outputParam,                                           \
        dimension,                                             \
        outputParam.tensorStride[dimension]);                  \
    sycl_kernel_submit(global_range_, local_range_, queue, f); \
  }

    switch (nDims) {
      case 1:
        HANDLE_CASE(1);
        break;
      case 2:
        HANDLE_CASE(2);
        break;
      case 3:
        HANDLE_CASE(3);
        break;
      case 4:
        HANDLE_CASE(4);
        break;
    }
#undef HANDLE_CASE
  }
}

void cat_out_kernel(
    const ITensorListRef& tensors,
    int64_t dim,
    int64_t valid,
    bool all_contiguous,
    bool all_same_dtype,
    bool all_same_sizes_and_stride,
    MemoryFormat memory_format,
    const Tensor& result) {
  if (result.numel() == 0) {
    return;
  }

  auto materialized = tensors.materialize();

  // We parallelize the copy if all 6 conditions pass:
  //
  // 1. There is more than one input tensor
  // 2. The out tensor is 32-bit indexable
  // 3. The number of dimensions is <= 4
  // 4. All input tensors are contiguous (output tensor may be non-contig)
  // 5. All input tensors can use 32-bit indexing

  const bool all32BitIndexable = std::all_of(
      materialized.begin(), materialized.end(), [](const Tensor& t) {
        return at::native::canUse32BitIndexMath(t);
      });

  int nDims = materialized[valid].get().dim();

  // We support the contiguous inputs and non-contiguous input (<=4 dims) in
  // different ways For contiguous input, we don't need to pass stride meta data
  // to xpu kernel through constant memory. Therefore, we could pass more inputs
  // to xpu threads. For non-contiguous, we reduce the number of inputs passed
  // to xpu kernel due to the limitation of constant memory.

  if (materialized.size() > 1 && result.dim() <= CAT_ARRAY_MAX_INPUT_DIMS &&
      at::native::canUse32BitIndexMath(result) && all_contiguous &&
      all32BitIndexable && all_same_dtype) {
    if (isBitsType(result.scalar_type())) {
      AT_DISPATCH_BIT_TYPES(result.scalar_type(), "cat_xpu", [&]() {
        using dtype = OpaqueType<sizeof(scalar_t)>;
        parallel_cat<dtype, CAT_ARRAY_BATCH_SIZE, 1>(
            result, materialized, dim, nDims, memory_format);
      });
    } else {
      AT_DISPATCH_V2(
          result.scalar_type(),
          "cat_xpu",
          AT_WRAP([&]() {
            using dtype = OpaqueType<sizeof(scalar_t)>;
            parallel_cat<dtype, CAT_ARRAY_BATCH_SIZE, 1>(
                result, materialized, dim, nDims, memory_format);
          }),
          AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX),
          kComplexHalf,
          kHalf,
          kBool,
          kBFloat16,
          AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
    }
  } else if (
      materialized.size() > 1 && result.dim() <= CAT_ARRAY_MAX_INPUT_DIMS &&
      at::native::canUse32BitIndexMath(result) &&
      nDims <= CAT_ARRAY_MAX_INPUT_DIMS && all32BitIndexable &&
      all_same_dtype && memory_format == c10::MemoryFormat::Contiguous) {
    if (isBitsType(result.scalar_type())) {
      AT_DISPATCH_BIT_TYPES(result.scalar_type(), "cat_xpu", [&]() {
        using dtype = OpaqueType<sizeof(scalar_t)>;
        parallel_cat<dtype, CAT_ARRAY_BATCH_SIZE / 2, CAT_ARRAY_BATCH_SIZE / 2>(
            result, materialized, dim, nDims, memory_format);
      });
    } else {
      AT_DISPATCH_V2(
          result.scalar_type(),
          "cat_xpu",
          AT_WRAP([&]() {
            using dtype = OpaqueType<sizeof(scalar_t)>;
            parallel_cat<
                dtype,
                CAT_ARRAY_BATCH_SIZE / 2,
                CAT_ARRAY_BATCH_SIZE / 2>(
                result, materialized, dim, nDims, memory_format);
          }),
          AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX),
          kComplexHalf,
          kHalf,
          kBool,
          kBFloat16,
          AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
    }
  } else {
    int64_t offset = 0;
    for (const Tensor& t : materialized) {
      if (cat_should_skip_tensor(t))
        continue;
      int64_t dimSize = t.size(dim);
      Tensor nt = at::narrow(result, dim, offset, dimSize);
      copy_(nt, t);
      offset += dimSize;
    }
  }
}

} // namespace at::native::xpu
