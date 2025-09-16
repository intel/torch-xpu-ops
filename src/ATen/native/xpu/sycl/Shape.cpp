#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/NumericUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/ceil_div.h>
#include <ATen/native/CanUse32BitIndexMath.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/TensorShape.h>
#include <ATen/native/TypeProperties.h>
#include <ATen/native/xpu/sycl/MemoryAccessUtils.h>
#include <ATen/xpu/CachingHostAllocator.h>
#include <c10/core/MemoryFormat.h>
#include <comm/SYCLContext.h>
#include <comm/xpu_aten.h>

#include <ATen/ops/cat_native.h>
#include <ATen/ops/copy_native.h>
#include <ATen/ops/narrow.h>
#include <ATen/ops/size_native.h>

#include <ATen/native/xpu/sycl/Philox4x32.h>
#include <ATen/native/xpu/sycl/ShapeKernels.h>

namespace at::native::xpu {

constexpr int CAT_ARRAY_BATCH_SIZE = 64;
constexpr int CAT_ARRAY_MAX_INPUT_DIMS = 4;
constexpr int ALIGNED_VEC_LOAD_BYTES_16 = 16;
constexpr int ALIGNED_VEC_LOAD_BYTES_8 = 8;

inline bool is_aligned_vec4(const void* ptr) {
  auto iptr = reinterpret_cast<uintptr_t>(ptr);
  return !(iptr % alignof(uint4));
}

inline std::tuple<sycl::range<2>, sycl::range<2>> getCatRange(
    unsigned int max_elements_per_tensor,
    ptrdiff_t nTensors) {
  std::cout << "getCatRange---" << std::endl;
  constexpr unsigned int items_per_group = 256;
  constexpr unsigned int elements_per_item = 8;
  constexpr unsigned int max_group_per_eu = 32;

  unsigned int max_items = ceil_div(max_elements_per_tensor, elements_per_item);
  unsigned int item_groups = ceil_div(max_items, items_per_group);

  const unsigned int num_eu = syclGpuEUCountPerSubslice();
  item_groups = std::min(num_eu * max_group_per_eu, item_groups);

  sycl::range<2> global_range(
      (long long)nTensors, items_per_group * item_groups);
  sycl::range<2> local_range(1, item_groups);
  return std::make_tuple(global_range, local_range);
}

template <typename T, int aligned_vec_load_bytes>
inline std::tuple<sycl::range<2>, sycl::range<2>> getCatRangeContig(
    unsigned int max_elements_per_tensor,
    ptrdiff_t nTensors) {
  std::cout << "getCatRangeContig---" << std::endl;
  constexpr unsigned int items_per_group = 256;
  constexpr unsigned int min_aligned_vec_per_item = 1;
  constexpr unsigned int max_group_per_eu = 32;

  unsigned int elements_per_item =
      aligned_vec_load_bytes / sizeof(T) * min_aligned_vec_per_item;
  unsigned int max_items = ceil_div(max_elements_per_tensor, elements_per_item);
  unsigned int item_groups = ceil_div(max_items, items_per_group);

  const unsigned int num_eu = syclGpuEUCountPerSubslice();
  item_groups = std::min(num_eu * max_group_per_eu, item_groups);

  sycl::range<2> global_range(
      (long long)nTensors, item_groups * items_per_group);
  sycl::range<2> local_range(1, items_per_group);
  return std::make_tuple(global_range, local_range);
}

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
      IndexType curDimSize = i == concatDim ? dimSize : tensorSize[i];
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
struct CatArrayBatchedCopy {
  void operator()(sycl::nd_item<2> item) const {
    IndexType tid =
        item.get_group(1) * item.get_local_range(1) + item.get_local_id(1);
    IndexType nElements = inputs.nElements[item.get_group(0)];
    TensorSizeStride<IndexType, CAT_ARRAY_MAX_INPUT_DIMS> ins = stride_size > 1
        ? inputs.tensorStride[item.get_group(0)]
        : inputs.tensorStride[0];
    bool isContig = inputs.isContiguous[item.get_group(0)];

    if (tid >= nElements)
      return;

    const T* data = inputs.input[item.get_group(0)];
    IndexType offset = inputs.offset[item.get_group(0)];
    IndexType dimSize = inputs.dimSize[item.get_group(0)];
    IndexType dataOffset = offset * dimStride;

    IndexType stride = item.get_group_range(1) * item.get_local_range(1);

    while (tid < nElements) {
      IndexType elementOffset = CatArrIndexToOffset<IndexType, Dims>::compute(
          os.tensorSize, os.tensorStride, dimSize, concatDim, tid);
      if (isContig) {
        output[dataOffset + elementOffset] = data[tid];
      } else {
        IndexType inElementOffset =
            CatArrIndexToOffset<IndexType, Dims>::compute(
                ins.tensorSize, ins.tensorStride, dimSize, concatDim, tid);
        output[dataOffset + elementOffset] = data[inElementOffset];
      }
      tid += stride;
    }
  }

  CatArrayBatchedCopy(
      T* output,
      CatArrInputTensorMetadata<T, IndexType, batch_size, stride_size> inputs,
      TensorSizeStride<IndexType, CAT_ARRAY_MAX_INPUT_DIMS> os,
      const int concatDim,
      IndexType dimStride)
      : output(output),
        inputs(inputs),
        os(os),
        concatDim(concatDim),
        dimStride(dimStride) {}

 private:
  T* output;
  CatArrInputTensorMetadata<T, IndexType, batch_size, stride_size> inputs;
  TensorSizeStride<IndexType, CAT_ARRAY_MAX_INPUT_DIMS> os;
  const int concatDim;
  IndexType dimStride;
};

template <
    typename T,
    typename IndexType,
    int Dims,
    int batch_size,
    int stride_size>
struct CatArrayBatchedCopy_contig {
  void operator()(sycl::nd_item<2> item) const {
    IndexType tid =
        item.get_group(1) * item.get_local_range(1) + item.get_local_id(1);
    IndexType nElements = inputs.nElements[item.get_group(0)];

    if (tid >= nElements)
      return;

    const T* data = inputs.input[item.get_group(0)];
    IndexType offset = inputs.offset[item.get_group(0)];
    IndexType dimSize = inputs.dimSize[item.get_group(0)];
    IndexType dataOffset = offset * dimStride;

    IndexType stride = item.get_group_range(1) * item.get_local_range(1);

    while (tid < nElements) {
      IndexType elementOffset = CatArrIndexToOffset<IndexType, Dims>::compute(
          os.tensorSize, os.tensorStride, dimSize, concatDim, tid);
      output[dataOffset + elementOffset] = data[tid];
      tid += stride;
    }
  }

  CatArrayBatchedCopy_contig(
      T* output,
      CatArrInputTensorMetadata<T, IndexType, batch_size, stride_size> inputs,
      TensorSizeStride<IndexType, CAT_ARRAY_MAX_INPUT_DIMS> os,
      const int concatDim,
      IndexType dimStride)
      : output(output),
        inputs(inputs),
        os(os),
        concatDim(concatDim),
        dimStride(dimStride) {}

 private:
  T* output;
  CatArrInputTensorMetadata<T, IndexType, batch_size, stride_size> inputs;
  TensorSizeStride<IndexType, CAT_ARRAY_MAX_INPUT_DIMS> os;
  const int concatDim;
  IndexType dimStride;
};

/*
  Specialized implementation of the CatArrayBatchedCopy written to generate wide
  memory loads to improve memory bandwidth throughput.
*/

template <
    typename T,
    typename IndexType,
    int Dims,
    int batch_size,
    int stride_size,
    int aligned_vec_load_bytes>
struct CatArrayBatchedCopy_alignedK_contig {
  void operator()(sycl::nd_item<2> item) const {
    // This kernel tries to use aligned_vec_load_bytes*8 bit loads
    // Special case 2-byte types to use 8-byte vec loads to reduce register
    // pressure The below lambda is to allow cc compiler to pass kILP>0 checks
    // for large types (e.g. ComplexDouble, 16 bytes)
    constexpr int kILP = aligned_vec_load_bytes / sizeof(T) > 0
        ? aligned_vec_load_bytes / sizeof(T)
        : ALIGNED_VEC_LOAD_BYTES_16 / sizeof(T);

    IndexType inputOffset =
        (item.get_group(1) * item.get_local_range(1) + item.get_local_id(1)) *
        kILP;
    IndexType inputStride =
        item.get_group_range(1) * item.get_local_range(1) * kILP;

    IndexType nElements = inputs.nElements[item.get_group(0)];
    if (inputOffset >= nElements) {
      return;
    }

    const T* data = inputs.input[item.get_group(0)];
    IndexType offset = inputs.offset[item.get_group(0)];
    IndexType dimSize = inputs.dimSize[item.get_group(0)];
    IndexType dataOffset = offset * dimStride;

    IndexType v_elementOffset[kILP];
    T reg_data[kILP];

    while (inputOffset + kILP <= nElements) {
      for (int i = 0; i < kILP; ++i) {
        v_elementOffset[i] = CatArrIndexToOffset<IndexType, Dims>::compute(
            os.tensorSize,
            os.tensorStride,
            dimSize,
            concatDim,
            inputOffset + i);
      }

      using LT = memory::aligned_vector<T, kILP>;
      ((LT*)reg_data)[0] = const_cast<LT*>((LT*)(data + inputOffset))[0];

#pragma unroll
      for (int i = 0; i < kILP; ++i) {
        output[dataOffset + v_elementOffset[i]] = reg_data[i];
      }

      inputOffset += inputStride;
    }

    // Handle remaining tail in case nElements does not divide
    // exactly to kILP

    while (inputOffset < nElements) {
      v_elementOffset[0] = CatArrIndexToOffset<IndexType, Dims>::compute(
          os.tensorSize, os.tensorStride, dimSize, concatDim, inputOffset);
      output[dataOffset + v_elementOffset[0]] = data[inputOffset];
      inputOffset++;
    }
  }

  CatArrayBatchedCopy_alignedK_contig(
      T* output,
      CatArrInputTensorMetadata<T, IndexType, batch_size, stride_size> inputs,
      TensorSizeStride<IndexType, CAT_ARRAY_MAX_INPUT_DIMS> os,
      const int concatDim,
      IndexType dimStride)
      : output(output),
        inputs(inputs),
        os(os),
        concatDim(concatDim),
        dimStride(dimStride) {}

 private:
  T* output;
  CatArrInputTensorMetadata<T, IndexType, batch_size, stride_size> inputs;
  TensorSizeStride<IndexType, CAT_ARRAY_MAX_INPUT_DIMS> os;
  const int concatDim;
  IndexType dimStride;
};

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
    TORCH_CHECK(false, "unsupported memory format");
  }

  // If all batches are contiguous we can call a specialized implementation
  // which requires the input tensor addresses to be aligned to a
  // 16 Byte boundary.

  bool isContig = true;
  bool isAligned = true;
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
      // CatArrayBatchedCopy_alignedK_contig
      isAligned &= is_aligned_vec4(catMetaData.input[batchCounter]);

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

      // We need max elements per tensor to compute range parameters
      max_elements_per_tensor = std::max(
          max_elements_per_tensor, catMetaData.nElements[batchCounter]);
    }

    // Skip if the tensor is empty. Otherwise, the range dim is invalid
    if (max_elements_per_tensor == 0)
      continue;

    isContig = false;
    sycl::range<2> applyGroup, catRange;
    if (isContig && sizeof(scalar_t) > 2) {
      std::tie(catRange, applyGroup) =
          getCatRangeContig<scalar_t, ALIGNED_VEC_LOAD_BYTES_16>(
              max_elements_per_tensor, batchCounter);
    } else if (isContig && sizeof(scalar_t) == 2) {
      std::tie(catRange, applyGroup) =
          getCatRangeContig<scalar_t, ALIGNED_VEC_LOAD_BYTES_8>(
              max_elements_per_tensor, batchCounter);
    } else {
      std::tie(catRange, applyGroup) =
          getCatRange(max_elements_per_tensor, batchCounter);
    }

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
#define HANDLE_CASE(DIMS)                                                      \
  if (isContig && isAligned && sizeof(scalar_t) > 2 &&                         \
      sizeof(scalar_t) <= 8) {                                                 \
    CatArrayBatchedCopy_alignedK_contig<                                       \
        scalar_t,                                                              \
        unsigned int,                                                          \
        DIMS,                                                                  \
        batch_size,                                                            \
        stride_size,                                                           \
        ALIGNED_VEC_LOAD_BYTES_16>                                             \
        kfn(data,                                                              \
            catMetaData,                                                       \
            outputParam,                                                       \
            dimension,                                                         \
            outputParam.tensorStride[dimension]);                              \
    auto& q = getCurrentSYCLQueue();                                           \
    sycl_kernel_submit(catRange, applyGroup, q, kfn);                          \
  } else if (isContig && isAligned && sizeof(scalar_t) == 2) {                 \
    CatArrayBatchedCopy_alignedK_contig<                                       \
        scalar_t,                                                              \
        unsigned int,                                                          \
        DIMS,                                                                  \
        batch_size,                                                            \
        stride_size,                                                           \
        ALIGNED_VEC_LOAD_BYTES_8>                                              \
        kfn(data,                                                              \
            catMetaData,                                                       \
            outputParam,                                                       \
            dimension,                                                         \
            outputParam.tensorStride[dimension]);                              \
    auto& q = getCurrentSYCLQueue();                                           \
    sycl_kernel_submit(catRange, applyGroup, q, kfn);                          \
  } else if (isContig) {                                                       \
    CatArrayBatchedCopy_contig<                                                \
        scalar_t,                                                              \
        unsigned int,                                                          \
        DIMS,                                                                  \
        batch_size,                                                            \
        stride_size>                                                           \
        kfn(data,                                                              \
            catMetaData,                                                       \
            outputParam,                                                       \
            dimension,                                                         \
            outputParam.tensorStride[dimension]);                              \
    auto& q = getCurrentSYCLQueue();                                           \
    sycl_kernel_submit(catRange, applyGroup, q, kfn);                          \
  } else {                                                                     \
    CatArrayBatchedCopy<scalar_t, unsigned int, DIMS, batch_size, stride_size> \
        kfn(data,                                                              \
            catMetaData,                                                       \
            outputParam,                                                       \
            dimension,                                                         \
            outputParam.tensorStride[dimension]);                              \
    auto& q = getCurrentSYCLQueue();                                           \
    sycl_kernel_submit(catRange, applyGroup, q, kfn);                          \
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

// The kernels are templated on an opaque, self-aligned type of the correct
// size to avoid redundant kernels for different types of the same size.
template <unsigned N>
struct alignas(N) OpaqueType {
  char data[N];
};

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
        return canUse32BitIndexMath(t);
      });

  int nDims = materialized[valid].get().dim();

  // We support the contiguous inputs and non-contiguous input (<=4 dims) in
  // different ways For contiguous input, we don't need to pass stride meta data
  // to kernel through constant memory. Therefore, we could pass more inputs to
  // threads. For non-contiguous, we reduce the number of inputs passed to
  // kernel due to the limitation of constant memory.

  if (materialized.size() > 1 && result.dim() <= CAT_ARRAY_MAX_INPUT_DIMS &&
      canUse32BitIndexMath(result) && all_contiguous && all32BitIndexable &&
      all_same_dtype) {
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
          AT_EXPAND(AT_FLOAT8_TYPES),
          AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES),
          kFloat4_e2m1fn_x2);
    }
  } else if (
      materialized.size() > 1 && result.dim() <= CAT_ARRAY_MAX_INPUT_DIMS &&
      canUse32BitIndexMath(result) && nDims <= CAT_ARRAY_MAX_INPUT_DIMS &&
      all32BitIndexable && all_same_dtype &&
      memory_format == c10::MemoryFormat::Contiguous) {
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
          kFloat8_e4m3fn,
          kFloat8_e4m3fnuz,
          kFloat8_e5m2,
          kFloat8_e5m2fnuz,
          AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES),
          kFloat4_e2m1fn_x2);
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
