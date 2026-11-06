/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/Dispatch_v2.h>
#include <ATen/NumericUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/ceil_div.h>
#include <ATen/native/CanUse32BitIndexMath.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/TensorShape.h>
#include <ATen/native/TypeProperties.h>
#include <ATen/native/xpu/sycl/IntegerDivider.h>
#include <ATen/native/xpu/sycl/MemoryAccessUtils.h>
#include <ATen/xpu/XPUContext.h>
#include <c10/core/MemoryFormat.h>
#include <c10/util/SmallVector.h>
#include <comm/SYCLContext.h>
#include <comm/xpu_aten.h>

#include <ATen/ops/cat_native.h>
#include <ATen/ops/copy_native.h>
#include <ATen/ops/narrow.h>
#include <ATen/ops/size_native.h>

#include <algorithm>
#include <cstddef>
#include <numeric>

#include <ATen/native/xpu/sycl/Philox4x32.h>
#include <ATen/native/xpu/sycl/ShapeKernels.h>

namespace at::native::xpu {

constexpr int CAT_ARRAY_BATCH_SIZE = 64;
constexpr int CAT_ARRAY_BATCH_SIZE_STRIDED = 16;
constexpr int CAT_ARRAY_MAX_INPUT_DIMS = 4;
constexpr int ALIGNED_VEC_LOAD_BYTES_16 = 16;
constexpr int ALIGNED_VEC_LOAD_BYTES_8 = 8;

inline bool is_aligned_vec4(const void* ptr) {
  auto iptr = reinterpret_cast<uintptr_t>(ptr);
  return !(iptr % alignof(uint4));
}

inline bool is_aligned_to(const void* ptr, int64_t bytes) {
  return !(reinterpret_cast<uintptr_t>(ptr) % static_cast<uintptr_t>(bytes));
}

inline std::tuple<sycl::range<2>, sycl::range<2>> getCatRange(
    unsigned int max_elements_per_tensor,
    ptrdiff_t nTensors) {
  constexpr unsigned int items_per_group = 256;
  constexpr unsigned int elements_per_item = 8;

  unsigned int max_items = ceil_div(max_elements_per_tensor, elements_per_item);
  unsigned int item_groups = ceil_div(max_items, items_per_group);

  // One wave is enough to saturate the tile. The kernel is grid-strided, so
  // additional groups do not increase parallelism and can increase idle work
  // for smaller tensors in the batch.
  const unsigned int max_item_groups = static_cast<unsigned int>(
      at::xpu::getDeviceMaxWorkItems() / items_per_group);
  item_groups = std::min(max_item_groups, item_groups);

  sycl::range<2> global_range(
      (long long)nTensors, items_per_group * item_groups);
  sycl::range<2> local_range(1, items_per_group);
  return std::make_tuple(global_range, local_range);
}

template <typename T, int aligned_vec_load_bytes>
inline std::tuple<sycl::range<2>, sycl::range<2>> getCatRangeContig(
    unsigned int max_elements_per_tensor,
    ptrdiff_t nTensors) {
  constexpr unsigned int items_per_group = 256;
  constexpr unsigned int min_aligned_vec_per_item = 1;

  unsigned int elements_per_item =
      aligned_vec_load_bytes / sizeof(T) * min_aligned_vec_per_item;
  unsigned int max_items = ceil_div(max_elements_per_tensor, elements_per_item);
  unsigned int item_groups = ceil_div(max_items, items_per_group);

  const unsigned int max_item_groups = static_cast<unsigned int>(
      at::xpu::getDeviceMaxWorkItems() / items_per_group);
  item_groups = std::min(max_item_groups, item_groups);

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
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<2>))
void cat_array_batched_copy_kernel_impl(
    T* output,
    CatArrInputTensorMetadata<T, IndexType, batch_size, stride_size> inputs,
    TensorSizeStride<IndexType, CAT_ARRAY_MAX_INPUT_DIMS> os,
    const int concatDim,
    IndexType dimStride) {
  auto item = syclext::this_work_item::get_nd_item<2>();
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
      IndexType inElementOffset = CatArrIndexToOffset<IndexType, Dims>::compute(
          ins.tensorSize, ins.tensorStride, dimSize, concatDim, tid);
      output[dataOffset + elementOffset] = data[inElementOffset];
    }
    tid += stride;
  }
}

template <
    typename T,
    typename IndexType,
    int Dims,
    int batch_size,
    int stride_size>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<2>))
void cat_array_batched_copy_contig_kernel_impl(
    T* output,
    CatArrInputTensorMetadata<T, IndexType, batch_size, stride_size> inputs,
    TensorSizeStride<IndexType, CAT_ARRAY_MAX_INPUT_DIMS> os,
    const int concatDim,
    IndexType dimStride) {
  auto item = syclext::this_work_item::get_nd_item<2>();
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
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<2>))
void cat_array_batched_copy_aligned_k_contig_kernel_impl(
    T* output,
    CatArrInputTensorMetadata<T, IndexType, batch_size, stride_size> inputs,
    TensorSizeStride<IndexType, CAT_ARRAY_MAX_INPUT_DIMS> os,
    const int concatDim,
    IndexType dimStride) {
  auto item = syclext::this_work_item::get_nd_item<2>();
  // Use 8-byte vector loads for 2-byte element types to reduce register
  // pressure. Fall back to 16-byte loads so kILP stays positive for supported
  // element types wider than the requested load width (for example,
  // ComplexDouble).
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

  while (inputOffset + kILP <= nElements) {
    for (int i = 0; i < kILP; ++i) {
      v_elementOffset[i] = CatArrIndexToOffset<IndexType, Dims>::compute(
          os.tensorSize, os.tensorStride, dimSize, concatDim, inputOffset + i);
    }

    using LT = memory::aligned_vector<T, kILP>;
    LT vec_data = const_cast<const LT*>((const LT*)(data + inputOffset))[0];

#pragma unroll
    for (int i = 0; i < kILP; ++i) {
      output[dataOffset + v_elementOffset[i]] = vec_data.val[i];
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

// Like CatArrayBatchedCopy_alignedK_contig, but vectorizes both loads and
// stores. When each input slice and the output row pitch are multiples of the
// vector width, each input vector maps to one contiguous output vector. This
// allows a single wide store and a simple direct output-offset calculation.
template <
    typename T,
    typename IndexType,
    int batch_size,
    int stride_size,
    int aligned_vec_load_bytes>
struct CatArrayBatchedCopy_vectorized {
  void operator()(sycl::nd_item<2> item) const {
    static_assert(
        aligned_vec_load_bytes % sizeof(T) == 0,
        "the vector width must be a whole number of elements");
    constexpr int kILP = aligned_vec_load_bytes / sizeof(T);
    using LT = memory::aligned_vector<T, kILP>;

    const IndexType group = item.get_group(0);

    // Exact: the host guarantees whole vectors per slice.
    const IndexType nVectors = inputs.nElements[group] / kILP;

    IndexType vectorIndex =
        item.get_group(1) * item.get_local_range(1) + item.get_local_id(1);

    const IndexType inputSliceVectors =
        inputs.dimSize[group] * dimStride / kILP;
    const IndexType outputSliceVectors = outputDimSize * dimStride / kILP;
    const IndexType outputOffsetVectors =
        inputs.offset[group] * dimStride / kILP;
    const IndexType stride = item.get_group_range(1) * item.get_local_range(1);

    const LT* in = reinterpret_cast<const LT*>(inputs.input[group]);
    LT* out = reinterpret_cast<LT*>(output);

    while (vectorIndex < nVectors) {
      const IndexType outerIndex = vectorIndex / inputSliceVectors;
      const IndexType innerIndex = vectorIndex - outerIndex * inputSliceVectors;
      out[outerIndex * outputSliceVectors + outputOffsetVectors + innerIndex] =
          in[vectorIndex];
      vectorIndex += stride;
    }
  }

  CatArrayBatchedCopy_vectorized(
      T* output,
      CatArrInputTensorMetadata<T, IndexType, batch_size, stride_size> inputs,
      IndexType outputDimSize,
      IndexType dimStride)
      : output(output),
        inputs(inputs),
        outputDimSize(outputDimSize),
        dimStride(dimStride) {}

 private:
  T* output;
  CatArrInputTensorMetadata<T, IndexType, batch_size, stride_size> inputs;
  IndexType outputDimSize;
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
  constexpr int kAlignedKernelBytes = 16;

  const int64_t logical_dimension = dimension;
  int64_t mapped_dimension = logical_dimension;
  if (memory_format != c10::MemoryFormat::Contiguous) {
    switch (mapped_dimension) {
      case 0:
        break;
      case 1:
        mapped_dimension = nDims - mapped_dimension;
        break;
      default:
        mapped_dimension--;
    }
  }
  bool isOutputAligned = is_aligned_vec4(data) &&
      (logical_dimension == 0 ||
       (outputParam.tensorStride[mapped_dimension] * sizeof(scalar_t)) %
               kAlignedKernelBytes ==
           0);

  // Cache per-input geometry for vectorization, sorting, batching, and metadata
  // setup. Empty tensors are handled here to avoid querying an invalid
  // dimension.
  constexpr size_t kInlineInputs = 8;
  const unsigned nInputs = static_cast<unsigned>(inputs.size());
  c10::SmallVector<int64_t, kInlineInputs> nElements(nInputs);
  c10::SmallVector<int64_t, kInlineInputs> dimSize(nInputs);
  // Keep output offsets in the original input order, independent of batching
  // order.
  c10::SmallVector<int64_t, kInlineInputs> inputOffset(nInputs);
  c10::SmallVector<unsigned, kInlineInputs> order(nInputs);
  {
    int64_t running = 0;
    for (unsigned k = 0; k < nInputs; ++k) {
      const Tensor& input = inputs[k].get();
      nElements[k] = input.numel();
      dimSize[k] = nElements[k] > 0 ? input.size(logical_dimension) : 0;
      inputOffset[k] = running;
      running += dimSize[k];
      order[k] = k;
    }
    // Sort inputs by size so tensors with similar workloads are batched
    // together, reducing idle work from smaller tensors in the same launch.
    std::stable_sort(order.begin(), order.end(), [&](unsigned a, unsigned b) {
      return nElements[a] > nElements[b];
    });
  }

  // Vectorized stores must stay within each input slice. Require both input
  // slice sizes and the output slice size to be multiples of the vector width,
  // so all slice offsets remain aligned.
  const int64_t dimStride = outputParam.tensorStride[mapped_dimension];
  const int64_t outputSliceBytes =
      static_cast<int64_t>(outputParam.tensorSize[mapped_dimension]) *
      dimStride * static_cast<int64_t>(sizeof(scalar_t));
  auto fitsVector = [&](int64_t bytes) {
    if (bytes < static_cast<int64_t>(sizeof(scalar_t)) ||
        !is_aligned_to(data, bytes) || outputSliceBytes % bytes != 0) {
      return false;
    }
    for (unsigned k = 0; k < nInputs; ++k) {
      if ((dimSize[k] * dimStride * static_cast<int64_t>(sizeof(scalar_t))) %
              bytes !=
          0) {
        return false;
      }
    }
    return true;
  };
  // The strided instantiation cannot address its inputs linearly.
  const int64_t vectorBytes = stride_size > 1 ? 0
      : fitsVector(ALIGNED_VEC_LOAD_BYTES_16) ? ALIGNED_VEC_LOAD_BYTES_16
      : fitsVector(ALIGNED_VEC_LOAD_BYTES_8)  ? ALIGNED_VEC_LOAD_BYTES_8
                                              : 0;

  // Process inputs in size-sorted batches.
  int batchCounter = 0;
  unsigned i = 0;
  while (i < nInputs) {
    // The first input is the largest in this batch.
    const int64_t batchMax = nElements[order[i]];
    // All remaining inputs are empty, so there is nothing left to launch.
    if (batchMax == 0)
      break;

    // Grow the batch while idle work stays within the useful-work budget.
    int64_t usefulElements = batchMax;
    int64_t idleElements = 0;
    unsigned batchEnd = i + 1;
    while (batchEnd < nInputs &&
           (batchEnd - i) < static_cast<unsigned>(batch_size)) {
      const int64_t n = nElements[order[batchEnd]];
      if (n == 0)
        break;
      // Limit batching when the padding overhead becomes too large. Smaller
      // tensors leave idle elements up to batchMax; stop once idle work exceeds
      // useful work.
      if (idleElements + (batchMax - n) > usefulElements + n)
        break;
      idleElements += batchMax - n;
      usefulElements += n;
      ++batchEnd;
    }
    const unsigned batchStart = i;
    // Advance before dispatch since LAUNCH_VECTORIZED may continue the loop,
    // which would otherwise skip the cursor update.
    i = batchEnd;

    bool isContig = true;
    bool isAligned = true;
    // OR input addresses so a single alignment check covers the whole batch.
    uintptr_t inputAddressBits = 0;
    // Inputs are sorted by size, so batchMax is the batch maximum.
    const unsigned int max_elements_per_tensor =
        static_cast<unsigned int>(batchMax);
    for (batchCounter = 0; batchStart + batchCounter < batchEnd;
         ++batchCounter) {
      const unsigned inputIndex = order[batchStart + batchCounter];
      catMetaData.input[batchCounter] = static_cast<const scalar_t*>(
          inputs[inputIndex].get().const_data_ptr());
      catMetaData.offset[batchCounter] = inputOffset[inputIndex];
      catMetaData.dimSize[batchCounter] = dimSize[inputIndex];
      catMetaData.nElements[batchCounter] = nElements[inputIndex];

      // If at least one of the inputs is not aligned, we can't call the
      // CatArrayBatchedCopy_alignedK_contig
      isAligned &= is_aligned_vec4(catMetaData.input[batchCounter]);
      inputAddressBits |=
          reinterpret_cast<uintptr_t>(catMetaData.input[batchCounter]);

      if (stride_size > 1) {
        auto strides = inputs[inputIndex].get().strides();
        auto sizes = inputs[inputIndex].get().sizes();
        if (memory_format == c10::MemoryFormat::Contiguous) {
          for (int j = 0; j < nDims; j++) {
            catMetaData.tensorStride[batchCounter].tensorSize[j] = sizes[j];
            catMetaData.tensorStride[batchCounter].tensorStride[j] = strides[j];
          }
        } else {
          catMetaData.tensorStride[batchCounter].tensorSize[0] = sizes[0];
          catMetaData.tensorStride[batchCounter].tensorStride[0] = strides[0];
          for (int j = 1; j < nDims - 1; ++j) {
            catMetaData.tensorStride[batchCounter].tensorSize[j] = sizes[j + 1];
            catMetaData.tensorStride[batchCounter].tensorStride[j] =
                strides[j + 1];
          }
          catMetaData.tensorStride[batchCounter].tensorSize[nDims - 1] =
              sizes[1];
          catMetaData.tensorStride[batchCounter].tensorStride[nDims - 1] =
              strides[1];
        }
        catMetaData.isContiguous[batchCounter] = false;
        isContig = false;
      } else {
        catMetaData.isContiguous[batchCounter] = true;
      }
    }

// Independent of Dims, so dispatched ahead of the HANDLE_CASE switch.
#define LAUNCH_VECTORIZED(VEC_BYTES)                                       \
  {                                                                        \
    sycl::range<2> vecGroup, vecRange;                                     \
    std::tie(vecRange, vecGroup) = getCatRangeContig<scalar_t, VEC_BYTES>( \
        max_elements_per_tensor, batchCounter);                            \
    CatArrayBatchedCopy_vectorized<                                        \
        scalar_t,                                                          \
        unsigned int,                                                      \
        batch_size,                                                        \
        stride_size,                                                       \
        VEC_BYTES>                                                         \
        kfn(data,                                                          \
            catMetaData,                                                   \
            outputParam.tensorSize[mapped_dimension],                      \
            outputParam.tensorStride[mapped_dimension]);                   \
    auto& q = getCurrentSYCLQueue();                                       \
    sycl_kernel_submit(vecRange, vecGroup, q, kfn);                        \
    continue;                                                              \
  }

    if (isContig && vectorBytes > 0 &&
        !(inputAddressBits % static_cast<uintptr_t>(vectorBytes))) {
      if constexpr (ALIGNED_VEC_LOAD_BYTES_16 % sizeof(scalar_t) == 0) {
        if (vectorBytes == ALIGNED_VEC_LOAD_BYTES_16)
          LAUNCH_VECTORIZED(ALIGNED_VEC_LOAD_BYTES_16);
      }
      if constexpr (ALIGNED_VEC_LOAD_BYTES_8 % sizeof(scalar_t) == 0) {
        if (vectorBytes == ALIGNED_VEC_LOAD_BYTES_8)
          LAUNCH_VECTORIZED(ALIGNED_VEC_LOAD_BYTES_8);
      }
    }
#undef LAUNCH_VECTORIZED

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

// Template Declarations for dim = 1, 2, 3, 4
#define HANDLE_CASE(DIMS)                                                   \
  if (isContig && isAligned && isOutputAligned && sizeof(scalar_t) > 2 &&   \
      sizeof(scalar_t) <= 8) {                                              \
    auto& q = getCurrentSYCLQueue();                                        \
    sycl_kernel_submit<cat_array_batched_copy_aligned_k_contig_kernel_impl< \
        scalar_t,                                                           \
        unsigned int,                                                       \
        DIMS,                                                               \
        batch_size,                                                         \
        stride_size,                                                        \
        ALIGNED_VEC_LOAD_BYTES_16>>(                                        \
        catRange,                                                           \
        applyGroup,                                                         \
        q,                                                                  \
        0,                                                                  \
        data,                                                               \
        catMetaData,                                                        \
        outputParam,                                                        \
        mapped_dimension,                                                   \
        outputParam.tensorStride[mapped_dimension]);                        \
  } else if (                                                               \
      isContig && isAligned && isOutputAligned && sizeof(scalar_t) == 2) {  \
    auto& q = getCurrentSYCLQueue();                                        \
    sycl_kernel_submit<cat_array_batched_copy_aligned_k_contig_kernel_impl< \
        scalar_t,                                                           \
        unsigned int,                                                       \
        DIMS,                                                               \
        batch_size,                                                         \
        stride_size,                                                        \
        ALIGNED_VEC_LOAD_BYTES_8>>(                                         \
        catRange,                                                           \
        applyGroup,                                                         \
        q,                                                                  \
        0,                                                                  \
        data,                                                               \
        catMetaData,                                                        \
        outputParam,                                                        \
        mapped_dimension,                                                   \
        outputParam.tensorStride[mapped_dimension]);                        \
  } else if (isContig) {                                                    \
    auto& q = getCurrentSYCLQueue();                                        \
    sycl_kernel_submit<cat_array_batched_copy_contig_kernel_impl<           \
        scalar_t,                                                           \
        unsigned int,                                                       \
        DIMS,                                                               \
        batch_size,                                                         \
        stride_size>>(                                                      \
        catRange,                                                           \
        applyGroup,                                                         \
        q,                                                                  \
        0,                                                                  \
        data,                                                               \
        catMetaData,                                                        \
        outputParam,                                                        \
        mapped_dimension,                                                   \
        outputParam.tensorStride[mapped_dimension]);                        \
  } else {                                                                  \
    auto& q = getCurrentSYCLQueue();                                        \
    sycl_kernel_submit<cat_array_batched_copy_kernel_impl<                  \
        scalar_t,                                                           \
        unsigned int,                                                       \
        DIMS,                                                               \
        batch_size,                                                         \
        stride_size>>(                                                      \
        catRange,                                                           \
        applyGroup,                                                         \
        q,                                                                  \
        0,                                                                  \
        data,                                                               \
        catMetaData,                                                        \
        outputParam,                                                        \
        mapped_dimension,                                                   \
        outputParam.tensorStride[mapped_dimension]);                        \
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

namespace {

constexpr int kCatMaxDims = 6;
constexpr int kCatMaxInputs = 32;
// Largest cat-dim size of the SLM cat-coordinate -> input table; larger cat
// dims search the inputs' offsets.
constexpr int kCatMaxTable = 4096;
constexpr int kCatGroupSize = 256;
// Output vectors per work-item of the SLM kernel (amortizes its setup).
constexpr int kCatChunk = 2;
// Below this output size, the SLM setup outweighs the register metadata
// blend also for element gathers (measured with cold caches).
constexpr int64_t kCatSmallBytes = int64_t(12) << 20;

struct CatInputMeta {
  uint64_t ptr;
  int32_t begin; // first output coordinate of this input along the cat dim
  int32_t end; // one past its last one
  int32_t stride[kCatMaxDims];
  int32_t vec_ok; // its V-unit vectors are aligned contiguous loads
};

// Output dims in memory order (dim 0 innermost) after merging neighbours;
// dim-0 sizes and offsets and all strides are in units of U bytes.
template <int N>
struct CatParams {
  int ndim = 0;
  int cat_dim = 0;
  int ninputs = 0;
  uint32_t nvec = 0;
  at::detail::IntDivider<uint32_t> size[kCatMaxDims];
  CatInputMeta in[N];
};

// Splits output unit index e into outer coordinates c[1..ndim) and returns
// the dim-0 one. Static loop bounds keep c in registers.
template <int N>
inline int32_t cat_decompose(const CatParams<N>& p, uint32_t e, uint32_t* c) {
  const auto qr = p.size[0].divmod(e);
  uint32_t r = qr.div;
#pragma unroll
  for (int d = 1; d < kCatMaxDims; ++d) {
    if (d == p.ndim - 1) {
      c[d] = r;
    } else if (d < p.ndim) {
      const auto q = p.size[d].divmod(r);
      c[d] = q.mod;
      r = q.div;
    }
  }
  return qr.mod;
}

// Moves c to the next dim-0 row, whose first unit is e.
template <int N>
inline void cat_next_row(const CatParams<N>& p, uint32_t e, uint32_t* c) {
  if (p.ndim > 1 && c[1] + 1 < p.size[1].divisor) {
    ++c[1];
  } else {
    cat_decompose(p, e, c);
  }
}

// Output-driven cat: each work-item writes V consecutive output units of U
// bytes as one aligned store and gathers them from the input holding them,
// so sub-group stores cover whole cache lines whatever the inputs' widths
// (the input-driven kernels below write per-input stripes, which turn into
// partial-line writes for narrow slices, e.g. channels-last cat along C).
// kRegInputs > 0: up to kRegInputs inputs, whose metadata is selected from
// the kernel parameters by a branch-free blend (a dynamic index would move
// them to private memory). kRegInputs == 0: up to kCatMaxInputs inputs; the
// metadata and a cat-coordinate -> input table are staged in SLM.
// kWrap: a vector may run past the end of an output row.
template <int U, int V, bool kWrap, int kRegInputs>
struct CatGatherFunctor : public __SYCL_KER_CONFIG_CONVENTION__ {
  using T = OpaqueType<U>;
  using VecT = memory::aligned_vector<T, V>;
  static constexpr bool kReg = kRegInputs > 0;
  static constexpr int N = kReg ? kRegInputs : kCatMaxInputs;

  // Start of the outer coordinates' row in input m. Coordinates are taken
  // relative to the input (cat coordinate minus begin), so the offset stays
  // within the input's 32-bit indexable extent.
  const T* row_base(const CatInputMeta& m, const uint32_t* c) const {
    int32_t off = 0;
#pragma unroll
    for (int d = 1; d < kCatMaxDims; ++d) {
      if (d < p_.ndim) {
        const int32_t x = static_cast<int32_t>(c[d]) -
            (d == p_.cat_dim ? m.begin : 0);
        off += x * m.stride[d];
      }
    }
    return reinterpret_cast<const T*>(m.ptr) + off;
  }

  CatInputMeta meta(const CatInputMeta* slm, int k) const {
    if constexpr (kReg) {
      CatInputMeta m{};
#pragma unroll
      for (int j = 0; j < N; ++j) {
        const uint32_t mask = 0u - static_cast<uint32_t>(k == j);
        const CatInputMeta& s = p_.in[j];
        m.ptr |= s.ptr & (0ull - static_cast<uint64_t>(k == j));
        m.begin |= s.begin & mask;
        m.end |= s.end & mask;
#pragma unroll
        for (int d = 0; d < kCatMaxDims; ++d) {
          m.stride[d] |= s.stride[d] & mask;
        }
        m.vec_ok |= s.vec_ok & mask;
      }
      return m;
    } else {
      return slm[k];
    }
  }

  // Input holding cat coordinate x.
  int find(const CatInputMeta* slm, const uint8_t* table, int32_t x) const {
    int k = 0;
    if constexpr (kReg) {
#pragma unroll
      for (int j = 1; j < N; ++j) {
        k += (j < p_.ninputs && x >= p_.in[j].begin) ? 1 : 0;
      }
    } else if (table_size_ > 0) {
      k = table[x];
    } else {
      for (int j = 1; j < p_.ninputs; ++j) {
        k += x >= slm[j].begin ? 1 : 0;
      }
    }
    return k;
  }

  // slm and table are only read on the SLM path; the register path (kReg)
  // passes nullptr.
  VecT gather(const CatInputMeta* slm, const uint8_t* table, uint32_t vid)
      const {
    const int32_t n0 = p_.size[0].divisor;
    uint32_t c[kCatMaxDims];
    int32_t x = cat_decompose(p_, vid * V, c);
    VecT v;
    if (p_.cat_dim == 0) {
      // Several inputs per row: switch input when x leaves the current one.
      int k = find(slm, table, x);
      CatInputMeta m = meta(slm, k);
      const T* base = row_base(m, c);
      if (m.vec_ok && x + V <= m.end && ((x - m.begin) & (V - 1)) == 0) {
        return *reinterpret_cast<const VecT*>(base + (x - m.begin));
      }
#pragma unroll
      for (int i = 0; i < V; ++i) {
        if (kWrap && x == n0) {
          x = 0;
          cat_next_row(p_, vid * V + i, c);
          k = 0;
          m = meta(slm, k);
          base = row_base(m, c);
        } else if (x >= m.end) {
          m = meta(slm, ++k);
          base = row_base(m, c);
        }
        v[i] = base[(x - m.begin) * m.stride[0]];
        ++x;
      }
      return v;
    }
    // The cat dim is an outer dim: one input per row. Its coordinate is read
    // with static indices (a dynamic index would put c in private memory).
    auto lookup = [&](CatInputMeta& m) {
      uint32_t cc = 0;
#pragma unroll
      for (int d = 1; d < kCatMaxDims; ++d) {
        cc = d == p_.cat_dim ? c[d] : cc;
      }
      m = meta(slm, find(slm, table, cc));
      return row_base(m, c);
    };
    CatInputMeta m;
    const T* base = lookup(m);
    if (m.vec_ok) {
      return *reinterpret_cast<const VecT*>(base + x);
    }
#pragma unroll
    for (int i = 0; i < V; ++i) {
      if (kWrap && x == n0) {
        x = 0;
        cat_next_row(p_, vid * V + i, c);
        base = lookup(m);
      }
      v[i] = base[x * m.stride[0]];
      ++x;
    }
    return v;
  }

  void operator()(sycl::nd_item<1> item) const {
    VecT* out = reinterpret_cast<VecT*>(out_);
    if constexpr (kReg) {
      const uint32_t vid = item.get_global_linear_id();
      if (vid < p_.nvec) {
        out[vid] = gather(nullptr, nullptr, vid);
      }
    } else {
      const int lid = item.get_local_linear_id();
      const int lsize = item.get_local_range(0);
      CatInputMeta* meta =
          slm_meta_.template get_multi_ptr<sycl::access::decorated::no>().get();
      uint8_t* table =
          slm_table_.template get_multi_ptr<sycl::access::decorated::no>()
              .get();
#pragma unroll
      for (int k = 0; k < N; ++k) {
        if (lid == 0 && k < p_.ninputs) {
          meta[k] = p_.in[k];
        }
      }
      sycl::group_barrier(item.get_group());
      for (int k = 0; k < (table_size_ > 0 ? p_.ninputs : 0); ++k) {
        for (int32_t x = meta[k].begin + lid; x < meta[k].end; x += lsize) {
          table[x] = static_cast<uint8_t>(k);
        }
      }
      sycl::group_barrier(item.get_group());
      // Each work-group writes kCatChunk consecutive blocks of lsize vectors.
      const uint32_t span = static_cast<uint32_t>(lsize) * kCatChunk;
      const uint32_t first =
          static_cast<uint32_t>(item.get_group_linear_id()) * span;
      const uint32_t last = p_.nvec - first < span ? p_.nvec : first + span;
      for (uint32_t vid = first + lid; vid < last; vid += lsize) {
        out[vid] = gather(meta, table, vid);
      }
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    if constexpr (!kReg) {
      slm_meta_ = sycl::local_accessor<CatInputMeta, 1>(N, cgh);
      slm_table_ = sycl::local_accessor<uint8_t, 1>(
          std::max<int64_t>(table_size_, 1), cgh);
    }
  }

  CatGatherFunctor(void* out, const CatParams<N>& p, int64_t table_size)
      : out_(static_cast<T*>(out)), p_(p), table_size_(table_size) {}

 private:
  T* out_;
  CatParams<N> p_;
  int64_t table_size_;
  sycl::local_accessor<CatInputMeta, 1> slm_meta_;
  sycl::local_accessor<uint8_t, 1> slm_table_;
};

// Stack along the last dim: K inputs each give one unit per output row of K
// units (cat_dim 0, size[0] == K). A work-item writes R rows (16 bytes) with
// one R-unit load per input, interleaved in registers; the gather kernel
// would load unit by unit.
template <int U, int R, int K>
struct CatInterleaveFunctor {
  using T = OpaqueType<U>;

  void operator()(sycl::nd_item<1> item) const {
    const uint32_t vid = item.get_global_linear_id();
    if (vid >= p_.nvec) {
      return;
    }
    uint32_t c[kCatMaxDims];
    cat_decompose(p_, vid * (R * K), c);
    memory::aligned_vector<T, R * K> v;
#pragma unroll
    for (int k = 0; k < K; ++k) {
      const CatInputMeta& m = p_.in[k];
      int32_t off = 0;
#pragma unroll
      for (int d = 1; d < kCatMaxDims; ++d) {
        if (d < p_.ndim) {
          off += static_cast<int32_t>(c[d]) * m.stride[d];
        }
      }
      const T* src = reinterpret_cast<const T*>(m.ptr) + off;
      memory::aligned_vector<T, R> r;
      if (m.vec_ok) {
        r = *reinterpret_cast<const memory::aligned_vector<T, R>*>(src);
      } else {
#pragma unroll
        for (int j = 0; j < R; ++j) {
          r[j] = src[j * m.stride[1]];
        }
      }
#pragma unroll
      for (int j = 0; j < R; ++j) {
        v[j * K + k] = r[j];
      }
    }
    reinterpret_cast<decltype(v)*>(out_)[vid] = v;
  }

  CatInterleaveFunctor(void* out, const CatParams<K>& p)
      : out_(static_cast<T*>(out)), p_(p) {}

 private:
  T* out_;
  CatParams<K> p_;
};

template <int U, int R, int K>
void launch_cat_interleave(const Tensor& out, CatParams<K>& p) {
  p.nvec = static_cast<uint32_t>(out.numel() / (R * K));
  CatInterleaveFunctor<U, R, K> kfn(out.mutable_data_ptr(), p);
  sycl_kernel_submit(
      sycl::range<1>(ceil_div<int64_t>(p.nvec, kCatGroupSize) * kCatGroupSize),
      sycl::range<1>(kCatGroupSize),
      getCurrentSYCLQueue(),
      kfn);
}

template <int U, int V, bool kWrap, int kRegInputs, typename FillFn>
void launch_cat_gather(
    const Tensor& out,
    int64_t table_size,
    const FillFn& fill) {
  using Kernel = CatGatherFunctor<U, V, kWrap, kRegInputs>;
  // Zeroed: the register blend in meta() reads all N entries.
  CatParams<Kernel::N> p{};
  fill(p);
  const int64_t per_group = kCatGroupSize * (kRegInputs ? 1 : kCatChunk);
  const int64_t groups = ceil_div<int64_t>(p.nvec, per_group);
  Kernel kfn(out.mutable_data_ptr(), p, table_size);
  sycl_kernel_submit(
      sycl::range<1>(groups * kCatGroupSize),
      sycl::range<1>(kCatGroupSize),
      getCurrentSYCLQueue(),
      kfn);
}

// Register metadata for up to 2 or 4 inputs (kRegInputs), else SLM. Vectors
// may run across rows (wrap) only if they are 16 bytes of several units.
template <int U, int V, typename FillFn>
void launch_cat(
    const Tensor& out,
    bool wrap,
    int reg_inputs,
    int64_t table_size,
    const FillFn& fill) {
  if constexpr (V == 1 && U >= 8) {
    if (reg_inputs) {
      return reg_inputs == 2 ? launch_cat_gather<U, V, false, 2>(out, 0, fill)
                             : launch_cat_gather<U, V, false, 4>(out, 0, fill);
    }
  }
  // Rows wrap only for units below 8 bytes, and V > 1 with 8-byte units
  // means more than 4 inputs, which use SLM.
  if constexpr (V > 1 && U < 8 && U * V == 16) {
    if (wrap) {
      return reg_inputs
          ? launch_cat_gather<U, V, true, 4>(out, 0, fill)
          : launch_cat_gather<U, V, true, 0>(out, table_size, fill);
    }
  }
  if constexpr (V > 1 && U < 8) {
    if (reg_inputs) {
      return launch_cat_gather<U, V, false, 4>(out, 0, fill);
    }
  }
  launch_cat_gather<U, V, false, 0>(out, table_size, fill);
}

// Cat of same-dtype `inputs` along `dim` into the dense `out` with
// CatGatherFunctor. Returns false if unsupported.
bool cat_output_driven(
    const MaterializedITensorListRef& inputs,
    int64_t dim,
    const Tensor& out) {
  const int64_t ndim = out.dim();
  if (ndim == 0 || !out.is_non_overlapping_and_dense() ||
      !canUse32BitIndexMath(out)) {
    return false;
  }
  c10::SmallVector<const Tensor*, kCatMaxInputs> in;
  for (const Tensor& t : inputs) {
    if (t.numel() == 0) {
      continue;
    }
    if (in.size() == kCatMaxInputs || t.dim() != ndim ||
        !canUse32BitIndexMath(t)) {
      return false;
    }
    in.push_back(&t);
  }
  const int nin = in.size();
  if (nin == 0) {
    return false;
  }
  c10::SmallVector<int64_t, kCatMaxDims> order(ndim);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
    return out.stride(a) < out.stride(b);
  });

  // Walk dims innermost first; merge neighbours that are not the cat dim
  // and are contiguous with each other in every input.
  int64_t size[kCatMaxDims];
  int64_t stride[kCatMaxInputs][kCatMaxDims];
  int nd = 0, cat_dim = -1;
  int64_t prev = -1;
  for (const int64_t d : order) {
    const int64_t n = out.size(d);
    if (n == 1) {
      continue;
    }
    bool merge = nd > 0 && d != dim && prev != dim;
    for (int k = 0; k < nin && merge; ++k) {
      merge = in[k]->stride(d) == stride[k][nd - 1] * size[nd - 1];
    }
    prev = d;
    if (merge) {
      size[nd - 1] *= n;
      continue;
    }
    if (nd == kCatMaxDims) {
      return false;
    }
    cat_dim = d == dim ? nd : cat_dim;
    size[nd] = n;
    for (int k = 0; k < nin; ++k) {
      stride[k][nd] = in[k]->stride(d);
    }
    ++nd;
  }
  if (cat_dim < 0) {
    return false;
  }
  int64_t begin[kCatMaxInputs + 1] = {0};
  for (int k = 0; k < nin; ++k) {
    begin[k + 1] = begin[k] + in[k]->size(dim);
  }
  auto addr = [](const Tensor& t) {
    return static_cast<int64_t>(
        reinterpret_cast<uintptr_t>(t.const_data_ptr()));
  };

  // Unit: the largest power of two (<= 16 bytes) dividing every byte offset
  // the kernel forms, so it moves U-byte units instead of elements. Needs
  // contiguous inner dims.
  const int64_t es = out.element_size();
  const int64_t out_addr = addr(out);
  bool inner_contiguous = true;
  int64_t bits = 16 | out_addr | size[0] * es;
  for (int k = 0; k < nin; ++k) {
    inner_contiguous &= stride[k][0] == 1;
    bits |= addr(*in[k]) | (cat_dim == 0 ? begin[k] * es : 0);
    for (int d = 1; d < nd; ++d) {
      bits |= stride[k][d] * es;
    }
  }
  const int64_t u = inner_contiguous ? bits & -bits : es;
  const int64_t q = u / es;
  size[0] /= q;
  for (int k = 0; k <= nin; ++k) {
    begin[k] /= cat_dim == 0 ? q : 1;
  }
  for (int k = 0; k < nin; ++k) {
    for (int d = 1; d < nd; ++d) {
      stride[k][d] /= q;
    }
  }
  // Store width: 16 or 8 bytes that end on output row boundaries, else 16
  // bytes running across rows. Up to 4 inputs with units of at least 8 bytes
  // store one unit, so that every vector is a single aligned load.
  const int64_t total_bytes = out.numel() * es;
  int64_t vec_bytes = u;
  bool wrap = false;
  if (u < 16 && !(nin <= 4 && u >= 8)) {
    for (const int64_t b : {16, 8}) {
      if (size[0] * u % b == 0 && out_addr % b == 0) {
        vec_bytes = std::max(b, u);
        break;
      }
    }
    if (vec_bytes < 8 && total_bytes % 16 == 0 && out_addr % 16 == 0) {
      vec_bytes = 16;
      wrap = true;
    }
  }
  const int64_t v = vec_bytes / u;

  // Interleave: every input gives one unit per row (e.g. stack along the
  // last dim).
  const int64_t rows = 16 / (nin * u);
  if (cat_dim == 0 && size[0] == nin && nd >= 2 && (nin == 2 || nin == 4) &&
      rows * nin * u == 16 && rows >= 2 && size[1] % rows == 0 &&
      out_addr % 16 == 0) {
    auto launch = [&]<int U, int R, int K>() {
      CatParams<K> p{};
      p.ndim = nd;
      for (int d = 0; d < nd; ++d) {
        p.size[d] = at::detail::IntDivider<uint32_t>(size[d]);
      }
      for (int k = 0; k < K; ++k) {
        CatInputMeta& m = p.in[k];
        m = CatInputMeta{};
        m.ptr = static_cast<uint64_t>(addr(*in[k]));
        m.vec_ok = stride[k][1] == 1 && m.ptr % (R * U) == 0;
        for (int d = 0; d < nd; ++d) {
          m.stride[d] = static_cast<int32_t>(stride[k][d]);
          m.vec_ok &= d < 2 || stride[k][d] % R == 0;
        }
      }
      launch_cat_interleave<U, R, K>(out, p);
    };
    switch (u * 8 + nin) {
      case 1 * 8 + 2:
        launch.template operator()<1, 8, 2>();
        return true;
      case 1 * 8 + 4:
        launch.template operator()<1, 4, 4>();
        return true;
      case 2 * 8 + 2:
        launch.template operator()<2, 4, 2>();
        return true;
      case 2 * 8 + 4:
        launch.template operator()<2, 2, 4>();
        return true;
      case 4 * 8 + 2:
        launch.template operator()<4, 2, 2>();
        return true;
    }
  }
  // Up to 4 inputs: metadata in registers when every vector is one unit
  // (one lookup each) or the output is small. Else SLM, which needs the
  // cat-coordinate table.
  const bool use_reg =
      nin <= 4 && (v == 1 ? u >= 8 : total_bytes < kCatSmallBytes);
  const int reg_inputs = !use_reg ? 0 : (v == 1 && nin <= 2) ? 2 : 4;
  const int64_t table_size = size[cat_dim] <= kCatMaxTable ? size[cat_dim] : 0;

  auto fill = [&](auto& p) {
    p.ndim = nd;
    p.cat_dim = cat_dim;
    p.ninputs = nin;
    p.nvec = static_cast<uint32_t>(total_bytes / vec_bytes);
    for (int d = 0; d < nd; ++d) {
      p.size[d] = at::detail::IntDivider<uint32_t>(size[d]);
    }
    for (int k = 0; k < nin; ++k) {
      CatInputMeta& m = p.in[k];
      m = CatInputMeta{};
      m.ptr = static_cast<uint64_t>(addr(*in[k]));
      m.begin = static_cast<int32_t>(begin[k]);
      m.end = static_cast<int32_t>(begin[k + 1]);
      m.vec_ok = !wrap && stride[k][0] == 1 && addr(*in[k]) % vec_bytes == 0;
      for (int d = 0; d < nd; ++d) {
        m.stride[d] = static_cast<int32_t>(stride[k][d]);
        m.vec_ok &= d == 0 || stride[k][d] % v == 0;
      }
    }
  };

#define CAT_CASE(U, V)                                         \
  case U * 32 + V:                                             \
    launch_cat<U, V>(out, wrap, reg_inputs, table_size, fill); \
    return true;
  switch (u * 32 + v) {
    CAT_CASE(1, 1)
    CAT_CASE(1, 8)
    CAT_CASE(1, 16)
    CAT_CASE(2, 1)
    CAT_CASE(2, 4)
    CAT_CASE(2, 8)
    CAT_CASE(4, 1)
    CAT_CASE(4, 2)
    CAT_CASE(4, 4)
    CAT_CASE(8, 1)
    CAT_CASE(8, 2)
    CAT_CASE(16, 1)
  }
#undef CAT_CASE
  return false;
}

} // namespace

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

  if (all_same_dtype &&
      materialized[valid].get().scalar_type() == result.scalar_type() &&
      cat_output_driven(materialized, dim, result)) {
    return;
  }

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
      all_same_dtype &&
      (materialized[valid].get().scalar_type() == result.scalar_type())) {
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
          kBComplex32,
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
      (materialized[valid].get().scalar_type() == result.scalar_type()) &&
      (memory_format == c10::MemoryFormat::Contiguous ||
       memory_format == c10::MemoryFormat::ChannelsLast)) {
    if (isBitsType(result.scalar_type())) {
      AT_DISPATCH_BIT_TYPES(result.scalar_type(), "cat_xpu", [&]() {
        using dtype = OpaqueType<sizeof(scalar_t)>;
        parallel_cat<
            dtype,
            CAT_ARRAY_BATCH_SIZE_STRIDED,
            CAT_ARRAY_BATCH_SIZE_STRIDED>(
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
                CAT_ARRAY_BATCH_SIZE_STRIDED,
                CAT_ARRAY_BATCH_SIZE_STRIDED>(
                result, materialized, dim, nDims, memory_format);
          }),
          AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX),
          kComplexHalf,
          kBComplex32,
          kHalf,
          kBool,
          kBFloat16,
          AT_EXPAND(AT_FLOAT8_TYPES),
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
      nt.copy_(t, /*non_blocking=*/false);
      offset += dimSize;
    }
  }
}

} // namespace at::native::xpu
