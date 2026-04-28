/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

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

// ============================================================
// V1 constants and structs (for contiguous fast path)
// ============================================================
constexpr int CAT_ARRAY_BATCH_SIZE_V1 = 1024;
constexpr int CAT_ARRAY_MAX_INPUT_DIMS_V1 = 5;

template <typename T, typename IndexType>
struct CatArrInputTensor {
  T* input;
  IndexType offset;
  IndexType dimSize;
  IndexType nElements;
};

template <typename IndexType, unsigned int MaxDims>
struct OutputTensorSizeStride {
  IndexType outputSize[MaxDims];
  IndexType outputStride[MaxDims];
};

// V1 IndexToOffset
template <typename IndexType, int Dims>
struct CatArrIndexToOffset_V1 {
  static inline IndexType compute(
      const IndexType outputSize[Dims],
      const IndexType outputStride[Dims],
      const IndexType dimSize,
      const unsigned int concatDim,
      IndexType linearIndex) {
    IndexType offset = 0;
#pragma unroll
    for (int i = Dims - 1; i >= 1; --i) {
      IndexType curDimSize = i == concatDim ? dimSize : outputSize[i];
      IndexType nextDimIndex = linearIndex / curDimSize;
      IndexType curDimIndex = linearIndex - curDimSize * nextDimIndex;
      IndexType curDimOffset = curDimIndex * outputStride[i];
      offset += curDimOffset;
      linearIndex = nextDimIndex;
    }
    return offset + linearIndex * outputStride[0];
  }
};

// V1 kernel functor (scalar copy, pointer-based metadata)
template <
    typename Tout,
    typename underlying_out_t,
    typename Tin,
    typename underlying_in_t,
    typename IndexType,
    int Dims>
struct CatArrayBatchedCopyKernelFunctor {
  void operator()(sycl::nd_item<2> item) const {
    IndexType tid = item.get_global_id(1);
    IndexType in = item.get_group(0);

    IndexType nElements = inputs[in].nElements;

    if (tid >= nElements)
      return;

    Tin* data = inputs[in].input;
    IndexType offset = inputs[in].offset;
    IndexType dimSize = inputs[in].dimSize;
    IndexType dataOffset = offset * dimStride;

    IndexType stride = item.get_global_range(1);

    while (tid < nElements) {
      IndexType elementOffset =
          CatArrIndexToOffset_V1<IndexType, Dims>::compute(
              os.outputSize, os.outputStride, dimSize, concatDim, tid);
      output[dataOffset + elementOffset] = data[tid];
      tid += stride;
    }
  }

  CatArrayBatchedCopyKernelFunctor(
      Tout* output_,
      CatArrInputTensor<Tin, IndexType>* inputs_,
      OutputTensorSizeStride<IndexType, CAT_ARRAY_MAX_INPUT_DIMS_V1> os_,
      const int concatDim_,
      IndexType dimStride_)
      : output(output_),
        inputs(inputs_),
        os(os_),
        concatDim(concatDim_),
        dimStride(dimStride_) {}

 private:
  Tout* output;
  CatArrInputTensor<Tin, IndexType>* inputs;
  OutputTensorSizeStride<IndexType, CAT_ARRAY_MAX_INPUT_DIMS_V1> os;
  const int concatDim;
  IndexType dimStride;
};

// V1 kernel launch (with V1 WG config)
template <
    typename Tout,
    typename underlying_out_t,
    typename Tin,
    typename underlying_in_t,
    typename IndexType,
    int Dims>
void CatArrayBatchedCopy_V1(
    Tout* output,
    CatArrInputTensor<Tin, IndexType>* inputs,
    OutputTensorSizeStride<IndexType, CAT_ARRAY_MAX_INPUT_DIMS_V1> os,
    const int concatDim,
    IndexType dimStride,
    int batchCounter) {
  CatArrayBatchedCopyKernelFunctor<
      Tout,
      underlying_out_t,
      Tin,
      underlying_in_t,
      IndexType,
      Dims>
      kfn(output, inputs, os, concatDim, dimStride);

  int64_t numWI = syclMaxWorkGroupSize(kfn);
  int64_t numWG;
  if (batchCounter > 32)
    numWG = 64;
  else
    numWG = 128;
  sycl::range<2> global_range(batchCounter, numWG * numWI);
  sycl::range<2> local_range(1, numWI);
  auto& q = getCurrentSYCLQueue();
  sycl_kernel_submit(global_range, local_range, q, kfn);
}

// V1 parallel_cat (pinned memcpy metadata, batch_size=1024)
template <
    typename scalar_out_t,
    typename underlying_out_t,
    typename scalar_in_t,
    typename underlying_in_t>
void parallel_cat_v1(
    const Tensor& out,
    const MaterializedITensorListRef& inputs,
    int64_t dimension,
    int nDims) {
  scalar_out_t* data = static_cast<scalar_out_t*>(out.mutable_data_ptr());

  int64_t tensorMetadataSize =
      sizeof(CatArrInputTensor<scalar_in_t, unsigned int>) *
      CAT_ARRAY_BATCH_SIZE_V1;
  auto d_inputs_storage =
      at::empty({tensorMetadataSize}, out.options().dtype(at::kByte));
  auto d_inputs = static_cast<CatArrInputTensor<scalar_in_t, unsigned int>*>(
      d_inputs_storage.mutable_data_ptr());

  OutputTensorSizeStride<unsigned int, CAT_ARRAY_MAX_INPUT_DIMS_V1> param;

  for (int i = 0; i < nDims; ++i) {
    param.outputSize[i] = at::native::size(out, i);
    param.outputStride[i] = out.stride(i);
  }

  auto& q = getCurrentSYCLQueue();
  int batchCounter = 0;
  int64_t offset = 0;
  for (int i = 0; i < inputs.size(); i += CAT_ARRAY_BATCH_SIZE_V1) {
    {
      CatArrInputTensor<scalar_in_t, unsigned int>* stackInputs;

      auto stackInputs_dptr =
          at::getHostAllocator(at::kXPU)->allocate(tensorMetadataSize);
      stackInputs =
          (CatArrInputTensor<scalar_in_t, unsigned int>*)stackInputs_dptr.get();

      for (batchCounter = 0; batchCounter < CAT_ARRAY_BATCH_SIZE_V1 &&
           (i + batchCounter) < inputs.size();
           ++batchCounter) {
        int64_t dimSize =
            at::native::size(inputs[i + batchCounter].get(), dimension);

        stackInputs[batchCounter].input =
            (scalar_in_t*)(inputs[i + batchCounter].get().const_data_ptr());
        stackInputs[batchCounter].offset = offset;
        stackInputs[batchCounter].dimSize = dimSize;
        stackInputs[batchCounter].nElements =
            inputs[i + batchCounter].get().numel();

        offset += dimSize;
      }

      q.memcpy((void*)d_inputs, (void*)stackInputs, tensorMetadataSize);
      at::getHostAllocator(at::kXPU)->record_event(
          (void*)stackInputs,
          stackInputs_dptr.get_context(),
          at::xpu::getCurrentXPUStream());
    }

#define HANDLE_CASE_V1(DIMS)         \
  CatArrayBatchedCopy_V1<            \
      scalar_out_t,                  \
      underlying_out_t,              \
      scalar_in_t,                   \
      underlying_in_t,               \
      unsigned int,                  \
      DIMS>(                         \
      data,                          \
      d_inputs,                      \
      param,                         \
      dimension,                     \
      param.outputStride[dimension], \
      batchCounter);
    switch (nDims) {
      case 1:
        HANDLE_CASE_V1(1);
        break;
      case 2:
        HANDLE_CASE_V1(2);
        break;
      case 3:
        HANDLE_CASE_V1(3);
        break;
      case 4:
        HANDLE_CASE_V1(4);
        break;
      case 5:
        HANDLE_CASE_V1(5);
        break;
      default:
        break;
    }
#undef HANDLE_CASE_V1
  }
}

// ============================================================
// V2 constants and structs (for non-contiguous / strided path)
// ============================================================
constexpr int CAT_ARRAY_BATCH_SIZE_V2 = 64;
constexpr int CAT_ARRAY_BATCH_SIZE_STRIDED = 16;
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
  constexpr unsigned int items_per_group = 256;
  constexpr unsigned int elements_per_item = 8;
  constexpr unsigned int max_group_per_eu = 32;

  unsigned int max_items = ceil_div(max_elements_per_tensor, elements_per_item);
  unsigned int item_groups = ceil_div(max_items, items_per_group);

  const unsigned int num_eu = syclGpuEUCountPerSubslice();
  item_groups = std::min(num_eu * max_group_per_eu, item_groups);

  sycl::range<2> global_range(
      (long long)nTensors, items_per_group * item_groups);
  sycl::range<2> local_range(1, items_per_group);
  return std::make_tuple(global_range, local_range);
}

// V2 IndexToOffset
template <typename IndexType, int Dims>
struct CatArrIndexToOffset {
  static inline IndexType compute(
      const IndexType tensorSize[Dims],
      const IndexType tensorStride[Dims],
      const IndexType dimSize,
      const unsigned int concatDim,
      IndexType linearIndex) {
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

// V2 kernel: scalar copy with stride support
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

// V2 parallel_cat for non-contiguous / strided path
template <typename scalar_t, int batch_size, int stride_size>
void parallel_cat_v2(
    const Tensor& out,
    const MaterializedITensorListRef& inputs,
    int64_t dimension,
    int nDims,
    c10::MemoryFormat memory_format) {
  scalar_t* data = (scalar_t*)(out.mutable_data_ptr());
  CatArrInputTensorMetadata<scalar_t, unsigned int, batch_size, stride_size>
      catMetaData;
  TensorSizeStride<unsigned int, CAT_ARRAY_MAX_INPUT_DIMS> outputParam;

  if (memory_format == c10::MemoryFormat::Contiguous) {
    for (int i = 0; i < nDims; ++i) {
      outputParam.tensorSize[i] = out.size(i);
      outputParam.tensorStride[i] = out.stride(i);
    }
  } else if (
      memory_format == c10::MemoryFormat::ChannelsLast ||
      memory_format == c10::MemoryFormat::ChannelsLast3d) {
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

  int batchCounter = 0;
  int64_t offset = 0;
  for (unsigned i = 0; i < inputs.size(); i += batch_size) {
    unsigned int max_elements_per_tensor = 0;
    for (batchCounter = 0;
         batchCounter < batch_size && (i + batchCounter) < inputs.size();
         ++batchCounter) {
      int64_t dimSize = 0;
      if (inputs[i + batchCounter].get().numel() > 0) {
        dimSize = inputs[i + batchCounter].get().size(logical_dimension);
      }
      catMetaData.input[batchCounter] = static_cast<const scalar_t*>(
          inputs[i + batchCounter].get().const_data_ptr());
      catMetaData.offset[batchCounter] = offset;
      catMetaData.dimSize[batchCounter] = dimSize;
      catMetaData.nElements[batchCounter] =
          inputs[i + batchCounter].get().numel();

      if (stride_size > 1) {
        auto strides = inputs[i + batchCounter].get().strides();
        auto sizes = inputs[i + batchCounter].get().sizes();
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
      } else {
        catMetaData.isContiguous[batchCounter] = true;
      }

      offset += dimSize;

      max_elements_per_tensor = std::max(
          max_elements_per_tensor, catMetaData.nElements[batchCounter]);
    }

    if (max_elements_per_tensor == 0)
      continue;

    sycl::range<2> applyGroup, catRange;
    std::tie(catRange, applyGroup) =
        getCatRange(max_elements_per_tensor, batchCounter);

#define HANDLE_CASE_V2(DIMS)                                                   \
  {                                                                            \
    CatArrayBatchedCopy<scalar_t, unsigned int, DIMS, batch_size, stride_size> \
        kfn(data,                                                              \
            catMetaData,                                                       \
            outputParam,                                                       \
            mapped_dimension,                                                  \
            outputParam.tensorStride[mapped_dimension]);                       \
    auto& q = getCurrentSYCLQueue();                                           \
    sycl_kernel_submit(catRange, applyGroup, q, kfn);                          \
  }

    switch (nDims) {
      case 1:
        HANDLE_CASE_V2(1);
        break;
      case 2:
        HANDLE_CASE_V2(2);
        break;
      case 3:
        HANDLE_CASE_V2(3);
        break;
      case 4:
        HANDLE_CASE_V2(4);
        break;
    }

#undef HANDLE_CASE_V2
  }
}

// The kernels are templated on an opaque, self-aligned type of the correct
// size to avoid redundant kernels for different types of the same size.
template <unsigned N>
struct alignas(N) OpaqueType {
  char data[N];
};

// ============================================================
// Entry point: cat_out_kernel
// Contiguous path → V1 (pinned memcpy + scalar kernel + V1 WG)
// Non-contiguous/strided path → V2 (kernel arg metadata + stride support)
// ============================================================
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

  const bool all32BitIndexable = std::all_of(
      materialized.begin(), materialized.end(), [](const Tensor& t) {
        return canUse32BitIndexMath(t);
      });

  int nDims = materialized[valid].get().dim();

  // Path 1: all contiguous → use V1 implementation (fast on all GPUs)
  if (materialized.size() > 1 && result.dim() <= CAT_ARRAY_MAX_INPUT_DIMS_V1 &&
      canUse32BitIndexMath(result) && all_contiguous && all32BitIndexable &&
      all_same_dtype && memory_format == c10::MemoryFormat::Contiguous &&
      (materialized[valid].get().scalar_type() == result.scalar_type())) {
    if (isBitsType(result.scalar_type())) {
      AT_DISPATCH_BIT_TYPES(result.scalar_type(), "cat_xpu", [&]() {
        using dtype = OpaqueType<sizeof(scalar_t)>;
        parallel_cat_v1<dtype, dtype, dtype, dtype>(
            result, materialized, dim, nDims);
      });
    } else {
      AT_DISPATCH_V2(
          result.scalar_type(),
          "cat_xpu",
          AT_WRAP([&]() {
            parallel_cat_v1<scalar_t, scalar_t, scalar_t, scalar_t>(
                result, materialized, dim, nDims);
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
  }
  // Path 2a: ChannelsLast + all_contiguous → V2 with stride_size=1 (same as V2
  // original Path 1)
  else if (
      materialized.size() > 1 && result.dim() <= CAT_ARRAY_MAX_INPUT_DIMS &&
      canUse32BitIndexMath(result) && all_contiguous && all32BitIndexable &&
      all_same_dtype &&
      (materialized[valid].get().scalar_type() == result.scalar_type()) &&
      (memory_format == c10::MemoryFormat::ChannelsLast ||
       memory_format == c10::MemoryFormat::ChannelsLast3d)) {
    if (isBitsType(result.scalar_type())) {
      AT_DISPATCH_BIT_TYPES(result.scalar_type(), "cat_xpu", [&]() {
        using dtype = OpaqueType<sizeof(scalar_t)>;
        parallel_cat_v2<dtype, CAT_ARRAY_BATCH_SIZE_V2, 1>(
            result, materialized, dim, nDims, memory_format);
      });
    } else {
      AT_DISPATCH_V2(
          result.scalar_type(),
          "cat_xpu",
          AT_WRAP([&]() {
            using dtype = OpaqueType<sizeof(scalar_t)>;
            parallel_cat_v2<dtype, CAT_ARRAY_BATCH_SIZE_V2, 1>(
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
  }
  // Path 2b: non-contiguous strided → V2 with stride_size=batch_size
  else if (
      materialized.size() > 1 && result.dim() <= CAT_ARRAY_MAX_INPUT_DIMS &&
      canUse32BitIndexMath(result) && nDims <= CAT_ARRAY_MAX_INPUT_DIMS &&
      all32BitIndexable && all_same_dtype &&
      (materialized[valid].get().scalar_type() == result.scalar_type()) &&
      (memory_format == c10::MemoryFormat::Contiguous ||
       memory_format == c10::MemoryFormat::ChannelsLast ||
       memory_format == c10::MemoryFormat::ChannelsLast3d)) {
    if (isBitsType(result.scalar_type())) {
      AT_DISPATCH_BIT_TYPES(result.scalar_type(), "cat_xpu", [&]() {
        using dtype = OpaqueType<sizeof(scalar_t)>;
        parallel_cat_v2<
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
            parallel_cat_v2<
                dtype,
                CAT_ARRAY_BATCH_SIZE_STRIDED,
                CAT_ARRAY_BATCH_SIZE_STRIDED>(
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
  }
  // Path 3: fallback narrow+copy
  else {
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
