#include <ATen/Dispatch_v2.h>
#include <ATen/NumericUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/native/CanUse32BitIndexMath.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/TensorShape.h>
#include <ATen/xpu/CachingHostAllocator.h>
#include <comm/SYCLContext.h>
#include <comm/xpu_aten.h>

#include <ATen/ops/narrow.h>
#include <ATen/ops/size_native.h>

#include <ATen/native/xpu/sycl/ShapeKernels.h>

namespace at::native::xpu {

// The best performance is achieved for parallel computing with 1024 batch sizes
// at a time.
constexpr int CAT_ARRAY_BATCH_SIZE = 1024;

// Maximum parallel dimension to supporte
constexpr int CAT_ARRAY_MAX_INPUT_DIMS = 5;

// Similar to any other IndexToOffset calculation for copying along a given
// dimension.
template <typename IndexType, int Dims>
struct CatArrIndexToOffset {
  static inline IndexType compute(
      const IndexType outputSize[Dims],
      const IndexType outputStride[Dims],
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
      IndexType elementOffset = CatArrIndexToOffset<IndexType, Dims>::compute(
          os.outputSize, os.outputStride, dimSize, concatDim, tid);
      output[dataOffset + elementOffset] = data[tid];
      tid += stride;
    }
  }

  CatArrayBatchedCopyKernelFunctor(
      Tout* output_,
      CatArrInputTensor<Tin, IndexType>* inputs_,
      OutputTensorSizeStride<IndexType, CAT_ARRAY_MAX_INPUT_DIMS> os_,
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
  OutputTensorSizeStride<IndexType, CAT_ARRAY_MAX_INPUT_DIMS> os;
  const int concatDim;
  IndexType dimStride;
};

/**
 * Kernel used to concatenated grimDim.y tensors into an output tensor. Uses a
 * grid-stride loop based off of the blockIdx.x, threadIdx.x for each input to
 * copy each element from each input tensor into the output.
 *
 * output: base pointer to the storage associated with the output tensor
 * inputs: GPU-allocated array of input metadata for each input to concatenate
 *         in the kernel
 * os: the size/stride vectors for the output tensor
 * concatDim: dimension along which we are concatenating
 * dimStride: the stride of the output tensor at the concatDim
 *
 * The most important assumption made is that the input tensors are contiguous.
 */
template <
    typename Tout,
    typename underlying_out_t,
    typename Tin,
    typename underlying_in_t,
    typename IndexType,
    int Dims>
void CatArrayBatchedCopy(
    Tout* output,
    CatArrInputTensor<Tin, IndexType>* inputs,
    OutputTensorSizeStride<IndexType, CAT_ARRAY_MAX_INPUT_DIMS> os,
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

  // Get grid where x dim fills half gpu and y dim is number of tensors.
  // This will have cating two tensors fill the entire grid, but prevent
  // many threads from needlessly load meta data if their sizes is small.
  int64_t numWI = syclMaxWorkGroupSize(kfn);

  // We set limited numWG to prevent over schedule.
  // numWG = 512 EUs * 8 threads * SIMD lanes 32 / max_compute_units
  // (1024 on PVC).
  // When input tensors less than 32, we choose 128 numWG to handle a tensor,
  // then we have one tile per tensor.
  // When input tensors more than 32, we choose 64 numWG to handle a tensor,
  // half tile per tensor, the other half is occupied by next input tensor.
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

template <
    typename scalar_out_t,
    typename underlying_out_t,
    typename scalar_in_t,
    typename underlying_in_t>
void parallel_cat(
    const Tensor& out,
    const MaterializedITensorListRef& inputs,
    int64_t dimension,
    int nDims) {
  // First, let's set up our kernel parameters. We start with a raw pointer to
  // the storage for the output Tensor.
  scalar_out_t* data = static_cast<scalar_out_t*>(out.mutable_data_ptr());

  // Kernel Parameter
  int64_t tensorMetadataSize =
      sizeof(CatArrInputTensor<scalar_in_t, unsigned int>) *
      CAT_ARRAY_BATCH_SIZE;
  auto d_inputs_storage =
      at::empty({tensorMetadataSize}, out.options().dtype(at::kByte));
  auto d_inputs = static_cast<CatArrInputTensor<scalar_in_t, unsigned int>*>(
      d_inputs_storage.mutable_data_ptr());

  OutputTensorSizeStride<unsigned int, CAT_ARRAY_MAX_INPUT_DIMS> param;

  for (int i = 0; i < nDims; ++i) {
    param.outputSize[i] = at::native::size(out, i);
    param.outputStride[i] = out.stride(i);
  }

  // Now we loop
  auto& q = getCurrentSYCLQueue();
  int batchCounter = 0;
  int64_t offset = 0;
  for (int i = 0; i < inputs.size(); i += CAT_ARRAY_BATCH_SIZE) {
    // Re-allocate stackInputs every iteration to avoid read-after-write hazard
    {
      CatArrInputTensor<scalar_in_t, unsigned int>* stackInputs;

      auto stackInputs_dptr = at::xpu::HostAlloc(tensorMetadataSize);
      stackInputs =
          (CatArrInputTensor<scalar_in_t, unsigned int>*)stackInputs_dptr.get();

      for (batchCounter = 0; batchCounter < CAT_ARRAY_BATCH_SIZE &&
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

        // update offset
        offset += dimSize;
      }

      q.memcpy((void*)d_inputs, (void*)stackInputs, tensorMetadataSize);
      at::xpu::CachingHostAllocator_recordEvent(
          (void*)stackInputs,
          stackInputs_dptr.get_context(),
          at::xpu::getCurrentXPUStream());
    }

#define HANDLE_CASE(DIMS)            \
  CatArrayBatchedCopy<               \
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
      case 5:
        HANDLE_CASE(5);
        break;
      default:
        break;
    }
#undef HANDLE_CASE
  }
}

void check_shape_except_dim(Tensor& first, Tensor& second, int dimension) {
  int first_dims = first.dim();
  int second_dims = second.dim();
  TORCH_CHECK(
      first_dims == second_dims, "Tensors must have same number of dimensions");
  for (int dim = 0; dim < first_dims; dim++) {
    if (dim == dimension) {
      continue;
    }
    int64_t first_dim_size = first.size(dim);
    int64_t second_dim_size = second.size(dim);
    TORCH_CHECK(
        first_dim_size == second_dim_size,
        "Sizes of tensors must match except in dimension");
  }
}

// TODO: Evaluate latest PyTorch CUDA implementation for performance
void cat_out_kernel(
    const ITensorListRef& container,
    int64_t dimension,
    int64_t valid,
    bool all_contiguous,
    bool all_same_dtype,
    bool all_same_sizes_and_stride,
    MemoryFormat memory_format,
    const Tensor& result) {
  if (result.numel() == 0) {
    return;
  }

  MaterializedITensorListRef inputs = container.materialize();
  int numInputs = inputs.size();

  int i, j;
  int64_t offset;
  bool hasSkippedInput = false;
  Tensor notSkippedTensor; // non-owning reference
  // empty tensor includes size[0], size[0, 0, ..., 0] (n-dim).
  // here we only skip size[0], other empty sizes are not skipped.
  auto should_skip = [](const Tensor& t) {
    return t.numel() == 0 && t.dim() == 1;
  };
  int nDims = 0;

  for (i = 0; i < numInputs; i++) {
    if (should_skip(inputs[i])) {
      hasSkippedInput = true;
      continue;
    }
    nDims = inputs[i].get().dim();
    notSkippedTensor = inputs[i];
  }

  // If all inputs are empty tensors, return an empty tensor
  if (!notSkippedTensor.defined()) {
    return;
  }

  TORCH_CHECK(numInputs > 0, "invalid number of inputs");
  TORCH_CHECK(dimension >= 0, "invalid dimension");

  Tensor first_tensor = inputs[0];

  std::vector<int64_t> size(nDims);

  int64_t cat_dim_size = 0;
  for (int i = 0; i < numInputs; i++) {
    Tensor tensor = inputs[i];
    if (should_skip(tensor)) {
      continue;
    }
    check_shape_except_dim(notSkippedTensor, tensor, dimension);
    cat_dim_size += tensor.size(dimension);
  }

  for (int dim = 0; dim < nDims; dim++) {
    int64_t result_dim_size = notSkippedTensor.size(dim);
    if (dim == dimension) {
      result_dim_size = cat_dim_size;
    }
    size[dim] = result_dim_size;
  }

  const bool all32BitIndexable =
      std::all_of(inputs.begin(), inputs.end(), [](const Tensor& t) {
        return canUse32BitIndexMath(t);
      });
  const bool allContiguous =
      std::all_of(inputs.begin(), inputs.end(), [](const Tensor& t) {
        return !t.defined() || t.is_contiguous();
      });

  if (inputs.size() > 1 && !hasSkippedInput &&
      result.dim() <= CAT_ARRAY_MAX_INPUT_DIMS &&
      canUse32BitIndexMath(result) && allContiguous && all32BitIndexable &&
      all_same_dtype &&
      (inputs[0].get().scalar_type() == result.scalar_type())) {
    AT_DISPATCH_V2(
        result.scalar_type(),
        "cat_xpu",
        AT_WRAP([&]() {
          parallel_cat<scalar_t, scalar_t, scalar_t, scalar_t>(
              result, inputs, dimension, nDims);
        }),
        AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX),
        kComplexHalf,
        kHalf,
        kBool,
        kBFloat16,
        AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
  } else {
    offset = 0;
    for (j = 0; j < numInputs; j++) {
      if (should_skip(inputs[j]))
        continue;
      int64_t dimSize = inputs[j].get().size(dimension);
      Tensor nt = at::narrow(result, dimension, offset, dimSize);
      nt.copy_(inputs[j], false);
      offset += dimSize;
    }
  }
}

} // namespace at::native::xpu
