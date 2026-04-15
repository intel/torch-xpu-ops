/*
 * Faithful SYCL translation of CUDA sbtopk (single-block topk).
 *
 * CUDA sources translated 1:1:
 *   - SortingRadixSelect.cuh: countRadixUsingMask (line 176), findPattern (239)
 *   - ScanUtils.cuh: inclusiveBinaryPrefixScan (16), exclusiveBinaryPrefixScan (64)
 *   - TensorTopK.cu: gatherTopK (lines 40-182), radixSelect (860)
 *
 * Key CUDA -> SYCL mappings:
 *   WARP_BALLOT + __popc  -> sub-group reduce_over_group
 *   getLaneId()           -> sg.get_local_linear_id()
 *   atomicAdd (smem)      -> sycl::atomic_ref (local_space)
 *   __syncthreads()       -> sycl::group_barrier(item.get_group())
 *   getLaneMaskLe()+ballot -> sycl::inclusive_scan_over_group
 *   smem[]                -> int* from local accessor
 *
 * withinSliceStride = 1 (input is .contiguous() before calling sbtopk).
 * RADIX_BITS, RADIX_SIZE, RADIX_MASK are defined in SortingRadixSelect.h.
 */

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/xpu/sycl/SortingRadixSelect.h>
#include <ATen/native/xpu/sycl/TensorTopKSbtopkKernel.h>

namespace at {
namespace native {
namespace xpu {

// Block size = 1024 threads, matching CUDA C10_LAUNCH_BOUNDS_1(1024)
constexpr int SBTOPK_BLOCK = 1024;

// SLM layout:
//   [0..63]  : used by countRadixUsingMask (smem[0..RADIX_SIZE-1] for counts)
//              and by inclusiveBinaryPrefixScan (smem[0..num_sgs-1] for carries)
//   [64..65] : used by findPattern (flag + found index)
//   Total: 68 ints = 272 bytes
constexpr int SMEM_INTS = 68;

template <typename scalar_t>
struct SbtopkGatherFunctor : public __SYCL_KER_CONFIG_CONVENTION__ {
  using RadixT = typename TopKTypeConfig<scalar_t>::RadixType;
  static constexpr int NUM_BITS = sizeof(RadixT) * 8;

  // ================================================================
  // countRadixUsingMask (SortingRadixSelect.cuh:176)
  //
  // Counts distribution of radix digits at digitPos for elements
  // matching (convert(v) & desiredMask) == desired.
  // Result: all threads have identical counts[0..RADIX_SIZE-1].
  // ================================================================
  void countRadixUsingMask(
      sycl::nd_item<1> item,
      sycl::sub_group sg,
      int* smem,
      int counts[RADIX_SIZE],
      RadixT desired,
      RadixT desiredMask,
      int digitPos,
      const scalar_t* data,
      int sliceSize) const {
    int lid = item.get_local_id(0);
    int block_size = item.get_local_range(0);
    int sg_lid = sg.get_local_linear_id();

    // CUDA: counts[i] = 0
#pragma unroll
    for (int i = 0; i < RADIX_SIZE; ++i) {
      counts[i] = 0;
    }
    // CUDA: if (threadIdx.x < RadixSize) smem[threadIdx.x] = 0
    if (lid < RADIX_SIZE) {
      smem[lid] = 0;
    }
    sycl::group_barrier(item.get_group());

    // CUDA: per-warp counting via WARP_BALLOT + __popc, accumulated
    // SYCL: per-thread counting + sub_group reduce (semantically identical)
    for (int i = lid; i < sliceSize; i += block_size) {
      RadixT val = TopKTypeConfig<scalar_t>::convert(data[i]);
      bool hasVal = ((val & desiredMask) == desired);
      RadixT digit = (val >> digitPos) & RADIX_MASK;
#pragma unroll
      for (int j = 0; j < RADIX_SIZE; ++j) {
        counts[j] += (hasVal && (digit == j)) ? 1 : 0;
      }
    }

    // CUDA: accumulated warp-level counts -> lane 0 atomicAdd to smem
    // SYCL: sub_group reduce -> lane 0 atomicAdd to smem
#pragma unroll
    for (int j = 0; j < RADIX_SIZE; ++j) {
      counts[j] = sycl::reduce_over_group(sg, counts[j], sycl::plus<int>());
    }

    if (sg_lid == 0) {
#pragma unroll
      for (int j = 0; j < RADIX_SIZE; ++j) {
        sycl::atomic_ref<int, sycl::memory_order::relaxed,
                         sycl::memory_scope::work_group,
                         sycl::access::address_space::local_space>
            ref(smem[j]);
        ref.fetch_add(counts[j]);
      }
    }
    sycl::group_barrier(item.get_group());

    // CUDA: all threads read smem[0..RadixSize-1] = block-level totals
#pragma unroll
    for (int j = 0; j < RADIX_SIZE; ++j) {
      counts[j] = smem[j];
    }
    sycl::group_barrier(item.get_group());
  }

  // ================================================================
  // findPattern (SortingRadixSelect.cuh:239)
  //
  // Finds the unique scalar_t value whose convert() matches desired.
  // CUDA uses smem cast to scalar_t* for flag+value.
  // SYCL uses smem[64]=flag(int), smem[65]=index(int), then data[index].
  // ================================================================
  scalar_t findPattern(
      sycl::nd_item<1> item,
      int* smem,
      const scalar_t* data,
      int sliceSize,
      RadixT desired,
      RadixT desiredMask) const {
    int lid = item.get_local_id(0);
    int block_size = item.get_local_range(0);

    if (lid == 0) {
      smem[64] = 0;  // found flag
      smem[65] = -1; // found index
    }
    sycl::group_barrier(item.get_group());

    // CUDA: numIterations = round_up(sliceSize, blockDim.x)
    int numIterations =
        ((sliceSize + block_size - 1) / block_size) * block_size;

    for (int i = lid; i < numIterations; i += block_size) {
      bool inRange = (i < sliceSize);
      scalar_t v = inRange ? data[i] : static_cast<scalar_t>(0);

      if (inRange &&
          ((TopKTypeConfig<scalar_t>::convert(v) & desiredMask) == desired)) {
        smem[64] = 1; // flag
        smem[65] = i; // index of found value
      }
      sycl::group_barrier(item.get_group());

      int found = smem[64];
      int foundIdx = smem[65];
      sycl::group_barrier(item.get_group());

      if (found != 0) {
        return data[foundIdx];
      }
    }
    // Should not reach here
    return static_cast<scalar_t>(0);
  }

  // ================================================================
  // inclusiveBinaryPrefixScan (ScanUtils.cuh:16)
  //
  // CUDA:
  //   vote = WARP_BALLOT(in)
  //   index = __popc(getLaneMaskLe() & vote)   // inclusive within warp
  //   carry = __popc(vote)                      // warp total
  //   lane 0: smem[warp] = carry
  //   __syncthreads()
  //   thread 0: for i in 0..num_warps-1:
  //     v = smem[i]; smem[i] += current; current += v
  //   __syncthreads()
  //   if (warp >= 1): index += smem[warp - 1]
  //   return index
  //
  // SYCL: sub-group inclusive_scan + smem cross-sub-group serial scan.
  // ================================================================
  int inclusiveBinaryPrefixScan(
      sycl::nd_item<1> item,
      sycl::sub_group sg,
      int* smem,
      bool in) const {
    int sg_lid = sg.get_local_linear_id();
    int sg_size = sg.get_local_range()[0];
    int sg_id = sg.get_group_linear_id();
    int block_size = item.get_local_range(0);
    int num_sgs = (block_size + sg_size - 1) / sg_size;

    int val = in ? 1 : 0;

    // Intra-sub-group inclusive scan
    // (= CUDA __popc(getLaneMaskLe() & WARP_BALLOT(in)))
    int index = sycl::inclusive_scan_over_group(sg, val, sycl::plus<int>());

    // Sub-group carry = last lane's inclusive value
    // (= CUDA __popc(vote))
    int carry = sycl::group_broadcast(sg, index, sg_size - 1);

    // Lane 0 writes carry to smem
    // (= CUDA if (getLaneId() == 0) smem[warp] = carry)
    if (sg_lid == 0) {
      smem[sg_id] = carry;
    }
    sycl::group_barrier(item.get_group());

    // Thread 0 serial inclusive prefix sum over sub-group carries
    // (= CUDA if (threadIdx.x == 0) { for ... smem[i] = smem[i] + current; current += v; })
    if (item.get_local_id(0) == 0) {
      int current = 0;
      for (int i = 0; i < num_sgs; ++i) {
        int v = smem[i];
        smem[i] = smem[i] + current;
        current = current + v;
      }
    }
    sycl::group_barrier(item.get_group());

    // Add preceding sub-groups' prefix
    // (= CUDA if (warp >= 1) index += smem[warp - 1])
    if (sg_id >= 1) {
      index += smem[sg_id - 1];
    }

    return index;
  }

  // ================================================================
  // exclusiveBinaryPrefixScan (ScanUtils.cuh:64)
  //
  // CUDA:
  //   inclusiveBinaryPrefixScan(smem, in, &out, binop)
  //   *out -= (T)in                                     // inclusive->exclusive
  //   *carry = smem[ceil_div(blockDim.x, WARP_SIZE)-1]  // total
  //   __syncthreads()                                   // KillWARDependency
  // ================================================================
  void exclusiveBinaryPrefixScan(
      sycl::nd_item<1> item,
      sycl::sub_group sg,
      int* smem,
      bool in,
      int& out,
      int& carry) const {
    int block_size = item.get_local_range(0);
    int sg_size = sg.get_local_range()[0];
    int num_sgs = (block_size + sg_size - 1) / sg_size;

    int inclusive = inclusiveBinaryPrefixScan(item, sg, smem, in);

    // Inclusive to exclusive
    out = inclusive - (in ? 1 : 0);

    // Carry = total across all threads
    carry = smem[num_sgs - 1];

    // KillWARDependency = true
    sycl::group_barrier(item.get_group());
  }

  // ================================================================
  // radixSelect (SortingRadixSelect.cuh:860, non-ROCm path)
  //
  // Iterates MSB to LSB in RADIX_BITS steps.
  // At each step: count digits, scan to find which digit contains k-th.
  // found_unique (count==1, kToFind==1): findPattern + return
  // found_non_unique (count>=kToFind): narrow desired/desiredMask, continue
  // End: *topK = deconvert(desired)
  // ================================================================
  void radixSelect(
      sycl::nd_item<1> item,
      sycl::sub_group sg,
      int* smem,
      const scalar_t* data,
      int k,
      bool largest,
      int sliceSize,
      scalar_t* topKValue) const {
    int counts[RADIX_SIZE];
    RadixT desired = 0;
    RadixT desiredMask = 0;
    int kToFind = k;

    for (int digitPos = NUM_BITS - RADIX_BITS; digitPos >= 0;
         digitPos -= RADIX_BITS) {
      countRadixUsingMask(
          item, sg, smem, counts,
          desired, desiredMask, digitPos,
          data, sliceSize);

      // All threads execute the same scan logic (counts are identical).
      // Replicates CUDA found_unique / found_non_unique lambdas exactly.
      if (largest) {
        for (int i = RADIX_SIZE - 1; i >= 0; --i) {
          int count = counts[i];

          // found_unique: return from radixSelect
          if (count == 1 && kToFind == 1) {
            desired = Bitfield<RadixT>::setBitfield(
                desired, i, digitPos, RADIX_BITS);
            desiredMask = Bitfield<RadixT>::setBitfield(
                desiredMask, RADIX_MASK, digitPos, RADIX_BITS);
            *topKValue = findPattern(
                item, smem, data, sliceSize, desired, desiredMask);
            return;
          }

          // found_non_unique: break inner loop, continue outer
          if (count >= kToFind) {
            desired = Bitfield<RadixT>::setBitfield(
                desired, i, digitPos, RADIX_BITS);
            desiredMask = Bitfield<RadixT>::setBitfield(
                desiredMask, RADIX_MASK, digitPos, RADIX_BITS);
            break;
          }

          kToFind -= count;
        }
      } else {
        for (int i = 0; i < RADIX_SIZE; ++i) {
          int count = counts[i];

          if (count == 1 && kToFind == 1) {
            desired = Bitfield<RadixT>::setBitfield(
                desired, i, digitPos, RADIX_BITS);
            desiredMask = Bitfield<RadixT>::setBitfield(
                desiredMask, RADIX_MASK, digitPos, RADIX_BITS);
            *topKValue = findPattern(
                item, smem, data, sliceSize, desired, desiredMask);
            return;
          }

          if (count >= kToFind) {
            desired = Bitfield<RadixT>::setBitfield(
                desired, i, digitPos, RADIX_BITS);
            desiredMask = Bitfield<RadixT>::setBitfield(
                desiredMask, RADIX_MASK, digitPos, RADIX_BITS);
            break;
          }

          kToFind -= count;
        }
      }
    }

    // No unique result; desired fully determined
    *topKValue = TopKTypeConfig<scalar_t>::deconvert(desired);
  }

  // ================================================================
  // operator() — gatherTopK (TensorTopK.cu:40-182)
  //
  // 1. radixSelect to find k-th value
  // 2. Gather values strictly > topK (largest) or < topK (!largest)
  // 3. Fill remaining with values == topK
  // ================================================================
  void operator()(sycl::nd_item<1> item) const {
    int slice = item.get_group_linear_id();
    if (slice >= numSlices_) return;

    sycl::sub_group sg = item.get_sub_group();

    // Get raw int* pointer from local accessor
    int* smem = local_mem_.template get_multi_ptr<
        sycl::access::decorated::no>().get();

    const scalar_t* inputSlice = inputData_ + (int64_t)slice * sliceSize_;
    scalar_t* topKSlice = topKData_ + (int64_t)slice * k_;
    int64_t* indicesSlice = indicesData_ + (int64_t)slice * k_;

    // Step 1: radixSelect
    scalar_t topKValue = static_cast<scalar_t>(0);
    radixSelect(item, sg, smem, inputSlice, k_, largest_, sliceSize_, &topKValue);

    RadixT topKConverted = TopKTypeConfig<scalar_t>::convert(topKValue);

    // Step 2: Gather values strictly greater/less than topKValue
    // CUDA: numIterations = round_up(inputSliceSize, blockDim.x)
    int numIterations =
        ((sliceSize_ + SBTOPK_BLOCK - 1) / SBTOPK_BLOCK) * SBTOPK_BLOCK;
    int writeIndexStart = 0;

    for (int i = item.get_local_id(0); i < numIterations; i += SBTOPK_BLOCK) {
      bool inRange = (i < sliceSize_);
      scalar_t v = inRange ? inputSlice[i] : static_cast<scalar_t>(0);
      RadixT convertedV = TopKTypeConfig<scalar_t>::convert(v);

      bool hasTopK;
      if (largest_) {
        hasTopK = inRange && (convertedV > topKConverted);
      } else {
        hasTopK = inRange && (convertedV < topKConverted);
      }

      int index, carry;
      exclusiveBinaryPrefixScan(item, sg, smem, hasTopK, index, carry);

      if (hasTopK) {
        int writeIndex = writeIndexStart + index;
        topKSlice[writeIndex] = v;
        indicesSlice[writeIndex] = i;
      }
      writeIndexStart += carry;
    }

    // Step 3: Fill remaining with values == topKValue
    int topKRemaining = k_ - writeIndexStart;

    for (int i = item.get_local_id(0); i < numIterations; i += SBTOPK_BLOCK) {
      bool inRange = (i < sliceSize_);
      scalar_t v = inRange ? inputSlice[i] : static_cast<scalar_t>(0);
      RadixT convertedV = TopKTypeConfig<scalar_t>::convert(v);

      bool hasTopK = inRange && (convertedV == topKConverted);

      int index, carry;
      exclusiveBinaryPrefixScan(item, sg, smem, hasTopK, index, carry);

      if (hasTopK && index < topKRemaining) {
        int writeIndex = writeIndexStart + index;
        topKSlice[writeIndex] = v;
        indicesSlice[writeIndex] = i;
      }

      if (carry >= topKRemaining) {
        break;
      }
      topKRemaining -= carry;
      writeIndexStart += carry;
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    local_mem_ = sycl_local_acc_t<int>(SMEM_INTS, cgh);
  }

  SbtopkGatherFunctor(
      const scalar_t* inputData,
      scalar_t* topKData,
      int64_t* indicesData,
      int numSlices,
      int sliceSize,
      int k,
      bool largest)
      : inputData_(inputData),
        topKData_(topKData),
        indicesData_(indicesData),
        numSlices_(numSlices),
        sliceSize_(sliceSize),
        k_(k),
        largest_(largest) {}

  const scalar_t* inputData_;
  scalar_t* topKData_;
  int64_t* indicesData_;
  int numSlices_;
  int sliceSize_;
  int k_;
  bool largest_;
  sycl_local_acc_t<int> local_mem_;
};

// ================================================================
// Launch function
// ================================================================
template <typename scalar_t>
static void sbtopk_launch_kernel(
    const scalar_t* input,
    scalar_t* topK,
    int64_t* indices,
    int numSlices,
    int sliceSize,
    int k,
    bool largest) {
  SbtopkGatherFunctor<scalar_t> functor(
      input, topK, indices, numSlices, sliceSize, k, largest);

  sycl_kernel_submit(
      sycl::range<1>(numSlices * SBTOPK_BLOCK),
      sycl::range<1>(SBTOPK_BLOCK),
      at::xpu::getCurrentSYCLQueue(),
      functor);
}

bool sbtopk_try_launch(
    const at::Tensor& self,
    int64_t nsegments,
    int64_t nelements,
    int64_t k,
    bool largest,
    const at::Tensor& values,
    const at::Tensor& indices) {
  // Only handle cases where sbtopk is beneficial:
  // large dim, small k, contiguous last-dim
  if (nelements < 1024 || k > 256) {
    return false;
  }

  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "sbtopk_xpu",
      [&]() {
        sbtopk_launch_kernel<scalar_t>(
            static_cast<const scalar_t*>(self.const_data_ptr()),
            static_cast<scalar_t*>(values.data_ptr()),
            static_cast<int64_t*>(indices.data_ptr()),
            static_cast<int>(nsegments),
            static_cast<int>(nelements),
            static_cast<int>(k),
            largest);
      });

  return true;
}

} // namespace xpu
} // namespace native
} // namespace at
