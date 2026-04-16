/*
 * Faithful SYCL translation of CUDA sbtopk (single-block topk).
 *
 * CUDA sources translated 1:1:
 *   - SortingRadixSelect.cuh: countRadixUsingMask (line 176), findPattern (239)
 *   - ScanUtils.cuh: inclusiveBinaryPrefixScan (16), exclusiveBinaryPrefixScan (64)
 *   - TensorTopK.cu: gatherTopK (lines 40-182), radixSelect (860)
 *
 * Key CUDA -> SYCL mappings:
 *   WARP_BALLOT(pred)         -> sycl::ext::oneapi::group_ballot(sg, pred)
 *   __popc(ballot)            -> ballot.count()  (or extract_bits + __builtin_popcount)
 *   getLaneMaskLe() & ballot  -> extract_bits + manual le_mask + __builtin_popcount
 *   getLaneId()               -> sg.get_local_linear_id()
 *   atomicAdd (smem)          -> sycl::atomic_ref (local_space)
 *   __syncthreads()           -> sycl::group_barrier(item.get_group())
 *   doLdg(ptr)                -> direct load (no read-only cache hint in SYCL)
 *   Bitfield<T>::getBitfield  -> software shift+mask (no PTX BFE/BFI in SYCL)
 *   smem[]                    -> int* from local accessor
 *
 * withinSliceStride = 1 (input is .contiguous() before calling sbtopk).
 * RADIX_BITS, RADIX_SIZE, RADIX_MASK are defined in SortingRadixSelect.h.
 */

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/xpu/sycl/SortingRadixSelect.h>
#include <ATen/native/xpu/sycl/TensorTopKSbtopkKernel.h>
#include <sycl/ext/oneapi/sub_group_mask.hpp>

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
  // CUDA uses sizeof(scalar_t)*8, NOT sizeof(RadixType)*8.
  // For fp16: sizeof(Half)=2 -> 16 bits, but sizeof(uint32_t)=4 -> 32 bits.
  // Using RadixT would scan garbage upper bits and break Half/BFloat16.
  static constexpr int NUM_BITS = sizeof(scalar_t) * 8;

  // ================================================================
  // countRadixUsingMask (SortingRadixSelect.cuh:176)
  //
  // CUDA: per-iteration WARP_BALLOT + __popc for each digit.
  // Each iteration, each thread processes one element. The warp collectively
  // counts via ballot+popc. counts[j] accumulates the warp-level count
  // across iterations (all lanes have the same value).
  // SYCL: per-iteration group_ballot + count() for each digit (1:1 match).
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

    // CUDA: WARP_BALLOT + __popc per iteration, per digit.
    // In CUDA, threads exit the loop when i >= sliceSize, and __ballot_sync
    // uses a mask to track active threads. In SYCL, group_ballot requires
    // converged control flow, so we round up the loop and have out-of-range
    // threads pass false for the vote (same result).
    int numIterations =
        ((sliceSize + block_size - 1) / block_size) * block_size;

    for (int i = lid; i < numIterations; i += block_size) {
      bool inRange = (i < sliceSize);
      // CUDA: TopKTypeConfig<scalar_t>::convert(doLdg(&data[i]))
      RadixT val = inRange
          ? TopKTypeConfig<scalar_t>::convert(data[i])
          : static_cast<RadixT>(0);

      bool hasVal = inRange && ((val & desiredMask) == desired);
      // CUDA: Bitfield<bitwise_t>::getBitfield(val, radixDigitPos, RadixBits)
      RadixT digit = Bitfield<RadixT>::getBitfield(val, digitPos, RADIX_BITS);

#pragma unroll
      for (int j = 0; j < RADIX_SIZE; ++j) {
        bool vote = hasVal && (digit == static_cast<RadixT>(j));
        // CUDA: counts[j] += __popc(WARP_BALLOT(vote));
        // group_ballot returns 64-bit sub_group_mask; .count() on 64-bit
        // compiles to software Brian Kernighan loop. Extract to uint32_t
        // first so sycl::popcount maps to hardware cbit instruction.
        auto ballot = sycl::ext::oneapi::group_ballot(sg, vote);
        uint32_t bits;
        ballot.extract_bits(bits);
        counts[j] += sycl::popcount(bits);
      }
    }

    // CUDA: if (getLaneId() == 0) atomicAdd(&smem[i], counts[i])
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
    // Barrier removed: post-read WAR; next call's barrier@85 covers it.
  }

  // ================================================================
  // findPattern (SortingRadixSelect.cuh:239)
  //
  // Finds the unique value whose convert() matches desired.
  // Returns RadixT (converted form) directly — no deconvert needed.
  // SYCL uses smem[64]=flag(int), smem[65]=index(int), then convert(data[index]).
  // ================================================================
  RadixT findPattern(
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
    // Barrier removed: post-write; barrier@177 covers the first read at line 179.

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
      // Barrier removed: post-read WAR; next iteration's barrier@177 covers it.

      if (found != 0) {
        return TopKTypeConfig<scalar_t>::convert(data[foundIdx]);
      }
    }
    // Should not reach here
    return static_cast<RadixT>(0);
  }

  // ================================================================
  // inclusiveBinaryPrefixScan (ScanUtils.cuh:16, non-ROCm path)
  //
  // CUDA:
  //   T vote = WARP_BALLOT(in);
  //   T index = __popc(getLaneMaskLe() & vote);   // inclusive within warp
  //   T carry = __popc(vote);                      // warp total
  //   lane 0: smem[warp] = carry
  //   __syncthreads()
  //   thread 0: serial prefix sum over smem
  //   __syncthreads()
  //   if (warp >= 1): index += smem[warp - 1]
  //   return index
  //
  // SYCL: group_ballot + extract_bits + manual le_mask + __builtin_popcount
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
    // CUDA: blockDim.x / C10_WARP_SIZE
    int num_sgs = block_size / sg_size;

    // CUDA: T vote = WARP_BALLOT(in)
    auto ballot = sycl::ext::oneapi::group_ballot(sg, in);
    uint32_t vote;
    ballot.extract_bits(vote);

    // CUDA: getLaneMaskLe() — PTX %%lanemask_le special register
    // Bitmask with bits 0..laneId set (inclusive)
    uint32_t le_mask =
        (static_cast<uint32_t>(sg_lid) >= 31u)
            ? ~0u
            : ((1u << (static_cast<uint32_t>(sg_lid) + 1)) - 1u);

    // CUDA: T index = __popc(getLaneMaskLe() & vote)
    int index = sycl::popcount(le_mask & vote);

    // CUDA: T carry = __popc(vote)
    int carry = sycl::popcount(vote);

    // CUDA: if (getLaneId() == 0) smem[warp] = carry
    if (sg_lid == 0) {
      smem[sg_id] = carry;
    }
    sycl::group_barrier(item.get_group());

    // CUDA: if (threadIdx.x == 0) { serial prefix sum over smem }
    // Note: binop = plus. smem[i] = smem[i] + current; current += v
    if (item.get_local_id(0) == 0) {
      int current = 0;
      for (int i = 0; i < num_sgs; ++i) {
        int v = smem[i];
        smem[i] = smem[i] + current;
        current = current + v;
      }
    }
    sycl::group_barrier(item.get_group());

    // CUDA: if (warp >= 1) index = binop(index, smem[warp - 1])
    if (sg_id >= 1) {
      index += smem[sg_id - 1];
    }

    return index;
  }

  // ================================================================
  // exclusiveBinaryPrefixScan (ScanUtils.cuh:64)
  //
  // CUDA:
  //   inclusiveBinaryPrefixScan<T, false>(smem, in, &out, binop)
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
    int num_sgs = block_size / sg_size;

    int inclusive = inclusiveBinaryPrefixScan(item, sg, smem, in);

    // Inclusive to exclusive
    out = inclusive - (in ? 1 : 0);

    // Carry = total across all threads
    carry = smem[num_sgs - 1];

    // Barrier removed: post-read KillWARDependency; next call's barrier@241 covers it.
  }

  // ================================================================
  // radixSelect (SortingRadixSelect.cuh:860, non-ROCm path)
  //
  // Iterates MSB to LSB in RADIX_BITS steps.
  // At each step: count digits, scan to find which digit contains k-th.
  // found_unique (count==1, kToFind==1): findPattern + return RadixT
  // found_non_unique (count>=kToFind): narrow desired/desiredMask, continue
  // End: return desired (RadixT, fully determined)
  // ================================================================
  RadixT radixSelect(
      sycl::nd_item<1> item,
      sycl::sub_group sg,
      int* smem,
      const scalar_t* data,
      int k,
      bool largest,
      int sliceSize) const {
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
            return findPattern(
                item, smem, data, sliceSize, desired, desiredMask);
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
            return findPattern(
                item, smem, data, sliceSize, desired, desiredMask);
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
    return desired;
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

    // Step 1: radixSelect — returns RadixT directly (no deconvert/convert round-trip)
    RadixT topKConverted = radixSelect(item, sg, smem, inputSlice, k_, largest_, sliceSize_);

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
