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
#include <ATen/native/xpu/sycl/MemoryAccessUtils.h>
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

template <typename scalar_t, int VEC_SIZE = 4>
struct SbtopkGatherFunctor : public __SYCL_KER_CONFIG_CONVENTION__ {
  using RadixT = typename TopKTypeConfig<scalar_t>::RadixType;
  // CUDA uses sizeof(scalar_t)*8, NOT sizeof(RadixType)*8.
  // For fp16: sizeof(Half)=2 -> 16 bits, but sizeof(uint32_t)=4 -> 32 bits.
  // Using RadixT would scan garbage upper bits and break Half/BFloat16.
  static constexpr int NUM_BITS = sizeof(scalar_t) * 8;

  // ================================================================
  // countRadixUsingMask — per-thread counting + sub-group/work-group reduce
  //
  // Replaces ballot-based counting. Each thread:
  //   1. Loads VEC_SIZE elements per iteration (vectorized)
  //   2. Locally increments counts[digit] (pure ALU, no cross-lane)
  //   3. After loop: sub-group reduce + lane0 atomicAdd to smem + broadcast
  //
  // Eliminates all group_ballot calls in counting.
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

#pragma unroll
    for (int i = 0; i < RADIX_SIZE; ++i) {
      counts[i] = 0;
    }
    if (lid < RADIX_SIZE) {
      smem[lid] = 0;
    }
    sycl::group_barrier(item.get_group());

    // Each thread processes VEC_SIZE consecutive elements per iteration.
    // Stride = block_size * VEC_SIZE for coalesced access across threads.
    using LoadT = memory::aligned_vector<scalar_t, VEC_SIZE>;
    int stride = block_size * VEC_SIZE;

    // Vectorized main loop — full VEC_SIZE loads
    int base = lid * VEC_SIZE;
    for (; base + VEC_SIZE <= sliceSize; base += stride) {
      scalar_t src[VEC_SIZE];
      *reinterpret_cast<LoadT*>(&src) =
          *reinterpret_cast<const LoadT*>(&data[base]);
#pragma unroll
      for (int v = 0; v < VEC_SIZE; ++v) {
        RadixT val = TopKTypeConfig<scalar_t>::convert(src[v]);
        if ((val & desiredMask) == desired) {
          RadixT digit = Bitfield<RadixT>::getBitfield(val, digitPos, RADIX_BITS);
          counts[digit]++;
        }
      }
    }
    // Scalar tail — remaining elements
    for (int idx = base; idx < sliceSize && idx < base + VEC_SIZE; ++idx) {
      RadixT val = TopKTypeConfig<scalar_t>::convert(data[idx]);
      if ((val & desiredMask) == desired) {
        RadixT digit = Bitfield<RadixT>::getBitfield(val, digitPos, RADIX_BITS);
        counts[digit]++;
      }
    }

    // Sub-group reduce + work-group reduce via atomicAdd
#pragma unroll
    for (int j = 0; j < RADIX_SIZE; ++j) {
      int sg_total = sycl::reduce_over_group(sg, counts[j], sycl::plus<int>());
      if (sg_lid == 0) {
        sycl::atomic_ref<int, sycl::memory_order::relaxed,
                         sycl::memory_scope::work_group,
                         sycl::access::address_space::local_space>
            ref(smem[j]);
        ref.fetch_add(sg_total);
      }
    }
    sycl::group_barrier(item.get_group());

    // All threads read block-level totals
#pragma unroll
    for (int j = 0; j < RADIX_SIZE; ++j) {
      counts[j] = smem[j];
    }
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
  // exclusiveIntPrefixScan — integer exclusive prefix scan
  //
  // Each thread provides an integer count (0..VEC_SIZE). Returns:
  //   out: exclusive prefix sum (write offset for this thread)
  //   carry: total sum across all threads in the work-group
  //
  // Sub-group level: exclusive_scan_over_group + reduce_over_group
  // Cross sub-group: smem serial scan (same pattern as binary version)
  // ================================================================
  void exclusiveIntPrefixScan(
      sycl::nd_item<1> item,
      sycl::sub_group sg,
      int* smem,
      int local_count,
      int& out,
      int& carry) const {
    int sg_lid = sg.get_local_linear_id();
    int sg_id = sg.get_group_linear_id();
    int block_size = item.get_local_range(0);
    int sg_size = sg.get_local_range()[0];
    int num_sgs = block_size / sg_size;

    // Manual sub-group inclusive prefix scan (Kogge-Stone)
    // Sub-group size is 32 on B580 (Xe2/BMG).
    int val = local_count;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
      int n = sycl::shift_group_right(sg, val, offset);
      if (sg_lid >= offset) val += n;
    }
    // val = inclusive scan within sub-group
    int sg_inclusive = val;
    int sg_exclusive = sg_inclusive - local_count;
    int sg_total = sycl::group_broadcast(sg, sg_inclusive, sg_size - 1);

    // Lane 0 writes sub-group total to smem
    if (sg_lid == 0) {
      smem[sg_id] = sg_total;
    }
    sycl::group_barrier(item.get_group());

    // Thread 0: serial inclusive prefix sum over smem
    if (item.get_local_id(0) == 0) {
      int current = 0;
      for (int i = 0; i < num_sgs; ++i) {
        int v = smem[i];
        smem[i] = v + current;
        current += v;
      }
    }
    sycl::group_barrier(item.get_group());

    // Add cross-sub-group prefix
    int cross_sg_prefix = (sg_id >= 1) ? smem[sg_id - 1] : 0;
    out = sg_exclusive + cross_sg_prefix;

    // Carry = total across all threads
    carry = smem[num_sgs - 1];
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
  //
  // v3: Vectorized gatherTopK — each thread processes VEC_SIZE elements
  // per iteration with integer prefix scan instead of binary prefix scan.
  // Reduces ballot count from 2048 to 512 per sub-group (8x in step2, 4x step3).
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

    // Vectorized gather setup
    using LoadT = memory::aligned_vector<scalar_t, VEC_SIZE>;
    int lid = item.get_local_id(0);

    // Uniform iteration count: all threads must participate in every prefix scan.
    // Each iteration covers SBTOPK_BLOCK * VEC_SIZE elements.
    // Total iterations = ceil(sliceSize_ / (SBTOPK_BLOCK * VEC_SIZE))
    int vec_stride = SBTOPK_BLOCK * VEC_SIZE;
    int numVecIters = (sliceSize_ + vec_stride - 1) / vec_stride;

    // Step 2: Gather values strictly greater/less than topKValue
    int writeIndexStart = 0;

    for (int iter = 0; iter < numVecIters; ++iter) {
      int base = iter * vec_stride + lid * VEC_SIZE;

      scalar_t vals[VEC_SIZE];
      int match_offsets[VEC_SIZE]; // local offsets of matching elements within VEC_SIZE
      int local_count = 0;

      if (base + VEC_SIZE <= sliceSize_) {
        // Full vector load
        *reinterpret_cast<LoadT*>(&vals) =
            *reinterpret_cast<const LoadT*>(&inputSlice[base]);
#pragma unroll
        for (int v = 0; v < VEC_SIZE; ++v) {
          RadixT cv = TopKTypeConfig<scalar_t>::convert(vals[v]);
          bool match = largest_ ? (cv > topKConverted) : (cv < topKConverted);
          if (match) {
            match_offsets[local_count] = v;
            local_count++;
          }
        }
      } else if (base < sliceSize_) {
        // Tail: scalar loads for remaining elements
        for (int v = 0; v < VEC_SIZE && base + v < sliceSize_; ++v) {
          vals[v] = inputSlice[base + v];
          RadixT cv = TopKTypeConfig<scalar_t>::convert(vals[v]);
          bool match = largest_ ? (cv > topKConverted) : (cv < topKConverted);
          if (match) {
            match_offsets[local_count] = v;
            local_count++;
          }
        }
      }
      // else: base >= sliceSize_, local_count stays 0, thread still participates in scan

      int offset, carry;
      exclusiveIntPrefixScan(item, sg, smem, local_count, offset, carry);

      for (int j = 0; j < local_count; ++j) {
        int writeIndex = writeIndexStart + offset + j;
        int v_idx = match_offsets[j];
        topKSlice[writeIndex] = vals[v_idx];
        indicesSlice[writeIndex] = base + v_idx;
      }
      writeIndexStart += carry;
    }

    // Step 3: Fill remaining with values == topKValue
    int topKRemaining = k_ - writeIndexStart;

    for (int iter = 0; iter < numVecIters; ++iter) {
      int base = iter * vec_stride + lid * VEC_SIZE;

      scalar_t vals[VEC_SIZE];
      int match_offsets[VEC_SIZE];
      int local_count = 0;

      if (base + VEC_SIZE <= sliceSize_) {
        *reinterpret_cast<LoadT*>(&vals) =
            *reinterpret_cast<const LoadT*>(&inputSlice[base]);
#pragma unroll
        for (int v = 0; v < VEC_SIZE; ++v) {
          RadixT cv = TopKTypeConfig<scalar_t>::convert(vals[v]);
          if (cv == topKConverted) {
            match_offsets[local_count] = v;
            local_count++;
          }
        }
      } else if (base < sliceSize_) {
        for (int v = 0; v < VEC_SIZE && base + v < sliceSize_; ++v) {
          vals[v] = inputSlice[base + v];
          RadixT cv = TopKTypeConfig<scalar_t>::convert(vals[v]);
          if (cv == topKConverted) {
            match_offsets[local_count] = v;
            local_count++;
          }
        }
      }

      int offset, carry;
      exclusiveIntPrefixScan(item, sg, smem, local_count, offset, carry);

      for (int j = 0; j < local_count; ++j) {
        if (offset + j < topKRemaining) {
          int writeIndex = writeIndexStart + offset + j;
          int v_idx = match_offsets[j];
          topKSlice[writeIndex] = vals[v_idx];
          indicesSlice[writeIndex] = base + v_idx;
        }
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
  // VEC_SIZE: number of elements per vectorized load in countRadixUsingMask.
  // fp32/int32/int64/double: 4 elements (128-bit / 256-bit load)
  // fp16/bf16: 8 elements (128-bit load)
  constexpr int VEC_SIZE = sizeof(scalar_t) <= 2 ? 8 : 4;
  SbtopkGatherFunctor<scalar_t, VEC_SIZE> functor(
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
