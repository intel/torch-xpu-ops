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
 * RADIX_BITS, RADIX_SIZE, RADIX_MASK are defined in SortingRadixSelect.h (=2,4,3).
 * sbtopk uses SBTOPK_RADIX_BITS=4, SBTOPK_RADIX_SIZE=16, SBTOPK_RADIX_MASK=15.
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

// sbtopk uses RADIX_BITS=4 (16 digits per pass), halving radix passes for fp32.
// Cannot reuse RADIX_BITS/SIZE/MASK from SortingRadixSelect.h (constexpr int, can't #undef).
constexpr int SBTOPK_RADIX_BITS = 4;
constexpr int SBTOPK_RADIX_SIZE = 16; // 2 ^ SBTOPK_RADIX_BITS
constexpr int SBTOPK_RADIX_MASK = (SBTOPK_RADIX_SIZE - 1);

// Block size = 1024 threads, matching CUDA C10_LAUNCH_BOUNDS_1(1024)
constexpr int SBTOPK_BLOCK = 1024;

// SLM layout:
//   [0..63]  : used by countRadixUsingMask (smem[0..SBTOPK_RADIX_SIZE-1] for counts)
//              and by exclusiveIntPrefixScan (smem[0..num_sgs-1] for carries)
//   [64..65] : used by findPattern (flag + found index)
//   Total: 68 ints = 272 bytes
constexpr int SMEM_INTS = 68;

template <typename scalar_t, int VEC_SIZE = 4, int ELEMS_PER_THREAD = 32, int SIMD = 32>
struct SbtopkGatherFunctor {
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
  __attribute__((noinline))
  void countRadixUsingMask(
      sycl::nd_item<1> item,
      sycl::sub_group sg,
      int* smem,
      int counts[SBTOPK_RADIX_SIZE],
      RadixT desired,
      RadixT desiredMask,
      int digitPos,
      const scalar_t* data,
      int sliceSize) const {
    int lid = item.get_local_id(0);
    int block_size = item.get_local_range(0);
    int sg_lid = sg.get_local_linear_id();

#pragma unroll
    for (int i = 0; i < SBTOPK_RADIX_SIZE; ++i) {
      counts[i] = 0;
    }
    if (lid < SBTOPK_RADIX_SIZE) {
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
          RadixT digit = Bitfield<RadixT>::getBitfield(val, digitPos, SBTOPK_RADIX_BITS);
          counts[digit]++;
        }
      }
    }
    // Scalar tail — remaining elements
    for (int idx = base; idx < sliceSize && idx < base + VEC_SIZE; ++idx) {
      RadixT val = TopKTypeConfig<scalar_t>::convert(data[idx]);
      if ((val & desiredMask) == desired) {
        RadixT digit = Bitfield<RadixT>::getBitfield(val, digitPos, SBTOPK_RADIX_BITS);
        counts[digit]++;
      }
    }

    // Sub-group reduce + lane0 atomicAdd to smem.
    // Testing: reduce_over_group instead of manual Kogge-Stone.
#pragma unroll
    for (int j = 0; j < SBTOPK_RADIX_SIZE; ++j) {
      int total = sycl::reduce_over_group(sg, counts[j], sycl::plus<int>());
      if (sg_lid == 0) {
        sycl::atomic_ref<int, sycl::memory_order::relaxed,
                         sycl::memory_scope::work_group,
                         sycl::access::address_space::local_space>
            ref(smem[j]);
        ref.fetch_add(total);
      }
    }
    sycl::group_barrier(item.get_group());

    // All threads read block-level totals
#pragma unroll
    for (int j = 0; j < SBTOPK_RADIX_SIZE; ++j) {
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
  __attribute__((noinline))
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
    // Barrier required: init must be visible before any thread enters the loop
    sycl::group_barrier(item.get_group());

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

      if (found != 0) {
        return TopKTypeConfig<scalar_t>::convert(data[foundIdx]);
      }

      // WAR barrier: protect smem writes in next iteration from current reads
      sycl::group_barrier(item.get_group());
    }
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
  __attribute__((noinline))
  void exclusiveIntPrefixScan(
      sycl::nd_item<1> item,
      sycl::sub_group sg,
      int* smem,
      int local_count,
      int& out,
      int& carry) const {
    int sg_lid = sg.get_local_linear_id();
    int sg_id = sg.get_group_linear_id();
    constexpr int num_sgs = SBTOPK_BLOCK / SIMD;

    // inclusive_scan_over_group instead of manual Kogge-Stone
    int sg_inclusive = sycl::inclusive_scan_over_group(sg, local_count, sycl::plus<int>());
    int sg_exclusive = sg_inclusive - local_count;

    // group_broadcast to get sub-group total (last lane's inclusive value)
    int sg_total = sycl::group_broadcast(sg, sg_inclusive, SIMD - 1);
    if (sg_lid == SIMD - 1) {
      smem[sg_id] = sg_total;
    }
    sycl::group_barrier(item.get_group());

    // Thread 0: serial inclusive prefix sum over sub-group totals
    if (item.get_local_id(0) == 0) {
      int current = 0;
      for (int i = 0; i < num_sgs; ++i) {
        int v = smem[i];
        smem[i] = v + current;
        current += v;
      }
    }
    sycl::group_barrier(item.get_group());

    int cross_sg_prefix = (sg_id >= 1) ? smem[sg_id - 1] : 0;
    out = sg_exclusive + cross_sg_prefix;
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
  __attribute__((noinline))
  RadixT radixSelect(
      sycl::nd_item<1> item,
      sycl::sub_group sg,
      int* smem,
      const scalar_t* data,
      int k,
      bool largest,
      int sliceSize) const {
    int counts[SBTOPK_RADIX_SIZE];
    RadixT desired = 0;
    RadixT desiredMask = 0;
    int kToFind = k;

    for (int digitPos = NUM_BITS - SBTOPK_RADIX_BITS; digitPos >= 0;
         digitPos -= SBTOPK_RADIX_BITS) {
      countRadixUsingMask(
          item, sg, smem, counts,
          desired, desiredMask, digitPos,
          data, sliceSize);

      // All threads execute the same scan logic (counts are identical).
      // Replicates CUDA found_unique / found_non_unique lambdas exactly.
      if (largest) {
        for (int i = SBTOPK_RADIX_SIZE - 1; i >= 0; --i) {
          int count = counts[i];

          // found_unique: return from radixSelect
          if (count == 1 && kToFind == 1) {
            desired = Bitfield<RadixT>::setBitfield(
                desired, i, digitPos, SBTOPK_RADIX_BITS);
            desiredMask = Bitfield<RadixT>::setBitfield(
                desiredMask, SBTOPK_RADIX_MASK, digitPos, SBTOPK_RADIX_BITS);
            return findPattern(
                item, smem, data, sliceSize, desired, desiredMask);
          }

          // found_non_unique: break inner loop, continue outer
          if (count >= kToFind) {
            desired = Bitfield<RadixT>::setBitfield(
                desired, i, digitPos, SBTOPK_RADIX_BITS);
            desiredMask = Bitfield<RadixT>::setBitfield(
                desiredMask, SBTOPK_RADIX_MASK, digitPos, SBTOPK_RADIX_BITS);
            break;
          }

          kToFind -= count;
        }
      } else {
    for (int i = 0; i < SBTOPK_RADIX_SIZE; ++i) {
          int count = counts[i];

          if (count == 1 && kToFind == 1) {
            desired = Bitfield<RadixT>::setBitfield(
                desired, i, digitPos, SBTOPK_RADIX_BITS);
            desiredMask = Bitfield<RadixT>::setBitfield(
                desiredMask, SBTOPK_RADIX_MASK, digitPos, SBTOPK_RADIX_BITS);
            return findPattern(
                item, smem, data, sliceSize, desired, desiredMask);
          }

          if (count >= kToFind) {
            desired = Bitfield<RadixT>::setBitfield(
                desired, i, digitPos, SBTOPK_RADIX_BITS);
            desiredMask = Bitfield<RadixT>::setBitfield(
                desiredMask, SBTOPK_RADIX_MASK, digitPos, SBTOPK_RADIX_BITS);
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
  // v4: Each thread processes ELEMS_PER_THREAD elements per iteration
  // (LOADS_PER_ITER × vec4 loads), then ONE prefix scan.
  // v3: 32 iterations × 2 barriers = 64 barriers for step2
  // v4 (ELEMS_PER_THREAD=32): 4 iterations × 2 barriers = 8 barriers (8x reduction)
  // ================================================================
  [[sycl::reqd_sub_group_size(SIMD)]]
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
    // ELEMS_PER_THREAD: each thread processes this many elements per iteration.
    // Multiple vec4 loads per iteration, then ONE prefix scan.
    // v3: ELEMS_PER_THREAD=4 (=VEC_SIZE) → 32 iterations for dim=131072
    // v4: ELEMS_PER_THREAD=32 → 4 iterations → 8x fewer prefix scans/barriers
    static constexpr int LOADS_PER_ITER = ELEMS_PER_THREAD / VEC_SIZE;
    using LoadT = memory::aligned_vector<scalar_t, VEC_SIZE>;
    int lid = item.get_local_id(0);

    // Each iteration covers SBTOPK_BLOCK * ELEMS_PER_THREAD elements.
    int iter_stride = SBTOPK_BLOCK * ELEMS_PER_THREAD;
    int numIters = (sliceSize_ + iter_stride - 1) / iter_stride;

    // Step 2: Gather values strictly greater/less than topKValue
    int writeIndexStart = 0;

    for (int iter = 0; iter < numIters; ++iter) {
      // Each thread loads ELEMS_PER_THREAD elements from LOADS_PER_ITER vec4 chunks.
      // Thread layout: consecutive threads handle consecutive VEC_SIZE chunks.
      // Thread t handles chunks at offsets: t*VEC_SIZE, (t+SBTOPK_BLOCK)*VEC_SIZE, ...
      scalar_t vals[ELEMS_PER_THREAD];
      int match_indices[ELEMS_PER_THREAD]; // global index of matching elements
      int local_count = 0;

      int iter_base = iter * iter_stride;

#pragma unroll
      for (int L = 0; L < LOADS_PER_ITER; ++L) {
        int base = iter_base + L * SBTOPK_BLOCK * VEC_SIZE + lid * VEC_SIZE;

        if (base + VEC_SIZE <= sliceSize_) {
          scalar_t src[VEC_SIZE];
          *reinterpret_cast<LoadT*>(&src) =
              *reinterpret_cast<const LoadT*>(&inputSlice[base]);
#pragma unroll
          for (int v = 0; v < VEC_SIZE; ++v) {
            RadixT cv = TopKTypeConfig<scalar_t>::convert(src[v]);
            bool match = largest_ ? (cv > topKConverted) : (cv < topKConverted);
            if (match) {
              vals[local_count] = src[v];
              match_indices[local_count] = base + v;
              local_count++;
            }
          }
        } else if (base < sliceSize_) {
          for (int v = 0; v < VEC_SIZE && base + v < sliceSize_; ++v) {
            scalar_t sv = inputSlice[base + v];
            RadixT cv = TopKTypeConfig<scalar_t>::convert(sv);
            bool match = largest_ ? (cv > topKConverted) : (cv < topKConverted);
            if (match) {
              vals[local_count] = sv;
              match_indices[local_count] = base + v;
              local_count++;
            }
          }
        }
      }

      int offset, carry;
      exclusiveIntPrefixScan(item, sg, smem, local_count, offset, carry);

      for (int j = 0; j < local_count; ++j) {
        int writeIndex = writeIndexStart + offset + j;
        if (writeIndex < k_) {
          topKSlice[writeIndex] = vals[j];
          indicesSlice[writeIndex] = match_indices[j];
        }
      }
      writeIndexStart += carry;
    }

    // Step 3: Fill remaining with values == topKValue
    int topKRemaining = k_ - writeIndexStart;

    for (int iter = 0; iter < numIters; ++iter) {
      scalar_t vals[ELEMS_PER_THREAD];
      int match_indices[ELEMS_PER_THREAD];
      int local_count = 0;

      int iter_base = iter * iter_stride;

#pragma unroll
      for (int L = 0; L < LOADS_PER_ITER; ++L) {
        int base = iter_base + L * SBTOPK_BLOCK * VEC_SIZE + lid * VEC_SIZE;

        if (base + VEC_SIZE <= sliceSize_) {
          scalar_t src[VEC_SIZE];
          *reinterpret_cast<LoadT*>(&src) =
              *reinterpret_cast<const LoadT*>(&inputSlice[base]);
#pragma unroll
          for (int v = 0; v < VEC_SIZE; ++v) {
            RadixT cv = TopKTypeConfig<scalar_t>::convert(src[v]);
            if (cv == topKConverted) {
              vals[local_count] = src[v];
              match_indices[local_count] = base + v;
              local_count++;
            }
          }
        } else if (base < sliceSize_) {
          for (int v = 0; v < VEC_SIZE && base + v < sliceSize_; ++v) {
            scalar_t sv = inputSlice[base + v];
            RadixT cv = TopKTypeConfig<scalar_t>::convert(sv);
            if (cv == topKConverted) {
              vals[local_count] = sv;
              match_indices[local_count] = base + v;
              local_count++;
            }
          }
        }
      }

      int offset, carry;
      exclusiveIntPrefixScan(item, sg, smem, local_count, offset, carry);

      for (int j = 0; j < local_count; ++j) {
        if (offset + j < topKRemaining) {
          int writeIndex = writeIndexStart + offset + j;
          topKSlice[writeIndex] = vals[j];
          indicesSlice[writeIndex] = match_indices[j];
        }
      }

      if (carry >= topKRemaining) {
        break;
      }
      topKRemaining -= carry;
      writeIndexStart += carry;
    }
  }

  SbtopkGatherFunctor(
      const scalar_t* inputData,
      scalar_t* topKData,
      int64_t* indicesData,
      int numSlices,
      int sliceSize,
      int k,
      bool largest,
      sycl::local_accessor<int, 1> local_mem)
      : inputData_(inputData),
        topKData_(topKData),
        indicesData_(indicesData),
        numSlices_(numSlices),
        sliceSize_(sliceSize),
        k_(k),
        largest_(largest),
        local_mem_(local_mem) {}

  const scalar_t* inputData_;
  scalar_t* topKData_;
  int64_t* indicesData_;
  int numSlices_;
  int sliceSize_;
  int k_;
  bool largest_;
  sycl::local_accessor<int, 1> local_mem_;
};

// ================================================================
// Launch function
// ================================================================
template <typename scalar_t, int VEC_SIZE, int ELEMS_PER_THREAD>
static void sbtopk_launch_impl(
    const scalar_t* input,
    scalar_t* topK,
    int64_t* indices,
    int numSlices,
    int sliceSize,
    int k,
    bool largest) {
  namespace syclex = sycl::ext::oneapi::experimental;

  constexpr int SIMD = 32;
  using Functor = SbtopkGatherFunctor<scalar_t, VEC_SIZE, ELEMS_PER_THREAD, SIMD>;

  syclex::properties kernel_props{syclex::sub_group_size<SIMD>};

  auto& q = at::xpu::getCurrentSYCLQueue();
  q.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<int, 1> local_mem(SMEM_INTS, cgh);
    Functor functor(input, topK, indices, numSlices, sliceSize, k, largest, local_mem);
    cgh.parallel_for<Functor>(
        sycl::nd_range<1>(
            sycl::range<1>(numSlices * SBTOPK_BLOCK),
            sycl::range<1>(SBTOPK_BLOCK)),
        kernel_props,
        functor);
  });
}

// Dispatch macro to reduce boilerplate
#define SBTOPK_LAUNCH(V, E) \
  sbtopk_launch_impl<scalar_t, V, E>( \
      input, topK, indices, numSlices, sliceSize, k, largest)

template <typename scalar_t>
static void sbtopk_launch_kernel(
    const scalar_t* input,
    scalar_t* topK,
    int64_t* indices,
    int numSlices,
    int sliceSize,
    int k,
    bool largest) {
  // Determine ELEMS_PER_THREAD based on dim: target ~4 iterations
  int ept;
  if      (sliceSize >= 32 * SBTOPK_BLOCK) ept = 32;
  else if (sliceSize >= 16 * SBTOPK_BLOCK) ept = 16;
  else if (sliceSize >= 8  * SBTOPK_BLOCK) ept = 8;
  else if (sliceSize >= 4  * SBTOPK_BLOCK) ept = 4;
  else if (sliceSize >= 2  * SBTOPK_BLOCK) ept = 2;
  else                                      ept = 1;

  // Determine VEC_SIZE: largest power-of-2 dividing sliceSize,
  // capped by type max AND by EPT (vec <= ept required)
  constexpr int MAX_VEC = sizeof(scalar_t) <= 2 ? 8 : 4;
  int cap = MAX_VEC < ept ? MAX_VEC : ept;
  int vec = 1;
  if (cap >= 8 && sliceSize % 8 == 0) vec = 8;
  else if (cap >= 4 && sliceSize % 4 == 0) vec = 4;
  else if (cap >= 2 && sliceSize % 2 == 0) vec = 2;

  // Dispatch: VEC determines which EPT values are valid (EPT >= VEC, EPT % VEC == 0)
  if constexpr (MAX_VEC == 8) {
    // 16-bit types: VEC can be 8, 4, 2, 1
    if (vec == 8) {
      switch (ept) {
        case 8:  SBTOPK_LAUNCH(8, 8);  return;
        case 16: SBTOPK_LAUNCH(8, 16); return;
        default: SBTOPK_LAUNCH(8, 32); return;
      }
    } else if (vec == 4) {
      switch (ept) {
        case 4:  SBTOPK_LAUNCH(4, 4);  return;
        case 8:  SBTOPK_LAUNCH(4, 8);  return;
        case 16: SBTOPK_LAUNCH(4, 16); return;
        default: SBTOPK_LAUNCH(4, 32); return;
      }
    } else if (vec == 2) {
      switch (ept) {
        case 2:  SBTOPK_LAUNCH(2, 2);  return;
        case 4:  SBTOPK_LAUNCH(2, 4);  return;
        case 8:  SBTOPK_LAUNCH(2, 8);  return;
        case 16: SBTOPK_LAUNCH(2, 16); return;
        default: SBTOPK_LAUNCH(2, 32); return;
      }
    } else {
      switch (ept) {
        case 1:  SBTOPK_LAUNCH(1, 1);  return;
        case 2:  SBTOPK_LAUNCH(1, 2);  return;
        case 4:  SBTOPK_LAUNCH(1, 4);  return;
        case 8:  SBTOPK_LAUNCH(1, 8);  return;
        case 16: SBTOPK_LAUNCH(1, 16); return;
        default: SBTOPK_LAUNCH(1, 32); return;
      }
    }
  } else {
    // 32-bit types: VEC can be 4, 2, 1
    if (vec >= 4) {
      switch (ept) {
        case 4:  SBTOPK_LAUNCH(4, 4);  return;
        case 8:  SBTOPK_LAUNCH(4, 8);  return;
        case 16: SBTOPK_LAUNCH(4, 16); return;
        default: SBTOPK_LAUNCH(4, 32); return;
      }
    } else if (vec == 2) {
      switch (ept) {
        case 2:  SBTOPK_LAUNCH(2, 2);  return;
        case 4:  SBTOPK_LAUNCH(2, 4);  return;
        case 8:  SBTOPK_LAUNCH(2, 8);  return;
        case 16: SBTOPK_LAUNCH(2, 16); return;
        default: SBTOPK_LAUNCH(2, 32); return;
      }
    } else {
      switch (ept) {
        case 1:  SBTOPK_LAUNCH(1, 1);  return;
        case 2:  SBTOPK_LAUNCH(1, 2);  return;
        case 4:  SBTOPK_LAUNCH(1, 4);  return;
        case 8:  SBTOPK_LAUNCH(1, 8);  return;
        case 16: SBTOPK_LAUNCH(1, 16); return;
        default: SBTOPK_LAUNCH(1, 32); return;
      }
    }
  }
}

#undef SBTOPK_LAUNCH

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
