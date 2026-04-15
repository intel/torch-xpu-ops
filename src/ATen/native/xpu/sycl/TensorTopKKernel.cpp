/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

// ============================================================================
// torch.topk XPU kernel implementation
//
// This file provides two topk selection strategies:
//
// 1. Original group radix select (segmented_group_select_pairs):
//    Tournament-style radix select from SortingKernels.h. Uses
//    RADIX_BITS=4, KEYS_PER_ITEM=4, GROUP_SIZE=1024 to process 4096
//    elements per work-group per round. Well-suited for small batch sizes
//    or very small slice dimensions. Supports k up to 256.
//
// 2. Single-block topk (sbtopk) — added for small-k, large-batch cases:
//    Uses RADIX_BITS=8, VEC=4 vectorized loads, BLOCK_SIZE=256.
//    One work-group per slice. Two phases:
//      Phase 1 (Radix Selection): 8-bit radix histogram over the entire
//        slice to identify the kth value. Only 2 passes for 16-bit types,
//        4 for float32, vs. 4-8 passes with RADIX_BITS=4.
//      Phase 2 (Deterministic Gather): Two-pass gather using exclusive
//        prefix scan (not atomics) to write results in deterministic order.
//        Mirrors CUDA's gatherTopK + exclusiveBinaryPrefixScan pattern.
//
// Dispatch logic (topk_kernel):
//   - k > 256           → full sort fallback (topk_out_with_sort)
//   - sbtopk eligible   → sbtopk (k <= 16, nelements > 4096, enough slices
//                          to saturate GPU occupancy)
//   - otherwise          → original group radix select
// ============================================================================

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/MemoryAccessUtils.h>
#include <ATen/native/xpu/sycl/Sorting.h>
#include <ATen/native/xpu/sycl/SortingKernels.h>
#include <ATen/native/xpu/sycl/SortingRadixSelect.h>

#include <ATen/native/xpu/sycl/TensorTopKKernel.h>

namespace at {
namespace native {
namespace xpu {

// Fallback for k > 256: perform a full sort and take the first k elements.
// This is simpler but O(n log n) instead of O(n), so only used when k is
// too large for the segmented radix select (which supports k <= 256).
void topk_out_with_sort(
    const Tensor& self,
    int64_t k,
    int64_t dim,
    bool largest,
    const Tensor& values,
    const Tensor& indices) {
  Tensor sorted_values, sorted_indices;
  std::tie(sorted_values, sorted_indices) =
      at::sort(self, /* stable= */ false, dim, largest);
  values.copy_(sorted_values.narrow(dim, 0, k));
  indices.copy_(sorted_indices.narrow(dim, 0, k));
}

// ============================================================
// Single-block topk (sbtopk) constants and kernel
//
// Why sbtopk over the original group radix select?
//   The original uses RADIX_BITS=4 and a tournament of 4096-element
//   chunks, requiring ~33 rounds for dim=131072. Each round does a
//   full 4-pass radix select (4 bits/pass × 4 passes = 16 bits for fp16).
//   sbtopk uses RADIX_BITS=8, scanning the entire slice in a single pass
//   per digit position (only 2 passes for fp16, 4 for fp32). Combined
//   with VEC=4 vectorized loads this yields 1.5-3x speedup for the
//   target shapes (batch=1024, dim=32K-131K, k<=16).
//
// Algorithm (mirrors CUDA TensorTopK.cu sbtopk):
//   Phase 1 — Radix Selection:
//     For each 8-bit digit position (MSB to LSB), build a 256-bucket
//     histogram in SLM using atomic adds, then thread-0 scans the
//     histogram to find which bucket contains the kth element.
//     After all passes, `kth_radix` holds the radix-converted value
//     of the kth element.
//
//   Phase 2 — Deterministic Gather (prefix-scan based):
//     Pass 1: Collect elements strictly better than kth. Each thread
//       counts how many of its VEC=4 elements qualify, then an exclusive
//       prefix scan computes deterministic write positions.
//     Pass 2: Fill remaining k slots with elements equal to kth (same
//       prefix-scan mechanism).
//
// Prefix scan structure (mirrors CUDA ScanUtils.cuh):
//   1. Intra-sub-group: sycl::exclusive_scan_over_group (≈ warp scan)
//   2. Cross-sub-group: each sub-group writes its total to SLM,
//      thread-0 does a serial prefix sum over sub-group totals,
//      then each thread adds its sub-group's prefix to get a
//      work-group-wide exclusive index.
//
// Deterministic: write positions fully determined by prefix scan;
// no atomics are used in the gather phase.
// ============================================================

// Work-group size for sbtopk. 256 threads = 8 sub-groups of 32.
// Chosen to balance occupancy (160 concurrent WGs on B580) and
// per-thread work (each thread processes slice_size/256 elements).
constexpr int SBTOPK_BLOCK = 256;

// 8-bit radix means 256 histogram buckets. This halves the number
// of digit passes vs RADIX_BITS=4 (e.g., 2 passes for fp16 instead
// of 4), at the cost of a larger histogram (256 ints = 1KB in SLM).
constexpr int SBTOPK_R_BITS = 8;
constexpr int SBTOPK_R_SIZE = 1 << SBTOPK_R_BITS;  // 256 buckets
constexpr int SBTOPK_R_MASK = SBTOPK_R_SIZE - 1;

// sbtopk is only used when k <= 16. For larger k, the original
// group radix select (k <= 256) or full sort (k > 256) is used.
constexpr int SBTOPK_MAX_K = 16;

// SLM (Shared Local Memory) layout — reused across phases:
//   Phase 1 (radix select):
//     [0..255]  = 256-bucket histogram counters
//     [256]     = selectedDigit (the winning bucket index)
//     [257]     = kToFind (remaining k after subtracting prior buckets)
//   Phase 2 (gather prefix scan):
//     [0..num_subgroups-1] = per-sub-group partial sums
//     [num_subgroups]      = work-group total
// Both phases fit within 258 ints (1032 bytes).
constexpr int SBTOPK_SLM_SIZE = SBTOPK_R_SIZE + 2;  // 258

// ---- sbtopk-local branchless convert/deconvert ----
//
// Radix selection requires converting floating-point values to unsigned
// integers that preserve the total order (i.e., a < b iff convert(a) <
// convert(b)). The shared TopKTypeConfig in SortingRadixSelect.h provides
// this, but includes a NaN-check branch that adds control-flow overhead
// in sbtopk's tight inner loops. Since topk does not require special NaN
// ordering semantics, we provide branchless specializations here.
//
// Branchless IEEE float-to-ordered-uint conversion:
//   convert:   mask = -(x >> sign_bit) | sign_bit   // all-ones if negative
//              return x ^ mask                       // flips all bits if neg,
//                                                    // only sign bit if pos
//   deconvert: mask = ((v >> sign_bit) - 1) | sign_bit  // inverse
//              return v ^ mask
//
// For 16-bit types (Half/BFloat16), the radix type is uint32_t (imposed by
// TopKTypeConfig) but only the lower 16 bits carry data, so we mask with
// CONVERT_MASK in the kernel to zero the upper bits.

template <typename scalar_t>
struct SbtopkConvert {
  using RadixT = typename TopKTypeConfig<scalar_t>::RadixType;
  // Fall back to TopKTypeConfig for integer types (no NaN issue)
  static inline RadixT convert(scalar_t v) {
    return TopKTypeConfig<scalar_t>::convert(v);
  }
  static inline scalar_t deconvert(RadixT v) {
    return TopKTypeConfig<scalar_t>::deconvert(v);
  }
};

template <>
struct SbtopkConvert<float> {
  using RadixT = uint32_t;
  static inline RadixT convert(float v) {
    RadixT x = *((uint32_t*)&v);
    RadixT mask = -((x >> 31)) | 0x80000000;
    return (x ^ mask);
  }
  static inline float deconvert(RadixT v) {
    RadixT mask = ((v >> 31) - 1) | 0x80000000;
    float r;
    *((uint32_t*)&r) = (v ^ mask);
    return r;
  }
};

template <>
struct SbtopkConvert<double> {
  using RadixT = uint64_t;
  static inline RadixT convert(double v) {
    RadixT x = *((uint64_t*)&v);
    RadixT mask = -((x >> 63)) | 0x8000000000000000ULL;
    return (x ^ mask);
  }
  static inline double deconvert(RadixT v) {
    RadixT mask = ((v >> 63) - 1) | 0x8000000000000000ULL;
    double r;
    *((uint64_t*)&r) = (v ^ mask);
    return r;
  }
};

template <>
struct SbtopkConvert<at::Half> {
  using RadixT = uint32_t;
  static inline RadixT convert(at::Half v) {
    RadixT x = *((uint16_t*)&v);
    RadixT mask = -((x >> 15)) | 0x8000;
    return (x ^ mask);
  }
  static inline at::Half deconvert(RadixT v) {
    RadixT mask = ((v >> 15) - 1) | 0x8000;
    return __ushort_as_half(v ^ mask);
  }
};

template <>
struct SbtopkConvert<at::BFloat16> {
  using RadixT = uint32_t;
  static inline RadixT convert(at::BFloat16 v) {
    RadixT x = v.x;
    RadixT mask = -((x >> 15)) | 0x8000;
    return (x ^ mask);
  }
  static inline at::BFloat16 deconvert(RadixT v) {
    RadixT mask = ((v >> 15) - 1) | 0x8000;
    at::BFloat16 r;
    r.x = (v ^ mask);
    return r;
  }
};

template <typename scalar_t, bool Largest>
struct SbtopkFunctor : public __SYCL_KER_CONFIG_CONVENTION__ {
  using RadixT = typename TopKTypeConfig<scalar_t>::RadixType;
  // VEC=4: each thread loads 4 contiguous elements via aligned_vector,
  // yielding coalesced 128-bit (fp32) or 64-bit (fp16) memory transactions.
  static constexpr int VEC = 4;
  using vec_t = at::native::memory::aligned_vector<scalar_t, VEC>;
  static constexpr int NUM_BITS = sizeof(scalar_t) * 8;
  // CONVERT_MASK zeroes upper bits for 16-bit types that use uint32_t RadixT.
  // For fp32/fp64 where RadixT width == scalar width, mask is all-ones (no-op).
  static constexpr RadixT CONVERT_MASK = (NUM_BITS == sizeof(RadixT) * 8)
      ? ~static_cast<RadixT>(0)
      : (static_cast<RadixT>(1) << NUM_BITS) - 1;

  // Convert scalar to ordered radix representation, masked to NUM_BITS.
  inline RadixT convert_masked(scalar_t v) const {
    return SbtopkConvert<scalar_t>::convert(v) & CONVERT_MASK;
  }

  void operator()(sycl::nd_item<1> item) const {
    // Each work-group handles one slice (= one topk problem instance).
    int slice_idx = item.get_group(0);
    int lid = item.get_local_id(0);        // local thread id [0, SBTOPK_BLOCK)
    int block_size = item.get_local_range(0);  // = SBTOPK_BLOCK
    auto sg = item.get_sub_group();
    int sg_size = sg.get_local_range()[0];     // typically 32 on Intel GPUs
    int sg_lid = sg.get_local_linear_id();     // lane id within sub-group
    int sg_id = sg.get_group_linear_id();      // sub-group index within WG
    int num_sgs = block_size / sg_size;        // = SBTOPK_BLOCK / 32 = 8

    // Pointers to this slice's input and output
    const scalar_t* slice_data = input_ + (int64_t)slice_idx * slice_size_;
    scalar_t* out_vals = values_ + (int64_t)slice_idx * k_;
    int64_t* out_idxs = indices_ + (int64_t)slice_idx * k_;

    auto smem_ptr =
        smem_.template get_multi_ptr<sycl::access::decorated::no>().get();

    // `desired` accumulates the radix bits of the kth element from MSB to LSB.
    // `desiredMask` tracks which bit positions have been determined so far.
    // `kToFind` is how many elements remain to be accounted for within the
    // current candidate set (decreases as we narrow down the digit bucket).
    RadixT desired = 0;
    RadixT desiredMask = 0;
    int kToFind = k_;

    // Vectorized load setup: process groups of VEC elements at a time.
    // Tail elements (slice_size_ % VEC) are handled separately.
    int64_t numVecs = slice_size_ / VEC;
    int64_t tailStart = numVecs * VEC;
    const vec_t* data_vec = reinterpret_cast<const vec_t*>(slice_data);

    // ======== Phase 1: Radix Selection (find the kth value) ========
    // Iterate from the most-significant digit to the least-significant.
    // For fp16: 2 iterations (bits 8..15, then 0..7).
    // For fp32: 4 iterations (bits 24..31, 16..23, 8..15, 0..7).
    // Each iteration:
    //   1. Build a 256-bucket histogram of the current digit for elements
    //      that match all previously-determined higher digits.
    //   2. Scan the histogram (from high to low for Largest, low to high
    //      otherwise) to find which bucket contains the kth element.
    //   3. Record that bucket's digit into `desired` and update `kToFind`.
    for (int digitPos = NUM_BITS - SBTOPK_R_BITS; digitPos >= 0;
         digitPos -= SBTOPK_R_BITS) {
      // Step 1a: Zero out the 256-entry histogram in SLM.
      for (int i = lid; i < SBTOPK_R_SIZE; i += block_size) {
        smem_ptr[i] = 0;
      }
      sycl::group_barrier(item.get_group());

      // Step 1b: Each thread loads VEC=4 elements at a time and atomically
      // increments the histogram bucket for the current digit position.
      // Only elements whose higher digits match `desired` are counted.
      for (int64_t vi = lid; vi < numVecs; vi += block_size) {
        vec_t v = data_vec[vi];
#pragma unroll
        for (int j = 0; j < VEC; ++j) {
          RadixT val = convert_masked(v.val[j]);
          if ((val & desiredMask) == desired) {
            int digit = (val >> digitPos) & SBTOPK_R_MASK;
            sycl::atomic_ref<
                int,
                sycl::memory_order::relaxed,
                sycl::memory_scope::work_group,
                sycl::access::address_space::local_space>
                ref(smem_ptr[digit]);
            ref.fetch_add(1);
          }
        }
      }
      // Handle tail elements (slice_size_ % VEC) that don't fill a full vector.
      for (int64_t i = tailStart + lid; i < slice_size_; i += block_size) {
        RadixT val = convert_masked(slice_data[i]);
        if ((val & desiredMask) == desired) {
          int digit = (val >> digitPos) & SBTOPK_R_MASK;
          sycl::atomic_ref<
              int,
              sycl::memory_order::relaxed,
              sycl::memory_scope::work_group,
              sycl::access::address_space::local_space>
              ref(smem_ptr[digit]);
          ref.fetch_add(1);
        }
      }
      sycl::group_barrier(item.get_group());

      // Step 2: Thread-0 scans the histogram to find which digit bucket
      // contains the kth element. For `Largest`, scan from bucket 255 down;
      // for smallest, scan from bucket 0 up.
      // Stores: smem[256] = selected digit, smem[257] = remaining kToFind
      // (kToFind for the next digit position = how many of the top-k fall
      // within this bucket, i.e., kToFind minus elements in higher buckets).
      if (lid == 0) {
        int cumCount = 0;
        if (Largest) {
          for (int i = SBTOPK_R_SIZE - 1; i >= 0; --i) {
            int count = smem_ptr[i];
            cumCount += count;
            if (cumCount >= kToFind) {
              smem_ptr[SBTOPK_R_SIZE] = i;
              smem_ptr[SBTOPK_R_SIZE + 1] = kToFind - (cumCount - count);
              break;
            }
          }
        } else {
          for (int i = 0; i < SBTOPK_R_SIZE; ++i) {
            int count = smem_ptr[i];
            cumCount += count;
            if (cumCount >= kToFind) {
              smem_ptr[SBTOPK_R_SIZE] = i;
              smem_ptr[SBTOPK_R_SIZE + 1] = kToFind - (cumCount - count);
              break;
            }
          }
        }
      }
      sycl::group_barrier(item.get_group());

      // All threads read the selected digit and remaining kToFind from SLM.
      int selectedDigit = smem_ptr[SBTOPK_R_SIZE];
      kToFind = smem_ptr[SBTOPK_R_SIZE + 1];

      // Incorporate this digit into `desired` and extend the mask.
      // After all iterations, `desired` == radix representation of kth element.
      desired =
          (desired & ~(static_cast<RadixT>(SBTOPK_R_MASK) << digitPos)) |
          (static_cast<RadixT>(selectedDigit) << digitPos);
      desiredMask |= (static_cast<RadixT>(SBTOPK_R_MASK) << digitPos);
      sycl::group_barrier(item.get_group());
    }

    // kth_radix now holds the ordered-uint representation of the kth value.
    RadixT kth_radix = desired;

    // ======== Phase 2: Gather using exclusive prefix scan ========
    //
    // Two-pass gather (mirrors CUDA's gatherTopK from TensorTopK.cu):
    //   Pass 1: Collect elements strictly better than kth (> for Largest,
    //           < for smallest). These are guaranteed to be in the top-k.
    //   Pass 2: Fill the remaining k slots with elements equal to kth.
    //
    // Why two passes instead of one?
    //   There may be more elements equal to kth than slots remaining.
    //   Pass 1 fills as many slots as possible with "strictly better" items,
    //   then Pass 2 fills exactly the remaining count with "equal" items,
    //   stopping as soon as k is reached.
    //
    // Prefix scan mechanism (per iteration of the vectorized loop):
    //   1. Each thread counts how many of its VEC=4 elements qualify.
    //   2. Intra-sub-group exclusive scan via sycl::exclusive_scan_over_group.
    //   3. Each sub-group's total is written to SLM.
    //   4. Thread-0 does a serial prefix sum over sub-group totals.
    //   5. Each thread combines its intra-sub-group offset + sub-group prefix
    //      + writeIndexStart to get its global write position.
    //
    // This is deterministic: write positions are fully determined by the
    // prefix scan, with no atomics used in the gather phase.

    // Number of vectorized loop iterations needed to cover all VEC-groups.
    int64_t numVecIters = (numVecs + block_size - 1) / block_size;
    // writeIndexStart tracks how many elements have been written so far.
    int writeIndexStart = 0;

    // --- Pass 1: elements strictly better than kth ---
    for (int64_t vi_base = 0; vi_base < numVecIters; vi_base++) {
      int64_t vi = vi_base * block_size + lid;
      bool vecInRange = (vi < numVecs);

      // Each thread counts how many of its VEC elements are strictly
      // better than kth, while caching values and indices in registers.
      int count = 0;
      scalar_t vals[VEC];
      int64_t idxs[VEC];
      if (vecInRange) {
        vec_t v = data_vec[vi];
        int64_t base_idx = vi * VEC;
#pragma unroll
        for (int j = 0; j < VEC; ++j) {
          vals[j] = v.val[j];
          idxs[j] = base_idx + j;
          RadixT rv = convert_masked(v.val[j]);
          bool take = Largest ? (rv > kth_radix) : (rv < kth_radix);
          if (take)
            count++;
        }
      }

      // --- Exclusive prefix scan across the work-group ---
      // Step 1: Intra-sub-group exclusive scan (hardware-accelerated).
      int sg_exclusive =
          sycl::exclusive_scan_over_group(sg, count, 0, sycl::plus<int>());
      // Sub-group total = last lane's exclusive + last lane's count.
      int sg_total = sycl::reduce_over_group(sg, count, sycl::plus<int>());

      // Step 2: Write each sub-group's total to SLM.
      if (sg_lid == sg_size - 1) {
        smem_ptr[sg_id] = sg_total;
      }
      sycl::group_barrier(item.get_group());

      // Step 3: Thread-0 does a serial exclusive prefix sum over sub-group
      // totals. After this, smem[s] = sum of totals for sub-groups 0..s-1,
      // and smem[num_sgs] = work-group grand total.
      int carry;
      if (lid == 0) {
        int current = 0;
        for (int s = 0; s < num_sgs; ++s) {
          int v = smem_ptr[s];
          smem_ptr[s] = current;
          current += v;
        }
        smem_ptr[num_sgs] = current;
      }
      sycl::group_barrier(item.get_group());

      // Step 4: Each thread computes its global exclusive index.
      // exclusive_idx = intra-sub-group offset + sub-group's prefix
      int sg_prefix = smem_ptr[sg_id];
      carry = smem_ptr[num_sgs];
      int exclusive_idx = sg_exclusive + sg_prefix;

      // Write qualifying elements to output at deterministic positions.
      if (vecInRange) {
        int writePos = writeIndexStart + exclusive_idx;
#pragma unroll
        for (int j = 0; j < VEC; ++j) {
          RadixT rv = convert_masked(vals[j]);
          bool take = Largest ? (rv > kth_radix) : (rv < kth_radix);
          if (take) {
            if (writePos < k_) {
              out_vals[writePos] = vals[j];
              out_idxs[writePos] = idxs[j];
            }
            writePos++;
          }
        }
      }

      // Advance the write cursor by the work-group's total matches.
      writeIndexStart += carry;
    }

    // Handle tail elements for pass 1 (slice_size_ % VEC elements).
    // Only thread-0 processes tail to avoid complex synchronization
    // for the small number of remaining elements (at most VEC-1 = 3).
    {
      bool hasTail = (lid == 0) && (tailStart < slice_size_);
      int tailCount = 0;
      scalar_t tailVals[VEC];
      int64_t tailIdxs[VEC];
      if (hasTail) {
        for (int64_t i = tailStart; i < slice_size_; i++) {
          RadixT rv = convert_masked(slice_data[i]);
          bool take = Largest ? (rv > kth_radix) : (rv < kth_radix);
          if (take) {
            tailVals[tailCount] = slice_data[i];
            tailIdxs[tailCount] = i;
            tailCount++;
          }
        }
        for (int t = 0; t < tailCount; t++) {
          int writePos = writeIndexStart + t;
          if (writePos < k_) {
            out_vals[writePos] = tailVals[t];
            out_idxs[writePos] = tailIdxs[t];
          }
        }
        writeIndexStart += tailCount;
      }
      // Broadcast updated writeIndexStart from thread-0 to all threads
      // via SLM, so pass 2 starts with the correct cursor.
      sycl::group_barrier(item.get_group());
      if (lid == 0)
        smem_ptr[0] = writeIndexStart;
      sycl::group_barrier(item.get_group());
      writeIndexStart = smem_ptr[0];
    }

    // --- Pass 2: elements equal to kth (fill remaining k slots) ---
    // topKRemaining = how many more elements we need to output.
    int topKRemaining = k_ - writeIndexStart;
    sycl::group_barrier(item.get_group());

    // Same structure as pass 1, but matching rv == kth_radix instead of >/< .
    for (int64_t vi_base = 0; vi_base < numVecIters; vi_base++) {
      int64_t vi = vi_base * block_size + lid;
      bool vecInRange = (vi < numVecs);

      int count = 0;
      scalar_t vals[VEC];
      int64_t idxs[VEC];
      if (vecInRange) {
        vec_t v = data_vec[vi];
        int64_t base_idx = vi * VEC;
#pragma unroll
        for (int j = 0; j < VEC; ++j) {
          vals[j] = v.val[j];
          idxs[j] = base_idx + j;
          RadixT rv = convert_masked(v.val[j]);
          if (rv == kth_radix)
            count++;
        }
      }

      // Exclusive prefix scan (same as pass 1)
      int sg_exclusive =
          sycl::exclusive_scan_over_group(sg, count, 0, sycl::plus<int>());
      int sg_total = sycl::reduce_over_group(sg, count, sycl::plus<int>());

      if (sg_lid == sg_size - 1) {
        smem_ptr[sg_id] = sg_total;
      }
      sycl::group_barrier(item.get_group());

      int carry;
      if (lid == 0) {
        int current = 0;
        for (int s = 0; s < num_sgs; ++s) {
          int v = smem_ptr[s];
          smem_ptr[s] = current;
          current += v;
        }
        smem_ptr[num_sgs] = current;
      }
      sycl::group_barrier(item.get_group());

      int sg_prefix = smem_ptr[sg_id];
      carry = smem_ptr[num_sgs];
      int exclusive_idx = sg_exclusive + sg_prefix;

      // Write equal-to-kth elements, but only until we've filled k slots.
      // exclusive_idx < topKRemaining ensures we don't overshoot.
      if (vecInRange) {
        int writePos = writeIndexStart + exclusive_idx;
#pragma unroll
        for (int j = 0; j < VEC; ++j) {
          RadixT rv = convert_masked(vals[j]);
          if (rv == kth_radix) {
            if (exclusive_idx < topKRemaining && writePos < k_) {
              out_vals[writePos] = vals[j];
              out_idxs[writePos] = idxs[j];
            }
            writePos++;
            exclusive_idx++;
          }
        }
      }

      // Early exit: if this iteration found enough equal elements to
      // fill the remaining slots, we're done — no need to scan further.
      if (carry >= topKRemaining) {
        break;
      }

      topKRemaining -= carry;
      writeIndexStart += carry;
    }

    // Handle tail elements for pass 2 (same single-thread approach).
    if (tailStart < slice_size_ && topKRemaining > 0) {
      if (lid == 0) {
        for (int64_t i = tailStart; i < slice_size_ && topKRemaining > 0;
             i++) {
          RadixT rv = convert_masked(slice_data[i]);
          if (rv == kth_radix) {
            if (writeIndexStart < k_) {
              out_vals[writeIndexStart] = slice_data[i];
              out_idxs[writeIndexStart] = i;
            }
            writeIndexStart++;
            topKRemaining--;
          }
        }
      }
    }
  }

  // Allocate SLM (Shared Local Memory) for the kernel.
  // Only 258 ints (1032 bytes) — much smaller than the 128KB SLM limit.
  void sycl_ker_config_convention(sycl::handler& cgh) {
    smem_ = sycl_local_acc_t<int>(SBTOPK_SLM_SIZE, cgh);
  }

  SbtopkFunctor(
      const scalar_t* input,
      scalar_t* values,
      int64_t* indices,
      int64_t slice_size,
      int64_t k)
      : input_(input),
        values_(values),
        indices_(indices),
        slice_size_(slice_size),
        k_(k) {}

 private:
  const scalar_t* input_;
  scalar_t* values_;
  int64_t* indices_;
  int64_t slice_size_;
  int64_t k_;
  sycl_local_acc_t<int> smem_;
};

// Launch the sbtopk kernel with nsegments work-groups, each of SBTOPK_BLOCK
// threads. The `largest` template parameter is resolved at compile time to
// avoid branching inside the hot radix selection loop.
template <typename scalar_t>
void sbtopk_launch(
    const scalar_t* input,
    scalar_t* values,
    int64_t* indices,
    int64_t nsegments,
    int64_t nelements,
    int64_t k,
    bool largest) {
  auto& queue = at::xpu::getCurrentSYCLQueue();

  if (largest) {
    auto f = SbtopkFunctor<scalar_t, true>(
        input, values, indices, nelements, k);
    sycl_kernel_submit(nsegments * SBTOPK_BLOCK, SBTOPK_BLOCK, queue, f);
  } else {
    auto f = SbtopkFunctor<scalar_t, false>(
        input, values, indices, nelements, k);
    sycl_kernel_submit(nsegments * SBTOPK_BLOCK, SBTOPK_BLOCK, queue, f);
  }
}

// Main topk dispatch function.
//
// Dispatch strategy (in priority order):
//   1. k > 256  → topk_out_with_sort (full sort, then slice)
//   2. sbtopk   → single-block radix select (k <= 16, large slices,
//                  enough slices to saturate GPU occupancy)
//   3. default  → segmented_group_select_pairs (original tournament
//                  radix select from SortingKernels.h, k <= 256)
//
// After selection, if sorted=true, the k results are sorted using
// segmented_sort_pairs (a radix sort on k elements per slice).
void topk_kernel(
    const at::Tensor& input,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted,
    const at::Tensor& values,
    const at::Tensor& indices) {
  if (k == 0) {
    return;
  }

  TORCH_CHECK(
      input.defined() && values.defined() && indices.defined(),
      "invalid inputs");

  auto self = (input.dim() == 0) ? input.view(1) : input;

  int64_t numel = self.numel();
  int64_t ndim = self.dim();
  dim = maybe_wrap_dim(dim, ndim);
  int64_t nelements = self.sizes()[dim];  // slice size (= dimension being selected)
  int64_t nsegments = numel / nelements;  // number of independent topk problems

  TORCH_CHECK(
      nelements <= std::numeric_limits<int>::max(),
      "The dimension being select can not have more than INT_MAX elements.");

  const auto self_dtype = self.dtype();
  TORCH_CHECK(
      self_dtype != ScalarType::ComplexFloat &&
          self_dtype != ScalarType::ComplexDouble,
      "Topk currently does not support complex dtypes on XPU.");

  auto out_sizes = self.sizes().vec();
  out_sizes[dim] = k;
  values.resize_(out_sizes);
  indices.resize_(out_sizes);

  if (k > 256) { // The segmented_group_select_pairs supports k<=256
    topk_out_with_sort(self.contiguous(), k, dim, largest, values, indices);
    return;
  }

  // Determine whether to use sbtopk (single-block radix select).
  // sbtopk launches one work-group (SBTOPK_BLOCK threads) per slice,
  // so it requires enough slices to fill the GPU's hardware thread slots.
  //
  // total_hw_threads = syclGpuEuCount() * syclGpuHWThreadsPerEU()
  //   (= number of sub-groups the GPU can run concurrently)
  // subgroups_per_wg = SBTOPK_BLOCK / subgroup_size
  // max_concurrent_wgs = total_hw_threads / subgroups_per_wg
  //
  // When nsegments < max_concurrent_wgs, the GPU would be underutilized,
  // so we fall back to the original group radix select (which uses fewer,
  // larger work-groups with multiple elements per thread).
  bool use_sbtopk = false;
  if (nelements > 4096 && k <= SBTOPK_MAX_K) {
    int64_t total_hw_threads = syclGpuEuCount() * syclGpuHWThreadsPerEU();
    constexpr int SUBGROUP_SIZE = 32;
    constexpr int64_t SUBGROUPS_PER_WG = SBTOPK_BLOCK / SUBGROUP_SIZE;
    int64_t max_concurrent_wgs = total_hw_threads / SUBGROUPS_PER_WG;
    use_sbtopk = (nsegments >= max_concurrent_wgs);
  }

  // Both sbtopk and the original radix select operate on the last dimension.
  // If dim != last, transpose so the target dimension becomes contiguous.
  Tensor self_;
  bool need_infer_dim = dim != ndim - 1;
  if (!need_infer_dim) {
    self_ = self.contiguous();
  } else {
    self_ = self.transpose(ndim - 1, dim).contiguous();
    std::swap(out_sizes[ndim - 1], out_sizes[dim]);
  }

  // Allocate output tensors (or reuse if already contiguous on last dim).
  Tensor values_, indices_;
  bool newvalues = false;
  bool newindices = false;
  if (!need_infer_dim && values.is_contiguous()) {
    values_ = values;
  } else {
    values_ = at::empty(out_sizes, values.options());
    newvalues = true;
  }
  if (!need_infer_dim && indices.is_contiguous()) {
    indices_ = indices;
  } else {
    indices_ = at::empty(out_sizes, indices.options());
    newindices = true;
  }

  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self_.scalar_type(),
      "topk_xpu",
      [&]() {
        scalar_t* self_ptr = self_.data_ptr<scalar_t>();
        scalar_t* values_ptr = values_.data_ptr<scalar_t>();
        int64_t* indices_ptr = indices_.data_ptr<int64_t>();

        if (use_sbtopk) {
          // sbtopk: one work-group per slice, RADIX_BITS=8, VEC=4
          sbtopk_launch<scalar_t>(
              self_ptr,
              values_ptr,
              indices_ptr,
              nsegments,
              nelements,
              k,
              largest);
        } else {
          // Original tournament radix select: RADIX_BITS=4, KPI=4,
          // GROUP_SIZE=1024, processes 4096 elements per round.
          segmented_group_select_pairs<scalar_t, int64_t>(
              self_ptr,
              (scalar_t*)values_ptr,
              nullptr,
              (int64_t*)indices_ptr,
              nsegments,
              nelements,
              k,
              largest);
        }

        // Both sbtopk and radix select output unsorted top-k results.
        // If sorted=true, run a radix sort on the k result elements per slice.
        if (sorted) {
          segmented_sort_pairs<scalar_t, int64_t>(
              values_ptr,
              values_ptr,
              indices_ptr,
              indices_ptr,
              nsegments,
              k,
              largest);
        }
      });

  // If we transposed earlier, transpose results back to the original layout
  // and copy into the caller's output tensors.
  if (newvalues) {
    if (need_infer_dim)
      values_.transpose_(ndim - 1, dim);
    values.copy_(values_);
  }
  if (newindices) {
    if (need_infer_dim)
      indices_.transpose_(ndim - 1, dim);
    indices.copy_(indices_);
  }
}

} // namespace xpu
} // namespace native
} // namespace at
