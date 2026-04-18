/*
 * Subgroup top-k kernel for optimized topk on XPU.
 *
 * Each sub-group (32 lanes) handles one slice entirely in registers.
 * Zero SLM, zero barriers.
 *
 * Algorithm:
 *   Phase 1: Each lane scans dim/32 elements, maintains a sorted top-k
 *            buffer via insertion sort (fully unrolled, no branches on
 *            direction thanks to compile-time Largest template param).
 *   Phase 2: 5 levels of pairwise bitonic merge via sub-group shuffles
 *            to combine 32 per-lane buffers into one global top-k.
 *   Phase 3: Lane 0 writes k results. Output is already sorted.
 *
 * Dispatch: k <= 16 and enough segments (large batch) and dim >= 1024
 *           routes to subgroup top-k; otherwise falls back to original.
 */

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/xpu/sycl/MemoryAccessUtils.h>
#include <ATen/native/xpu/sycl/TensorTopKSbtopkKernel.h>
#include <comm/DeviceProperties.h>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <cstdint>
#include <limits>

namespace at {
namespace native {
namespace xpu {

namespace syclex = sycl::ext::oneapi::experimental;
namespace intelex = sycl::ext::intel::experimental;

static constexpr int SG_SIZE = 32;

// ================================================================
// SubgroupTopKFunctor
//
// K: compile-time max top-k (must be >= runtime k)
// VEC_SIZE: vectorized load width
// Largest: compile-time direction flag. Eliminates per-element branches
//          on largest_ that otherwise pessimize the tight insert/merge loops.
// IndexT: int32 when total elements (nsegments * nelements) <= INT_MAX
//         (common case), int64_t otherwise.  Mirrors CUDA's
//         canUse32BitIndexMath.  int32 avoids 64-bit arithmetic on slice
//         indices and element indices, reducing register pressure.
// ================================================================
template <
    typename scalar_t,
    int K,
    int VEC_SIZE,
    bool Largest,
    typename IndexT = int>
struct SubgroupTopKFunctor {
  // Insert val into a K-sorted buffer. For Largest=true the buffer is sorted
  // descending (top_vals[0] is max); for Largest=false it is sorted ascending
  // (top_vals[0] is min). The comparator `better(a, b)` means "a should sit
  // above b in the buffer" — i.e. strictly greater for largest, strictly less
  // for smallest.
  //
  // Fully unrolled, no early break — SIMD-friendly.
  inline void insert(
      scalar_t* top_vals,
      IndexT* top_idx,
      int count,
      scalar_t val,
      IndexT idx) const {
    // Threshold is at the bottom of the buffer (top_vals[K-1]).
    if constexpr (Largest) {
      if (count >= K && !(val > top_vals[K - 1]))
        return;
    } else {
      if (count >= K && !(val < top_vals[K - 1]))
        return;
    }
    bool inserted = false;
#pragma unroll
    for (int i = K - 1; i >= 0; --i) {
      // When count < K the buffer is partially filled: positions [0, count)
      // hold real values while [count, K) still contain sentinels.
      // Guard "i <= count" ensures we only stop at position i when
      // top_vals[i-1] is a real entry.  Without this guard, an input
      // value equal to the sentinel (e.g. all -inf for largest=true)
      // would always stop at position K-1, overwriting it repeatedly
      // instead of filling lower positions.
      bool stop;
      if constexpr (Largest) {
        stop = (i == 0) || (i <= count && !(val > top_vals[i - 1]));
      } else {
        stop = (i == 0) || (i <= count && !(val < top_vals[i - 1]));
      }
      if (!inserted && stop) {
        top_vals[i] = val;
        top_idx[i] = idx;
        inserted = true;
      } else if (!inserted) {
        top_vals[i] = top_vals[i - 1];
        top_idx[i] = top_idx[i - 1];
      }
    }
  }

  // Bitonic merge: A[K] and B[K] are both sorted in the "better" direction.
  // Step 1: A[i] = better(A[i], B[K-1-i]) — produces bitonic sequence.
  // Step 2: bitonic sort restores the sorted-by-better order on A.
  inline void bitonic_merge(
      scalar_t* A,
      IndexT* A_idx,
      const scalar_t* B,
      const IndexT* B_idx) const {
    // Step 1: compare with reversed partner
#pragma unroll
    for (int i = 0; i < K; ++i) {
      scalar_t bv = B[K - 1 - i];
      IndexT bi = B_idx[K - 1 - i];
      bool take;
      if constexpr (Largest) {
        take = bv > A[i];
      } else {
        take = bv < A[i];
      }
      if (take) {
        A[i] = bv;
        A_idx[i] = bi;
      }
    }
    // Step 2: bitonic sort — standard bitonic merge network.
    //
    // After step 1, A[0..K-1] is bitonic (first decreasing then increasing,
    // or vice versa).  At stride = K/2 we compare A[i] with A[i + K/2]
    // for i in [0, K/2) and swap so the "better" value goes to the low
    // half.  This guarantees:
    //   (a) every element in A[0..K/2-1] >= every element in A[K/2..K-1],
    //   (b) each half is itself bitonic (splitting a bitonic sequence at
    //       the midpoint with min/max produces two bitonic subsequences).
    // Recurse with stride K/4, K/8, ..., 1 and each sub-piece halves
    // again, until every piece has length 1 — the array is sorted.
    //
    // j = i ^ stride pairs each element with its partner at distance
    // `stride`.  The guard j > i ensures each pair is processed once.
    //
    // Example for K = 16:
    //   stride 8: (0,8) (1,9) (2,10) ... (7,15)   — 8 pairs
    //   stride 4: (0,4) (1,5) (2,6) (3,7)          — two groups of 4
    //             (8,12) (9,13) (10,14) (11,15)
    //   stride 2: (0,2) (1,3) (4,6) (5,7) ...      — four groups of 2
    //   stride 1: (0,1) (2,3) (4,5) ... (14,15)    — 8 adjacent pairs
#pragma unroll
    for (int stride = K / 2; stride >= 1; stride >>= 1) {
#pragma unroll
      for (int i = 0; i < K; ++i) {
        int j = i ^ stride;
        bool swap;
        if constexpr (Largest) {
          swap = (j > i) && (A[i] < A[j]);
        } else {
          swap = (j > i) && (A[i] > A[j]);
        }
        if (swap) {
          scalar_t tv = A[i];
          A[i] = A[j];
          A[j] = tv;
          IndexT ti = A_idx[i];
          A_idx[i] = A_idx[j];
          A_idx[j] = ti;
        }
      }
    }
  }

  void operator()(sycl::nd_item<1> item) const {
    sycl::sub_group sg = item.get_sub_group();
    int sg_lid = sg.get_local_linear_id();

    // Each sub-group handles one slice
    int sgs_per_wg = item.get_local_range(0) / SG_SIZE;
    IndexT slice =
        static_cast<IndexT>(item.get_group_linear_id()) * sgs_per_wg +
        sg.get_group_linear_id();
    if (slice >= numSlices_)
      return;

    const scalar_t* inputSlice =
        inputData_ + static_cast<int64_t>(slice) * sliceSize_;
    scalar_t* topKSlice = topKData_ + static_cast<int64_t>(slice) * k_;
    int64_t* indicesSlice = indicesData_ + static_cast<int64_t>(slice) * k_;

    // Initialize sorted top-K buffer
    scalar_t top_vals[K];
    IndexT top_idx_local[K];
    scalar_t init_val;
    if constexpr (Largest) {
      if constexpr (std::numeric_limits<scalar_t>::has_infinity) {
        init_val = -std::numeric_limits<scalar_t>::infinity();
      } else {
        init_val = std::numeric_limits<scalar_t>::lowest();
      }
    } else {
      if constexpr (std::numeric_limits<scalar_t>::has_infinity) {
        init_val = std::numeric_limits<scalar_t>::infinity();
      } else {
        init_val = std::numeric_limits<scalar_t>::max();
      }
    }
#pragma unroll
    for (int i = 0; i < K; ++i) {
      top_vals[i] = init_val;
      top_idx_local[i] = -1;
    }
    int count = 0;

    // ---- Phase 1: scan data with vec loads ----
    using LoadT = memory::aligned_vector<scalar_t, VEC_SIZE>;
    int stride = SG_SIZE * VEC_SIZE;

    int64_t base;
    for (base = sg_lid * VEC_SIZE; base + VEC_SIZE <= sliceSize_;
         base += stride) {
      alignas(alignof(LoadT)) scalar_t src[VEC_SIZE];
      *reinterpret_cast<LoadT*>(&src) =
          *reinterpret_cast<const LoadT*>(&inputSlice[base]);
#pragma unroll
      for (int v = 0; v < VEC_SIZE; ++v) {
        insert(
            top_vals,
            top_idx_local,
            count,
            src[v],
            static_cast<IndexT>(base + v));
        if (count < K)
          count++;
      }
    }
    // Scalar tail
    for (IndexT idx = static_cast<IndexT>(base);
         idx < sliceSize_ && idx < base + VEC_SIZE;
         ++idx) {
      scalar_t val = inputSlice[idx];
      insert(top_vals, top_idx_local, count, val, idx);
      if (count < K)
        count++;
    }

    // ---- Phase 2: sub-group bitonic merge (5 levels for sg_size=32) ----
#pragma unroll
    for (int d = 0; d < 5; ++d) {
      int partner = sg_lid ^ (1 << d);

      scalar_t partner_vals[K];
      IndexT partner_idx[K];
#pragma unroll
      for (int i = 0; i < K; ++i) {
        partner_vals[i] = sycl::select_from_group(sg, top_vals[i], partner);
        partner_idx[i] = sycl::select_from_group(sg, top_idx_local[i], partner);
      }

      bitonic_merge(top_vals, top_idx_local, partner_vals, partner_idx);
    }

    // ---- Phase 3: lane 0 writes output ----
    if (sg_lid == 0) {
      for (int i = 0; i < k_; ++i) {
        topKSlice[i] = top_vals[i];
        indicesSlice[i] = static_cast<int64_t>(top_idx_local[i]);
      }
    }
  }

  SubgroupTopKFunctor(
      const scalar_t* inputData,
      scalar_t* topKData,
      int64_t* indicesData,
      IndexT numSlices,
      int64_t sliceSize,
      int k)
      : inputData_(inputData),
        topKData_(topKData),
        indicesData_(indicesData),
        numSlices_(numSlices),
        sliceSize_(sliceSize),
        k_(k) {}

  const scalar_t* inputData_;
  scalar_t* topKData_;
  int64_t* indicesData_;
  IndexT numSlices_;
  int64_t sliceSize_;
  int k_;
};

// ================================================================
// Launch helpers
// ================================================================
template <typename scalar_t, int K, int VEC_SIZE, bool Largest, typename IndexT>
static void sbtopk_launch_impl(
    const scalar_t* input,
    scalar_t* topK,
    int64_t* indices,
    IndexT numSlices,
    int64_t sliceSize,
    int k) {
  constexpr int WG_SIZE = 256; // 8 sub-groups per work-group
  constexpr int SGS_PER_WG = WG_SIZE / SG_SIZE;
  auto num_wgs =
      (static_cast<int64_t>(numSlices) + SGS_PER_WG - 1) / SGS_PER_WG;

  SubgroupTopKFunctor<scalar_t, K, VEC_SIZE, Largest, IndexT> functor(
      input, topK, indices, numSlices, sliceSize, k);

  auto q = at::xpu::getCurrentSYCLQueue();
  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(num_wgs * WG_SIZE, WG_SIZE),
        syclex::properties{
            syclex::sub_group_size<SG_SIZE>, intelex::grf_size<128>},
        functor);
  });
}

// Vec-size dispatch: picks the largest VEC_SIZE compatible with
// (dtype, sliceSize).
template <typename scalar_t, int K, bool Largest, typename IndexT>
static void sbtopk_launch_vec_dispatch(
    const scalar_t* input,
    scalar_t* topK,
    int64_t* indices,
    IndexT numSlices,
    int64_t sliceSize,
    int k) {
  // Max VEC_SIZE for this dtype
  constexpr int MAX_VEC = sizeof(scalar_t) <= 2 ? 8 : 4;

  // Pick largest VEC_SIZE such that:
  //   1. SG_SIZE * VEC_SIZE <= sliceSize  (all threads get at
  //      least one full vector)
  //   2. sliceSize % VEC_SIZE == 0        (slice boundaries are aligned)
  //   3. input pointer is aligned to sizeof(scalar_t) * VEC_SIZE
  //      (usually guaranteed by PyTorch allocators, but a non-zero
  //      storage offset can break alignment)
  auto input_align = reinterpret_cast<uintptr_t>(input);
  if (MAX_VEC >= 8 && sliceSize % 8 == 0 && SG_SIZE * 8 <= sliceSize &&
      input_align % (sizeof(scalar_t) * 8) == 0) {
    sbtopk_launch_impl<scalar_t, K, 8, Largest, IndexT>(
        input, topK, indices, numSlices, sliceSize, k);
  } else if (
      MAX_VEC >= 4 && sliceSize % 4 == 0 && SG_SIZE * 4 <= sliceSize &&
      input_align % (sizeof(scalar_t) * 4) == 0) {
    sbtopk_launch_impl<scalar_t, K, 4, Largest, IndexT>(
        input, topK, indices, numSlices, sliceSize, k);
  } else if (
      sliceSize % 2 == 0 && SG_SIZE * 2 <= sliceSize &&
      input_align % (sizeof(scalar_t) * 2) == 0) {
    sbtopk_launch_impl<scalar_t, K, 2, Largest, IndexT>(
        input, topK, indices, numSlices, sliceSize, k);
  } else {
    sbtopk_launch_impl<scalar_t, K, 1, Largest, IndexT>(
        input, topK, indices, numSlices, sliceSize, k);
  }
}

template <typename scalar_t>
static void sbtopk_launch_kernel(
    const scalar_t* input,
    scalar_t* topK,
    int64_t* indices,
    int64_t numSlices,
    int64_t sliceSize,
    int k,
    bool largest) {
  constexpr int K = 16;
  // Dispatch on (largest, IndexT) at the outermost level:
  //   - largest: so tight insert/merge loops have no runtime direction branch
  //   - IndexT: int32 when total elements fit (common), int64 otherwise.
  //     Mirrors CUDA's canUse32BitIndexMath. int32 avoids 64-bit
  //     arithmetic and reduces register pressure.
  if (numSlices * sliceSize <= std::numeric_limits<int>::max()) {
    int numSlices32 = static_cast<int>(numSlices);
    if (largest) {
      sbtopk_launch_vec_dispatch<scalar_t, K, true, int>(
          input, topK, indices, numSlices32, sliceSize, k);
    } else {
      sbtopk_launch_vec_dispatch<scalar_t, K, false, int>(
          input, topK, indices, numSlices32, sliceSize, k);
    }
  } else {
    if (largest) {
      sbtopk_launch_vec_dispatch<scalar_t, K, true, int64_t>(
          input, topK, indices, numSlices, sliceSize, k);
    } else {
      sbtopk_launch_vec_dispatch<scalar_t, K, false, int64_t>(
          input, topK, indices, numSlices, sliceSize, k);
    }
  }
}

// ================================================================
// Dispatch: subgroup top-k vs original
//
//   - dim < 1024: original (kernel launch overhead dominates)
//   - dim >= 1024, large batch, k <= 16: subgroup top-k
// ================================================================
SbtopkResult sbtopk_try_launch(
    const at::Tensor& self,
    int64_t nsegments,
    int64_t nelements,
    int64_t k,
    bool largest,
    const at::Tensor& values,
    const at::Tensor& indices) {
  // Not beneficial for small dim
  if (nelements < 1024) {
    return SbtopkResult::FAILED;
  }

  // Subgroup top-k: best for large batch, k<=16.
  // Output is ALREADY SORTED (descending for largest, ascending for smallest).
  //
  // Threshold: nsegments >= thread_slots / 4.
  //   Subgroup top-k uses 1 sub-group per slice (reading data once), while
  //   the original kernel reads data multiple times (~3 radix passes). So
  //   subgroup top-k reaches memory-BW saturation at much lower occupancy.
  //   thread_slots/4 is the conservative cutoff.
  //
  // On B580: thread_slots = 160 EU * 8 HW threads = 1280, threshold = 320.
  int64_t thread_slots =
      ::xpu::sycl::syclGpuEuCount() * ::xpu::sycl::syclGpuHWThreadsPerEU();
  int64_t sg_threshold = thread_slots / 4;
  if (k <= 16 && nsegments >= sg_threshold) {
    AT_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        self.scalar_type(),
        "subgroup_topk_xpu",
        [&]() {
          sbtopk_launch_kernel<scalar_t>(
              static_cast<const scalar_t*>(self.const_data_ptr()),
              static_cast<scalar_t*>(values.data_ptr()),
              static_cast<int64_t*>(indices.data_ptr()),
              nsegments,
              nelements,
              static_cast<int>(k),
              largest);
        });
    return SbtopkResult::SORTED;
  }

  return SbtopkResult::FAILED;
}

} // namespace xpu
} // namespace native
} // namespace at
