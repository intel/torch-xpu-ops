/*
 * Sub-group level top-k: each sub-group (32 threads) handles one slice.
 * Each thread maintains a sorted top-k buffer in registers.
 * Data loaded with vec4. Merge across sub-group via bitonic shuffle merge.
 * Zero SLM, zero barriers.
 *
 * Algorithm:
 *   Phase 1: Each thread scans dim/32 elements, maintains sorted top-k buffer
 *   Phase 2: 5 levels of pairwise bitonic merge via sub-group shuffles
 *   Phase 3: Lane 0 writes k results
 */

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/xpu/sycl/MemoryAccessUtils.h>
#include <ATen/native/xpu/sycl/SortingRadixSelect.h>
#include <ATen/native/xpu/sycl/TensorTopKSbtopkKernel.h>

namespace at {
namespace native {
namespace xpu {

static constexpr int SG_SIZE = 32;

// ================================================================
// SubgroupTopKFunctor
//
// K: compile-time max top-k (must be >= runtime k)
// VEC_SIZE: vectorized load width
// ================================================================
template <typename scalar_t, int K, int VEC_SIZE = 4>
struct SubgroupTopKFunctor {

  // Insert val into sorted descending buffer top_vals[0..K-1].
  // top_vals[0] is max, top_vals[K-1] is min (threshold).
  // If val <= threshold and buffer full, skip.
  // No break — fully unrolled, SIMD-friendly.
  inline void insert_largest(
      scalar_t* top_vals, int* top_idx, int count,
      scalar_t val, int idx) const {
    if (count >= K && !(val > top_vals[K - 1])) return;
    // Iterate from bottom (K-1) to top (0).
    // Shift down while val > element above; insert when val <= element above or at top.
    bool inserted = false;
#pragma unroll
    for (int i = K - 1; i >= 0; --i) {
      if (!inserted && (i == 0 || !(val > top_vals[i - 1]))) {
        top_vals[i] = val;
        top_idx[i] = idx;
        inserted = true;
      } else if (!inserted) {
        top_vals[i] = top_vals[i - 1];
        top_idx[i] = top_idx[i - 1];
      }
    }
  }

  inline void insert_smallest(
      scalar_t* top_vals, int* top_idx, int count,
      scalar_t val, int idx) const {
    if (count >= K && !(val < top_vals[K - 1])) return;
    bool inserted = false;
#pragma unroll
    for (int i = K - 1; i >= 0; --i) {
      if (!inserted && (i == 0 || !(val < top_vals[i - 1]))) {
        top_vals[i] = val;
        top_idx[i] = idx;
        inserted = true;
      } else if (!inserted) {
        top_vals[i] = top_vals[i - 1];
        top_idx[i] = top_idx[i - 1];
      }
    }
  }

  // Bitonic merge: given A[K] (sorted descending) and B[K] (sorted descending),
  // keep top-K in A (for largest mode).
  // Step 1: A[i] = max(A[i], B[K-1-i]) — produces bitonic sequence
  // Step 2: Bitonic sort to restore descending order
  inline void bitonic_merge_largest(
      scalar_t* A, int* A_idx,
      const scalar_t* B, const int* B_idx) const {
    // Step 1: compare with reversed partner
#pragma unroll
    for (int i = 0; i < K; ++i) {
      scalar_t bv = B[K - 1 - i];
      int bi = B_idx[K - 1 - i];
      if (bv > A[i]) {
        A[i] = bv;
        A_idx[i] = bi;
      }
    }
    // Step 2: bitonic sort descending
#pragma unroll
    for (int stride = K / 2; stride >= 1; stride >>= 1) {
#pragma unroll
      for (int i = 0; i < K; ++i) {
        int j = i ^ stride;
        if (j > i && A[i] < A[j]) {
          scalar_t tv = A[i]; A[i] = A[j]; A[j] = tv;
          int ti = A_idx[i]; A_idx[i] = A_idx[j]; A_idx[j] = ti;
        }
      }
    }
  }

  inline void bitonic_merge_smallest(
      scalar_t* A, int* A_idx,
      const scalar_t* B, const int* B_idx) const {
#pragma unroll
    for (int i = 0; i < K; ++i) {
      scalar_t bv = B[K - 1 - i];
      int bi = B_idx[K - 1 - i];
      if (bv < A[i]) {
        A[i] = bv;
        A_idx[i] = bi;
      }
    }
#pragma unroll
    for (int stride = K / 2; stride >= 1; stride >>= 1) {
#pragma unroll
      for (int i = 0; i < K; ++i) {
        int j = i ^ stride;
        if (j > i && A[i] > A[j]) {
          scalar_t tv = A[i]; A[i] = A[j]; A[j] = tv;
          int ti = A_idx[i]; A_idx[i] = A_idx[j]; A_idx[j] = ti;
        }
      }
    }
  }

  [[sycl::reqd_sub_group_size(32)]]
  void operator()(sycl::nd_item<1> item) const {
    sycl::sub_group sg = item.get_sub_group();
    int sg_lid = sg.get_local_linear_id();

    // Each sub-group handles one slice
    int sgs_per_wg = item.get_local_range(0) / SG_SIZE;
    int slice = item.get_group_linear_id() * sgs_per_wg
                + sg.get_group_linear_id();
    if (slice >= numSlices_) return;

    const scalar_t* inputSlice = inputData_ + (int64_t)slice * sliceSize_;
    scalar_t* topKSlice = topKData_ + (int64_t)slice * k_;
    int64_t* indicesSlice = indicesData_ + (int64_t)slice * k_;

    // Initialize sorted top-K buffer
    scalar_t top_vals[K];
    int top_idx_local[K];
    scalar_t init_val = largest_
        ? -std::numeric_limits<scalar_t>::infinity()
        :  std::numeric_limits<scalar_t>::infinity();
#pragma unroll
    for (int i = 0; i < K; ++i) {
      top_vals[i] = init_val;
      top_idx_local[i] = -1;
    }
    int count = 0;

    // ---- Phase 1: scan data with vec4 loads ----
    using LoadT = memory::aligned_vector<scalar_t, VEC_SIZE>;
    int stride = SG_SIZE * VEC_SIZE; // 128 elements per sub-group iteration

    int base;
    for (base = sg_lid * VEC_SIZE; base + VEC_SIZE <= sliceSize_; base += stride) {
      scalar_t src[VEC_SIZE];
      *reinterpret_cast<LoadT*>(&src) =
          *reinterpret_cast<const LoadT*>(&inputSlice[base]);
#pragma unroll
      for (int v = 0; v < VEC_SIZE; ++v) {
        if (largest_) {
          insert_largest(top_vals, top_idx_local, count, src[v], base + v);
        } else {
          insert_smallest(top_vals, top_idx_local, count, src[v], base + v);
        }
        if (count < K) count++;
      }
    }
    // Scalar tail
    for (int idx = base; idx < sliceSize_ && idx < base + VEC_SIZE; ++idx) {
      scalar_t val = inputSlice[idx];
      if (largest_) {
        insert_largest(top_vals, top_idx_local, count, val, idx);
      } else {
        insert_smallest(top_vals, top_idx_local, count, val, idx);
      }
      if (count < K) count++;
    }

    // ---- Phase 2: sub-group bitonic merge (5 levels for sg_size=32) ----
#pragma unroll
    for (int d = 0; d < 5; ++d) {
      int partner = sg_lid ^ (1 << d);

      scalar_t partner_vals[K];
      int partner_idx[K];
#pragma unroll
      for (int i = 0; i < K; ++i) {
        partner_vals[i] = sycl::select_from_group(sg, top_vals[i], partner);
        partner_idx[i] = sycl::select_from_group(sg, top_idx_local[i], partner);
      }

      if (largest_) {
        bitonic_merge_largest(top_vals, top_idx_local, partner_vals, partner_idx);
      } else {
        bitonic_merge_smallest(top_vals, top_idx_local, partner_vals, partner_idx);
      }
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
};

// ================================================================
// Launch function
// ================================================================
template <typename scalar_t, int K, int VEC_SIZE>
static void sbtopk_launch_impl(
    const scalar_t* input,
    scalar_t* topK,
    int64_t* indices,
    int numSlices,
    int sliceSize,
    int k,
    bool largest) {
  constexpr int WG_SIZE = 256; // 8 sub-groups per work-group
  constexpr int SGS_PER_WG = WG_SIZE / SG_SIZE;
  int num_wgs = (numSlices + SGS_PER_WG - 1) / SGS_PER_WG;

  SubgroupTopKFunctor<scalar_t, K, VEC_SIZE> functor(
      input, topK, indices, numSlices, sliceSize, k, largest);

  sycl_kernel_submit(
      sycl::range<1>(num_wgs * WG_SIZE),
      sycl::range<1>(WG_SIZE),
      at::xpu::getCurrentSYCLQueue(),
      functor);
}

template <typename scalar_t>
static void sbtopk_launch_kernel(
    const scalar_t* input,
    scalar_t* topK,
    int64_t* indices,
    int numSlices,
    int sliceSize,
    int k,
    bool largest) {
  constexpr int K = 16;
  // Max VEC_SIZE for this dtype
  constexpr int MAX_VEC = sizeof(scalar_t) <= 2 ? 8 : 4;

  // Pick largest VEC_SIZE such that:
  //   1. 32 * VEC_SIZE <= sliceSize  (all threads get at least one full vector)
  //   2. sliceSize % VEC_SIZE == 0   (slice boundaries are aligned)
  if (MAX_VEC >= 8 && sliceSize % 8 == 0 && SG_SIZE * 8 <= sliceSize) {
    sbtopk_launch_impl<scalar_t, K, 8>(input, topK, indices, numSlices, sliceSize, k, largest);
  } else if (MAX_VEC >= 4 && sliceSize % 4 == 0 && SG_SIZE * 4 <= sliceSize) {
    sbtopk_launch_impl<scalar_t, K, 4>(input, topK, indices, numSlices, sliceSize, k, largest);
  } else if (sliceSize % 2 == 0 && SG_SIZE * 2 <= sliceSize) {
    sbtopk_launch_impl<scalar_t, K, 2>(input, topK, indices, numSlices, sliceSize, k, largest);
  } else {
    sbtopk_launch_impl<scalar_t, K, 1>(input, topK, indices, numSlices, sliceSize, k, largest);
  }
}

bool sbtopk_try_launch(
    const at::Tensor& self,
    int64_t nsegments,
    int64_t nelements,
    int64_t k,
    bool largest,
    const at::Tensor& values,
    const at::Tensor& indices) {
  // Only handle cases where sub-group topk is beneficial
  if (k > 16) {
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
