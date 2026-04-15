/*
 * Direct SYCL translation of CUDA sbtopk::gatherTopK (non-ROCm path)
 * from pytorch/aten/src/ATen/native/cuda/TensorTopK.cu lines 40-182.
 *
 * Faithfully mirrors CUDA:
 *   - RADIX_BITS=2 (4 buckets, 16 passes for fp32)
 *   - Scalar loads (no vectorization)
 *   - block_size=1024
 *   - TopKTypeConfig::convert (same as CUDA)
 *   - `largest` as runtime parameter (same as CUDA)
 *   - exclusiveBinaryPrefixScan -> sub-group scan + SLM cross-sub-group scan
 *
 * Compiled as a separate .so to avoid regressing the original topk kernel.
 */

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/xpu/sycl/SortingRadixSelect.h>
#include <ATen/native/xpu/sycl/TensorTopKSbtopkKernel.h>

namespace at {
namespace native {
namespace xpu {

// Matching CUDA: block_size = min(ceil_align(sliceSize, warpSize), 1024)
// For all sbtopk cases (dim > 4096), this is always 1024.
constexpr int SBTOPK_BLOCK = 1024;
constexpr int SBTOPK_MAX_K = 16;

// RADIX_BITS=2, matching CUDA SortingRadixSelect.cuh

// SLM reused across phases:
//   Radix select: RADIX_SIZE + 2 = 6 entries
//   Prefix scan:  num_subgroups + 1 = 33 entries (block=1024, sg=32)
// Use 34 for alignment.
constexpr int SLM_SIZE = 34;

template <typename scalar_t>
struct SbtopkFunctor : public __SYCL_KER_CONFIG_CONVENTION__ {
  using RadixT = typename TopKTypeConfig<scalar_t>::RadixType;
  static constexpr int NUM_BITS = sizeof(RadixT) * 8;

  void operator()(sycl::nd_item<1> item) const {
    int lid = item.get_local_id(0);
    int block_size = item.get_local_range(0);
    auto sg = item.get_sub_group();
    int sg_size = sg.get_local_range()[0];
    int sg_lid = sg.get_local_linear_id();
    int sg_id = sg.get_group_linear_id();
    int num_sgs = block_size / sg_size;

    int slice = item.get_group(0);

    auto smem =
        smem_.template get_multi_ptr<sycl::access::decorated::no>().get();

    const scalar_t* inputSliceStart = input_ + (int64_t)slice * inputSliceSize_;
    scalar_t* topKSliceStart = topK_ + (int64_t)slice * k_;
    int64_t* indicesSliceStart = indices_ + (int64_t)slice * k_;

    // ======== Phase 0: Radix Select (find kth value) ========
    // RADIX_BITS=2, 4 buckets per pass, 16 passes for fp32.
    // Direct translation of CUDA radixSelect().
    RadixT desired = 0;
    RadixT desiredMask = 0;
    int kToFind = k_;

    for (int digitPos = NUM_BITS - RADIX_BITS; digitPos >= 0;
         digitPos -= RADIX_BITS) {
      // Zero histogram (4 buckets)
      if (lid < RADIX_SIZE) {
        smem[lid] = 0;
      }
      sycl::group_barrier(item.get_group());

      // Count elements per bucket (scalar loads, matching CUDA doLdg)
      for (int i = lid; i < inputSliceSize_; i += block_size) {
        RadixT val = TopKTypeConfig<scalar_t>::convert(inputSliceStart[i]);
        if ((val & desiredMask) == desired) {
          int digit = (val >> digitPos) & RADIX_MASK;
          sycl::atomic_ref<
              int,
              sycl::memory_order::relaxed,
              sycl::memory_scope::work_group,
              sycl::access::address_space::local_space>
              ref(smem[digit]);
          ref.fetch_add(1);
        }
      }
      sycl::group_barrier(item.get_group());

      // Thread 0: scan histogram to find target bucket
      if (lid == 0) {
        int cumCount = 0;
        if (largest_) {
          for (int i = RADIX_SIZE - 1; i >= 0; --i) {
            int count = smem[i];
            cumCount += count;
            if (cumCount >= kToFind) {
              smem[RADIX_SIZE] = i;
              smem[RADIX_SIZE + 1] = kToFind - (cumCount - count);
              break;
            }
          }
        } else {
          for (int i = 0; i < RADIX_SIZE; ++i) {
            int count = smem[i];
            cumCount += count;
            if (cumCount >= kToFind) {
              smem[RADIX_SIZE] = i;
              smem[RADIX_SIZE + 1] = kToFind - (cumCount - count);
              break;
            }
          }
        }
      }
      sycl::group_barrier(item.get_group());

      int selectedDigit = smem[RADIX_SIZE];
      kToFind = smem[RADIX_SIZE + 1];

      desired =
          (desired & ~(static_cast<RadixT>(RADIX_MASK) << digitPos)) |
          (static_cast<RadixT>(selectedDigit) << digitPos);
      desiredMask |= (static_cast<RadixT>(RADIX_MASK) << digitPos);
      sycl::group_barrier(item.get_group());
    }

    // After full radix select, desired == convert(kthValue).
    // No need for findPattern — we work directly with the radix representation.
    RadixT topKConverted = desired;

    // ======== Phase 1: Gather elements strictly better than kth ========
    // Direct translation of CUDA gatherTopK phase 1.
    // Round up to multiple of block_size so all threads participate in prefix scan.
    int numIterations =
        ((inputSliceSize_ + block_size - 1) / block_size) * block_size;
    int writeIndexStart = 0;

    for (int i = lid; i < numIterations; i += block_size) {
      bool inRange = (i < inputSliceSize_);
      scalar_t v =
          inRange ? inputSliceStart[i] : static_cast<scalar_t>(0);
      RadixT convertedV = TopKTypeConfig<scalar_t>::convert(v);
      bool hasTopK;
      if (largest_) {
        hasTopK = inRange && (convertedV > topKConverted);
      } else {
        hasTopK = inRange && (convertedV < topKConverted);
      }

      // -- exclusiveBinaryPrefixScan equivalent --
      // CUDA: warp shuffle scan + smem cross-warp scan
      // SYCL: sub-group scan + smem cross-sub-group scan
      int val = hasTopK ? 1 : 0;
      int sg_exclusive =
          sycl::exclusive_scan_over_group(sg, val, 0, sycl::plus<int>());
      int sg_total =
          sycl::reduce_over_group(sg, val, sycl::plus<int>());

      if (sg_lid == sg_size - 1) {
        smem[sg_id] = sg_total;
      }
      sycl::group_barrier(item.get_group());

      int carry;
      if (lid == 0) {
        int current = 0;
        for (int s = 0; s < num_sgs; ++s) {
          int tmp = smem[s];
          smem[s] = current;
          current += tmp;
        }
        smem[num_sgs] = current;
      }
      sycl::group_barrier(item.get_group());

      int sg_prefix = smem[sg_id];
      carry = smem[num_sgs];
      int index = sg_exclusive + sg_prefix;

      if (hasTopK) {
        int writeIndex = writeIndexStart + index;
        topKSliceStart[writeIndex] = v;
        indicesSliceStart[writeIndex] = i;
      }

      writeIndexStart += carry;
    }

    // ======== Phase 2: Gather elements equal to kth ========
    // Direct translation of CUDA gatherTopK phase 2.
    int topKRemaining = k_ - writeIndexStart;

    for (int i = lid; i < numIterations; i += block_size) {
      bool inRange = (i < inputSliceSize_);
      scalar_t v =
          inRange ? inputSliceStart[i] : static_cast<scalar_t>(0);
      RadixT convertedV = TopKTypeConfig<scalar_t>::convert(v);
      bool hasTopK = inRange && (convertedV == topKConverted);

      int val = hasTopK ? 1 : 0;
      int sg_exclusive =
          sycl::exclusive_scan_over_group(sg, val, 0, sycl::plus<int>());
      int sg_total =
          sycl::reduce_over_group(sg, val, sycl::plus<int>());

      if (sg_lid == sg_size - 1) {
        smem[sg_id] = sg_total;
      }
      sycl::group_barrier(item.get_group());

      int carry;
      if (lid == 0) {
        int current = 0;
        for (int s = 0; s < num_sgs; ++s) {
          int tmp = smem[s];
          smem[s] = current;
          current += tmp;
        }
        smem[num_sgs] = current;
      }
      sycl::group_barrier(item.get_group());

      int sg_prefix = smem[sg_id];
      carry = smem[num_sgs];
      int index = sg_exclusive + sg_prefix;

      if (hasTopK && index < topKRemaining) {
        int writeIndex = writeIndexStart + index;
        topKSliceStart[writeIndex] = v;
        indicesSliceStart[writeIndex] = i;
      }

      if (carry >= topKRemaining) {
        break;
      }

      topKRemaining -= carry;
      writeIndexStart += carry;
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    smem_ = sycl_local_acc_t<int>(SLM_SIZE, cgh);
  }

  SbtopkFunctor(
      const scalar_t* input,
      scalar_t* topK,
      int64_t* indices,
      int inputSliceSize,
      int k,
      bool largest)
      : input_(input),
        topK_(topK),
        indices_(indices),
        inputSliceSize_(inputSliceSize),
        k_(k),
        largest_(largest) {}

 private:
  const scalar_t* input_;
  scalar_t* topK_;
  int64_t* indices_;
  int inputSliceSize_;
  int k_;
  bool largest_;
  sycl_local_acc_t<int> smem_;
};

// Launch sbtopk. Returns true if used.
bool sbtopk_try_launch(
    const at::Tensor& self,
    int64_t nsegments,
    int64_t nelements,
    int64_t k,
    bool largest,
    const at::Tensor& values,
    const at::Tensor& indices) {
  if (nelements <= 4096 || k > SBTOPK_MAX_K) {
    return false;
  }

  // Occupancy check: need enough slices to saturate the GPU.
  int64_t total_hw_threads = syclGpuEuCount() * syclGpuHWThreadsPerEU();
  constexpr int SG_SIZE = 32;
  constexpr int64_t SGS_PER_WG = SBTOPK_BLOCK / SG_SIZE;
  int64_t max_concurrent_wgs = total_hw_threads / SGS_PER_WG;
  if (nsegments < max_concurrent_wgs) {
    return false;
  }

  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "sbtopk_xpu",
      [&]() {
        auto f = SbtopkFunctor<scalar_t>(
            static_cast<const scalar_t*>(self.const_data_ptr()),
            static_cast<scalar_t*>(values.data_ptr()),
            static_cast<int64_t*>(indices.data_ptr()),
            static_cast<int>(nelements),
            static_cast<int>(k),
            largest);
        auto& queue = at::xpu::getCurrentSYCLQueue();
        sycl_kernel_submit(
            nsegments * SBTOPK_BLOCK, SBTOPK_BLOCK, queue, f);
      });

  return true;
}

} // namespace xpu
} // namespace native
} // namespace at
