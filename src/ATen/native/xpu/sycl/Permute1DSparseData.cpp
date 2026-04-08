/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/native/xpu/sycl/Permute1DSparseData.h>
#include <comm/SYCLContext.h>

namespace at::native::xpu {

// ============================================================================
// SYCL Kernel Functors - Lengths Permutation
// ============================================================================

/**
 * @brief SYCL kernel functor for permuting lengths (int32 version)
 *
 * Simple element-parallel kernel: permuted_lengths[i] = lengths[permute[i]]
 */
template <typename index_t>
class Permute1DLengthsKernel {
public:
    Permute1DLengthsKernel(
        int64_t _permuted_lengths_size,
        const index_t* _lengths,
        const int32_t* _permute,
        index_t* _permuted_lengths)
        : permuted_lengths_size(_permuted_lengths_size),
          lengths(_lengths),
          permute(_permute),
          permuted_lengths(_permuted_lengths) {}

    void operator()(const sycl::nd_item<1>& item) const {
        const int64_t i = item.get_global_id(0);
        if (i < permuted_lengths_size) {
            permuted_lengths[i] = lengths[permute[i]];
        }
    }

private:
    int64_t permuted_lengths_size;
    const index_t* lengths;
    const int32_t* permute;
    index_t* permuted_lengths;
};

// ============================================================================
// SYCL Kernel Functors - Data Permutation (without weights)
// ============================================================================

/**
 * @brief SYCL kernel functor for permuting data without weights
 *
 * 2D parallel decomposition:
 * - Dimension 0 (y): segments (one row of work-items per segment)
 * - Dimension 1 (x): threads cooperating on one segment
 */
template <typename offsets_t, typename indices_t>
class Permute1DDataKernel {
public:
    Permute1DDataKernel(
        int64_t _permuted_indices_size,
        int64_t _permuted_lengths_size,
        const indices_t* _indices,
        const int32_t* _permute,
        const offsets_t* _input_offsets,
        const offsets_t* _output_offsets,
        indices_t* _permuted_indices)
        : permuted_indices_size(_permuted_indices_size),
          permuted_lengths_size(_permuted_lengths_size),
          indices(_indices),
          permute(_permute),
          input_offsets(_input_offsets),
          output_offsets(_output_offsets),
          permuted_indices(_permuted_indices) {}

    void operator()(const sycl::nd_item<2>& item) const {
        // Get segment ID and thread ID within segment
        const int32_t segment_base = item.get_group(0) * item.get_local_range(1);
        const int32_t segment_local = item.get_local_id(1);
        const int32_t segment_id = segment_base + segment_local;
        const int32_t tid = item.get_local_id(0);
        const int32_t threads_per_segment = item.get_local_range(0);

        if (segment_id >= permuted_lengths_size) {
            return;
        }

        // Calculate segment boundaries
        const offsets_t output_start = output_offsets[segment_id];
        const offsets_t output_end = (segment_id == permuted_lengths_size - 1)
            ? static_cast<offsets_t>(permuted_indices_size)
            : output_offsets[segment_id + 1];
        const int32_t segment_length = static_cast<int32_t>(output_end - output_start);
        const offsets_t input_start = input_offsets[permute[segment_id]];

        // Copy data using strided access pattern
        for (int32_t i = tid; i < segment_length; i += threads_per_segment) {
            permuted_indices[output_start + i] = indices[input_start + i];
        }
    }

private:
    int64_t permuted_indices_size;
    int64_t permuted_lengths_size;
    const indices_t* indices;
    const int32_t* permute;
    const offsets_t* input_offsets;
    const offsets_t* output_offsets;
    indices_t* permuted_indices;
};

// ============================================================================
// SYCL Kernel Functors - Data Permutation (with weights)
// ============================================================================

/**
 * @brief SYCL kernel functor for permuting data with weights
 *
 * Same structure as Permute1DDataKernel but also copies weight values.
 */
template <typename offsets_t, typename indices_t, typename weights_t>
class Permute1DDataWithWeightsKernel {
public:
    Permute1DDataWithWeightsKernel(
        int64_t _permuted_indices_size,
        int64_t _permuted_lengths_size,
        const indices_t* _indices,
        const weights_t* _weights,
        const int32_t* _permute,
        const offsets_t* _input_offsets,
        const offsets_t* _output_offsets,
        indices_t* _permuted_indices,
        weights_t* _permuted_weights)
        : permuted_indices_size(_permuted_indices_size),
          permuted_lengths_size(_permuted_lengths_size),
          indices(_indices),
          weights(_weights),
          permute(_permute),
          input_offsets(_input_offsets),
          output_offsets(_output_offsets),
          permuted_indices(_permuted_indices),
          permuted_weights(_permuted_weights) {}

    void operator()(const sycl::nd_item<2>& item) const {
        const int32_t segment_base = item.get_group(0) * item.get_local_range(1);
        const int32_t segment_local = item.get_local_id(1);
        const int32_t segment_id = segment_base + segment_local;
        const int32_t tid = item.get_local_id(0);
        const int32_t threads_per_segment = item.get_local_range(0);

        if (segment_id >= permuted_lengths_size) {
            return;
        }

        const offsets_t output_start = output_offsets[segment_id];
        const offsets_t output_end = (segment_id == permuted_lengths_size - 1)
            ? static_cast<offsets_t>(permuted_indices_size)
            : output_offsets[segment_id + 1];
        const int32_t segment_length = static_cast<int32_t>(output_end - output_start);
        const offsets_t input_start = input_offsets[permute[segment_id]];

        // Copy both indices and weights
        for (int32_t i = tid; i < segment_length; i += threads_per_segment) {
            permuted_indices[output_start + i] = indices[input_start + i];
            permuted_weights[output_start + i] = weights[input_start + i];
        }
    }

private:
    int64_t permuted_indices_size;
    int64_t permuted_lengths_size;
    const indices_t* indices;
    const weights_t* weights;
    const int32_t* permute;
    const offsets_t* input_offsets;
    const offsets_t* output_offsets;
    indices_t* permuted_indices;
    weights_t* permuted_weights;
};

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * @brief Compute exclusive cumsum (prefix sum) for offsets
 *
 * output[0] = 0
 * output[i] = sum(input[0:i]) for i > 0
 */
template <typename T>
at::Tensor exclusive_cumsum_xpu(const at::Tensor& input) {
    auto output = at::empty({input.numel() + 1}, input.options().dtype(at::kLong));
    output[0] = 0;
    if (input.numel() > 0) {
        auto cumsum = input.cumsum(0, at::kLong);
        output.slice(0, 1).copy_(cumsum);
    }
    return output;
}

/**
 * @brief Compute complete cumsum (inclusive + final element)
 *
 * output[0] = 0
 * output[i] = sum(input[0:i]) for i > 0
 * output[n] = sum(all input)
 */
template <typename T>
at::Tensor complete_cumsum_xpu(const at::Tensor& input) {
    auto output = at::empty({input.numel() + 1}, input.options().dtype(at::kLong));
    output[0] = 0;
    if (input.numel() > 0) {
        auto cumsum = input.cumsum(0, at::kLong);
        output.slice(0, 1).copy_(cumsum);
    }
    return output;
}

// ============================================================================
// Host Function - XPU Implementation
// ============================================================================

/**
 * @brief XPU implementation of permute_1D_sparse_data
 *
 * Permutes sparse data represented in jagged format according to permutation indices.
 *
 * @param permute Permutation indices tensor [P] - int32
 * @param lengths Segment lengths tensor [L] - int32/int64
 * @param indices Concatenated values tensor [V] - any type
 * @param weights Optional weights tensor [V] - float/double
 * @param permuted_lengths_sum Optional precomputed sum of permuted lengths
 * @return Tuple of (permuted_lengths, permuted_indices, permuted_weights)
 */
std::tuple<at::Tensor, at::Tensor, std::optional<at::Tensor>>
permute_1D_sparse_data_xpu(
    const at::Tensor& permute,
    const at::Tensor& lengths,
    const at::Tensor& indices,
    const std::optional<at::Tensor>& weights,
    const std::optional<int64_t>& permuted_lengths_sum) {

    // Device validation
    TORCH_INTERNAL_ASSERT(permute.device().type() == at::DeviceType::XPU,
                         "permute must be on XPU device");
    TORCH_INTERNAL_ASSERT(lengths.device().type() == at::DeviceType::XPU,
                         "lengths must be on XPU device");
    TORCH_INTERNAL_ASSERT(indices.device().type() == at::DeviceType::XPU,
                         "indices must be on XPU device");

    // Input validation
    TORCH_CHECK(permute.dim() == 1, "permute must be 1D");
    TORCH_CHECK(lengths.dim() == 1, "lengths must be 1D");
    TORCH_CHECK(indices.dim() == 1, "indices must be 1D");
    TORCH_CHECK(permute.dtype() == at::kInt, "permute must be int32");

    // Ensure contiguous
    const auto permute_contig = permute.contiguous();
    const auto lengths_contig = lengths.contiguous();
    const auto indices_contig = indices.contiguous();

    const int64_t permuted_lengths_size = permute.numel();
    const int64_t lengths_size = lengths.numel();

    // Handle empty input
    if (permuted_lengths_size == 0) {
        // Empty permutation returns empty outputs
        return {
            at::empty({0}, lengths.options()),
            at::empty({0}, indices.options()),
            weights.has_value() ? std::make_optional(at::empty({0}, weights->options())) : std::nullopt
        };
    }

    if (lengths_size == 0) {
        // Empty lengths but non-empty permute - return zeros for lengths and empty indices
        // Warning: if permute contains valid indices they would be out of bounds,
        // but typically 0 lengths implies 0 segments, so permute should be empty too.
        // If we really want to support "selecting from 0 segments" it implies permute indices are invalid
        // unless we assume they map to implicit 0 length segments?
        // Following CPU logic:
        return {
            at::zeros({permuted_lengths_size}, lengths.options()),
            at::empty({0}, indices.options()),
            weights.has_value() ? std::make_optional(at::empty({0}, weights->options())) : std::nullopt
        };
    }

    // Get SYCL queue
    sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue();

    // Phase 1: Allocate and compute permuted_lengths
    at::Tensor permuted_lengths = at::empty({permuted_lengths_size}, lengths.options());

    // Launch lengths permutation kernel
    {
        constexpr int LOCAL_SIZE = 256;
        const int64_t global_size = ((permuted_lengths_size + LOCAL_SIZE - 1) / LOCAL_SIZE) * LOCAL_SIZE;

        AT_DISPATCH_INDEX_TYPES(
            lengths.scalar_type(), "permute_1D_lengths_xpu", [&] {
              sycl_kernel_submit<Permute1DLengthsKernel<index_t>>(
                  sycl::range<1>(global_size),
                  sycl::range<1>(LOCAL_SIZE),
                  getCurrentSYCLQueue(),
                  Permute1DLengthsKernel<index_t>(
                      permuted_lengths_size,
                      lengths_contig.data_ptr<index_t>(),
                      permute_contig.data_ptr<int32_t>(),
                      permuted_lengths.data_ptr<index_t>()));
            }
        );
    }

    // Phase 2: Compute input and output offsets
    at::Tensor input_offsets = exclusive_cumsum_xpu<int64_t>(lengths_contig);
    at::Tensor output_offsets = complete_cumsum_xpu<int64_t>(permuted_lengths);

    // Phase 3: Determine output size
    int64_t permuted_indices_size = 0;
    if (permuted_lengths_sum.has_value()) {
        permuted_indices_size = permuted_lengths_sum.value();
    } else {
        // Need to sync to get the value from device
        permuted_indices_size = output_offsets[permuted_lengths_size].item<int64_t>();
    }

    // Phase 4: Allocate output tensors
    at::Tensor permuted_indices = at::empty(permuted_indices_size, indices.options());
    std::optional<at::Tensor> permuted_weights = std::nullopt;

    // Phase 5: Launch data permutation kernel
    constexpr int32_t THREADS_PER_SEGMENT = 64;
    constexpr int32_t SEGMENTS_PER_BLOCK = 16;

    const int64_t num_blocks = (permuted_lengths_size + SEGMENTS_PER_BLOCK - 1) / SEGMENTS_PER_BLOCK;
    sycl::range<2> global_range{static_cast<size_t>(num_blocks * THREADS_PER_SEGMENT),
                                 static_cast<size_t>(SEGMENTS_PER_BLOCK)};
    sycl::range<2> local_range{THREADS_PER_SEGMENT, SEGMENTS_PER_BLOCK};

    if (weights.has_value()) {
        // With weights
        const auto weights_contig = weights->contiguous();
        permuted_weights = at::empty(permuted_indices_size, weights->options());

        AT_DISPATCH_INDEX_TYPES(
            input_offsets.scalar_type(), "permute_1D_data_xpu_offsets", [&] {
                using offsets_t = index_t;
                AT_DISPATCH_ALL_TYPES_AND2(
                    at::kHalf, at::kBFloat16,
                    indices.scalar_type(), "permute_1D_data_xpu_indices", [&] {
                        using indices_t = scalar_t;
                        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                            weights->scalar_type(), "permute_1D_data_xpu_weights", [&] {
                                using weights_t = scalar_t;
                                sycl_kernel_submit<Permute1DDataWithWeightsKernel<offsets_t, indices_t, weights_t>>(
                                  sycl::range<2>(global_range),
                                  sycl::range<2>(local_range),
                                  getCurrentSYCLQueue(),
                                  Permute1DDataWithWeightsKernel<
                                          offsets_t,
                                          indices_t,
                                          weights_t>(
                                          permuted_indices_size,
                                          permuted_lengths_size,
                                          indices_contig.data_ptr<indices_t>(),
                                          weights_contig.data_ptr<weights_t>(),
                                          permute_contig.data_ptr<int32_t>(),
                                          input_offsets.data_ptr<offsets_t>(),
                                          output_offsets.data_ptr<offsets_t>(),
                                          permuted_indices.data_ptr<indices_t>(),
                                          permuted_weights->data_ptr<weights_t>()));
                            }
                        );
                    }
                );
            }
        );
    } else {
        // Without weights
        AT_DISPATCH_INDEX_TYPES(
            input_offsets.scalar_type(), "permute_1D_data_xpu_offsets", [&] {
                using offsets_t = index_t;
                AT_DISPATCH_ALL_TYPES_AND2(
                    at::kHalf, at::kBFloat16,
                    indices.scalar_type(), "permute_1D_data_xpu_indices", [&] {
                        using indices_t = scalar_t;
                        sycl_kernel_submit<Permute1DDataKernel<offsets_t, indices_t>>(
                          sycl::range<2>(global_range),
                          sycl::range<2>(local_range),
                          getCurrentSYCLQueue(),
                          Permute1DDataKernel<offsets_t, indices_t>(
                                  permuted_indices_size,
                                  permuted_lengths_size,
                                  indices_contig.data_ptr<indices_t>(),
                                  permute_contig.data_ptr<int32_t>(),
                                  input_offsets.data_ptr<offsets_t>(),
                                  output_offsets.data_ptr<offsets_t>(),
                                  permuted_indices.data_ptr<indices_t>()));
                    }
                );
            }
        );
    }

    return {permuted_lengths, permuted_indices, permuted_weights};
  }

} // namespace at::native::xpu
