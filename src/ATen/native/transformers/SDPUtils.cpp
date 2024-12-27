#include <ATen/native/transformers/sdp_utils_cpp.h>
#include <c10/util/Array.h>
#include <c10/util/Exception.h>

namespace sdp {

using c10::array_of;

bool check_all_tensors_on_device(sdp_params const& params, bool debug) {
  // Check that all tensors are on the GPU device
  // This should be handled by the stub dispatch, but whe call
  // can_use_*_attention directly from python we need to ensure that the tensors
  // are on cuda
  if (params.query.device().type() != at::DeviceType::XPU) {
    if (debug) {
      TORCH_WARN(
          "All tensors need to be on cuda device. Got query on device: ",
          params.query.device(),
          ", key on device: ",
          params.key.device(),
          ", value on device: ",
          params.value.device());
    }
    return false;
  }
  return true;
}

inline bool check_head_dim(sdp_params const& params, bool debug) {
  if (params.query.sym_size(-1) > 512) {
    return false;
  }
  return true;
}

bool can_use_mem_efficient_attention(sdp::sdp_params params, bool debug) {
  //  Define gate functions that determine if a flash kernel can be ran
  constexpr auto general_constraints =
      array_of<bool (*)(sdp_params const&, bool)>(
          sdp::check_runtime_disabled_mem_efficient,
          check_all_tensors_on_device,
          sdp::check_tensor_shapes,
          check_head_dim);
  for (auto& constraint : general_constraints) {
    if (!constraint(params, debug)) {
      return false;
    }
  }

  if (has_for_nested_inputs(params)) {
    constexpr auto nested_constraints =
        array_of<bool (*)(sdp_params const&, bool)>(
            sdp::check_requires_grad_and_nested,
            sdp::check_batch_size_nested,
            sdp::check_for_seq_len_0_nested_tensor);
    for (auto& constraint : nested_constraints) {
      if (!constraint(params, debug)) {
        return false;
      }
    }
  }

  if (has_only_dense_inputs(params)) {
    constexpr auto dense_constraints =
        array_of<bool (*)(sdp_params const&, bool)>(
            sdp::check_nonzero_sequence_lengths_dense,
            sdp::check_last_dim_stride_equals_1_dense<
                false /*ignore_singleton_dim=*/>,
            sdp::check_batch_size_and_num_heads_dense<
                false /*supports_grouped_query_attention=*/>);
    for (auto& constraint : dense_constraints) {
      if (!constraint(params, debug)) {
        return false;
      }
    }
  }

  return true;
}

} // namespace sdp
