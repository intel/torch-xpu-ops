#include <comm/SYCLContext.h>
#include <torch/csrc/autograd/profiler.h>
#include <torch/csrc/autograd/custom_function.h>
#include <ATen/native/xpu/sycl/SplitEmbeddingLookupOps.h>

#include <torch/csrc/autograd/record_function_ops.h>

#include <ATen/native/xpu/sycl/fbgemm_utils/tensor_utils.h>
#include <ATen/native/xpu/sycl/fbgemm_utils/feature_gates.h>
#include <ATen/native/xpu/sycl/fbgemm_utils/pt2_arg_utils.h>
#include <ATen/native/xpu/sycl/fbgemm_utils/torch_library.h>
#include <ATen/native/xpu/sycl/fbgemm_utils/utils.h>

using Tensor = at::Tensor;
using namespace fbgemm_utils;
namespace profiler = torch::autograd::profiler;
namespace config = fbgemm_utils::config;

namespace at::native::xpu {

#define GET_OPTIONAL_TENSOR_VALUE(name, empty_tensor) name.has_value() ? name.value() : empty_tensor;


class SplitNoBagLookupFunction_rowwise_adagrad_Op_pt2 :
    public torch::autograd::Function<SplitNoBagLookupFunction_rowwise_adagrad_Op_pt2> {
 public:
  static constexpr bool is_traceable = true;
  static torch::autograd::variable_list forward(
    torch::autograd::AutogradContext* ctx,
    const Tensor& placeholder_autograd_tensor,
    const int64_t output_dtype,
    const at::TensorList weights,
    const c10::SymInt D,
    const Tensor& hash_size_cumsum,
    const int64_t total_hash_size_bits,
    const Tensor& indices,
    const Tensor& offsets, 
    std::vector<std::optional<at::Tensor>> aux_tensor,
    std::vector<int64_t> aux_int,
    std::vector<double> aux_float,
    c10::List<bool> aux_bool,
    at::TensorList momentum1, 
    Tensor learning_rate_tensor, 
    std::vector<int64_t> optim_int, 
    std::vector<double> optim_float) {

    // unpack Tensor lists
    
    Tensor weights_host;
    Tensor weights_dev;
    Tensor weights_uvm;
    Tensor weights_placements;
    Tensor weights_offsets;
    Tensor weights_lxu_cache;

    if (weights.size() == 3) {
      TENSOR_ON_CPU_OR_MTIA(weights[0]);
      TENSORS_EMPTY_OR_ON_SAME_DEVICE(weights[0], weights[1]);
      TENSORS_EMPTY_OR_ON_SAME_DEVICE(weights[0], weights[2]);
      weights_host = weights[0];
      weights_placements = weights[1];
      weights_offsets = weights[2]; 
    }
    else if (weights.size() == 5)  {
      TENSOR_ON_SYCL_XPU(weights[0]);
      TENSORS_EMPTY_OR_ON_SAME_DEVICE(weights[0], weights[1]);
      TENSORS_EMPTY_OR_ON_SAME_DEVICE(weights[0], weights[2]);
      TENSORS_EMPTY_OR_ON_SAME_DEVICE(weights[0], weights[3]);
      TENSORS_EMPTY_OR_ON_SAME_DEVICE(weights[0], weights[4]);
      weights_dev = weights[0]; 
      weights_uvm = weights[1];
      weights_placements = weights[2];
      weights_offsets = weights[3];
      weights_lxu_cache = weights[4];
    }
    else {
      TORCH_CHECK(false, "Invalid size of weights, expected 3 for CPU or 5 for XPU but got ", weights.size());
    }
    
    Tensor momentum1_host;
    Tensor momentum1_dev;
    Tensor momentum1_uvm;
    Tensor momentum1_placements;
    Tensor momentum1_offsets;

    if (momentum1.size() == 3) {
      TENSOR_ON_CPU_OR_MTIA(momentum1[0]);
      TENSORS_EMPTY_OR_ON_SAME_DEVICE(momentum1[0], momentum1[1]);
      TENSORS_EMPTY_OR_ON_SAME_DEVICE(momentum1[0], momentum1[2]);
      momentum1_host = momentum1[0];
      momentum1_placements = momentum1[1];
      momentum1_offsets = momentum1[2]; 
    }
    else if (momentum1.size() == 4)  {
      TENSOR_ON_SYCL_XPU(momentum1[0]);
      TENSORS_EMPTY_OR_ON_SAME_DEVICE(momentum1[0], momentum1[1]);
      TENSORS_EMPTY_OR_ON_SAME_DEVICE(momentum1[0], momentum1[2]);
      TENSORS_EMPTY_OR_ON_SAME_DEVICE(momentum1[0], momentum1[3]);
      momentum1_dev = momentum1[0]; 
      momentum1_uvm = momentum1[1];
      momentum1_placements = momentum1[2];
      momentum1_offsets = momentum1[3];
    }
    else {
      TORCH_CHECK(false, "Invalid size of momentum1, expected 3 for CPU or 4 for XPU but got ", momentum1.size());
    }

    const auto T = weights_offsets.sym_numel();
    const auto max_B_ = offsets.sym_size(0) / T;

    // Annotate Kineto trace
    const static bool is_annotate_trace_enabled = config::is_feature_enabled(
        config::FeatureGateName::TBE_ANNOTATE_KINETO_TRACE);
    std::string op_annotation = "";
    c10::intrusive_ptr<profiler::PythonRecordFunction> record_trace;
    if (is_annotate_trace_enabled) {
      std::stringstream ss;
      ss << "["
        << "weighted=F,"
        << "pooled=F,"
        << "vbe=F,"
        << "avg_B=" << (max_B_) << ","
        << "max_B=" << max_B_ << ","
        << "T=" << T << ","
        << "avg_D=" << (D) << ","
        << "max_D=" << D << ","
        << "num_indices=" << indices.sym_numel() << ","
        << "avg_pooling_fac=" << (static_cast<c10::SymFloat>(indices.sym_numel()) / T / max_B_)
        << "]";
      op_annotation = ss.str();
      record_trace = profiler::record_function_enter_new(
        "split_tbe_fwd" + op_annotation);
      ctx->saved_data["op_annotation"] = op_annotation;
    }
    // NOTE: The `local_uvm_cache_stats` variable held by the nn.Module has dtype int32_t
    // TODO: Hook up with frontend code
    at::TensorOptions uvm_options = weights_host.numel() > 0 ? weights_host.options() : weights_dev.options();
    const auto uvm_cache_stats = GET_OPTIONAL_TENSOR_VALUE(aux_tensor[IDX_UVM_CACHE_STATS], at::empty({0}, uvm_options.dtype(at::kInt)));
    TORCH_CHECK(aux_tensor[IDX_LXU_CACHE_LOCATIONS].has_value(), "lxu_cache_locations should have value.");
    const auto lxu_cache_locations = aux_tensor[IDX_LXU_CACHE_LOCATIONS].value();
    const auto is_experimental = aux_bool[IDX_IS_EXPERIMENTAL_TBE];

    // Default values for Dynamo tracing
    // SymInt does not support bitshifts operator
    // Constanting info_B_num_bits, info_B_mask for Dynamo for now.
    const auto info_B_num_bits = static_cast<int32_t>(aux_int[IDX_INFO_B_NUM_BITS]);
    const auto info_B_mask = static_cast<uint32_t>(aux_int[IDX_INFO_B_MASK]);
    TORCH_SYM_CHECK(max_B_.sym_le(info_B_mask), "Not enough bits to accommodate B"); // vbe

    // Setting learning rate tensor with `.fill_()` breaks apf_dlrm bento kernel with 
    // `RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation.`
    // This is because if a tensor is saved for backward and it is mutated later, this can cause correctness problems. 
    // Since the forward compute and backward compute see different data values for this tensor.
    // To work around, we pass the cloned tensor instead the mutated tensor
    Tensor learning_rate_tensor_cloned = learning_rate_tensor.clone();

    ctx->save_for_backward({
        weights_host,
        weights_dev,
        weights_uvm,
        weights_lxu_cache,
        weights_placements,
        weights_offsets,
        hash_size_cumsum,
        indices,
        offsets,
        lxu_cache_locations,
        momentum1_host,
        momentum1_dev,
        momentum1_uvm,
        momentum1_placements,
        momentum1_offsets,
        learning_rate_tensor_cloned
    });
    ctx->saved_data["D"] = D;
    ctx->saved_data["total_hash_size_bits"] = total_hash_size_bits;
    ctx->saved_data["gradient_clipping"] = static_cast<bool>(aux_bool[IDX_GRADIENT_CLIPPING]);
    ctx->saved_data["max_gradient"] = aux_float[IDX_MAX_GRADIENT];
    ctx->saved_data["stochastic_rounding"] = static_cast<bool>(aux_bool[IDX_STOCHASTIC_ROUNDING]);
    ctx->saved_data["info_B_num_bits"] = info_B_num_bits;
    const auto info_B_mask_int64 = static_cast<int64_t>(info_B_mask);
    ctx->saved_data["info_B_mask"] = info_B_mask_int64;
    ctx->saved_data["use_uniq_cache_locations_bwd"] = static_cast<bool>(aux_bool[IDX_USE_UNIQ_CACHE_LOCATIONS_BWD]);
    ctx->saved_data["use_homogeneous_placements"] = static_cast<bool>(aux_bool[IDX_USE_HOMOGENEOUS_PLACEMENTS]);

    const auto iter = aux_int[IDX_ITER];
    ctx->saved_data["iter"] = iter;
    // unpack optim args
    auto eps = optim_float[0];
    ctx->saved_data["eps"] = eps;
    auto weight_decay = optim_float[1];
    ctx->saved_data["weight_decay"] = weight_decay;
    auto weight_decay_mode = optim_int[0];
    ctx->saved_data["weight_decay_mode"] = weight_decay_mode;
    auto max_norm = optim_float[2];
    ctx->saved_data["max_norm"] = max_norm;
    const auto& flatten_weights_dev = weights_dev;
    // nobag
    static auto embedding_codegen_forward_op =
        Dispatcher::singleton()
        .findSchemaOrThrow("fbgemm::split_embedding_nobag_codegen_forward_unweighted_pt2_wrapper", "")
            .typed<Tensor(
                const Tensor& /*weights_host*/,
                const Tensor& /*weights_dev*/,
                const Tensor& /*weights_uvm*/,
                const Tensor& /*weights_lxu_cache*/,
                const Tensor& /*weights_placements*/,
                const Tensor& /*weights_offsets*/,
                const c10::SymInt /*D*/,
                const Tensor& /*hash_size_cumsum*/,
                const Tensor& /*indices*/,
                const Tensor& /*offsets*/,
                const Tensor& /*ssd_row_addrs or lxu_cache_locations*/,
                const Tensor& /*uvm_cache_stats*/,
                const bool /*is_experimental*/,
                const int64_t /*output_dtype*/
            )>();

    auto output = embedding_codegen_forward_op.call(
      weights_host,
      flatten_weights_dev,
      weights_uvm,
      weights_lxu_cache,
      weights_placements,
      weights_offsets,
      D,
      hash_size_cumsum,
      indices,
      offsets, 
      lxu_cache_locations,
      uvm_cache_stats, 
      is_experimental,
      output_dtype
    );

    if (is_annotate_trace_enabled) {
      record_trace->record.end();
    }
    
    return {output};
  }

static torch::autograd::variable_list backward(
    torch::autograd::AutogradContext* ctx,
    torch::autograd::variable_list grad_outputs) {
    const auto saved = ctx->get_saved_variables();
    auto savedItr = std::begin(saved);
    auto weights_host = *savedItr++;
    auto weights_dev = *savedItr++;
    auto weights_uvm = *savedItr++;
    auto weights_lxu_cache = *savedItr++;
    auto weights_placements = *savedItr++;
    auto weights_offsets = *savedItr++;
    auto hash_size_cumsum = *savedItr++;
    auto indices = *savedItr++;
    auto offsets = *savedItr++;
    auto lxu_cache_locations = *savedItr++;
    auto momentum1_host = *savedItr++;
    auto momentum1_dev = *savedItr++;
    auto momentum1_uvm = *savedItr++;
    auto momentum1_placements = *savedItr++;
    auto momentum1_offsets = *savedItr++;
    auto learning_rate_tensor = *savedItr++;
    auto D = ctx->saved_data["D"].toInt();
    auto total_hash_size_bits = ctx->saved_data["total_hash_size_bits"].toInt();
    auto gradient_clipping = ctx->saved_data["gradient_clipping"].toBool();
    auto max_gradient = ctx->saved_data["max_gradient"].toDouble();
    auto stochastic_rounding = ctx->saved_data["stochastic_rounding"].toBool();
    [[maybe_unused]] const int32_t info_B_num_bits = ctx->saved_data["info_B_num_bits"].toInt();
    [[maybe_unused]] const int64_t info_B_mask_int64 = ctx->saved_data["info_B_mask"].toInt();
    const auto use_uniq_cache_locations_bwd = ctx->saved_data["use_uniq_cache_locations_bwd"].toBool();
    const auto use_homogeneous_placements = ctx->saved_data["use_homogeneous_placements"].toBool();
    auto eps = ctx->saved_data["eps"].toDouble();
    auto weight_decay = ctx->saved_data["weight_decay"].toDouble();
    auto weight_decay_mode = ctx->saved_data["weight_decay_mode"].toInt();
    auto max_norm = ctx->saved_data["max_norm"].toDouble();

    const static bool is_annotate_trace_enabled = config::is_feature_enabled(
        config::FeatureGateName::TBE_ANNOTATE_KINETO_TRACE);
    c10::intrusive_ptr<profiler::PythonRecordFunction> record_trace;
    if (is_annotate_trace_enabled) {
      auto& op_annotation = ctx->saved_data["op_annotation"].toStringRef();
      record_trace = profiler::record_function_enter_new(
          "split_tbe_bwd" + op_annotation);
    }

    TORCH_CHECK_EQ(grad_outputs.size(), 1);

    constexpr int32_t BT_block_size = 32;
    constexpr int32_t max_segment_length_per_warp = 32;

    using torch::autograd::Variable;
    auto grad_output = gradient_clipping ? clamp(grad_outputs[0], -max_gradient, max_gradient) : grad_outputs[0];
    // nobag
    Tensor grad_weights_dev;
      
    static auto embedding_codegen_unweighted_backward_op =
        Dispatcher::singleton()
            .findSchemaOrThrow("fbgemm::split_embedding_nobag_backward_codegen_rowwise_adagrad_unweighted_pt2_wrapper", "")
            .typed<Tensor(
                const Tensor& /*grad_output*/,
                const Tensor& /*weights_host*/,
                const Tensor& /*weights_dev*/,
                const Tensor& /*weights_uvm*/,
                const Tensor& /*lxu_cache_weight*/,
                const Tensor& /*weights_placements*/,
                const Tensor& /*weights_offsets*/,
                const c10::SymInt /*D*/,
                const Tensor& /*hash_size_cumsum*/,
                const int64_t /*total_hash_size_bits*/,
                const Tensor& /*indices*/,
                const Tensor& /*offsets*/,
                const Tensor& /*lxu_cache_locations*/,
                const int64_t /*BT_block_size*/,
                const int64_t /*max_segment_length_per_warp*/,
                const bool /*stochastic_rounding*/,
                const int64_t /*info_B_num_bits*/,
                const int64_t /*info_B_mask_int64*/,
                const bool /*use_uniq_cache_locations_bwd*/,
                const bool /*use_homogeneous_placements*/, 
                Tensor,
                Tensor,
                Tensor,
                Tensor,
                Tensor,
                Tensor,
                double,
                double,
                int64_t,
                double
            )>();

    grad_weights_dev = embedding_codegen_unweighted_backward_op.call(
          grad_output,
          weights_host,
          weights_dev,
          weights_uvm,
          weights_lxu_cache,
          weights_placements,
          weights_offsets,
          D, 
          hash_size_cumsum,
          total_hash_size_bits,
          indices,
          offsets, 
          lxu_cache_locations,
          BT_block_size,
          max_segment_length_per_warp,
          stochastic_rounding,
          info_B_num_bits,
          info_B_mask_int64,
          use_uniq_cache_locations_bwd,
          use_homogeneous_placements,
          momentum1_host,
          momentum1_dev,
          momentum1_uvm,
          momentum1_placements,
          momentum1_offsets,
          learning_rate_tensor,
          eps,
          weight_decay,
          weight_decay_mode,
          max_norm
    );

    if (is_annotate_trace_enabled) {
      record_trace->record.end();
    }

    // Number of returned gradients have to match the input to Autograd's forward
    // The number of items in the tensorlist differ between devices and is determined at runtime
    std::vector<Tensor> ret;
    ret.push_back(Variable()); // placeholder autograd tensor
    ret.push_back(Variable()); // output_dtype
    if (weights_host.numel() > 0) {
      ret.push_back(Tensor()); // host_weights
    }
    else {
      ret.push_back(grad_weights_dev); // dev_weights
      ret.push_back(Variable()); // weights_uvm
      ret.push_back(Variable()); // weights_lxu_cache
    }
    ret.push_back(Variable()); // weights_placement
    ret.push_back(Variable()); // weights_offsets
    ret.push_back(Variable()); // D
    ret.push_back(Variable()); // hash_size_cumsum
    ret.push_back(Variable()); // total_hash_size_bits
    ret.push_back(Variable()); // indices
    ret.push_back(Variable()); // offsets 
    ret.push_back(Variable()); // aux_tensor
    ret.push_back(Variable()); // aux_int
    ret.push_back(Variable()); // aux_float
    ret.push_back(Variable()); // aux_bool
    ret.push_back(Variable()); // momentum1_dev or host
    ret.push_back(Variable()); // momentum1_placements
    ret.push_back(Variable()); // momentum1_offsets
    if (momentum1_host.numel() == 0) {
    ret.push_back(Variable()); // momentum1_uvm
    }
    ret.push_back(Variable()); // learning_rate_tensor
    ret.push_back(Variable()); // optim_int
    ret.push_back(Variable()); // optim_float
    return ret;

}
};

Tensor split_embedding_codegen_lookup_rowwise_adagrad_function_pt2_xpu(
    const Tensor& placeholder_autograd_tensor,
    const at::TensorList weights,
    const Tensor& D_offsets,
    const c10::SymInt total_D,
    const c10::SymInt max_D,
    const Tensor& hash_size_cumsum,
    const int64_t total_hash_size_bits,
    const Tensor& indices,
    const Tensor& offsets,
    const int64_t pooling_mode,
    const std::optional<Tensor>& indice_weights,
    const std::optional<Tensor>& feature_requires_grad,
    const int64_t output_dtype,
    const std::vector<std::optional<at::Tensor>>& aux_tensor,
    const std::vector<int64_t>& aux_int,
    const std::vector<double>& aux_float,
    c10::List<bool> aux_bool,
    at::TensorList momentum1, 
    Tensor learning_rate_tensor, 
    std::vector<int64_t> optim_int, 
    std::vector<double> optim_float,
    const c10::SymInt max_B,
    const c10::SymInt max_B_feature_rank,
    const c10::SymInt vbe_output_size
) {
    // Load the config value from JK once
    static auto is_tbev2_enabled = config::is_feature_enabled(config::FeatureGateName::TBE_V2);

    // Set to experimental if either the feature is enabled in JK, or the user specifies to use TBEv2
    aux_bool[IDX_IS_EXPERIMENTAL_TBE] = is_tbev2_enabled || aux_bool[IDX_IS_EXPERIMENTAL_TBE];

    // has vbe support and on xpu
    if (aux_tensor[IDX_B_OFFSETS].has_value()) {
        if (aux_bool[IDX_APPLY_GLOBAL_WEIGHT_DECAY] && optim_float[1] > 0) {
            TORCH_CHECK(false, "VBE is not supported yet in SYCL backend");
        }
      // vbe and no gwd support
        TORCH_CHECK(false, "VBE is not supported yet in SYCL backend");
    }
    // has gwd support
     if (aux_bool[IDX_APPLY_GLOBAL_WEIGHT_DECAY] && optim_float[1] > 0) {
        // not vbe and gwd
        TORCH_CHECK(false, "Global weight decay is not supported yet in SYCL backend");
    }

    if (static_cast<PoolingMode>(pooling_mode) == PoolingMode::NONE) {
        // no bag
        return SplitNoBagLookupFunction_rowwise_adagrad_Op_pt2::apply(
            placeholder_autograd_tensor,
            output_dtype,
            weights,
            max_D,
            hash_size_cumsum,
            total_hash_size_bits,
            indices,
            offsets, 
            aux_tensor,
            aux_int,
            aux_float,
            aux_bool,
            momentum1, 
            learning_rate_tensor, 
            optim_int, 
            optim_float
            )[0];
    }
    else {
        TORCH_CHECK(false, "split_embedding_codegen_lookup_rowwise_adagrad_function_pt2 is not implemented yet for pooled case in SYCL backend");
    } 
  }
} // namespace at::native::xpu
