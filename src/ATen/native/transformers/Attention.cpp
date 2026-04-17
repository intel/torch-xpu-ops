/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/NestedTensorImpl.h>
#include <ATen/ceil_div.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/nested/NestedTensorUtils.h>
#include <ATen/native/transformers/attention.h>
#include <ATen/native/transformers/sdp_utils_cpp.h>
#include <ATen/native/xpu/sycl/DropoutKernels.h>
#include <ATen/ops/_scaled_dot_product_attention_math.h>
#include <ATen/xpu/XPUGeneratorImpl.h>
#include <ATen/xpu/XPUGraphsUtils.h>
#include <c10/core/InferenceMode.h>
#include <torch/autograd.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_scaled_dot_product_efficient_attention_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/from_blob.h>
#include <ATen/ops/full.h>
#include <ATen/ops/linear.h>
#include <ATen/ops/scalar_tensor.h>
#include <ATen/ops/scaled_dot_product_attention.h>
#include <ATen/ops/split_native.h>
#endif

#include <ATen/native/transformers/SDPUtils.h>
#include <ATen/native/transformers/sycl/AttentionKernels.h>

#include <comm/SYCLContext.h>

namespace at {
namespace native {

// compute q = (q + q_bias) / sqrt(dim_per_head), k = k + k_bias, v = v + v_bias
// Note: current only support contiguous indexing, since nested tensor is all
// contiguous
std::tuple<Tensor, Tensor, Tensor> transform_bias_rescale_qkv_xpu(
    const Tensor& qkv,
    const Tensor& qkv_bias,
    const int64_t num_head) {
  // for nested tensor, B is most outer size, but T is not regular, it should be
  // the large size on dim1
  auto B = qkv.is_nested()
      ? native::get_nested_tensor_impl(qkv)->get_nested_sizes().size(0)
      : qkv.size(0);

  auto T = qkv.is_nested() ? native::NestedTensor_get_max_size(
                                 *native::get_nested_tensor_impl(qkv))[0]
                           : qkv.size(1);
  if (qkv.is_nested()) {
    // Don't mess with non-nested case for now since it's not set up to fiddle
    // with mask size.

    // Round T up to next multiple of 8 so as to be able to utilize Tensor
    // cores. Otherwise, sometimes with padding, *no* row will have the maximum
    // sequence length and so we'll have a non-divisible-by-8 dimension even if
    // the model author chose a multiple of 8.
    T = T + (8 - (T % 8)) % 8;
  }
  auto _3D = qkv_bias.size(0);
  auto D = _3D / 3;
  TORCH_CHECK(D % num_head == 0);
  const auto dim_per_head = D / num_head;

  // q_k_v B T 3D -> 3, B, num_head, T, dim_per_head
  auto q_k_v = at::empty({3, B, num_head, T, dim_per_head}, qkv_bias.options());

  xpu::_transform_bias_rescale_qkv_kernel(
      qkv, qkv_bias, num_head, q_k_v, B, T, D, dim_per_head);

  auto q_k_v_s =
      at::native::split(q_k_v.view({3 * B, num_head, T, dim_per_head}), B, 0);
  return std::make_tuple(q_k_v_s[0], q_k_v_s[1], q_k_v_s[2]);
}

static bool check_for_seq_len_1_nested_tensor(
    sdp::sdp_params params,
    bool debug) {
  // When this function is called we are assured that the nt is dim==4
  if (!params.query.is_nested()) {
    return true;
  }

  const auto nt_q_tensor_impl =
      at::native::get_nested_tensor_impl(params.query);
  const at::Tensor& sizes = nt_q_tensor_impl->get_nested_sizes();
  auto* sizes_ptr = sizes.data_ptr<int64_t>();
  const int64_t n_tensors = params.query.size(0);
  const int64_t size_tensor_stride = sizes.stride(0);

  // This is being called inside sdp with shape [batch, heads, {seq_len}, dim]
  for (const auto i : c10::irange(n_tensors)) {
    if (sizes_ptr[(i * size_tensor_stride) + 1] <= 1) {
      if (debug) {
        TORCH_WARN(
            "Packed projection for fused kernels does not support sequence_length <= 1");
      }
      return false;
    }
  }

  return true;
}

std::tuple<Tensor, Tensor> native_multi_head_attention_xpu(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const int64_t embed_dim,
    const int64_t num_head,
    const Tensor& qkv_weight,
    const Tensor& qkv_bias,
    const Tensor& proj_weight,
    const Tensor& proj_bias,
    const std::optional<Tensor>& mask,
    bool need_weights,
    bool average_attn_weights,
    const std::optional<int64_t> mask_type) {
  // query shape: [B, T, D]
  // qkv_weight shape: [3 * D, D]

  TORCH_CHECK(
      !mask || !query.is_nested(),
      "NestedTensor with mask is not supported yet");
  const auto D = embed_dim;
  TORCH_CHECK(
      query.dim() == 3, "expected 3-D `query`, got ", query.dim(), "-D tensor");
  TORCH_CHECK(
      query.is_nested() || query.sizes()[2] == embed_dim,
      "passed-in embed_dim ",
      embed_dim,
      " didn't match last dim of query ",
      query.sizes()[2]);
  TORCH_CHECK(
      key.dim() == 3, "expected 3-D `key`, got ", key.dim(), "-D tensor");
  TORCH_CHECK(
      value.dim() == 3, "expected 3-D `value`, got ", value.dim(), "-D tensor");
  TORCH_CHECK(
      query.is_nested() || key.is_nested() || value.is_nested() ||
          (query.sizes() == key.sizes() && key.sizes() == value.sizes()),
      "expected `query`/`key`/`value` shapes to match");
  TORCH_CHECK(
      qkv_weight.dim() == 2,
      "expected 2-D `qkv_weight`, got ",
      qkv_weight.dim(),
      "-D tensor");
  TORCH_CHECK(
      D * 3 == qkv_weight.sizes()[0],
      "expected `qkv_weight` first dim to be 3x embed_dim");
  TORCH_CHECK(
      D == qkv_weight.sizes()[1],
      "expected `qkv_weight` second dim to be embed_Dim");
  TORCH_CHECK(
      qkv_bias.dim() == 1,
      "expected 1-D `qkv_bias`, got ",
      qkv_bias.dim(),
      "-D tensor");
  TORCH_CHECK(
      qkv_bias.sizes()[0] == 3 * D,
      "expected `qkv_bias` first dim and first dim of query to be equal");
  TORCH_CHECK(
      D % num_head == 0, "`embed_dim` must divide evenly by `num_heads`");

#ifndef NDEBUG
  const auto B = query.is_nested()
      ? native::get_nested_tensor_impl(query)->get_nested_sizes().size(0)
      : query.sizes()[0];
  auto T = query.is_nested() ? 0 : query.sizes()[1];

#endif
  const auto dim_per_head = D / num_head;
  if ((query.is_same(key) && key.is_same(value)) && dim_per_head % 8 == 0 &&
      !need_weights) {
    // We have not done linear projection yet but the input for SDP
    // Is expected to be 4 dimensional. We "cheaply" create view tensors
    // That will then be used for checking hot path conditions with
    // select_sd_backend
    auto q =
        query.view({query.size(0), -1, num_head, dim_per_head}).transpose(1, 2);
    auto k =
        key.view({key.size(0), -1, num_head, dim_per_head}).transpose(1, 2);
    auto v =
        value.view({value.size(0), -1, num_head, dim_per_head}).transpose(1, 2);

    sdp::sdp_params kernel_params{q, k, v, mask, 0.0, false, false};

    sdp::SDPBackend backend = sdp::SDPBackend::math;
    if (_fused_sdp_choice_stub.is_device_supported(q.device().type())) {
      backend = static_cast<sdp::SDPBackend>(_fused_sdp_choice_stub(
          q.device().type(), q, k, v, mask, 0.0, false, std::nullopt, false));
    }

    // strides from packed projection for nested tensors when seq_len is 1 will
    // be and will trigger a contiguous call in the kernel, so we prevent this
    bool no_seq_len_1_nested = query.is_nested()
        ? check_for_seq_len_1_nested_tensor(kernel_params, false)
        : true;
    // The API for transformer_encoder is a mask of shape (Batch_Size,
    // Seq_len_q) For mem-eff attention this will cause the expand call to error
    // For now I am going to turn of that path not have to deal with all the
    // annoying Mask type shape grossness
    if (!mask.has_value() && no_seq_len_1_nested &&
        (backend == sdp::SDPBackend::flash_attention ||
         backend == sdp::SDPBackend::efficient_attention)) {
      auto x = at::linear(query, qkv_weight, qkv_bias);
      auto chunks = x.chunk(3, -1);
      auto x_size_0 = x.size(0);

      chunks[0] = (chunks[0].view({x_size_0, -1, num_head, dim_per_head}))
                      .transpose(1, 2);
      chunks[1] = (chunks[1].view({x_size_0, -1, num_head, dim_per_head}))
                      .transpose(1, 2);
      chunks[2] = (chunks[2].view({x_size_0, -1, num_head, dim_per_head}))
                      .transpose(1, 2);
      auto y = at::scaled_dot_product_attention(
          chunks[0], chunks[1], chunks[2], mask, 0.0, false, std::nullopt);

      auto past_sdp = y.transpose(1, 2).reshape({x_size_0, -1, embed_dim});
      return std::make_tuple(
          at::linear(past_sdp, proj_weight, proj_bias), Tensor());
    }
    // Returned math or error lets not use it
  }

  // shape: [B, T, 3 x D]
  auto qkv = native::qkv_projection(query, key, value, embed_dim, qkv_weight);

  if (!qkv.is_nested() && qkv.numel() == 0) {
    if (query.is_nested()) {
      return std::make_tuple(Tensor(), Tensor());
    }
    return std::make_tuple(at::empty_like(query), Tensor());
  }

#ifndef NDEBUG
  if (!query.is_nested() || !qkv.is_nested()) {
    if (query.is_nested()) {
      T = qkv.size(1);
    }
    native::debug_assert_shape(__LINE__, qkv, {B, T, 3 * D});
  }
#endif

#ifdef DEBUG_PRINT_EACH_STEP
  if (!qkv.is_nested()) {
    std::cerr << "qkv: " << qkv << std::endl;
  }
#endif
  // shape: 3 x [B, num_head, T, dim_per_head]
  auto q_k_v = transform_bias_rescale_qkv_xpu(qkv, qkv_bias, num_head);
  qkv = Tensor(); // Not used any more, allow free
  auto& q = std::get<0>(q_k_v);
  const auto& k = std::get<1>(q_k_v);
  const auto& v = std::get<2>(q_k_v);
#ifndef NDEBUG
  native::debug_assert_shape(__LINE__, q, {B, num_head, T, dim_per_head});
  native::debug_assert_shape(__LINE__, k, {B, num_head, T, dim_per_head});
  native::debug_assert_shape(__LINE__, v, {B, num_head, T, dim_per_head});
#endif
#ifdef DEBUG_PRINT_EACH_STEP
  std::cerr << "q: " << q << std::endl;
  std::cerr << "k: " << k << std::endl;
  std::cerr << "v: " << v << std::endl;
#endif

  // shape: [B, num_head, T, T]
  auto qkt = native::bmm_nt(q, k);
  // q & k are dead but cannot be freed because they were packed with v
#ifndef NDEBUG
  native::debug_assert_shape(__LINE__, qkt, {B, num_head, T, T});
#endif
#ifdef DEBUG_PRINT_EACH_STEP
  std::cerr << "qkt: " << qkt << std::endl;
#endif

  // shape: [B, num_head, T, T]
  // TODO: long-term, have a kernel that works with
  // NestedTensor directly if there is no mask passed
  qkt = native::masked_softmax(qkt, mask, query, mask_type);
#ifdef DEBUG_PRINT_EACH_STEP
  std::cerr << "qkt after softmax: " << qkt << std::endl;
#endif

  // shape: [B, num_head, T, dim_per_head]
  // reuse storage for q; we're done with it
  auto attn_ctx = native::bmm_nn(q, qkt, v);
  // qkv is not dead; we just reused storage for q!
  if (!need_weights) {
    qkt = Tensor();
  }
#ifndef NDEBUG
  native::debug_assert_shape(
      __LINE__, attn_ctx, {B, num_head, T, dim_per_head});
#endif
#ifdef DEBUG_PRINT_EACH_STEP
  std::cerr << "attn_ctx: " << attn_ctx << std::endl;
#endif

  // shape: [B, T, D]
  // Fuse transform_0213 inside
  auto proj = native::transform0213_gemm_nt_bias(
      attn_ctx, proj_weight, proj_bias, query);
#ifndef NDEBUG
  native::debug_assert_shape(__LINE__, proj, {B, T, D});
#endif
  if (need_weights && average_attn_weights) {
    // weights are not needed for full transformer, so don't worry too
    // much about performance -- we implement this just to make use
    // cases that don't disable need_weights still get some speedup.
    qkt = qkt.sum(1);
    qkt /= num_head;
  }
  return std::make_tuple(std::move(proj), std::move(qkt));
}

// Reproduces the dropout mask from a host-side seed/offset (normal path).
at::Tensor& _fill_mem_eff_dropout_mask_(
    Tensor& self,
    double dropout_p,
    const int64_t seed,
    const int64_t offset) {
  auto state = c10::make_intrusive<at::XPUGeneratorState>(
      static_cast<uint64_t>(seed), static_cast<uint64_t>(offset));
  auto gen = at::make_generator<at::XPUGeneratorImpl>(
      self.device().index(), std::move(state));
  auto mask =
      std::get<1>(xpu::fused_dropout_kernel(self, 1.0 - dropout_p, gen));
  self.copy_(mask);
  return self;
}

// Reproduces the dropout mask from device-side seed/offset tensors (graph
// capture path). Constructs a temporary generator whose extragraph tensors
// alias the provided device tensors so the SYCL kernel reads seed/offset
// directly from device memory during graph replay — no D2H transfer.
static at::Tensor& _fill_mem_eff_dropout_mask_from_device_tensors_(
    Tensor& self,
    double dropout_p,
    const Tensor& philox_seed_t,
    const Tensor& philox_offset_t) {
  auto state = c10::make_intrusive<at::XPUGeneratorState>();
  state->capturing_ = true;
  state->seed_extragraph_ = philox_seed_t;
  state->offset_extragraph_ = philox_offset_t;
  state->offset_intragraph_ = 0;
  auto gen = at::make_generator<at::XPUGeneratorImpl>(
      self.device().index(), std::move(state));
  auto mask =
      std::get<1>(xpu::fused_dropout_kernel(self, 1.0 - dropout_p, gen));
  self.copy_(mask);
  return self;
}

/**
 * Fall back implementation of efficient attention
 */
std::tuple<Tensor, Tensor, Tensor, Tensor>
_scaled_dot_product_efficient_attention_xpu(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const std::optional<at::Tensor>& attn_bias,
    bool compute_log_sumexp,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale) {
  // Used for tracking usage statistics
  C10_LOG_API_USAGE_ONCE("torch.sdpa.mem_efficient_attention");
  constexpr int64_t MAX_BATCH_SIZE = (1LL << 16) - 1;
  int64_t batch_size = query.size(0);

  if (batch_size > MAX_BATCH_SIZE) {
    TORCH_CHECK(
        dropout_p == 0.0,
        "Efficient attention cannot produce valid seed and offset outputs when "
        "the batch size exceeds (",
        MAX_BATCH_SIZE,
        ").");
  }

  // Snapshot the philox seed/offset *before* running the math kernel so that
  // the backward pass can replay the same dropout mask.
  // During XPU graph capture we cannot do host-side reads; instead we emit a
  // device-side copy of the generator's extragraph tensors into output scalar
  // tensors so each graph replay writes the correct values into them.
  Tensor philox_seed_tensor, philox_offset_tensor;
  if (dropout_p > 0.0) {
    auto gen = get_generator_or_default<at::XPUGeneratorImpl>(
        std::nullopt, at::xpu::detail::getDefaultXPUGenerator());
    if (at::xpu::currentStreamCaptureStatus() !=
        at::xpu::CaptureStatus::Executing) {
      // Graph capture path: output device tensors that alias the extragraph
      // seed/offset buffers, updated on each replay by replay_prologue().
      philox_seed_tensor = at::empty({1}, query.options().dtype(at::kLong));
      philox_offset_tensor = at::empty({1}, query.options().dtype(at::kLong));
      PhiloxXpuState pstate;
      {
        std::lock_guard<std::mutex> lock(gen->mutex_);
        pstate = gen->philox_xpu_state(0);
      }
      auto dev_opts =
          at::TensorOptions().dtype(at::kLong).device(query.device());
      at::Tensor seed_eg = at::from_blob(pstate.seed_.ptr, {1}, dev_opts);
      at::Tensor offset_eg = at::from_blob(pstate.offset_.ptr, {1}, dev_opts);
      philox_seed_tensor.copy_(seed_eg);
      philox_offset_tensor.copy_(offset_eg);
      philox_offset_tensor.add_(
          static_cast<int64_t>(pstate.offset_intragraph_));
    } else {
      // Normal path: snapshot host-side seed/offset.
      std::lock_guard<std::mutex> lock(gen->mutex_);
      philox_seed_tensor = at::scalar_tensor(
          static_cast<int64_t>(gen->current_seed()),
          query.options().dtype(at::kLong));
      philox_offset_tensor = at::scalar_tensor(
          static_cast<int64_t>(gen->philox_offset_per_thread()),
          query.options().dtype(at::kLong));
    }
  } else {
    philox_seed_tensor =
        at::scalar_tensor(int64_t(0), query.options().dtype(at::kLong));
    philox_offset_tensor =
        at::scalar_tensor(int64_t(0), query.options().dtype(at::kLong));
  }

  auto res = at::_scaled_dot_product_attention_math(
      query,
      key,
      value,
      attn_bias,
      dropout_p,
      is_causal,
      std::nullopt, /*dropout_mask*/
      scale,
      true);
  auto attention = std::get<0>(res);
  // logsumexp is padded along the query dimension to kAlignLSE
  // so that backward kernels can perform vectorized loads safely.
  // This matches the contract expected by memory-efficient attention kernels.
  // Align with CUDA: kAlignLSE = 32.
  constexpr int64_t kAlignLSE = 32;
  int64_t B = query.size(0);
  int64_t H = query.size(1);
  int64_t L = query.size(2);
  Tensor out =
      attention.permute({0, 2, 1, 3}).contiguous().permute({0, 2, 1, 3});
  return std::make_tuple(
      out,
      at::full(
          {B, H, (compute_log_sumexp ? ceil_div(L, kAlignLSE) * kAlignLSE : 0)},
          0.0,
          attention.options()),
      std::move(philox_seed_tensor),
      std::move(philox_offset_tensor));
}

/**
 * Fall back implementation of efficient attention backward.
 * Since the forward path uses _scaled_dot_product_attention_math (which is
 * fully differentiable), we re-run the math forward with autograd enabled
 * and use torch::autograd::grad to compute the gradients for query, key,
 * value (and optionally attn_bias).
 */
std::tuple<Tensor, Tensor, Tensor, Tensor>
_scaled_dot_product_efficient_attention_backward_xpu(
    const Tensor& grad_out_,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& attn_bias,
    const Tensor& out,
    const Tensor& logsumexp,
    const Tensor& philox_seed,
    const Tensor& philox_offset,
    double dropout_p,
    std::array<bool, 4> grad_input_mask,
    bool is_causal,
    std::optional<double> scale) {
  if (!grad_out_.defined()) {
    return std::make_tuple(Tensor{}, Tensor{}, Tensor{}, Tensor{});
  }

  // Not used in this fallback path; the re-run forward recomputes attention
  // from scratch rather than reading saved outputs or log-sum-exp.
  (void)out;
  (void)logsumexp;

  // Nothing to differentiate — return empty gradients immediately.
  bool any_grad_needed = grad_input_mask[0] || grad_input_mask[1] ||
      grad_input_mask[2] || (grad_input_mask[3] && attn_bias.defined());
  if (!any_grad_needed) {
    return std::make_tuple(Tensor{}, Tensor{}, Tensor{}, Tensor{});
  }

  // Detach the inputs so we control which ones participate in the autograd
  // graph.  detach() returns a view sharing storage — no copy is made.
  // requires_grad_() is then set only for inputs whose grad_input_mask bit
  // is true; inputs that do not need a gradient are passed as plain detached
  // views and are never added to the `inputs` list for autograd::grad.
  auto q = query.detach();
  auto k = key.detach();
  auto v = value.detach();
  if (grad_input_mask[0])
    q.requires_grad_(true);
  if (grad_input_mask[1])
    k.requires_grad_(true);
  if (grad_input_mask[2])
    v.requires_grad_(true);

  TORCH_CHECK(
      !grad_input_mask[3] || attn_bias.defined(),
      "bias_requires_grad is true but no bias was provided");

  std::optional<Tensor> attn_bias_opt;
  Tensor ab;
  if (attn_bias.defined()) {
    ab = attn_bias.detach();
    if (grad_input_mask[3])
      ab.requires_grad_(true);
    attn_bias_opt = ab;
  }

  // When dropout was used in the forward pass, rebuild the exact same mask
  // using the captured philox seed/offset via a temporary generator.
  std::optional<Tensor> dropout_mask_opt;
  if (dropout_p > 0.0 && philox_seed.defined() && philox_offset.defined()) {
    int64_t B = query.size(0);
    int64_t H = query.size(1);
    int64_t L_q = query.size(2);
    int64_t L_k = key.size(2);
    Tensor mask = at::empty(
        {B, H, L_q, L_k},
        query.options().dtype(at::kFloat).device(query.device()));
    if (at::xpu::currentStreamCaptureStatus() !=
        at::xpu::CaptureStatus::Executing) {
      // Graph capture path: philox_seed/offset are XPU device tensors.
      _fill_mem_eff_dropout_mask_from_device_tensors_(
          mask, dropout_p, philox_seed, philox_offset);
    } else {
      at::native::_fill_mem_eff_dropout_mask_(
          mask,
          dropout_p,
          philox_seed.item<int64_t>(),
          philox_offset.item<int64_t>());
    }
    dropout_mask_opt = std::move(mask);
  }

  Tensor attention;
  {
    // The autograd engine executes backward nodes with the GraphTask's saved
    // ThreadLocalState restored.  That state was captured during the forward
    // pass, at which point VariableType had installed an
    // AutoDispatchBelowADInplaceOrView guard.  That guard adds the autograd
    // dispatch keys to the TLS *excluded* set.  When the engine restores this
    // state, all autograd keys stay excluded for the entire backward node —
    // even if AutoGradMode(true) is set.  IncludeDispatchKeyGuard only
    // modifies the *included* set and cannot override the excluded set.
    //
    // The correct fix mirrors what InferenceMode(false) does: remove the
    // autograd keys from the excluded set directly, then re-enable grad_mode.
    // We use _force_tls_local_dispatch_key_set for atomic RAII replacement.
    at::AutoGradMode enable_grad(true);
    auto saved_ks = c10::impl::tls_local_dispatch_key_set();
    c10::impl::PODLocalDispatchKeySet new_ks{};
    new_ks.set_included(saved_ks.included_);
    new_ks.set_excluded(saved_ks.excluded_ - c10::autograd_dispatch_keyset);
    c10::impl::_force_tls_local_dispatch_key_set(new_ks);
    struct RestoreKS {
      c10::impl::LocalDispatchKeySet saved;
      ~RestoreKS() {
        c10::impl::_force_tls_local_dispatch_key_set(saved);
      }
    } restore_ks{saved_ks};

    // Re-run the forward with the reproduced mask. We must pass the actual
    // dropout_p (not 0.0) so that _scaled_dot_product_attention_math applies
    // the correct scaling factor 1/(1-dropout_p) alongside the mask, matching
    // exactly what the original forward computed.
    auto res = at::_scaled_dot_product_attention_math(
        q,
        k,
        v,
        attn_bias_opt,
        dropout_mask_opt.has_value() ? dropout_p : 0.0,
        is_causal,
        dropout_mask_opt,
        scale);
    attention = std::get<0>(res);
  }

  TORCH_INTERNAL_ASSERT(
      attention.requires_grad(),
      "_scaled_dot_product_efficient_attention_backward_xpu: "
      "re-run of _scaled_dot_product_attention_math did not produce a tensor "
      "with requires_grad=True. This is a bug — the autograd graph was not "
      "recorded despite removing autograd keys from the excluded dispatch key set.");

  // Build the inputs list only from tensors that need gradients so that
  // autograd::grad does not compute — or error on — unrequested gradients.
  std::vector<Tensor> inputs;
  if (grad_input_mask[0])
    inputs.push_back(q);
  if (grad_input_mask[1])
    inputs.push_back(k);
  if (grad_input_mask[2])
    inputs.push_back(v);
  if (grad_input_mask[3] && ab.defined())
    inputs.push_back(ab);

  Tensor grad_q, grad_k, grad_v, grad_bias;
  auto grads = torch::autograd::grad(
      {attention}, inputs, {grad_out_}, /*retain_graph=*/false);

  int idx = 0;
  if (grad_input_mask[0])
    grad_q = grads[idx++];
  if (grad_input_mask[1])
    grad_k = grads[idx++];
  if (grad_input_mask[2])
    grad_v = grads[idx++];
  if (grad_input_mask[3] && ab.defined())
    grad_bias = grads[idx++];

  return std::make_tuple(
      std::move(grad_q),
      std::move(grad_k),
      std::move(grad_v),
      std::move(grad_bias));
}

} // namespace native
} // namespace at
