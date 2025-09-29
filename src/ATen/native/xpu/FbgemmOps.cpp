#include <ATen/ATen.h>
#include <ATen/DeviceGuard.h>
#include <ATen/record_function.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include <torch/torch.h>

#include <ATen/native/xpu/ScanKernels.h>
#include <ATen/native/xpu/sycl/FbgemmKernels.h>

namespace at {
namespace native {
namespace xpu {

#define XPU_DEVICE_GUARD(TENSOR) \
  const OptionalDeviceGuard device_guard(device_of(TENSOR));

Tensor asynchronous_complete_cumsum_xpu(const Tensor& t_in) {
  TORCH_CHECK(t_in.is_contiguous());
  TORCH_CHECK(t_in.dtype() == at::kInt || t_in.dtype() == at::kLong);
  TORCH_CHECK(t_in.dim() == 1 || t_in.dim() == 2);

  if (t_in.dim() == 1) {
    Tensor t_out = at::zeros({t_in.numel() + 1}, t_in.options());
    auto r_out = t_out.slice(0, 1);
    at::cumsum_out(r_out, t_in, 0);
    return t_out;
  }

  Tensor t_out = at::zeros({t_in.size(0), t_in.size(1) + 1}, t_in.options());
  auto r_out = t_out.slice(1, 1);
  at::cumsum_out(r_out, t_in, 1);
  return t_out;
}

Tensor dense_to_jagged_forward_xpu(
    const Tensor& dense,
    const std::vector<Tensor>& offsets,
    std::optional<at::SymInt> total_L) {
  TORCH_CHECK(dense.is_xpu(), "value must be a xpu tensor");
  for (auto& offset : offsets) {
    TORCH_CHECK(offset.is_xpu(), "offset must be a xpu tensor");
  }

  const int num_jagged_dim = dense.dim() - 2;
  TORCH_CHECK(
      offsets.size() == static_cast<size_t>(num_jagged_dim),
      "x_offsets.size(), ",
      offsets.size(),
      " != num_jagged_dim, ",
      num_jagged_dim);

  // D is the embedding dimension
  auto D = dense.size(-1);

  // If total_L is not given then compute it
  at::SymInt total_L_computed;
  if (total_L.has_value()) {
    total_L_computed = total_L.value();
  } else {
    total_L_computed = (int64_t)offsets.back().max().item<int64_t>();
  }
  auto values = at::empty_symint({total_L_computed, D}, dense.options());
  auto output = at::empty_like(values); // not used

  if (dense.numel() == 0 || values.numel() == 0) {
    return output;
  }

  XPU_DEVICE_GUARD(dense);

  dense_to_jagged_forward_xpu_kernel(values, offsets, dense, output);

  return output;
}

Tensor jagged_to_padded_dense_forward_xpu(
    const Tensor& values,
    const std::vector<Tensor>& offsets,
    c10::SymIntArrayRef max_lengths,
    const double padding_value) {
  size_t num_jagged_dim = offsets.size();
  TORCH_CHECK(
      max_lengths.size() == num_jagged_dim,
      "max_lengths.size(), ",
      max_lengths.size(),
      " != num_jagged_dim, ",
      num_jagged_dim);

  TORCH_CHECK(values.is_xpu(), "value must be a xpu tensor");
  for (auto& offset : offsets) {
    TORCH_CHECK(offset.is_xpu(), "offset must be a xpu tensor");
  }

  XPU_DEVICE_GUARD(values);

  const Tensor values_canonicalized = values.view(
      {values.size(0),
       std::accumulate(
           values.sizes().begin() + 1,
           values.sizes().end(),
           1,
           std::multiplies<size_t>())});
  at::SymDimVector padded_values_shape({at::SymInt(offsets[0].size(0) - 1)});
  padded_values_shape.insert(
      padded_values_shape.end(), max_lengths.begin(), max_lengths.end());

  // Canonicalize padded_values by unsqueeze the last dim if the inner dense
  // dimension is 1 and folded.
  const bool D_folded = values.dim() == 1;
  if (!D_folded) {
    padded_values_shape.push_back(values.size(-1));
  }
  Tensor padded_values =
      at::empty_symint(padded_values_shape, values.options());
  Tensor padded_values_view =
      D_folded ? padded_values.unsqueeze(-1) : padded_values;

  num_jagged_dim = padded_values_view.dim() - 2;
  TORCH_CHECK(
      offsets.size() == static_cast<size_t>(num_jagged_dim),
      "x_offsets.size(), ",
      offsets.size(),
      " != num_jagged_dim ",
      num_jagged_dim);

  if (padded_values_view.numel() == 0) {
    return padded_values;
  }

  jagged_to_padded_dense_forward_xpu_kernel(
      values_canonicalized,
      offsets,
      padded_values_view,
      padded_values_view,
      padding_value);

  return padded_values;
}

class DenseToJaggedOp : public torch::autograd::Function<DenseToJaggedOp> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const Tensor& dense,
      const std::vector<Tensor>& offsets,
      const std::optional<at::SymInt>& total_L) {
    // uncomment when implement backward

    // dims of dense tensor: <batch, [maxlen0, maxlen1, ...], embedding_dim>
    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("fbgemm::dense_to_jagged_forward", "")
            .typed<Tensor(
                const Tensor& dense,
                const std::vector<Tensor>& offsets,
                std::optional<at::SymInt> total_L)>();
    auto output = op.call(dense, offsets, total_L);

    return {output};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    // TODO: backward kernel
    return {
        torch::autograd::Variable(),
        torch::autograd::Variable(), // offsets
        torch::autograd::Variable() // total_L
    };
  }
};

// output = x + y where x is jagged, y is dense, and output is jagged
std::tuple<Tensor, std::vector<Tensor>> dense_to_jagged(
    const Tensor& dense,
    const std::vector<Tensor>& offsets,
    std::optional<at::SymInt> total_L) {
  return {DenseToJaggedOp::apply(dense, offsets, total_L)[0], offsets};
}

class JaggedToPaddedDenseOp
    : public torch::autograd::Function<JaggedToPaddedDenseOp> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const Tensor& values,
      const std::vector<Tensor>& offsets,
      const c10::SymIntArrayRef max_lengths,
      const double padding_value) {
    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("fbgemm::jagged_to_padded_dense_forward", "")
            .typed<at::Tensor(
                const Tensor& values,
                const std::vector<Tensor>& offsets,
                at::ArrayRef<at::SymInt> max_lengths,
                const double padding_value)>();
    Tensor padded_values = op.call(values, offsets, max_lengths, padding_value);

    return {padded_values};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    // TODO: backward kernel
    return {
        torch::autograd::Variable(),
        torch::autograd::Variable(), // offsets
        torch::autograd::Variable(), // max_lengths
        torch::autograd::Variable(), // padding_value
    };
  }
};

Tensor jagged_to_padded_dense(
    const Tensor& values,
    const std::vector<Tensor>& offsets,
    const c10::SymIntArrayRef max_lengths,
    const double padding_value = 0.0) {
  return JaggedToPaddedDenseOp::apply(
      values, offsets, max_lengths, padding_value)[0];
}

class JaggedDenseAddJaggedOutputOp
    : public torch::autograd::Function<JaggedDenseAddJaggedOutputOp> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const Tensor& x_values,
      const std::vector<Tensor>& offsets,
      const Tensor& dense) {
    TORCH_CHECK(x_values.is_xpu(), "value must be a xpu tensor");
    for (auto& offset : offsets) {
      TORCH_CHECK(offset.is_xpu(), "offset must be a xpu tensor");
    }
    TORCH_CHECK(dense.is_xpu(), "dense must be a xpu tensor");

    const int num_jagged_dim = dense.dim() - 2;
    TORCH_CHECK(
        offsets.size() == static_cast<size_t>(num_jagged_dim),
        "x_offsets.size(), ",
        offsets.size(),
        " != num_jagged_dim, ",
        num_jagged_dim);

    auto output = at::empty_like(x_values);
    if (dense.numel() == 0 || x_values.numel() == 0) {
      return {output};
    }

    XPU_DEVICE_GUARD(dense);
    jagged_dense_elementwise_add_jagged_output_fwd_xpu_kn(
        x_values, offsets, dense, output);

    return {output};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    // TODO: backward kernel
    return {
        torch::autograd::Variable(),
        torch::autograd::Variable(), // offsets
        torch::autograd::Variable()};
  }
};

std::tuple<Tensor, std::vector<Tensor>>
jagged_dense_elementwise_add_jagged_output_xpu(
    const Tensor& x_values,
    const std::vector<Tensor>& x_offsets,
    const Tensor& y) {
  auto sum_values =
      JaggedDenseAddJaggedOutputOp::apply(x_values, x_offsets, y)[0];

  return {sum_values, x_offsets};
}

Tensor reorder_batched_ad_lengths_xpu(
    const Tensor& cat_ad_lengths,
    const Tensor& batch_offsets,
    const int64_t num_ads_in_batch,
    const bool broadcast_lengths,
    const int64_t max_batch_size = 0) {
  TORCH_CHECK_LE(max_batch_size, 0);
  TENSORS_ON_SAME_XPU_IF_NOT_OPTIONAL(cat_ad_lengths, batch_offsets);

  XPU_DEVICE_GUARD(cat_ad_lengths);

  const int64_t B = batch_offsets.numel() - 1;
  const int64_t T = broadcast_lengths
      ? cat_ad_lengths.numel() / B
      : cat_ad_lengths.numel() / num_ads_in_batch;

  Tensor reordered_cat_ad_lengths = broadcast_lengths
      ? at::empty({T * num_ads_in_batch}, cat_ad_lengths.options())
      : at::empty_like(cat_ad_lengths);

  const int64_t grid_size = (B * T + 32 - 1) / 32;
  TORCH_CHECK(
      grid_size >= 0,
      "grid_size must be positive, got ",
      grid_size,
      " where B =",
      B,
      " and T =",
      T);

  reorder_batched_ad_lengths_xpu_kernel(
      cat_ad_lengths,
      batch_offsets,
      reordered_cat_ad_lengths,
      T,
      broadcast_lengths,
      grid_size);

  return reordered_cat_ad_lengths;
}

Tensor reorder_batched_ad_indices_xpu(
    const at::Tensor& cat_ad_offsets,
    const at::Tensor& cat_ad_indices,
    const at::Tensor& reordered_cat_ad_offsets,
    const at::Tensor& batch_offsets,
    const int64_t num_ads_in_batch,
    const bool broadcast_indices = false,
    const int64_t num_indices_after_broadcast = -1) {
  TENSORS_ON_SAME_XPU_IF_NOT_OPTIONAL(
      cat_ad_offsets, cat_ad_indices, reordered_cat_ad_offsets, batch_offsets);

  XPU_DEVICE_GUARD(cat_ad_offsets);

  const int64_t B = batch_offsets.numel() - 1;
  const int64_t T = (reordered_cat_ad_offsets.numel() - 1) / num_ads_in_batch;
  Tensor reordered_cat_ad_indices;
  if (broadcast_indices) {
    TORCH_CHECK_GE(num_indices_after_broadcast, 0);
    reordered_cat_ad_indices =
        at::empty({num_indices_after_broadcast}, cat_ad_indices.options());
  } else {
    reordered_cat_ad_indices = at::empty_like(cat_ad_indices);
  }

  reorder_batched_ad_indices_xpu_kernel(
      cat_ad_offsets,
      cat_ad_indices,
      reordered_cat_ad_offsets,
      batch_offsets,
      reordered_cat_ad_indices,
      num_ads_in_batch,
      B,
      T,
      broadcast_indices);

  return reordered_cat_ad_indices;
}

Tensor asynchronous_exclusive_cumsum_(const Tensor& t_in) {
  torch_tensor_on_xpu_check(t_in);
  XPU_DEVICE_GUARD(t_in);

  if (t_in.numel() == 0) {
    return at::empty_like(t_in);
  }

  TORCH_CHECK(t_in.is_contiguous());
  TORCH_CHECK(t_in.dtype() == at::kInt || t_in.dtype() == at::kLong);
  // only handles up to INT_MAX elements.
  TORCH_CHECK(t_in.numel() < std::numeric_limits<int32_t>::max());
  auto t_in_flatten = t_in.flatten();
  auto t_out = at::empty_like(t_in_flatten);

  cumsum_kernel(t_out, t_in_flatten, 0);

  // make it exclusive
  t_out = t_out.roll(1, 0);
  // set all first elemnts 0
  t_out[0] = 0;
  return t_out.view_as(t_in);
}

std::tuple<at::Tensor, at::Tensor, std::optional<at::Tensor>>
permute_2D_sparse_data_xpu(
    const at::Tensor& permute,
    const at::Tensor& lengths,
    const at::Tensor& indices,
    const std::optional<at::Tensor>& weights,
    const std::optional<int64_t>& permuted_lengths_sum) {
  TENSORS_ON_SAME_XPU_IF_NOT_OPTIONAL(permute, lengths, indices, weights);
  TORCH_CHECK(lengths.dim() == 2);

  XPU_DEVICE_GUARD(indices);

  const auto permute_contig = permute.contiguous();
  const auto lengths_contig = lengths.contiguous();
  const auto indices_contig = indices.contiguous();
  // the data to permute over can be less or more with or without
  // repetitions
  const auto T = permute.numel();
  const auto B = lengths.size(1);

  if (T == 0 || B == 0) {
    // When T = 0 or B = 0, permutation will not be performed.  Return the
    // input tensors.
    return {
        lengths.clone(),
        indices.clone(),
        weights.has_value() ? std::make_optional(weights->clone())
                            : std::nullopt};
  }

  Tensor permuted_lengths = at::empty({T, B}, lengths.options());
  Tensor permuted_indices;
  Tensor permuted_weights;

  permute_2D_lengths_kernel_xpu(
      T, B, lengths_contig, permute_contig, permuted_lengths);

  // convert lengths to offsets
  const auto input_offsets = asynchronous_exclusive_cumsum_(lengths_contig);
  const auto output_offsets =
      asynchronous_complete_cumsum_xpu(permuted_lengths.flatten());
  int64_t permuted_indices_size = 0;
  if (permuted_lengths_sum.has_value()) {
    permuted_indices_size = permuted_lengths_sum.value();
  } else {
    permuted_indices_size = output_offsets[-1].item<int64_t>();
  }

  permuted_indices = at::empty(permuted_indices_size, indices.options());

  if (weights.has_value()) {
    const Tensor weights_value = weights.value();
    int32_t weights_columns = 1;
    if (weights_value.dense_dim() > 1) {
      weights_columns = weights_value.size(1);
      permuted_weights = at::empty(
          {permuted_indices_size, weights_columns}, weights_value.options());
    } else {
      permuted_weights =
          at::empty(permuted_indices_size, weights_value.options());
    }
    permute_2D_data_kernel_xpu(
        permuted_indices_size,
        T,
        B,
        indices_contig,
        std::optional<const at::Tensor>{weights_value},
        weights_columns,
        permute_contig,
        input_offsets,
        output_offsets,
        permuted_indices,
        std::optional<at::Tensor>{permuted_weights});
  } else {
    permute_2D_data_kernel_xpu(
        permuted_indices_size,
        T,
        B,
        indices_contig,
        std::nullopt,
        0,
        permute_contig,
        input_offsets,
        output_offsets,
        permuted_indices,
        std::nullopt);
  }

  return {permuted_lengths, permuted_indices, permuted_weights};
}

} // namespace xpu
} // namespace native
} // namespace at

namespace {

TORCH_LIBRARY(fbgemm, m) {
  m.def("asynchronous_complete_cumsum(Tensor t_in) -> Tensor");
  m.def(
      "dense_to_jagged(Tensor dense, Tensor[] offsets, SymInt? total_L=None) -> (Tensor, Tensor[])");
  m.def(
      "dense_to_jagged_forward(Tensor dense, Tensor[] offsets, SymInt? total_L=None) -> Tensor");
  m.def(
      "jagged_to_padded_dense(Tensor values, Tensor[] offsets, SymInt[] max, float padding_value=0.0) -> Tensor");
  m.def(
      "jagged_to_padded_dense_forward(Tensor values, Tensor[] offsets, SymInt[] max, float padding_value=0.0) -> Tensor");
  m.def(
      "jagged_dense_elementwise_add_jagged_output(Tensor values, Tensor[] offsets, Tensor y) -> (Tensor, Tensor[])");
  m.def(
      "reorder_batched_ad_lengths(Tensor cat_ad_lengths, Tensor batch_offsets, int num_ads_in_batch, bool broadcast_lengths, int max_batch_size=0) -> Tensor");
  m.def(
      "reorder_batched_ad_indices(Tensor cat_ad_offsets, Tensor cat_ad_indices, Tensor reordered_cat_ad_offsets, Tensor batch_offsets, int num_ads_in_batch, bool broadcast_indices, int num_indices_after_broadcast) -> Tensor");
  m.def(
      "permute_2D_sparse_data(Tensor permute, Tensor lengths, Tensor indices, Tensor? weights=None, int? permuted_lengths_sum=None) -> (Tensor, Tensor, Tensor?)");
}

TORCH_LIBRARY_IMPL(fbgemm, XPU, m) {
  m.impl(
      "asynchronous_complete_cumsum",
      &at::native::xpu::asynchronous_complete_cumsum_xpu);
}

// Autograd backend register in fbgemm
TORCH_LIBRARY_IMPL(fbgemm, XPU, m) {
  m.impl("dense_to_jagged", &at::native::xpu::dense_to_jagged);
}

TORCH_LIBRARY_IMPL(fbgemm, XPU, m) {
  m.impl(
      "dense_to_jagged_forward", &at::native::xpu::dense_to_jagged_forward_xpu);
}

// Autograd backend register in fbgemm
TORCH_LIBRARY_IMPL(fbgemm, XPU, m) {
  m.impl("jagged_to_padded_dense", &at::native::xpu::jagged_to_padded_dense);
}

TORCH_LIBRARY_IMPL(fbgemm, XPU, m) {
  m.impl(
      "jagged_to_padded_dense_forward",
      &at::native::xpu::jagged_to_padded_dense_forward_xpu);
}

TORCH_LIBRARY_IMPL(fbgemm, XPU, m) {
  m.impl(
      "jagged_dense_elementwise_add_jagged_output",
      &at::native::xpu::jagged_dense_elementwise_add_jagged_output_xpu);
}

TORCH_LIBRARY_IMPL(fbgemm, XPU, m) {
  m.impl(
      "reorder_batched_ad_lengths",
      &at::native::xpu::reorder_batched_ad_lengths_xpu);
}

TORCH_LIBRARY_IMPL(fbgemm, XPU, m) {
  m.impl(
      "reorder_batched_ad_indices",
      &at::native::xpu::reorder_batched_ad_indices_xpu);
}

TORCH_LIBRARY_IMPL(fbgemm, XPU, m) {
  m.impl(
      "permute_2D_sparse_data", &at::native::xpu::permute_2D_sparse_data_xpu);
}
} // namespace
