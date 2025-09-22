#include <ATen/ATen.h>
#include <ATen/DeviceGuard.h>
#include <ATen/record_function.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include <torch/torch.h>

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
} // namespace
