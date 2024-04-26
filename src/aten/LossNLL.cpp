#include <ATen/ATen.h>
#include <ATen/XPUNativeFunctions.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/op_registration/adaption.h>
#include <comm/RegisterUtils.h>
#include <aten/sycl/LossNLLKernel.h>

namespace at {
void torch_meta_func_nll_loss_forward (const Tensor& self,
    const Tensor& target,
    const OptionalTensorRef weight_opt,
    int64_t reduction,
    int64_t ignore_index) {
  const Tensor& weight = weight_opt.getTensorRef();

  TORCH_CHECK(
      self.dim() > 0 && self.dim() <= 2, "input tensor should be 1D or 2D");
  TORCH_CHECK(
      target.dim() <= 1,
      "0D or 1D target tensor expected, multi-target not supported");

  auto no_batch_dim = self.dim() == 1  && target.dim() == 0;
  TORCH_CHECK(
      no_batch_dim || (self.size(0) == target.size(0)),
      "size mismatch (got input: ",
      self.sizes(),
      ", target: ",
      target.sizes(),
      ")")

  const auto n_classes = self.size(-1);

  TORCH_CHECK(
      !weight.defined() || (weight.dim() == 1 && weight.numel() == n_classes),
      "weight tensor should be defined either for all ",
      n_classes,
      " classes or no classes"
      " but got weight tensor of shape: ",
      weight.sizes());

  const auto n_dims = self.dim();
  const auto batch_size = self.size(0);

  if (reduction == Reduction::None && n_dims == 2) {
    set_output_raw_strided(0, {batch_size}, {}, self.options());
  } else {
    // produce scalar output when reducing or input is 1d
    set_output_raw_strided(0, {}, {}, self.options());
  }

  set_output_raw_strided(1, {}, {}, self.options());
}

std::tuple<Tensor &,Tensor &> nll_loss_forward_out(const Tensor & self, const Tensor & target, const c10::optional<Tensor> & weight, int64_t reduction, c10::SymInt ignore_index, Tensor & output, Tensor & total_weight) {

}

void torch_meta_func_nll_loss_backward(const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    OptionalTensorRef weight_opt,
    int64_t reduction,
    int64_t ignore_index,
    const Tensor& total_weight) {
  TORCH_CHECK(
      self.dim() > 0 && self.dim() <= 2, "input tensor should be 1D or 2D");
  TORCH_CHECK(
      target.dim() <= 1,
      "0D or 1D target tensor expected, multi-target not supported");

  auto no_batch_dim = self.dim() == 1  && target.dim() == 0;
  TORCH_CHECK(
      no_batch_dim || (self.size(0) == target.size(0)),
      "size mismatch (got input: ",
      self.sizes(),
      ", target: ",
      target.sizes(),
      ")")
  TORCH_CHECK(
      total_weight.numel() == 1,
      "expected total_weight to be a  single element tensor, got: ",
      total_weight.sizes(),
      " (",
      total_weight.numel(),
      " elements)");

  const auto& weight = weight_opt.getTensorRef();

  TORCH_CHECK(
      !weight.defined() || weight.numel() == self.size(-1),
      "weight tensor should be defined either for all or no classes");

  const auto n_dims = self.dim();

  if (reduction == Reduction::None && n_dims == 2) {
    const auto batch_size = self.size(0);
    check_dim_size(grad_output, 1, 0, batch_size);
  } else {
    TORCH_CHECK(
        grad_output.dim() <= 1 && grad_output.numel() == 1,
        "Expected a single element grad_output tensor, but got: ",
        grad_output.sizes());
  }

  set_output_raw_strided(0, self.sizes(), {}, self.options().memory_format(LEGACY_CONTIGUOUS_MEMORY_FORMAT));
}

Tensor & nll_loss_backward_out(const Tensor & grad_output, const Tensor & self, const Tensor & target, const c10::optional<Tensor> & weight, int64_t reduction, c10::SymInt ignore_index, const Tensor & total_weight, Tensor & grad_input) {

}
} // namespace at