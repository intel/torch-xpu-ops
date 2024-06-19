
#include <ATen/core/Reduction.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/xpu/sycl/LossNLLKernel.h>
#include <comm/RegisterUtils.h>
#include <comm/xpu_aten.h>

#include <ATen/xpu/ops/nll_loss_backward_native.h>
#include <ATen/xpu/ops/nll_loss_forward_native.h>

namespace at {
namespace native {
TORCH_IMPL_FUNC(nll_loss_forward_out_xpu)
(const Tensor& self,
 const Tensor& target,
 const OptionalTensorRef weight_opt,
 int64_t reduction,
 int64_t ignore_index,
 const Tensor& output,
 const Tensor& total_weight) {
  xpu::nll_loss_forward_kernel(
      self,
      target,
      ((weight_opt.has_value() && (*weight_opt).defined())
           ? at::OptionalTensorRef(*weight_opt)
           : at::OptionalTensorRef()),
      reduction,
      ignore_index,
      output,
      total_weight);
}

TORCH_IMPL_FUNC(nll_loss_backward_out_xpu)
(const Tensor& grad_output,
 const Tensor& self,
 const Tensor& target,
 OptionalTensorRef weight_opt,
 int64_t reduction,
 int64_t ignore_index,
 const Tensor& total_weight,
 const Tensor& grad_input) {
  const Tensor& weight = weight_opt.getTensorRef();
  grad_input.zero_();
  xpu::nll_loss_backward_kernel(
      grad_output,
      self,
      target,
      ((weight_opt.has_value() && (*weight_opt).defined())
           ? at::OptionalTensorRef(*weight_opt)
           : at::OptionalTensorRef()),
      reduction,
      ignore_index,
      total_weight,
      grad_input);
}

} // namespace native

// void nll_loss_forward_meta(
//     const Tensor& self,
//     const Tensor& target,
//     const OptionalTensorRef weight_opt,
//     int64_t reduction,
//     int64_t ignore_index,
//     Tensor& output,
//     Tensor& total_weight) {
//   const Tensor& weight = weight_opt.getTensorRef();

//   TORCH_CHECK(
//       self.dim() > 0 && self.dim() <= 2, "input tensor should be 1D or 2D");
//   TORCH_CHECK(
//       target.dim() <= 1,
//       "0D or 1D target tensor expected, multi-target not supported");

//   auto no_batch_dim = self.dim() == 1 && target.dim() == 0;
//   TORCH_CHECK(
//       no_batch_dim || (self.size(0) == target.size(0)),
//       "size mismatch (got input: ",
//       self.sizes(),
//       ", target: ",
//       target.sizes(),
//       ")")

//   const auto n_classes = self.size(-1);

//   TORCH_CHECK(
//       !weight.defined() || (weight.dim() == 1 && weight.numel() ==
//       n_classes), "weight tensor should be defined either for all ",
//       n_classes,
//       " classes or no classes"
//       " but got weight tensor of shape: ",
//       weight.sizes());

//   const auto n_dims = self.dim();
//   const auto batch_size = self.size(0);

//   if (reduction == Reduction::None && n_dims == 2) {
//     if (output.defined()) {
//       at::xpu::resize_out(output, {batch_size}, {}, self.options());
//     } else {
//       output = at::xpu::create_out({batch_size}, {}, self.options());
//     }
//   } else {
//     // produce scalar output when reducing or input is 1d
//     if (output.defined()) {
//       at::xpu::resize_out(output, {}, {}, self.options());
//     } else {
//       output = at::xpu::create_out({}, {}, self.options());
//     }
//   }
//   if (total_weight.defined()) {
//     at::xpu::resize_out(total_weight, {}, {}, self.options());
//   } else {
//     total_weight = at::xpu::create_out({}, {}, self.options());
//   }
// }

// std::tuple<Tensor&, Tensor&> XPUNativeFunctions::nll_loss_forward_out(
//     const Tensor& self,
//     const Tensor& target,
//     const c10::optional<Tensor>& weight,
//     int64_t reduction,
//     int64_t ignore_index,
//     Tensor& output,
//     Tensor& total_weight) {
//   std::optional<Device> common_device = std::nullopt;
//   c10::impl::check_and_update_common_device(
//       common_device, output, "xpu::nll_loss_forward_out", "output");
//   c10::impl::check_and_update_common_device(
//       common_device, total_weight, "xpu::nll_loss_forward_out",
//       "total_weight");
//   c10::impl::check_and_update_common_device(
//       common_device, self, "xpu::nll_loss_forward_out", "self");
//   c10::impl::check_and_update_common_device(
//       common_device, target, "xpu::nll_loss_forward_out", "target");
//   c10::impl::check_and_update_common_device(
//       common_device, weight, "xpu::nll_loss_forward_out", "weight");
//   nll_loss_forward_meta(
//       self,
//       target,
//       ((weight.has_value() && (*weight).defined())
//            ? at::OptionalTensorRef(*weight)
//            : at::OptionalTensorRef()),
//       reduction,
//       ignore_index,
//       output,
//       total_weight);
//   return native::xpu::nll_loss_forward_kernel(
//       self,
//       target,
//       ((weight.has_value() && (*weight).defined())
//            ? at::OptionalTensorRef(*weight)
//            : at::OptionalTensorRef()),
//       reduction,
//       ignore_index,
//       output,
//       total_weight);
// }

// std::tuple<Tensor, Tensor> XPUNativeFunctions::nll_loss_forward(
//     const Tensor& self,
//     const Tensor& target,
//     const c10::optional<Tensor>& weight,
//     int64_t reduction,
//     int64_t ignore_index) {
//   Tensor output;
//   Tensor total_weight;
//   return nll_loss_forward_out(
//       self, target, weight, reduction, ignore_index, output, total_weight);
// }

// void nll_loss_backward_meta(
//     const Tensor& grad_output,
//     const Tensor& self,
//     const Tensor& target,
//     OptionalTensorRef weight_opt,
//     int64_t reduction,
//     int64_t ignore_index,
//     const Tensor& total_weight,
//     Tensor& grad_input) {
//   TORCH_CHECK(
//       self.dim() > 0 && self.dim() <= 2, "input tensor should be 1D or 2D");
//   TORCH_CHECK(
//       target.dim() <= 1,
//       "0D or 1D target tensor expected, multi-target not supported");

//   auto no_batch_dim = self.dim() == 1 && target.dim() == 0;
//   TORCH_CHECK(
//       no_batch_dim || (self.size(0) == target.size(0)),
//       "size mismatch (got input: ",
//       self.sizes(),
//       ", target: ",
//       target.sizes(),
//       ")")
//   TORCH_CHECK(
//       total_weight.numel() == 1,
//       "expected total_weight to be a  single element tensor, got: ",
//       total_weight.sizes(),
//       " (",
//       total_weight.numel(),
//       " elements)");

//   const auto& weight = weight_opt.getTensorRef();

//   TORCH_CHECK(
//       !weight.defined() || weight.numel() == self.size(-1),
//       "weight tensor should be defined either for all or no classes");

//   const auto n_dims = self.dim();

//   if (reduction == Reduction::None && n_dims == 2) {
//     const auto batch_size = self.size(0);
//     check_dim_size(grad_output, 1, 0, batch_size);
//   } else {
//     TORCH_CHECK(
//         grad_output.dim() <= 1 && grad_output.numel() == 1,
//         "Expected a single element grad_output tensor, but got: ",
//         grad_output.sizes());
//   }
//   if (grad_input.defined()) {
//     at::xpu::resize_out(
//         grad_input,
//         self.sizes(),
//         {},
//         self.options().memory_format(LEGACY_CONTIGUOUS_MEMORY_FORMAT));
//   } else {
//     grad_input = at::xpu::create_out(
//         self.sizes(),
//         {},
//         self.options().memory_format(LEGACY_CONTIGUOUS_MEMORY_FORMAT));
//   }
// }

// Tensor& XPUNativeFunctions::nll_loss_backward_out(
//     const Tensor& grad_output,
//     const Tensor& self,
//     const Tensor& target,
//     const c10::optional<Tensor>& weight,
//     int64_t reduction,
//     int64_t ignore_index,
//     const Tensor& total_weight,
//     Tensor& grad_input) {
//   std::optional<Device> common_device = std::nullopt;
//   c10::impl::check_and_update_common_device(
//       common_device, grad_input, "xpu::nll_loss_backward_out", "grad_input");
//   c10::impl::check_and_update_common_device(
//       common_device, grad_output, "xpu::nll_loss_backward_out",
//       "grad_output");
//   c10::impl::check_and_update_common_device(
//       common_device, self, "xpu::nll_loss_backward_out", "self");
//   c10::impl::check_and_update_common_device(
//       common_device, target, "xpu::nll_loss_backward_out", "target");
//   c10::impl::check_and_update_common_device(
//       common_device, weight, "xpu::nll_loss_backward_out", "weight");
//   c10::impl::check_and_update_common_device(
//       common_device,
//       total_weight,
//       "xpu::nll_loss_backward_out",
//       "total_weight");
//   nll_loss_backward_meta(
//       grad_output,
//       self,
//       target,
//       ((weight.has_value() && (*weight).defined())
//            ? at::OptionalTensorRef(*weight)
//            : at::OptionalTensorRef()),
//       reduction,
//       ignore_index,
//       total_weight,
//       grad_input);
//   return native::xpu::nll_loss_backward_kernel(
//       grad_output,
//       self,
//       target,
//       ((weight.has_value() && (*weight).defined())
//            ? at::OptionalTensorRef(*weight)
//            : at::OptionalTensorRef()),
//       reduction,
//       ignore_index,
//       total_weight,
//       grad_input);
// }

// Tensor XPUNativeFunctions::nll_loss_backward(
//     const Tensor& grad_output,
//     const Tensor& self,
//     const Tensor& target,
//     const c10::optional<Tensor>& weight,
//     int64_t reduction,
//     int64_t ignore_index,
//     const Tensor& total_weight) {
//   Tensor grad_input;
//   return nll_loss_backward_out(
//       grad_output,
//       self,
//       target,
//       weight,
//       reduction,
//       ignore_index,
//       total_weight,
//       grad_input);
// }
} // namespace at