#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
#include <ATen/native/xpu/sycl/CrossKernel.h>
#include <ATen/xpu/XPUNativeFunctions.h>
#include <comm/RegisterUtils.h>

namespace at {
void linalg_cross_meta(
    const Tensor& input,
    const Tensor& other,
    int64_t dim,
    Tensor& output) {
  auto x_d = input.dim();
  auto y_d = other.dim();
  // This is to avoid things like
  // linalg.cross(torch.randn(2, 3), torch.randn(5, 2, 3), dim=2)
  TORCH_CHECK(
      x_d == y_d,
      "linalg.cross: inputs must have the same number of dimensions.");
  TORCH_CHECK(
      input.size(dim) == 3 && other.size(dim) == 3,
      "linalg.cross: inputs dimension ",
      dim,
      " must have length 3. Got ",
      input.size(dim),
      " and ",
      other.size(dim));

  // Broadcast the batch dimension of input and other.
  // Since the non-batch dimensions agree, this is the same as broadcast all the
  // inputs
  auto out_size = infer_size(input.sizes(), other.sizes());

  if (output.defined()) {
    at::xpu::resize_out(output, out_size, {}, input.options());
  } else {
    output = at::xpu::create_out(out_size, {}, input.options());
  }
}

Tensor& XPUNativeFunctions::linalg_cross_out(
    const Tensor& self,
    const Tensor& other,
    int64_t dim,
    Tensor& out) {
  linalg_cross_meta(self, other, dim, out);

  dim = maybe_wrap_dim(dim, self.dim());
  auto out_size = out.sizes();
  Tensor input_broadcasted = self.expand(out_size);
  Tensor other_broadcasted = other.expand(out_size);
  native::xpu::linalg_cross_kernel(
      out, input_broadcasted, other_broadcasted, dim);
  return out;
}

Tensor XPUNativeFunctions::linalg_cross(
    const Tensor& self,
    const Tensor& other,
    int64_t dim) {
  Tensor out;
  return linalg_cross_out(self, other, dim, out);
}
} // namespace at