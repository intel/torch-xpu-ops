#include <ATen/ScalarOps.h>
#include <ATen/native/xpu/sycl/ActivationGluKernels.h>
#include <ATen/xpu/XPUNativeFunctions.h>

namespace at {

void check_even_dimension(const Tensor& self, int64_t dim) {
  // this can't pass anyway because a 0-dimensional tensor has "size" 1, which
  // can't be evenly halved, but give a nicer error message here.
  TORCH_CHECK(self.dim() > 0, "glu does not support 0-dimensional tensors");
  auto wrap_dim = maybe_wrap_dim(dim, self.dim());
  const int64_t nIn = self.size(wrap_dim);
  TORCH_CHECK(
      nIn % 2 == 0,
      "Halving dimension must be even, but dimension ",
      wrap_dim,
      " is size ",
      nIn);
}

Tensor& XPUNativeFunctions::glu_out(
    const Tensor& self,
    int64_t dim,
    Tensor& out) {
  check_even_dimension(self, dim);
  native::xpu::glu_kernel(self, dim, out);
  return out;
}

Tensor XPUNativeFunctions::glu(const Tensor& self, int64_t dim) {
  Tensor out = at::empty({}, self.options());
  return XPUNativeFunctions::glu_out(self, dim, out);
}

Tensor& XPUNativeFunctions::glu_backward_out(
    const Tensor& grad_output,
    const Tensor& self,
    int64_t dim,
    Tensor& grad_input) {
  check_even_dimension(self, dim);
  native::xpu::glu_backward_kernel(grad_output, self, dim, grad_input);
  return grad_input;
}

Tensor XPUNativeFunctions::glu_backward(
    const Tensor& grad_output,
    const Tensor& self,
    int64_t dim) {
  Tensor grad_input = at::empty({}, self.options());
  return glu_backward_out(grad_output, self, dim, grad_input);
}
} // namespace at
