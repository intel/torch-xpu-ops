#include <ATen/native/DispatchStub.h>
#include <ATen/native/Distance.h>
#include <ATen/native/xpu/sycl/DistanceKernels.h>

namespace at {

namespace native {
REGISTER_XPU_DISPATCH(cdist_stub, &xpu::cdist_kernel);
}

Tensor XPUNativeFunctions::_cdist_forward(
    const Tensor& x1,
    const Tensor& x2,
    const double p,
    c10::optional<int64_t> compute_mode) {
  TORCH_CHECK(
      x1.dim() >= 2,
      "cdist only supports at least 2D tensors, X1 got: ",
      x1.dim(),
      "D");
  TORCH_CHECK(
      x2.dim() >= 2,
      "cdist only supports at least 2D tensors, X2 got: ",
      x2.dim(),
      "D");
  TORCH_CHECK(
      x1.size(-1) == x2.size(-1),
      "X1 and X2 must have the same number of columns. X1: ",
      x1.size(-1),
      " X2: ",
      x2.size(-1));

  return cdist_impl(x1, x2, p, compute_mode);
}

Tensor XPUNativeFunctions::_cdist_backward(
    const Tensor& _grad,
    const Tensor& _x1,
    const Tensor& _x2,
    const double p,
    const Tensor& _cdist) {
  // Broadcasting might generate non-contiguous Tensors, so handle it before
  // doing checks
  int64_t c1 = _x1.size(-1);
  int64_t c2 = _x2.size(-1);
  int64_t r1 = _x1.size(-2);
  int64_t r2 = _x2.size(-2);
  auto dim1 = _x1.dim();
  auto dim2 = _x2.dim();
  IntArrayRef batch_tensor1(_x1.sizes().data(), dim1 - 2);
  IntArrayRef batch_tensor2(_x2.sizes().data(), dim2 - 2);
  std::vector<int64_t> expand_batch_portion =
      infer_size(batch_tensor1, batch_tensor2);
  std::vector<int64_t> tensor1_expand_size(expand_batch_portion);
  tensor1_expand_size.insert(tensor1_expand_size.end(), {r1, c1});
  std::vector<int64_t> tensor2_expand_size(expand_batch_portion);
  tensor2_expand_size.insert(tensor2_expand_size.end(), {r2, c2});

  // Compute the linearized batch size
  const int64_t batch_product = c10::multiply_integers(expand_batch_portion);

  // Gracefully handle empty Tensors
  if (r1 == 0 || r2 == 0 || c1 == 0 || batch_product == 0) {
    return at::zeros_like(_x1, _x1.options());
  }

  Tensor x1 = _x1;
  if (tensor1_expand_size != x1.sizes()) {
    x1 = x1.expand(tensor1_expand_size);
  }
  Tensor x2 = _x2;
  if (tensor2_expand_size != x2.sizes()) {
    x2 = x2.expand(tensor2_expand_size);
  }

  x1 = x1.contiguous();
  x2 = x2.contiguous();
  auto cdist = _cdist.contiguous();
  auto grad = _grad.contiguous();
  int64_t n = x1.size(-2);
  int64_t m = x1.size(-1);
  auto device1 = x1.device().type();
  TORCH_CHECK(
      device1 == kCPU || device1 == kXPU,
      "_cdist_backward only supports CPU and XPU devices, X1 got: ",
      device1);
  auto device2 = x2.device().type();
  TORCH_CHECK(
      device2 == kCPU || device2 == kXPU,
      "_cdist_backward only supports CPU and XPU devices, X2 got: ",
      device2);

  Tensor grad_x1 = at::empty(
      {batch_product, n, m}, x1.options(), LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  grad_x1 = native::xpu::cdist_backward_kernel(grad_x1, grad, x1, x2, p, cdist);
  return grad_x1.view(x1.sizes());
}

Tensor XPUNativeFunctions::_pdist_forward(const Tensor& self, const double p) {
  TORCH_CHECK(self.is_contiguous(), "_pdist_forward requires contiguous input");
  Tensor result = at::empty({0}, self.options());
  if (self.size(0) <= 1) {
    result.resize_({0});
  } else {
    int64_t n = self.size(0);
    int64_t c = n * (n - 1) / 2;
    result.resize_({c});
    if (self.size(1) == 0) {
      result.fill_(0);
    } else {
      native::xpu::pdist_forward_kernel(result, self, p);
    }
  }
  return result;
}

Tensor XPUNativeFunctions::_pdist_backward(
    const Tensor& grad,
    const Tensor& self,
    const double p,
    const Tensor& pdist) {
  TORCH_CHECK(
      self.is_contiguous(), "_pdist_backward requires self to be contiguous");
  TORCH_CHECK(
      pdist.is_contiguous(), "_pdist_backward requires pdist to be contiguous");

  Tensor result = at::empty_like(self);
  native::xpu::pdist_backward_kernel(result, grad, self, p, pdist);
  return result;
}

} // namespace at
