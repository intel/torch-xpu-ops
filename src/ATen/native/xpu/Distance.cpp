#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/utils/ParamUtils.h>
#include <ATen/native/xpu/sycl/DistanceKernels.h>
#include <ATen/xpu/XPUNativeFunctions.h>

namespace at {
Tensor cdist_impl(
    const Tensor& x1,
    const Tensor& x2,
    const double p,
    c10::optional<int64_t> compute_mode) {
  TORCH_CHECK(
      at::isFloatingType(x1.scalar_type()),
      "cdist only supports floating-point dtypes, X1 got: ",
      x1.scalar_type());
  auto device1 = x1.device().type();
  TORCH_CHECK(
      at::isFloatingType(x2.scalar_type()),
      "cdist only supports floating-point dtypes, X2 got: ",
      x2.scalar_type());
  auto device2 = x2.device().type();
  TORCH_CHECK(p >= 0, "cdist only supports non-negative p values");
  TORCH_CHECK(
      device1 == device2,
      "X1 and X2 must have the same device type. X1: ",
      device1,
      " X2: ",
      device2);
  // TODO: This is bad; this test should apply universally
  TORCH_CHECK(
      !x1.is_xpu() || x1.get_device() == x2.get_device(),
      "device of X1 (",
      x1.get_device(),
      ") must match device of X2 (",
      x2.get_device(),
      ")");

  SymInt c1 = x1.sym_size(-1);
  SymInt c2 = x2.sym_size(-1);
  // 0 - default value. If p = 2 and r1 > 25 or r2 > 25 (these values are based
  // on performance metrics), it will try to compute distance using matrix
  // multiplication approach 1 - force to use matrix multiplication for p = 2 2
  // - do not use matrix multiplication for p = 2
  int64_t mode = compute_mode.value_or(0);
  TORCH_CHECK(
      mode >= 0 && mode <= 2, "possible modes: 0, 1, 2, but was: ", mode);
  SymInt r1 = x1.size(-2);
  SymInt r2 = x2.size(-2);
  if (!(p == 2 && (mode == 1 || (mode == 0 && (r1 > 25 || r2 > 25))))) {
    TORCH_CHECK(
        device1 == kCPU || device1 == kXPU,
        "cdist only supports CPU and XPU devices, X1 got: ",
        device1);
    TORCH_CHECK(
        device2 == kCPU || device2 == kXPU,
        "cdist only supports CPU and XPU devices, X2 got: ",
        device2);
  }
  int64_t dim1 = x1.dim();
  int64_t dim2 = x2.dim();
  SymIntArrayRef batch_tensor1(x1.sym_sizes().data(), dim1 - 2);
  SymIntArrayRef batch_tensor2(x2.sym_sizes().data(), dim2 - 2);
  std::vector<SymInt> expand_batch_portion =
      at::infer_size_symint(batch_tensor1, batch_tensor2);
  std::vector<SymInt> x1_expand_size(expand_batch_portion);
  x1_expand_size.insert(x1_expand_size.end(), {r1, c1});
  std::vector<SymInt> x2_expand_size(expand_batch_portion);
  x2_expand_size.insert(x2_expand_size.end(), {r2, c2});

  const SymInt expand_batch_product =
      c10::multiply_integers(expand_batch_portion);
  std::vector<SymInt> x1_view{expand_batch_product, r1, c1};
  std::vector<SymInt> x2_view{expand_batch_product, r2, c2};

  Tensor x1_expanded =
      x1.expand_symint(x1_expand_size).contiguous().view_symint(x1_view);
  Tensor x2_expanded =
      x2.expand_symint(x2_expand_size).contiguous().view_symint(x2_view);

  std::vector<SymInt> output_shape(std::move(expand_batch_portion));
  output_shape.insert(output_shape.end(), {r1, r2});

  Tensor result;
  if (r1 == 0 || r2 == 0 || expand_batch_product == 0) {
    result = at::empty_symint(output_shape, x1.options());
  } else if (c1 == 0) {
    result = at::zeros_symint(output_shape, x1.options());
  } else if (p == 2 && (mode == 1 || (mode == 0 && (r1 > 25 || r2 > 25)))) {
    Tensor dist = (expand_batch_product == 1)
        ? at::_euclidean_dist(x1, x2)
        : at::_euclidean_dist(x1_expanded, x2_expanded);
    result = dist.view_symint(output_shape);
  } else {
    result = at::empty_symint(output_shape, x1.options());
    native::xpu::cdist_kernel_impl(result, x1_expanded, x2_expanded, p);
  }
  return result;
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

} // namespace at