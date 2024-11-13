#include <ATen/ATen.h>
#include <ATen/native/xpu/sycl/RreluWithNoiseKernels.h>

namespace at {
namespace native {

Tensor& rrelu_with_noise_out_xpu(
    const Tensor& self,
    const Tensor& noise,
    const Scalar& lower,
    const Scalar& upper,
    bool training,
    std::optional<Generator> generator,
    Tensor& output) {
  return xpu::rrelu_with_noise_kernel(
      self, noise, lower, upper, training, generator, output);
}

Tensor rrelu_with_noise_xpu(
    const Tensor& self,
    const Tensor& noise,
    const Scalar& lower,
    const Scalar& upper,
    bool training,
    std::optional<Generator> generator) {
  Tensor output = at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  return at::native::rrelu_with_noise_out_xpu(
      self, noise, lower, upper, training, generator, output);
}

Tensor& rrelu_with_noise_xpu_(
    Tensor& self,
    const Tensor& noise,
    const Scalar& lower,
    const Scalar& upper,
    bool training,
    std::optional<Generator> generator) {
  return at::native::rrelu_with_noise_out_xpu(
      self, noise, lower, upper, training, generator, self);
}

} // namespace native
} // namespace at
