#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/ScalarOps.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/DistributionTemplates.h>
#include <ATen/native/Distributions.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/UnaryOps.h>

#include <ATen/native/xpu/sycl/DistributionKernels.h>
#include <ATen/native/xpu/sycl/Distributions.h>
#include <ATen/native/xpu/sycl/MultinomialKernel.h>
#include <ATen/ops/div.h>
#include <comm/xpu_aten.h>

namespace at {
namespace native {
REGISTER_XPU_DISPATCH(normal_stub, &xpu::normal_kernel);
REGISTER_XPU_DISPATCH(uniform_stub, &xpu::uniform_kernel);
REGISTER_XPU_DISPATCH(bernoulli_scalar_stub, &xpu::bernoulli_scalar_kernel);
REGISTER_XPU_DISPATCH(bernoulli_tensor_stub, &xpu::bernoulli_tensor_kernel);
REGISTER_XPU_DISPATCH(random_stub, &xpu::random_kernel);
REGISTER_XPU_DISPATCH(random_from_to_stub, &xpu::random_from_to_kernel);
REGISTER_XPU_DISPATCH(exponential_stub, &xpu::exponential_kernel);
REGISTER_XPU_DISPATCH(
    random_full_64_bits_range_stub,
    &xpu::random_full_64_bits_range_kernel);
REGISTER_XPU_DISPATCH(
    multinomial_with_replacement_stub,
    &xpu::multinomial_kernel);
REGISTER_XPU_DISPATCH(log_normal_stub, &xpu::log_normal_kernel);
REGISTER_XPU_DISPATCH(cauchy_stub, &xpu::cauchy_kernel);
REGISTER_XPU_DISPATCH(geometric_stub, &xpu::geometric_kernel);

Tensor _s_poisson_xpu(const Tensor& lambda, std::optional<Generator> gen_) {
  auto gen = get_generator_or_default<at::XPUGeneratorImpl>(
      gen_, at::xpu::detail::getDefaultXPUGenerator());
  Tensor ret = at::empty(lambda.sizes(), lambda.options());
  xpu::launch_poisson_kernel(ret, lambda, gen);
  return ret;
}

Tensor _s_binomial_xpu(
    const Tensor& count,
    const Tensor& prob,
    std::optional<Generator> generator) {
  auto gen = get_generator_or_default<at::XPUGeneratorImpl>(
      generator, at::xpu::detail::getDefaultXPUGenerator());
  Tensor ret = at::empty(count.sizes(), count.options());
  at::TensorIterator iter = at::TensorIteratorConfig()
                                .add_output(ret)
                                .add_input(count)
                                .add_input(prob)
                                .build();
  xpu::launch_binomial_kernel(iter, gen);
  return ret;
}

Tensor _s_gamma_xpu(const Tensor& alpha, c10::optional<Generator> gen_) {
  auto gen = get_generator_or_default<at::XPUGeneratorImpl>(
      gen_, at::xpu::detail::getDefaultXPUGenerator());
  Tensor ret = at::empty(alpha.sizes(), alpha.options());
  xpu::launch_gamma_kernel(ret, alpha, gen);
  return ret;
}

Tensor _sample_dirichlet_xpu(
    const Tensor& alpha,
    std::optional<Generator> generator) {
  auto gen = get_generator_or_default<at::XPUGeneratorImpl>(
      generator, at::xpu::detail::getDefaultXPUGenerator());
  Tensor ret = at::empty(alpha.sizes(), alpha.options());
  xpu::launch_gamma_kernel(ret, alpha, gen);
  auto gamma_sum = ret.sum(/*dim=*/-1, /*keepdim=*/true);
  auto iter = at::TensorIteratorConfig()
                  .add_output(ret)
                  .add_input(ret)
                  .add_input(gamma_sum)
                  .build();
  xpu::launch_dirichlet_kernel(iter);
  return ret;
}

Tensor _standard_gamma_grad_xpu(const Tensor& self, const Tensor& output) {
  Tensor ret = at::empty(self.sizes(), self.options());
  TensorIterator iter = TensorIteratorConfig()
                            .add_output(ret)
                            .add_input(self)
                            .add_input(output)
                            .build();
  xpu::launch_standard_gamma_grad_kernel(iter);
  return ret;
}

Tensor _dirichlet_grad_xpu(
    const Tensor& x,
    const Tensor& alpha,
    const Tensor& total) {
  Tensor ret = at::empty(x.sizes(), x.options());
  auto iter = at::TensorIteratorConfig()
                  .add_output(ret)
                  .add_input(x)
                  .add_input(alpha)
                  .add_input(total)
                  .build();
  xpu::launch_dirichlet_grad_kernel(iter);
  return ret;
}

} // namespace native
} // namespace at
