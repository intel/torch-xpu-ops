#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/DistributionKernels.h>
#include <ATen/native/xpu/sycl/DistributionTemplates.h>
#include <ATen/xpu/XPUGeneratorImpl.h>

namespace at::native::xpu {

void cauchy_kernel(
    TensorIteratorBase& iter,
    double median,
    double sigma,
    std::optional<Generator> gen) {
  auto generator = get_generator_or_default<at::XPUGeneratorImpl>(
      gen, at::xpu::detail::getDefaultXPUGenerator());
  at::native::templates::xpu::cauchy_kernel(iter, median, sigma, generator);
}

} // namespace at::native::xpu
