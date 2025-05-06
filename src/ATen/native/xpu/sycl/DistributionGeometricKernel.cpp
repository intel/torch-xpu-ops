#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/DistributionKernels.h>
#include <ATen/native/xpu/sycl/DistributionTemplates.h>
#include <ATen/xpu/XPUGeneratorImpl.h>

namespace at::native::xpu {

void geometric_kernel(
    TensorIteratorBase& iter,
    double p_,
    std::optional<Generator> gen) {
  auto generator = get_generator_or_default<at::XPUGeneratorImpl>(
      gen, at::xpu::detail::getDefaultXPUGenerator());
  at::native::templates::xpu::geometric_kernel(iter, p_, generator);
}

} // namespace at::native::xpu
