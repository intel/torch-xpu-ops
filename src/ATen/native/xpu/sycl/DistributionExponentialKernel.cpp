#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/DistributionTemplates.h>
#include <ATen/native/xpu/sycl/Philox4x32.h>
#include <ATen/xpu/XPUGeneratorImpl.h>
#include <comm/DeviceProperties.h>
#include <comm/Runtime.h>

#include <ATen/native/xpu/sycl/DistributionKernels.h>

namespace at::native::xpu {

void exponential_kernel(
    TensorIteratorBase& iter,
    double lambda,
    std::optional<Generator> gen) {
  auto generator = get_generator_or_default<at::XPUGeneratorImpl>(
      gen, at::xpu::detail::getDefaultXPUGenerator());
  at::native::templates::xpu::exponential_kernel(iter, lambda, generator);
}

} // namespace at::native::xpu
