#include <ATen/AccumulateType.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/DistributionTemplates.h>
#include <ATen/native/xpu/sycl/MemoryAccess.h>
#include <ATen/native/xpu/sycl/OffsetCalculator.h>
#include <ATen/native/xpu/sycl/Philox4x32.h>
#include <ATen/xpu/XPUGeneratorImpl.h>
#include <comm/DeviceProperties.h>
#include <comm/Runtime.h>
#include <comm/xpu_aten.h>

#include <ATen/native/xpu/sycl/DistributionKernels.h>

namespace at {
namespace native {
namespace xpu {

void uniform_kernel(
    TensorIteratorBase& iter,
    double from,
    double to,
    std::optional<Generator> gen) {
  auto generator = get_generator_or_default<at::XPUGeneratorImpl>(
      gen, at::xpu::detail::getDefaultXPUGenerator());
  at::native::templates::xpu::uniform_kernel(iter, from, to, generator);
}

} // namespace xpu
} // namespace native
} // namespace at
