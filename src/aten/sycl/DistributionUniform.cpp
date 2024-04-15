#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/xpu/XPUGeneratorImpl.h>
#include <aten/sycl/DistributionTemplates.h>
#include <aten/sycl/MemoryAccess.h>
#include <aten/sycl/OffsetCalculator.h>
#include <aten/sycl/Philox4x32.h>
#include <comm/DeviceProperties.h>
#include <comm/Runtime.h>

namespace at {
namespace native {
namespace xpu {

void uniform_kernel(
    TensorIteratorBase& iter,
    double from,
    double to,
    c10::optional<Generator> gen) {
  auto generator = get_generator_or_default<at::XPUGeneratorImpl>(
      gen, at::xpu::detail::getDefaultXPUGenerator());
  at::native::templates::xpu::uniform_kernel(iter, from, to, generator);
}

} // namespace xpu
} // namespace native
} // namespace at
