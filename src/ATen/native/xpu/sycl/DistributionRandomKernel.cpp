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

void random_from_to_kernel(
    TensorIteratorBase& iter,
    uint64_t range,
    int64_t base,
    std::optional<Generator> gen_) {
  auto gen = get_generator_or_default<at::XPUGeneratorImpl>(
      gen_, at::xpu::detail::getDefaultXPUGenerator());
  at::native::templates::xpu::random_from_to_kernel(iter, range, base, gen);
}

void random_full_64_bits_range_kernel(
    TensorIteratorBase& iter,
    std::optional<Generator> gen_) {
  auto gen = get_generator_or_default<at::XPUGeneratorImpl>(
      gen_, at::xpu::detail::getDefaultXPUGenerator());
  at::native::templates::xpu::random_full_64_bits_range_kernel(iter, gen);
}

void random_kernel(TensorIteratorBase& iter, std::optional<Generator> gen_) {
  auto gen = get_generator_or_default<at::XPUGeneratorImpl>(
      gen_, at::xpu::detail::getDefaultXPUGenerator());
  at::native::templates::xpu::random_kernel(iter, gen);
}

} // namespace xpu
} // namespace native
} // namespace at
