#include <ATen/ATen.h>
#include <ATen/XPUNativeFunctions.h>
#include <ATen/core/Tensor.h>

namespace at {

Tensor& XPUNativeFunctions::tril_out(
    const Tensor& self,
    int64_t diagonal,
    Tensor& out) {
  Tensor out_cpu = out.to(Device(kCPU));
  Tensor self_cpu = self.to(Device(kCPU));
  at::tril_out(out_cpu, self_cpu, diagonal);
  out.copy_(out_cpu);
  return out;
}

} // namespace at