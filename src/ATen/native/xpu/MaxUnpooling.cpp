#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/layer_norm.h>
#include <ATen/native/xpu/sycl/MaxUnpoolingKernels.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#endif

namespace at::native {

Tensor& max_unpooling2d_forward_out_xpu(
    const Tensor& self,
    const Tensor& indices,
    IntArrayRef output_size,
    Tensor& out) {
  native::xpu::max_unpooling2d_forward_kernel(out, self, indices, output_size);
  return out;
}

Tensor max_unpooling2d_forward_xpu(
    const Tensor& self,
    const Tensor& indices,
    IntArrayRef output_size) {
  auto out = at::empty({0}, self.options());
  max_unpooling2d_forward_out_xpu(self, indices, output_size, out);
  return out;
}

Tensor& max_unpooling3d_forward_out_xpu(
    const Tensor& self,
    const Tensor& indices,
    IntArrayRef output_size,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor& out) {
  native::xpu::max_unpooling3d_forward_kernel(
      out, self, indices, output_size, stride, padding);
  return out;
}

Tensor max_unpooling3d_forward_xpu(
    const Tensor& self,
    const Tensor& indices,
    IntArrayRef output_size,
    IntArrayRef stride,
    IntArrayRef padding) {
  auto out = at::empty({0}, self.options());
  max_unpooling3d_forward_out_xpu(
      self, indices, output_size, stride, padding, out);
  return out;
}

} // namespace at::native
