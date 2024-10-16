#include <ATen/native/UnaryOps.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <comm/xpu_aten.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/xpu/sycl/BesselJ0Kernel.h>
#include <ATen/native/xpu/sycl/BesselJ1Kernel.h>
#include <ATen/native/xpu/sycl/BesselY0Kernel.h>
#include <ATen/native/xpu/sycl/BesselY1Kernel.h>

namespace at {
namespace native {
REGISTER_XPU_DISPATCH(special_bessel_j0_stub, &xpu::bessel_j0_kernel);
REGISTER_XPU_DISPATCH(special_bessel_j1_stub, &xpu::bessel_j1_kernel);
REGISTER_XPU_DISPATCH(special_bessel_y0_stub, &xpu::bessel_y0_kernel);
REGISTER_XPU_DISPATCH(special_bessel_y1_stub, &xpu::bessel_y1_kernel);
} // namespace native
} // namespace at
