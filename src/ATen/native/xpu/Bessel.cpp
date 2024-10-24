#include <ATen/native/UnaryOps.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/xpu/sycl/BesselJ0Kernel.h>
#include <ATen/native/xpu/sycl/BesselJ1Kernel.h>
#include <ATen/native/xpu/sycl/BesselY0Kernel.h>
#include <ATen/native/xpu/sycl/BesselY1Kernel.h>
#include <ATen/native/xpu/sycl/ModifiedBesselI0Kernel.h>
#include <ATen/native/xpu/sycl/ModifiedBesselI1Kernel.h>
#include <ATen/native/xpu/sycl/ModifiedBesselK0Kernel.h>
#include <ATen/native/xpu/sycl/ModifiedBesselK1Kernel.h>
#include <ATen/native/xpu/sycl/SphericalBesselJ0Kernel.h>

namespace at {
namespace native {
REGISTER_XPU_DISPATCH(special_bessel_j0_stub, &xpu::bessel_j0_kernel);
REGISTER_XPU_DISPATCH(special_bessel_j1_stub, &xpu::bessel_j1_kernel);
REGISTER_XPU_DISPATCH(special_bessel_y0_stub, &xpu::bessel_y0_kernel);
REGISTER_XPU_DISPATCH(special_bessel_y1_stub, &xpu::bessel_y1_kernel);
REGISTER_XPU_DISPATCH(special_modified_bessel_i0_stub, &xpu::modified_bessel_i0_kernel);
REGISTER_XPU_DISPATCH(special_modified_bessel_i1_stub, &xpu::modified_bessel_i1_kernel);
REGISTER_XPU_DISPATCH(special_modified_bessel_k0_stub, &xpu::modified_bessel_k0_kernel);
REGISTER_XPU_DISPATCH(special_modified_bessel_k1_stub, &xpu::modified_bessel_k1_kernel);
REGISTER_XPU_DISPATCH(special_spherical_bessel_j0_stub, &xpu::spherical_bessel_j0_kernel);
} // namespace native
} // namespace at
