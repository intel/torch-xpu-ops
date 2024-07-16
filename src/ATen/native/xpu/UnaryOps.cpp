#include <ATen/ScalarOps.h>
#include <ATen/core/Tensor.h>

#include <ATen/native/DispatchStub.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/UnaryOps.h>

#include <ATen/native/xpu/sycl/AbsKernel.h>
#include <ATen/native/xpu/sycl/UnaryComplexKernels.h>
#include <ATen/native/xpu/sycl/UnaryFractionKernels.h>
#include <ATen/native/xpu/sycl/UnaryGeometricAcosKernel.h>
#include <ATen/native/xpu/sycl/UnaryGeometricAcoshKernel.h>
#include <ATen/native/xpu/sycl/UnaryGeometricAsinKernel.h>
#include <ATen/native/xpu/sycl/UnaryGeometricAsinhKernel.h>
#include <ATen/native/xpu/sycl/UnaryGeometricAtanKernel.h>
#include <ATen/native/xpu/sycl/UnaryGeometricAtanhKernel.h>
#include <ATen/native/xpu/sycl/UnaryGeometricCosKernel.h>
#include <ATen/native/xpu/sycl/UnaryGeometricCoshKernel.h>
#include <ATen/native/xpu/sycl/UnaryGeometricSinKernel.h>
#include <ATen/native/xpu/sycl/UnaryGeometricSinhKernel.h>
#include <ATen/native/xpu/sycl/UnaryGeometricTanKernel.h>
#include <ATen/native/xpu/sycl/UnaryGeometricTanhKernel.h>

#include <ATen/native/xpu/sycl/UnaryKernels.h>
#include <ATen/native/xpu/sycl/UnaryLogKernels.h>
#include <ATen/native/xpu/sycl/UnarySignKernels.h>
#include <ATen/native/xpu/sycl/UnarySpecialOpsKernels.h>

#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/real.h>

namespace at {
namespace native {
REGISTER_XPU_DISPATCH(abs_stub, &xpu::abs_kernel);
REGISTER_XPU_DISPATCH(sin_stub, &xpu::sin_kernel);
REGISTER_XPU_DISPATCH(cos_stub, &xpu::cos_kernel);
REGISTER_XPU_DISPATCH(log_stub, &xpu::log_kernel);
REGISTER_XPU_DISPATCH(sqrt_stub, &xpu::sqrt_kernel);
REGISTER_XPU_DISPATCH(rsqrt_stub, &xpu::rsqrt_kernel);
REGISTER_XPU_DISPATCH(tanh_stub, &xpu::tanh_kernel);
REGISTER_XPU_DISPATCH(neg_stub, &xpu::neg_kernel);
REGISTER_XPU_DISPATCH(logical_not_stub, &xpu::logical_not_kernel);
REGISTER_XPU_DISPATCH(reciprocal_stub, &xpu::reciprocal_kernel);
REGISTER_XPU_DISPATCH(bitwise_not_stub, &xpu::bitwise_not_kernel);
REGISTER_XPU_DISPATCH(exp_stub, &xpu::exp_kernel);
REGISTER_XPU_DISPATCH(sigmoid_stub, &xpu::sigmoid_kernel);
REGISTER_XPU_DISPATCH(sgn_stub, &xpu::sgn_kernel);
REGISTER_XPU_DISPATCH(sign_stub, &xpu::sign_kernel);
REGISTER_XPU_DISPATCH(acos_stub, &xpu::acos_kernel);
REGISTER_XPU_DISPATCH(acosh_stub, &xpu::acosh_kernel);
REGISTER_XPU_DISPATCH(erf_stub, &xpu::erf_kernel);
REGISTER_XPU_DISPATCH(erfc_stub, &xpu::erfc_kernel);
REGISTER_XPU_DISPATCH(conj_physical_stub, &xpu::conj_physical_kernel);
REGISTER_XPU_DISPATCH(ceil_stub, &xpu::ceil_kernel);
REGISTER_XPU_DISPATCH(sinh_stub, &xpu::sinh_kernel);
REGISTER_XPU_DISPATCH(asinh_stub, &xpu::asinh_kernel);
REGISTER_XPU_DISPATCH(tan_stub, &xpu::tan_kernel);
REGISTER_XPU_DISPATCH(atan_stub, &xpu::atan_kernel);
REGISTER_XPU_DISPATCH(atanh_stub, &xpu::atanh_kernel);
REGISTER_XPU_DISPATCH(cosh_stub, &xpu::cosh_kernel);
REGISTER_XPU_DISPATCH(nan_to_num_stub, &xpu::nan_to_num_kernel);
} // namespace native
} // namespace at
