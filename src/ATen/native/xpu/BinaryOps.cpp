#include <ATen/ScalarOps.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>

#include <xpu/ATen/ops/add_native.h>

#include <ATen/native/xpu/sycl/BinaryBitwiseOpsKernels.h>
#include <ATen/native/xpu/sycl/BinaryGeometricKernels.h>
#include <ATen/native/xpu/sycl/BinaryKernels.h>
#include <ATen/native/xpu/sycl/BinaryLogicalOpsKernels.h>
#include <ATen/native/xpu/sycl/BinaryMiscBackwardOpsKernels.h>
#include <ATen/native/xpu/sycl/BinaryMiscOpsKernels.h>
#include <ATen/native/xpu/sycl/BinaryRemainderKernel.h>
#include <ATen/native/xpu/sycl/BinaryShiftOpsKernels.h>
#include <ATen/native/xpu/sycl/ChebyshevPolynomialKernels.h>
#include <ATen/native/xpu/sycl/CopysignKernel.h>
#include <ATen/native/xpu/sycl/GcdLcmKernels.h>
#include <ATen/native/xpu/sycl/HermitePolynomialHKernel.h>
#include <ATen/native/xpu/sycl/HermitePolynomialHeKernel.h>
#include <ATen/native/xpu/sycl/IGammaKernel.h>
#include <ATen/native/xpu/sycl/LaguerrePolynomialLKernel.h>
#include <ATen/native/xpu/sycl/LegendrePolynomialPKernel.h>
#include <ATen/native/xpu/sycl/LogAddExpKernels.h>
#include <ATen/native/xpu/sycl/MaxMinElementwiseKernels.h>
#include <ATen/native/xpu/sycl/ShiftedChebyshevPolynomialKernels.h>
#include <ATen/native/xpu/sycl/StepKernels.h>
#include <ATen/native/xpu/sycl/ZetaKernel.h>

namespace at {
namespace native {

REGISTER_XPU_DISPATCH(add_stub, &xpu::add_kernel)
REGISTER_XPU_DISPATCH(sub_stub, &xpu::sub_kernel);
REGISTER_XPU_DISPATCH(mul_stub, &xpu::mul_kernel);
REGISTER_XPU_DISPATCH(div_true_stub, &xpu::div_true_kernel);
REGISTER_XPU_DISPATCH(div_trunc_stub, &xpu::div_trunc_kernel);
REGISTER_XPU_DISPATCH(div_floor_stub, &xpu::div_floor_kernel);
REGISTER_XPU_DISPATCH(remainder_stub, &xpu::remainder_kernel);
REGISTER_XPU_DISPATCH(fmod_stub, &xpu::fmod_kernel);
REGISTER_XPU_DISPATCH(tanh_backward_stub, &xpu::tanh_backward_kernel);
REGISTER_XPU_DISPATCH(bitwise_and_stub, &xpu::bitwise_and_kernel);
REGISTER_XPU_DISPATCH(bitwise_or_stub, &xpu::bitwise_or_kernel);
REGISTER_XPU_DISPATCH(bitwise_xor_stub, &xpu::bitwise_xor_kernel);
REGISTER_XPU_DISPATCH(gcd_stub, &xpu::gcd_kernel);
REGISTER_XPU_DISPATCH(lcm_stub, &xpu::lcm_kernel);
REGISTER_XPU_DISPATCH(maximum_stub, &xpu::maximum_kernel);
REGISTER_XPU_DISPATCH(minimum_stub, &xpu::minimum_kernel);
REGISTER_XPU_DISPATCH(sigmoid_backward_stub, &xpu::sigmoid_backward_kernel);
REGISTER_XPU_DISPATCH(nextafter_stub, &xpu::nextafter_kernel);
REGISTER_XPU_DISPATCH(heaviside_stub, &xpu::heaviside_kernel);
REGISTER_XPU_DISPATCH(hypot_stub, &xpu::hypot_kernel);
REGISTER_XPU_DISPATCH(igamma_stub, &xpu::igamma_kernel);
REGISTER_XPU_DISPATCH(igammac_stub, &xpu::igammac_kernel);
REGISTER_XPU_DISPATCH(atan2_stub, &xpu::atan2_kernel);
REGISTER_XPU_DISPATCH(copysign_stub, &xpu::copysign_kernel);
REGISTER_XPU_DISPATCH(logical_and_stub, &xpu::logical_and_kernel);
REGISTER_XPU_DISPATCH(logical_or_stub, &xpu::logical_or_kernel);
REGISTER_XPU_DISPATCH(logical_xor_stub, &xpu::logical_xor_kernel);
REGISTER_XPU_DISPATCH(logit_backward_stub, &xpu::logit_backward_kernel);
REGISTER_XPU_DISPATCH(logaddexp_stub, &xpu::logaddexp_kernel);
REGISTER_XPU_DISPATCH(logaddexp2_stub, &xpu::logaddexp2_kernel);
REGISTER_XPU_DISPATCH(fmax_stub, &xpu::fmax_kernel);
REGISTER_XPU_DISPATCH(fmin_stub, &xpu::fmin_kernel);
REGISTER_XPU_DISPATCH(lshift_stub, &xpu::lshift_kernel);
REGISTER_XPU_DISPATCH(rshift_stub, &xpu::rshift_kernel);
REGISTER_XPU_DISPATCH(xlogy_stub, &xpu::xlogy_kernel);
REGISTER_XPU_DISPATCH(xlog1py_stub, &xpu::xlog1py_kernel);
REGISTER_XPU_DISPATCH(zeta_stub, &xpu::zeta_kernel);
REGISTER_XPU_DISPATCH(
    hermite_polynomial_h_stub,
    &xpu::hermite_polynomial_h_kernel);
REGISTER_XPU_DISPATCH(
    hermite_polynomial_he_stub,
    &xpu::hermite_polynomial_he_kernel);
REGISTER_XPU_DISPATCH(
    laguerre_polynomial_l_stub,
    &xpu::laguerre_polynomial_l_kernel);
REGISTER_XPU_DISPATCH(
    legendre_polynomial_p_stub,
    &xpu::legendre_polynomial_p_kernel);
REGISTER_XPU_DISPATCH(
    chebyshev_polynomial_t_stub,
    &xpu::chebyshev_polynomial_t_kernel);
REGISTER_XPU_DISPATCH(
    chebyshev_polynomial_u_stub,
    &xpu::chebyshev_polynomial_u_kernel);
REGISTER_XPU_DISPATCH(
    chebyshev_polynomial_v_stub,
    &xpu::chebyshev_polynomial_v_kernel);
REGISTER_XPU_DISPATCH(
    chebyshev_polynomial_w_stub,
    &xpu::chebyshev_polynomial_w_kernel);
REGISTER_XPU_DISPATCH(
    shifted_chebyshev_polynomial_t_stub,
    &xpu::shifted_chebyshev_polynomial_t_kernel);
REGISTER_XPU_DISPATCH(
    shifted_chebyshev_polynomial_u_stub,
    &xpu::shifted_chebyshev_polynomial_u_kernel);
REGISTER_XPU_DISPATCH(
    shifted_chebyshev_polynomial_v_stub,
    &xpu::shifted_chebyshev_polynomial_v_kernel);
REGISTER_XPU_DISPATCH(
    shifted_chebyshev_polynomial_w_stub,
    &xpu::shifted_chebyshev_polynomial_w_kernel);

TORCH_IMPL_FUNC(add_out_xpu)
(const Tensor& self,
 const Tensor& other,
 const Scalar& alpha,
 const Tensor& output) {
  auto iter = TensorIterator::borrowing_binary_op(output, self, other);
  xpu::add_kernel(iter, alpha);
}

} // namespace native
} // namespace at
