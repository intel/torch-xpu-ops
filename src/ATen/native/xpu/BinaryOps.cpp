#include <ATen/ScalarOps.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/TensorIterator.h>

#include <ATen/native/DispatchStub.h>

#include <ATen/native/xpu/sycl/BinaryBitwiseOpsKernels.h>
#include <ATen/native/xpu/sycl/BinaryKernels.h>
#include <ATen/native/xpu/sycl/BinaryMiscBackwardOpsKernels.h>
#include <ATen/native/xpu/sycl/BinaryRemainderKernel.h>
#include <ATen/native/xpu/sycl/GcdLcmKernels.h>
#include <ATen/native/xpu/sycl/MaxMinElementwiseKernels.h>

namespace at {

namespace native {
REGISTER_XPU_DISPATCH(add_stub, xpu::add_kernel);
REGISTER_XPU_DISPATCH(sub_stub, xpu::sub_kernel);
REGISTER_XPU_DISPATCH(mul_stub, xpu::mul_kernel);
// REGISTER_XPU_DISPATCH(div_stub, xpu::div_kernel);
// div needs 3 stub, div_true_stub, div_trunc_stub, div_floor_stub
REGISTER_XPU_DISPATCH(remainder_stub, xpu::remainder_kernel);
REGISTER_XPU_DISPATCH(fmod_stub, xpu::fmod_kernel);
REGISTER_XPU_DISPATCH(tanh_backward_stub, xpu::tanh_backward_kernel);
REGISTER_XPU_DISPATCH(bitwise_and_stub, xpu::bitwise_and_kernel);
REGISTER_XPU_DISPATCH(bitwise_or_stub, xpu::bitwise_or_kernel);
REGISTER_XPU_DISPATCH(bitwise_xor_stub, xpu::bitwise_xor_kernel);
REGISTER_XPU_DISPATCH(gcd_stub, xpu::gcd_kernel);

REGISTER_XPU_DISPATCH(maximum_stub, xpu::maximum_kernel);
REGISTER_XPU_DISPATCH(minimum_stub, xpu::minimum_kernel);
REGISTER_XPU_DISPATCH(sigmoid_backward_stub, xpu::sigmoid_backward_kernel);

} // namespace native
} // namespace at
