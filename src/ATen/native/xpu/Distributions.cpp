#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/ScalarOps.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/DistributionTemplates.h>
#include <ATen/native/Distributions.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/UnaryOps.h>

#include <ATen/native/xpu/sycl/DistributionKernels.h>
#include <ATen/native/xpu/sycl/MultinomialKernel.h>
#include <ATen/ops/div.h>
#include <comm/xpu_aten.h>

namespace at {
namespace native {
REGISTER_XPU_DISPATCH(normal_stub, &xpu::normal_kernel);
REGISTER_XPU_DISPATCH(uniform_stub, &xpu::uniform_kernel);
REGISTER_XPU_DISPATCH(bernoulli_scalar_stub, &xpu::bernoulli_scalar_kernel);
REGISTER_XPU_DISPATCH(bernoulli_tensor_stub, &xpu::bernoulli_tensor_kernel);
REGISTER_XPU_DISPATCH(random_stub, &xpu::random_kernel);
REGISTER_XPU_DISPATCH(random_from_to_stub, &xpu::random_from_to_kernel);
REGISTER_XPU_DISPATCH(exponential_stub, &xpu::exponential_kernel);
REGISTER_XPU_DISPATCH(
    random_full_64_bits_range_stub,
    &xpu::random_full_64_bits_range_kernel);
REGISTER_XPU_DISPATCH(
    multinomial_with_replacement_stub,
    &xpu::multinomial_kernel);
} // namespace native
} // namespace at
