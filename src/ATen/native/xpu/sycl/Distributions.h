#pragma once

#include <ATen/native/TensorIterator.h>
#include <ATen/xpu/XPUGeneratorImpl.h>

namespace at::native::xpu {

TORCH_XPU_API void launch_binomial_xpu_kernel(
    TensorIteratorBase& iter,
    XPUGeneratorImpl* gen);

TORCH_XPU_API void launch_gamma_kernel(
    Tensor& ret,
    const Tensor& alpha,
    XPUGeneratorImpl* gen);

TORCH_XPU_API void launch_dirichlet_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void launch_dirichlet_grad_kernel(TensorIteratorBase& iter);

} // namespace at::native::xpu
