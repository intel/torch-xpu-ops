#pragma once

#include <ATen/native/TensorIterator.h>
#include <ATen/xpu/XPUGeneratorImpl.h>

namespace at::native::xpu {

void launch_binomial_xpu_kernel(
    TensorIteratorBase& iter,
    XPUGeneratorImpl* gen);

void launch_gamma_kernel(
    Tensor& ret,
    const Tensor& alpha,
    XPUGeneratorImpl* gen);

void launch_dirichlet_kernel(TensorIteratorBase& iter);

void launch_dirichlet_grad_kernel(TensorIteratorBase& iter);

} // namespace at::native::xpu
