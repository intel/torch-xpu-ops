/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/ExpandUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/LinearAlgebra.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/utils/ParamUtils.h>
#include <comm/xpu_aten.h>

#include <ATen/native/xpu/sycl/LinearAlgebraKernels.h>
#include <ATen/native/xpu/sycl/ReduceNormKernel.h>
#include <comm/RegisterUtils.h>

namespace at {
namespace native {
REGISTER_XPU_DISPATCH(addr_stub, &xpu::addr_kernel);
REGISTER_XPU_DISPATCH(norm_stub, &xpu::norm_kernel);
} // namespace native
} // namespace at
