/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/Context.h>
#include <ATen/Device.h>
#include <ATen/DeviceGuard.h>
#include <ATen/DimVector.h>
#include <ATen/Dispatch.h>
#include <ATen/Formatting.h>
#include <ATen/NamedTensor.h>
#include <ATen/ScalarOps.h>
#include <ATen/Tensor.h>
#include <ATen/TensorGeometry.h>
#include <ATen/TensorIndexing.h>
#include <ATen/TensorOperators.h>
#include <ATen/Version.h>
#include <ATen/core/ATenGeneral.h>
#include <ATen/core/Generator.h>
#include <ATen/core/Reduction.h>
#include <ATen/core/Scalar.h>
#include <ATen/core/UnsafeFromTH.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <c10/core/Allocator.h>
#include <c10/core/InferenceMode.h>
#include <c10/core/Layout.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Exception.h>