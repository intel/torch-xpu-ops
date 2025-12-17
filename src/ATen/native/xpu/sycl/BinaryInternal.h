/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#pragma once

#include <comm/XPUMathCompat.h>

namespace at::native::xpu {

template <typename scalar_t>
struct DivFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    return c10::xpu::compat::div(a, b);
  }
};

template <typename scalar_t>
struct MulFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    return a * b;
  }
};

// Workaround for the error: '*' in boolean context, suggest '&&' instead
// [-Werror=int-in-bool-context]
template <>
struct MulFunctor<bool> {
  bool operator()(bool a, bool b) const {
    return a && b;
  }
};

} // namespace at::native::xpu
