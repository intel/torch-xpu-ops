/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#pragma once

#include <limits>

namespace at {

template <typename T>
struct numeric_limits {
  static constexpr T lowest() {
    return std::numeric_limits<T>::lowest();
  }

  static constexpr T max() {
    return std::numeric_limits<T>::max();
  }

  static constexpr T lower_bound() {
    if constexpr (std::numeric_limits<T>::has_infinity) {
      return -std::numeric_limits<T>::infinity();
    } else {
      return std::numeric_limits<T>::lowest();
    }
  }

  static constexpr T upper_bound() {
    if constexpr (std::numeric_limits<T>::has_infinity) {
      return std::numeric_limits<T>::infinity();
    } else {
      return std::numeric_limits<T>::max();
    }
  }
};

} // namespace at
