/*
 * Portions of this file are derived from FBGEMM
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <ATen/core/dispatch/Dispatcher.h>

namespace fbgemm_utils {

inline bool schemaExists(const std::string& qualified_name) {
  return c10::Dispatcher::singleton()
      .findSchema({qualified_name, ""})
      .has_value();
}

} // namespace fbgemm_utils
