/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "function_types.h"

#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>

namespace fbgemm_utils::config {

  #define ENUMERATE_ALL_FEATURE_FLAGS       \
  X(TBE_V2)                               \
  X(TBE_ENSEMBLE_ROWWISE_ADAGRAD)         \
  X(TBE_ANNOTATE_KINETO_TRACE)            \
  X(TBE_ROCM_INFERENCE_PACKED_BAGS)       \
  X(TBE_ROCM_HIP_BACKWARD_KERNEL)         \
  X(BOUNDS_CHECK_INDICES_V2)              \
  X(TBE_REPORT_INPUT_PARAMS)              \
  X(TBE_CPU_OUTPUT_DISABLE_PINNED_MEMORY) \
  X(TBE_USE_TUNED_SEGMENT_LENGTHS_CTA_B200)

enum class FeatureGateName {
#define X(value) value,
  ENUMERATE_ALL_FEATURE_FLAGS
#undef X
};

std::string to_string(const FeatureGateName& value) {
  switch (value) {
#define X(value)               \
  case FeatureGateName::value: \
    return #value;
    ENUMERATE_ALL_FEATURE_FLAGS
#undef X
  }
  return "UNKNOWN";
}

bool ev_check_key(const std::string& key) {
  const auto env_var = "FBGEMM_" + key;

  const auto value = std::getenv(env_var.c_str());
  if (!value) {
    return false;
  }

  try {
    return std::stoi(value) == 1;
  } catch (const std::invalid_argument&) {
    return false;
  }
}

static bool check_feature_gate_key_impl(
    const std::string& key,
    bool check_env_vars_only [[maybe_unused]]) {
  // Cache feature flags to avoid repeated JK and env var checks
  static std::map<std::string, bool> feature_flags_cache;
  if (const auto search = feature_flags_cache.find(key);
      search != feature_flags_cache.end()) {
    return search->second;
  }

  const auto value = ev_check_key(key);

  feature_flags_cache.insert({key, value});
  return value;
}

DLL_PUBLIC bool check_feature_gate_key(const std::string& key) {
  static const auto no_jk = false;

  return check_feature_gate_key_impl(key, no_jk);
}

DLL_PUBLIC bool is_feature_enabled(const FeatureGateName& feature) {
  return check_feature_gate_key(to_string(feature));
}

DLL_PUBLIC bool is_feature_enabled_from_env(const FeatureGateName& feature) {
  return check_feature_gate_key_impl(
      to_string(feature), /* check_env_vars_only */ true);
}


} // namespace fbgemm_utils::config
