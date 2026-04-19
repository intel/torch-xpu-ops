/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <torch/library.h>
#include <experimental/source_location>

namespace fbgemm_utils::utils {

using source_location = std::experimental::source_location;

////////////////////////////////////////////////////////////////////////////////
// Source Context
//
// This is a wrapper abstraction around some source context information,
// including the source location, template filepath, and summary string.  It is
// used to generate consistent descriptions in log messages around kernel
// executions.
////////////////////////////////////////////////////////////////////////////////

struct SourceContext {
  // The source location
  const source_location location;
  // A summary of the context (usually the kernel name)
  const std::string_view summary;
  // The originating template filepath (for template-generated source files)
  const std::string_view template_;
  // The file descriptor for DSA error reporting (needs to be generated at
  // compile-time)
  const std::string_view dsa_file_descriptor_;

  constexpr inline SourceContext(
      const source_location& loc_,
      const std::string_view& sum_,
      const std::string_view& tmpl_,
      const std::string_view& dsa_) noexcept
      : location(loc_),
        summary(sum_),
        template_(tmpl_),
        dsa_file_descriptor_(dsa_) {}

  inline const std::string description() const noexcept {
    // Generate and cache the description if it hasn't been generated yet
    std::stringstream ss;

    // Append template source file location if it exists
    if (!template_.empty()) {
      ss << "[" << template_ << "] ";
    }

    ss << "[" << location.file_name() << '(' << location.line() << ':'
       << location.column() << ")] [" << summary << "]";

    return ss.str();
  }

  inline SourceContext withSummary(
      const std::string_view& sum_) const noexcept {
    return SourceContext(location, sum_, template_, dsa_file_descriptor_);
  }
};

} // namespace fbgemm_utils::utils

#define SOURCE_CONTEXT_CURRENT(LABEL)                \
  fbgemm_utils::utils::SourceContext(                  \
      fbgemm_utils::utils::source_location::current(), \
      #LABEL,                                        \
      _FBGEMM_TFILE_,                                \
      _FBGEMM_DSA_FILESRC_);

#define FBGEMM_LAUNCH_KERNEL(KERNEL, GRID, BLOCK, SMEM, STREAM, ...)        \
  ([&] {                                                                    \
    constexpr auto context = SOURCE_CONTEXT_CURRENT(KERNEL);                \
    auto& kernel = KERNEL;                                                  \
                                                                            \
    return fbgemm_utils::utils::                                              \
        KernelLauncher<false, _FKL_BLOCKING_, _FKL_TENSORCHECK_>(context)   \
            .launch_kernel(kernel, GRID, BLOCK, SMEM, STREAM, __VA_ARGS__); \
  }())


#define PRIVATE_CASE_TYPE_CACHE(enum_type, type, ...) \
  case enum_type: {                                   \
    using cache_t = type;                             \
    return __VA_ARGS__();                             \
  }


#define PRIVATE_CASE_TYPE_EMB(enum_type1, enum_type2, type1, NAME, ...)    \
  case enum_type1: {                                                       \
    using emb_t = type1;                                                   \
    switch (enum_type2) {                                                  \
      PRIVATE_CASE_TYPE_CACHE(at::ScalarType::Float, float, __VA_ARGS__)   \
      PRIVATE_CASE_TYPE_CACHE(at::ScalarType::Half, at::Half, __VA_ARGS__) \
      default:                                                             \
        AT_ERROR(                                                          \
            #NAME,                                                         \
            " not implemented for cache_t '",                              \
            toString(enum_type2),                                          \
            "'");                                                          \
    }                                                                      \
  }

#define _DISPATCH_EMB_CACHE_TYPES(emb_enum_type, cache_enum_type, NAME, ...)  \
  at::ScalarType _emb_t = emb_enum_type;                                      \
  at::ScalarType _cache_t = cache_enum_type;                                  \
  switch (_emb_t) {                                                           \
    PRIVATE_CASE_TYPE_EMB(                                                    \
        at::ScalarType::Float, _cache_t, float, NAME, __VA_ARGS__)            \
    PRIVATE_CASE_TYPE_EMB(                                                    \
        at::ScalarType::Half, _cache_t, at::Half, NAME, __VA_ARGS__)          \
    PRIVATE_CASE_TYPE_EMB(                                                    \
        at::ScalarType::Float8_e4m3fnuz,                                      \
        _cache_t,                                                             \
        at::Float8_e4m3fnuz,                                                  \
        NAME,                                                                 \
        __VA_ARGS__)                                                          \
    default:                                                                  \
      AT_ERROR(#NAME, " not implemented for emb_t '", toString(_emb_t), "'"); \
  }


#define PRIVATE_CASE_TYPE_OUTPUT(                            \
    output_enum_type1,                                       \
    emb_enum_type1,                                          \
    cache_enum_type1,                                        \
    output_type1,                                            \
    NAME,                                                    \
    ...)                                                     \
  case output_enum_type1: {                                  \
    using output_t = output_type1;                           \
    _DISPATCH_EMB_CACHE_TYPES(                               \
        emb_enum_type1, cache_enum_type1, NAME, __VA_ARGS__) \
  }

#define DISPATCH_EMB_CACHE_OUTPUT_TYPES(                           \
    EMB_TYPE, CACHE_TYPE, OUTPUT_TYPE, NAME, ...)                  \
  [&] {                                                            \
    const at::ScalarType _output_t = OUTPUT_TYPE; /*            */ \
    const auto& emb_type = EMB_TYPE;                               \
    const auto& cache_type = CACHE_TYPE;                           \
    switch (_output_t) {                                           \
      PRIVATE_CASE_TYPE_OUTPUT(                                    \
          at::ScalarType::Half,                                    \
          emb_type,                                                \
          cache_type,                                              \
          at::Half,                                                \
          NAME,                                                    \
          __VA_ARGS__)                                             \
      PRIVATE_CASE_TYPE_OUTPUT(                                    \
          at::ScalarType::Float,                                   \
          emb_type,                                                \
          cache_type,                                              \
          float,                                                   \
          NAME,                                                    \
          __VA_ARGS__)                                             \
      PRIVATE_CASE_TYPE_OUTPUT(                                    \
          at::ScalarType::BFloat16,                                \
          emb_type,                                                \
          cache_type,                                              \
          at::BFloat16,                                            \
          NAME,                                                    \
          __VA_ARGS__)                                             \
      default:                                                     \
        AT_ERROR(                                                  \
            #NAME,                                                 \
            " not implemented for output_t '",                     \
            toString(_output_t),                                   \
            "'");                                                  \
    }                                                              \
  }()

#define PRIVATE_CASE_TYPE_CACHE(enum_type, type, ...) \
  case enum_type: {                                   \
    using cache_t = type;                             \
    return __VA_ARGS__();                             \
  }

#define PRIVATE_CASE_TYPE_CACHE_EMB(                                       \
    grad_enum_type, _cache_t, _emb_t, grad_cxx_type, NAME, ...)            \
  case grad_enum_type: {                                                   \
    using grad_t = grad_cxx_type;                                          \
    switch (_emb_t) {                                                      \
      PRIVATE_CASE_TYPE_EMB(                                               \
          at::ScalarType::Float, _cache_t, float, NAME, __VA_ARGS__)       \
      PRIVATE_CASE_TYPE_EMB(                                               \
          at::ScalarType::Half, _cache_t, at::Half, NAME, __VA_ARGS__)     \
      default:                                                             \
        AT_ERROR(                                                          \
            #NAME, " not implemented for emb_t '", toString(_emb_t), "'"); \
    }                                                                      \
  }


#define DISPATCH_EMB_GRAD_CACHE_TYPES(                                         \
    EMB_TYPE, GRAD_TYPE, CACHE_TYPE, NAME, ...)                                \
  [&] {                                                                        \
    const auto& emb_type = EMB_TYPE;                                           \
    const auto& grad_type = GRAD_TYPE;                                         \
    const auto& cache_type = CACHE_TYPE;                                       \
    at::ScalarType _emb_t = emb_type;                                          \
    at::ScalarType _grad_t = grad_type;                                        \
    at::ScalarType _cache_t = cache_type;                                      \
    switch (_grad_t) {                                                         \
      PRIVATE_CASE_TYPE_CACHE_EMB(                                             \
          at::ScalarType::Float, _cache_t, _emb_t, float, NAME, __VA_ARGS__)   \
      PRIVATE_CASE_TYPE_CACHE_EMB(                                             \
          at::ScalarType::Half, _cache_t, _emb_t, at::Half, NAME, __VA_ARGS__) \
      PRIVATE_CASE_TYPE_CACHE_EMB(                                             \
          at::ScalarType::BFloat16,                                            \
          _cache_t,                                                            \
          _emb_t,                                                              \
          at::BFloat16,                                                        \
          NAME,                                                                \
          __VA_ARGS__)                                                         \
      default:                                                                 \
        AT_ERROR(                                                              \
            #NAME, " not implemented for grad_t '", toString(_grad_t), "'");   \
    }                                                                          \
  }()
