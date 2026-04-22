/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

/*
 * BSD License
 * 
 * For FBGEMM software
 * 
 * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 * 
 *  * Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 
 *  * Neither the name Facebook nor the names of its contributors may be used to
 *    endorse or promote products derived from this software without specific
 *    prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 * ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <ATen/ATen.h>
#include <cstdint>

namespace fbgemm_utils {
  inline std::optional<int64_t> get_device_index_from_tensor(
      const at::Tensor& ten) {
    return {ten.device().index()};
  }

  inline bool torch_tensor_on_sycl_xpu_check(const at::Tensor& ten) {
    return ten.is_xpu();
  }

  inline std::string torch_tensor_device_name(const at::Tensor& ten) {
    return c10::DeviceTypeName(ten.device().type());
  }

  inline bool torch_tensor_undefined(const at::Tensor& ten) {
    return ten.defined();
  }

  inline bool torch_tensor_on_cpu_or_on_mtia_check(const at::Tensor& ten) {
    return ten.is_cpu() || ten.is_mtia();
  }

  inline bool torch_tensor_on_same_device_check(
      const at::Tensor& ten1,
      const std::optional<at::Tensor>& ten2) {
    return !ten2.has_value() || ten1.get_device() == ten2->get_device();
  }

  inline std::string torch_tensor_device_name(
      const std::optional<at::Tensor>& ten) {
    if (ten.has_value()) {
      return torch_tensor_device_name(ten.value());
    } else {
      return "N/A";
    }
  }

  inline bool torch_tensor_on_sycl_xpu_check(
      const std::optional<at::Tensor>& ten) {
    return !ten.has_value() || torch_tensor_on_sycl_xpu_check(ten.value());
  }

  #define TENSOR_ON_CPU_OR_MTIA(x)                                      \
    TORCH_CHECK(                                                        \
        torch_tensor_on_cpu_or_on_mtia_check(x),                        \
        #x " must be a CPU or MTIA tensor; it is currently on device ", \
        torch_tensor_device_name(x))

  #define TENSORS_EMPTY_OR_ON_SAME_DEVICE(x, y)                           \
    TORCH_CHECK(                                                          \
        torch_tensor_on_same_device_check(x, y) || (x.numel() == 0),      \
        #x " must be empty or a XPU tensor; it is currently on device ", \
        torch_tensor_device_name(x))

  #define TENSOR_ON_SYCL_XPU(x)                                  \
    TORCH_CHECK(                                                 \
        torch_tensor_on_sycl_xpu_check(x),                       \
        #x " must be a SYCL XPU tensor; it is currently on device ", \
        torch_tensor_device_name(x))

  // Generate constexpr array of variable names to improve diagnostic output and
  // raise a message if any non-empty tensor is not on a XPU or not on the same
  // XPU as all the other non-empty tensors.
  #define TENSORS_ON_SAME_SYCL_XPU_IF_NOT_OPTIONAL(...)                        \
    do {                                                                       \
      const auto tensors_on_same_xpu =                                         \
          tensor_on_same_xpu_if_not_optional_check(#__VA_ARGS__, __VA_ARGS__); \
      TORCH_CHECK(tensors_on_same_xpu.empty(), tensors_on_same_xpu);           \
    } while (false)

  inline at::Tensor aligned_grad_output_tensor_for_xpu_backwards(
      const at::Tensor& grad_output) {
    auto aligned_grad_output = grad_output;
    // FIXME: to support aligned memory access in Vec4T load/store function
    // 16 for FP32 and 8 for FP16
    if (!aligned_grad_output.is_contiguous()) {
      aligned_grad_output = aligned_grad_output.contiguous();
    }
    if (reinterpret_cast<uint64_t>(aligned_grad_output.data_ptr()) % 16 != 0) {
      aligned_grad_output =
          at::empty_like(aligned_grad_output).copy_(aligned_grad_output);
    }
    TORCH_CHECK(aligned_grad_output.is_contiguous());
    TORCH_CHECK(
        reinterpret_cast<uint64_t>(aligned_grad_output.data_ptr()) % 16 == 0);
    return aligned_grad_output;
  }

  template <typename... Tensors>
  std::string tensor_on_same_xpu_if_not_optional_check(
      const std::string& var_names_str,
      const Tensors&... tensors) {
    std::optional<int64_t> xpu_index;
    bool on_same_xpu = true;

    // Collect the GPU index of the first non-empty optional tensor and make sure
    // that all tensors are on this same index.
    (
        [&](const auto& tensor) {
          if (!torch_tensor_undefined(tensor)) {
            return;
          }
          if (!torch_tensor_on_sycl_xpu_check(tensor)) {
            on_same_xpu = false;
            return;
          }
          const auto my_xpu_index = get_device_index_from_tensor(tensor);
          if (my_xpu_index) {
            if (!xpu_index) {
              xpu_index = my_xpu_index;
            } else if (*xpu_index != my_xpu_index) {
              on_same_xpu = false;
            }
          }
        }(tensors),
        ...);

    if (on_same_xpu) {
      return "";
    }

    std::vector<std::string> var_names;
    {
      std::string temp;
      for (const auto& x : var_names_str) {
        if (x == ',') {
          var_names.push_back(temp);
          temp = "";
        } else {
          temp.push_back(x);
        }
      }
      var_names.push_back(temp);
    }

    // Not all the tensors on a XPU or on the same XPU, generate a message.
    std::string msg = "Not all tensors were on the same XPU: ";
    size_t current_idx = 0;
    (
        [&](const auto& tensor) {
          if (current_idx > 0) {
            msg.append(", ");
          }
          msg.append(
              var_names.at(current_idx++) + "(" +
              torch_tensor_device_name(tensor));
          const auto xpu_device_index = get_device_index_from_tensor(tensor);
          if (xpu_device_index) {
            msg.append(":" + std::to_string(*xpu_device_index));
          }
          msg.append(")");
        }(tensors),
        ...);

    return msg;
  }
} // namespace fbgemm_utils
