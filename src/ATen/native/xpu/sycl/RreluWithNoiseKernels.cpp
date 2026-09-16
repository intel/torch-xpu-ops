/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/Dispatch.h>
#include <ATen/native/Math.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/DistributionTemplates.h>
#include <ATen/xpu/XPUGeneratorImpl.h>

#include <ATen/native/xpu/sycl/RreluWithNoiseKernels.h>

#include <concepts>

namespace at::native::xpu {

template <typename scalar_t, int unroll_factor, typename transform_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void rrelu_with_noise_kernel_fn(
    int numel,
    PhiloxXpuState philox_args,
    scalar_t* output,
    const scalar_t* input,
    scalar_t* noise,
    double lower,
    double upper,
    transform_t random_func) {
  auto item = syclext::this_work_item::get_nd_item<1>();
  auto seeds = at::xpu::philox::unpack(philox_args);
  int group_size = item.get_local_range(0);
  int num_groups = item.get_group_range(0);
  int idx = item.get_global_linear_id();

  randStatePhilox4_32_10_t state;
  rand_init(std::get<0>(seeds), idx, std::get<1>(seeds), &state);

  int full_tile_work_size = group_size * num_groups * unroll_factor;
  int rounded_size =
      ((numel - 1) / full_tile_work_size + 1) * full_tile_work_size;
  double range = upper - lower;

  for (int linear_index = idx; linear_index < rounded_size;
       linear_index += full_tile_work_size) {
    auto rand = random_func(&state);

    // ensure that (&rand.x)[ii] is safe
    static_assert(sizeof(rand) / sizeof(rand.x) == unroll_factor);

#pragma unroll
    for (int ii = 0; ii < unroll_factor; ii++) {
      int li = linear_index + group_size * num_groups * ii;
      if (li >= numel) {
        continue;
      }
      scalar_t r = static_cast<scalar_t>((&rand.x)[ii]);
      r = r * range + lower;
      if (input[li] <= 0) {
        output[li] = input[li] * r;
        noise[li] = r;
      } else {
        output[li] = input[li];
        noise[li] = static_cast<scalar_t>(1);
      }
    }
    // Some state (e.g. MTGP32) need to add barrier there.
  }
}

template <typename scalar_t>
inline void _rrelu_with_noise_xpu_train(
    Tensor& output,
    const Tensor& input_,
    Tensor& noise_,
    const Scalar& lower_,
    const Scalar& upper_,
    const std::optional<Generator>& generator) {
  auto input = input_.contiguous();
  auto noise = noise_.contiguous();
  Tensor tmp_output = output.contiguous();

  int64_t numel = input.numel();
  constexpr int unroll_factor = std::is_same_v<scalar_t, double> ? 2 : 4;
  auto execution_policy = calc_execution_policy<unroll_factor>(numel);

  auto counter_offset = std::get<0>(execution_policy);
  auto num_groups = std::get<1>(execution_policy);
  auto group_size = std::get<2>(execution_policy);

  auto gen = get_generator_or_default<at::XPUGeneratorImpl>(
      generator, at::xpu::detail::getDefaultXPUGenerator());
  PhiloxXpuState rng_engine_inputs;
  {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_xpu_state(counter_offset);
  }

  const scalar_t* input_data = input.const_data_ptr<scalar_t>();
  scalar_t* noise_data = noise.mutable_data_ptr<scalar_t>();
  scalar_t* output_data = tmp_output.mutable_data_ptr<scalar_t>();

  double lower = lower_.to<double>();
  double upper = upper_.to<double>();

  if constexpr (std::same_as<scalar_t, double>) {
    templates::xpu::Uniform2DistributionFunctor tfn;
    constexpr auto kptr =
        rrelu_with_noise_kernel_fn<scalar_t, 2, decltype(tfn)>;
    sycl_kernel_submit<kptr>(
        num_groups * group_size,
        group_size,
        getCurrentSYCLQueue(),
        0,
        numel,
        rng_engine_inputs,
        output_data,
        input_data,
        noise_data,
        lower,
        upper,
        tfn);
  } else {
    // half and float
    templates::xpu::Uniform4DistributionFunctor tfn;
    constexpr auto kptr =
        rrelu_with_noise_kernel_fn<scalar_t, 4, decltype(tfn)>;
    sycl_kernel_submit<kptr>(
        num_groups * group_size,
        group_size,
        getCurrentSYCLQueue(),
        0,
        numel,
        rng_engine_inputs,
        output_data,
        input_data,
        noise_data,
        lower,
        upper,
        tfn);
  }

  if (!output.is_contiguous()) {
    output.copy_(tmp_output);
  }
}

Tensor& rrelu_with_noise_kernel(
    const Tensor& self,
    Tensor& noise,
    const Scalar& lower,
    const Scalar& upper,
    bool training,
    std::optional<Generator> generator,
    Tensor& output) {
  at::native::resize_output(output, self.sizes());

  if (self.numel() == 0) {
    return output;
  }

  TensorArg self_arg{self, "self", 1}, noise_arg{noise, "noise", 2},
      output_arg{output, "output", 3};
  checkAllSameGPU(
      "rrelu_with_noise_out_xpu", {self_arg, noise_arg, output_arg});

  if (training) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        self.scalar_type(),
        "rrelu_with_noise_out_xpu",
        [&] {
          _rrelu_with_noise_xpu_train<scalar_t>(
              output, self, noise, lower, upper, generator);
        });
  } else {
    auto lower_tensor = lower.to<double>();
    auto upper_tensor = upper.to<double>();
    Scalar negative_slope = (lower_tensor + upper_tensor) / 2;
    at::leaky_relu_out(output, self, negative_slope);
  }
  return output;
}

} // namespace at::native::xpu
