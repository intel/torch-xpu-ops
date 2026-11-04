/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <comm/xpu_aten.h>

#include <ATen/native/xpu/sycl/DeviceAddCmulCdiv.h>
#include <ATen/native/xpu/sycl/Loops.h>
#include <ATen/native/xpu/sycl/PointwiseOpsKernels.h>

namespace at::native::xpu {

// Preserve the common dtype rounding before storing to a different output
// dtype. Using float I/O for reduced precision also retains Loops.h's mixed FP
// fast path.
template <typename scalar_t, typename io_t>
io_t addc_cast_result(at::acc_type_device<scalar_t, kXPU> value) {
  if constexpr (std::is_same_v<scalar_t, Half> && !std::is_same_v<io_t, Half>) {
    // Use the bit conversion to preserve intermediate half rounding without
    // a float -> half -> float round trip that SYCL can eliminate.
    return static_cast<io_t>(
        c10::detail::fp16_ieee_to_fp32_value(Half(value).x));
  }
  return static_cast<io_t>(static_cast<scalar_t>(value));
}

template <typename scalar_t, typename io_t, bool cast_inputs>
struct AddcInputCaster {
  explicit AddcInputCaster(uint8_t) {}

  at::acc_type_device<scalar_t, kXPU> load(io_t value, int) const {
    return static_cast<at::acc_type_device<scalar_t, kXPU>>(value);
  }
};

template <typename scalar_t, typename io_t>
struct AddcInputCaster<scalar_t, io_t, true> {
  using accscalar_t = at::acc_type_device<scalar_t, kXPU>;

  explicit AddcInputCaster(uint8_t mask) : mask_(mask) {}

  accscalar_t load(io_t value, int operand) const {
    auto input = static_cast<accscalar_t>(value);
    return mask_ & (1 << operand)
        ? addc_cast_result<scalar_t, accscalar_t>(input)
        : input;
  }

 private:
  uint8_t mask_;
};

template <
    template <typename, typename, bool>
    class functor_t,
    typename scalar_t,
    typename io_t,
    typename... Args>
void launch_addc_kernel_with_cast(TensorIteratorBase& iter, Args... args) {
  if constexpr (
      std::is_same_v<scalar_t, Half> || std::is_same_v<scalar_t, BFloat16>) {
    // Zero-dimensional and integral inputs may differ from the common dtype.
    uint8_t mask = 0;
    for (int i = 1; i < iter.ntensors(); ++i) {
      if (iter.dtype(i) != iter.common_dtype()) {
        mask |= 1 << (i - 1);
      }
    }
    if (mask != 0) {
      gpu_kernel(iter, functor_t<scalar_t, io_t, true>(args..., mask));
      return;
    }
  }
  gpu_kernel(iter, functor_t<scalar_t, io_t, false>(args...));
}

template <
    template <typename, typename, bool>
    class functor_t,
    typename scalar_t,
    typename... Args>
void launch_addc_kernel(TensorIteratorBase& iter, Args... args) {
  if constexpr (
      std::is_same_v<scalar_t, Half> || std::is_same_v<scalar_t, BFloat16> ||
      std::is_same_v<scalar_t, float>) {
    // Double I/O retains the smaller unroll factor for noncontiguous outputs.
    // Arithmetic and result rounding still use the common dtype.
    if (iter.dtype() == ScalarType::Double) {
      launch_addc_kernel_with_cast<functor_t, scalar_t, double>(iter, args...);
      return;
    }
  }
  if constexpr (
      std::is_same_v<scalar_t, Half> || std::is_same_v<scalar_t, BFloat16>) {
    if (iter.dtype() != iter.common_dtype()) {
      launch_addc_kernel_with_cast<functor_t, scalar_t, float>(iter, args...);
      return;
    }
  }
  gpu_kernel(iter, functor_t<scalar_t, scalar_t, false>(args...));
}

template <typename scalar_t, typename io_t = scalar_t, bool cast_inputs = false>
struct AddcmulFunctor : AddcInputCaster<scalar_t, io_t, cast_inputs> {
  using input_caster_t = AddcInputCaster<scalar_t, io_t, cast_inputs>;
  using accscalar_t = at::acc_type_device<scalar_t, kXPU>;
  io_t operator()(io_t a, io_t b, io_t c) const {
    auto input = this->load(a, 0);
    auto t1 = this->load(b, 1);
    auto t2 = this->load(c, 2);
    return addc_cast_result<scalar_t, io_t>(pointwise_op_impl(
        input, t1, t2, alpha_, std::multiplies<accscalar_t>{}));
  }

  AddcmulFunctor(accscalar_t alpha, uint8_t mask = 0)
      : input_caster_t(mask), alpha_(alpha) {}

 private:
  accscalar_t alpha_;
};

template <typename scalar_t, typename io_t = scalar_t, bool cast_inputs = false>
struct AddcmulCpuScalarFunctor : AddcInputCaster<scalar_t, io_t, cast_inputs> {
  using input_caster_t = AddcInputCaster<scalar_t, io_t, cast_inputs>;
  using accscalar_t = at::acc_type_device<scalar_t, kXPU>;
  io_t operator()(io_t a, io_t b) const {
    return addc_cast_result<scalar_t, io_t>(
        this->load(a, 0) + alpha_ * this->load(b, 1) * c_val_);
  }

  AddcmulCpuScalarFunctor(
      accscalar_t alpha,
      accscalar_t c_val,
      uint8_t mask = 0)
      : input_caster_t(mask), alpha_(alpha), c_val_(c_val) {}

 private:
  accscalar_t alpha_;
  accscalar_t c_val_;
};

template <typename scalar_t>
struct AddcmulComplexFunctor {
  scalar_t operator()(scalar_t a, scalar_t b, scalar_t c) const {
    return a + alpha_ * b * c;
  }

  AddcmulComplexFunctor(scalar_t alpha) : alpha_(alpha) {}

 private:
  scalar_t alpha_;
};

template <typename scalar_t>
struct AddcmulComplexCpuScalarFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    return a + alpha_ * b * c_val_;
  }

  AddcmulComplexCpuScalarFunctor(scalar_t alpha, scalar_t c_val)
      : alpha_(alpha), c_val_(c_val) {}

 private:
  scalar_t alpha_;
  scalar_t c_val_;
};

void addcmul_kernel(TensorIteratorBase& iter, const Scalar& value) {
  TORCH_CHECK(
      !iter.is_cpu_scalar(1),
      "CPU Scalar support for self argument is not supported when "
      "calling addcmul on XPU tensors.");

  TORCH_CHECK(
      !iter.is_cpu_scalar(2),
      "CPU Scalar support for tensor1 argument is not supported when "
      "calling addcmul on XPU tensors. "
      "However, CPU Scalar support for tensor2 is supported, "
      "please swap your tensor1 and tensor2 terms.");

  auto dtype = iter.common_dtype();
  if (iter.is_cpu_scalar(3)) {
    if (at::isComplexType(dtype)) {
      AT_DISPATCH_COMPLEX_TYPES(dtype, "addcmul_xpu", [&]() {
        auto alpha = value.to<scalar_t>();
        auto c_val = iter.scalar_value<scalar_t>(3);
        iter.remove_operand(3);
        gpu_kernel(
            iter, AddcmulComplexCpuScalarFunctor<scalar_t>(alpha, c_val));
      });
    } else {
      AT_DISPATCH_ALL_TYPES_AND2(
          at::ScalarType::Half,
          at::ScalarType::BFloat16,
          dtype,
          "addcmul_xpu",
          [&]() {
            using accscalar_t = at::acc_type_device<scalar_t, kXPU>;
            auto alpha = value.to<accscalar_t>();
            auto c_val = iter.scalar_value<scalar_t>(3);
            iter.remove_operand(3);
            launch_addc_kernel<AddcmulCpuScalarFunctor, scalar_t>(
                iter, alpha, c_val);
          });
    }
  } else if (at::isComplexType(dtype)) {
    AT_DISPATCH_COMPLEX_TYPES(dtype, "addcmul_xpu", [&]() {
      auto alpha = value.to<scalar_t>();
      gpu_kernel(iter, AddcmulComplexFunctor<scalar_t>(alpha));
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        dtype,
        "addcmul_xpu",
        [&]() {
          using accscalar_t = at::acc_type_device<scalar_t, kXPU>;
          auto alpha = value.to<accscalar_t>();
          launch_addc_kernel<AddcmulFunctor, scalar_t>(iter, alpha);
        });
  }
}

template <typename scalar_t, typename io_t = scalar_t, bool cast_inputs = false>
struct AddcdivFunctor : AddcInputCaster<scalar_t, io_t, cast_inputs> {
  using input_caster_t = AddcInputCaster<scalar_t, io_t, cast_inputs>;
  using accscalar_t = at::acc_type_device<scalar_t, kXPU>;
  io_t operator()(io_t a, io_t b, io_t c) const {
    auto input = this->load(a, 0);
    auto t1 = this->load(b, 1);
    auto t2 = this->load(c, 2);
    return addc_cast_result<scalar_t, io_t>(input + alpha_ * (t1 / t2));
  }

  AddcdivFunctor(accscalar_t alpha, uint8_t mask = 0)
      : input_caster_t(mask), alpha_(alpha) {}

 private:
  accscalar_t alpha_;
};

template <typename scalar_t>
struct AddcdivComplexFunctor {
  scalar_t operator()(scalar_t a, scalar_t b, scalar_t c) const {
    return a + alpha_ * (b / c);
  }

  AddcdivComplexFunctor(scalar_t alpha) : alpha_(alpha) {}

 private:
  scalar_t alpha_;
};

void addcdiv_kernel(TensorIteratorBase& iter, const Scalar& value) {
  auto dtype = iter.common_dtype();
  if (at::isComplexType(dtype)) {
    AT_DISPATCH_COMPLEX_TYPES(dtype, "addcdiv_xpu", [&]() {
      auto alpha = value.to<scalar_t>();
      AddcdivComplexFunctor<scalar_t> f(alpha);
      gpu_kernel(iter, f);
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        dtype,
        "addcdiv_xpu",
        [&]() {
          using accscalar_t = at::acc_type_device<scalar_t, kXPU>;
          auto alpha = value.to<accscalar_t>();
          launch_addc_kernel<AddcdivFunctor, scalar_t>(iter, alpha);
        });
  }
}

template <typename scalar_t>
struct MSEBackwardFunctor {
  scalar_t operator()(scalar_t a, scalar_t b, scalar_t c) const {
    return alpha_ * (a - b) * c;
  }
  MSEBackwardFunctor(scalar_t alpha) : alpha_(alpha) {}

 private:
  scalar_t alpha_;
};

void mse_backward_kernel(TensorIterator& iter, const Scalar& value) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "mse_backward_xpu",
      [&]() {
        auto alpha = value.to<scalar_t>();
        gpu_kernel(iter, MSEBackwardFunctor<scalar_t>(alpha));
      });
}

template <typename scalar_t>
struct SmoothL1BackwardFunctor {
  scalar_t operator()(scalar_t input, scalar_t target, scalar_t grad_output)
      const {
    const auto x = input - target;
    if (x < -beta_val)
      return -norm_val * grad_output;
    else if (x > beta_val)
      return norm_val * grad_output;
    else
      return norm_val * x * grad_output / beta_val;
  }
  SmoothL1BackwardFunctor(scalar_t norm_val, scalar_t beta_val)
      : norm_val(norm_val), beta_val(beta_val) {}

 private:
  scalar_t norm_val;
  scalar_t beta_val;
};

void smooth_l1_backward_kernel(
    TensorIterator& iter,
    const Scalar& norm,
    double beta) {
  AT_DISPATCH_ALL_TYPES_AND2(
      kHalf,
      kBFloat16,
      iter.dtype(),
      "smooth_l1_backward_xpu",
      [&iter, &norm, beta] {
        auto norm_val = norm.to<scalar_t>();
        scalar_t beta_val(beta);
        SmoothL1BackwardFunctor<scalar_t> f(norm_val, beta_val);
        gpu_kernel(iter, f);
      });
}

template <typename scalar_t>
struct HuberBackwardFunctor {
  scalar_t operator()(scalar_t input, scalar_t target, scalar_t grad_output)
      const {
    const auto x = input - target;
    if (x < -delta_val_) {
      return -norm_val_ * grad_output * delta_val_;
    } else if (x > delta_val_) {
      return norm_val_ * grad_output * delta_val_;
    } else {
      return norm_val_ * x * grad_output;
    }
  }
  HuberBackwardFunctor(scalar_t norm_val, scalar_t delta_val)
      : norm_val_(norm_val), delta_val_(delta_val) {}

 private:
  scalar_t norm_val_;
  scalar_t delta_val_;
};

void huber_backward_kernel(
    TensorIterator& iter,
    const Scalar& norm,
    double delta) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kBFloat16,
      kHalf,
      iter.dtype(),
      "huber_backward_xpu",
      [&iter, &norm, delta] {
        auto norm_val = norm.to<scalar_t>();
        scalar_t delta_val(delta);
        gpu_kernel(iter, HuberBackwardFunctor<scalar_t>(norm_val, delta_val));
      });
}

} // namespace at::native::xpu
