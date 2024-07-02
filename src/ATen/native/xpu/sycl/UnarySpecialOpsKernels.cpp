#include <ATen/ATen.h>

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/TensorIterator.h>
#include <c10/core/ScalarType.h>

#include <ATen/native/xpu/sycl/Loops.h>

namespace at::native::xpu {

template <typename scalar_t>
struct SigmoidFunctor {
  scalar_t operator()(scalar_t a) const {
    using opmath_t = at::opmath_type<scalar_t>;
    const auto one = opmath_t{1.0};
    return one / (one + std::exp(-static_cast<opmath_t>(a)));
  }
};

void sigmoid_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND3(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      kComplexHalf,
      iter.common_dtype(),
      "sigmoid_xpu",
      [&]() { gpu_kernel(iter, SigmoidFunctor<scalar_t>()); });
}

template <typename scalar_t>
struct ErfFunctor {
  scalar_t operator()(scalar_t a) const {
    return std::erf(float(a));
  }
};

template <>
struct ErfFunctor<double> {
  double operator()(double a) const {
    return std::erf(a);
  }
};

template <typename scalar_t>
struct ErfcFunctor {
  scalar_t operator()(scalar_t a) const {
    return std::erfc(float(a));
  }
};

template <>
struct ErfcFunctor<double> {
  double operator()(double a) const {
    return std::erfc(a);
  }
};

void erf_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::Half,
      ScalarType::BFloat16,
      iter.common_dtype(),
      "erf_xpu",
      [&]() { gpu_kernel(iter, ErfFunctor<scalar_t>()); });
}

void erfc_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::Half,
      ScalarType::BFloat16,
      iter.common_dtype(),
      "erfc_xpu",
      [&]() { gpu_kernel(iter, ErfcFunctor<scalar_t>()); });
}

template <typename scalar_t>
struct ErfinvFunctor {
  using opmath_type = at::opmath_type<scalar_t>;

  scalar_t operator()(scalar_t in) const {
    scalar_t out;
    opmath_type z, num, dem;

    auto x = static_cast<opmath_type>(in);
    if (std::fabs(x) > 1.0f) {
      out = static_cast<scalar_t>(NAN);
      return out;
    }
    if (std::fabs(x) == 1.0f) {
      out = static_cast<scalar_t>(
          (std::copysign(1.0, static_cast<double>(x))) *
          (std::numeric_limits<double>::infinity()));
      return out;
    }
    if (std::fabs(x) <= 0.7f) {
      z = x * x;
      num = (((a_[3] * z + a_[2]) * z + a_[1]) * z + a_[0]);
      dem =
          ((((b_[3] * z + b_[2]) * z + b_[1]) * z + b_[0]) * z +
           static_cast<opmath_type>(1.0));
      out = x * num / dem;
    } else {
      z = static_cast<opmath_type>(
          std::sqrt(-std::log((1.0 - std::fabs(x)) / 2.0)));
      num = ((c_[3] * z + c_[2]) * z + c_[1]) * z + c_[0];
      dem = (d_[1] * z + d_[0]) * z + static_cast<opmath_type>(1.0);
      out = static_cast<scalar_t>(
          static_cast<opmath_type>(std::copysign(1.0, static_cast<double>(x))) *
          num / dem);
    }
    out = out -
        static_cast<scalar_t>(
              (std::erf(static_cast<double>(out)) - x) /
              ((2.0 / std::sqrt(PI_f64_)) * std::exp(-x * x)));
    out = out -
        static_cast<scalar_t>(
              (std::erf(static_cast<double>(out)) - x) /
              ((2.0 / std::sqrt(PI_f64_)) * std::exp(-x * x)));
    return out;
  }

  static constexpr double PI_f64_ = 3.14159265358979323846;
  static constexpr std::array<opmath_type, 4> a_ = {
      0.886226899,
      -1.645349621,
      0.914624893,
      -0.140543331};
  static constexpr std::array<opmath_type, 4> b_ = {
      -2.118377725,
      1.442710462,
      -0.329097515,
      0.012229801};
  static constexpr std::array<opmath_type, 4> c_ = {
      -1.970840454,
      -1.624906493,
      3.429567803,
      1.641345311};
  static constexpr std::array<opmath_type, 2> d_ = {3.543889200, 1.637067800};
};

void erfinv_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::Half,
      ScalarType::BFloat16,
      iter.common_dtype(),
      "erfinv_xpu",
      [&]() { gpu_kernel(iter, ErfinvFunctor<scalar_t>()); });
}

} // namespace at::native::xpu
