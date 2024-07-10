#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>

#include <ATen/native/xpu/sycl/Loops.h>

namespace at::native::xpu {

template <typename scalar_t>
struct AddcmulFunctor {
  using accscalar_t = at::acc_type<scalar_t, true>;
  scalar_t operator()(scalar_t a, scalar_t b, scalar_t c) const {
    return static_cast<accscalar_t>(a) +
        alpha_ * static_cast<accscalar_t>(b) * static_cast<accscalar_t>(c);
  }

  AddcmulFunctor(accscalar_t alpha) : alpha_(alpha) {}

 private:
  accscalar_t alpha_;
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

void addcmul_kernel(TensorIterator& iter, Scalar value) {
  auto dtype = iter.common_dtype();
  if (at::isComplexType(dtype)) {
    AT_DISPATCH_COMPLEX_TYPES(dtype, "addcmul_xpu", [&]() {
      auto alpha = value.to<scalar_t>();
      gpu_kernel(iter, AddcmulComplexFunctor<scalar_t>(alpha));
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        iter.dtype(),
        "addcmul_xpu",
        [&]() {
          using accscalar_t = at::acc_type<scalar_t, true>;
          auto alpha = value.to<accscalar_t>();
          gpu_kernel(iter, AddcmulFunctor<scalar_t>(alpha));
        });
  }
}

template <typename scalar_t>
struct AddcdivFunctor {
  using accscalar_t = at::acc_type<scalar_t, true>;
  scalar_t operator()(scalar_t a, scalar_t b, scalar_t c) const {
    return a + alpha_ * (b / static_cast<accscalar_t>(c));
  }

  AddcdivFunctor(accscalar_t alpha) : alpha_(alpha) {}

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

void addcdiv_kernel(TensorIterator& iter, Scalar value) {
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
        iter.dtype(),
        "addcdiv_xpu",
        [&]() {
          using accscalar_t = at::acc_type<scalar_t, true>;
          auto alpha = value.to<accscalar_t>();
          AddcdivFunctor<scalar_t> f(alpha);
          gpu_kernel(iter, f);
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

} // namespace at::native::xpu
