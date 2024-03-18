#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/NumericUtils.h>
#include <ATen/native/TensorIterator.h>

#include <aten/sycl/Loops.h>

namespace at {
namespace native {
namespace xpu {

template <typename opmath_t>
struct EqFunctor {
  opmath_t operator()(opmath_t a, opmath_t b) const {
    return a == b;
  }
};

template <typename opmath_t>
struct NeFunctor {
  opmath_t operator()(opmath_t a, opmath_t b) const {
    return a != b;
  }
};

template <typename opmath_t>
struct LtFunctor {
  opmath_t operator()(opmath_t a, opmath_t b) const {
    return a < b;
  }
};

template <typename opmath_t>
struct LeFunctor {
  opmath_t operator()(opmath_t a, opmath_t b) const {
    return a <= b;
  }
};

template <typename opmath_t>
struct GtFunctor {
  opmath_t operator()(opmath_t a, opmath_t b) const {
    return a > b;
  }
};

template <typename opmath_t>
struct GeFunctor {
  opmath_t operator()(opmath_t a, opmath_t b) const {
    return a >= b;
  }
};

template <typename opmath_t>
struct ClampFunctor {
  opmath_t operator()(opmath_t v) const {
    if (_isnan(v)) {
      return v;
    } else {
      return std::min(std::max(v, lower_), upper_);
    }
  }
  ClampFunctor(opmath_t lower, opmath_t upper) : lower_(lower), upper_(upper) {}

 private:
  opmath_t lower_;
  opmath_t upper_;
};

template <typename opmath_t>
struct ClampMinFunctor {
  opmath_t operator()(opmath_t v) const {
    if (_isnan(v)) {
      return v;
    } else {
      return std::max(v, lower_);
    }
  }
  ClampMinFunctor(opmath_t lower) : lower_(lower) {}

 private:
  opmath_t lower_;
};

template <typename opmath_t>
struct ClampMaxFunctor {
  opmath_t operator()(opmath_t v) const {
    if (_isnan(v)) {
      return v;
    } else {
      return std::min(v, upper_);
    }
  }
  ClampMaxFunctor(opmath_t upper) : upper_(upper) {}

 private:
  opmath_t upper_;
};

template <typename scalar_t>
struct WhereFunctor {
  scalar_t operator()(bool cond_val, scalar_t self_val, scalar_t other_val)
      const {
    return cond_val ? self_val : other_val;
  }
};

void eq_kernel(TensorIteratorBase& iter) {
  auto common_dtype = iter.common_dtype();
  if (common_dtype == kComplexHalf) {
    using scalar_t = c10::complex<c10::Half>;
    using opmath_t = opmath_type<scalar_t>;
    opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(
        iter, EqFunctor<opmath_t>());
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
        kHalf, kBFloat16, kBool, iter.common_dtype(), "eq_xpu", [&]() {
          using opmath_t = opmath_type<scalar_t>;
          opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(
              iter, EqFunctor<opmath_t>());
        });
  }
}

void ne_kernel(TensorIteratorBase& iter) {
  auto common_dtype = iter.common_dtype();
  if (common_dtype == kComplexHalf) {
    using scalar_t = c10::complex<c10::Half>;
    using opmath_t = opmath_type<scalar_t>;
    opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(
        iter, NeFunctor<opmath_t>());
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
        kHalf, kBFloat16, kBool, iter.common_dtype(), "ne_xpu", [&]() {
          using opmath_t = opmath_type<scalar_t>;
          opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(
              iter, NeFunctor<opmath_t>());
        });
  }
}

void lt_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_ALL_TYPES_AND3(
      kHalf, kBFloat16, kBool, iter.common_dtype(), "lt_xpu", [&]() {
        using opmath_t = opmath_type<scalar_t>;
        opmath_gpu_kernel_with_scalars<scalar_t>(iter, LtFunctor<opmath_t>());
      });
}

void le_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_ALL_TYPES_AND3(
      kHalf, kBFloat16, kBool, iter.common_dtype(), "le_xpu", [&]() {
        using opmath_t = opmath_type<scalar_t>;
        opmath_gpu_kernel_with_scalars<scalar_t>(iter, LeFunctor<opmath_t>());
      });
}

void gt_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_ALL_TYPES_AND3(
      kHalf, kBFloat16, kBool, iter.common_dtype(), "gt_xpu", [&]() {
        using opmath_t = opmath_type<scalar_t>;
        opmath_gpu_kernel_with_scalars<scalar_t>(iter, GtFunctor<opmath_t>());
      });
}

void ge_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_ALL_TYPES_AND3(
      kHalf, kBFloat16, kBool, iter.common_dtype(), "ge_xpu", [&]() {
        using opmath_t = opmath_type<scalar_t>;
        opmath_gpu_kernel_with_scalars<scalar_t>(iter, GeFunctor<opmath_t>());
      });
}

void clamp_kernel(
    TensorIteratorBase& iter,
    const Scalar& min_value,
    const Scalar& max_value) {
  AT_DISPATCH_ALL_TYPES_AND2(
      kHalf, kBFloat16, iter.dtype(), "clamp_xpu", [&]() {
        auto lower = min_value.to<scalar_t>();
        auto upper = max_value.to<scalar_t>();
        gpu_kernel(iter, ClampFunctor<scalar_t>(lower, upper));
      });
}

void clamp_min_kernel(TensorIteratorBase& iter, const Scalar& min_value) {
  AT_DISPATCH_ALL_TYPES_AND2(
      kHalf, kBFloat16, iter.dtype(), "clamp_min_xpu", [&]() {
        auto lower = min_value.to<scalar_t>();
        gpu_kernel(iter, ClampMinFunctor<scalar_t>(lower));
      });
}

void clamp_max_kernel(TensorIteratorBase& iter, const Scalar& max_value) {
  AT_DISPATCH_ALL_TYPES_AND2(
      kHalf, kBFloat16, iter.dtype(), "clamp_max_xpu", [&]() {
        auto upper = max_value.to<scalar_t>();
        gpu_kernel(iter, ClampMaxFunctor<scalar_t>(upper));
      });
}

void where_kernel(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
      kComplexHalf, kHalf, kBFloat16, kBool, iter.dtype(), "where_xpu", [&] {
        gpu_kernel(iter, WhereFunctor<scalar_t>());
      });
}

} // namespace xpu
} // namespace native
} // namespace at
