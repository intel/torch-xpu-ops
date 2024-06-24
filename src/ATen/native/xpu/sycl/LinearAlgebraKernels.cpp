#include <ATen/Dispatch.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/Loops.h>
#include <ATen/native/xpu/sycl/Reduce.h>
#include <ATen/ops/imag.h>
#include <c10/core/ScalarType.h>

namespace at::native::xpu {

template <typename scalar_t>
struct AddrKernelFunctorForBetaIsFalse {
  scalar_t operator()(scalar_t self_val, scalar_t vec1_val, scalar_t vec2_val)
      const {
    return alpha_val_ && vec1_val && vec2_val;
  }
  AddrKernelFunctorForBetaIsFalse(scalar_t alpha_val) : alpha_val_(alpha_val) {}

 private:
  scalar_t alpha_val_;
};

template <typename scalar_t>
struct AddrKernelFunctorForBetaIsTrue {
  scalar_t operator()(scalar_t self_val, scalar_t vec1_val, scalar_t vec2_val)
      const {
    return (beta_val_ && self_val) || (alpha_val_ && vec1_val && vec2_val);
  }

  AddrKernelFunctorForBetaIsTrue(scalar_t alpha_val, scalar_t beta_val)
      : alpha_val_(alpha_val), beta_val_(beta_val) {}

 private:
  scalar_t alpha_val_;
  scalar_t beta_val_;
};

template <typename scalar_t>
struct AddrKernelFunctorForBetaIsZero {
  scalar_t operator()(scalar_t self_val, scalar_t vec1_val, scalar_t vec2_val)
      const {
    return alpha_val_ * vec1_val * vec2_val;
  }
  AddrKernelFunctorForBetaIsZero(scalar_t alpha_val) : alpha_val_(alpha_val) {}

 private:
  scalar_t alpha_val_;
};

template <typename scalar_t>
struct AddrKernelFunctorForBetaIsNotZero {
  scalar_t operator()(scalar_t self_val, scalar_t vec1_val, scalar_t vec2_val)
      const {
    return beta_val_ * self_val + alpha_val_ * vec1_val * vec2_val;
  }

  AddrKernelFunctorForBetaIsNotZero(scalar_t alpha_val, scalar_t beta_val)
      : alpha_val_(alpha_val), beta_val_(beta_val) {}

 private:
  scalar_t alpha_val_;
  scalar_t beta_val_;
};

void addr_kernel(
    TensorIterator& iter,
    const Scalar& beta,
    const Scalar& alpha) {
  if (iter.dtype() == at::ScalarType::Bool) {
    using scalar_t = bool;
    auto beta_val = beta.to<scalar_t>();
    auto alpha_val = alpha.to<scalar_t>();

    // when beta is false, values in self should be ignored,
    // nans and infs in self should not propagate.
    if (beta_val == false) {
      AddrKernelFunctorForBetaIsFalse<scalar_t> f(alpha_val);
      gpu_kernel(iter, f);
    } else {
      AddrKernelFunctorForBetaIsTrue<scalar_t> f(alpha_val, beta_val);
      gpu_kernel(iter, f);
    }
    return;
  }

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      kBFloat16, kHalf, iter.dtype(), "addr_xpu", [&] {
        auto beta_val = beta.to<scalar_t>();
        auto alpha_val = alpha.to<scalar_t>();

        scalar_t zero_val(0);
        // when beta==0, values in self should be ignored,
        // nans and infs in self should not propagate.
        if (beta_val == zero_val) {
          AddrKernelFunctorForBetaIsZero<scalar_t> f(alpha_val);
          gpu_kernel(iter, f);
        } else {
          AddrKernelFunctorForBetaIsNotZero<scalar_t> f(alpha_val, beta_val);
          gpu_kernel(iter, f);
        }
      });
}
// This reduction accumulates results as the type `acc_t`. By default, when
// `scalar_t` is complex, `acc_t` is the downgraded real number type.
// Otherwise, `acc_t` and `scalar_t` are the same type.
template <
    typename scalar_t,
    typename acc_t = typename scalar_value_type<scalar_t>::type,
    typename out_t = typename scalar_value_type<scalar_t>::type>
void norm_kernel_xpu_impl(TensorIterator& iter, double p) {
  if (p == static_cast<double>(0)) {
    gpu_reduce_kernel<scalar_t, out_t>(
        iter, NormZeroOps<scalar_t, acc_t, out_t>(), 0);
  } else if (p == static_cast<double>(1)) {
    gpu_reduce_kernel<scalar_t, out_t>(
        iter, NormOneOps<scalar_t, acc_t, out_t>(), 0);
  } else if (p == static_cast<double>(2)) {
    gpu_reduce_kernel<scalar_t, out_t>(
        iter, NormTwoOps<scalar_t, acc_t, out_t>(), 0);
  } else if (p == static_cast<double>(INFINITY)) {
    gpu_reduce_kernel<scalar_t, out_t>(
        iter, AbsMaxOps<scalar_t, acc_t, out_t>(), 0);
  } else if (p == static_cast<double>(-INFINITY)) {
    gpu_reduce_kernel<scalar_t, out_t>(
        iter,
        AbsMinOps<scalar_t, acc_t, out_t>(),
        std::numeric_limits<acc_t>::infinity());
  } else {
    gpu_reduce_kernel<scalar_t, out_t>(
        iter, NormOps<scalar_t, acc_t, out_t>{acc_t(p)}, 0);
  }
}

void norm_launch_kernel(TensorIterator& iter, double ord) {
  if (iter.dtype(0) == kHalf) {
    return norm_kernel_xpu_impl<at::Half, float>(iter, ord);
  } else if (iter.input_dtype() == kHalf && iter.dtype(0) == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    return norm_kernel_xpu_impl<at::Half, float, float>(iter, ord);
  } else if (iter.dtype(0) == kBFloat16) {
    return norm_kernel_xpu_impl<at::BFloat16, float>(iter, ord);
  } else if (iter.input_dtype() == kBFloat16 && iter.dtype(0) == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    return norm_kernel_xpu_impl<at::BFloat16, float, float>(iter, ord);
  }
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(iter.input_dtype(), "norm_cuda", [&] {
    norm_kernel_xpu_impl<scalar_t>(iter, ord);
  });
}

void norm_kernel(TensorIterator& iter, const Scalar& val) {
  double p;
  if (val.isIntegral(false)) {
    p = val.to<int64_t>();
  } else if (val.isFloatingPoint()) {
    p = val.to<double>();
  } else {
    TORCH_CHECK(
        false, "norm_kernel_xpu_impl expects norm to be integer or float");
  }
  if (iter.numel() == 0) {
    iter.output().fill_((p < 0) ? INFINITY : 0);
    return;
  }

  norm_launch_kernel(iter, p);

  if (isComplexType(iter.output().scalar_type())) {
    at::imag(iter.output()).zero_();
  }
}

} // namespace at::native::xpu
