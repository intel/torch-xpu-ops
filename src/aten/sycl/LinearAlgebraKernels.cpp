#include <ATen/ATen.h>

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/TensorIterator.h>
#include <c10/core/ScalarType.h>

#include <aten/sycl/CopyKernel.h>
#include <aten/sycl/Loops.h>
#include <comm/SYCLContext.h>

namespace at::native::xpu {

template <typename scalar_t>
struct AddrKernelFunctor {
  scalar_t operator()(scalar_t self_val, scalar_t vec1_val, scalar_t vec2_val)
      const {
    return alpha_val_ && vec1_val && vec2_val;
  }
  AddrKernelFunctor(scalar_t alpha_val) : alpha_val_(alpha_val) {}

 private:
  scalar_t alpha_val_;
};

template <typename scalar_t>
struct AddrKernelFunctor2 {
  scalar_t operator()(scalar_t self_val, scalar_t vec1_val, scalar_t vec2_val)
      const {
    return (beta_val_ && self_val) || (alpha_val_ && vec1_val && vec2_val);
  }

  AddrKernelFunctor2(scalar_t alpha_val, scalar_t beta_val)
      : alpha_val_(alpha_val), beta_val_(beta_val) {}

 private:
  scalar_t alpha_val_;
  scalar_t beta_val_;
};

template <typename scalar_t>
struct AddrKernelFunctor3 {
  scalar_t operator()(scalar_t self_val, scalar_t vec1_val, scalar_t vec2_val)
      const {
    return alpha_val_ * vec1_val * vec2_val;
  }
  AddrKernelFunctor3(scalar_t alpha_val) : alpha_val_(alpha_val) {}

 private:
  scalar_t alpha_val_;
};

template <typename scalar_t>
struct AddrKernelFunctor4 {
  scalar_t operator()(scalar_t self_val, scalar_t vec1_val, scalar_t vec2_val)
      const {
    return beta_val_ * self_val + alpha_val_ * vec1_val * vec2_val;
  }

  AddrKernelFunctor4(scalar_t alpha_val, scalar_t beta_val)
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
      AddrKernelFunctor<scalar_t> f(alpha_val);
      gpu_kernel(iter, f);
    } else {
      AddrKernelFunctor2<scalar_t> f(alpha_val, beta_val);
      gpu_kernel(iter, f);
    }
    return;
  }

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kBFloat16, kHalf, kBool, iter.dtype(), "addr_xpu", [&] {
        auto beta_val = beta.to<scalar_t>();
        auto alpha_val = alpha.to<scalar_t>();

        scalar_t zero_val(0);
        // when beta==0, values in self should be ignored,
        // nans and infs in self should not propagate.
        if (beta_val == zero_val) {
          AddrKernelFunctor3<scalar_t> f(alpha_val);
          gpu_kernel(iter, f);
        } else {
          AddrKernelFunctor4<scalar_t> f(alpha_val, beta_val);
          gpu_kernel(iter, f);
        }
      });
}
} // namespace at::native::xpu