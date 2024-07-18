#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/NumericLimits.h>
#include <ATen/native/xpu/sycl/Reduce.h>

namespace at {
namespace native {
namespace xpu {

template <typename acc_t>
struct SumFunctor {
  inline acc_t operator()(acc_t a, acc_t b) const {
    return a + b;
  }
};

template <>
struct SumFunctor<c10::complex<at::Half>> {
  using scalar_t = c10::complex<at::Half>;
  using acc_t = at::opmath_type<scalar_t>;
  inline acc_t operator()(acc_t a, acc_t b) const {
    return a + b;
  }
};

template <
    typename scalar_t,
    typename acc_t = scalar_t,
    typename out_t = scalar_t>
struct sum_functor {
  void operator()(TensorIterator& iter) {
    gpu_reduce_kernel<scalar_t, out_t>(
        iter, func_wrapper<out_t>(SumFunctor<acc_t>()));
  }
};

void sum_kernel(TensorIterator& iter) {
  if (iter.dtype() == kHalf) {
    return sum_functor<at::Half, float>{}(iter);
  } else if (iter.dtype(1) == kHalf && iter.dtype() == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    return sum_functor<at::Half, float, float>{}(iter);
  } else if (iter.dtype() == kBFloat16) {
    return sum_functor<at::BFloat16, float>{}(iter);
  } else if (iter.dtype(1) == kBFloat16 && iter.dtype() == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    return sum_functor<at::BFloat16, float, float>{}(iter);
  }

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      kBool, kComplexHalf, iter.dtype(), "sum_xpu", [&]() {
        sum_functor<scalar_t>{}(iter);
      });
}

template <typename acc_t>
struct ProdFunctor {
  inline acc_t operator()(acc_t a, acc_t b) const {
    return a * b;
  }
};

template <>
struct ProdFunctor<bool> {
  inline bool operator()(bool a, bool b) const {
    return a && b;
  }
};

template <
    typename scalar_t,
    typename acc_t = scalar_t,
    typename out_t = scalar_t>
struct prod_functor {
  void operator()(TensorIterator& iter) {
    gpu_reduce_kernel<scalar_t, out_t>(
        iter, func_wrapper<out_t>(ProdFunctor<acc_t>()), 1.);
  }
};

template <>
struct prod_functor<bool> {
  void operator()(TensorIterator& iter) {
    gpu_reduce_kernel<bool, bool>(
        iter, func_wrapper<bool>(ProdFunctor<bool>()), 1);
  }
};

template <>
struct prod_functor<c10::complex<at::Half>> {
  void operator()(TensorIterator& iter) {
    using scalar_t = c10::complex<at::Half>;
    using acc_t = at::opmath_type<scalar_t>;
    gpu_reduce_kernel<scalar_t, scalar_t>(
        iter, func_wrapper<scalar_t>(ProdFunctor<acc_t>()), acc_t{1.});
  }
};

void prod_kernel(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      kComplexHalf, kBool, iter.dtype(), "prod_xpu", [&]() {
        prod_functor<scalar_t>{}(iter);
      });
};

} // namespace xpu
} // namespace native
} // namespace at
