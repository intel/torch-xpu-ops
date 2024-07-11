#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/NumericLimits.h>
#include <ATen/native/xpu/sycl/Reduce.h>

namespace at {
namespace native {
namespace xpu {

// The function `reduce_dispatch` below dispatches to the kernel based
// on the type of `iter`. It takes care of the common logic
// for handling Half-Precision floating types.
// Otherwise the functor `op` is called to dispatch to the kernel
// of relevant type.
//
// Note: Functor `op` should take care of all the types to be supported
//       except for `at::Half` and `at::BFloat16`.
template <
    template <
        typename scalar_t,
        typename acc_t = scalar_t,
        typename out_t = scalar_t>
    typename OpFunctor,
    typename GeneralDispatcher>
static void reduce_dispatch(TensorIterator& iter, GeneralDispatcher op) {
  if (iter.dtype() == kHalf) {
    return OpFunctor<at::Half, float>{}(iter);
  } else if (iter.dtype(1) == kHalf && iter.dtype() == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    return OpFunctor<at::Half, float, float>{}(iter);
  } else if (iter.dtype() == kBFloat16) {
    return OpFunctor<at::BFloat16, float>{}(iter);
  } else if (iter.dtype(1) == kBFloat16 && iter.dtype() == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    return OpFunctor<at::BFloat16, float, float>{}(iter);
  }
  op(iter);
}

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
  auto general_dispatcher = [](TensorIterator& iter) {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
        kBool, kComplexHalf, iter.dtype(), "sum_xpu", [&]() {
          sum_functor<scalar_t>{}(iter);
        });
  };
  reduce_dispatch<sum_functor>(iter, general_dispatcher);
}

template <typename acc_t>
struct NanSumFunctor {
  inline acc_t operator()(acc_t a, acc_t b) const {
    return a + (at::_isnan(b) ? acc_t{0.} : acc_t{b});
  }
};

template <>
struct NanSumFunctor<c10::complex<at::Half>> {
  using scalar_t = c10::complex<at::Half>;
  using acc_t = at::opmath_type<scalar_t>;
  inline acc_t operator()(acc_t a, acc_t b) const {
    return a + (at::_isnan(b) ? acc_t{0.} : acc_t{b});
  }
};

template <
    typename scalar_t,
    typename acc_t = scalar_t,
    typename out_t = scalar_t>
struct nansum_functor {
  void operator()(TensorIterator& iter) {
    gpu_reduce_kernel<scalar_t, out_t>(
        iter, func_wrapper<out_t>(NanSumFunctor<acc_t>()));
  }
};

void nansum_kernel(TensorIterator& iter) {
  auto general_dispatcher = [](TensorIterator& iter) {
    auto dtype = iter.dtype();
    if (at::isComplexType(dtype)) {
      AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, dtype, "nansum_xpu", [&]() {
        nansum_functor<scalar_t>{}(iter);
      });
    } else {
      AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "nansum_xpu", [&]() {
        nansum_functor<scalar_t>{}(iter);
      });
    }
  };
  reduce_dispatch<nansum_functor>(iter, general_dispatcher);
}

} // namespace xpu
} // namespace native
} // namespace at
