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
#include <ATen/OpMathType.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/NumericLimits.h>
#include <ATen/native/xpu/sycl/Reduce.h>

#include <ATen/native/xpu/sycl/ReduceOpsKernels.h>

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
  using native_sycl_op = sycl::plus<acc_t>;
  inline acc_t operator()(acc_t a, acc_t b) const {
    return native_sycl_op{}(a, b);
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

template <>
struct sum_functor<c10::complex<at::Half>> {
  void operator()(TensorIterator& iter) {
    using scalar_t = c10::complex<at::Half>;
    using acc_t = at::opmath_type<scalar_t>;
    gpu_reduce_kernel<scalar_t, scalar_t>(
        iter, func_wrapper<scalar_t>(SumFunctor<acc_t>()));
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

template <
    typename scalar_t,
    typename acc_t = scalar_t,
    typename out_t = scalar_t>
struct NansumFunctor {
  void operator()(TensorIterator& iter) {
    gpu_reduce_kernel<scalar_t, out_t>(iter, NanSumOps<acc_t, out_t>{});
  }
};

template <typename scalar_t>
struct NansumComplexFunctor {
  void operator()(TensorIterator& iter) {
    using acc_t = at::opmath_type<scalar_t>;
    gpu_reduce_kernel<scalar_t, scalar_t>(iter, NanSumOps<acc_t, scalar_t>{});
  }
};

void nansum_kernel(TensorIterator& iter) {
  auto general_dispatcher = [](TensorIterator& iter) {
    auto dtype = iter.dtype();
    if (at::isComplexType(dtype)) {
      AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, dtype, "nansum_xpu", [&]() {
        NansumComplexFunctor<scalar_t>{}(iter);
      });
    } else {
      AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "nansum_xpu", [&]() {
        NansumFunctor<scalar_t>{}(iter);
      });
    }
  };
  reduce_dispatch<NansumFunctor>(iter, general_dispatcher);
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
  auto general_dispatcher = [](TensorIterator& iter) {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
        kComplexHalf, kBool, iter.dtype(), "prod_xpu", [&]() {
          prod_functor<scalar_t>{}(iter);
        });
  };
  reduce_dispatch<prod_functor>(iter, general_dispatcher);
}

// XOR reduction used by hash_tensor (mode=0, XOR_SUM).
// The accumulator is uint64_t for integral/bool inputs, and double for
// floating point inputs (where the XOR is performed on the bit pattern).
struct XorSumOps {
  inline uint64_t operator()(uint64_t a, uint64_t b) const {
    return a ^ b;
  }
};

// Named functor (not a lambda): torch-xpu-ops builds with
// -fno-sycl-unnamed-lambda.
struct XorSumDoubleOps {
  inline double operator()(double a, double b) const {
    union {
      double d;
      uint64_t u;
    } a_converter, b_converter, result_converter;

    a_converter.d = a;
    b_converter.d = b;
    result_converter.u = a_converter.u ^ b_converter.u;
    // return a double, otherwise uint64_t will be cast to double
    // when accumulating and the result will be wrong
    return result_converter.d;
  }
};

void xor_sum_kernel(TensorIterator& iter) {
  // Use iter.dtype(1) to dispatch based on the type of the input tensor
  AT_DISPATCH_ALL_TYPES_AND3(
      kBFloat16, kHalf, kBool, iter.dtype(1), "xor_sum_xpu", [&]() {
        if constexpr (std::is_same_v<scalar_t, bool>) {
          gpu_reduce_kernel<bool, uint64_t>(
              iter, func_wrapper<uint64_t>(XorSumOps()));
        } else if constexpr (std::is_integral_v<scalar_t>) {
          gpu_reduce_kernel<scalar_t, uint64_t>(
              iter, func_wrapper<uint64_t>(XorSumOps()));
        } else {
          // Floating point inputs are hashed via their bit pattern.
          // The input value is implicitly upcast to double, then bit-cast to
          // uint64_t before XORing.
          gpu_reduce_kernel<scalar_t, double>(
              iter, func_wrapper<double>(XorSumDoubleOps()));
        }
      });
}

} // namespace xpu
} // namespace native
} // namespace at
