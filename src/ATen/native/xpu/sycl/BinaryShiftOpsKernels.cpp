#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>

#include <ATen/native/xpu/sycl/Loops.h>

namespace at::native::xpu {

template <typename scalar_t>
struct LshiftFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    constexpr scalar_t max_shift = sizeof(scalar_t) * CHAR_BIT;
    if ((static_cast<std::make_signed_t<scalar_t>>(b) < 0) ||
        (b >= max_shift)) {
      return 0;
    }
    return static_cast<std::make_unsigned_t<scalar_t>>(a) << b;
  }
};

void lshift_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "lshift_xpu", [&]() {
    gpu_kernel_with_scalars(iter, LshiftFunctor<scalar_t>());
  });
}

template <typename scalar_t>
struct RshiftFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    // right shift value to retain sign bit for signed and no bits for
    // unsigned
    constexpr scalar_t max_shift =
        sizeof(scalar_t) * CHAR_BIT - std::is_signed_v<scalar_t>;
    if ((static_cast<std::make_signed_t<scalar_t>>(b) < 0) ||
        (b >= max_shift)) {
      return a >> max_shift;
    }
    return a >> b;
  }
};

void rshift_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "rshift_xpu", [&]() {
    gpu_kernel_with_scalars(iter, RshiftFunctor<scalar_t>());
  });
}

} // namespace at::native::xpu