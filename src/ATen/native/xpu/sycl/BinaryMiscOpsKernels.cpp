#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <comm/xpu_aten.h>

#include <ATen/native/xpu/sycl/Loops.h>

namespace at::native::xpu {

template <typename scalar_t>
struct MSEFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    auto diff = a - b;
    return diff * diff;
  }
};

void mse_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "mse_xpu",
      [&]() { gpu_kernel(iter, MSEFunctor<scalar_t>()); });
}

template <typename scalar_t>
struct HuberFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    auto z = std::abs(a - b);
    return z < delta_val_ ? scalar_t(0.5) * z * z
                          : delta_val_ * (z - scalar_t(0.5) * delta_val_);
  }
  HuberFunctor(scalar_t delta_val) : delta_val_(delta_val) {}

 private:
  scalar_t delta_val_;
};

void huber_kernel(TensorIterator& iter, double delta) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kBFloat16, kHalf, iter.dtype(), "huber_xpu", [&iter, delta] {
        scalar_t delta_val(delta);
        gpu_kernel(iter, HuberFunctor<scalar_t>(delta_val));
      });
}

} // namespace at::native::xpu
