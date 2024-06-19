#include <ATen/ATen.h>
#include <ATen/OpMathType.h>
#include <ATen/native/TensorIterator.h>

#include <aten/sycl/Loops.h>

namespace at::native::xpu {

template <typename scalar_t, typename accscalar_t>
struct HardsigmoidOutFunctor {
  scalar_t operator()(scalar_t self_val) const {
    accscalar_t x = static_cast<accscalar_t>(self_val);
    return std::min(std::max(x + three_, zero_), six_) * one_sixth_;
  }

  HardsigmoidOutFunctor(
      const accscalar_t zero,
      const accscalar_t one_sixth,
      const accscalar_t three,
      const accscalar_t six)
      : zero_(zero), one_sixth_(one_sixth), three_(three), six_(six) {}

 private:
  const accscalar_t zero_;
  const accscalar_t one_sixth_;
  const accscalar_t three_;
  const accscalar_t six_;
};

void hardsigmoid_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "hardsigmoid_out_xpu",
      [&]() {
        using accscalar_t = at::opmath_type<scalar_t>;
        const accscalar_t zero(0.0f);
        const accscalar_t one_sixth(1.0f / 6.0f);
        const accscalar_t three(3.0f);
        const accscalar_t six(6.0f);
        HardsigmoidOutFunctor<scalar_t, accscalar_t> f(
            zero, one_sixth, three, six);
        gpu_kernel(iter, f);
      });
}

template <typename scalar_t, typename accscalar_t>
struct HardsigmoidBackwardOutFunctor {
  scalar_t operator()(scalar_t grad_val_, scalar_t self_val_) const {
    accscalar_t grad_val = static_cast<accscalar_t>(grad_val_);
    accscalar_t self_val = static_cast<accscalar_t>(self_val_);
    return (self_val > neg_three_ && self_val < three_) ? grad_val * one_sixth_
                                                        : zero_;
  }

  HardsigmoidBackwardOutFunctor(
      const accscalar_t zero,
      const accscalar_t three,
      const accscalar_t neg_three,
      const accscalar_t one_sixth)
      : zero_(zero),
        three_(three),
        neg_three_(neg_three),
        one_sixth_(one_sixth) {}

 private:
  const accscalar_t zero_;
  const accscalar_t three_;
  const accscalar_t neg_three_;
  const accscalar_t one_sixth_;
};

void hardsigmoid_backward_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "hardsigmoid_backward_out_xpu",
      [&]() {
        using accscalar_t = at::opmath_type<scalar_t>;
        const accscalar_t zero(0.0f);
        const accscalar_t three(3.0f);
        const accscalar_t neg_three(-3.0f);
        const accscalar_t one_sixth(1.0f / 6.0f);
        HardsigmoidBackwardOutFunctor<scalar_t, accscalar_t> f(
            zero, three, neg_three, one_sixth);
        gpu_kernel(iter, f);
      });
}

} // namespace at::native::xpu
