#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/ScalarOps.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/RangeFactories.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/RangeFactoriesKernel.h>
#include <comm/xpu_aten.h>
#include <torch/library.h>

#include <xpu/ATen/ops/arange_native.h>
#include <xpu/ATen/ops/linspace_native.h>
#include <xpu/ATen/ops/logspace_native.h>
#include <xpu/ATen/ops/range_native.h>

namespace at {

namespace native {
Tensor& arange_out_xpu(
    const Scalar& start,
    const Scalar& end,
    const Scalar& step,
    Tensor& out) {
  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      out.scalar_type(),
      "arange_xpu_preprocess",
      [&]() {
        using accscalar_t = at::acc_type_device<scalar_t, kXPU>;
        auto xstart = start.to<accscalar_t>();
        auto xend = end.to<accscalar_t>();
        auto xstep = step.to<accscalar_t>();

        TORCH_CHECK(xstep > 0 || xstep < 0, "step must be nonzero");
        TORCH_CHECK(
            std::isfinite(static_cast<double>(xstart)) &&
                std::isfinite(static_cast<double>(xend)),
            "unsupported range: ",
            xstart,
            " -> ",
            xend);
        TORCH_CHECK(
            ((xstep > 0) && (xend >= xstart)) ||
                ((xstep < 0) && (xend <= xstart)),
            "upper bound and larger bound inconsistent with step sign");

        // we use double precision for (start - end) / step
        // to compute size_d for consistency across devices.
        // The problem with using accscalar_t is that accscalar_t might be
        // float32 on gpu for a float32 scalar_t, but double on cpu for the
        // same, and the effective output size starts differing on CPU vs GPU
        // because of precision issues, which we dont want. the corner-case we
        // do want to take into account is int64_t, which has higher precision
        // than double
        double size_d;
        if constexpr (std::is_same_v<scalar_t, int64_t>) {
          int64_t sgn = (xstep > 0) - (xstep < 0);
          size_d = std::ceil((xend - xstart + xstep - sgn) / xstep);
        } else {
          size_d = std::ceil(
              static_cast<double>(end.to<double>() - start.to<double>()) /
              step.to<double>());
        }

        TORCH_CHECK(
            size_d >= 0 &&
                size_d <=
                    static_cast<double>(std::numeric_limits<int64_t>::max()),
            "invalid size, possible overflow?");
        int64_t size = static_cast<int64_t>(size_d);
        int64_t numel = out.numel();

        if (numel != size) {
          if (numel > 0) {
            TORCH_WARN(
                "The number of elements in the out tensor of shape ",
                out.sizes(),
                " is ",
                numel,
                " which does not match the computed number of elements ",
                size,
                ". Note that this may occur as a result of rounding error. "
                "The out tensor will be resized to a tensor of shape (",
                size,
                ",).");
          }
          out.resize_({size});
        }
      });

  return xpu::arange_kernel(start, end, step, out);
}

Tensor& range_xpu_out(
    const Scalar& start,
    const Scalar& end,
    const Scalar& step,
    Tensor& out) {
  auto xstart = start.to<double>();
  auto xend = end.to<double>();
  auto xstep = step.to<double>();

  TORCH_CHECK(xstep > 0 || xstep < 0, "step must be nonzero");
  TORCH_CHECK(
      std::isfinite(static_cast<double>(xstart)) &&
          std::isfinite(static_cast<double>(xend)),
      "unsupported range: ",
      xstart,
      " -> ",
      xend);
  TORCH_CHECK(
      ((xstep > 0) && (xend >= xstart)) || ((xstep < 0) && (xend <= xstart)),
      "upper bound and larger bound inconsistent with step sign");
  int64_t size = static_cast<int64_t>(((xend - xstart) / xstep) + 1);
  if (out.numel() != size) {
    out.resize_({size});
  }

  return at::native::xpu::range_kernel(start, end, step, out);
}

Tensor& linspace_xpu_out(
    const Scalar& start,
    const Scalar& end,
    int64_t steps,
    Tensor& out) {
  return at::native::xpu::linspace_kernel(start, end, steps, out);
}

Tensor& logspace_xpu_out(
    const Scalar& start,
    const Scalar& end,
    int64_t steps,
    double base,
    Tensor& result) {
  return at::native::xpu::logspace_kernel(start, end, steps, base, result);
}

} // namespace native
} // namespace at
