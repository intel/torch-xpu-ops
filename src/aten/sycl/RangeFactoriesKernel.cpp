#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/detail/FunctionTraits.h>
#include <comm/SYCLContext.h>

namespace at {
namespace native {
namespace xpu {

constexpr int nitem_per_wg = 256;
constexpr int item_work_size = 1;
constexpr int group_work_size = item_work_size * nitem_per_wg;

template <typename index_t, typename func_t>
struct ElementwiseKernelWithIndexFunctor {
  using res_t = typename function_traits<func_t>::result_type;
  void operator()(sycl::nd_item<1> item) const {
#pragma unroll
    for (int i = 0; i < item_work_size; i++) {
      index_t idx = group_work_size * item.get_group(0) + nitem_per_wg * i +
          item.get_local_id(0);
      if (idx < N_) {
        data_[idx] = f_(idx);
      }
    }
  }
  ElementwiseKernelWithIndexFunctor(index_t N, func_t f, res_t* data)
      : N_(N), f_(f), data_(data) {}

 private:
  index_t N_;
  func_t f_;
  res_t* data_;
};

template <typename func_t>
void gpu_kernel_with_index(at::Tensor& output, func_t f) {
  int64_t N = output.numel();
  if (N == 0) {
    return;
  }
  int64_t num_wg = (N + group_work_size - 1) / group_work_size;
  auto queue = at::xpu::getCurrentSYCLQueue();
  using scalar_t = typename function_traits<func_t>::result_type;
  if (N <= std::numeric_limits<int>::max()) {
    auto caller = ElementwiseKernelWithIndexFunctor<int, func_t>(
        N, f, output.mutable_data_ptr<scalar_t>());
    sycl_kernel_submit(num_wg * nitem_per_wg, nitem_per_wg, queue, caller);
  } else {
    auto caller = ElementwiseKernelWithIndexFunctor<int64_t, func_t>(
        N, f, output.mutable_data_ptr<scalar_t>());
    sycl_kernel_submit(num_wg * nitem_per_wg, nitem_per_wg, queue, caller);
  }
}

template <typename scalar_t, typename accscalar_t>
struct ArangeFunctor {
  scalar_t operator()(int64_t ind) const {
    accscalar_t inc = xstep_ * static_cast<accscalar_t>(ind);
    accscalar_t val = xstart_ + inc;
    return static_cast<scalar_t>(val);
  }
  ArangeFunctor(accscalar_t xstart, accscalar_t xstep)
      : xstart_(xstart), xstep_(xstep) {}

 private:
  accscalar_t xstart_;
  accscalar_t xstep_;
};

Tensor& arange_kernel(
    const Scalar& start,
    const Scalar& end,
    const Scalar& step,
    Tensor& result) {
  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      result.scalar_type(),
      "arange_xpu",
      [&]() {
        using accscalar_t = at::acc_type<scalar_t, true>;
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
        int64_t numel = result.numel();

        if (numel != size) {
          if (numel > 0) {
            TORCH_WARN(
                "The number of elements in the out tensor of shape ",
                result.sizes(),
                " is ",
                numel,
                " which does not match the computed number of elements ",
                size,
                ". Note that this may occur as a result of rounding error. "
                "The out tensor will be resized to a tensor of shape (",
                size,
                ",).");
          }
          result.resize_({size});
        }
        bool is_contiguous = result.is_contiguous();
        Tensor r = !is_contiguous
            ? at::empty_like(result, LEGACY_CONTIGUOUS_MEMORY_FORMAT)
            : result;

        auto f = ArangeFunctor<scalar_t, accscalar_t>(xstart, xstep);
        gpu_kernel_with_index(r, f);

        if (!is_contiguous) {
          result.copy_(r);
        }
      });

  return result;
}

} // namespace xpu
} // namespace native
} // namespace at
