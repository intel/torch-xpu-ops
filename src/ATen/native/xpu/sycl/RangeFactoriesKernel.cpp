#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/detail/FunctionTraits.h>
#include <comm/SYCLContext.h>

#include <ATen/native/xpu/sycl/RangeFactoriesKernel.h>

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
        using accscalar_t = at::acc_type_device<scalar_t, kXPU>;
        auto xstart = start.to<accscalar_t>();
        auto xstep = step.to<accscalar_t>();

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

template <typename scalar_t, typename accscalar_t>
struct RangeFunctor {
  scalar_t operator()(int64_t ind) const {
    accscalar_t inc = xstep_ * static_cast<accscalar_t>(ind);
    accscalar_t val = xstart_ + inc;
    return static_cast<scalar_t>(val);
  }
  RangeFunctor(accscalar_t xstart, accscalar_t xstep)
      : xstart_(xstart), xstep_(xstep) {}

 private:
  accscalar_t xstart_;
  accscalar_t xstep_;
};

Tensor& range_kernel(
    const Scalar& start,
    const Scalar& end,
    const Scalar& step,
    Tensor& result) {
  AT_DISPATCH_ALL_TYPES_AND(
      at::ScalarType::Half, result.scalar_type(), "range_xpu", [&]() {
        using accscalar_t = acc_type_device<scalar_t, kXPU>;
        auto xstart = start.to<accscalar_t>();
        auto xstep = step.to<accscalar_t>();

        bool is_contiguous = result.is_contiguous();
        Tensor r = !is_contiguous
            ? at::empty_like(result, LEGACY_CONTIGUOUS_MEMORY_FORMAT)
            : result;
        auto f = RangeFunctor<scalar_t, accscalar_t>(xstart, xstep);

        gpu_kernel_with_index(r, f);

        if (!result.is_contiguous()) {
          result.copy_(r);
        }
      });

  return result;
}

} // namespace xpu
} // namespace native
} // namespace at
