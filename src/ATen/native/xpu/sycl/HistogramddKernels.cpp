#pragma clang diagnostic push
#pragma GCC diagnostic push
// Avoid SYCL compiler return-type error
#pragma clang diagnostic ignored "-Wreturn-type"
#pragma GCC diagnostic ignored "-Wreturn-type"

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/xpu/sycl/Atomics.h>
#include <comm/SYCLContext.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/aminmax.h>
#include <ATen/ops/linspace.h>
#endif

namespace at::native::xpu {

template <typename scalar_t>
struct HistogramddKernelFunctor {
  void operator()(sycl::nd_item<1> item_id) const {
    int64_t wi_id = item_id.get_global_id();
    if (wi_id < input_size_ * bin_size_) {
      int64_t ele_idx = wi_id / bin_size_;
      int64_t bin_idx = wi_id % bin_size_;

      // [left, right)
      if (input_[ele_idx] >= bin_edges_[bin_idx] &&
          input_[ele_idx] < bin_edges_[bin_idx + 1]) {
        scalar_t value = weight_ ? weight_[ele_idx] : (scalar_t)1;
        atomicAdd((sycl_global_ptr<scalar_t>)(hist_ + bin_idx), value);
        return;
      }

      // For last bin, [left, right]
      if (bin_idx == 0 && input_[ele_idx] == bin_edges_[bin_size_]) {
        scalar_t value = weight_ ? weight_[ele_idx] : (scalar_t)1;
        atomicAdd((sycl_global_ptr<scalar_t>)(hist_ + bin_size_ - 1), value);
      }
    }
  }

  HistogramddKernelFunctor(
      const scalar_t* input,
      const scalar_t* bin_edges,
      scalar_t* hist,
      const scalar_t* weight,
      int64_t input_size,
      int64_t bin_size)
      : input_(input),
        bin_edges_(bin_edges),
        hist_(hist),
        weight_(weight),
        input_size_(input_size),
        bin_size_(bin_size) {}

 private:
  const scalar_t* input_;
  const scalar_t* bin_edges_;
  scalar_t* hist_;
  const scalar_t* weight_;
  int64_t input_size_;
  int64_t bin_size_;
};

// For one dimension case
template <typename scalar_t>
void histogramdd_template(
    const scalar_t* input,
    const scalar_t* bin_edges,
    scalar_t* hist,
    const scalar_t* weight,
    int64_t input_size,
    int64_t bin_size) {
  HistogramddKernelFunctor<scalar_t> kfn(
      input, bin_edges, hist, weight, input_size, bin_size);
  const int64_t work_group_size = syclMaxWorkGroupSize(kfn);
  const int64_t num_wg =
      (input_size * bin_size + work_group_size - 1) / work_group_size;
  sycl_kernel_submit(
      num_wg * work_group_size, work_group_size, getCurrentSYCLQueue(), kfn);
}

template <typename scalar_t>
struct HistogramddLinearKernelFunctor {
  void operator()(sycl::nd_item<1> item_id) const {
    int64_t wi_id = item_id.get_global_id();
    if (wi_id < input_size_) {
      scalar_t i_value = input_[wi_id];
      if (i_value >= leftmost_edge_ && i_value <= rightmost_edge_) {
        int64_t bin =
            (int64_t)(((i_value - leftmost_edge_)) * bin_size_ / (rightmost_edge_ - leftmost_edge_));
        if (bin == bin_size_)
          bin -= 1;
        scalar_t value = weight_ ? weight_[wi_id] : (scalar_t)1;
        atomicAdd((sycl_global_ptr<scalar_t>)(hist_ + bin), value);
      }
    }
  }

  HistogramddLinearKernelFunctor(
      const scalar_t* input,
      scalar_t* hist,
      const scalar_t* weight,
      int64_t input_size,
      int64_t bin_size,
      double leftmost_edge,
      double rightmost_edge)
      : input_(input),
        hist_(hist),
        weight_(weight),
        input_size_(input_size),
        bin_size_(bin_size),
        leftmost_edge_(leftmost_edge),
        rightmost_edge_(rightmost_edge) {}

 private:
  const scalar_t* input_;
  scalar_t* hist_;
  const scalar_t* weight_;
  int64_t input_size_;
  int64_t bin_size_;
  double leftmost_edge_;
  double rightmost_edge_;
};

// For one dimension case
template <typename scalar_t>
void histogramdd_linear_template(
    const scalar_t* input,
    scalar_t* hist,
    const scalar_t* weight,
    int64_t input_size,
    int64_t bin_size,
    double leftmost_edge,
    double rightmost_edge) {
  HistogramddLinearKernelFunctor<scalar_t> kfn(
      input, hist, weight, input_size, bin_size, leftmost_edge, rightmost_edge);
  const int64_t work_group_size = syclMaxWorkGroupSize(kfn);
  const int64_t num_wg = (input_size + work_group_size - 1) / work_group_size;
  sycl_kernel_submit(
      num_wg * work_group_size, work_group_size, getCurrentSYCLQueue(), kfn);
}

void histogramdd_kernel(
    const Tensor& self,
    const std::optional<Tensor>& weight,
    bool density,
    Tensor& hist,
    const Tensor& bin_edges_) {
  hist.fill_(0);
  Tensor bin_edges = bin_edges_.contiguous();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kBFloat16, kHalf, self.scalar_type(), "histogram_xpu", [&]() {
        histogramdd_template<scalar_t>(
            self.data_ptr<scalar_t>(),
            bin_edges.data_ptr<scalar_t>(),
            hist.data_ptr<scalar_t>(),
            weight.has_value() ? weight->data_ptr<scalar_t>() : nullptr,
            self.numel(),
            bin_edges.numel() - 1);
      });

  if (density) {
    const auto hist_sum = hist.sum();
    hist.div_(hist_sum);
  }
}

void histogramdd_linear_kernel(
    const Tensor& self,
    int64_t bin_ct,
    std::optional<c10::ArrayRef<double>> range,
    const std::optional<Tensor>& weight,
    bool density,
    Tensor& hist,
    Tensor& out_bin_edges) {
  hist.fill_(0);

  double leftmost_edge, rightmost_edge;
  if (!range.has_value()) {
    auto extrema = at::aminmax(self);
    leftmost_edge = std::get<0>(extrema).item<double>();
    rightmost_edge = std::get<1>(extrema).item<double>();
  } else {
    leftmost_edge = range.value()[0];
    rightmost_edge = range.value()[1];
  }

  at::linspace_out(out_bin_edges, leftmost_edge, rightmost_edge, bin_ct + 1);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kBFloat16, kHalf, self.scalar_type(), "histogram_linear_xpu", [&]() {
        histogramdd_linear_template<scalar_t>(
            self.data_ptr<scalar_t>(),
            hist.data_ptr<scalar_t>(),
            weight.has_value() ? weight->data_ptr<scalar_t>() : nullptr,
            self.numel(),
            bin_ct,
            leftmost_edge,
            rightmost_edge);
      });

  if (density) {
    const auto hist_sum = hist.sum();
    hist.div_(hist_sum);
  }
}

} // namespace at::native::xpu

#pragma GCC diagnostic pop
#pragma clang diagnostic pop