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

#include <ATen/native/xpu/sycl/HistogramKernels.h>

namespace at::native::xpu {

template <typename scalar_t>
struct HistogramddKernelFunctor {
  void operator()(sycl::nd_item<1> item_id) const {
    auto input_data = input_;
    int64_t wi_id = item_id.get_global_id();
    if (wi_id < input_size_ * bin_size_) {
      int64_t ele_idx = wi_id / bin_size_;
      int64_t bin_idx = wi_id % bin_size_;

      scalar_t value = (scalar_t)1;
      if (use_weight_) {
        auto weight_data = weight_;
        value = weight_data[ele_idx];
      }

      // [left, right)
      if (input_data[ele_idx][0] >= bin_edges_[bin_idx] &&
          input_data[ele_idx][0] < bin_edges_[bin_idx + 1]) {
        atomicAdd((sycl_global_ptr<scalar_t>)(hist_ + bin_idx), value);
        return;
      }

      // For last bin, [left, right]
      if (bin_idx == 0 && input_data[ele_idx][0] == bin_edges_[bin_size_]) {
        atomicAdd((sycl_global_ptr<scalar_t>)(hist_ + bin_size_ - 1), value);
      }
    }
  }

  HistogramddKernelFunctor(
      const PackedTensorAccessor64<const scalar_t, 2> input,
      const scalar_t* bin_edges,
      scalar_t* hist,
      const PackedTensorAccessor64<const scalar_t, 1> weight,
      int64_t input_size,
      int64_t bin_size,
      bool use_weight)
      : input_(input),
        bin_edges_(bin_edges),
        hist_(hist),
        weight_(weight),
        input_size_(input_size),
        bin_size_(bin_size),
        use_weight_(use_weight) {}

 private:
  const PackedTensorAccessor64<const scalar_t, 2> input_;
  const scalar_t* bin_edges_;
  scalar_t* hist_;
  const PackedTensorAccessor64<const scalar_t, 1> weight_;
  int64_t input_size_;
  int64_t bin_size_;
  bool use_weight_;
};

// For one dimension case
template <typename scalar_t>
void histogramdd_template(
    const PackedTensorAccessor64<const scalar_t, 2> input,
    const scalar_t* bin_edges,
    scalar_t* hist,
    const PackedTensorAccessor64<const scalar_t, 1> weight,
    int64_t input_size,
    int64_t bin_size,
    bool use_weight) {
  HistogramddKernelFunctor<scalar_t> kfn(
      input, bin_edges, hist, weight, input_size, bin_size, use_weight);
  const int64_t work_group_size = syclMaxWorkGroupSize(kfn);
  const int64_t num_wg =
      (input_size * bin_size + work_group_size - 1) / work_group_size;
  sycl_kernel_submit(
      num_wg * work_group_size, work_group_size, getCurrentSYCLQueue(), kfn);
}

template <typename scalar_t>
struct HistogramddLinearKernelFunctor {
  void operator()(sycl::nd_item<1> item_id) const {
    auto input_data = input_;
    int64_t wi_id = item_id.get_global_id();
    if (wi_id < input_size_) {
      scalar_t i_value = input_data[wi_id][0];
      scalar_t leftmost_edge = bin_edges_[0];
      scalar_t rightmost_edge = bin_edges_[bin_size_];
      if (i_value >= leftmost_edge && i_value <= rightmost_edge) {
        int64_t bin =
            (int64_t)(((i_value - leftmost_edge)) * bin_size_ / (rightmost_edge - leftmost_edge));
        if (bin == bin_size_)
          bin -= 1;
        scalar_t value = (scalar_t)1;
        if (use_weight_) {
          auto weight_data = weight_;
          value = weight_data[wi_id];
        }
        atomicAdd((sycl_global_ptr<scalar_t>)(hist_ + bin), value);
      }
    }
  }

  HistogramddLinearKernelFunctor(
      const PackedTensorAccessor64<const scalar_t, 2> input,
      const scalar_t* bin_edges,
      scalar_t* hist,
      const PackedTensorAccessor64<const scalar_t, 1> weight,
      int64_t input_size,
      int64_t bin_size,
      bool use_weight)
      : input_(input),
        bin_edges_(bin_edges),
        hist_(hist),
        weight_(weight),
        input_size_(input_size),
        bin_size_(bin_size),
        use_weight_(use_weight) {}

 private:
  const PackedTensorAccessor64<const scalar_t, 2> input_;
  const scalar_t* bin_edges_;
  scalar_t* hist_;
  const PackedTensorAccessor64<const scalar_t, 1> weight_;
  int64_t input_size_;
  int64_t bin_size_;
  bool use_weight_;
};

// For one dimension case
template <typename scalar_t>
void histogramdd_linear_template(
    const PackedTensorAccessor64<const scalar_t, 2> input,
    const scalar_t* bin_edges,
    scalar_t* hist,
    const PackedTensorAccessor64<const scalar_t, 1> weight,
    int64_t input_size,
    int64_t bin_size,
    bool use_weight) {
  HistogramddLinearKernelFunctor<scalar_t> kfn(
      input, bin_edges, hist, weight, input_size, bin_size, use_weight);
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
    const TensorList& bin_edges_) {
  globalContext().alertNotDeterministic("histogramdd_kernel_xpu");
  // remove this check once we support multi-dimension
  TORCH_CHECK(
      bin_edges_.size() == 1,
      "histogramdd_kernel xpu kernel doesn't support multi-dimensional histogram");

  hist.fill_(0);
  Tensor hist_c = hist;
  if (!hist.is_contiguous())
    hist_c = at::zeros_like(hist, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  Tensor bin_edges = bin_edges_[0].contiguous();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      kBFloat16, kHalf, self.scalar_type(), "histogram_xpu", [&]() {
        const auto self_accessor = self.packed_accessor64<const scalar_t, 2>();
        const auto weight_accessor = weight.has_value()
            ? weight.value().packed_accessor64<const scalar_t, 1>()
            : PackedTensorAccessor64<const scalar_t, 1>(
                  nullptr, self.sizes().data(), self.strides().data());
        histogramdd_template<scalar_t>(
            self_accessor,
            bin_edges.data_ptr<scalar_t>(),
            hist_c.data_ptr<scalar_t>(),
            weight_accessor,
            self.numel(),
            bin_edges.numel() - 1,
            weight.has_value());
      });

  if (!hist.is_same(hist_c))
    hist.copy_(hist_c);

  if (density) {
    const auto hist_sum = hist.sum();
    hist.div_(hist_sum);
    Tensor bin_lengths = bin_edges.diff();
    hist.div_(bin_lengths);
  }
}

void histogramdd_linear_kernel(
    const Tensor& self,
    const std::optional<Tensor>& weight,
    bool density,
    Tensor& hist,
    const TensorList& bin_edges_,
    bool local_search) {
  globalContext().alertNotDeterministic("histogramdd_linear_kernel_xpu");
  // remove this check once we support multi-dimension
  TORCH_CHECK(
      bin_edges_.size() == 1,
      "histogramdd_linear_kernel xpu kernel doesn't support multi-dimensional histogram");

  hist.fill_(0);
  Tensor hist_c = hist;
  if (!hist.is_contiguous())
    hist_c = at::zeros_like(hist, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  Tensor bin_edges = bin_edges_[0].contiguous();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      kBFloat16, kHalf, self.scalar_type(), "histogram_linear_xpu", [&]() {
        const auto self_accessor = self.packed_accessor64<const scalar_t, 2>();
        const auto weight_accessor = weight.has_value()
            ? weight.value().packed_accessor64<const scalar_t, 1>()
            : PackedTensorAccessor64<const scalar_t, 1>(
                  nullptr, self.sizes().data(), self.strides().data());
        histogramdd_linear_template<scalar_t>(
            self_accessor,
            bin_edges.data_ptr<scalar_t>(),
            hist_c.data_ptr<scalar_t>(),
            weight_accessor,
            self.numel(),
            bin_edges.numel() - 1,
            weight.has_value());
      });

  if (!hist.is_same(hist_c))
    hist.copy_(hist_c);

  if (density) {
    const auto hist_sum = hist.sum();
    hist.div_(hist_sum);
    Tensor bin_lengths = bin_edges.diff();
    hist.div_(bin_lengths);
  }
}

void histogram_select_outer_bin_edges_kernel(
    const Tensor& input,
    const int64_t N,
    std::vector<double>& leftmost_edges,
    std::vector<double>& rightmost_edges) {
  auto [min, max] = at::aminmax(input, 0);

  for (const auto i : c10::irange(N)) {
    leftmost_edges[i] = min[i].item().to<double>();
    rightmost_edges[i] = max[i].item().to<double>();
  }
}

} // namespace at::native::xpu

#pragma GCC diagnostic pop
#pragma clang diagnostic pop
