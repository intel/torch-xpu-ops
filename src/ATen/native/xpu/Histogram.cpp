#include <ATen/Context.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/Resize.h>
#include <ATen/native/xpu/sycl/HistogramKernels.h>
#include <ATen/xpu/XPUNativeFunctions.h>

namespace at {

/* Checks properties of input tensors input, bins, and weight.
 */
void histogramdd_check_inputs(
    const Tensor& input,
    const TensorList& bins,
    const std::optional<Tensor>& weight) {
  TORCH_CHECK(
      input.dim() >= 2,
      "torch.histogramdd: input tensor should have at least 2 dimensions, but got ",
      input.dim());

  const int64_t N = input.size(-1);

  TORCH_CHECK(
      static_cast<int64_t>(bins.size()) == N,
      "torch.histogramdd: expected ",
      N,
      " sequences of bin edges for a ",
      N,
      "-dimensional histogram but got ",
      bins.size());

  auto input_dtype = input.dtype();
  for (const auto dim : c10::irange(N)) {
    const Tensor& dim_bins = bins[dim];

    auto bins_dtype = dim_bins.dtype();
    TORCH_CHECK(
        input_dtype == bins_dtype,
        "torch.histogramdd: input tensor and bins tensors should",
        " have the same dtype, but got input with dtype ",
        input_dtype,
        " and bins for dimension ",
        dim,
        " with dtype ",
        bins_dtype);

    const int64_t dim_bins_dim = dim_bins.dim();
    TORCH_CHECK(
        dim_bins_dim == 1,
        "torch.histogramdd: bins tensor should have one dimension,",
        " but got ",
        dim_bins_dim,
        " dimensions in the bins tensor for dimension ",
        dim);

    const int64_t numel = dim_bins.numel();
    TORCH_CHECK(
        numel > 0,
        "torch.histogramdd: bins tensor should have at least 1 element,",
        " but got ",
        numel,
        " elements in the bins tensor for dimension ",
        dim);
  }

  if (weight.has_value()) {
    TORCH_CHECK(
        input.dtype() == weight.value().dtype(),
        "torch.histogramdd: if weight tensor is provided,"
        " input tensor and weight tensor should have the same dtype, but got input(",
        input.dtype(),
        ")",
        ", and weight(",
        weight.value().dtype(),
        ")");

    /* If a weight tensor is provided, we expect its shape to match that of
     * the input tensor excluding its innermost dimension N.
     */
    auto input_sizes = input.sizes().vec();
    input_sizes.pop_back();

    auto weight_sizes = weight.value().sizes().vec();
    if (weight_sizes.empty()) {
      // correctly handle scalars
      weight_sizes = {1};
    }

    TORCH_CHECK(
        input_sizes == weight_sizes,
        "torch.histogramdd: if weight tensor is provided it should have"
        " the same shape as the input tensor excluding its innermost dimension, but got input with shape ",
        input.sizes(),
        " and weight with shape ",
        weight.value().sizes());
  }
}

/* Checks properties of output tensors hist and bin_edges, then resizes them.
 */
void histogramdd_prepare_out(
    const Tensor& input,
    const std::vector<int64_t>& bin_ct,
    const Tensor& hist,
    const TensorList& bin_edges) {
  const int64_t N = input.size(-1);

  TORCH_INTERNAL_ASSERT((int64_t)bin_ct.size() == N);
  TORCH_INTERNAL_ASSERT((int64_t)bin_edges.size() == N);

  TORCH_CHECK(
      input.dtype() == hist.dtype(),
      "torch.histogram: input tensor and hist tensor should",
      " have the same dtype, but got input ",
      input.dtype(),
      " and hist ",
      hist.dtype());

  for (const auto dim : c10::irange(N)) {
    TORCH_CHECK(
        input.dtype() == bin_edges[dim].dtype(),
        "torch.histogram: input tensor and bin_edges tensor should",
        " have the same dtype, but got input ",
        input.dtype(),
        " and bin_edges ",
        bin_edges[dim].dtype(),
        " for dimension ",
        dim);

    TORCH_CHECK(
        bin_ct[dim] > 0,
        "torch.histogram(): bins must be > 0, but got ",
        bin_ct[dim],
        " for dimension ",
        dim);

    at::native::resize_output(bin_edges[dim], bin_ct[dim] + 1);
  }

  at::native::resize_output(hist, bin_ct);
}

void histogramdd_prepare_out(
    const Tensor& input,
    TensorList bins,
    const Tensor& hist,
    const TensorList& bin_edges) {
  std::vector<int64_t> bin_ct(bins.size());
  std::transform(bins.begin(), bins.end(), bin_ct.begin(), [](Tensor t) {
    return t.numel() - 1;
  });
  histogramdd_prepare_out(input, bin_ct, hist, bin_edges);
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

/* Determines the outermost bin edges. For simplicity when calling into aminmax,
 * assumes that input has already been reshaped to (M, N).
 */
std::pair<std::vector<double>, std::vector<double>> select_outer_bin_edges(
    const Tensor& input,
    std::optional<c10::ArrayRef<double>> range) {
  TORCH_INTERNAL_ASSERT(
      input.dim() == 2, "expected input to have shape (M, N)");
  const int64_t N = input.size(-1);

  // Default ranges for empty input matching numpy.histogram's default
  std::vector<double> leftmost_edges(N, 0.);
  std::vector<double> rightmost_edges(N, 1.);

  if (range.has_value()) {
    // range is specified
    TORCH_CHECK(
        (int64_t)range.value().size() == 2 * N,
        "torch.histogramdd: for a ",
        N,
        "-dimensional histogram",
        " range should have ",
        2 * N,
        " elements, but got ",
        range.value().size());

    for (const auto dim : c10::irange(N)) {
      leftmost_edges[dim] = range.value()[2 * dim];
      rightmost_edges[dim] = range.value()[2 * dim + 1];
    }
  } else if (input.numel() > 0) {
    // non-empty input

    histogram_select_outer_bin_edges_kernel(
        input, N, leftmost_edges, rightmost_edges);
  }

  for (const auto dim : c10::irange(N)) {
    double leftmost_edge = leftmost_edges[dim];
    double rightmost_edge = rightmost_edges[dim];

    TORCH_CHECK(
        std::isfinite(leftmost_edge) && std::isfinite(rightmost_edge),
        "torch.histogramdd: dimension ",
        dim,
        "'s range [",
        leftmost_edge,
        ", ",
        rightmost_edge,
        "] is not finite");

    TORCH_CHECK(
        leftmost_edge <= rightmost_edge,
        "torch.histogramdd: min should not exceed max, but got",
        " min ",
        leftmost_edge,
        " max ",
        rightmost_edge,
        " for dimension ",
        dim);

    // Expand empty range to match numpy behavior and avoid division by 0 in
    // normalization
    if (leftmost_edge == rightmost_edge) {
      leftmost_edges[dim] -= 0.5;
      rightmost_edges[dim] += 0.5;
    }
  }

  return std::make_pair(leftmost_edges, rightmost_edges);
}

static Tensor& histogramdd_out(
    const Tensor& self,
    TensorList bins,
    const std::optional<Tensor>& weight,
    bool density,
    Tensor& hist,
    TensorList& bin_edges) {
  histogramdd_check_inputs(self, bins, weight);
  histogramdd_prepare_out(self, bins, hist, bin_edges);

  for (const auto dim : c10::irange(bins.size())) {
    bin_edges[dim].copy_(bins[dim]);
  }

  at::native::xpu::histogramdd_kernel(self, weight, density, hist, bin_edges);
  return hist;
}

std::tuple<Tensor&, Tensor&> XPUNativeFunctions::histogram_out(
    const Tensor& self,
    const Tensor& bins,
    const std::optional<Tensor>& weight,
    bool density,
    Tensor& hist,
    Tensor& bin_edges) {
  Tensor reshaped_self = self.reshape({self.numel(), 1});
  std::optional<Tensor> reshaped_weight = weight.has_value()
      ? weight.value().reshape({weight.value().numel()})
      : weight;
  TensorList bins_in = bins;
  TensorList bins_out = bin_edges;

  histogramdd_out(
      reshaped_self, bins_in, reshaped_weight, density, hist, bins_out);

  return std::forward_as_tuple(hist, bin_edges);
}

std::tuple<Tensor, Tensor> XPUNativeFunctions::histogram(
    const Tensor& self,
    const Tensor& bins,
    const std::optional<Tensor>& weight,
    bool density) {
  Tensor hist = at::empty({0}, self.options(), MemoryFormat::Contiguous);
  Tensor bin_edges = at::empty({0}, bins.options(), MemoryFormat::Contiguous);
  return histogram_out(self, bins, weight, density, hist, bin_edges);
}

std::tuple<Tensor&, Tensor&> XPUNativeFunctions::histogram_out(
    const Tensor& self,
    int64_t bin_ct,
    std::optional<c10::ArrayRef<double>> range,
    const std::optional<Tensor>& weight,
    bool density,
    Tensor& hist,
    Tensor& bin_edges) {
  Tensor reshaped_self = self.reshape({self.numel(), 1});
  std::optional<Tensor> reshaped_weight = weight.has_value()
      ? weight.value().reshape({weight.value().numel()})
      : weight;
  TensorList bins_in = bin_edges;
  TensorList bins_out = bin_edges;

  histogramdd_prepare_out(
      reshaped_self, std::vector<int64_t>{bin_ct}, hist, bins_out);
  auto outer_bin_edges = select_outer_bin_edges(reshaped_self, range);
  at::linspace_out(
      bin_edges,
      outer_bin_edges.first[0],
      outer_bin_edges.second[0],
      bin_ct + 1);

  histogramdd_check_inputs(reshaped_self, bins_in, reshaped_weight);

  at::native::xpu::histogramdd_linear_kernel(
      reshaped_self, reshaped_weight, density, hist, bin_edges, true);
  return std::forward_as_tuple(hist, bin_edges);
}

std::tuple<Tensor, Tensor> XPUNativeFunctions::histogram(
    const Tensor& self,
    int64_t bin_ct,
    std::optional<c10::ArrayRef<double>> range,
    const std::optional<Tensor>& weight,
    bool density) {
  Tensor hist = at::empty({0}, self.options(), MemoryFormat::Contiguous);
  Tensor bin_edges_out = at::empty({0}, self.options());
  return histogram_out(
      self, bin_ct, range, weight, density, hist, bin_edges_out);
}

} // namespace at