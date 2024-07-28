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
    const Tensor& bins,
    const std::optional<Tensor>& weight) {
  if (weight.has_value()) {
    TORCH_CHECK(
        weight->device() == input.device(),
        "weight and input need to be on the same device.")
  }
  auto input_dtype = input.dtype();
  auto bins_dtype = bins.dtype();
  TORCH_CHECK(
      input_dtype == bins_dtype,
      "torch.histogramdd: input tensor and bins tensors should",
      " have the same dtype, but got input with dtype ",
      input_dtype,
      " and bins with dtype ",
      bins_dtype);

  const int64_t bins_dim = bins.dim();
  TORCH_CHECK(
      bins_dim == 1,
      "torch.histogramdd: bins tensor should have one dimension,",
      " but got ",
      bins_dim,
      " dimensions in the bin tensor");

  const int64_t numel = bins.numel();
  TORCH_CHECK(
      numel > 0,
      "torch.histogramdd: bins tensor should have at least 1 element,",
      " but got ",
      numel,
      " elements in the bin tensor");

  if (weight.has_value()) {
    TORCH_CHECK(
        input.dtype() == weight.value().dtype(),
        "torch.histogramdd: if weight tensor is provided, ",
        "input tensor and weight tensor should have the same dtype, ",
        "but got input(",
        input.dtype(),
        ")",
        ", and weight(",
        weight.value().dtype(),
        ")");

    auto input_sizes = input.sizes().vec();

    auto weight_sizes = weight.value().sizes().vec();
    if (weight_sizes.empty()) {
      // correctly handle scalars
      weight_sizes = {1};
    }

    TORCH_CHECK(
        input_sizes == weight_sizes,
        "torch.histogramdd: if weight tensor is provided it should have",
        " the same shape as the input tensor excluding its innermost ",
        "dimension, but got input with shape ",
        input.sizes(),
        " and weight ",
        "with shape ",
        weight.value().sizes());
  }
}

/* Checks properties of output tensors hist and bin_edges, then resizes them.
 */
void histogramdd_prepare_out(
    const Tensor& input,
    int64_t bin_ct,
    const Tensor& hist,
    const Tensor& bin_edges) {
  TORCH_CHECK(
      input.dtype() == hist.dtype(),
      "torch.histogram: input tensor and hist tensor should",
      " have the same dtype, but got input ",
      input.dtype(),
      " and hist ",
      hist.dtype());

  TORCH_CHECK(
      input.dtype() == bin_edges.dtype(),
      "torch.histogram: input tensor and bin_edges tensor should",
      " have the same dtype, but got input ",
      input.dtype(),
      " and bin_edges ",
      bin_edges.dtype());

  TORCH_CHECK(
      bin_ct > 0, "torch.histogram(): bins must be > 0, but got ", bin_ct);

  at::native::resize_output(bin_edges, {bin_ct + 1});

  at::native::resize_output(hist, {bin_ct});
}

void histogramdd_prepare_out(
    const Tensor& input,
    const Tensor& bins,
    const Tensor& hist,
    const Tensor& bin_edges) {
  int64_t bin_ct = bins.numel() - 1;
  histogramdd_prepare_out(input, bin_ct, hist, bin_edges);
}

static Tensor& histogramdd_out(
    const Tensor& self,
    const Tensor& bins,
    const std::optional<Tensor>& weight,
    bool density,
    Tensor& hist,
    Tensor& bin_edges) {
  histogramdd_check_inputs(self, bins, weight);
  histogramdd_prepare_out(self, bins, hist, bin_edges);

  bin_edges.copy_(bins);

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
  Tensor reshaped_self = self.reshape({self.numel()});
  std::optional<Tensor> reshaped_weight = weight.has_value()
      ? weight.value().reshape({weight.value().numel()})
      : weight;

  histogramdd_out(
      reshaped_self, bins, reshaped_weight, density, hist, bin_edges);

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
  Tensor reshaped_self = self.reshape({self.numel()});
  std::optional<Tensor> reshaped_weight = weight.has_value()
      ? weight.value().reshape({weight.value().numel()})
      : weight;

  histogramdd_prepare_out(reshaped_self, bin_ct, hist, bin_edges);
  histogramdd_check_inputs(reshaped_self, bin_edges, reshaped_weight);

  at::native::xpu::histogramdd_linear_kernel(
      reshaped_self, bin_ct, range, reshaped_weight, density, hist, bin_edges);
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