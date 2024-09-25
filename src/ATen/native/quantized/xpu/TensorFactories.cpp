#include <ATen/ATen.h>
#include <ATen/xpu/XPUNativeFunctions.h>

#include <ATen/native/quantized/xpu/MakePerTensorQuantizedTensor.h>
#include <ATen/native/quantized/xpu/Quantizer.h>

namespace at {

Tensor XPUNativeFunctions::_empty_affine_quantized(
    IntArrayRef size,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory,
    double scale,
    int64_t zero_point,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  TensorOptions options_ =
      TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(
          pin_memory);
  TORCH_CHECK(
      !(options_.has_memory_format() && optional_memory_format.has_value()),
      "Cannot set memory_format both in TensorOptions and explicit argument; "
      "please delete "
      "the redundant setter.");
  auto options = options_.memory_format(optional_memory_format);
  TORCH_CHECK(
      options.has_dtype(),
      "Must provide data type for Tensor creation functions.");
  return native::xpu::new_qtensor(
      size,
      options,
      native::make_per_tensor_affine_quantizer_xpu(
          scale, zero_point, typeMetaToScalarType(options.dtype())));
}

Tensor XPUNativeFunctions::_empty_per_channel_affine_quantized(
    IntArrayRef size,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  TensorOptions options_ =
      TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(
          pin_memory);
  TORCH_CHECK(
      !(options_.has_memory_format() && optional_memory_format.has_value()),
      "Cannot set memory_format both in TensorOptions and explicit argument; "
      "please delete "
      "the redundant setter.");
  auto options = options_.memory_format(optional_memory_format);
  TORCH_CHECK(
      options.has_dtype(),
      "Must provide data type for Tensor creation functions.");
  TORCH_CHECK(
      options.dtype() == kQInt8 || options.dtype() == kQUInt8,
      "Supported data type for tensor creation is int8 or uint8");
  return native::xpu::new_qtensor(
      size,
      options,
      native::make_per_channel_affine_quantizer_xpu(
          scales, zero_points, axis, typeMetaToScalarType(options.dtype())));
}

} // namespace at
