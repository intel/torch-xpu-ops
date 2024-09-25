#include <ATen/ATen.h>
#include <ATen/ceil_div.h>
#include <ATen/native/TensorFactories.h>
#include <ATen/quantized/QTensorImpl.h>
#include <ATen/quantized/Quantizer.h>
#include <c10/core/Allocator.h>

namespace at::native::xpu {

static int64_t get_sub_byte_tensor_size(
    IntArrayRef sizes,
    size_t dtype_itemsize,
    at::ScalarType t) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int64_t element_per_byte;
  switch (t) {
    case at::ScalarType::QUInt4x2:
      element_per_byte = 2;
      break;
    case at::ScalarType::QUInt2x4:
      element_per_byte = 4;
      break;
    default:
      element_per_byte = 1;
  }
  // zero dim tensor
  if (sizes.empty()) {
    return c10::multiply_integers(sizes) * dtype_itemsize;
  }
  // Consider most inner dim as cols
  int64_t cols = sizes.at(sizes.size() - 1);
  int64_t bytes_per_row = cols * dtype_itemsize;
  // align qtensor most inner dim, compute ceil (bytes_per_row /
  // element_per_byte)
  return c10::multiply_integers(IntArrayRef(sizes.data(), sizes.size() - 1)) *
      at::ceil_div(bytes_per_row, element_per_byte);
}

Tensor new_qtensor(
    IntArrayRef sizes,
    const TensorOptions& options,
    QuantizerPtr quantizer) {
  at::DispatchKey tensorDispatchKey = options.computeDispatchKey();
  native::check_size_nonnegative(sizes);
  auto dtype = options.dtype();
  TORCH_CHECK(
      isQIntType(typeMetaToScalarType(dtype)),
      dtype,
      " is not supported in new_qtensor on xpu device.");
  auto scalar_type = typeMetaToScalarType(dtype);
  int64_t size_bytes =
      get_sub_byte_tensor_size(sizes, dtype.itemsize(), scalar_type);

  auto memory_format =
      options.memory_format_opt().value_or(MemoryFormat::Contiguous);
  at::Allocator* allocator = c10::GetAllocator(kXPU);

  auto storage = c10::make_intrusive<StorageImpl>(
      StorageImpl::use_byte_size_t(),
      size_bytes,
      allocator->allocate(size_bytes),
      allocator,
      /*resizable=*/true);
  auto tensor = detail::make_tensor<QTensorImpl>(
      storage, at::DispatchKeySet(tensorDispatchKey), dtype, quantizer);

  get_qtensorimpl(tensor)->set_sizes_contiguous(sizes);
  get_qtensorimpl(tensor)->empty_tensor_restride(memory_format);

  return tensor;
}

} // namespace at::native::xpu
