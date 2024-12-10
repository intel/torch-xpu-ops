#include <ATen/ATen.h>
#include <ATen/NestedTensorImpl.h>
#include <ATen/native/nested/NestedTensorTransformerFunctions.h>
#include <ATen/native/xpu/NestedTensorTransformerFunctions.h>

namespace at::native {
namespace {
int64_t padded_tensor_numel(const Tensor& sizes) {
  const auto sizes_num_rows = sizes.sizes()[0];
  const auto sizes_row_length = sizes.sizes()[1];
  const auto* sizes_data = sizes.data_ptr<int64_t>();
  int64_t numel = 0;
  for (const auto row_num : c10::irange(sizes_num_rows)) {
    const auto* row_ptr = sizes_data + row_num * sizes_row_length;
    int64_t prod = 1;
    for (const auto idx : c10::irange(sizes_row_length)) {
      prod *= row_ptr[idx];
    }
    numel += prod;
  }
  return numel;
}
} // namespace
Tensor nested_from_padded_xpu(
    const Tensor& padded,
    const Tensor& sizes,
    bool do_transform_0213) {
  if (padded.dim() > 1 && padded.dim() < 5) {
    // Instead of erroring, call the generic version
    if (!(padded.dim() == 4 && do_transform_0213) &&
        !(padded.dim() == 3 && !do_transform_0213)) {
      return at::native::nested_from_padded_generic(
          padded, sizes, do_transform_0213);
    }
    if (padded.dtype() != at::kFloat && padded.dtype() != kHalf) {
      TORCH_WARN_ONCE(
          "nested_from_padded CUDA kernels only support fp32/fp16; falling "
          "back to slower generic kernel");
      return at::native::nested_from_padded_generic(
          padded, sizes, do_transform_0213);
    }
    Tensor target_offsets =
        at::native::NestedTensor_batch_offsets_from_size_tensor(sizes, 0);
    Tensor padded_sizes_tensor = at::tensor(padded.sizes());
    Tensor output = at::empty({padded_tensor_numel(sizes)}, padded.options());
    Tensor target_size_sizes = sizes.reshape(-1);

    target_offsets = target_offsets.to(at::Device(kXPU), at::kInt);
    padded_sizes_tensor = padded_sizes_tensor.to(at::Device(kXPU), at::kInt);
    target_size_sizes = target_size_sizes.to(at::Device(kXPU), at::kInt);

    auto output_size_ptr = target_size_sizes.data_ptr<int>();
    auto input_size_ptr = padded_sizes_tensor.data_ptr<int>();
    auto offsets_ptr = target_offsets.data_ptr<int>();

    Tensor padded_contiguous = padded.contiguous();

    if (padded.dtype() == at::kFloat) {
      if (do_transform_0213) {
        xpu::launch_remove_padding_transform0213_kernel(
            padded_contiguous.data_ptr<float>(),
            output.data_ptr<float>(),
            offsets_ptr,
            input_size_ptr,
            output_size_ptr,
            padded_contiguous.dim() - 2,
            padded_contiguous.sizes()[0]);
      } else {
        xpu::launch_remove_padding_kernel(
            padded_contiguous.data_ptr<float>(),
            output.data_ptr<float>(),
            offsets_ptr,
            input_size_ptr,
            output_size_ptr,
            padded_contiguous.dim() - 1,
            padded_contiguous.sizes()[0]);
      }
    } else if (padded.dtype() == at::kHalf) {
      if (do_transform_0213) {
        xpu::launch_remove_padding_transform0213_kernel(
            padded_contiguous.data_ptr<c10::Half>(),
            output.data_ptr<c10::Half>(),
            offsets_ptr,
            input_size_ptr,
            output_size_ptr,
            padded_contiguous.dim() - 2,
            padded_contiguous.sizes()[0]);
      } else {
        xpu::launch_remove_padding_kernel(
            padded_contiguous.data_ptr<c10::Half>(),
            output.data_ptr<c10::Half>(),
            offsets_ptr,
            input_size_ptr,
            output_size_ptr,
            padded_contiguous.dim() - 1,
            padded_contiguous.sizes()[0]);
      }
    } else {
      AT_ERROR("Only support fp32/fp16 for padded input");
    }
    return at::detail::make_tensor<at::native::NestedTensorImpl>(
        std::move(output), sizes);
  } else {
    return at::native::nested_from_padded_generic(padded, sizes);
  }
}

} // namespace at::native