#include <ATen/ATen.h>
#include <ATen/XPUNativeFunctions.h>
#include <ATen/core/Tensor.h>
#include <aten/sycl/TensorTransformationsKernel.h>

namespace at{

Tensor XPUNativeFunctions::flip(const Tensor& self, IntArrayRef dims) {
  std::optional<Device> common_device = std::nullopt;
  c10::impl::check_and_update_common_device(common_device, self, "xpu::flip", "self");

  const int64_t total_dims = self.dim();
  // It wraps the dims and checks that there are no repeated dims
  auto flip_dims_b = at::dim_list_to_bitset(dims, total_dims);

  Tensor out_tensor = at::empty_like(self, MemoryFormat::Preserve);

  // Count dimensions in which we need to do work
  int n = 0;
  auto strides = DimVector(self.strides());
  for (int64_t i = 0; i < total_dims; i++) {
    if (flip_dims_b[i] && self.size(i) > 1 && self.stride(i) != 0) {
      n++;
      strides[i] = 0;
    }
  }

  // Nothing to do, we return fast
  if (n == 0 || self.numel() <= 1) {
    out_tensor.copy_(self);
    return out_tensor;
  }

  // create dummy output with 0 strides at flipped dimension, to prevent
  // tensorIterator from coalescing flipped dims
  const auto restrided_self = self.as_strided(self.sizes(), strides);
  auto iter =
      TensorIteratorConfig()
          .set_check_mem_overlap(false)
          .check_all_same_dtype(false)
          .declare_static_dtype_and_device(self.scalar_type(), self.device())
          .add_output(out_tensor)
          .add_input(self)
          .add_input(restrided_self)
          .build();

  auto* data = reinterpret_cast<char*>(iter.data_ptr(0));
  const auto sizes = iter.shape();
  // This is a SmallVector of _signed_ ints
  auto strides_bytes = DimVector(iter.strides(0));
  const auto strides_self = iter.strides(1);
  const auto strides_dummy = iter.strides(2);

  // To understand this transformation, think of a 3D cube.
  //   - The data ptr points to the lower-left most vertex of the cube
  //   - The strides tell us how to move in each dimension,
  //     that is, data + stride[i] advances one element in the dimension i
  // To flip a dimension:
  //   - We move the pointer to the opposite vertex of the cube
  //   - We iterate in the opposite direction (invert the strides)
  for (int i = 0; i < iter.ndim(); i++) {
    // We know that an dimension has a zero stride and self[i] does not, as we
    // defined above Note that it may be the case that strides_dummy[i] = 0
    // not because we set it, but because strides_self[i] == 0. We do not want
    // to do anything there
    if (strides_dummy[i] == 0 && strides_self[i] != 0) {
      data += strides_bytes[i] * (sizes[i] - 1);
      strides_bytes[i] *= -1;
    }
  }
  iter._unsafe_set_arg_strides(0, strides_bytes);
  iter._unsafe_set_arg_data(0, reinterpret_cast<void*>(data));

  at::native::xpu::flip_xpu_kernel(iter);
  return out_tensor;
}
} // namespace at 