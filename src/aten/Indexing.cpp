#include <ATen/ATen.h>
#include <src/ATen/WrapDimUtils.h>
#include <ATen/XPUNativeFunctions.h>

#include <aten/sycl/IndexingKernel.h>
#include <comm/TensorInfo.h>

namespace at {

Tensor& XPUNativeFunctions::index_select_out(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    Tensor& out) {
  static constexpr string_view DIM_WARNING =
      "Tensor too large or too many (> 12) dimensions";
  at::assert_no_internal_overlap(out);
  at::assert_no_overlap(out, self);
  at::assert_no_overlap(out, index);

  dim = at::maybe_wrap_dim(dim, self);
  TORCH_CHECK(self.dim() <= XPU_MAX_TENSORINFO_DIMS, DIM_WARNING);
  TORCH_CHECK(index.dim() <= XPU_MAX_TENSORINFO_DIMS, DIM_WARNING);
  AT_DISPATCH_V2(
      out.scalar_type(),
      "index_select_xpu",
      AT_WRAP([&] { index_select_kernel<scalar_t>(out, self, dim, index); }),
      AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX), AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES),
      kComplexHalf,
      kHalf,
      kBool,
      kBFloat16
      );

  return out;
}

Tensor XPUNativeFunctions::index_select(
    const Tensor& self,
    int64_t dim,
    const Tensor& index) {
  auto out = at::empty({0}, self.options());
  return at::native::xpu::index_select_out_kernel(self, dim, index, out);
}
} // namespace at
