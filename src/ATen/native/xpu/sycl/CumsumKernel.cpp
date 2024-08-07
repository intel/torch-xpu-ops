#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>

#include <ATen/native/xpu/sycl/ScanUtils.h>

namespace at::native::xpu {

void launch_cumsum_kernel(
    const Tensor& result,
    const Tensor& self,
    int64_t dim) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      ScalarType::Half,
      ScalarType::BFloat16,
      self.scalar_type(),
      "cumsum_xpu",
      [&]() {
        scalar_t init = 0;
        scan<INCLUSIVE_TYPE, scalar_t, scalar_t>(
            result, self, dim, init, std::plus<scalar_t>());
      });
}

static c10::MaybeOwned<Tensor> contiguous_out_arg(const Tensor& tensor) {
  if (tensor.is_contiguous()) {
    return c10::MaybeOwned<Tensor>::borrowed(tensor);
  }
  return c10::MaybeOwned<Tensor>::owned(
      at::empty(tensor.sizes(), tensor.options()));
}

void cumsum_kernel(const Tensor& result, const Tensor& self, int64_t dim) {
  auto result_ = contiguous_out_arg(result);

  launch_cumsum_kernel(*result_, self, dim);

  if (!result.is_same(*result_)) {
    result.copy_(*result_);
  }
}

} // namespace at::native::xpu
