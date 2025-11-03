#include <ATen/native/xpu/sycl/QRKernel.h>
#include <ATen/ops/linalg_qr_native.h>

namespace at::native {

TORCH_IMPL_FUNC(linalg_qr_xpu_out)
(const at::Tensor& A,
 std::string_view mode,
 const at::Tensor& Q,
 const at::Tensor& R) {
  xpu::linalg_qr_kernel(A, mode, Q, R);
}

} // namespace at::native
