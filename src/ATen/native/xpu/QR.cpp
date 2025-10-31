#include <ATen/core/Tensor.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/xpu/sycl/QRKernel.h>
#include <comm/RegisterUtils.h>
#include <comm/xpu_aten.h>

#include <ATen/ops/linalg_qr_native.h>

namespace at::native {

TORCH_IMPL_FUNC(linalg_qr_xpu_out)
(const Tensor& A,
 c10::string_view mode,
 const Tensor& Q,
 const Tensor& R) {
  xpu::linalg_qr_kernel(A, mode, Q, R);
}

} // namespace at::native
