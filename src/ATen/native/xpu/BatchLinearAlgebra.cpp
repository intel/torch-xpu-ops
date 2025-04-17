#include <ATen/core/Tensor.h>
#include <ATen/native/BatchLinearAlgebra.h>
#include <ATen/native/DispatchStub.h>
#if defined(USE_ONEMKL)
#include <ATen/native/xpu/mkl/BatchLinearAlgebra.h>
#endif // USE_ONEMKL

namespace at::native {

void svd_kernel_xpu(
    const Tensor& A,
    const bool full_matrices,
    const bool compute_uv,
    const c10::optional<c10::string_view>& driver,
    const Tensor& U,
    const Tensor& S,
    const Tensor& Vh,
    const Tensor& info) {
#if defined(USE_ONEMKL)
  native::xpu::svd_mkl(A, full_matrices, compute_uv, driver, U, S, Vh, info);
#else
  AT_ERROR(
      "SVD is not supported on XPU without oneMKL. Include oneMKL library in compilation");
#endif // USE_ONEMKL
}

REGISTER_XPU_DISPATCH(svd_stub, &svd_kernel_xpu);

} // namespace at::native
