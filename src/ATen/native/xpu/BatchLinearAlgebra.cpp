#if defined(USE_ONEMKL)
#include <ATen/native/xpu/mkl/BatchLinearAlgebra.h>
#else
#include <ATen/native/DispatchStub.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/ops/_linalg_svd.h>
#include <ATen/ops/_linalg_svd_meta.h>
#include <ATen/ops/_linalg_svd_native.h>
#endif // USE_ONEMKL

namespace at::native {

std::tuple<Tensor&, Tensor&, Tensor&> _linalg_svd_out(
    const Tensor& A,
    bool full_matrices,
    bool compute_uv,
    c10::optional<c10::string_view> driver,
    Tensor& U,
    Tensor& S,
    Tensor& Vh) {
#if defined(USE_ONEMKL)
  std::cout << "enter _linalg_svd_out-----------------------" << std::endl;
  return native::xpu::svd_mkl(A, full_matrices, compute_uv, driver, U, S, Vh);
#else
  Tensor out_cpu = native::_linalg_svd_out(
      A.to(Device(at::kCPU)), full_matrices, compute_uv, driver, U, S, Vh);
  return out_cpu.to(Device(at::kXPU));
#endif // USE_ONEMKL
}

// REGISTER_XPU_DISPATCH(svd_stub, &native::xpu::svd_kernel);

} // namespace at::native
