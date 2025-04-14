#if defined(USE_ONEMKL)
#include <ATen/native/BatchLinearAlgebra.h>
#include <ATen/native/xpu/mkl/BatchLinearAlgebra.h>
#include <ATen/ops/zeros.h>
#include <torch/library.h>

namespace at::native {

void _linalg_svd_xpu_out(
    const Tensor& A,
    const bool full_matrices,
    const bool compute_uv,
    std::optional<std::string_view> driver,
    const Tensor& U,
    const Tensor& S,
    const Tensor& Vh) {
  if (A.numel() == 0) {
    // Needed in the case that we have e.g. A.shape == (3, 0) and
    // full_matrices=True We fill U or Vh with the identity matrix as it's a
    // valid SVD for the empty matrix
    if (compute_uv && full_matrices) {
      if (U.numel() != 0) {
        U.zero_();
        U.diagonal(0, -2, -1).fill_(1.);
      }
      if (Vh.numel() != 0) {
        Vh.zero_();
        Vh.diagonal(0, -2, -1).fill_(1.);
      }
    }
    return;
  }

  native::xpu::svd_mkl(A, full_matrices, compute_uv, driver, U, S, Vh);
}

std::tuple<Tensor&, Tensor&, Tensor&> _linalg_svd_U_xpu(
    const Tensor& A,
    bool full_matrices,
    bool compute_uv,
    std::optional<c10::string_view> driver,
    Tensor& U,
    Tensor& S,
    Tensor& Vh) {
  _linalg_svd_xpu_out(A, full_matrices, compute_uv, driver, U, S, Vh);
  return std::forward_as_tuple(U, S, Vh);
}

TORCH_LIBRARY_IMPL(aten, XPU, m) {
  m.impl("_linalg_svd.U", TORCH_FN(_linalg_svd_U_xpu));
}

} // namespace at::native
#endif // USE_ONEMKL
