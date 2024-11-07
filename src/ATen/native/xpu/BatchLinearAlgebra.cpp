#include <ATen/ATen.h>
#include <ATen/native/BatchLinearAlgebra.h>
#include <ATen/native/DispatchStub.h>

namespace at {
namespace native {

namespace xpu {

void linalg_eigh_kernel(
    const Tensor& eigenvalues,
    const Tensor& eigenvectors,
    const Tensor& infos,
    bool upper,
    bool compute_eigenvectors) {
  // transfer to CPU, compute the result and copy back to GPU
  Tensor eigenvalues_cpu =
      at::empty_like(eigenvalues, eigenvalues.options().device(kCPU));
  if (compute_eigenvectors) {
    Tensor eigenvectors_cpu =
        at::empty_like(eigenvectors, eigenvectors.options().device(kCPU));
    at::linalg_eigh_out(
        eigenvalues_cpu,
        eigenvectors_cpu,
        eigenvectors.to(kCPU),
        upper ? "U" : "L");
    eigenvectors.copy_(eigenvectors_cpu);
  } else {
    at::linalg_eigvalsh_out(
        eigenvalues_cpu, eigenvectors.to(kCPU), upper ? "U" : "L");
  }
  eigenvalues.copy_(eigenvalues_cpu);
}

void svd_kernel(
    const Tensor& A,
    const bool full_matrices,
    const bool compute_uv,
    const std::optional<c10::string_view>& driver,
    const Tensor& U,
    const Tensor& S,
    const Tensor& Vh,
    const Tensor& info) {
  const auto A_ = A.to(A.options()
                           .device(kCPU)
                           .memory_format(at::MemoryFormat::Contiguous)
                           .pinned_memory(true));
  // U, S, Vh, info are the right size and strides, but are on GPU
  // We copy them into CPU in pinned_memory
  const auto empty_like_cpu = [](const Tensor& t) {
    return at::empty_like(t, t.options().device(kCPU).pinned_memory(true));
  };
  auto U_ = compute_uv ? empty_like_cpu(U) : Tensor{};
  auto S_ = empty_like_cpu(S);
  auto Vh_ = compute_uv ? empty_like_cpu(Vh) : Tensor{};
  auto info_ = empty_like_cpu(info);
  svd_stub(
      A_.device().type(),
      A_,
      full_matrices,
      compute_uv,
      driver,
      U_,
      S_,
      Vh_,
      info_);

  // Copy from CPU back to XPU
  // We can do a non_blocking copy, as there is an unconditional check of the
  // infos in the calling function
  if (compute_uv) {
    U.copy_(U_, /*non_blocking*/ true);
    Vh.copy_(Vh_, /*non_blocking*/ true);
  }
  S.copy_(S_, /*non_blocking*/ true);
  info.copy_(info, /*non_blocking*/ true);
}

} // namespace xpu

REGISTER_XPU_DISPATCH(linalg_eigh_stub, &xpu::linalg_eigh_kernel);
REGISTER_XPU_DISPATCH(svd_stub, &xpu::svd_kernel)

} // namespace native
} // namespace at
