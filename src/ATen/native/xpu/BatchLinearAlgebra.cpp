#include <ATen/core/Tensor.h>
#include <ATen/native/BatchLinearAlgebra.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/ops/empty_like.h>
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
  const auto A_cpu = A.to(A.options()
                              .device(kCPU)
                              .memory_format(at::MemoryFormat::Contiguous)
                              .pinned_memory(true));
  // U, S, Vh, info are the right size and strides, but are on GPU
  // We copy them into CPU in pinned_memory
  const auto empty_like_cpu = [](const Tensor& t) {
    return at::empty_like(t, t.options().device(kCPU).pinned_memory(true));
  };

  auto U_cpu = compute_uv ? empty_like_cpu(U) : Tensor{};
  auto S_cpu = empty_like_cpu(S);
  auto Vh_cpu = compute_uv ? empty_like_cpu(Vh) : Tensor{};
  auto info_cpu = empty_like_cpu(info);

  svd_stub(
      at::kCPU,
      A_cpu,
      full_matrices,
      compute_uv,
      driver,
      U_cpu,
      S_cpu,
      Vh_cpu,
      info_cpu);

  // Copy from CPU back to XPU
  // We can do a non_blocking copy, as there is an unconditional check of the
  // infos in the calling function
  if (compute_uv) {
    U.copy_(U_cpu, /*non_blocking*/ true);
    Vh.copy_(Vh_cpu, /*non_blocking*/ true);
  }
  S.copy_(S_cpu, /*non_blocking*/ true);
  info.copy_(info_cpu, /*non_blocking*/ true);
#endif // USE_ONEMKL
}

REGISTER_XPU_DISPATCH(svd_stub, &svd_kernel_xpu);

} // namespace at::native
