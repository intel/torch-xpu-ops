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

void lu_solve_kernel_xpu(
    const Tensor& LU,
    const Tensor& pivots,
    const Tensor& B,
    TransposeType trans) {
#if defined(USE_ONEMKL)
  native::xpu::lu_solve_mkl(LU, pivots, B, trans);
#else
  const auto LU_cpu = LU.to(LU.options().device(kCPU).pinned_memory(true));
  const auto pivots_cpu =
      pivots.to(pivots.options().device(kCPU).pinned_memory(true));
  auto B_cpu = B.to(B.options().device(kCPU).pinned_memory(true));

  lu_solve_stub(at::kCPU, LU_cpu, pivots_cpu, B_cpu, trans);

  B.copy_(B_cpu, /*non_blocking*/ true);
#endif // USE_ONEMKL
}

REGISTER_XPU_DISPATCH(lu_solve_stub, &lu_solve_kernel_xpu);

void lu_factor_kernel_xpu(
    const Tensor& input,
    const Tensor& pivots,
    const Tensor& infos,
    bool compute_pivots) {
#if defined(USE_ONEMKL)
  native::xpu::lu_factor_mkl(input, pivots, infos, compute_pivots);
#else
  auto input_cpu = input.to(input.options().device(kCPU).pinned_memory(true));
  auto pivots_cpu =
      pivots.to(pivots.options().device(kCPU).pinned_memory(true));
  const auto infos_cpu =
      infos.to(infos.options().device(kCPU).pinned_memory(true));

  lu_factor_stub(at::kCPU, input_cpu, pivots_cpu, infos_cpu, compute_pivots);

  input.copy_(input_cpu);
  pivots.copy_(pivots_cpu);
  infos.copy_(infos_cpu);
#endif // USE_ONEMKL
}

REGISTER_XPU_DISPATCH(lu_factor_stub, &lu_factor_kernel_xpu);

// void matmal_complex_kernel_xpu() {
//   std::cout << "======== call matmul_complex_xpu =============" << std::endl;
//   return;
// }
// REGISTER_XPU_DISPATCH(matmul_complex_stub, &matmal_complex_kernel_xpu);

} // namespace at::native
