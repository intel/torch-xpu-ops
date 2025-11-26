#include <ATen/core/Tensor.h>
#include <ATen/native/BatchLinearAlgebra.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/ops/linalg_qr_native.h>
#if defined(USE_ONEMKL_XPU)
#include <ATen/native/xpu/mkl/BatchLinearAlgebra.h>
#endif // USE_ONEMKL_XPU

namespace at::native {

void lu_solve_kernel_xpu(
    const Tensor& LU,
    const Tensor& pivots,
    const Tensor& B,
    TransposeType trans) {
#if defined(USE_ONEMKL_XPU)
  native::xpu::lu_solve_mkl(LU, pivots, B, trans);
#else
  const auto LU_cpu = LU.to(LU.options().device(kCPU));
  const auto pivots_cpu = pivots.to(pivots.options().device(kCPU));
  auto B_cpu = B.to(B.options().device(kCPU));

  lu_solve_stub(at::kCPU, LU_cpu, pivots_cpu, B_cpu, trans);

  B.copy_(B_cpu);
#endif // USE_ONEMKL_XPU
}

REGISTER_XPU_DISPATCH(lu_solve_stub, &lu_solve_kernel_xpu);

void lu_factor_kernel_fallback(
    const Tensor& input,
    const Tensor& pivots,
    const Tensor& infos,
    bool compute_pivots) {
  auto input_cpu = input.to(input.options().device(kCPU));
  auto pivots_cpu = pivots.to(pivots.options().device(kCPU));
  const auto infos_cpu = infos.to(infos.options().device(kCPU));

  lu_factor_stub(at::kCPU, input_cpu, pivots_cpu, infos_cpu, compute_pivots);

  input.copy_(input_cpu);
  pivots.copy_(pivots_cpu);
  infos.copy_(infos_cpu);
}

void lu_factor_kernel_xpu(
    const Tensor& input,
    const Tensor& pivots,
    const Tensor& infos,
    bool compute_pivots) {
#if defined(USE_ONEMKL_XPU)
  int64_t batch_size = native::batchCount(input);
  // TODO: optimize lu_factor performance on XPU when batch_size = 1
  if (batch_size == 1) {
    lu_factor_kernel_fallback(input, pivots, infos, compute_pivots);
  } else {
    native::xpu::lu_factor_mkl(input, pivots, infos, compute_pivots);
  }
#else
  lu_factor_kernel_fallback(input, pivots, infos, compute_pivots);
#endif // USE_ONEMKL_XPU
}

REGISTER_XPU_DISPATCH(lu_factor_stub, &lu_factor_kernel_xpu);

TORCH_IMPL_FUNC(linalg_qr_xpu_out)(const Tensor& A,
                               std::string_view mode,
                               const Tensor & Q,
                               const Tensor & R) {
#if defined(USE_ONEMKL_XPU)
  xpu::linalg_qr_kernel(A, mode, Q, R);
#else
  auto A_cpu = A.to(A.options().device(kCPU));
  auto Q_cpu = Q.to(Q.options().device(kCPU));
  auto R_cpu = R.to(R.options().device(kCPU));
  at::linalg_qr_out(Q_cpu, R_cpu, A_cpu, mode);
  Q.copy_(Q_cpu);
  R.copy_(R_cpu);
#endif // USE_ONEMKL_XPU
}


} // namespace at::native
