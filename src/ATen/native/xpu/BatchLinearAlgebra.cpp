#if defined(USE_ONEMKL)
#include <ATen/native/BatchLinearAlgebra.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/LinearAlgebra.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/mkl/BatchLinearAlgebra.h>

#include <ATen/ops/_linalg_svd.h>
#include <ATen/ops/_linalg_svd_meta.h>
#include <ATen/ops/_linalg_svd_native.h>
#include <ATen/ops/linalg_lu_factor_native.h>
#include <ATen/ops/linalg_lu_meta.h>
#include <ATen/ops/linalg_lu_native.h>
#include <ATen/ops/linalg_lu_solve.h>
#include <ATen/ops/linalg_lu_solve_meta.h>
#include <ATen/ops/linalg_lu_solve_native.h>

namespace at::native {

REGISTER_XPU_DISPATCH(svd_stub, &native::xpu::svd_mkl);
REGISTER_XPU_DISPATCH(lu_solve_stub, &native::xpu::lu_solve_mkl);
REGISTER_XPU_DISPATCH(lu_factor_stub, &native::xpu::lu_factor_mkl);

} // namespace at::native
#endif // USE_ONEMKL