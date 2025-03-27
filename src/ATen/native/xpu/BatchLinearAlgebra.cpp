#if defined(USE_ONEMKL)
#include <ATen/native/BatchLinearAlgebra.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/mkl/BatchLinearAlgebra.h>

#include <ATen/ops/_linalg_svd.h>
#include <ATen/ops/_linalg_svd_meta.h>
#include <ATen/ops/_linalg_svd_native.h>

namespace at::native {

REGISTER_XPU_DISPATCH(svd_stub, &native::xpu::svd_mkl);

} // namespace at::native
#endif // USE_ONEMKL