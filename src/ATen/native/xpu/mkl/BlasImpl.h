#include <ATen/core/Tensor.h>

namespace at::native::xpu {

Tensor& mm_complex_out_xpu_mkl(
    const Tensor& self,
    const Tensor& mat2,
    Tensor& out);

Tensor& bmm_complex_out_xpu_mkl(
    const Tensor& self,
    const Tensor& mat2,
    Tensor& out);

Tensor& addmm_complex_out_xpu_mkl(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& out);

Tensor& baddbmm_complex_out_xpu_mkl(
    const Tensor& self,
    const Tensor& batch1,
    const Tensor& batch2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& out);

} // namespace at::native::xpu
