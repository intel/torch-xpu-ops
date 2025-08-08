#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/xpu/XPUContext.h>
#include <ATen/native/mkldnn/xpu/Blas.h>
#include <ATen/native/DispatchStub.h>
#include <oneapi/mkl/blas.hpp>
#include <comm/Runtime.h>

namespace at::native {

  at::Tensor& mm_complex_out_xpu(const at::Tensor &self, const at::Tensor &mat2, at::Tensor &out) {
    at::Tensor self_cont = self.contiguous();
    at::Tensor mat2_cont = mat2.contiguous();
    at::Tensor out_cont = out.contiguous();

    const int64_t m = self_cont.sizes().at(0);
    const int64_t n = mat2_cont.sizes().at(1);
    const int64_t k = self_cont.sizes().at(1);

    constexpr std::complex<float> alpha = {1.0f, 0.0f};
    constexpr std::complex<float> beta = {0.0f, 0.0f};

    oneapi::mkl::blas::row_major::gemm(
        at::xpu::getCurrentSYCLQueue(),
        oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::nontrans,
        m,
        n,
        k,
        alpha,
        reinterpret_cast<const std::complex<float>*>(self_cont.const_data_ptr()), 
        k,
        reinterpret_cast<const std::complex<float>*>(mat2_cont.const_data_ptr()),
        n,
        beta,
        reinterpret_cast<std::complex<float>*>(out_cont.data_ptr()),
        n);

    return out;
}

REGISTER_XPU_DISPATCH(mm_complex_stub, &mm_complex_out_xpu)

} // namespace at::native