#pragma once
#include <xpu/ATen/ops/_convert_indices_from_coo_to_csr_native.h>
#include <ATen/native/sparse/xpu/sycl/SparseCsrTensorMathKernels.h>

namespace at::native{

TORCH_IMPL_FUNC(_convert_indices_from_coo_to_csr_structured_xpu)(
    const Tensor& input, 
    const int64_t size, 
    const bool out_int32, 
    const Tensor& result){
        xpu::convert_indices_from_coo_to_csr_structured_kernel(
            input, 
            size, 
            out_int32, 
            result);
    };

} // namespace at::native