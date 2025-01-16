#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/SparseCsrTensorImpl.h>
#include <ATen/SparseCsrTensorUtils.h>
#include <ATen/WrapDimUtilsMulti.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/Resize.h>
#include <ATen/native/SparseTensorUtils.h>
#include <algorithm>
#include <ATen/AccumulateType.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
//#include <ATen/ops/_convert_indices_from_csr_to_coo_native.h>
#include <ATen/ops/_sparse_csr_tensor_unsafe_native.h>
#include <ATen/ops/_unique.h>
#include <ATen/ops/add_native.h>
#include <ATen/ops/resize_as_sparse_native.h>
#include <ATen/ops/tensor.h>
#include <ATen/ops/zeros.h>
#endif

#include <ATen/native/xpu/sycl/Loops.h>
#include <ATen/native/xpu/sycl/pstl/PSTLFunctions.h>
#include <comm/SYCLContext.h>
#include <xpu/ATen/ops/_convert_indices_from_coo_to_csr_native.h>

namespace at::native{

template <typename input_t, typename output_t>
struct convertIndicesFromCooToCsrXPUFunctor {
void operator()(sycl::nd_item<1> itemId) const {
    auto linear_id = itemId.get_global_linear_id();
    if (linear_id == 0) {
    for (int64_t i = 0; i <= data_in[0]; i++)
        data_out[i] = static_cast<output_t>(0);
    } else if (linear_id < numel) {
    for (int64_t i = data_in[linear_id - 1]; i < data_in[linear_id]; i++)
        data_out[i + 1] = static_cast<output_t>(linear_id);
    } else if (linear_id == numel) {
    for (int64_t i = data_in[numel - 1] + 1; i < size + 1; i++)
        data_out[i] = static_cast<output_t>(numel);
    }
}
convertIndicesFromCooToCsrXPUFunctor(
    int64_t numel_,
    const input_t* data_in_,
    output_t* data_out_,
    const int64_t size_)
    : numel(numel_), data_in(data_in_), data_out(data_out_), size(size_) {}

private:
int64_t numel;
const input_t* data_in;
output_t* data_out;
const int64_t size;
};

//With reference to coalesce_sparse_kernel
void convert_indices_from_coo_to_csr_xpu_kernel(
    const Tensor& input,
    const int64_t size,
    const bool out_int32,
    const Tensor& result){

    int64_t numel = input.numel();
    if (numel == 0) {
        result.zero_();
        return;
    }

    //How to define the global_range and local_range? --numel
    int64_t wgroup_size = 64;
    int64_t ngroups = (numel + wgroup_size - 1) / wgroup_size;
    sycl::range<1> global_range(ngroups * wgroup_size);
    sycl::range<1> local_range(wgroup_size);

    if (out_int32) {
        AT_DISPATCH_INTEGRAL_TYPES(
            input.scalar_type(), 
            "convert_indices_from_coo_to_csr_xpu", [&] {
            const scalar_t* data_in = input.data_ptr<scalar_t>();
            int* data_out = result.data_ptr<int>();
            auto functor = convertIndicesFromCooToCsrXPUFunctor<scalar_t, int>(
                numel,
                data_in,
                data_out, 
                size);
            sycl_kernel_submit(
            global_range, local_range, getCurrentSYCLQueue(), functor);
        });
    } else {
        AT_DISPATCH_INTEGRAL_TYPES(
            input.scalar_type(), 
            "convert_indices_from_coo_to_csr_xpu", [&] {
            const scalar_t* data_in = input.data_ptr<scalar_t>();
            int64_t* data_out = result.data_ptr<int64_t>();
            auto functor = convertIndicesFromCooToCsrXPUFunctor<scalar_t, int64_t>(
                numel,
                data_in,
                data_out, 
                size);
            sycl_kernel_submit(
            global_range, local_range, getCurrentSYCLQueue(), functor);
        });
    }
}

TORCH_IMPL_FUNC(_convert_indices_from_coo_to_csr_structured_xpu) (
const Tensor& input, const int64_t size, const bool out_int32, const Tensor& result){
    convert_indices_from_coo_to_csr_xpu_kernel(
        input,
        size,
        out_int32,
        result);
}   

} // namespace at::native

namespace at::native::xpu {
} // namespace at::native::xpu


