#include <ATen/native/sparse/ParamUtils.h>
#include <ATen/native/sparse/SparseStubs.h>
#include <ATen/native/sparse/xpu/sycl/SparseSoftmaxKernels.h>

namespace at::native {

using namespace at::sparse;

Tensor softmax_sparse_xpu(
    const Tensor& input_,
    const int64_t dim_,
    const bool half_to_float) {
  return xpu::softmax_sparse_xpu_kernel(input_, dim_, half_to_float);
}

Tensor log_softmax_sparse_xpu(
    const Tensor& input_,
    const int64_t dim_,
    const bool half_to_float) {
  return xpu::log_softmax_sparse_xpu_kernel(input_, dim_, half_to_float);
}

Tensor softmax_backward_sparse_xpu(
    const Tensor& grad_,
    const Tensor& output_,
    int64_t dim_,
    const Tensor& input_) {
  return xpu::softmax_backward_sparse_xpu_kernel(grad_, output_, dim_, input_);
}

Tensor log_softmax_backward_sparse_xpu(
    const Tensor& grad_,
    const Tensor& output_,
    int64_t dim_,
    const Tensor& input_) {
  return xpu::log_softmax_backward_sparse_xpu_kernel(
      grad_, output_, dim_, input_);
}
} // namespace at::native
