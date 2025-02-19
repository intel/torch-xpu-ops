#include <ATen/native/sparse/xpu/sycl/SparseTensorMathKernels.h>

namespace at::native {

using namespace at::sparse;

SparseTensor& add_out_sparse_xpu(
    const SparseTensor& t,
    const SparseTensor& src,
    const Scalar& value,
    SparseTensor& r_) {
  return xpu::add_sparse_kernel(t, src, value, r_);
}

SparseTensor& mul_out_sparse_xpu(
    const Tensor& t_,
    const Tensor& src_,
    SparseTensor& r_) {
  return xpu::mul_sparse_kernel(t_, src_, r_);
}

Tensor _sparse_sum_backward_xpu(
    const Tensor& grad_,
    const SparseTensor& input_,
    IntArrayRef dims_to_sum) {
  return xpu::_sparse_sum_backward_kernel(grad_, input_, dims_to_sum);
}

} // namespace at::native
