#include <ATen/native/sparse/xpu/sycl/SparseTensorMathKernels.h>
#include <ATen/native/TensorConversions.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/addmm.h>
#endif

#include <ATen/ExpandUtils.h>

#include <iostream>

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

Tensor& s_addmm_out_sparse_dense_xpu(Tensor& r_, const Tensor& t, const SparseTensor& sparse_, const Tensor& dense, const Scalar& beta, const Scalar& alpha) {
  TORCH_CHECK(t.is_xpu(), "Expected all tensors to be on the same device. addmm: expected 'self' to be XPU, but got CPU");
  TORCH_CHECK(r_.is_xpu(), "Expected all tensors to be on the same device. addmm: expected 'out' to be XPU, but got CPU");
  TORCH_CHECK(sparse_.is_xpu(), "Expected all tensors to be on the same device. addmm: expected 'mat1' to be XPU, but got CPU");
  TORCH_CHECK(dense.is_xpu(), "Expected all tensors to be on the same device. addmm: expected 'mat2' to be XPU, but got CPU");

  // TORCH_CHECK(xpu::check_device({sparse_, r_, t, dense}));

  TORCH_CHECK(dense.dim() == 2, "addmm: 2D tensor expected, got ", dense.dim(), "D tensor");
  TORCH_CHECK(sparse_.sparse_dim() == 2, "addmm: expected first two dims to be sparse (indices has size 2 at first dim), but got ", sparse_.sparse_dim(), " sparse dims");
  // no need to check dense_dim because dense_dim + sparse_dim = dim

  // change sparse to dense
  Tensor mat1_dense = at::native::sparse_to_dense(sparse_);
  // calculate
  at::addmm_out(r_, t, mat1_dense, dense, beta, alpha);

  return r_;
}

Tensor s_addmm_sparse_dense_xpu(
    const Tensor& t,
    const SparseTensor& sparse,
    const Tensor& dense,
    const Scalar& beta,
    const Scalar& alpha
) {
  Tensor r = at::empty({0}, t.options());
  s_addmm_out_sparse_dense_xpu(r, t, sparse, dense, beta, alpha);
  return r;
}


Tensor& addmm_out_sparse_dense_xpu(
    const Tensor& self,
    const SparseTensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& result
) {
  std::cout << "Call addmm_out_sparse_dense_xpu" << std::endl;
  c10::MaybeOwned<Tensor> b_self = expand_size(self, {mat1.size(0), mat2.size(1)}, "addmm_out");
  return s_addmm_sparse_dense_xpu(*b_self, mat1, mat2, beta, alpha);
}

Tensor& s_addmm_sparse_dense_xpu_(
    Tensor& t,
    const SparseTensor& sparse,
    const Tensor& dense,
    const Scalar& beta,
    const Scalar& alpha
) {
  return s_addmm_out_sparse_dense_xpu(t, t, sparse, dense, beta, alpha);
}

} // namespace at::native
