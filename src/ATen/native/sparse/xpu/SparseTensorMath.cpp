#include <ATen/native/sparse/xpu/sycl/SparseTensorMathKernels.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/addmm.h>
#include <ATen/ops/matmul.h>
#include <ATen/ops/bmm.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/_sparse_addmm.h>
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

  Tensor mat1_dense = sparse_._to_dense(std::nullopt, std::nullopt);

  r_ = t.mul(beta).add(mat1_dense.mm(dense).mul(alpha));

  // at::addmm_out(r_, t, mat1_dense, dense, beta, alpha);

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
  c10::MaybeOwned<Tensor> b_self = expand_size(self, {mat1.size(0), mat2.size(1)}, "addmm_out");
  return s_addmm_out_sparse_dense_xpu(result, *b_self, mat1, mat2, beta, alpha);
}

Tensor addmm_sparse_dense_xpu(
    const Tensor& self,
    const SparseTensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha
) {
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

Tensor sparse_sparse_matmul_xpu(const Tensor& mat1_, const Tensor& mat2_) {
  TORCH_INTERNAL_ASSERT(mat1_.is_sparse());
  TORCH_INTERNAL_ASSERT(mat2_.is_sparse());
  TORCH_CHECK(mat1_.dim() == 2);
  TORCH_CHECK(mat2_.dim() == 2);
  TORCH_CHECK(mat1_.dense_dim() == 0, "sparse_mm: scalar values expected, mat1 got ", mat1_.dense_dim(), "D values");
  TORCH_CHECK(mat2_.dense_dim() == 0, "sparse_mm: scalar values expected, mat2 got ", mat2_.dense_dim(), "D values");

  TORCH_CHECK(
      mat1_.size(1) == mat2_.size(0), "mat1 and mat2 shapes cannot be multiplied (",
      mat1_.size(0), "x", mat1_.size(1), " and ", mat2_.size(0), "x", mat2_.size(1), ")");

  TORCH_CHECK(mat1_.scalar_type() == mat2_.scalar_type(),
           "mat1 dtype ", mat1_.scalar_type(), " does not match mat2 dtype ", mat2_.scalar_type());

  Tensor mat1_dense = mat1_._to_dense(std::nullopt, std::nullopt);
  Tensor mat2_dense = mat2_._to_dense(std::nullopt, std::nullopt);

  Tensor output_dense = at::matmul(mat1_dense, mat2_dense);

  Tensor output_sparse = output_dense._to_sparse(mat1_.layout());

  return output_sparse;

  // auto output = at::native::empty_like(mat1_);
  // output.sparse_resize_and_clear_({mat1_.size(0), mat2_.size(1)}, mat1_.sparse_dim(), 0);
}

Tensor& bmm_out_sparse_xpu(const SparseTensor& self, const Tensor& mat2, Tensor& result) {
  TORCH_CHECK(!mat2.is_sparse(), "bmm_sparse: Tensor 'mat2' must be dense");
  TORCH_CHECK(self.dense_dim() == 0, "bmm_sparse: Tensor 'self' must have 0 dense dims, but has ", self.dense_dim());
  TORCH_CHECK(self.sparse_dim() == 3, "bmm_sparse: Tensor 'self' must have 3 sparse dims, but has ", self.sparse_dim());
  TORCH_CHECK(mat2.dim() == 3, "bmm_sparse: Tensor 'mat2' must have 3 dims, but has ", mat2.dim());
  TORCH_CHECK(self.size(0) == mat2.size(0), "bmm_sparse: 'self.size(0)' and 'mat2.size(0)' must match");
  TORCH_CHECK(self.size(2) == mat2.size(1), "bmm_sparse: 'self.size(2)' and 'mat2.size(1)' must match");

  int64_t num_matrices = self.size(0);
  int64_t dim_i = self.size(1);
  int64_t dim_j = self.size(2);
  int64_t dim_k = mat2.size(2);

  result.resize_({num_matrices, dim_k, dim_i});

  if ((self._nnz() == 0) || (dim_j == 0) || (dim_k == 0)) {
    result.zero_().transpose_(1, 2);
    return result;
  }

  Tensor tmp_result;
  bool need_copy_result;

  // If the result tensor is contiguous, we can just write results directly to it.
  // Otherwise, we'll need to write results to a temp buffer and then copy.
  if (result.is_contiguous()) {
    tmp_result = result;
    need_copy_result = false;
  } else {
    tmp_result = at::empty({num_matrices, dim_k, dim_i}, result.options(), at::MemoryFormat::Contiguous);
    need_copy_result = true;
  }
  Tensor mat1_dense = self._to_dense(std::nullopt, std::nullopt);
  at::bmm_out(tmp_result, mat1_dense, mat2);
  if (need_copy_result) {
    result.copy_(tmp_result);
  }
  // Need to transpose the result matrices since cusparse stores
  // them in column-major order in memory
  result.transpose_(1,2);

  return result;
}

Tensor bmm_sparse_xpu(const SparseTensor& self, const Tensor& mat2) {
  Tensor result = at::empty({self.size(0), mat2.size(2), self.size(1)}, mat2.options(), at::MemoryFormat::Contiguous);
  return bmm_out_sparse_xpu(self, mat2, result);
}

SparseTensor& hspmm_out_sparse_xpu(
    const SparseTensor& sparse_,
    const Tensor& dense,
    SparseTensor& r_
    /* , const Scalar& alpha */) {
  TORCH_CHECK(sparse_.is_xpu(), "hspmm: expected 'self' to be XPU, but got CPU");
  TORCH_CHECK(r_.is_xpu(), "hspmm: expected 'out' to be XPU, but got CPU");
  TORCH_CHECK(dense.is_xpu(), "hspmm: expected 'mat2' to be XPU, but got CPU");

  // TORCH_CHECK(xpu::check_device({r_, sparse_, dense}));

  TORCH_CHECK(sparse_.sparse_dim() == 2,
      "hspmm: Argument #2: 2D tensor expected, got ", sparse_.sparse_dim(), "D tensor");
  TORCH_CHECK(sparse_.dense_dim() == 0,
      "hspmm: Argument #2: scalar values expected, got ", sparse_.dense_dim(), "D values");
  TORCH_CHECK(dense.dim() == 2,
      "hspmm: Argument #3: 2D tensor expected, got ", dense.dim(), "D tensor");

  int64_t m = sparse_.size(0);
  int64_t k = sparse_.size(1);
  int64_t n = dense.size(1);

  TORCH_CHECK(dense.size(0) == k,
      "hspmm: Argument #3: Expected dim 0 size ", k, ", got ", dense.size(0));

                                                                                                      (r_)->resize_and_clear_(1, 1, {m, n});
  Tensor t = at::zeros({sparse_.size(-2), dense.size(-1)}, dense.options());
  Tensor output_dense = at::_sparse_addmm(t, sparse_, dense, 0, 1);
  std::cout << "output_dense" << output_dense << std::endl;
  std::cout << "sparse_.layout()" << sparse_.layout() << std::endl;
  Tensor output_sparse = output_dense._to_sparse(sparse_.layout());
  std::cout << "output_sparse" << output_sparse << std::endl;
  std::cout << "output_sparse._indices()" << output_sparse._indices() << std::endl;
  std::cout << "output_sparse._values()" << output_sparse._values() << std::endl;
  // get_sparse_impl(r_)->set_indices_and_values_unsafe(output_sparse._indices(), output_sparse._values());

  return r_;
}

SparseTensor hspmm_sparse_xpu(const SparseTensor& sparse, const Tensor& dense) {
  SparseTensor r = at::empty({0}, sparse.options());
  hspmm_out_sparse_xpu(sparse, dense, r);
  return r;
}

} // namespace at::native
