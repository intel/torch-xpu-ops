#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/ExpandUtils.h>
#include <ATen/SparseCsrTensorUtils.h>
#include <ATen/TensorOperators.h>
#include <ATen/native/Resize.h>
//#include <ATen/native/sparse/cuda/SparseBlasImpl.h>
#include <ATen/native/sparse/SparseBlas.h>
//#include <ATen/native/sparse/SparseCsrTensorMath.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/abs.h>
#include <ATen/ops/mm.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/sign.h>
#include <ATen/ops/zeros_like.h>
//#include <ATen/ops/addmv_native.h>
//#include <ATen/ops/copy_native.h>
//#include <ATen/ops/empty.h>
//#include <ATen/ops/mul.h>
#include <ATen/ops/resize_as_sparse_native.h>
#include <ATen/ops/scalar_tensor_native.h>
#include <ATen/ops/sparse_sampled_addmm_native.h>
#endif

//#include <c10/util/MaybeOwned.h>

namespace at::native {

/*
  Computes `result` <- α*(A @ B) * spy(C) + β*C, where spy(C) is the sparsity pattern matrix of C.

  Args:
  * `mat1` - [in] dense Tensor A of size m × k.
  * `mat2` - [in] dense Tensor B of size k × n.
  * `self` - [in] sparse Tensor C of size m × n.
  * `result` - [out] sparse Tensor of size m × n.
*/
Tensor& sparse_sampled_addmm_out_sparse_csr_xpu(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& result) {
  at::native::sparse::sparse_sampled_addmm_check_inputs(
      self, mat1, mat2, beta, alpha, result);

  if (&result != &self) {
    //printf(">>> Branch 1\n");
    // We allow self to be a single matrix when mat1 and mat2 are batched
    auto result_sizes = DimVector(mat1.sizes().slice(0, mat1.dim() - 2));
    result_sizes.push_back(self.size(-2));
    result_sizes.push_back(self.size(-1));
    at::sparse_csr::get_sparse_csr_impl(result)->resize_(self._nnz(), result_sizes);
    //printf(">>> Signal 0: %ld vs. %ld\n", result.numel(), self.numel());
    result.copy_(self);
    //printf(">>> Signal 1\n");
  }

  if (mat1.numel() == 0 || mat2.numel() == 0 || result._nnz() == 0) {
    result.mul_(beta);
    return result;
  }

  Tensor self_dense;
  if (self.numel() != 0) {
    self_dense = self.to_dense();
  } else {
    self_dense = at::zeros_like(self);
  }
  Tensor mask = at::abs(at::sign(self_dense));
  Tensor masked_mm = at::mm(mat1, mat2) * mask;
  // Tensor result_dense = at::addmm(self_dense, mat1, mat2, beta, alpha);
  Tensor result_dense = self_dense * beta + masked_mm * alpha;
  //printf(">>> Signal 2: %ld vs. %ld\n", result.numel(), result_dense.numel());
  result.copy_(result_dense.to_sparse_csr());
  //printf(">>> Signal 3\n");

  return result;
}

Tensor sparse_sampled_addmm_sparse_csr_xpu(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha) {
  auto result = at::empty({0, 0}, self.options());
  at::native::sparse_sampled_addmm_out_sparse_csr_xpu(self, mat1, mat2, beta, alpha, result);
  return result;
}

} // namespace at::native
