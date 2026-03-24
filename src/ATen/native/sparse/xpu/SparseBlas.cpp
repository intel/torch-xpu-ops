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
#include <ATen/ops/matmul.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/sgn.h>
#include <ATen/ops/zeros_like.h>
//#include <ATen/ops/addmv_native.h>
//#include <ATen/ops/copy_native.h>
//#include <ATen/ops/empty.h>
//#include <ATen/ops/mul.h>
#include <ATen/ops/resize_as_sparse_native.h>
#include <ATen/ops/scalar_tensor_native.h>
#include <ATen/ops/sparse_sampled_addmm_native.h>
#include <ATen/ops/repeat_interleave.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/ones_like.h>
#include <ATen/ops/add.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/_sparse_csr_tensor_unsafe_native.h>
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
  std::cout << "mat1: " << mat1.to(Device(at::kCPU)) << std::endl;
  std::cout << "mat2: " << mat2.to(Device(at::kCPU)) << std::endl;

  // resize result if have batch
  if (&result != &self) {
    printf(">>> Branch 1\n");
    // We allow self to be a single matrix when mat1 and mat2 are batched
    auto result_sizes = DimVector(mat1.sizes().slice(0, mat1.dim() - 2));
    result_sizes.push_back(self.size(-2));
    result_sizes.push_back(self.size(-1));
    std::cout << "result_sizes: " << result_sizes << std::endl;
    at::sparse_csr::get_sparse_csr_impl(result)->resize_(self._nnz(), result_sizes);
    //printf(">>> Signal 0: %ld vs. %ld\n", result.numel(), self.numel());
    result.copy_(self);
    printf(">>> Signal 1\n");
  }

  if (mat1.numel() == 0 || mat2.numel() == 0 || result._nnz() == 0) {
    result.mul_(beta);
    return result;
  }

  std::cout << "self.numel(): " << self.numel() << std::endl;
  std::cout << "self.sizes(): " << self.sizes() << std::endl;
  std::cout << "self.value(): " << self.values() << std::endl;
  std::cout << "Values array: ";
  auto values = self.values();
  // auto val_ptr = values.data_ptr<float>();
  // for (int i = 0; i < values.numel(); ++i) {
  //     // Print each element with 2 decimal places
  //     printf("%.10f ", val_ptr[i]);
  // }
  // Tensor self_dense;
  // if (self.numel() != 0) {
  //   self_dense = self.to_dense();
  // } else {
  //   self_dense = at::zeros_like(self);
  // }
  // std::cout << "self_dense: " << self_dense << std::endl;

  auto sizes = result.sizes();
  auto ndim = sizes.size();
  auto M = sizes[ndim - 2];
  auto N = sizes[ndim - 1];
  auto total_batch = result.numel() / (M * N);

  std::cout << "sizes: " << sizes << " M: " << M << " N: "<< N << "total_batch: " << total_batch << std::endl;

  auto crow = result.crow_indices(); // [B1, B2, ..., M+1]
  auto col = result.col_indices();   // [B1, B2, ..., nnz_per_batch]
  auto val = result.values();        // [B1, B2, ..., nnz_per_batch]

  // elment counts of all rows [B1, B2, ..., M]
  auto row_counts = (crow.narrow(-1, 1, M) - crow.narrow(-1, 0, M)).reshape((-1));
  std::cout << "row_counts: " << row_counts << std::endl;
    
  // row_base of [0, 1, ..., M-1] to flat to all batches
  auto row_base = at::arange(M, crow.options()).repeat((total_batch));
  auto row_indices = at::repeat_interleave(row_base, row_counts);

  std::cout << "row_indices: " << row_indices << std::endl;

  auto nnz_per_batch = (crow.select(-1, M) - crow.select(-1, 0)).reshape({-1});
  auto batch_indices_flat = at::repeat_interleave(
        at::arange(total_batch, crow.options()), 
        nnz_per_batch
    );
  std::cout << "batch_indices_flat: " << batch_indices_flat << std::endl;
  // restore to Multi-dim Batch Indices
  std::vector<at::Tensor> batch_coords;
  auto temp_batch_idx = batch_indices_flat;
  auto batch_sizes = sizes.slice(0, ndim - 2); // [B1, B2, ...]
    
  for (int i = batch_sizes.size() - 1; i >= 0; --i) {
      batch_coords.insert(batch_coords.begin(), temp_batch_idx % batch_sizes[i]);
      temp_batch_idx = temp_batch_idx.divide(batch_sizes[i], "trunc");
  }

  // indice [B1_idx, B2_idx, Row_idx]
  std::vector<at::Tensor> a_indices = batch_coords;
  a_indices.push_back(row_indices);
  auto a_sub = mat1.index(at::ArrayRef<at::Tensor>(a_indices)); // (Total_NNZ, K)

  std::vector<at::Tensor> b_indices = batch_coords;
  b_indices.push_back(col.reshape({-1}));
  // mat2 transpose to [..., N, K] to get column indice
  auto b_sub = mat2.transpose(-1, -2).index(at::ArrayRef<at::Tensor>(b_indices)); // (Total_NNZ, K)

  // dot product (Total_NNZ)
  auto dot_products = (a_sub * b_sub).sum(-1);

  auto new_values = (dot_products * alpha) + (val.reshape({-1}) * beta);
    
  auto result_temp = at::native::_sparse_csr_tensor_unsafe(
        crow, col, new_values.view_as(val), sizes, result.options());

  // auto crow_indices = self.crow_indices(); // (rows + 1)
  // std::cout << "crow_indices: " << crow_indices << std::endl;
  // auto col_indices = self.col_indices();   // (nnz)
  // std::cout << "col_indices: " << col_indices << std::endl;

  // auto num_rows = self.size(0);
  // // auto nnz = values.size(0);
  // auto row_indices = at::repeat_interleave(
  //       at::arange(num_rows, crow_indices.options()),
  //       crow_indices.diff()
  //   );
  // Tensor mask = at::zeros(self.sizes(), values.options());
  // Tensor ones = at::ones_like(values);
  // List<std::optional<Tensor>> indices;
  // indices.emplace_back(row_indices);
  // indices.emplace_back(col_indices);
  // mask.index_put_(indices, ones);
  // // Tensor mask = at::abs(at::sgn(self_dense));
  // std::cout << "mask: " << mask.to(Device(at::kCPU)) << std::endl;
  // Tensor masked_mm = at::matmul(mat1, mat2) * mask;
  // // Tensor result_dense = at::addmm(self_dense, mat1, mat2, beta, alpha);
  // Tensor result_dense = self_dense * beta + masked_mm * alpha;
  // printf(">>> Signal 2: %ld vs. %ld\n", result.numel(), result_dense.numel());
  // auto tmp = result_dense.to_sparse_csr();
  // std::cout << "result_dense: " << result_dense << std::endl;
  // std::cout << "result nnz: " << result._nnz() << std::endl;
  // result.copy_(result_dense.to_sparse_csr());
  // printf(">>> Signal 3\n");

  // auto a_sub = mat1.index_select(0, row_indices);
  // auto b_sub = mat2.t().index_select(0, col_indices);

  // auto sampled_product = (a_sub * b_sub).sum(1);
  // std::cout << "sampled_product: " << sampled_product << std::endl;

  // auto new_values = at::add(at::mul(sampled_product, alpha), values, beta);
  // std::cout << "new_values: " << new_values << std::endl;

  // auto result_temp = at::native::_sparse_csr_tensor_unsafe(
  //       crow_indices, col_indices, new_values, self.sizes(), new_values.scalar_type(), self.layout(), new_values.device());
  result.copy_(result_temp);
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
