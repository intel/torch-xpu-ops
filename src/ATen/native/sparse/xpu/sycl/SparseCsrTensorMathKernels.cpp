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
#include <ATen/ops/_sparse_csr_tensor_unsafe_native.h>
#include <ATen/ops/_unique.h>
#include <ATen/ops/add_native.h>
#include <ATen/ops/resize_as_sparse_native.h>
#include <ATen/ops/tensor.h>
#include <ATen/ops/zeros.h>
#endif

#include <ATen/native/xpu/sycl/Reduce.h>
#include <ATen/native/xpu/sycl/Loops.h>
#include <ATen/native/xpu/sycl/pstl/PSTLFunctions.h>
#include <ATen/native/sparse/xpu/sycl/SparseCsrTensorMathKernels.h>
#include <comm/SYCLContext.h>

namespace at::native::xpu{

template <typename scalar_t>
struct ReductionAddOp {
  static inline scalar_t operator()(const scalar_t a, const scalar_t b) const {
    return a + b;
  }
  static inline scalar_t identity() const { return 0; }
  inline scalar_t identity_cpu() const { return 0; }
};

template <typename scalar_t>
struct ReductionMulOp {
  static inline scalar_t operator()(const scalar_t a, const scalar_t b) const {
    return a * b;
  }
  static inline scalar_t identity() const { return 1; }
  inline scalar_t identity_cpu() const { return 1; }
};

template <typename scalar_t, typename index_t, typename ReductionOp, typename acc_t>
struct ReduceSparseCsrDim0KernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    int64_t tid = item.get_global_linear_id();
    if (tid < new_nnz) {
      index_t col = new_col_indices[tid];
      acc_t v = rop.identity();
      for (int64_t j=0; j < nnz; j++) {
        if (col == col_indices[j]) v = rop(v, acc_t(values[j]));
      }
    }
    new_values[tid] = v;
  }

  ReduceSparseCsrDim0KernelFunctor(
    acc_t* new_values,
    const index_t* new_col_indices,
    const int64_t new_nnz,
    const scalar_t* values,
    const index_t* col_indices,
    const int64_t nnz,
    ReductionOp rop)
    : new_values(new_values),
      new_col_indices(new_col_indices),
      new_nnz(new_nnz),
      values(values),
      col_indices(col_indices),
      nnz(nnz),
      rop(rop) {}

  private:
    acc_t* new_values;
    const index_t* new_col_indices;
    const int64_t new_nnz;
    const scalar_t* values;
    const index_t* col_indices;
    const int64_t nnz;
    ReductionOp rop;
};

template <typename scalar_t, typename ReductionOp>
Tensor reduce_sparse_csr_dim0_xpu_template(const Tensor& sparse, ReductionOp rop) {
  Tensor col_indices = sparse.col_indices();
  Tensor values = sparse.values();
  auto ncols = sparse.size(1);
  auto nnz = col_indices.numel();

  auto new_col_indices = std::get<0>(at::_unique(col_indices, true, false));
  auto new_nnz = new_col_indices.numel();
  Tensor new_crow_indices = at::tensor(ArrayRef<int64_t>{0, new_nnz}, col_indices.options());

  using acc_t = at::acc_type<scalar_t, true>;
  auto acc_buffer = at::sparse_csr::create_acc_buffer<acc_t, scalar_t>(
      values.options(), values.scalar_type(), new_nnz);
  Tensor new_values = std::get<0>(acc_buffer);
  Tensor new_values_acc = std::get<1>(acc_buffer);
  scalar_t* values_ptr = values.data_ptr<scalar_t>();
  acc_t* new_values_acc_ptr = new_values_acc.data_ptr<acc_t>();
  auto queue = getCurrentSYCLQueue();

  AT_DISPATCH_INDEX_TYPES(col_indices.scalar_type(), "reduce_sparse_csr_dim0_xpu_indices",
    [&]() {
      index_t* col_indices_ptr = col_indices.data_ptr<index_t>();
      index_t* new_col_indices_ptr = new_col_indices.data_ptr<index_t>();
      using KernelFn = ReduceSparseCsrDim0KernelFunctor<scalar_t, index_t, ReductionOp, acc_t>;
      int64_t work_group_size = syclMaxWorkGroupSize<KernelFn>();
      int64_t work_group_num = (new_nnz + work_group_size - 1) / work_group_size; // fix -1
      auto kfn = KernelFn(
        new_values_acc_ptr,
        new_col_indices_ptr,
        new_nnz,
        values_ptr,
        col_indices_ptr,
        nnz,
        rop);
      sycl_kernel_submit(
        sycl::range<1>(work_group_num * work_group_size),
        sycl::range<1>(work_group_size),
        queue,
        kfn);
    });
  copy_from_acc_buffer(new_values, new_values_acc);
  return at::native::_sparse_csr_tensor_unsafe(new_crow_indices, new_col_indices, new_values,
    {1, ncols},
    new_values.scalar_type(),
    sparse.layout(),
    new_values.device());
}

template <typename index_t>
struct ReduceCrowIndicesDim1KernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    int64_t nnz = 0;
    new_crow_indices[0] = 0;
    for (int64_t i=0; i<nrows; i++) {
      if (crow_indices[i] != crow_indices[i + 1]) {
        row_map[i] = nnz;
        nnz++;
      }
      new_crow_indices[i + 1] = nnz;
    }
  }

  ReduceCrowIndicesDim1KernelFunctor(
    index_t* new_crow_indices,
    index_t* row_map,
    const index_t* crow_indices,
    const int64_t nrows)
    : new_crow_indices(new_crow_indices),
      row_map(row_map),
      crow_indices(crow_indices),
      nrows(nrows) {}
  private:
    index_t* new_crow_indices;
    index_t* row_map;
    const index_t* crow_indices;
    const int64_t nrows;
};

template <typename scalar_t, typename index_t, typename ReductionOp, typename acc_t>
struct ReduceSparseCsrDim1KernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    int64_t tid = item.get_global_linear_id();
    if (tid < nrows) {
      index_t i_start = crow_indices[tid];
      index_t i_end = crow_indices[tid+1];
      if (i_start != i_end) {
        acc_t acc = rop.identity();
        for (index_t i = i_start; i < i_end; i++) {
          acc = rop(acc, acc_t(values[i]));
        }
        new_values[row_map[tid]] = acc;
      }
    }
  }

  ReduceSparseCsrDim1KernelFunctor(
    acc_t* new_values,
    const scalar_t* values,
    const index_t* crow_indices,
    const index_t* row_map,
    const int64_t nrows,
    ReductionOp rop)
    : new_values(new_values),
      values(values),
      crow_indices(crow_indices),
      row_map(row_map),
      nrows(nrows),
      rop(rop) {}
  private:
    acc_t* new_values;
    const scalar_t* values;
    const index_t* crow_indices;
    const index_t* row_map;
    const int64_t nrows;
    ReductionOp rop;
};

template <typename scalar_t, typename ReductionOp>
Tensor reduce_sparse_csr_dim1_xpu_template(const Tensor& sparse, ReductionOp rop) {
  Tensor crow_indices = sparse.crow_indices();
  auto ioptions = crow_indices.options();
  Tensor values = sparse.values();
  auto nrows = sparse.size(0);
  auto numel = values.numel();

  Tensor new_crow_indices = at::empty({crow_indices.numel()}, ioptions);
  Tensor new_col_indices = at::empty({}, ioptions);
  Tensor row_map = at::empty({nrows}, ioptions);

  using acc_t = at::acc_type<scalar_t, true>;
  auto acc_buffer = at::sparse_csr::create_acc_buffer<acc_t, scalar_t>(
      values.options(), values.scalar_type());
  Tensor new_values = std::get<0>(acc_buffer);
  Tensor new_values_acc = std::get<1>(acc_buffer);
  auto queue = getCurrentSYCLQueue();

  AT_DISPATCH_INDEX_TYPES(col_indices.scalar_type(), "reduce_sparse_csr_dim1_xpu_indices",
    [&]() {
      using KernelFn = ReduceSparseCsrDim1KernelFunctor<scalar_t, index_t, ReductionOp, acc_t>;
      int64_t work_group_size = syclMaxWorkGroupSize<KernelFn>();
      int64_t work_group_num = (nrows + work_group_size - 1) / work_group_size; // fix -1

      index_t* crow_indices_ptr = crow_indices.data_ptr<index_t>();
      index_t* new_crow_indices_ptr = new_crow_indices.data_ptr<index_t>();
      index_t* row_map_ptr = row_map.data_ptr<index_t>();
      ReduceCrowIndicesDim1KernelFunctor<index_t> kfn_crow(
        new_crow_indices_ptr,
        row_map_ptr,
        crow_indices_ptr,
        nrows);
      sycl_kernel_submit(
        sycl::range<1>(1),
        sycl::range<1>(1),
        queue,
        kfn_crow);

      index_t new_nnz = new_crow_indices[-1].item<index_t>();
      new_col_indices.resize_(new_nnz);
      new_col_indices.fill_(index_t(0));
      new_values.resize_(new_nnz);
      new_values_acc.resize_(new_nnz);

      scalar_t* values_ptr = values.data_ptr<scalar_t>();
      acc_t* new_values_acc_ptr = new_values_acc.data_ptr<acc_t>();
      auto kfn = KernelFn(
        new_values_acc_ptr,
        values_ptr,
        crow_indices_ptr,
        row_map_ptr,
        nrows,
        rop);
      sycl_kernel_submit(
        sycl::range<1>(work_group_num * work_group_size),
        sycl::range<1>(work_group_size),
        queue,
        kfn);
    });
  copy_from_acc_buffer(new_values, new_values_acc);
  return at::native::_sparse_csr_tensor_unsafe(new_crow_indices, new_col_indices, new_values,
         {sparse.size(0), 1},
         new_values.scalar_type(),
         sparse.layout(),
         new_values.device());
}

template <typename scalar_t, typename ReductionOp>
Tensor reduce_sparse_csr_dim01_xpu_template(const Tensor& sparse, ReductionOp rop) {
  auto ioptions = sparse.col_indices().options();
  Tensor values = sparse.values();
  auto numel = values.numel();
  auto nnz = std::min<int64_t>(1, numel);

  auto result_dtype = at::isIntegralType(values.scalar_type(), /*includeBool=*/true) ? ScalarType::Long : values.scalar_type();
  Tensor new_values, new_values_acc;
  if (numel > 0) {
    new_values = at::empty({1}, values.options().dtype(result_dtype));
    new_values_acc = at::empty({1}, values.options());
    auto iter = TensorIterator::reduce_op(new_values_acc, values);
    gpu_reduce_kernel<scalar_t, scalar_t>(iter, func_wrapper<scalar_t>(rop), rop.identity_cpu());
    new_values.copy_(new_values_acc);
  } else {
    new_values = at::empty({}, values.options().dtype(result_dtype));
  }
  Tensor new_col_indices = at::zeros({nnz}, ioptions);
  Tensor new_crow_indices = at::tensor(ArrayRef<int64_t>{0, nnz}, ioptions);
  return at::native::_sparse_csr_tensor_unsafe(new_crow_indices, new_col_indices, new_values,
                                               {1, std::min<int64_t>(1, sparse.size(1))},
                                               new_values.scalar_type(),
                                               sparse.layout(),
                                               new_values.device());
}

template <typename scalar_t, typename ReductionOp>
Tensor reduce_sparse_csr_xpu_template(const Tensor& sparse, std::vector<int64_t> dims, ReductionOp rop) {
  if (dims.size() == 1) {
    if (dims[0] == 0) {
      return reduce_sparse_csr_dim0_xpu_template<scalar_t>(sparse, rop);
    } else {
      TORCH_INTERNAL_ASSERT(dims[0] == 1);
      return reduce_sparse_csr_dim1_xpu_template<scalar_t>(sparse, rop);
    }
  } else if (dims.size() == 2) {
    TORCH_INTERNAL_ASSERT(((dims[0] == 0 && dims[1] == 1) || (dims[0] == 1 && dims[1] == 0)));
    return reduce_sparse_csr_dim01_xpu_template<scalar_t>(sparse, rop);
  }
  TORCH_INTERNAL_ASSERT(dims.size() == 0);
  // effective after gh-29137 has been resolved
  return sparse.clone();
}

template <typename scalar_t, typename ReductionOp>
Tensor reduce_sparse_csr_xpu_template(const Tensor& sparse, IntArrayRef dims_to_sum, bool keepdim, ReductionOp rop) {
  TORCH_INTERNAL_ASSERT(sparse.is_sparse_csr());
  TORCH_CHECK(keepdim, "reduction operations on CSR tensors with keepdim=False is unsupported");
  TORCH_INTERNAL_ASSERT(sparse.is_xpu());

  const int64_t input_dim = sparse.dim();
  TORCH_INTERNAL_ASSERT(input_dim == 2);
  auto dims = dims_to_sum.vec();
  maybe_wrap_dims(dims, input_dim);
  if (dims.size() == 0) {
    dims.emplace_back(0);
    dims.emplace_back(1);
  }
  return reduce_sparse_csr_xpu_template<scalar_t>(sparse, dims, rop);
}

Tensor _sparse_csr_sum_xpu_kernel(const Tensor& input, IntArrayRef dims_to_sum, bool keepdim, std::optional<ScalarType> dtype) {
  ScalarType dtype_ = dtype.value_or(input.scalar_type());
  Tensor input_ = at::sparse_csr::to_type(input, dtype_);
  Tensor result;
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      kHalf, kBFloat16, input_.scalar_type(), "_sparse_csr_sum_xpu", [&] {
      using acc_t = at::acc_type<scalar_t, true>;
        result = reduce_sparse_csr_xpu_template<scalar_t>(
          input_, dims_to_sum, keepdim, ReductionAddOp<acc_t>());
      });
  return result;
}

Tensor _sparse_csr_prod_xpu_kernel(const Tensor& input, IntArrayRef dims_to_reduce, bool keepdim, std::optional<ScalarType> dtype) {
  ScalarType dtype_ = dtype.value_or(input.scalar_type());
  Tensor input_ = input.to(dtype_);
  Tensor result;
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
    kHalf, kBFloat16, input_.scalar_type(), "_sparse_csr_prod_xpu",
    [&] {
      result = reduce_sparse_csr_xpu_template<scalar_t>(
        input_, dims_to_reduce, keepdim, ReductionMulOp<scalar_t>());
    });
  return result;
}

} // namespace at::native::xpu