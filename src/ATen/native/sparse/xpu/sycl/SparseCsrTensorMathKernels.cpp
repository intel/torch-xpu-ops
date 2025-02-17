#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/SparseCsrTensorImpl.h>
#include <ATen/SparseCsrTensorUtils.h>
#include <ATen/WrapDimUtilsMulti.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/Resize.h>
#include <ATen/native/SparseTensorUtils.h>
#include <algorithm>

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

#include <ATen/native/sparse/xpu/sycl/SparseCsrTensorMathKernels.h>
#include <ATen/native/xpu/sycl/Loops.h>
#include <ATen/native/xpu/sycl/Reduce.h>
#include <ATen/native/xpu/sycl/pstl/PSTLFunctions.h>
#include <comm/SYCLContext.h>

namespace at::native::xpu {

using namespace at::sparse_csr;
using namespace at::sparse;

template <typename scalar_t>
struct ReductionAddOp {
  inline scalar_t operator()(const scalar_t a, const scalar_t b) const {
    return a + b;
  }
  inline scalar_t identity() const {
    return 0;
  }
  inline scalar_t identity_cpu() const {
    return 0;
  }
};

template <typename scalar_t>
struct ReductionMulOp {
  inline scalar_t operator()(const scalar_t a, const scalar_t b) const {
    return a * b;
  }
  inline scalar_t identity() const {
    return 1;
  }
  inline scalar_t identity_cpu() const {
    return 1;
  }
};

template <
    typename scalar_t,
    typename index_t,
    typename ReductionOp,
    typename acc_t>
struct ReduceSparseCsrDim0KernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    int64_t tid = item.get_global_linear_id();
    if (tid < new_nnz) {
      index_t col = new_col_indices[tid];
      acc_t v = rop.identity();
      for (int64_t j = 0; j < nnz; j++) {
        if (col == col_indices[j])
          v = rop(v, acc_t(values[j]));
      }
      new_values[tid] = v;
    }
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
Tensor reduce_sparse_csr_dim0_xpu_template(
    const Tensor& sparse,
    ReductionOp rop) {
  Tensor col_indices = sparse.col_indices();
  Tensor values = sparse.values();
  auto ncols = sparse.size(1);
  auto nnz = col_indices.numel();

  auto new_col_indices = std::get<0>(at::_unique(col_indices, true, false));
  auto new_nnz = new_col_indices.numel();
  Tensor new_crow_indices =
      at::tensor(ArrayRef<int64_t>{0, new_nnz}, col_indices.options());

  using acc_t = at::acc_type<scalar_t, true>;
  auto acc_buffer = at::sparse_csr::create_acc_buffer<acc_t, scalar_t>(
      values.options(), values.scalar_type(), new_nnz);
  Tensor new_values = std::get<0>(acc_buffer);
  Tensor new_values_acc = std::get<1>(acc_buffer);
  scalar_t* values_ptr = values.data_ptr<scalar_t>();
  acc_t* new_values_acc_ptr = new_values_acc.data_ptr<acc_t>();
  auto queue = getCurrentSYCLQueue();

  AT_DISPATCH_INDEX_TYPES(
      col_indices.scalar_type(), "reduce_sparse_csr_dim0_xpu_indices", [&]() {
        index_t* col_indices_ptr = col_indices.data_ptr<index_t>();
        index_t* new_col_indices_ptr = new_col_indices.data_ptr<index_t>();
        using KernelFn = ReduceSparseCsrDim0KernelFunctor<
            scalar_t,
            index_t,
            ReductionOp,
            acc_t>;
        int64_t work_group_size = syclMaxWorkGroupSize<KernelFn>();
        int64_t work_group_num =
            (new_nnz + work_group_size - 1) / work_group_size;
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
  return at::native::_sparse_csr_tensor_unsafe(
      new_crow_indices,
      new_col_indices,
      new_values,
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
    for (int64_t i = 0; i < nrows; i++) {
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

template <
    typename scalar_t,
    typename index_t,
    typename ReductionOp,
    typename acc_t>
struct ReduceSparseCsrDim1KernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    int64_t tid = item.get_global_linear_id();
    if (tid < nrows) {
      index_t i_start = crow_indices[tid];
      index_t i_end = crow_indices[tid + 1];
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
Tensor reduce_sparse_csr_dim1_xpu_template(
    const Tensor& sparse,
    ReductionOp rop) {
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

  AT_DISPATCH_INDEX_TYPES(
      crow_indices.scalar_type(), "reduce_sparse_csr_dim1_xpu_indices", [&]() {
        using KernelFn = ReduceSparseCsrDim1KernelFunctor<
            scalar_t,
            index_t,
            ReductionOp,
            acc_t>;
        int64_t work_group_size = syclMaxWorkGroupSize<KernelFn>();
        int64_t work_group_num =
            (nrows + work_group_size - 1) / work_group_size;

        index_t* crow_indices_ptr = crow_indices.data_ptr<index_t>();
        index_t* new_crow_indices_ptr = new_crow_indices.data_ptr<index_t>();
        index_t* row_map_ptr = row_map.data_ptr<index_t>();
        ReduceCrowIndicesDim1KernelFunctor<index_t> kfn_crow(
            new_crow_indices_ptr, row_map_ptr, crow_indices_ptr, nrows);
        sycl_kernel_submit(
            sycl::range<1>(1), sycl::range<1>(1), queue, kfn_crow);

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
  return at::native::_sparse_csr_tensor_unsafe(
      new_crow_indices,
      new_col_indices,
      new_values,
      {sparse.size(0), 1},
      new_values.scalar_type(),
      sparse.layout(),
      new_values.device());
}

template <typename scalar_t, typename ReductionOp>
Tensor reduce_sparse_csr_dim01_xpu_template(
    const Tensor& sparse,
    ReductionOp rop) {
  auto ioptions = sparse.col_indices().options();
  Tensor values = sparse.values();
  auto numel = values.numel();
  auto nnz = std::min<int64_t>(1, numel);

  auto result_dtype =
      at::isIntegralType(values.scalar_type(), /*includeBool=*/true)
      ? ScalarType::Long
      : values.scalar_type();
  Tensor new_values, new_values_acc;
  if (numel > 0) {
    new_values = at::empty({1}, values.options().dtype(result_dtype));
    new_values_acc = at::empty({1}, values.options());
    auto iter = TensorIterator::reduce_op(new_values_acc, values);
    gpu_reduce_kernel<scalar_t, scalar_t>(
        iter, func_wrapper<scalar_t>(rop), rop.identity_cpu());
    new_values.copy_(new_values_acc);
  } else {
    new_values = at::empty({}, values.options().dtype(result_dtype));
  }
  Tensor new_col_indices = at::zeros({nnz}, ioptions);
  Tensor new_crow_indices = at::tensor(ArrayRef<int64_t>{0, nnz}, ioptions);
  return at::native::_sparse_csr_tensor_unsafe(
      new_crow_indices,
      new_col_indices,
      new_values,
      {1, std::min<int64_t>(1, sparse.size(1))},
      new_values.scalar_type(),
      sparse.layout(),
      new_values.device());
}

template <typename scalar_t, typename ReductionOp>
Tensor reduce_sparse_csr_xpu_template(
    const Tensor& sparse,
    std::vector<int64_t> dims,
    ReductionOp rop) {
  if (dims.size() == 1) {
    if (dims[0] == 0) {
      return reduce_sparse_csr_dim0_xpu_template<scalar_t>(sparse, rop);
    } else {
      TORCH_INTERNAL_ASSERT(dims[0] == 1);
      return reduce_sparse_csr_dim1_xpu_template<scalar_t>(sparse, rop);
    }
  } else if (dims.size() == 2) {
    TORCH_INTERNAL_ASSERT(
        ((dims[0] == 0 && dims[1] == 1) || (dims[0] == 1 && dims[1] == 0)));
    return reduce_sparse_csr_dim01_xpu_template<scalar_t>(sparse, rop);
  }
  TORCH_INTERNAL_ASSERT(dims.size() == 0);
  return sparse.clone();
}

template <typename scalar_t, typename ReductionOp>
Tensor reduce_sparse_csr_xpu_template(
    const Tensor& sparse,
    IntArrayRef dims_to_sum,
    bool keepdim,
    ReductionOp rop) {
  TORCH_INTERNAL_ASSERT(sparse.is_sparse_csr());
  TORCH_CHECK(
      keepdim,
      "reduction operations on CSR tensors with keepdim=False is unsupported");
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

Tensor _sparse_csr_sum_xpu_kernel(
    const Tensor& input,
    IntArrayRef dims_to_sum,
    bool keepdim,
    std::optional<ScalarType> dtype) {
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

Tensor _sparse_csr_prod_xpu_kernel(
    const Tensor& input,
    IntArrayRef dims_to_reduce,
    bool keepdim,
    std::optional<ScalarType> dtype) {
  ScalarType dtype_ = dtype.value_or(input.scalar_type());
  Tensor input_ = input.to(dtype_);
  Tensor result;
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      kHalf, kBFloat16, input_.scalar_type(), "_sparse_csr_prod_xpu", [&] {
        result = reduce_sparse_csr_xpu_template<scalar_t>(
            input_, dims_to_reduce, keepdim, ReductionMulOp<scalar_t>());
      });
  return result;
}

template <typename input_t, typename output_t>
struct ConvertIndicesFromCooToCsrXPUFunctor {
  void operator()(sycl::nd_item<1> item) const {
    auto linear_id = item.get_global_linear_id();
    if (linear_id == 0) {
      for (int64_t i = 0; i <= data_in_[0]; i++)
        data_out_[i] = static_cast<output_t>(0);
    } else if (linear_id < numel_) {
      for (int64_t i = data_in_[linear_id - 1]; i < data_in_[linear_id]; i++)
        data_out_[i + 1] = static_cast<output_t>(linear_id);
    } else if (linear_id == numel_) {
      for (int64_t i = data_in_[numel_ - 1] + 1; i < size_ + 1; i++)
        data_out_[i] = static_cast<output_t>(numel_);
    }
  }
  ConvertIndicesFromCooToCsrXPUFunctor(
      int64_t numel,
      const input_t* data_in,
      output_t* data_out,
      const int64_t size)
      : numel_(numel), data_in_(data_in), data_out_(data_out), size_(size) {}

 private:
  int64_t numel_;
  const input_t* data_in_;
  output_t* data_out_;
  const int64_t size_;
};

template <typename input_t, typename output_t>
struct ConvertIndicesFromCsrToCooXPUFunctor {
  void operator()(sycl::nd_item<1> item) const {
    int64_t tid = item.get_global_linear_id();
    if (tid < nrows_ * nbatches_) {
      int64_t b = tid / nrows_;
      int64_t i_ = b * (nrows_ + 1) + tid % nrows_;
      for (int64_t i = data_in_[i_]; i < data_in_[i_ + 1]; i++) {
        data_out_[b * nnz_ + i] = static_cast<output_t>(tid % nrows_);
      }
    }
  }
  ConvertIndicesFromCsrToCooXPUFunctor(
      output_t* data_out,
      const input_t* data_in,
      const int64_t nrows,
      const int64_t nnz,
      const int64_t nbatches)
      : data_out_(data_out),
        data_in_(data_in),
        nrows_(nrows),
        nnz_(nnz),
        nbatches_(nbatches) {}

 private:
  output_t* data_out_;
  const input_t* data_in_;
  const int64_t nrows_;
  const int64_t nnz_;
  const int64_t nbatches_;
};

template <typename input_t, typename output_t>
void launch_convert_indices_from_coo_to_csr_xpu_kernel(
    const Tensor& result,
    const Tensor& input,
    const int64_t size) {
  int64_t numel = input.numel();
  if (numel == 0) {
    result.zero_();
    return;
  }

  const input_t* data_in = input.const_data_ptr<input_t>();
  output_t* data_out = result.data_ptr<output_t>();

  auto functor = ConvertIndicesFromCooToCsrXPUFunctor<input_t, output_t>(
      numel, data_in, data_out, size);

  int64_t wgroup_size = syclMaxWorkGroupSize(functor);
  int64_t ngroups = (numel + wgroup_size - 1) / wgroup_size;
  sycl::range<1> global_range(ngroups * wgroup_size);
  sycl::range<1> local_range(wgroup_size);

  sycl_kernel_submit(global_range, local_range, getCurrentSYCLQueue(), functor);
}

template <typename input_t, typename output_t>
void launch_convert_indices_from_csr_to_coo_xpu_kernel(
    const Tensor& indices,
    const Tensor& crow_indices,
    const Tensor& col_indices,
    const bool transpose = false) {
  int64_t nrows = crow_indices.size(-1) - 1;
  int64_t nnz = col_indices.size(-1);
  if (nrows == 0 || nnz == 0) {
    indices.zero_();
    return;
  }
  int64_t total_nnz = col_indices.numel();
  int64_t batch_ndim = crow_indices.dim() - 1;
  if (batch_ndim > 0) {
    auto batch_indices = indices.narrow(0, 0, batch_ndim);
    batch_indices.copy_(
        at::sparse::full_coo_indices(
            crow_indices.sizes().slice(0, batch_ndim), indices.options())
            .repeat_interleave(nnz, 1));
  }

  auto crow_indices_ = crow_indices.expect_contiguous();
  const input_t* crow_indices_data_in =
      crow_indices_->const_data_ptr<input_t>();
  TORCH_INTERNAL_ASSERT(indices.is_contiguous());
  auto row0 = indices.select(0, transpose ? batch_ndim + 1 : batch_ndim + 0);
  auto row1 = indices.select(0, transpose ? batch_ndim + 0 : batch_ndim + 1);
  auto col_indices_ = col_indices.expect_contiguous();
  row1.copy_(col_indices_->view({-1}));
  output_t* data_out = row0.data_ptr<output_t>();

  // Run nrows * nbatches threads...
  int64_t nbatches = total_nnz / nnz;
  auto functor = ConvertIndicesFromCsrToCooXPUFunctor<input_t, output_t>(
      data_out, crow_indices_data_in, nrows, nnz, nbatches);

  int64_t THREADS = syclMaxWorkGroupSize(functor);
  int64_t GROUPS = (nrows * nbatches + THREADS) / THREADS;

  sycl::range<1> global_range(GROUPS * THREADS);
  sycl::range<1> local_range(THREADS);

  sycl_kernel_submit(global_range, local_range, getCurrentSYCLQueue(), functor);
}

void convert_indices_from_coo_to_csr_structured_kernel(
    const Tensor& input,
    const int64_t size,
    const bool out_int32,
    const Tensor& result) {
  if (out_int32) {
    AT_DISPATCH_INTEGRAL_TYPES(
        input.scalar_type(), "convert_indices_from_coo_to_csr_xpu", [&] {
          launch_convert_indices_from_coo_to_csr_xpu_kernel<scalar_t, int>(
              result, input, size);
        });
  } else {
    AT_DISPATCH_INTEGRAL_TYPES(
        input.scalar_type(), "convert_indices_from_coo_to_csr_xpu", [&] {
          launch_convert_indices_from_coo_to_csr_xpu_kernel<scalar_t, int64_t>(
              result, input, size);
        });
  }
}

void convert_indices_from_csr_to_coo_structured_kernel(
    const Tensor& crow_indices,
    const Tensor& col_indices,
    const bool out_int32,
    const bool transpose,
    const Tensor& result) {
  if (out_int32) {
    AT_DISPATCH_INTEGRAL_TYPES(
        crow_indices.scalar_type(), "convert_indices_from_csr_to_coo_xpu", [&] {
          launch_convert_indices_from_csr_to_coo_xpu_kernel<scalar_t, int>(
              result, crow_indices, col_indices, transpose);
        });
  } else {
    AT_DISPATCH_INTEGRAL_TYPES(
        crow_indices.scalar_type(), "convert_indices_from_csr_to_coo_xpu", [&] {
          launch_convert_indices_from_csr_to_coo_xpu_kernel<scalar_t, int64_t>(
              result, crow_indices, col_indices, transpose);
        });
  }
}

} // namespace at::native::xpu
