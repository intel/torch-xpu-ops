#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/native/SparseTensorUtils.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/sparse/SparseTensorMath.h>
#include <c10/core/ScalarType.h>

#include <comm/Runtime.h>
#include <comm/SYCLContext.h>
#include <comm/SYCLHelpers.h>
#include <comm/TensorInfo.h>

#include <ATen/native/sparse/Utils.h>
#include <ATen/native/sparse/xpu/sycl/SparseTensorMathKernel.h>
#include <ATen/xpu/XPUUtils.h>

#define I_INFO(tensor) at::xpu::detail::getTensorInfo<int64_t, uint64_t>(tensor)
#define V_INFO(tensor) \
  at::xpu::detail::getTensorInfo<scalar_t, uint64_t>(tensor)

namespace at::native::xpu {

namespace apply {
using at::xpu::detail::TensorInfo;
using indexT = int64_t;

template <typename Op, typename IndexType, typename Real>
struct SparseElementwiseKernellFunctor {
  void operator()(sycl::nd_item<1> item) const {
    IndexType ind_skip = indices.strides[0];
    IndexType ind_nnz_skip = indices.strides[1];
    IndexType value_size = values.strides[0];

    for (IndexType linearId = (IndexType)item.get_group_linear_id();
         linearId < nnz;
         linearId += (IndexType)item.get_group_range()[0]) {
      IndexType index = 0;
      for (IndexType d = 0; d < indices.sizes[0]; d++) {
        index = dense.sizes[d] * index +
            indices.data[d * ind_skip + linearId * ind_nnz_skip];
      }
      Real* dst = dense.data + index * value_size;
      Real* src = values.data + linearId * value_size;
      for (IndexType linearId2 = (IndexType)item.get_local_id()[0];
           linearId2 < value_size;
           linearId2 += (IndexType)item.get_local_range()[0]) {
        op(dst + linearId2, src + linearId2);
      }
    }
  }
  SparseElementwiseKernellFunctor(
      Op op,
      TensorInfo<Real, IndexType> dense_info,
      TensorInfo<indexT, IndexType> indices_info,
      TensorInfo<Real, IndexType> values_info,
      const IndexType nnz_value)
      : op(op),
        dense(dense_info),
        indices(indices_info),
        values(values_info),
        nnz(nnz_value) {}

 private:
  Op op;
  TensorInfo<Real, IndexType> dense;
  TensorInfo<indexT, IndexType> indices;
  TensorInfo<Real, IndexType> values;
  const IndexType nnz;
};

template <typename Op, typename IndexType, typename Real>
void sparseElementwiseKernel(
    Op op,
    TensorInfo<Real, IndexType> dense,
    TensorInfo<indexT, IndexType> indices,
    TensorInfo<Real, IndexType> values,
    const IndexType nnz) {
  using KernelClass = SparseElementwiseKernellFunctor<Op, IndexType, Real>;
  auto& queue = getCurrentSYCLQueue();
  IndexType group_size = (IndexType)syclMaxWorkGroupSize<KernelClass>();
  IndexType target_global_size = (IndexType)syclMaxWorkItemsPerTile();
  auto max_work_group_num = target_global_size / group_size;

  auto num_groups = CeilDiv(nnz, group_size);
  if (num_groups > max_work_group_num)
    num_groups = max_work_group_num;
  auto total_items = num_groups * group_size;

  KernelClass kfn(op, dense, indices, values, nnz);
  sycl_kernel_submit(total_items, group_size, queue, kfn);
}

template <typename Op, typename IndexType, typename Real>
struct SparseElementwiseKernelScalarFunctor {
  void operator()(sycl::nd_item<1> item) const {
    IndexType ind_skip = indices.strides[0];
    IndexType ind_nnz_skip = indices.strides[1];
    IndexType value_skip = values.strides[0];

    for (IndexType linearId = (IndexType)item.get_group_linear_id() *
                 (IndexType)item.get_local_range()[0] +
             (IndexType)item.get_local_id()[0];
         linearId < nnz;
         linearId += (IndexType)item.get_group_range()[0] *
             (IndexType)item.get_local_range()[0]) {
      IndexType index = 0;
      for (IndexType d = 0; d < indices.sizes[0]; d++) {
        index = dense.sizes[d] * index +
            indices.data[d * ind_skip + linearId * ind_nnz_skip];
      }
      op(dense.data + index, values.data + linearId * value_skip);
    }
  }
  SparseElementwiseKernelScalarFunctor(
      Op op,
      TensorInfo<Real, IndexType> dense_info,
      TensorInfo<indexT, IndexType> indices_info,
      TensorInfo<Real, IndexType> values_info,
      const IndexType nnz_value)
      : op(op),
        dense(dense_info),
        indices(indices_info),
        values(values_info),
        nnz(nnz_value) {}

 private:
  Op op;
  TensorInfo<Real, IndexType> dense;
  TensorInfo<indexT, IndexType> indices;
  TensorInfo<Real, IndexType> values;
  const IndexType nnz;
};

template <typename Op, typename IndexType, typename Real>
void sparseElementwiseKernelScalar(
    Op op,
    TensorInfo<Real, IndexType> dense,
    TensorInfo<indexT, IndexType> indices,
    TensorInfo<Real, IndexType> values,
    const IndexType nnz) {
  using KernelClass = SparseElementwiseKernelScalarFunctor<Op, IndexType, Real>;
  auto& queue = getCurrentSYCLQueue();
  IndexType group_size = (IndexType)syclMaxWorkGroupSize<KernelClass>();
  IndexType target_global_size = (IndexType)syclMaxWorkItemsPerTile();
  auto max_work_group_num = target_global_size / group_size;

  auto num_groups = CeilDiv(nnz * group_size, group_size);
  if (num_groups > max_work_group_num)
    num_groups = max_work_group_num;
  auto total_items = num_groups * group_size;

  KernelClass kfn(op, dense, indices, values, nnz);
  sycl_kernel_submit(total_items, group_size, queue, kfn);
}
} // namespace apply

template <typename T>
struct TensorCAddOp {
  TensorCAddOp(T v) : val(v) {}

  void operator()(T* out, T* in) const {
    *out += val * *in;
  }

  void operator()(T* out, T* in1, T* in2) const {
    *out = *in1 + val * *in2;
  }

  T val;
};

TORCH_XPU_API void add_out_dense_sparse_kernel(
    Tensor& r_,
    const Tensor& dense,
    const SparseTensor& sparse,
    const Scalar& value) {
  TORCH_CHECK(
      dense.is_xpu(),
      "add: expected 'self' to be a XPU tensor, but got a other device tensor");
  TORCH_CHECK(
      sparse.is_xpu(),
      "add: expected 'other' to be a XPU tensor, but got a other device tensor");
  TORCH_CHECK(
      r_.is_xpu(),
      "add: expected 'out' to be a XPU tensor, but got a other device tensor");

  TORCH_CHECK(at::xpu::check_device({sparse, r_, dense}));

  TORCH_CHECK(
      dense.sizes().equals(sparse.sizes()),
      "add: expected 'self' and 'other' to have same size, but self has size ",
      dense.sizes(),
      " while other has size ",
      sparse.sizes(),
      " (FYI: dense-sparse addition does not currently support broadcasting)");

  const int64_t nnz = sparse._nnz();
  if (nnz == 0) {
    r_.resize_as_(dense);
    r_.copy_(dense);
    return;
  }

  auto commonDtype = at::result_type(dense, sparse);
  TORCH_CHECK(
      canCast(commonDtype, r_.scalar_type()),
      "Can't convert result type ",
      commonDtype,
      " to output ",
      r_.scalar_type());

  Tensor r = r_;
  if (r_.scalar_type() != commonDtype) {
    r = at::empty_like(dense, r_.options().dtype(commonDtype));
  }

  Tensor dense_buffer = dense.to(commonDtype);
  Tensor values = sparse._values().to(commonDtype);

  if (!is_same_tensor(r, dense_buffer)) {
    r.resize_as_(dense);
    r.copy_(dense_buffer);
  }

  Tensor indices = sparse._indices();
  int64_t nDim = dense.dim();
  int64_t nDimI = sparse.sparse_dim();

  if (values.numel() == 0) {
    return;
  }

  if (sparse.is_coalesced()) {
    // TODO benchmark to decide whether to remove this special case
    if (sparse.dense_dim() == 0) {
      AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
          at::ScalarType::ComplexHalf,
          at::ScalarType::Bool,
          at::ScalarType::Half,
          at::ScalarType::BFloat16,
          commonDtype,
          "add_out_dense_sparse_xpu",
          [&] {
            apply::sparseElementwiseKernelScalar(
                TensorCAddOp<scalar_t>(value.to<scalar_t>()),
                V_INFO(r),
                I_INFO(indices),
                V_INFO(values),
                static_cast<uint64_t>(nnz));
          });
    } else {
      // sparseElementwiseKernel needs values to be contiguous too
      values = values.contiguous();

      AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
          at::ScalarType::ComplexHalf,
          at::ScalarType::Bool,
          at::ScalarType::Half,
          at::ScalarType::BFloat16,
          commonDtype,
          "add_out_dense_sparse_xpu",
          [&] {
            apply::sparseElementwiseKernel(
                TensorCAddOp<scalar_t>(value.to<scalar_t>()),
                V_INFO(r),
                I_INFO(indices),
                V_INFO(values),
                static_cast<uint64_t>(nnz));
          });
    }
  } else {
    Tensor indices1D = flatten_indices(indices, sparse.sizes(), 0);

    int64_t view_rows = 1;
    int64_t view_columns = 1;
    for (int i = 0; i < nDimI; i++) {
      view_rows *= r.size(i);
    }
    for (int i = nDimI; i < nDim; i++) {
      view_columns *= r.size(i);
    }

    Tensor r_view = r.view({view_rows, view_columns});
    values = values.reshape({nnz, view_columns});
    r_view.index_add_(0, indices1D, values, value);
  }

  r_.copy_(r);
}

TORCH_XPU_API void add_out_sparse_kernel(
    const SparseTensor& t,
    const SparseTensor& src,
    const Scalar& value,
    SparseTensor& r_) {
  // TODO: This test seems a bit goofy
  TORCH_CHECK(
      src.is_sparse(),
      "add(sparse, dense) is not supported. Use add(dense, sparse) instead.");

  TORCH_CHECK(
      t.is_xpu(), "add: expected 'self' to be XPU, but got other device");
  TORCH_CHECK(
      src.is_xpu(), "add: expected 'other' to be XPU, but got other device");
  TORCH_CHECK(
      r_.is_xpu(), "add: expected 'out' to be XPU, but got other device");

  TORCH_CHECK(at::xpu::check_device({r_, t, src}));

  auto commonDtype = at::result_type(t, src);
  TORCH_CHECK(
      canCast(commonDtype, r_.scalar_type()),
      "Can't convert result type ",
      commonDtype,
      " to output ",
      r_.scalar_type());

  TORCH_CHECK(
      t.sizes().equals(src.sizes()),
      "add: expected 'self' and 'other' to have same size, but ",
      t.sizes(),
      " != ",
      src.sizes());

  if (src._nnz() == 0) {
    copy_sparse_to_sparse_(r_, t);
    return;
  }
  if (t._nnz() == 0) {
    mul_out_sparse_scalar(r_, src, value);
    return;
  }

  TORCH_CHECK(
      is_same_density(t, src),
      "add: expected 'self' and 'other' to have same density, but 'self' has ",
      t.sparse_dim(),
      " sparse dimensions while 'other' has ",
      src.sparse_dim(),
      " sparse dimensions");

  // We deliberately choose to simply concat the indices and values tensors
  // rather than merging them. This removes the need to synchronously fetch nnz
  // at the end of the operation, at the cost of having a non-coalesced result.
  // This trade-off is preferable for the common use-case of gradient
  // accumulation.
  Tensor t_indices_ = t._indices();
  Tensor s_indices_ = src._indices();

  Tensor t_values_ = t._values().to(commonDtype);
  Tensor s_values_ = src._values().to(commonDtype);

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      commonDtype,
      "add_out_sparse_xpu",
      [&] {
        if (value.to<scalar_t>() != scalar_t(1)) {
          s_values_ = s_values_.mul(value);
        }
      });
  Tensor r_indices_ = at::cat({t_indices_, s_indices_}, 1);
  Tensor r_values_ = at::cat({t_values_, s_values_}, 0);

  if (r_.scalar_type() != commonDtype) {
    SparseTensor promoted = at::empty({0}, r_.options().dtype(commonDtype));
    promoted.resize_as_(src);
    alias_into_sparse(promoted, r_indices_, r_values_);
    // performs the addition under the common dtype.
    promoted = promoted.coalesce();
    r_values_ = promoted._values().to(r_.scalar_type());
    r_indices_ = promoted._indices();
  } else {
    r_.resize_as_(src);
  }

  alias_into_sparse(r_, r_indices_, r_values_);

  // Prevent unbounded growth of nnz
  // TODO: Improved heuristic on when to coalesce or remove need to coalesce
  if (r_._nnz() > r_.numel()) {
    auto c = r_.coalesce();
    alias_into_sparse(r_, c._indices(), c._values());
  }
}

} // namespace at::native::xpu
