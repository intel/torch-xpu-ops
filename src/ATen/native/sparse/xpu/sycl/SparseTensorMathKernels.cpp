#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/SparseCsrTensorUtils.h>
#include <ATen/WrapDimUtilsMulti.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/SparseTensorUtils.h>
#include <ATen/native/sparse/SparseTensorMath.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_sparse_coo_tensor_with_dims_and_tensors.h>
#include <ATen/ops/_sparse_sum_native.h>
#include <ATen/ops/add_native.h>
#include <ATen/ops/addmm_native.h>
#include <ATen/ops/bmm_native.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/copy_sparse_to_sparse.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/hspmm_native.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/result_type.h>
#include <ATen/ops/scalar_tensor.h>
#include <ATen/ops/zeros_like.h>
#endif

#include <ATen/native/sparse/xpu/sycl/SparseTensorMathKernels.h>
#include <comm/SYCLContext.h>
#include <comm/TensorInfo.h>

namespace at::native::xpu {

#define I_INFO(tensor) at::xpu::detail::getTensorInfo<int64_t, uint64_t>(tensor)
#define V_INFO(tensor) \
  at::xpu::detail::getTensorInfo<scalar_t, uint64_t>(tensor)

template <typename T>
struct TensorCAddOp {
  TensorCAddOp(T v) : val(v) {}

  inline void operator()(T* out, T* in) const {
    *out += val * *in;
  }

  inline void operator()(T* out, T* in1, T* in2) const {
    *out = *in1 + val * *in2;
  }

  T val;
};

namespace apply {

using at::xpu::detail::TensorInfo;
using indexT = int64_t;

template <typename Op, typename IndexType, typename Real>
struct SparseElementwiseKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    IndexType ind_skip = indices_.strides[0];
    IndexType ind_nnz_skip = indices_.strides[1];
    IndexType value_size = values_.strides[0];

    for (IndexType linearId = (IndexType)item.get_group(0); linearId < nnz_;
         linearId += (IndexType)item.get_group_range(0)) {
      IndexType index = 0;
      for (IndexType d = 0; d < indices_.sizes[0]; d++) {
        index = dense_.sizes[d] * index +
            indices_.data[d * ind_skip + linearId * ind_nnz_skip];
      }
      Real* dst = dense_.data + index * value_size;
      Real* src = values_.data + linearId * value_size;
      for (IndexType linearId2 = (IndexType)item.get_local_id(0);
           linearId2 < value_size;
           linearId2 += (IndexType)item.get_local_range(0)) {
        op_(dst + linearId2, src + linearId2);
      }
    }
  }
  SparseElementwiseKernelFunctor(
      Op op,
      TensorInfo<Real, IndexType> dense_info,
      TensorInfo<indexT, IndexType> indices_info,
      TensorInfo<Real, IndexType> values_info,
      const IndexType nnz_value)
      : op_(op),
        dense_(dense_info),
        indices_(indices_info),
        values_(values_info),
        nnz_(nnz_value) {}

 private:
  Op op_;
  TensorInfo<Real, IndexType> dense_;
  TensorInfo<indexT, IndexType> indices_;
  TensorInfo<Real, IndexType> values_;
  const IndexType nnz_;
};

// template <typename Op, typename IndexType, typename Real>
// void sparseElementwiseKernel(
//     Op op,
//     TensorInfo<Real, IndexType> dense,
//     TensorInfo<indexT, IndexType> indices,
//     TensorInfo<Real, IndexType> values,
//     const IndexType nnz) {
//   using KernelClass = SparseElementwiseKernellFunctor<Op, IndexType, Real>;
//   auto& queue = getCurrentSYCLQueue();
//   IndexType group_size = (IndexType)syclMaxWorkGroupSize<KernelClass>();
//   IndexType target_global_size = (IndexType)syclMaxWorkItemsPerTile();
//   auto max_work_group_num = target_global_size / group_size;

//   auto num_groups = CeilDiv(nnz, group_size);
//   if (num_groups > max_work_group_num)
//     num_groups = max_work_group_num;
//   auto total_items = num_groups * group_size;

//   KernelClass kfn(op, dense, indices, values, nnz);
//   sycl_kernel_submit(total_items, group_size, queue, kfn);
// }

template <typename Op, typename IndexType, typename Real>
struct SparseElementwiseKernelScalarFunctor {
  void operator()(sycl::nd_item<1> item) const {
    IndexType ind_skip = indices_.strides[0];
    IndexType ind_nnz_skip = indices_.strides[1];
    IndexType value_skip = values_.strides[0];

    for (IndexType linearId =
             (IndexType)item.get_group(0) * (IndexType)item.get_local_range(0) +
             (IndexType)item.get_local_id(0);
         linearId < nnz_;
         linearId += (IndexType)item.get_group_range(0) *
             (IndexType)item.get_local_range(0)) {
      IndexType index = 0;
      for (IndexType d = 0; d < indices_.sizes[0]; d++) {
        index = dense_.sizes[d] * index +
            indices_.data[d * ind_skip + linearId * ind_nnz_skip];
      }
      op_(dense_.data + index, values_.data + linearId * value_skip);
    }
  }
  SparseElementwiseKernelScalarFunctor(
      Op op,
      TensorInfo<Real, IndexType> dense_info,
      TensorInfo<indexT, IndexType> indices_info,
      TensorInfo<Real, IndexType> values_info,
      const IndexType nnz_value)
      : op_(op),
        dense_(dense_info),
        indices_(indices_info),
        values_(values_info),
        nnz_(nnz_value) {}

 private:
  Op op_;
  TensorInfo<Real, IndexType> dense_;
  TensorInfo<indexT, IndexType> indices_;
  TensorInfo<Real, IndexType> values_;
  const IndexType nnz_;
};

// template <typename Op, typename IndexType, typename Real>
// void sparseElementwiseKernelScalar(
//     Op op,
//     TensorInfo<Real, IndexType> dense,
//     TensorInfo<indexT, IndexType> indices,
//     TensorInfo<Real, IndexType> values,
//     const IndexType nnz) {
//   using KernelClass = SparseElementwiseKernelScalarFunctor<Op, IndexType,
//   Real>; auto& queue = getCurrentSYCLQueue(); IndexType group_size =
//   (IndexType)syclMaxWorkGroupSize<KernelClass>(); IndexType
//   target_global_size = (IndexType)syclMaxWorkItemsPerTile(); auto
//   max_work_group_num = target_global_size / group_size;

//   auto num_groups = CeilDiv(nnz * group_size, group_size);
//   if (num_groups > max_work_group_num)
//     num_groups = max_work_group_num;
//   auto total_items = num_groups * group_size;

//   KernelClass kfn(op, dense, indices, values, nnz);
//   sycl_kernel_submit(total_items, group_size, queue, kfn);
// }

} // namespace apply

// Check if every tensor in a list of tensors matches the current
// device.
inline bool check_device(ArrayRef<Tensor> ts) {
  if (ts.empty()) {
    return true;
  }
  Device curDevice = Device(kXPU, c10::xpu::current_device());
  for (const Tensor& t : ts) {
    if (t.device() != curDevice)
      return false;
  }
  return true;
}

Tensor& add_out_dense_sparse_kernel(
    Tensor& r_,
    const Tensor& dense,
    const SparseTensor& sparse,
    const at::Scalar& value) {
  TORCH_CHECK(
      dense.is_xpu(),
      "add: expected 'self' to be a XPU tensor, but got a CPU tensor");
  TORCH_CHECK(
      sparse.is_xpu(),
      "add: expected 'other' to be a XPU tensor, but got a CPU tensor");
  TORCH_CHECK(
      r_.is_xpu(),
      "add: expected 'out' to be a XPU tensor, but got a CPU tensor");
  TORCH_CHECK(check_device({sparse, r_, dense}));
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
    return r_;
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
    return r_;
  }

  if (sparse.is_coalesced()) {
    size_t target_global_size = syclMaxWorkItemsPerTile();

    if (sparse.dense_dim() == 0) {
      AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
          at::ScalarType::ComplexHalf,
          at::ScalarType::Bool,
          at::ScalarType::Half,
          at::ScalarType::BFloat16,
          commonDtype,
          "add_out_dense_sparse_xpu",
          [&] {
            auto caller = apply::SparseElementwiseKernelScalarFunctor<
                TensorCAddOp<scalar_t>,
                uint64_t,
                scalar_t>(
                TensorCAddOp<scalar_t>(value.to<scalar_t>()),
                V_INFO(r),
                I_INFO(indices),
                V_INFO(values),
                static_cast<uint64_t>(nnz));
            size_t group_size = syclMaxWorkGroupSize(caller);
            size_t max_work_group_num = target_global_size / group_size;
            size_t num_groups = (nnz + group_size - 1) / group_size;
            if (num_groups > max_work_group_num)
              num_groups = max_work_group_num;
            sycl_kernel_submit(
                num_groups * group_size,
                group_size,
                getCurrentSYCLQueue(),
                caller);
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
            auto caller = apply::SparseElementwiseKernelFunctor<
                TensorCAddOp<scalar_t>,
                uint64_t,
                scalar_t>(
                TensorCAddOp<scalar_t>(value.to<scalar_t>()),
                V_INFO(r),
                I_INFO(indices),
                V_INFO(values),
                static_cast<uint64_t>(nnz));
            size_t group_size = syclMaxWorkGroupSize(caller);
            size_t max_work_group_num = target_global_size / group_size;
            size_t num_groups = (nnz + group_size - 1) / group_size;
            if (num_groups > max_work_group_num)
              num_groups = max_work_group_num;
            sycl_kernel_submit(
                num_groups * group_size,
                group_size,
                getCurrentSYCLQueue(),
                caller);
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
  return r_;
}

SparseTensor& add_sparse_kernel(
    const SparseTensor& t,
    const SparseTensor& src,
    const Scalar& value,
    SparseTensor& r_) {
  if (!t.is_sparse()) {
    return add_out_dense_sparse_kernel(r_, t, src, value);
  }

  // TODO: This test seems a bit goofy
  TORCH_CHECK(
      src.is_sparse(),
      "add(sparse, dense) is not supported. Use add(dense, sparse) instead.");

  TORCH_CHECK(t.is_xpu(), "add: expected 'self' to be XPU, but got CPU");
  TORCH_CHECK(src.is_xpu(), "add: expected 'other' to be XPU, but got CPU");
  TORCH_CHECK(r_.is_xpu(), "add: expected 'out' to be XPU, but got CPU");

  TORCH_CHECK(check_device({r_, t, src}));

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
    return copy_sparse_to_sparse_(r_, t);
  }
  if (t._nnz() == 0) {
    return mul_out_sparse_scalar(r_, src, value);
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
      ScalarType::Half,
      ScalarType::BFloat16,
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

  return r_;
}

SparseTensor& mul_sparse_kernel(
    const Tensor& t_,
    const Tensor& src_,
    SparseTensor& r_) {
  TORCH_CHECK(r_.is_xpu(), "mul: expected 'out' to be XPU, but got CPU");

  // case mul(sparse, dense)
  if (!src_.is_sparse()) {
    return _mul_dense_sparse_out(src_, t_, r_);
  }
  // case mul(dense, sparse)
  if (!t_.is_sparse()) {
    return _mul_dense_sparse_out(t_, src_, r_);
  }

  // case mul(sparse, sparse) with a 0-dim input.
  if (!src_.dim()) {
    return _mul_sparse_sparse_zero_dim_out(src_, t_, r_);
  }
  if (!t_.dim()) {
    return _mul_sparse_sparse_zero_dim_out(t_, src_, r_);
  }

  TORCH_CHECK(t_.is_xpu(), "mul: expected 'self' to be XPU, but got CPU");
  TORCH_CHECK(src_.is_xpu(), "mul: expected 'other' to be XPU, but got CPU");
  TORCH_CHECK(check_device({r_, t_, src_}));

  // mul(sparse, sparse)

  // Short circuit when there is zero nnz.
  // Not strictly necessary, but there are tests checking whether
  // resize in mul fails if run on tensors coming from .data/.detach.
  if (t_.sizes().equals(src_.sizes()) && (!t_._nnz() || !src_._nnz())) {
    r_.resize_as_(t_);
    return r_.zero_();
  }
  return _mul_sparse_sparse_out(t_, src_, r_);
}

} // namespace at::native::xpu
