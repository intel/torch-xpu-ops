#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/SparseCsrTensorUtils.h>
#include <ATen/WrapDimUtilsMulti.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/SparseTensorUtils.h>
#include <ATen/native/sparse/SparseTensorMath.h>
#include <ATen/native/xpu/sycl/pstl/PSTLFunctions.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_sparse_coo_tensor_with_dims_and_tensors.h>
#include <ATen/ops/_sparse_sum.h>
#include <ATen/ops/_sparse_sum_backward_native.h>
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

using at::xpu::detail::getTensorInfo;
using at::xpu::detail::TensorInfo;

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

template <typename scalar_t>
struct SparseSumBackwardFunctor {
  void operator()(sycl::nd_item<1> item) const {
    const int64_t i =
        item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
    if (i >= total_threads)
      return;
    const int64_t j = input_indices_pos_ti.data[i];

    bool has_match = false;
    if (grad_indices_ti.data[j] == input_indices_ti.data[i]) {
      has_match = true;
    }

    int64_t grad_input_values_stride0 = grad_input_values_ti.strides[0];
    int64_t out_start = i * grad_input_values_stride0;
    int64_t out_end = (i + 1) * grad_input_values_stride0;
    int64_t in_start = j * grad_values_expand_ti.strides[0];

    if (has_match) {
      for (int64_t out_i = out_start, in_i = in_start; out_i < out_end;
           out_i++, in_i++) {
        grad_input_values_ti.data[out_i] = grad_values_expand_ti.data[in_i];
      }
    } else {
      for (int64_t out_i = out_start; out_i < out_end; out_i++) {
        grad_input_values_ti.data[out_i] = scalar_t(0);
      }
    }
  }

  SparseSumBackwardFunctor(
      int64_t total_threads,
      const TensorInfo<int64_t, int64_t> grad_indices_ti,
      const TensorInfo<int64_t, int64_t> input_indices_ti,
      const TensorInfo<int64_t, int64_t> input_indices_pos_ti,
      const TensorInfo<scalar_t, int64_t> grad_values_expand_ti,
      TensorInfo<scalar_t, int64_t> grad_input_values_ti)
      : total_threads(total_threads),
        grad_indices_ti(grad_indices_ti),
        input_indices_ti(input_indices_ti),
        input_indices_pos_ti(input_indices_pos_ti),
        grad_values_expand_ti(grad_values_expand_ti),
        grad_input_values_ti(grad_input_values_ti) {}

 private:
  int64_t total_threads;
  const TensorInfo<int64_t, int64_t> grad_indices_ti;
  const TensorInfo<int64_t, int64_t> input_indices_ti;
  const TensorInfo<int64_t, int64_t> input_indices_pos_ti;
  const TensorInfo<scalar_t, int64_t> grad_values_expand_ti;
  TensorInfo<scalar_t, int64_t> grad_input_values_ti;
};

Tensor _sparse_sum_backward_kernel(
    const Tensor& grad_,
    const SparseTensor& input_,
    IntArrayRef dims_to_sum) {
  TORCH_CHECK(
      grad_.is_xpu(),
      "_sparse_sum_backward_xpu: expected 'grad_' to be XPU tensor, but got CPU tensor");
  TORCH_CHECK(
      input_.is_xpu(),
      "_sparse_sum_backward_xpu: expected 'input_' to be XPU tensor, but got CPU tensor");

  // Short circuit if grad is either zero or empty
  if (((grad_.is_sparse() || at::sparse_csr::is_sparse_compressed(grad_)) &&
       !grad_._nnz()) ||
      !grad_.numel()) {
    return at::zeros_like(input_);
  }

  auto input = input_.coalesce();
  const int64_t input_dim = input.dim();
  auto dims_to_sum_b = dim_list_to_bitset(dims_to_sum, input_dim);
  auto dims_to_sum_v = dims_to_sum.vec();
  maybe_wrap_dims(dims_to_sum_v, input_dim);

  Tensor input_indices = input._indices();
  Tensor input_values = input._values();
  IntArrayRef input_sizes = input.sizes();
  const int64_t input_sparse_dim = input.sparse_dim();
  const int64_t input_dense_dim = input.dense_dim();
  const int64_t input_nnz = input._nnz();

  int64_t sparse_dims_to_sum_size = 0;
  auto sparse_dims_to_keep_v = std::vector<int64_t>();
  auto dense_dims_to_sum_v = std::vector<int64_t>();
  for (int64_t d = 0; d < input_dim; d++) {
    if (dims_to_sum_b[d]) {
      if (d < input_sparse_dim)
        sparse_dims_to_sum_size++;
      else
        dense_dims_to_sum_v.emplace_back(d + 1 - input_sparse_dim);
    } else {
      if (d < input_sparse_dim)
        sparse_dims_to_keep_v.emplace_back(d);
    }
  }

  const bool sum_all_sparse_dim = (input_sparse_dim == sparse_dims_to_sum_size);
  const bool sum_dense_dim = (dense_dims_to_sum_v.size() > 0);
  const bool sum_sparse_dim = (sparse_dims_to_sum_size > 0);

  if (sum_all_sparse_dim) {
    TORCH_CHECK(
        !grad_.is_sparse(),
        "_sparse_sum_backward_xpu: expected grad Tensor to be dense since all sparse dims are summed");
    auto grad_input_values = grad_;
    auto expand_size = input_values.sizes().vec();
    if (sum_dense_dim) {
      auto dense_expand_size = std::vector<int64_t>(expand_size);
      dense_expand_size.erase(dense_expand_size.begin()); // remove nnz dim
      for (auto d : dense_dims_to_sum_v)
        grad_input_values =
            grad_input_values.unsqueeze(d - 1); // -1 since grad has no nnz dim
      grad_input_values = grad_input_values.expand(dense_expand_size);
    }
    grad_input_values = grad_input_values.expand(expand_size)
                            .clone(at::MemoryFormat::Contiguous);
    return at::_sparse_coo_tensor_with_dims_and_tensors(
        input_sparse_dim,
        input_dense_dim,
        input_sizes,
        input_indices.clone(at::MemoryFormat::Contiguous),
        grad_input_values,
        input.options().dtype(grad_.dtype())); // convert to grad dtype
  } else {
    TORCH_CHECK(
        grad_.is_sparse(),
        "_sparse_sum_backward_xpu: expected grad_ Tensor to be sparse, but got dense");
    auto grad = grad_.coalesce();
    Tensor grad_indices = grad._indices();
    Tensor grad_values = grad._values();
    const int64_t grad_sparse_dim = grad.sparse_dim();
    const int64_t grad_nnz = grad._nnz();

    Tensor grad_values_expand = grad_values;
    if (sum_dense_dim) {
      auto expand_size = input_values.sizes().vec();
      if (sum_sparse_dim)
        expand_size[0] = grad_values.size(0); // update nnz
      for (auto d : dense_dims_to_sum_v)
        grad_values_expand = grad_values_expand.unsqueeze(d);
      grad_values_expand = grad_values_expand.expand(expand_size)
                               .clone(at::MemoryFormat::Contiguous);
    }

    Tensor grad_input_values;
    if (!sum_sparse_dim) {
      grad_input_values = grad_values_expand;
    } else {
      grad_input_values = at::empty_like(
          input_values, grad_values.options(), LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      AT_ASSERT(grad_input_values.is_xpu());

      // get 1D indices
      auto grad_sparse_dim_to_keep_v = std::vector<int64_t>(grad_sparse_dim);
      std::iota(
          grad_sparse_dim_to_keep_v.begin(),
          grad_sparse_dim_to_keep_v.end(),
          0);

      auto grad_indices_1D = flatten_indices_by_dims(
          grad_indices,
          grad.sizes(),
          grad_sparse_dim_to_keep_v); // flatten indices on all sparse_dim
                                      // of grad, output indices is
                                      // coalesced and sorted
      auto input_indices_1D = flatten_indices_by_dims(
          input_indices, input_sizes, sparse_dims_to_keep_v);

      // store lower_bound of input indices at grad indices
      Tensor input_indices_pos =
          at::empty_like(input_indices_1D, LEGACY_CONTIGUOUS_MEMORY_FORMAT);

      pstl::lower_bound_tensor<int64_t>(
          grad_indices_1D.data_ptr<int64_t>(),
          grad_indices_1D.data_ptr<int64_t>() + grad_nnz,
          input_indices_1D.data_ptr<int64_t>(),
          input_indices_1D.data_ptr<int64_t>() + input_nnz,
          input_indices_pos.data_ptr<int64_t>());

      auto grad_indices_ti = getTensorInfo<int64_t, int64_t>(grad_indices_1D);
      auto input_indices_ti = getTensorInfo<int64_t, int64_t>(input_indices_1D);
      auto input_indices_pos_ti =
          getTensorInfo<int64_t, int64_t>(input_indices_pos);

      AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(
          kHalf, grad_values.scalar_type(), "_sparse_sum_backward_xpu", [&] {
            auto grad_values_expand_ti =
                getTensorInfo<scalar_t, int64_t>(grad_values_expand);
            auto grad_input_values_ti =
                getTensorInfo<scalar_t, int64_t>(grad_input_values);

            auto kfn = SparseSumBackwardFunctor<scalar_t>(
                input_nnz,
                grad_indices_ti,
                input_indices_ti,
                input_indices_pos_ti,
                grad_values_expand_ti,
                grad_input_values_ti);

            size_t target_global_size = syclMaxWorkItemsPerTile();
            size_t group_size = std::min(input_nnz, syclMaxWorkGroupSize(kfn));
            size_t max_work_group_num = target_global_size / group_size;
            size_t num_groups = (input_nnz + group_size - 1) / group_size;
            if (num_groups > max_work_group_num)
              num_groups = max_work_group_num;
            sycl_kernel_submit(
                num_groups * group_size,
                group_size,
                getCurrentSYCLQueue(),
                kfn);
          });
    }

    return at::_sparse_coo_tensor_with_dims_and_tensors(
        input_sparse_dim,
        input_dense_dim,
        input_sizes,
        input_indices.clone(at::MemoryFormat::Contiguous),
        grad_input_values,
        grad.options());
  }
}

} // namespace at::native::xpu
