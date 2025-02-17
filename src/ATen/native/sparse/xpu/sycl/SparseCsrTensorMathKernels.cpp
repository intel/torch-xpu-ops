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
#include <ATen/native/xpu/sycl/pstl/PSTLFunctions.h>
#include <comm/SYCLContext.h>

namespace at::native::xpu {

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
