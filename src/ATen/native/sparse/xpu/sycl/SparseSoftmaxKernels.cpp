#include <cstdint>
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/WrapDimUtilsMulti.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/SparseTensorUtils.h>
#include <ATen/native/sparse/ParamUtils.h>
#include <ATen/native/sparse/SparseTensorMath.h>

#include <ATen/native/xpu/sycl/Loops.h>
#include <ATen/native/xpu/sycl/pstl/PSTLFunctions.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_log_softmax.h>
#include <ATen/ops/_log_softmax_backward_data.h>
#include <ATen/ops/_log_softmax_backward_data_native.h>
#include <ATen/ops/_log_softmax_native.h>
#include <ATen/ops/_softmax.h>
#include <ATen/ops/_softmax_backward_data.h>
#include <ATen/ops/_softmax_backward_data_native.h>
#include <ATen/ops/_softmax_native.h>
#include <ATen/ops/_sparse_log_softmax.h>
#include <ATen/ops/_sparse_log_softmax_backward_data.h>
#include <ATen/ops/_sparse_log_softmax_backward_data_native.h>
#include <ATen/ops/_sparse_log_softmax_native.h>
#include <ATen/ops/_sparse_softmax.h>
#include <ATen/ops/_sparse_softmax_backward_data.h>
#include <ATen/ops/_sparse_softmax_backward_data_native.h>
#include <ATen/ops/_sparse_softmax_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/equal.h>
#include <ATen/ops/equal_native.h>
#include <ATen/ops/full.h>
#include <ATen/ops/log_softmax.h>
#include <ATen/ops/log_softmax_native.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/ones_like.h>
#include <ATen/ops/softmax.h>
#include <ATen/ops/softmax_native.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like.h>
#endif

#include <bitset>

#include <c10/macros/Macros.h>
#include <comm/Memory.h>

#include <ATen/native/sparse/xpu/sycl/SparseSoftmaxKernels.h>
#include <ATen/native/xpu/sycl/Loops.h>
#include <comm/SYCLContext.h>
#include <comm/TensorInfo.h>

namespace at::native::xpu {

template <typename T, class InputIt1, class InputIt2, class OutputIt>
struct MaxRowKernelFunctor {
  void operator()(sycl::item<1> item_id) const {
    int64_t curr_pool_size = pool_sizes_ptr[item_id];
    auto mx_row = mx_buffer_ptr + static_cast<int64_t>(item_id * nvalues);
    int64_t offset = pool_offsets_ptr[item_id];
    for (int64_t p = 0; p < curr_pool_size; p++) {
      int64_t i = *(sorted_indices_ptr + offset + p);
      auto values_row = values_accessor[i].data();
      for (int64_t j = 0; j < nvalues; j++) {
        mx_row[j] = std::max(mx_row[j], values_row[j]);
      }
    }
  }

  MaxRowKernelFunctor(
      InputIt1 pool_sizes_ptr,
      InputIt2 values_accessor,
      InputIt1 sorted_indices_ptr,
      InputIt1 pool_offsets_ptr,
      OutputIt mx_buffer_ptr,
      T nvalues)
      : pool_sizes_ptr(pool_sizes_ptr),
        values_accessor(values_accessor),
        sorted_indices_ptr(sorted_indices_ptr),
        pool_offsets_ptr(pool_offsets_ptr),
        mx_buffer_ptr(mx_buffer_ptr),
        nvalues(nvalues) {}

 private:
  InputIt1 pool_sizes_ptr;
  InputIt2 values_accessor;
  InputIt1 sorted_indices_ptr;
  InputIt1 pool_offsets_ptr;
  OutputIt mx_buffer_ptr;
  T nvalues;
};

template <typename T, class InputIt1, class InputIt2, class OutputIt>
OutputIt max_row(
    InputIt1 pool_sizes_first,
    InputIt1 pool_sizes_last,
    InputIt2 values_accessor,
    InputIt1 sorted_indices_ptr,
    InputIt1 pool_offsets_ptr,
    OutputIt mx_buffer_ptr,
    T nvalues) {
  RECORD_FUNCTION("max_row_xpu", {});
  const auto N = std::distance(pool_sizes_first, pool_sizes_last);
  auto& q = getCurrentSYCLQueue();

  MaxRowKernelFunctor<T, InputIt1, InputIt2, OutputIt> mfn(
      pool_sizes_first,
      values_accessor,
      sorted_indices_ptr,
      pool_offsets_ptr,
      mx_buffer_ptr,
      nvalues);
  sycl_kernel_submit(sycl::range<1>(N), q, mfn);

  return mx_buffer_ptr;
}

// Number of threads in a block given an input size up to MAX_BLOCK_SIZE
static int getNumThreads(int nElem) {
  int threadSizes[5] = {32, 64, 128, 256, 512};
  for (int i = 0; i != 5; ++i) {
    if (nElem <= threadSizes[i]) {
      return threadSizes[i];
    }
  }
  return threadSizes[4];
}

int64_t get_nvalues(const IntArrayRef& sizes, int64_t sparse_dim) {
  /* Return the number of entries in the dense part of a sparse tensor.
     `sizes` is a vector of sparse tensor dimensions.
     `sparse_dim` is the dimension of the sparse part of a sparse tensor.
   */
  return c10::multiply_integers(sizes.begin() + sparse_dim, sizes.end());
}

template <typename T>
struct PoolPred {
  bool operator()(const T& x, const T& y) const {
    return offsets_ptr[x] < offsets_ptr[y];
  }
  PoolPred(T* offsets_ptr) : offsets_ptr(offsets_ptr) {}

 private:
  T* offsets_ptr;
};

template <typename index_t>
struct SortFunctor {
  auto operator()(index_t a, index_t b) const {
    return (a < b);
  }
};

template <typename T>
struct ReducePred {
  bool operator()(const T& x, const T& y) const {
    return offsets_ptr[x] == offsets_ptr[y];
  }
  ReducePred(T* offsets_ptr) : offsets_ptr(offsets_ptr) {}

 private:
  T* offsets_ptr;
};

template <typename scalar_t, bool LogSoftMax>
struct SparseCooSoftmaxFunctor {
  /*
    See ATen/native/sparse/SoftMax.cpp:cpu_sparse_coo_softmax for the CPU
    implementation of the sparse softmax algorithm that this implementation is
    based on.
  */
  void operator()(sycl::nd_item<1> item) const {
    int tid = item.get_local_id(0);
    int blkid = item.get_group(0);
    int blksz = item.get_local_range(0);
    int gridsz = item.get_group_range(0);

    int index = tid + blkid * blksz;
    int step = blksz * gridsz;

    while (index < pool_size) {
      int64_t offset = pool_offsets[index];
      int64_t* pool_indices = sorted_pool_indices + offset;
      int64_t pool_indices_size = pool_sizes[index];
      scalar_t* mx_row = mx_rows + index * nvalues;

      for (int64_t j = 0; j < nvalues; j++) {
        scalar_t exp_sums = 0;
        for (int64_t p = 0; p < pool_indices_size; p++) {
          auto i = pool_indices[p];
          auto values_row = input_values_acc[i];
          auto out_values_row = output_values_acc[i];

          auto v = std::exp(values_row[j] - mx_row[j]);
          if (!LogSoftMax) {
            out_values_row[j] = v;
          }
          exp_sums += v;
        }
        for (int64_t p = 0; p < pool_indices_size; p++) {
          auto i = pool_indices[p];
          auto values_row = input_values_acc[i];
          auto out_values_row = output_values_acc[i];

          if (LogSoftMax) {
            out_values_row[j] = values_row[j] - mx_row[j] - std::log(exp_sums);
          } else {
            out_values_row[j] *= 1.0 / exp_sums;
          }
        }
      }
      index += step;
    }
  }

  SparseCooSoftmaxFunctor(
      int64_t* sorted_pool_indices,
      int64_t pool_size,
      int64_t* pool_sizes,
      int64_t* pool_offsets,
      int64_t nvalues,
      scalar_t* mx_rows,
      GenericPackedTensorAccessor<scalar_t, 2> input_values_acc,
      GenericPackedTensorAccessor<scalar_t, 2> output_values_acc)
      : sorted_pool_indices(sorted_pool_indices),
        pool_size(pool_size),
        pool_sizes(pool_sizes),
        pool_offsets(pool_offsets),
        nvalues(nvalues),
        mx_rows(mx_rows),
        input_values_acc(input_values_acc),
        output_values_acc(output_values_acc) {}

 private:
  int64_t* sorted_pool_indices;
  int64_t pool_size;
  int64_t* pool_sizes;
  int64_t* pool_offsets;
  int64_t nvalues;
  scalar_t* mx_rows;
  GenericPackedTensorAccessor<scalar_t, 2> input_values_acc;
  GenericPackedTensorAccessor<scalar_t, 2> output_values_acc;
};

template <typename scalar_t, bool LogSoftMax>
struct SparseCooSoftmaxbBackwardFunctor {
  /*
    See ATen/native/sparse/SoftMax.cpp:cpu_sparse_coo_softmax_backward for
    the CPU implementation of the sparse softmax backward algorithm that this
    implementation is based on.
  */
  void operator()(sycl::nd_item<1> item) const {
    int tid = item.get_local_id(0);
    int blkid = item.get_group(0);
    int blksz = item.get_local_range(0);
    int gridsz = item.get_group_range(0);

    int index = tid + blkid * blksz;
    int step = blksz * gridsz;

    while (index < size) {
      int64_t offset = pool_offsets[index];
      int64_t* pool_indices = sorted_pool_indices + offset;
      int64_t pool_indices_size = pool_sizes[index];

      for (int64_t k = 0; k < nvalues; k++) {
        scalar_t tmp_row{0};

        /* Compute tmp = - sum_j output_j * grad_j */
        for (int64_t p = 0; p < pool_indices_size; p++) {
          auto i = pool_indices[p];
          auto out_values_row = out_values_accessor[i];
          auto j = lower_bound_values[i];

          /* Update `tmp_row` accumulator only when limits and pools are valid
           */
          if (j < grad_nnz && (out_offsets[i] == grad_offsets[j])) {
            auto grad_values_row = grad_values_accessor[j];
            if (LogSoftMax) {
              tmp_row -= grad_values_row[k];
            } else {
              tmp_row -= out_values_row[k] * grad_values_row[k];
            }
          }
        }

        /* Compute grad_input = output * (grad + tmp)*/
        for (int64_t p = 0; p < pool_indices_size; p++) {
          auto i = pool_indices[p];
          auto out_values_row = out_values_accessor[i];
          auto values_row = values_accessor[i];
          auto j = lower_bound_values[i];
          if (j < grad_nnz && (out_offsets[i] == grad_offsets[j])) {
            auto grad_values_row = grad_values_accessor[j];
            if (LogSoftMax) {
              values_row[k] =
                  grad_values_row[k] + std::exp(out_values_row[k]) * tmp_row;
            } else {
              values_row[k] =
                  out_values_row[k] * (grad_values_row[k] + tmp_row);
            }
          } else {
            if (LogSoftMax) {
              values_row[k] = std::exp(out_values_row[k]) * tmp_row;
            } else {
              values_row[k] = out_values_row[k] * tmp_row;
            }
          }
        }
      }
      index += step;
    }
  }

  SparseCooSoftmaxbBackwardFunctor(
      int64_t* sorted_pool_indices,
      int64_t size,
      int64_t* pool_sizes,
      int64_t* pool_offsets,
      int64_t nvalues,
      int64_t grad_nnz,
      int64_t* grad_offsets,
      int64_t* out_offsets,
      int64_t* lower_bound_values,
      GenericPackedTensorAccessor<scalar_t, 2> values_accessor,
      GenericPackedTensorAccessor<scalar_t, 2> out_values_accessor,
      GenericPackedTensorAccessor<scalar_t, 2> grad_values_accessor)
      : sorted_pool_indices(sorted_pool_indices),
        size(size),
        pool_sizes(pool_sizes),
        pool_offsets(pool_offsets),
        nvalues(nvalues),
        grad_nnz(grad_nnz),
        grad_offsets(grad_offsets),
        out_offsets(out_offsets),
        lower_bound_values(lower_bound_values),
        values_accessor(values_accessor),
        out_values_accessor(out_values_accessor),
        grad_values_accessor(grad_values_accessor) {}

 private:
  int64_t* sorted_pool_indices;
  int64_t size;
  int64_t* pool_sizes;
  int64_t* pool_offsets;
  int64_t nvalues;
  int64_t grad_nnz;
  int64_t* grad_offsets;
  int64_t* out_offsets;
  int64_t* lower_bound_values;
  GenericPackedTensorAccessor<scalar_t, 2> values_accessor;
  GenericPackedTensorAccessor<scalar_t, 2> out_values_accessor;
  GenericPackedTensorAccessor<scalar_t, 2> grad_values_accessor;
};

Tensor get_offsets(
    const Tensor& indices,
    const IntArrayRef& sizes,
    const int64_t dim) {
  /*
    See ATen/native/sparse/SoftMax.cpp:get_offsets for the CPU
    implementation of get_offsets function that this implementation is based
    on.
  */

  auto ndim = indices.size(0);
  auto nnz = indices.size(1);
  std::vector<int64_t> host_strides(ndim, 1);
  if (ndim > 1) {
    for (int64_t i = ndim - 2; i >= 0; i--) {
      host_strides[i] = host_strides[i + 1] * (i + 1 == dim ? 1 : sizes[i + 1]);
    }
  }
  // auto strides = host_strides;
  auto strides = at::empty({ndim}, indices.options());
  auto strides_ptr = strides.data_ptr<int64_t>();

  // syclMemcpyAsync(
  //     strides_ptr,
  //     host_strides.data(),
  //     host_strides.size() * sizeof(int64_t),
  //     HostToDevice);

  for (int kk = 0; kk < ndim; kk++) {
    strides[kk] = host_strides[kk];
  }

  auto indices_accessor = indices.packed_accessor64<int64_t, 2>();
  Tensor offsets = at::ones({nnz}, indices.options());

  for (int i = 0; i < nnz; i++) {
    int64_t pool_index = 0;
    for (int64_t j = 0; j < ndim; j++) {
      if (j != dim) {
        offsets[i] += (strides[j] * indices[j][i]);
      }
    }
  }
  return offsets;
}

template <class scalar_t, bool requireMxRows = true>
std::tuple<Tensor, Tensor, Tensor, Tensor> compute_pool_max(
    const Tensor& indices,
    const Tensor& values,
    const IntArrayRef& sizes,
    int64_t nvalues,
    const int64_t dim) {
  /*
    Return pools of indices that align with the given dimension and the
    corresponding max values for each pool.

    See ATen/native/sparse/SoftMax.cpp:get_offsets and
    ATen/native/sparse/SoftMax.cpp:cpu_sparse_coo_softmax for the CPU
    implementation that this implementation is based on.
  */
  auto nnz = indices.size(1);

  auto offsets = get_offsets(indices, sizes, dim);
  int64_t* offsets_ptr = offsets.data_ptr<int64_t>();
  auto offsets_sort = get_offsets(indices, sizes, dim);
  int64_t* offsets_sort_ptr = offsets_sort.data_ptr<int64_t>();

  auto sorted_indices = at::empty({nnz}, indices.options());
  auto sorted_indices_ptr = sorted_indices.data_ptr<int64_t>();
  pstl::iota<int64_t>(sorted_indices_ptr, sorted_indices_ptr + nnz, (int64_t)0);

  SortFunctor<int64_t> sfn;
  pstl::sort<int64_t, int64_t>(offsets_sort_ptr, sorted_indices_ptr, nnz, sfn);

  auto pool_sizes = at::ones({nnz}, indices.options());
  auto constant_it = at::ones({nnz}, indices.options());
  auto discard_it = at::zeros({nnz}, indices.options());
  // sorted_indices_ptr = sorted_indices.data_ptr<int64_t>();

  auto new_end = pstl::reduce_by_key<int64_t>(
      sorted_indices_ptr,
      sorted_indices_ptr + nnz,
      constant_it.data_ptr<int64_t>(),
      discard_it.data_ptr<int64_t>(),
      pool_sizes.data_ptr<int64_t>(),
      ReducePred<int64_t>(offsets_ptr));
  auto new_sz = std::distance(pool_sizes.data_ptr<int64_t>(), new_end);

  pool_sizes.resize_({new_sz});

  auto pool_offsets = pool_sizes.clone();
  auto pool_offsets_ptr = pool_offsets.data_ptr<int64_t>();
  pstl::exclusive_scan(
      pool_offsets_ptr,
      pool_offsets_ptr + new_sz,
      pool_offsets_ptr,
      static_cast<int64_t>(0));

  Tensor mx_buffer;
  if (requireMxRows) {
    auto values_accessor =
        values.packed_accessor64<scalar_t, 2>(); // {nnz, nvalues}

    mx_buffer = at::full(
        {new_sz * nvalues},
        Scalar(-std::numeric_limits<scalar_t>::infinity()),
        values.options());
    auto mx_buffer_ptr = mx_buffer.data_ptr<scalar_t>();

    auto pool_sizes_ptr = pool_sizes.data_ptr<int64_t>();
    auto sorted_indices_ptr = sorted_indices.data_ptr<int64_t>();
    auto pool_offsets_ptr = pool_offsets.data_ptr<int64_t>();

    max_row<scalar_t>(
        pool_sizes_ptr,
        pool_sizes_ptr + new_sz,
        values_accessor,
        sorted_indices_ptr,
        pool_offsets_ptr,
        mx_buffer_ptr,
        nvalues);
  }

  return std::make_tuple(sorted_indices, pool_offsets, pool_sizes, mx_buffer);
}

template <typename scalar_t, bool LogSoftMax>
void xpu_sparse_coo_softmax(
    Tensor& output,
    const Tensor& input,
    const int64_t dim) {
  /*
    See ATen/native/sparse/SoftMax.cpp:cpu_sparse_coo_softmax for the CPU
    implementation of the sparse softmax algorithm that this implementation is
    based on.
  */
  auto sparse_dim = input.sparse_dim();
  auto indices = input._indices().contiguous();
  auto values = input._values().contiguous();
  auto out_values = output._values();
  auto out_indices = output._indices();
  out_values.resize_as_(values);
  out_indices.resize_as_(indices);
  out_indices.copy_(indices);

  if (dim >= sparse_dim) {
    if (LogSoftMax) {
      auto new_values = _log_softmax(values, dim - sparse_dim + 1, false);
      out_values.set_(new_values);
    } else {
      auto new_values = _softmax(values, dim - sparse_dim + 1, false);
      out_values.set_(new_values);
    }
    return;
  }

  auto nnz = values.size(0);
  auto sizes = input.sizes();
  auto nvalues = get_nvalues(sizes, sparse_dim);

  /* Prepare accessors */
  auto values_2 = values.view({nnz, nvalues});
  auto values_accessor = values_2.packed_accessor64<scalar_t, 2>();

  auto out_values_2 = out_values.view({nnz, nvalues});
  auto out_values_accessor = out_values_2.packed_accessor64<scalar_t, 2>();

  auto [sorted_indices, pool_offsets, pool_sizes, mx_buffer] =
      compute_pool_max<scalar_t, true>(indices, values_2, sizes, nvalues, dim);

  auto pool_size = pool_offsets.size(0);
  int block_size = getNumThreads(pool_size);
  const int grid_size = (pool_size + block_size - 1) / block_size;

  sycl::range<1> global_range(grid_size * block_size);
  sycl::range<1> local_range(block_size);

  // If either nvalues or pool_size are zero, then
  // Sparse_coo_softmax_kernel won't actually perform any computation.
  // Further, they will be invalid configuration parameters for the launch. So
  // let's not launch a kernel unless both are non-zero.
  if (nvalues > 0 && pool_size > 0) {
    auto kfn = SparseCooSoftmaxFunctor<scalar_t, LogSoftMax>(
        sorted_indices.template data_ptr<int64_t>(),
        pool_size,
        pool_sizes.template data_ptr<int64_t>(),
        pool_offsets.template data_ptr<int64_t>(),
        nvalues,
        mx_buffer.template data_ptr<scalar_t>(),
        values_accessor,
        out_values_accessor);
    sycl_kernel_submit(global_range, local_range, getCurrentSYCLQueue(), kfn);
  }
}

template <typename scalar_t, bool LogSoftMax>
void xpu_sparse_coo_softmax_backward(
    Tensor& grad_input,
    const Tensor& grad,
    const Tensor& output,
    const int64_t dim,
    ScalarType input_dtype) {
  /*
    See ATen/native/sparse/SoftMax.cpp:cpu_sparse_coo_softmax_backward for
    the CPU implementation of the sparse softmax backward algorithm that this
    implementation is based on.
  */
  auto sparse_dim = output.sparse_dim();
  auto sizes = output.sizes().vec();
  auto grad_indices = grad._indices().contiguous();
  auto grad_values = grad._values().contiguous();
  auto out_indices = output._indices().contiguous();
  auto out_values = output._values().contiguous();
  auto values = grad_input._values();
  auto indices = grad_input._indices();
  auto out_nnz = out_values.size(0);
  auto grad_nnz = grad_values.size(0);

  values.resize_as_(out_values);
  values.zero_();
  indices.resize_as_(out_indices);
  indices.copy_(out_indices);

  auto out_offsets = get_offsets(out_indices, sizes, -1);
  auto grad_offsets = get_offsets(grad_indices, sizes, -1);

  /* when dim >= sparse_dim the dense backward is used */
  if (dim >= sparse_dim) {
    if (at::equal(out_offsets, grad_offsets) == true) {
      if (LogSoftMax) {
        auto r = at::_log_softmax_backward_data(
            grad_values, out_values, dim - sparse_dim + 1, input_dtype);
        values.set_(r);
      } else {
        auto r = at::_softmax_backward_data(
            grad_values, out_values, dim - sparse_dim + 1, input_dtype);
        values.set_(r);
      }
    } else {
      auto host_out_offsets =
          out_offsets.to(at::Device(kCPU), indices.dtype(), false, true);
      auto host_grad_offsets =
          grad_offsets.to(at::Device(kCPU), indices.dtype(), false, true);
      auto out_offsets_accessor = host_out_offsets.data_ptr<int64_t>();
      auto grad_offsets_accessor = host_grad_offsets.data_ptr<int64_t>();

      for (int64_t i = 0; i < out_nnz; i++) {
        auto low = std::lower_bound(
            grad_offsets_accessor,
            grad_offsets_accessor + grad_offsets.size(0),
            out_offsets_accessor[i]);
        auto j = low - grad_offsets_accessor;

        /*
          Compute output using dense backward only when limits and pools are
          valid If this check is false then a sparse tensor with full of zeros
          is returned
        */
        if (j < grad_nnz &&
            out_offsets_accessor[i] == grad_offsets_accessor[j]) {
          if (LogSoftMax) {
            auto r = at::_log_softmax_backward_data(
                grad_values[j], out_values[i], dim - sparse_dim, input_dtype);
            values[i].copy_(r);
          } else {
            auto r = at::_softmax_backward_data(
                grad_values[j], out_values[i], dim - sparse_dim, input_dtype);
            values[i].copy_(r);
          }
        }
      }
    }
    return;
  }

  auto nnz = values.size(0);
  auto nvalues = get_nvalues(sizes, sparse_dim);

  auto values_2 = values.view({nnz, nvalues});
  auto values_accessor = values_2.packed_accessor64<scalar_t, 2>();

  auto out_values_2 = out_values.view({out_nnz, nvalues});
  auto out_values_accessor = out_values_2.packed_accessor64<scalar_t, 2>();

  auto grad_values_2 = grad_values.view({grad_nnz, nvalues});
  auto grad_values_accessor = grad_values_2.packed_accessor64<scalar_t, 2>();

  Tensor lower_bound_values =
      at::empty({out_offsets.size(0)}, indices.options());

  pstl::lower_bound_tensor<int64_t>(
      grad_offsets.data_ptr<int64_t>(),
      grad_offsets.data_ptr<int64_t>() + grad_offsets.size(0),
      out_offsets.data_ptr<int64_t>(),
      out_offsets.data_ptr<int64_t>() + out_offsets.size(0),
      lower_bound_values.data_ptr<int64_t>());

  /* Compute independent pools of indices */
  auto [sorted_indices, pool_offsets, pool_sizes, _] =
      compute_pool_max<scalar_t, false>(
          out_indices, values_2, sizes, nvalues, dim);

  auto pool_size = pool_offsets.size(0);

  int block_size = getNumThreads(pool_size);
  const int grid_size = (pool_size + block_size - 1) / block_size;

  sycl::range<1> global_range(grid_size * block_size);
  sycl::range<1> local_range(block_size);

  if (nvalues > 0 && pool_size > 0) {
    auto kfn = SparseCooSoftmaxbBackwardFunctor<scalar_t, LogSoftMax>(
        sorted_indices.template data_ptr<int64_t>(),
        pool_size,
        pool_sizes.template data_ptr<int64_t>(),
        pool_offsets.template data_ptr<int64_t>(),
        nvalues,
        grad_nnz,
        grad_offsets.data_ptr<int64_t>(),
        out_offsets.data_ptr<int64_t>(),
        lower_bound_values.data_ptr<int64_t>(),
        values_accessor,
        out_values_accessor,
        grad_values_accessor);
    sycl_kernel_submit(global_range, local_range, getCurrentSYCLQueue(), kfn);
  }
}

Tensor softmax_sparse_xpu_kernel(
    const Tensor& input_,
    const int64_t dim_,
    const bool half_to_float) {
  Tensor input, output;
  int64_t dim;
  std::tie(input, output, dim) = softmax_sparse_input_preprocessing(
      input_, dim_, half_to_float, "softmax");
  if (input.numel() == 0) {
    return output;
  }
  if (input._values().numel() == 0) {
    return output;
  }

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "softmax", [&] {
    xpu_sparse_coo_softmax<scalar_t, false>(output, input, dim);
  });
  return output;
}

Tensor log_softmax_sparse_xpu_kernel(
    const Tensor& input_,
    const int64_t dim_,
    const bool half_to_float) {
  Tensor input, output;
  int64_t dim;
  std::tie(input, output, dim) = softmax_sparse_input_preprocessing(
      input_, dim_, half_to_float, "log_softmax");
  if (input.numel() == 0) {
    return output;
  }
  if (input._values().numel() == 0) {
    return output;
  }
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "log_softmax", [&] {
    xpu_sparse_coo_softmax<scalar_t, true>(output, input, dim);
  });
  return output;
}

Tensor softmax_backward_sparse_xpu_kernel(
    const Tensor& grad_,
    const Tensor& output_,
    int64_t dim_,
    const Tensor& input_) {
  Tensor grad_input, grad, output;
  int64_t dim;
  std::tie(grad_input, grad, output, dim) =
      softmax_backward_sparse_input_preprocessing(
          grad_, output_, dim_, input_, "softmax_backward");
  if (output.numel() == 0) {
    return grad_input;
  }
  if (output._values().numel() == 0) {
    return grad_input;
  }

  AT_DISPATCH_FLOATING_TYPES(grad.scalar_type(), "softmax_backward", [&] {
    xpu_sparse_coo_softmax_backward<scalar_t, false>(
        grad_input, grad, output, dim_, input_.scalar_type());
  });
  return grad_input;
}

Tensor log_softmax_backward_sparse_xpu_kernel(
    const Tensor& grad_,
    const Tensor& output_,
    int64_t dim_,
    const Tensor& input_) {
  Tensor grad_input, grad, output;
  int64_t dim;
  std::tie(grad_input, grad, output, dim) =
      softmax_backward_sparse_input_preprocessing(
          grad_, output_, dim_, input_, "log_softmax_backward");
  if (output.numel() == 0) {
    return grad_input;
  }
  if (output._values().numel() == 0) {
    return grad_input;
  }

  AT_DISPATCH_FLOATING_TYPES(grad.scalar_type(), "log_softmax_backward", [&] {
    xpu_sparse_coo_softmax_backward<scalar_t, true>(
        grad_input, grad, output, dim_, input_.scalar_type());
  });
  return grad_input;
}

} // namespace at::native::xpu
