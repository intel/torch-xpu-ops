#include <ATen/ATen.h>
#include <ATen/native/xpu/sycl/FbgemmKernels.h>
#include <ATen/native/xpu/sycl/KernelUtils.h>

#include <ATen/native/xpu/sycl/ScanUtils.h>

#include <comm/SYCLContext.h>

namespace syclext = sycl::ext::oneapi;
namespace syclexp = sycl::ext::oneapi::experimental;

namespace at {
namespace native {
namespace xpu {

template <typename T>
struct StackArray {
  T vals[kStackArrayMaxDims];
  size_t ndim;
};

template <typename T>
class SimpleAddFunctor3 {
 public:
  T operator()(T x, T y, T /*unused*/) {
    return x + y;
  }
};

template <typename T>
class SimpleRetSecondFunctor3 {
 public:
  T operator()(T /*unused*/, T y, T /*unused*/) {
    return y;
  }
};

template <typename T>
class SimpleRetFirstFunctor2 {
 public:
  T operator()(T x, T /*unused*/) {
    return x;
  }
};

#define JAGGED_TENSOR_DISPATCH_DIMS()                                         \
  AT_DISPATCH_INDEX_TYPES(x_offsets[0].scalar_type(), "jagged_indices", [=] { \
    switch (num_jagged_dim) {                                                 \
      case 1:                                                                 \
        INVOKE_KERNEL_WITH_DIM(1);                                            \
        break;                                                                \
      case 2:                                                                 \
        INVOKE_KERNEL_WITH_DIM(2);                                            \
        break;                                                                \
      case 3:                                                                 \
        INVOKE_KERNEL_WITH_DIM(3);                                            \
        break;                                                                \
      case 4:                                                                 \
        INVOKE_KERNEL_WITH_DIM(4);                                            \
        break;                                                                \
      case 5:                                                                 \
        INVOKE_KERNEL_WITH_DIM(5);                                            \
        break;                                                                \
      default:                                                                \
        TORCH_CHECK(                                                          \
            false, "unsupported number of jagged dim ", num_jagged_dim);      \
    }                                                                         \
  });

template <int NUM_JAGGED_DIM, typename index_t, typename scalar_t, typename F>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<2>))
void jagged_dense_elementwise_jagged_output_kernel_(
    GenericPackedTensorAccessor<scalar_t, 2, DefaultPtrTraits, int32_t>
        x_values,
    StackArray<index_t*> x_offsets,
    StackArray<int64_t> x_offsets_sizes,
    GenericPackedTensorAccessor<scalar_t, 3, DefaultPtrTraits, int32_t> y_0,
    GenericPackedTensorAccessor<scalar_t, 3, DefaultPtrTraits, int32_t> y_1,
    GenericPackedTensorAccessor<scalar_t, 2, DefaultPtrTraits, int32_t>
        output_values,
    StackArray<int64_t> jagged_dims,
    F f) {
  auto output_values_acc = output_values;
  const int outer_dense_size = y_0.size(0);
  const int inner_dense_size = y_0.size(2);
  const int nnz = x_values.size(0);

  auto item = syclext::this_work_item::get_nd_item<2>();
  const int offset_begin =
      item.get_group(0) * item.get_local_range(1) + item.get_local_id(1);
  const int offset_stride = item.get_global_range(0) * item.get_local_range(1);
  for (int offset = offset_begin; offset < nnz; offset += offset_stride) {
    int offset_temp = offset;
    int jidx = 0;
    bool truncated = false;
    int dim_prod = 1;
#pragma unroll
    for (int d = NUM_JAGGED_DIM - 1; d >= 0; --d) {
      // Binary search the first that is bigger than offset
      int count = x_offsets_sizes.vals[d] - 1;
      int first = 1;
      while (count > 0) {
        int idx = first;
        int step = count / 2;
        idx += step;
        if (x_offsets.vals[d][idx] <= offset_temp) {
          first = ++idx;
          count -= step + 1;
        } else {
          count = step;
        }
      }

      --first;
      int coord = offset_temp - x_offsets.vals[d][first];
      if (coord >= jagged_dims.vals[d]) {
        truncated = true;
        break;
      }
      jidx += coord * dim_prod;
      dim_prod *= jagged_dims.vals[d];
      offset_temp = first;
    }

    if (offset_temp >= outer_dense_size) {
      // This can happen when values have more elements than the last element of
      // offset
      truncated = true;
    }
    if (!truncated) {
      const int oidx = offset_temp;
      int iidx;
      for (iidx = item.get_local_id(0); iidx * 2 + 1 < inner_dense_size;
           iidx += item.get_local_range(0)) {
        output_values_acc[offset][2 * iidx] =
            f(x_values[offset][2 * iidx],
              y_0[oidx][jidx][2 * iidx],
              y_1[oidx][jidx][2 * iidx]);
        output_values_acc[offset][2 * iidx + 1] =
            f(x_values[offset][2 * iidx + 1],
              y_0[oidx][jidx][2 * iidx + 1],
              y_1[oidx][jidx][2 * iidx + 1]);
      }
      if (iidx * 2 + 1 == inner_dense_size) {
        output_values_acc[offset][2 * iidx] =
            f(x_values[offset][2 * iidx],
              y_0[oidx][jidx][2 * iidx],
              y_1[oidx][jidx][2 * iidx]);
      }
    } else {
      int iidx;
      for (iidx = item.get_local_id(0); iidx * 2 + 1 < inner_dense_size;
           iidx += item.get_local_range(0)) {
        output_values_acc[offset][2 * iidx] =
            f(x_values[offset][2 * iidx], 0, 0);
        output_values_acc[offset][2 * iidx + 1] =
            f(x_values[offset][2 * iidx + 1], 0, 0);
      }
      if (iidx * 2 + 1 == inner_dense_size) {
        output_values_acc[offset][2 * iidx] =
            f(x_values[offset][2 * iidx], 0, 0);
      }
    }
  }
}

template <int NUM_JAGGED_DIM, typename index_t, typename scalar_t, typename F>
void jagged_dense_elementwise_jagged_output_launch_(
    GenericPackedTensorAccessor<scalar_t, 2, DefaultPtrTraits, int32_t>
        x_values, // output
    StackArray<index_t*> x_offsets,
    StackArray<int64_t> x_offsets_sizes,
    GenericPackedTensorAccessor<scalar_t, 3, DefaultPtrTraits, int32_t>
        y_0, // not used
    GenericPackedTensorAccessor<scalar_t, 3, DefaultPtrTraits, int32_t> y_1,
    GenericPackedTensorAccessor<scalar_t, 2, DefaultPtrTraits, int32_t>
        output_values, // not used
    StackArray<int64_t> jagged_dims,
    F f,
    int64_t wg_0,
    int64_t wg_1,
    int64_t wg_num) {
  sycl_kernel_submit<jagged_dense_elementwise_jagged_output_kernel_<
      NUM_JAGGED_DIM,
      index_t,
      scalar_t,
      F>>(
      sycl::range<2>(wg_0 * wg_num, wg_1),
      sycl::range<2>(wg_0, wg_1),
      getCurrentSYCLQueue(),
      0,
      x_values,
      x_offsets,
      x_offsets_sizes,
      y_0,
      y_1,
      output_values,
      jagged_dims,
      f);
}

template <int NUM_JAGGED_DIM, typename index_t>
bool walk_down_tensor_storage_tree_(
    int& offset,
    const int flattened_jagged_idx,
    const StackArray<int64_t>& jagged_dims,
    const StackArray<index_t*>& x_offsets) {
  // compute coorindates
  int jagged_coords[NUM_JAGGED_DIM];
  int j_temp = flattened_jagged_idx;
#pragma unroll
  for (int d = NUM_JAGGED_DIM - 1; d >= 0; --d) {
    const int jagged_size = jagged_dims.vals[d];
    jagged_coords[d] = j_temp % jagged_size;
    j_temp /= jagged_size;
  }

  // walk down the tree
  bool is_zero = false;
#pragma unroll
  for (int d = 0; d < NUM_JAGGED_DIM; ++d) {
    const int begin = x_offsets.vals[d][offset];
    const int end = x_offsets.vals[d][offset + 1];
    if (jagged_coords[d] >= end - begin) {
      is_zero = true;
      break;
    }
    offset = begin + jagged_coords[d];
  }
  return is_zero;
}

inline std::tuple<int64_t, int64_t, int64_t, StackArray<int64_t>>
check_shape_and_partition_(
    const Tensor& values,
    const std::vector<Tensor>& offsets,
    const Tensor& dense_tensor) {
  const int32_t outer_dense_size = dense_tensor.size(0);
  TORCH_CHECK(
      outer_dense_size == offsets[0].numel() - 1,
      "outer_dense_size, ",
      outer_dense_size,
      " != offsets[0].numel() - 1, ",
      offsets[0].numel() - 1);
  const int32_t inner_dense_size = dense_tensor.size(-1);
  TORCH_CHECK(
      inner_dense_size == values.size(-1),
      "inner_dense_size, ",
      inner_dense_size,
      " != values.size(-1), ",
      values.size(-1));
  const int32_t jagged_folded_size =
      dense_tensor.numel() / (outer_dense_size * inner_dense_size);

  const int32_t sub_group_size = syclMaxSubGroupSize();
  const int64_t wg_size_0 = inner_dense_size >= sub_group_size / 2
      ? sub_group_size
      : inner_dense_size;
  const int64_t wg_size_1 = syclDeviceMaxWorkGroupSize() / sub_group_size;
  const int64_t wg_num =
      CeilDivUp(outer_dense_size * jagged_folded_size, (int32_t)wg_size_1);

  StackArray<int64_t> jagged_dims_tensor;
  const int32_t num_jagged_dim = dense_tensor.dim() - 2;
  TORCH_CHECK(num_jagged_dim <= kStackArrayMaxDims);
  jagged_dims_tensor.ndim = num_jagged_dim;
  std::memcpy(
      &(jagged_dims_tensor.vals[0]),
      dense_tensor.sizes().data() + 1,
      num_jagged_dim * sizeof(int64_t));
  return {wg_size_0, wg_size_1, wg_num, jagged_dims_tensor};
}

template <typename scalar_t, typename F>
void jagged_dense_elementwise_jagged_output_(
    const Tensor& x_values,
    const std::vector<Tensor>& x_offsets,
    const Tensor& y,
    const Tensor& output_values,
    F f) {
  // Canonicalize y to 3D, collapsing jagged dimensions.
  const int num_jagged_dim = y.dim() - 2;
  const Tensor y_reshaped = y.view({y.size(0), -1, y.size(-1)});
#define INVOKE_KERNEL_WITH_DIM(NUM_JAGGED_DIM)                               \
  {                                                                          \
    int64_t wg_0, wg_1, wg_num;                                              \
    StackArray<int64_t> jagged_dims_tensor;                                  \
    std::tie(wg_0, wg_1, wg_num, jagged_dims_tensor) =                       \
        check_shape_and_partition_(x_values, x_offsets, y);                  \
    wg_num = CeilDivUp(x_values.size(0), wg_1);                              \
    std::vector<Tensor> x_offsets_contig;                                    \
    x_offsets_contig.resize(num_jagged_dim);                                 \
    StackArray<index_t*> x_offset_ptrs;                                      \
    x_offset_ptrs.ndim = num_jagged_dim;                                     \
    StackArray<int64_t> x_offset_sizes;                                      \
    x_offset_sizes.ndim = num_jagged_dim;                                    \
    for (int d = 0; d < num_jagged_dim; ++d) {                               \
      x_offsets_contig[d] = x_offsets[d].contiguous();                       \
      x_offset_ptrs.vals[d] =                                                \
          x_offsets_contig[d].template data_ptr<index_t>();                  \
      x_offset_sizes.vals[d] = x_offsets[d].numel();                         \
    }                                                                        \
    jagged_dense_elementwise_jagged_output_launch_<NUM_JAGGED_DIM, index_t>( \
        x_values.packed_accessor32<scalar_t, 2>(),                           \
        x_offset_ptrs,                                                       \
        x_offset_sizes,                                                      \
        y_reshaped.packed_accessor32<scalar_t, 3>(),                         \
        y_reshaped.packed_accessor32<scalar_t, 3>(),                         \
        output_values.packed_accessor32<scalar_t, 2>(),                      \
        jagged_dims_tensor,                                                  \
        f,                                                                   \
        wg_0,                                                                \
        wg_1,                                                                \
        wg_num);                                                             \
  }

  JAGGED_TENSOR_DISPATCH_DIMS();
#undef INVOKE_KERNEL_WITH_DIM
}

void dense_to_jagged_forward_xpu_kernel(
    const Tensor& x_values,
    const std::vector<Tensor>& x_offsets,
    const Tensor& y,
    const Tensor& output_values) {
  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      x_values.scalar_type(),
      "dense_to_jagged_forward_xpu_kernel",
      [&]() {
        jagged_dense_elementwise_jagged_output_<scalar_t>(
            x_values,
            x_offsets,
            y,
            output_values,
            SimpleRetSecondFunctor3<scalar_t>());
      });
}

template <int NUM_JAGGED_DIM, typename index_t, typename scalar_t, typename F>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<2>))
void jagged_dense_elementwise_dense_output_kernel_(
    GenericPackedTensorAccessor<scalar_t, 2, DefaultPtrTraits, int32_t>
        x_values,
    StackArray<index_t*> x_offsets,
    GenericPackedTensorAccessor<scalar_t, 3, DefaultPtrTraits, int32_t> y,
    GenericPackedTensorAccessor<scalar_t, 3, DefaultPtrTraits, int32_t> output,
    StackArray<int64_t> jagged_dims,
    const scalar_t padding_value,
    F f) {
  auto output_acc = output;
  const int outer_dense_size = y.size(0);
  const int jagged_folded_size = y.size(1);
  const int inner_dense_size = y.size(2);

  auto item = syclext::this_work_item::get_nd_item<2>();
  const int outer_begin =
      item.get_group(0) * item.get_local_range(1) + item.get_local_id(1);
  const int outer_stride = item.get_global_range(0) * item.get_local_range(1);
  for (int outer = outer_begin; outer < outer_dense_size * jagged_folded_size;
       outer += outer_stride) {
    const int oidx = outer / jagged_folded_size;
    const int jidx = outer % jagged_folded_size;

    int offset = oidx;
    const bool is_zero = walk_down_tensor_storage_tree_<NUM_JAGGED_DIM>(
        offset, jidx, jagged_dims, x_offsets);

    if (is_zero) {
      int iidx;
      for (iidx = item.get_local_id(0); iidx * 2 + 1 < inner_dense_size;
           iidx += item.get_local_range(0)) {
        output_acc[oidx][jidx][2 * iidx] =
            f(padding_value, y[oidx][jidx][2 * iidx]);
        output_acc[oidx][jidx][2 * iidx + 1] =
            f(padding_value, y[oidx][jidx][2 * iidx + 1]);
      }
      if (iidx * 2 + 1 == inner_dense_size) {
        output_acc[oidx][jidx][2 * iidx] =
            f(padding_value, y[oidx][jidx][2 * iidx]);
      }
    } else {
      int iidx;
      for (iidx = item.get_local_id(0); iidx * 2 + 1 < inner_dense_size;
           iidx += item.get_local_range(0)) {
        output_acc[oidx][jidx][2 * iidx] =
            f(x_values[offset][2 * iidx], y[oidx][jidx][2 * iidx]);
        output_acc[oidx][jidx][2 * iidx + 1] =
            f(x_values[offset][2 * iidx + 1], y[oidx][jidx][2 * iidx + 1]);
      }
      if (iidx * 2 + 1 == inner_dense_size) {
        output_acc[oidx][jidx][2 * iidx] =
            f(x_values[offset][2 * iidx], y[oidx][jidx][2 * iidx]);
      }
    }
  }
}

template <int NUM_JAGGED_DIM, typename index_t, typename scalar_t>
void jagged_dense_elementwise_dense_output_launch_(
    const GenericPackedTensorAccessor<scalar_t, 2, DefaultPtrTraits, int32_t>
        x_values,
    StackArray<index_t*> x_offsets,
    const GenericPackedTensorAccessor<scalar_t, 3, DefaultPtrTraits, int32_t> y,
    GenericPackedTensorAccessor<scalar_t, 3, DefaultPtrTraits, int32_t> output,
    StackArray<int64_t> jagged_dims,
    const scalar_t padding_value,
    int64_t wg_0,
    int64_t wg_1,
    int64_t wg_num) {
  sycl_kernel_submit<jagged_dense_elementwise_dense_output_kernel_<
      NUM_JAGGED_DIM,
      index_t,
      scalar_t,
      SimpleRetFirstFunctor2<scalar_t>>>(
      sycl::range<2>(wg_0 * wg_num, wg_1),
      sycl::range<2>(wg_0, wg_1),
      getCurrentSYCLQueue(),
      0,
      x_values,
      x_offsets,
      y,
      output,
      jagged_dims,
      padding_value,
      SimpleRetFirstFunctor2<scalar_t>());
}

template <typename scalar_t>
void jagged_dense_elementwise_dense_output_(
    const Tensor& x_values,
    const std::vector<Tensor>& x_offsets,
    const Tensor& y,
    const Tensor& output,
    const scalar_t padding_value) {
  int64_t wg_0, wg_1, wg_num;
  StackArray<int64_t> jagged_dims_tensor;
  std::tie(wg_0, wg_1, wg_num, jagged_dims_tensor) =
      check_shape_and_partition_(x_values, x_offsets, y);

  // Canonicalize y and output to 3D, collapsing jagged dimensions.
  const int num_jagged_dim = y.dim() - 2;
  const Tensor y_reshaped = y.view({y.size(0), -1, y.size(-1)});
  Tensor output_reshaped = output.view(y_reshaped.sizes());

#define INVOKE_KERNEL_WITH_DIM(NUM_JAGGED_DIM)                              \
  {                                                                         \
    std::vector<Tensor> x_offsets_contig;                                   \
    x_offsets_contig.resize(num_jagged_dim);                                \
    StackArray<index_t*> x_offset_ptrs;                                     \
    x_offset_ptrs.ndim = num_jagged_dim;                                    \
    for (int d = 0; d < num_jagged_dim; ++d) {                              \
      x_offsets_contig[d] = x_offsets[d].contiguous();                      \
      x_offset_ptrs.vals[d] =                                               \
          x_offsets_contig[d].template data_ptr<index_t>();                 \
    }                                                                       \
    jagged_dense_elementwise_dense_output_launch_<NUM_JAGGED_DIM, index_t>( \
        x_values.packed_accessor32<scalar_t, 2>(),                          \
        x_offset_ptrs,                                                      \
        y_reshaped.packed_accessor32<scalar_t, 3>(),                        \
        output_reshaped.packed_accessor32<scalar_t, 3>(),                   \
        jagged_dims_tensor,                                                 \
        padding_value,                                                      \
        wg_0,                                                               \
        wg_1,                                                               \
        wg_num);                                                            \
  }

  JAGGED_TENSOR_DISPATCH_DIMS();
#undef INVOKE_KERNEL_WITH_DIM
}

void jagged_to_padded_dense_forward_xpu_kernel(
    const Tensor& x_values,
    const std::vector<Tensor>& x_offsets,
    const Tensor& y,
    const Tensor& output,
    const double padding_value) {
  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      x_values.scalar_type(),
      "jagged_to_padded_dense_forward_xpu_kernel",
      [&] {
        jagged_dense_elementwise_dense_output_<scalar_t>(
            x_values,
            x_offsets,
            y, // not used
            output,
            static_cast<scalar_t>(padding_value));
      });
}

// Check to see if the inputs to the op are amenable to the fast path
inline bool jagged_dense_dense_elementwise_jagged_output_matches_opt(
    const int& num_jagged_dim,
    const Tensor& x_values,
    const std::vector<Tensor>& x_offsets,
    const Tensor& y_0_reshaped,
    const Tensor& y_1_reshaped,
    const Tensor& output_values) {
  bool matches = true;
  matches &= (num_jagged_dim == 1);

  // Unit stride embedding dim
  matches &= (x_values.stride(-1) == 1);
  matches &= (output_values.stride(-1) == 1);
  matches &= (y_0_reshaped.stride(-1) == 1);
  matches &= (y_1_reshaped.stride(-1) == 1);

  // Each row is aligned to 128-bit
  matches &= ((x_values.stride(-2) & 0x7) == 0);
  matches &= ((output_values.stride(-2) & 0x7) == 0);
  matches &= ((y_0_reshaped.stride(-2) & 0x7) == 0);
  matches &= ((y_1_reshaped.stride(-2) & 0x7) == 0);

  // Base addresses aligned to 128-bit
  matches &= ((reinterpret_cast<uint64_t>(x_values.data_ptr()) & 0xF) == 0);
  matches &=
      ((reinterpret_cast<uint64_t>(output_values.data_ptr()) % 0xF) == 0);
  matches &= ((reinterpret_cast<uint64_t>(y_0_reshaped.data_ptr()) % 0xF) == 0);
  matches &= ((reinterpret_cast<uint64_t>(y_1_reshaped.data_ptr()) % 0xF) == 0);

  // Rows and col fit into int32_t
  matches &= (y_0_reshaped.size(0) < INT_MAX);
  matches &= (y_0_reshaped.size(1) < INT_MAX);

  // maximum shared local memory size
  int max_shared_bytes = syclLocalMemSize();
  // Use all shared memory, no L1 cache consideration
  int max_shared_kb = max_shared_bytes >> 10;
  int used_shared_kb = round_down(max_shared_kb, 16);
  TORCH_CHECK(used_shared_kb > 0);
  int used_shared_bytes = used_shared_kb << 10;
  AT_DISPATCH_INDEX_TYPES(
      x_offsets[0].scalar_type(), "check_shared_memory", [&] {
        auto B = y_0_reshaped.size(0);
        // the default shared memory on V100/A100/H100 is 48 KB from
        // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-8-x
        if ((B + 1) * sizeof(index_t) >= used_shared_bytes) {
          matches = false;
        }
      });
  return matches;
}

template <typename index_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<3>))
void jagged_dense_dense_elementwise_jagged_output_opt_search_kernel_(
    const GenericPackedTensorAccessor<
        index_t,
        1,
        at::RestrictPtrTraits,
        int32_t> offsets,
    GenericPackedTensorAccessor<int, 1, at::RestrictPtrTraits, int32_t> rows,
    GenericPackedTensorAccessor<int, 1, at::RestrictPtrTraits, int32_t> cols,
    int nnz,
    int B) {
  index_t* offsets_sh =
      reinterpret_cast<index_t*>(syclexp::get_work_group_scratch_memory());
  auto item = syclext::this_work_item::get_nd_item<3>();

  for (auto i = item.get_local_id(0); i < B + 1; i += item.get_local_range(0)) {
    offsets_sh[i] = offsets[i];
  }
  group_barrier(item.get_group());
  auto row = item.get_local_id(0) + item.get_group(0) * item.get_local_range(0);
  if (row >= nnz)
    return;
  int first = -1;
  int count = B - 1;
  first = 1;
  while (count > 0) {
    int idx = first;
    int step = count / 2;
    idx += step;
    if (offsets_sh[idx] <= row) {
      first = ++idx;
      count -= step + 1;
    } else {
      count = step;
    }
  }
  --first;

  int dense_row = first;
  int offset = offsets_sh[dense_row];
  int dense_col = row - offset;
  rows[row] = dense_row;
  cols[row] = dense_col;
}

template <typename index_t, typename F>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<3>))
void jagged_dense_dense_elementwise_jagged_output_opt_gather_kernel_(
    GenericPackedTensorAccessor<c10::Half, 2, at::RestrictPtrTraits, int32_t>
        values,
    const GenericPackedTensorAccessor<
        c10::Half,
        2,
        at::RestrictPtrTraits,
        int32_t> x_values,
    const GenericPackedTensorAccessor<
        c10::Half,
        3,
        at::RestrictPtrTraits,
        int32_t> y0,
    const GenericPackedTensorAccessor<
        c10::Half,
        3,
        at::RestrictPtrTraits,
        int32_t> y1,
    const GenericPackedTensorAccessor<int, 1, at::RestrictPtrTraits, int32_t>
        rows,
    const GenericPackedTensorAccessor<int, 1, at::RestrictPtrTraits, int32_t>
        cols,
    const int nnz,
    const int E,
    F f) {
  auto item = syclext::this_work_item::get_nd_item<3>();
  auto values_row =
      item.get_local_id(1) + item.get_group(1) * item.get_local_range(1);
  if (values_row >= nnz)
    return;
  for (int real_row = values_row; real_row < nnz;
       real_row += item.get_local_range(1) * item.get_group_range(1)) {
    int dense_row = rows[real_row];
    int dense_col = cols[real_row];
    sycl::half* values_ptr =
        reinterpret_cast<sycl::half*>(&values[real_row][0]);
    const sycl::half* x_ptr =
        reinterpret_cast<const sycl::half*>(&x_values[real_row][0]);
    const sycl::half* y0_ptr =
        reinterpret_cast<const sycl::half*>(&y0[dense_row][dense_col][0]);
    const sycl::half* y1_ptr =
        reinterpret_cast<const sycl::half*>(&y1[dense_row][dense_col][0]);
    if ((dense_col < y0.size(1)) && (dense_row < y0.size(0)) &&
        (dense_col < y1.size(1)) && (dense_row < y1.size(0)) &&
        (dense_col >= 0) && (dense_row >= 0)) {
      for (auto tid = item.get_local_id(0); tid < E / 8;
           tid += item.get_local_range(0)) {
        VecType128 v_x, v_out, v_y0, v_y1;
        v_x.data.mask =
            (reinterpret_cast<const VecType128::TType*>(x_ptr))[tid];
        v_y0.data.mask =
            (reinterpret_cast<const VecType128::TType*>(y0_ptr))[tid];
        v_y1.data.mask =
            (reinterpret_cast<const VecType128::TType*>(y1_ptr))[tid];
        f128(v_out, v_x, v_y0, v_y1, f);
        (reinterpret_cast<VecType128::TType*>(values_ptr))[tid] =
            v_out.data.mask;
      }
      for (auto tid = item.get_local_id(0) + (E / 8) * 8; tid < E / 4;
           tid += item.get_local_range(0)) {
        VecType64 v_x, v_out, v_y0, v_y1;
        v_x.data.mask = (reinterpret_cast<const VecType64::TType*>(x_ptr))[tid];
        v_y0.data.mask =
            (reinterpret_cast<const VecType64::TType*>(y0_ptr))[tid];
        v_y1.data.mask =
            (reinterpret_cast<const VecType64::TType*>(y1_ptr))[tid];
        f64(v_out, v_x, v_y0, v_y1, f);
        (reinterpret_cast<VecType64::TType*>(values_ptr))[tid] =
            v_out.data.mask;
      }
      for (auto tid = item.get_local_id(0) + (E / 4) * 4; tid < E / 2;
           tid += item.get_local_range(0)) {
        VecType32 v_x, v_out, v_y0, v_y1;
        v_x.data.mask = (reinterpret_cast<const VecType32::TType*>(x_ptr))[tid];
        v_y0.data.mask =
            (reinterpret_cast<const VecType32::TType*>(y0_ptr))[tid];
        v_y1.data.mask =
            (reinterpret_cast<const VecType32::TType*>(y1_ptr))[tid];
        f32(v_out, v_x, v_y0, v_y1, f);
        (reinterpret_cast<VecType32::TType*>(values_ptr))[tid] =
            v_out.data.mask;
      }
      for (auto tid = item.get_local_id(0) + (E / 2) * 2; tid < E;
           tid += item.get_local_range(0)) {
        sycl::half v_x, v_out, v_y0, v_y1;
        v_x = static_cast<sycl::half>(x_ptr[tid]);
        v_y0 = static_cast<sycl::half>(y0_ptr[tid]);
        v_y1 = static_cast<sycl::half>(y1_ptr[tid]);
        fh(v_out, v_x, v_y0, v_y1, f);
        values_ptr[tid] = v_out;
      }
    } else {
      for (auto tid = item.get_local_id(0); tid < E / 8;
           tid += item.get_local_range(0)) {
        VecType128 v_x, v_out, v_y0, v_y1;
        v_x.data.mask =
            (reinterpret_cast<const VecType128::TType*>(x_ptr))[tid];
        f128(v_out, v_x, v_y0, v_y1, f);
        (reinterpret_cast<VecType128::TType*>(values_ptr))[tid] =
            v_out.data.mask;
      }
      for (auto tid = item.get_local_id(0) + (E / 8) * 8; tid < E / 4;
           tid += item.get_local_range(0)) {
        VecType64 v_x, v_out, v_y0, v_y1;
        v_x.data.mask = (reinterpret_cast<const VecType64::TType*>(x_ptr))[tid];
        f64(v_out, v_x, v_y0, v_y1, f);
        (reinterpret_cast<VecType64::TType*>(values_ptr))[tid] =
            v_out.data.mask;
      }
      for (auto tid = item.get_local_id(0) + (E / 4) * 4; tid < E / 2;
           tid += item.get_local_range(0)) {
        VecType32 v_x, v_out, v_y0, v_y1;
        v_x.data.mask = (reinterpret_cast<const VecType32::TType*>(x_ptr))[tid];
        f32(v_out, v_x, v_y0, v_y1, f);
        (reinterpret_cast<VecType32::TType*>(values_ptr))[tid] =
            v_out.data.mask;
      }
      for (auto tid = item.get_local_id(0) + (E / 2) * 2; tid < E;
           tid += item.get_local_range(0)) {
        sycl::half v_x, v_out, v_y0, v_y1;
        v_x = static_cast<sycl::half>(x_ptr[tid]);
        fh(v_out, v_x, v_y0, v_y1, f);
        values_ptr[tid] = v_out;
      }
    }
  }
}

template <int NUM_JAGGED_DIM, typename index_t, typename scalar_t, typename F>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<2>))
void jagged_dense_dense_elementwise_jagged_output_kernel_(
    const GenericPackedTensorAccessor<
        scalar_t,
        2,
        at::RestrictPtrTraits,
        int32_t> x_values,
    StackArray<index_t*> x_offsets,
    StackArray<int64_t> x_offsets_sizes,
    const GenericPackedTensorAccessor<
        scalar_t,
        3,
        at::RestrictPtrTraits,
        int32_t> y_0,
    const GenericPackedTensorAccessor<
        scalar_t,
        3,
        at::RestrictPtrTraits,
        int32_t> y_1,
    GenericPackedTensorAccessor<scalar_t, 2, at::RestrictPtrTraits, int32_t>
        output_values,
    StackArray<int64_t> jagged_dims,
    F f) {
  const int outer_dense_size = y_0.size(0);
  const int inner_dense_size = y_0.size(2);
  const int nnz = x_values.size(0);

  auto item = syclext::this_work_item::get_nd_item<2>();

  const auto offset_begin =
      item.get_group(0) * item.get_local_range(1) + item.get_local_id(1);
  const auto offset_stride = item.get_group_range(0) * item.get_local_range(1);
  for (int offset = offset_begin; offset < nnz; offset += offset_stride) {
    int offset_temp = offset;
    int jidx = 0;
    bool truncated = false;
    int dim_prod = 1;
#pragma unroll
    for (int d = NUM_JAGGED_DIM - 1; d >= 0; --d) {
      // Binary search the first that is bigger than offset
      int count = x_offsets_sizes.vals[d] - 1;
      int first = 1;
      while (count > 0) {
        int idx = first;
        int step = count / 2;
        idx += step;
        if (x_offsets.vals[d][idx] <= offset_temp) {
          first = ++idx;
          count -= step + 1;
        } else {
          count = step;
        }
      }

      --first;
      int coord = offset_temp - x_offsets.vals[d][first];
      if (coord >= jagged_dims.vals[d]) {
        truncated = true;
        break;
      }
      jidx += coord * dim_prod;
      dim_prod *= jagged_dims.vals[d];
      offset_temp = first;
    }

    if (offset_temp >= outer_dense_size) {
      // This can happen when values have more elements than the last element of
      // offset
      truncated = true;
    }
    if (!truncated) {
      const int oidx = offset_temp;
      int iidx;
      for (iidx = item.get_local_id(0); iidx * 2 + 1 < inner_dense_size;
           iidx += item.get_local_range(0)) {
        output_values[offset][2 * iidx] =
            f(x_values[offset][2 * iidx],
              y_0[oidx][jidx][2 * iidx],
              y_1[oidx][jidx][2 * iidx]);
        output_values[offset][2 * iidx + 1] =
            f(x_values[offset][2 * iidx + 1],
              y_0[oidx][jidx][2 * iidx + 1],
              y_1[oidx][jidx][2 * iidx + 1]);
      }
      if (iidx * 2 + 1 == inner_dense_size) {
        output_values[offset][2 * iidx] =
            f(x_values[offset][2 * iidx],
              y_0[oidx][jidx][2 * iidx],
              y_1[oidx][jidx][2 * iidx]);
      }
    } else {
      int iidx;
      for (iidx = item.get_local_id(0); iidx * 2 + 1 < inner_dense_size;
           iidx += item.get_local_range(0)) {
        output_values[offset][2 * iidx] = f(x_values[offset][2 * iidx], 0, 0);
        output_values[offset][2 * iidx + 1] =
            f(x_values[offset][2 * iidx + 1], 0, 0);
      }
      if (iidx * 2 + 1 == inner_dense_size) {
        output_values[offset][2 * iidx] = f(x_values[offset][2 * iidx], 0, 0);
      }
    }
  }
}

// defined for jagged_dense_elementwise_jagged_output_opt_
// and jagged_dense_elementwise_jagged_output_
#define INVOKE_KERNEL_WITH_DIM(NUM_JAGGED_DIM)                                 \
  {                                                                            \
    int64_t wg_0, wg_1, wg_num;                                                \
    StackArray<int64_t> jagged_dims_tensor;                                    \
    std::tie(wg_0, wg_1, wg_num, jagged_dims_tensor) =                         \
        check_shape_and_partition_(x_values, x_offsets, y);                    \
    wg_num = CeilDivUp(x_values.size(0), wg_1);                                \
    std::vector<Tensor> x_offsets_contig;                                      \
    x_offsets_contig.resize(num_jagged_dim);                                   \
    StackArray<index_t*> x_offset_ptrs;                                        \
    x_offset_ptrs.ndim = num_jagged_dim;                                       \
    StackArray<int64_t> x_offset_sizes;                                        \
    x_offset_sizes.ndim = num_jagged_dim;                                      \
    for (int d = 0; d < num_jagged_dim; ++d) {                                 \
      x_offsets_contig[d] = x_offsets[d].contiguous();                         \
      x_offset_ptrs.vals[d] =                                                  \
          x_offsets_contig[d].template data_ptr<index_t>();                    \
      x_offset_sizes.vals[d] = x_offsets[d].numel();                           \
    }                                                                          \
    sycl_kernel_submit<jagged_dense_dense_elementwise_jagged_output_kernel_<   \
        NUM_JAGGED_DIM,                                                        \
        index_t,                                                               \
        scalar_t,                                                              \
        F>>(                                                                   \
        sycl::range<2>(wg_0 * wg_num, wg_1),                                   \
        sycl::range<2>(wg_0, wg_1),                                            \
        getCurrentSYCLQueue(),                                                 \
        0,                                                                     \
        x_values.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(),      \
        x_offset_ptrs,                                                         \
        x_offset_sizes,                                                        \
        y_reshaped.packed_accessor32<scalar_t, 3, at::RestrictPtrTraits>(),    \
        y_reshaped.packed_accessor32<scalar_t, 3, at::RestrictPtrTraits>(),    \
        output_values.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>(), \
        jagged_dims_tensor,                                                    \
        f);                                                                    \
  } // namespace xpu

void cumsum_kernel(const Tensor& result, const Tensor& self, int64_t dim) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      ScalarType::Half,
      ScalarType::BFloat16,
      self.scalar_type(),
      "cumsum_xpu",
      [&]() {
        scalar_t init = 0;
        scan<INCLUSIVE_TYPE, const scalar_t, scalar_t>(
            result, self, dim, init, std::plus<scalar_t>());
      });
}

template <typename scalar_t, typename F>
void jagged_dense_elementwise_jagged_output_opt_(
    const Tensor& x_values,
    const std::vector<Tensor>& x_offsets,
    const Tensor& y,
    const Tensor& output_values,
    F f) {
  // Canonicalize y to 3D, collapsing jagged dimensions.
  const int num_jagged_dim = y.dim() - 2;
  const Tensor y_reshaped = y.view({y.size(0), -1, y.size(-1)});
  if (jagged_dense_dense_elementwise_jagged_output_matches_opt(
          num_jagged_dim,
          x_values,
          x_offsets,
          y_reshaped,
          y_reshaped,
          output_values)) {
    AT_DISPATCH_INDEX_TYPES(
        x_offsets[0].scalar_type(), "jagged_indices_fast_path", [=] {
          auto nnz = output_values.size(0);
          auto B = y_reshaped.size(0);
          auto E = y_reshaped.size(2);
          Tensor t_rows_after_bs = at::empty(
              {nnz},
              at::TensorOptions().dtype(at::kInt).device(
                  at::kXPU, c10::xpu::current_device()));
          Tensor t_cols_after_bs = at::empty(
              {nnz},
              at::TensorOptions().dtype(at::kInt).device(
                  at::kXPU, c10::xpu::current_device()));

          // Binary search
          size_t dynamic_smem_size = (B + 1) * sizeof(index_t);
          auto max_shared_bytes = syclLocalMemSize();
          int max_shared_kb = max_shared_bytes >> 10;
          int used_shared_kb = round_down(max_shared_kb, 16);
          TORCH_CHECK(used_shared_kb > 0);
          int used_shared_bytes = used_shared_kb << 10;
          TORCH_CHECK(dynamic_smem_size <= used_shared_bytes);

          int max_wg_size = syclDeviceMaxWorkGroupSize();
          int wg_size = max_wg_size < 1024 ? max_wg_size : 1024;

          int nbr_of_wg = CeilDivUp(nnz, wg_size);
          sycl_kernel_submit<
              jagged_dense_dense_elementwise_jagged_output_opt_search_kernel_<
                  index_t>>(
              sycl::range<3>(nbr_of_wg * wg_size, 1, 1),
              sycl::range<3>(wg_size, 1, 1),
              getCurrentSYCLQueue(),
              dynamic_smem_size,
              x_offsets[0]
                  .packed_accessor32<index_t, 1, at::RestrictPtrTraits>(),
              t_rows_after_bs
                  .packed_accessor32<int, 1, at::RestrictPtrTraits>(),
              t_cols_after_bs
                  .packed_accessor32<int, 1, at::RestrictPtrTraits>(),
              nnz,
              B);

          // Gather kernel
          int dim_0_1 = 16;
          int nbr_of_wg_g = CeilDivUp(nnz, dim_0_1);
          if (nbr_of_wg_g > 65535) {
            nbr_of_wg_g = round_down(65535, dim_0_1);
          }

          sycl_kernel_submit<
              jagged_dense_dense_elementwise_jagged_output_opt_gather_kernel_<
                  index_t,
                  F>>(
              sycl::range<3>(dim_0_1 * 1, dim_0_1 * nbr_of_wg_g, 1),
              sycl::range<3>(dim_0_1, dim_0_1, 1),
              getCurrentSYCLQueue(),
              0,
              output_values
                  .packed_accessor32<c10::Half, 2, at::RestrictPtrTraits>(),
              x_values.packed_accessor32<c10::Half, 2, at::RestrictPtrTraits>(),
              y_reshaped
                  .packed_accessor32<c10::Half, 3, at::RestrictPtrTraits>(),
              y_reshaped
                  .packed_accessor32<c10::Half, 3, at::RestrictPtrTraits>(),
              t_rows_after_bs
                  .packed_accessor32<int, 1, at::RestrictPtrTraits>(),
              t_cols_after_bs
                  .packed_accessor32<int, 1, at::RestrictPtrTraits>(),
              nnz,
              E,
              f);
        }); // AT_DISPATCH
  } else {
    JAGGED_TENSOR_DISPATCH_DIMS();
  }
}

void jagged_dense_elementwise_add_jagged_output_fwd_xpu_kn(
    const Tensor& x_values,
    const std::vector<Tensor>& offsets,
    const Tensor& dense,
    const Tensor& output_values) {
  AT_DISPATCH_SWITCH(
      x_values.scalar_type(),
      "jagged_dense_elementwise_add_jagged_output_fwd_xpu_kn",
      AT_DISPATCH_CASE(
          at::ScalarType::Half,
          [&] {
            jagged_dense_elementwise_jagged_output_opt_<scalar_t>(
                x_values,
                offsets,
                dense,
                output_values,
                SimpleAddFunctor3<sycl::half>()); // device lambda
          } // lambda
          ) // CASE
      FBGEMM_DISPATCH_FLOAT_AND_BFLOAT16_CASE([&] {
        jagged_dense_elementwise_jagged_output_<scalar_t>(
            x_values,
            offsets,
            dense,
            output_values,
            SimpleAddFunctor3<scalar_t>()); // device lambda
      } // lambda
                                              ) // CASE_FLOATING_TYPES_AND
  ); // SWITCH
}

template <typename Dtype>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<2>))
void reorder_batched_ad_lengths_kernel_(
    // reorder lengths from (ragged) [B  x T x #num_ads_b)] to
    // [T][B][#num_ads_b], i.e. [T][sum(#num_ads_b)].
    const GenericPackedTensorAccessor<Dtype, 1, at::RestrictPtrTraits, int32_t>
        cat_ad_lengths,
    const GenericPackedTensorAccessor<
        int32_t,
        1,
        at::RestrictPtrTraits,
        int32_t> batch_offsets,
    GenericPackedTensorAccessor<Dtype, 1, at::RestrictPtrTraits, int32_t>
        reordered_cat_ad_lengths,
    const int32_t T,
    const bool broadcast_lengths) {
  const int32_t B = batch_offsets.size(0) - 1;

  const int32_t num_ads_in_batch = batch_offsets[B];
  // warp-per-segment.
  auto item = syclext::this_work_item::get_nd_item<2>();
  const auto b_t =
      item.get_group(0) * item.get_local_range(1) + item.get_local_id(1);
  const int32_t b = b_t % B;
  const int32_t t = b_t / B;
  if (t >= T) {
    return;
  }

  const int32_t num_ads_b = batch_offsets[b + 1] - batch_offsets[b];
  const int32_t input_segment_start =
      broadcast_lengths ? T * b + t : T * batch_offsets[b] + t * num_ads_b;
  const int32_t output_segment_start = t * num_ads_in_batch + batch_offsets[b];

  for (auto i = item.get_local_id(0); i < num_ads_b;
       i += item.get_local_range(0)) {
    reordered_cat_ad_lengths[output_segment_start + i] = broadcast_lengths
        ? cat_ad_lengths[input_segment_start]
        : cat_ad_lengths[input_segment_start + i];
  }
}

void reorder_batched_ad_lengths_xpu_kernel(
    const Tensor& cat_ad_lengths,
    const Tensor& batch_offsets,
    Tensor& reordered_cat_ad_lengths,
    const int32_t T,
    const bool broadcast_lengths,
    const int32_t grid_size) {
  FBGEMM_DISPATCH_ALL_TYPES(
      cat_ad_lengths.scalar_type(),
      "reorder_batched_ad_lengths_xpu_kernel",
      [&] {
        sycl_kernel_submit<reorder_batched_ad_lengths_kernel_<scalar_t>>(
            sycl::range<2>(32 * grid_size, 32),
            sycl::range<2>(32, 32),
            getCurrentSYCLQueue(),
            0,
            cat_ad_lengths
                .packed_accessor32<scalar_t, 1, at::RestrictPtrTraits>(),
            batch_offsets
                .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
            reordered_cat_ad_lengths
                .packed_accessor32<scalar_t, 1, at::RestrictPtrTraits>(),
            T,
            broadcast_lengths);
      });
}

template <typename Dtype, typename index_t = int32_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void narrow_broadcast_indices_kernel_(
    const GenericPackedTensorAccessor<
        index_t,
        1,
        at::RestrictPtrTraits,
        int32_t> cat_ad_offsets,
    const GenericPackedTensorAccessor<Dtype, 1, at::RestrictPtrTraits, int32_t>
        cat_ad_indices,
    GenericPackedTensorAccessor<Dtype, 1, at::RestrictPtrTraits, int32_t>
        reordered_cat_ad_indices,
    const int num_ads_in_batch,
    const int reordered_cat_ad_batches,
    const int subGroupSize) {
  auto item = syclext::this_work_item::get_nd_item<1>();
  const auto lane_id = item.get_local_id(0) % subGroupSize;
  const auto warp_id =
      (item.get_group(0) * item.get_local_range(0) + item.get_local_id(0)) /
      subGroupSize;
  const auto table_idx = warp_id / num_ads_in_batch;
  const auto ads_idx = warp_id % num_ads_in_batch;
  const auto start_offset = cat_ad_offsets[table_idx];
  const auto end_offset = cat_ad_offsets[table_idx + 1];
  const auto num_ads = end_offset - start_offset;
  if (warp_id < reordered_cat_ad_batches) {
    for (auto i = lane_id; i < num_ads; i += subGroupSize) {
      reordered_cat_ad_indices
          [start_offset * num_ads_in_batch + ads_idx * num_ads + i] =
              cat_ad_indices[start_offset + i];
    }
  }
}

template <typename Dtype, typename index_t = int32_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void narrow_batched_broadcast_indices_kernel_(
    const GenericPackedTensorAccessor<
        index_t,
        1,
        at::RestrictPtrTraits,
        int32_t> cat_ad_offsets,
    const GenericPackedTensorAccessor<Dtype, 1, at::RestrictPtrTraits, int32_t>
        cat_ad_indices,
    const GenericPackedTensorAccessor<
        index_t,
        1,
        at::RestrictPtrTraits,
        int32_t> reordered_cat_ad_offsets,
    GenericPackedTensorAccessor<Dtype, 1, at::RestrictPtrTraits, int32_t>
        reordered_cat_ad_indices,
    const GenericPackedTensorAccessor<
        int32_t,
        1,
        at::RestrictPtrTraits,
        int32_t> batch_offsets,
    const int32_t T,
    const int subGroupSize) {
  const auto B = batch_offsets.size(0) - 1;
  const auto num_ads_in_batch = static_cast<uint32_t>(batch_offsets[B]);
  // calculate table_id and batch_id for this warp
  auto item = syclext::this_work_item::get_nd_item<1>();
  const auto warp_id =
      (item.get_group(0) * item.get_local_range(0) + item.get_local_id(0)) /
      static_cast<uint32_t>(subGroupSize);
  const auto table_id = warp_id / num_ads_in_batch;
  const auto warp_id_in_table = warp_id % num_ads_in_batch;
  // warps in a table equally splited for each B
  const auto num_warp_in_batch = num_ads_in_batch / B;
  const auto batch_id = warp_id_in_table / num_warp_in_batch;
  if (table_id >= T || batch_id >= B) {
    return;
  }

  // all table_id and batch_id for this warp is the same
  const auto num_ads_b = batch_offsets[batch_id + 1] - batch_offsets[batch_id];
  const auto output_segment_offset_start =
      table_id * num_ads_in_batch + batch_offsets[batch_id];
  const auto output_segment_start =
      reordered_cat_ad_offsets[output_segment_offset_start];
  const auto input_segment_offset_start = T * batch_id + table_id;
  const auto input_segment_offset_end = input_segment_offset_start + 1;
  const auto input_segment_start = cat_ad_offsets[input_segment_offset_start];
  const auto input_segment_end = cat_ad_offsets[input_segment_offset_end];
  const auto num_elements = input_segment_end - input_segment_start;

  const auto warp_id_in_batch = warp_id_in_table % num_warp_in_batch;
  const auto lane_id_in_warp = item.get_local_id(0) % subGroupSize;
  for (auto i = warp_id_in_batch; i < num_ads_b; i += num_warp_in_batch) {
    for (auto j = lane_id_in_warp; j < num_elements; j += subGroupSize) {
      reordered_cat_ad_indices[output_segment_start + i * num_elements + j] =
          cat_ad_indices[input_segment_start + j];
    }
  }
}

template <typename Dtype, typename index_t = int32_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<2>))
void reorder_batched_ad_indices_kernel_(
    // reorder indices from (ragged) [B  x T x #num_ads_b x length_{b, t, a})]
    // to [T][B][#num_ads_b][length_{b, t, a}], i.e. [sum(length_{b, t, a})],
    // laid out as [T][B][A][L] (if all lengths were equal).

    // if broadcast_indices is enabled, all the indices will be copies of the
    // first batch of the cat_ad_indices, this is useful for request-only
    // broadcast
    const GenericPackedTensorAccessor<
        index_t,
        1,
        at::RestrictPtrTraits,
        int32_t> cat_ad_offsets,
    const GenericPackedTensorAccessor<Dtype, 1, at::RestrictPtrTraits, int32_t>
        cat_ad_indices,
    const GenericPackedTensorAccessor<
        index_t,
        1,
        at::RestrictPtrTraits,
        int32_t> reordered_cat_ad_offsets,
    GenericPackedTensorAccessor<Dtype, 1, at::RestrictPtrTraits, int32_t>
        reordered_cat_ad_indices,
    const GenericPackedTensorAccessor<
        int32_t,
        1,
        at::RestrictPtrTraits,
        int32_t> batch_offsets,
    const int32_t T,
    const bool broadcast_indices) {
  const int32_t B = batch_offsets.size(0) - 1;
  const int32_t num_ads_in_batch = batch_offsets[B];
  // warp-per-segment.
  auto item = syclext::this_work_item::get_nd_item<2>();
  const auto b_t =
      item.get_group(0) * item.get_local_range(1) + item.get_local_id(1);
  const int32_t b = b_t % B;
  const int32_t t = b_t / B;
  if (t >= T) {
    return;
  }

  const auto num_ads_b = batch_offsets[b + 1] - batch_offsets[b];
  const auto output_segment_offset_start =
      t * num_ads_in_batch + batch_offsets[b];
  const auto output_segment_start =
      reordered_cat_ad_offsets[output_segment_offset_start];
  const int32_t input_segment_offset_start =
      broadcast_indices ? T * b + t : T * batch_offsets[b] + t * num_ads_b;
  const int32_t input_segment_offset_end = broadcast_indices
      ? input_segment_offset_start + 1
      : input_segment_offset_start + num_ads_b;
  const auto input_segment_start = cat_ad_offsets[input_segment_offset_start];
  const auto input_segment_end = cat_ad_offsets[input_segment_offset_end];
  const auto num_elements = input_segment_end - input_segment_start;

  if (broadcast_indices) {
    for (auto i = item.get_local_id(0); i < num_ads_b * num_elements;
         i += item.get_local_range(0)) {
      reordered_cat_ad_indices[output_segment_start + i] =
          cat_ad_indices[input_segment_start + i % num_elements];
    }
  } else {
    // Idea: we want to copy the entire segment of size sum_a(length_{b, t, a})
    // from starting point (given by cat_ad_offsets[b, t])
    // to end point (given by reordered_cat_ad_indices[t][b])
    for (auto i = item.get_local_id(0);
         i < input_segment_end - input_segment_start;
         i += item.get_local_range(0)) {
      reordered_cat_ad_indices[output_segment_start + i] =
          cat_ad_indices[input_segment_start + i];
    }
  }
}

void reorder_batched_ad_indices_xpu_kernel(
    const at::Tensor& cat_ad_offsets,
    const at::Tensor& cat_ad_indices,
    const at::Tensor& reordered_cat_ad_offsets,
    const at::Tensor& batch_offsets,
    at::Tensor& reordered_cat_ad_indices,
    const int64_t num_ads_in_batch,
    const int64_t B,
    const int64_t T,
    const bool broadcast_indices) {
  const int subGroupSize = syclMaxSubGroupSize();
  if (broadcast_indices && T <= 320 && B < 64) {
    TORCH_CHECK(num_ads_in_batch * T == reordered_cat_ad_offsets.numel() - 1);
    if (B == 1) {
      // for B = 1 broadcast case
      constexpr auto NUM_WARPS = 16;
      const int workGroupSize = NUM_WARPS * subGroupSize;
      const int global_dim =
          xpu_calc_xblock_count(
              reordered_cat_ad_offsets.numel() - 1, NUM_WARPS) *
          workGroupSize;
      FBGEMM_DISPATCH_ALL_TYPES(
          cat_ad_indices.scalar_type(),
          "narrow_broadcast_indices_kernel_1",
          [&] {
            AT_DISPATCH_INDEX_TYPES(
                cat_ad_offsets.scalar_type(),
                "narrow_broadcast_indices_kernel_2",
                [&] {
                  sycl_kernel_submit<
                      narrow_broadcast_indices_kernel_<scalar_t, index_t>>(
                      sycl::range<1>(global_dim),
                      sycl::range<1>(workGroupSize),
                      getCurrentSYCLQueue(),
                      0,
                      cat_ad_offsets.packed_accessor32<
                          index_t,
                          1,
                          at::RestrictPtrTraits>(),
                      cat_ad_indices.packed_accessor32<
                          scalar_t,
                          1,
                          at::RestrictPtrTraits>(),
                      reordered_cat_ad_indices.packed_accessor32<
                          scalar_t,
                          1,
                          at::RestrictPtrTraits>(),
                      num_ads_in_batch,
                      reordered_cat_ad_offsets.numel() - 1,
                      subGroupSize);
                });
          });
      return;
    } else {
      // for B > 1 and B < 64 broadcast case
      constexpr auto NUM_WARPS = 16;
      const int workGroupSize = NUM_WARPS * subGroupSize;
      const int global_dim =
          xpu_calc_xblock_count(T * num_ads_in_batch, NUM_WARPS) *
          workGroupSize;
      FBGEMM_DISPATCH_ALL_TYPES(
          cat_ad_indices.scalar_type(),
          "narrow_batched_broadcast_indices_kernel_1",
          [&] {
            AT_DISPATCH_INDEX_TYPES(
                cat_ad_offsets.scalar_type(),
                "narrow_batched_broadcast_indices_kernel_2",
                [&] {
                  sycl_kernel_submit<narrow_batched_broadcast_indices_kernel_<
                      scalar_t,
                      index_t>>(
                      sycl::range<1>(global_dim),
                      sycl::range<1>(workGroupSize),
                      getCurrentSYCLQueue(),
                      0,
                      cat_ad_offsets.packed_accessor32<
                          index_t,
                          1,
                          at::RestrictPtrTraits>(),
                      cat_ad_indices.packed_accessor32<
                          scalar_t,
                          1,
                          at::RestrictPtrTraits>(),
                      reordered_cat_ad_offsets.packed_accessor32<
                          index_t,
                          1,
                          at::RestrictPtrTraits>(),
                      reordered_cat_ad_indices.packed_accessor32<
                          scalar_t,
                          1,
                          at::RestrictPtrTraits>(),
                      batch_offsets.packed_accessor32<
                          int32_t,
                          1,
                          at::RestrictPtrTraits>(),
                      T,
                      subGroupSize);
                });
          });
      return;
    }
  }
  FBGEMM_DISPATCH_ALL_TYPES(
      cat_ad_indices.scalar_type(),
      "reorder_batched_ad_indices_xpu_kernel_1",
      [&] {
        AT_DISPATCH_INDEX_TYPES(
            cat_ad_offsets.scalar_type(),
            "reorder_batched_ad_indices_xpu_kernel_2",
            [&] {
              constexpr auto NUM_WARPS = 32;
              const int maxWorkGroupSize = syclDeviceMaxWorkGroupSize();
              auto maxWarpSize = maxWorkGroupSize / NUM_WARPS;
              const int gloal_dim_y =
                  maxWarpSize < subGroupSize ? maxWarpSize : subGroupSize;
              const int global_dim_x =
                  xpu_calc_xblock_count(B * T, NUM_WARPS) * NUM_WARPS;
              sycl_kernel_submit<
                  reorder_batched_ad_indices_kernel_<scalar_t, index_t>>(
                  sycl::range<2>(global_dim_x, gloal_dim_y),
                  sycl::range<2>(NUM_WARPS, gloal_dim_y),
                  getCurrentSYCLQueue(),
                  0,
                  cat_ad_offsets
                      .packed_accessor32<index_t, 1, at::RestrictPtrTraits>(),
                  cat_ad_indices
                      .packed_accessor32<scalar_t, 1, at::RestrictPtrTraits>(),
                  reordered_cat_ad_offsets
                      .packed_accessor32<index_t, 1, at::RestrictPtrTraits>(),
                  reordered_cat_ad_indices
                      .packed_accessor32<scalar_t, 1, at::RestrictPtrTraits>(),
                  batch_offsets
                      .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                  T,
                  broadcast_indices);
            });
      });
}

// Kernel for permuting the lengths. Used for permutation of sparse features.
template <typename index_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void permute_2D_lengths_kernel_(
    int32_t T,
    int32_t B,
    const index_t* __restrict__ lengths,
    const int32_t* __restrict__ permute,
    index_t* __restrict__ permuted_lengths) {
  auto item = syclext::this_work_item::get_nd_item<1>();
  XPU_KERNEL_LOOP(item, b_t, B * T) {
    int32_t b = b_t % B;
    int32_t t = b_t / B;
    permuted_lengths[b_t] = lengths[permute[t] * B + b];
  }
}

void permute_2D_lengths_kernel_xpu(
    int32_t T,
    int32_t B,
    const at::Tensor& lengths_contig,
    const at::Tensor& permute_contig,
    at::Tensor& permuted_lengths) {
  constexpr int32_t threads_1 = 256;
  const auto blocks_1 = xpu_calc_xblock_count(B * T, threads_1);
  AT_DISPATCH_INDEX_TYPES(
      lengths_contig.scalar_type(), "permute_2D_lengths_kernel", [&] {
        sycl_kernel_submit<permute_2D_lengths_kernel_<index_t>>(
            sycl::range<1>(blocks_1 * threads_1),
            sycl::range<1>(threads_1),
            getCurrentSYCLQueue(),
            0,
            T,
            B,
            lengths_contig.data_ptr<index_t>(),
            permute_contig.data_ptr<int32_t>(),
            permuted_lengths.data_ptr<index_t>());
      });
}

template <
    bool has_weight,
    typename offsets_t,
    typename indices_t,
    typename weights_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<2>))
void permute_2D_data_kernel_(
    int32_t len,
    int32_t T,
    int32_t B,
    const indices_t* __restrict__ indices,
    const weights_t* __restrict__ weights,
    const int32_t weights_columns,
    const int32_t* __restrict__ permute,
    const offsets_t* __restrict__ input_offsets,
    const offsets_t* __restrict__ output_offsets,
    indices_t* __restrict__ permuted_indices,
    weights_t* __restrict__ permuted_weights) {
  auto item = syclext::this_work_item::get_nd_item<2>();
  auto b_t_start =
      item.get_group(0) * item.get_local_range(1) + item.get_local_id(1);
  const auto stride = item.get_group_range(0) * item.get_local_range(1);
  for (int b_t = b_t_start; b_t < B * T; b_t += stride) {
    int32_t b = b_t % B;
    int32_t t = b_t / B;
    offsets_t output_start = output_offsets[b_t];
    offsets_t segment_length;
    if (b_t == B * T - 1) {
      segment_length = len - output_offsets[b_t];
    } else {
      segment_length = output_offsets[b_t + 1] - output_offsets[b_t];
    }
    offsets_t input_start = input_offsets[permute[t] * B + b];
    for (auto i = item.get_local_id(0); i < segment_length;
         i += item.get_local_range(0)) {
      permuted_indices[output_start + i] = indices[input_start + i];
      if (has_weight) {
        for (auto w_col = 0; w_col < weights_columns; ++w_col) {
          permuted_weights[(output_start + i) * weights_columns + w_col] =
              weights[(input_start + i) * weights_columns + w_col];
        }
      }
    }
  }
}

void permute_2D_data_kernel_xpu(
    int32_t permuted_indices_size,
    int32_t T,
    int32_t B,
    const Tensor& indices_contig,
    const std::optional<const Tensor>& weights,
    const int32_t weights_columns,
    const Tensor& permute_contig,
    const Tensor& input_offsets,
    const Tensor& output_offsets,
    Tensor& permuted_indices,
    const std::optional<Tensor>& permuted_weights) {
  constexpr int32_t BT_blocks = 32;
  const auto blocks_2 = xpu_calc_xblock_count(B * T, BT_blocks);
  AT_DISPATCH_INDEX_TYPES(
      input_offsets.scalar_type(), "permute_2D_data_kernel_1", [&] {
        using offsets_t = index_t;
        FBGEMM_DISPATCH_ALL_TYPES(
            indices_contig.scalar_type(), "permute_2D_data_kernel_2", [&] {
              using indices_t = scalar_t;
              if (weights.has_value()) {
                const auto weights_value_contig = weights.value().contiguous();
                FBGEMM_DISPATCH_ALL_TYPES_AND_DOUBLE(
                    weights_value_contig.scalar_type(),
                    "permute_2D_data_kernel_3",
                    [&] {
                      using weights_t = scalar_t;
                      sycl_kernel_submit<permute_2D_data_kernel_<
                          true,
                          offsets_t,
                          indices_t,
                          weights_t>>(
                          sycl::range<2>(blocks_2 * 32, BT_blocks),
                          sycl::range<2>(32, BT_blocks),
                          getCurrentSYCLQueue(),
                          0,
                          permuted_indices_size,
                          T,
                          B,
                          indices_contig.data_ptr<indices_t>(),
                          weights_value_contig.data_ptr<weights_t>(),
                          weights_columns,
                          permute_contig.data_ptr<int32_t>(),
                          input_offsets.data_ptr<offsets_t>(),
                          output_offsets.data_ptr<offsets_t>(),
                          permuted_indices.data_ptr<indices_t>(),
                          permuted_weights.value().data_ptr<weights_t>());
                    }); // for each weights_t
              } else {
                sycl_kernel_submit<permute_2D_data_kernel_<
                    false,
                    offsets_t,
                    indices_t,
                    float>>( // false float type here as wa since
                             // std::nullptr_t cannot be
                             // supported in free function
                             // kernel
                    sycl::range<2>(blocks_2 * 32, BT_blocks),
                    sycl::range<2>(32, BT_blocks),
                    getCurrentSYCLQueue(),
                    0,
                    permuted_indices_size,
                    T,
                    B,
                    indices_contig.data_ptr<indices_t>(),
                    nullptr,
                    0,
                    permute_contig.data_ptr<int32_t>(),
                    input_offsets.data_ptr<offsets_t>(),
                    output_offsets.data_ptr<offsets_t>(),
                    permuted_indices.data_ptr<indices_t>(),
                    nullptr);
              }
            }); // for each indices_t
      }); // for each offsets_t
}

#undef INVOKE_KERNEL_WITH_DIM

} // namespace xpu
} // namespace native
} // namespace at

