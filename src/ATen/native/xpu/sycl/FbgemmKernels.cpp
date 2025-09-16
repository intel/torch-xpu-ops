#include <ATen/ATen.h>
#include <ATen/native/xpu/sycl/FbgemmKernels.h>

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

template <int NUM_JAGGED_DIM, typename index_t, typename scalar_t>
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
    StackArray<int64_t> jagged_dims) {
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

    auto f = [](scalar_t x, scalar_t y, scalar_t z) -> scalar_t { return y; };

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

template <int NUM_JAGGED_DIM, typename index_t, typename scalar_t>
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
    int64_t wg_0,
    int64_t wg_1,
    int64_t wg_num) {
  sycl_kernel_submit<jagged_dense_elementwise_jagged_output_kernel_<
      NUM_JAGGED_DIM,
      index_t,
      scalar_t>>(
      sycl::range<2>(wg_0 * wg_num, wg_1),
      sycl::range<2>(wg_0, wg_1),
      getCurrentSYCLQueue(),
      x_values,
      x_offsets,
      x_offsets_sizes,
      y_0,
      y_1,
      output_values,
      jagged_dims);
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
      CeilDiv(outer_dense_size * jagged_folded_size, (int32_t)wg_size_1);

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

template <typename scalar_t>
void jagged_dense_elementwise_jagged_output_(
    const Tensor& x_values,
    const std::vector<Tensor>& x_offsets,
    const Tensor& y,
    const Tensor& output_values) {
  // Canonicalize y to 3D, collapsing jagged dimensions.
  const int num_jagged_dim = y.dim() - 2;
  const Tensor y_reshaped = y.view({y.size(0), -1, y.size(-1)});
#define INVOKE_KERNEL_WITH_DIM(NUM_JAGGED_DIM)                               \
  {                                                                          \
    int64_t wg_0, wg_1, wg_num;                                              \
    StackArray<int64_t> jagged_dims_tensor;                                  \
    std::tie(wg_0, wg_1, wg_num, jagged_dims_tensor) =                       \
        check_shape_and_partition_(x_values, x_offsets, y);                  \
    wg_num = CeilDiv(x_values.size(0), wg_1);                                \
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
            x_values, x_offsets, y, output_values);
      });
}

template <int NUM_JAGGED_DIM, typename index_t, typename scalar_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<2>))
void jagged_dense_elementwise_dense_output_kernel_(
    GenericPackedTensorAccessor<scalar_t, 2, DefaultPtrTraits, int32_t>
        x_values,
    StackArray<index_t*> x_offsets,
    GenericPackedTensorAccessor<scalar_t, 3, DefaultPtrTraits, int32_t> y,
    GenericPackedTensorAccessor<scalar_t, 3, DefaultPtrTraits, int32_t> output,
    StackArray<int64_t> jagged_dims,
    const scalar_t padding_value) {
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

    auto f = [](scalar_t x, scalar_t y) -> scalar_t { return x; };

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
    GenericPackedTensorAccessor<scalar_t, 2, DefaultPtrTraits, int32_t>
        x_values,
    StackArray<index_t*> x_offsets,
    GenericPackedTensorAccessor<scalar_t, 3, DefaultPtrTraits, int32_t> y,
    GenericPackedTensorAccessor<scalar_t, 3, DefaultPtrTraits, int32_t> output,
    StackArray<int64_t> jagged_dims,
    const scalar_t padding_value,
    int64_t wg_0,
    int64_t wg_1,
    int64_t wg_num) {
  sycl_kernel_submit<jagged_dense_elementwise_dense_output_kernel_<
      NUM_JAGGED_DIM,
      index_t,
      scalar_t>>(
      sycl::range<2>(wg_0 * wg_num, wg_1),
      sycl::range<2>(wg_0, wg_1),
      getCurrentSYCLQueue(),
      x_values,
      x_offsets,
      y,
      output,
      jagged_dims,
      padding_value);
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

} // namespace xpu
} // namespace native
} // namespace at
