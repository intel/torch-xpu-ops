#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/NestedTensorImpl.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/Array.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/TensorAccessor.h>
#include <ATen/native/nested/NestedTensorUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/narrow_native.h>
#endif

#include <ATen/native/transformers/sycl/AttentionKernels.h>
#include <comm/SYCLContext.h>

namespace at::native::xpu {

static constexpr int TRANSFORM_BIAS_RESCALE_VEC = 4;

template <typename scalar_t, typename accscalar_t, int VEC>
struct TransformBiasRescaleQKVAddPaddingFunctor {
  void operator()(sycl::nd_item<1> item) const {
    const auto NH = q_k_v_.size(2);
    const auto T = q_k_v_.size(3);
    const auto DH = q_k_v_.size(4);

    const auto stride_0 = q_k_v_.stride(0);
    const auto stride_1 = q_k_v_.stride(1);
    const auto stride_2 = q_k_v_.stride(2);
    const auto stride_3 = q_k_v_.stride(3);
    const auto stride_4 = q_k_v_.stride(4);
    scalar_t* const data = q_k_v_.data();

    const int32_t local_id = item.get_local_id(0);
    const int32_t global_id = item.get_group(0);

    const auto t = global_id % T;
    const auto b = global_id / T;

    const auto D = NH * DH;
    const auto _3D = 3 * D;

    const auto offset_for_batch = offsets_[b];
    const auto input_dim = 1;
    const auto* sizes_i = input_sizes_ + b * input_dim;

    if (assume_aligned_) {
      using LoadT = at::detail::Array<scalar_t, VEC>;
      for (int32_t d_v = local_id; d_v < D / VEC;
           d_v += item.get_local_range(0)) {
        auto d = d_v * VEC;
        auto nh = d / DH;
        auto dh = d % DH;
        scalar_t qkv_bias_q[VEC];
        scalar_t qkv_bias_k[VEC];
        scalar_t qkv_bias_v[VEC];
        scalar_t qkv_q[VEC];
        scalar_t qkv_k[VEC];
        scalar_t qkv_v[VEC];

        const auto first_item_offset = t * _3D + d;
        const auto last_item_offset = first_item_offset + VEC - 1;
        const bool first_item_in_bounds = first_item_offset < sizes_i[0];
        const bool entire_vec_in_bounds = last_item_offset < sizes_i[0];

        // Here we require D % VEC == 0 for these vectorized loads.
        *reinterpret_cast<LoadT*>(&qkv_bias_q) =
            *reinterpret_cast<const LoadT*>(&qkv_bias_[d + 0 * D]);
        *reinterpret_cast<LoadT*>(&qkv_bias_k) =
            *reinterpret_cast<const LoadT*>(&qkv_bias_[d + 1 * D]);
        *reinterpret_cast<LoadT*>(&qkv_bias_v) =
            *reinterpret_cast<const LoadT*>(&qkv_bias_[d + 2 * D]);

        if (entire_vec_in_bounds) {
          const auto offset = offset_for_batch + first_item_offset;
          *reinterpret_cast<LoadT*>(&qkv_q) =
              *reinterpret_cast<const LoadT*>(&qkv_[offset + 0 * D]);
          *reinterpret_cast<LoadT*>(&qkv_k) =
              *reinterpret_cast<const LoadT*>(&qkv_[offset + 1 * D]);
          *reinterpret_cast<LoadT*>(&qkv_v) =
              *reinterpret_cast<const LoadT*>(&qkv_[offset + 2 * D]);

#pragma unroll
          for (auto ii = 0; ii < VEC; ++ii) {
            qkv_q[ii] = static_cast<scalar_t>(
                (static_cast<accscalar_t>(qkv_q[ii]) +
                 static_cast<accscalar_t>(qkv_bias_q[ii])) *
                static_cast<accscalar_t>(inv_sqrt_dim_per_head_));
            qkv_k[ii] = static_cast<scalar_t>(
                (static_cast<accscalar_t>(qkv_k[ii]) +
                 static_cast<accscalar_t>(qkv_bias_k[ii])));
            qkv_v[ii] = static_cast<scalar_t>(
                (static_cast<accscalar_t>(qkv_v[ii]) +
                 static_cast<accscalar_t>(qkv_bias_v[ii])));
          }
        } else if (first_item_in_bounds) {
          const auto offset = offset_for_batch + first_item_offset;
          qkv_q[0] = qkv_[offset + 0 * D];
          qkv_k[0] = qkv_[offset + 1 * D];
          qkv_v[0] = qkv_[offset + 2 * D];
          qkv_q[0] = static_cast<scalar_t>(
              (static_cast<accscalar_t>(qkv_q[0]) +
               static_cast<accscalar_t>(qkv_bias_q[0])) *
              static_cast<accscalar_t>(inv_sqrt_dim_per_head_));
          qkv_k[0] = static_cast<scalar_t>(
              (static_cast<accscalar_t>(qkv_k[0]) +
               static_cast<accscalar_t>(qkv_bias_k[0])));
          qkv_v[0] = static_cast<scalar_t>(
              (static_cast<accscalar_t>(qkv_v[0]) +
               static_cast<accscalar_t>(qkv_bias_v[0])));
#pragma unroll
          for (auto ii = 1; ii < VEC; ++ii) {
            const auto loop_offset = offset + ii;
            if (loop_offset < sizes_i[0]) {
              qkv_q[ii] = qkv_[loop_offset + 0 * D];
              qkv_k[ii] = qkv_[loop_offset + 1 * D];
              qkv_v[ii] = qkv_[loop_offset + 2 * D];
              qkv_q[ii] = static_cast<scalar_t>(
                  (static_cast<accscalar_t>(qkv_q[ii]) +
                   static_cast<accscalar_t>(qkv_bias_q[ii])) *
                  static_cast<accscalar_t>(inv_sqrt_dim_per_head_));
              qkv_k[ii] = static_cast<scalar_t>(
                  (static_cast<accscalar_t>(qkv_k[ii]) +
                   static_cast<accscalar_t>(qkv_bias_k[ii])));
              qkv_v[ii] = static_cast<scalar_t>(
                  (static_cast<accscalar_t>(qkv_v[ii]) +
                   static_cast<accscalar_t>(qkv_bias_v[ii])));
            } else {
              qkv_q[ii] = 0;
              qkv_k[ii] = 0;
              qkv_v[ii] = 0;
            }
          }
        } else {
#pragma unroll
          for (auto ii = 0; ii < VEC; ++ii) {
            qkv_q[ii] = 0;
            qkv_k[ii] = 0;
            qkv_v[ii] = 0;
          }
        }

        // Here we require DH % VEC == 0 for these vectorized stores.
        *reinterpret_cast<LoadT*>(
            &data
                [0 * stride_0 + b * stride_1 + nh * stride_2 + t * stride_3 +
                 dh * stride_4]) = *reinterpret_cast<const LoadT*>(&qkv_q);
        *reinterpret_cast<LoadT*>(
            &data
                [1 * stride_0 + b * stride_1 + nh * stride_2 + t * stride_3 +
                 dh * stride_4]) = *reinterpret_cast<const LoadT*>(&qkv_k);
        *reinterpret_cast<LoadT*>(
            &data
                [2 * stride_0 + b * stride_1 + nh * stride_2 + t * stride_3 +
                 dh * stride_4]) = *reinterpret_cast<const LoadT*>(&qkv_v);
      }
    } else {
      for (int32_t d = local_id; d < D; d += item.get_local_range(0)) {
        auto nh = d / DH;
        auto dh = d % DH;
        scalar_t qkv_bias_q = qkv_bias_[d + 0 * D];
        scalar_t qkv_bias_k = qkv_bias_[d + 1 * D];
        scalar_t qkv_bias_v = qkv_bias_[d + 2 * D];

        const auto item_offset = t * _3D + d;
        const bool in_bounds = item_offset < sizes_i[0];
        scalar_t qkv_q, qkv_k, qkv_v;

        if (in_bounds) {
          const auto qkv_offset = offset_for_batch + item_offset;
          qkv_q = qkv_[qkv_offset + 0 * D];
          qkv_k = qkv_[qkv_offset + 1 * D];
          qkv_v = qkv_[qkv_offset + 2 * D];
          qkv_q = static_cast<scalar_t>(
              (static_cast<accscalar_t>(qkv_q) +
               static_cast<accscalar_t>(qkv_bias_q)) *
              static_cast<accscalar_t>(inv_sqrt_dim_per_head_));
          qkv_k = static_cast<scalar_t>(
              (static_cast<accscalar_t>(qkv_k) +
               static_cast<accscalar_t>(qkv_bias_k)));
          qkv_v = static_cast<scalar_t>(
              (static_cast<accscalar_t>(qkv_v) +
               static_cast<accscalar_t>(qkv_bias_v)));
        } else {
          qkv_q = 0;
          qkv_k = 0;
          qkv_v = 0;
        }

        data
            [0 * stride_0 + b * stride_1 + nh * stride_2 + t * stride_3 +
             dh * stride_4] = qkv_q;
        data
            [1 * stride_0 + b * stride_1 + nh * stride_2 + t * stride_3 +
             dh * stride_4] = qkv_k;
        data
            [2 * stride_0 + b * stride_1 + nh * stride_2 + t * stride_3 +
             dh * stride_4] = qkv_v;
      }
    }
  }

  TransformBiasRescaleQKVAddPaddingFunctor(
      const PackedTensorAccessor64<scalar_t, 1> qkv,
      const PackedTensorAccessor64<scalar_t, 1> qkv_bias,
      const int* offsets,
      const int* input_sizes,
      PackedTensorAccessor64<scalar_t, 5> q_k_v,
      const scalar_t inv_sqrt_dim_per_head,
      const bool assume_aligned)
      : qkv_(qkv),
        qkv_bias_(qkv_bias),
        offsets_(offsets),
        input_sizes_(input_sizes),
        q_k_v_(q_k_v),
        inv_sqrt_dim_per_head_(inv_sqrt_dim_per_head),
        assume_aligned_(assume_aligned) {}

 private:
  // [B, T, 3 * D], but it's a NestedTensor buffer
  const PackedTensorAccessor64<scalar_t, 1> qkv_;
  // [3 * D]
  const PackedTensorAccessor64<scalar_t, 1> qkv_bias_;
  const int* offsets_;
  const int* input_sizes_;
  // [3, B, NH, T, DH]
  PackedTensorAccessor64<scalar_t, 5> q_k_v_;
  const scalar_t inv_sqrt_dim_per_head_;
  const bool assume_aligned_;
};

// each nd_range is dealing with one qkv tensor, each nd_group is dealing with D
// dim
template <typename scalar_t, typename accscalar_t, int VEC>
struct TransformBiasRescaleQKVKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    auto NH = q_k_v_.size(2);
    auto T = q_k_v_.size(3);
    auto DH = q_k_v_.size(4);

    const auto qkv_stride_0 = q_k_v_.stride(0);
    const auto qkv_stride_1 = q_k_v_.stride(1);
    const auto qkv_stride_2 = q_k_v_.stride(2);
    const auto qkv_stride_3 = q_k_v_.stride(3);
    const auto qkv_stride_4 = q_k_v_.stride(4);
    scalar_t* const qkv_data = q_k_v_.data();

    const auto group_id = item.get_group(0);
    const auto local_id = item.get_local_id(0);
    const auto local_range = item.get_local_range(0);

    auto t = group_id % T;
    auto b = group_id / T;

    const auto D = NH * DH;

    if (assume_aligned_) {
      using LoadT = at::detail::Array<scalar_t, VEC>;
      // here is aligned, no more need ceiling for D / VEC
      for (int32_t d_v = local_id; d_v < D / VEC; d_v += local_range) {
        auto d = d_v * VEC;
        auto nh = d / DH;
        auto dh = d % DH;
        scalar_t qkv_bias_q[VEC];
        scalar_t qkv_bias_k[VEC];
        scalar_t qkv_bias_v[VEC];
        scalar_t qkv_q[VEC];
        scalar_t qkv_k[VEC];
        scalar_t qkv_v[VEC];

        *reinterpret_cast<LoadT*>(&qkv_bias_q) =
            *reinterpret_cast<const LoadT*>(&qkv_bias_[d + 0 * D]);
        *reinterpret_cast<LoadT*>(&qkv_bias_k) =
            *reinterpret_cast<const LoadT*>(&qkv_bias_[d + 1 * D]);
        *reinterpret_cast<LoadT*>(&qkv_bias_v) =
            *reinterpret_cast<const LoadT*>(&qkv_bias_[d + 2 * D]);

        *reinterpret_cast<LoadT*>(&qkv_q) =
            *reinterpret_cast<const LoadT*>(&qkv_[b][t][d + 0 * D]);
        *reinterpret_cast<LoadT*>(&qkv_k) =
            *reinterpret_cast<const LoadT*>(&qkv_[b][t][d + 1 * D]);
        *reinterpret_cast<LoadT*>(&qkv_v) =
            *reinterpret_cast<const LoadT*>(&qkv_[b][t][d + 2 * D]);

#pragma unroll
        for (auto ii = 0; ii < VEC; ii++) {
          qkv_q[ii] = static_cast<scalar_t>(
              (static_cast<accscalar_t>(qkv_q[ii]) +
               static_cast<accscalar_t>(qkv_bias_q[ii])) *
              static_cast<accscalar_t>(inv_sqrt_dim_per_head_));
          qkv_k[ii] = static_cast<scalar_t>(
              (static_cast<accscalar_t>(qkv_k[ii]) +
               static_cast<accscalar_t>(qkv_bias_k[ii])));
          qkv_v[ii] = static_cast<scalar_t>(
              (static_cast<accscalar_t>(qkv_v[ii]) +
               static_cast<accscalar_t>(qkv_bias_v[ii])));
        }

        // Here we require DH % VEC == 0 for these vectorized stores.
        *reinterpret_cast<LoadT*>(
            &qkv_data
                [0 * qkv_stride_0 + b * qkv_stride_1 + nh * qkv_stride_2 +
                 t * qkv_stride_3 + dh * qkv_stride_4]) =
            *reinterpret_cast<const LoadT*>(&qkv_q);
        *reinterpret_cast<LoadT*>(
            &qkv_data
                [1 * qkv_stride_0 + b * qkv_stride_1 + nh * qkv_stride_2 +
                 t * qkv_stride_3 + dh * qkv_stride_4]) =
            *reinterpret_cast<const LoadT*>(&qkv_k);
        *reinterpret_cast<LoadT*>(
            &qkv_data
                [2 * qkv_stride_0 + b * qkv_stride_1 + nh * qkv_stride_2 +
                 t * qkv_stride_3 + dh * qkv_stride_4]) =
            *reinterpret_cast<const LoadT*>(&qkv_v);
      }
    } else {
      // without vectorize load and store
      for (int32_t d = local_id; d < D; d += local_range) {
        auto nh = d / DH;
        auto dh = d % DH;
        scalar_t qkv_bias_q = qkv_bias_[0 * D + d];
        scalar_t qkv_bias_k = qkv_bias_[1 * D + d];
        scalar_t qkv_bias_v = qkv_bias_[2 * D + d];
        scalar_t qkv_q = qkv_[b][t][0 * D + d];
        scalar_t qkv_k = qkv_[b][t][1 * D + d];
        scalar_t qkv_v = qkv_[b][t][2 * D + d];

        qkv_q = static_cast<scalar_t>(
            (static_cast<accscalar_t>(qkv_q) +
             static_cast<accscalar_t>(qkv_bias_q)) *
            static_cast<accscalar_t>(inv_sqrt_dim_per_head_));
        qkv_k = static_cast<scalar_t>(
            static_cast<accscalar_t>(qkv_k) +
            static_cast<accscalar_t>(qkv_bias_k));
        qkv_v = static_cast<scalar_t>(
            static_cast<accscalar_t>(qkv_v) +
            static_cast<accscalar_t>(qkv_bias_v));

        qkv_data
            [0 * qkv_stride_0 + b * qkv_stride_1 + nh * qkv_stride_2 +
             t * qkv_stride_3 + dh * qkv_stride_4] = qkv_q;
        qkv_data
            [1 * qkv_stride_0 + b * qkv_stride_1 + nh * qkv_stride_2 +
             t * qkv_stride_3 + dh * qkv_stride_4] = qkv_k;
        qkv_data
            [2 * qkv_stride_0 + b * qkv_stride_1 + nh * qkv_stride_2 +
             t * qkv_stride_3 + dh * qkv_stride_4] = qkv_v;
      }
    }
  }

  TransformBiasRescaleQKVKernelFunctor(
      const PackedTensorAccessor64<scalar_t, 3> qkv,
      const PackedTensorAccessor64<scalar_t, 1> qkv_bias,
      PackedTensorAccessor64<scalar_t, 5> q_k_v,
      const scalar_t inv_sqrt_dim_per_head,
      const bool assume_aligned)
      : qkv_(qkv),
        qkv_bias_(qkv_bias),
        q_k_v_(q_k_v),
        inv_sqrt_dim_per_head_(inv_sqrt_dim_per_head),
        assume_aligned_(assume_aligned) {}

 private:
  // [B, T, 3 * D]
  const PackedTensorAccessor64<scalar_t, 3> qkv_;
  // [3 * D]
  const PackedTensorAccessor64<scalar_t, 1> qkv_bias_;
  // [3, B, num_heads, T, dim_per_head]
  PackedTensorAccessor64<scalar_t, 5> q_k_v_;
  const scalar_t inv_sqrt_dim_per_head_;
  const bool assume_aligned_;
};

static Tensor collapse_dims_1_and_2(const Tensor& sizes) {
  auto sizes_dim1 = at::native::narrow_symint(sizes, 1, 0, 1);
  auto sizes_dim2 = at::native::narrow_symint(sizes, 1, 1, 1);

  return (sizes_dim1 * sizes_dim2).contiguous();
}

static Tensor NestedTensor_batch_offsets_from_size_tensor(
    const Tensor& sizes,
    int64_t extra_elements) {
  int64_t* const sizes_ptr = sizes.data_ptr<int64_t>();
  Tensor offsets = at::empty({1 + sizes.size(0) + extra_elements}, at::kInt);
  int32_t* const offsets_ptr = offsets.mutable_data_ptr<int32_t>();
  offsets_ptr[0] = 0;
  const auto sizes_size_1 = sizes.size(1);
  const auto sizes_size_0 = sizes.size(0);
  for (const auto i : c10::irange(sizes_size_0)) {
    int64_t prod = 1;
    for (const auto j : c10::irange(sizes_size_1)) {
      prod *= sizes_ptr[i * sizes_size_1 + j];
    }
    offsets_ptr[i + 1] = offsets_ptr[i] + prod;
  }
  return offsets;
}

void _transform_bias_rescale_qkv_kernel(
    const Tensor& qkv,
    const Tensor& qkv_bias,
    const int64_t num_head,
    Tensor& q_k_v,
    int64_t B,
    int64_t T,
    int64_t D,
    int64_t dim_per_head) {
  auto max_wg_size = syclDeviceMaxWorkGroupSize();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      qkv.scalar_type(),
      "transform_bias_rescale_qkv_xpu",
      [&] {
        using accscalar_t = at::acc_type<scalar_t, true>;
        auto local_range = std::max(
            std::min<int32_t>(
                max_wg_size,
                (D + TRANSFORM_BIAS_RESCALE_VEC - 1) /
                    TRANSFORM_BIAS_RESCALE_VEC),
            1);
        auto global_range = B * T;
        const bool aligned =
            ((dim_per_head % TRANSFORM_BIAS_RESCALE_VEC) == 0) &&
            ((reinterpret_cast<intptr_t>(qkv_bias.data_ptr()) %
              TRANSFORM_BIAS_RESCALE_VEC) == 0);

        if (aligned) {
          TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
              D % TRANSFORM_BIAS_RESCALE_VEC == 0,
              "D = num_heads * dim_per_head, so we should have dim_per_head %"
              "TRANSFORM_BIAS_RESCALE_VEC == 0 => "
              "D %TRANSFORM_BIAS_RESCALE_VEC == 0");
        }

        if (qkv.is_nested()) {
          auto* nt_qkv = native::get_nested_tensor_impl(qkv);
          const Tensor& nt_qkv_buffer = nt_qkv->get_buffer();

          auto sizes = collapse_dims_1_and_2(nt_qkv->get_nested_sizes());
          auto offsets =
              NestedTensor_batch_offsets_from_size_tensor(sizes, sizes.numel());

          native::narrow_symint(offsets, 0, sizes.numel() + 1, sizes.numel())
              .copy_(sizes.reshape({-1}));
          auto metadata = offsets.to(at::Device(kXPU), at::kInt, true, true);
          const auto offsets_ptr = metadata.data_ptr<int>();
          const auto sizes_ptr = offsets_ptr + sizes.numel() + 1;
          // const auto input_dim = sizes.sizes()[1];
          auto qkv_acc = q_k_v.packed_accessor64<scalar_t, 5>();
          if (aligned &&
              ((reinterpret_cast<intptr_t>(qkv.data_ptr()) %
                TRANSFORM_BIAS_RESCALE_VEC) == 0)) {
            TransformBiasRescaleQKVAddPaddingFunctor<
                scalar_t,
                accscalar_t,
                TRANSFORM_BIAS_RESCALE_VEC>
                kfn(nt_qkv_buffer.packed_accessor64<scalar_t, 1>(),
                    qkv_bias.packed_accessor64<scalar_t, 1>(),
                    offsets_ptr,
                    sizes_ptr,
                    qkv_acc,
                    1.0 / std::sqrt(static_cast<scalar_t>(dim_per_head)),
                    true);
            sycl_kernel_submit(
                global_range * local_range,
                local_range,
                getCurrentSYCLQueue(),
                kfn);
          } else {
            TransformBiasRescaleQKVAddPaddingFunctor<
                scalar_t,
                accscalar_t,
                TRANSFORM_BIAS_RESCALE_VEC>
                kfn(nt_qkv_buffer.packed_accessor64<scalar_t, 1>(),
                    qkv_bias.packed_accessor64<scalar_t, 1>(),
                    offsets_ptr,
                    sizes_ptr,
                    qkv_acc,
                    1.0 / std::sqrt(static_cast<scalar_t>(dim_per_head)),
                    false);

            sycl_kernel_submit(
                global_range * local_range,
                local_range,
                getCurrentSYCLQueue(),
                kfn);
          }
        } else {
          TransformBiasRescaleQKVKernelFunctor<
              scalar_t,
              accscalar_t,
              TRANSFORM_BIAS_RESCALE_VEC>
              kfn(qkv.packed_accessor64<scalar_t, 3>(),
                  qkv_bias.packed_accessor64<scalar_t, 1>(),
                  q_k_v.packed_accessor64<scalar_t, 5>(),
                  1.0 / std::sqrt(static_cast<scalar_t>(dim_per_head)),
                  aligned);

          sycl_kernel_submit(
              global_range * local_range,
              local_range,
              getCurrentSYCLQueue(),
              kfn);
        }
      });
}

} // namespace at::native::xpu
