#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/NestedTensorImpl.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/Array.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/TensorAccessor.h>
#include <ATen/native/nested/NestedTensorUtils.h>
#include <ATen/native/transformers/attention.h>
#include <ATen/native/transformers/sdp_utils_cpp.h>
#include <ATen/xpu/XPUNativeFunctions.h>

#include <ATen/native/transformers/xpu/sdp_utils.h>

#include <comm/SYCLContext.h>

namespace at {

static constexpr int TRANSFORM_BIAS_RESCALE_VEC = 4;

template <typename scalar_t, typename accscalar_t>
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
      constexpr int VEC = TRANSFORM_BIAS_RESCALE_VEC;
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
template <typename scalar_t, typename accscalar_t>
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
      constexpr int VEC = TRANSFORM_BIAS_RESCALE_VEC;
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

// compute q = (q + q_bias) / sqrt(dim_per_head), k = k + k_bias, v = v + v_bias
// Note: current only support contiguous indexing, since nested tensor is all
// contiguous
std::tuple<Tensor, Tensor, Tensor> XPUNativeFunctions::
    _transform_bias_rescale_qkv(
        const Tensor& qkv,
        const Tensor& qkv_bias,
        const int64_t num_head) {
  // for nested tensor, B is most outer size, but T is not regular, it should be
  // the large size on dim1
  auto B = qkv.is_nested()
      ? native::get_nested_tensor_impl(qkv)->get_nested_sizes().size(0)
      : qkv.size(0);

  auto T = qkv.is_nested() ? native::NestedTensor_get_max_size(
                                 *native::get_nested_tensor_impl(qkv))[0]
                           : qkv.size(1);

  // qkv_bias size should be same with finall linear projection layer, which
  // size is 3 * D
  auto _3D = qkv_bias.size(0);
  auto D = _3D / 3;
  TORCH_CHECK(D % num_head == 0);
  const auto dim_per_head = D / num_head;

  // q_k_v B T 3D -> 3, B, num_head, T, dim_per_head
  auto q_k_v = at::empty({3, B, num_head, T, dim_per_head}, qkv_bias.options());

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
            TransformBiasRescaleQKVAddPaddingFunctor<scalar_t, accscalar_t> kfn(
                nt_qkv_buffer.packed_accessor64<scalar_t, 1>(),
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
            TransformBiasRescaleQKVAddPaddingFunctor<scalar_t, accscalar_t> kfn(
                nt_qkv_buffer.packed_accessor64<scalar_t, 1>(),
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
          TransformBiasRescaleQKVKernelFunctor<scalar_t, accscalar_t> kfn(
              qkv.packed_accessor64<scalar_t, 3>(),
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

  auto q_k_v_s =
      at::native::split(q_k_v.view({3 * B, num_head, T, dim_per_head}), B, 0);
  return std::make_tuple(q_k_v_s[0], q_k_v_s[1], q_k_v_s[2]);
}

static bool check_for_seq_len_1_nested_tensor(
    sdp::sdp_params params,
    bool debug) {
  // When this function is called we are assured that the nt is dim==4
  if (!params.query.is_nested()) {
    return true;
  }

  const auto nt_q_tensor_impl =
      at::native::get_nested_tensor_impl(params.query);
  const at::Tensor& sizes = nt_q_tensor_impl->get_nested_sizes();
  auto* sizes_ptr = sizes.data_ptr<int64_t>();
  const int64_t n_tensors = params.query.size(0);
  const int64_t size_tensor_stride = sizes.stride(0);

  // This is being called inside sdp with shape [batch, heads, {seq_len}, dim]
  for (const auto i : c10::irange(n_tensors)) {
    if (sizes_ptr[(i * size_tensor_stride) + 1] <= 1) {
      if (debug) {
        TORCH_WARN(
            "Packed projection for fused kernels does not support sequence_length <= 1");
      }
      return false;
    }
  }

  return true;
}

int64_t XPUNativeFunctions::_fused_sdp_choice(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const std::optional<Tensor>& attn_mask_,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale,
    bool enable_gqa) {
  // We have implemented efficient_attention backend with xetla, flash_attention
  // backend is not supported now, which will be implemented in the future. So
  // we provide two backends here.
  sdp::sdp_params kernel_params{
      query, key, value, attn_mask_, dropout_p, is_causal, enable_gqa};
  // Because TORCHCHECK checks if condition is true we negate debug so that
  // The statements will be printed when debug is true
  bool print_debug = false;
  sdp::SDPBackend backend =
      sdp::use_mem_efficient_attention(kernel_params, print_debug)
      ? sdp::SDPBackend::efficient_attention
      : sdp::SDPBackend::math;
  if (backend == sdp::SDPBackend::error) {
    TORCH_CHECK(
        false,
        "No viable backend for scaled_dot_product_attention was found. ",
        "This is likely due to turning off both the math kernel and the fused kernels.");
  }
  return static_cast<int64_t>(backend);
}

std::tuple<Tensor, Tensor> XPUNativeFunctions::_native_multi_head_attention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const int64_t embed_dim,
    const int64_t num_head,
    const Tensor& qkv_weight,
    const Tensor& qkv_bias,
    const Tensor& proj_weight,
    const Tensor& proj_bias,
    const std::optional<Tensor>& mask,
    bool need_weights,
    bool average_attn_weights,
    const std::optional<int64_t> mask_type) {
  // query shape: [B, T, D]
  // qkv_weight shape: [3 * D, D]

  TORCH_CHECK(
      !mask || !query.is_nested(),
      "NestedTensor with mask is not supported yet");
  const auto D = embed_dim;
  TORCH_CHECK(
      query.dim() == 3, "expected 3-D `query`, got ", query.dim(), "-D tensor");
  TORCH_CHECK(
      query.is_nested() || query.sizes()[2] == embed_dim,
      "passed-in embed_dim ",
      embed_dim,
      " didn't match last dim of query ",
      query.sizes()[2]);
  TORCH_CHECK(
      key.dim() == 3, "expected 3-D `key`, got ", key.dim(), "-D tensor");
  TORCH_CHECK(
      value.dim() == 3, "expected 3-D `value`, got ", value.dim(), "-D tensor");
  TORCH_CHECK(
      query.is_nested() || key.is_nested() || value.is_nested() ||
          (query.sizes() == key.sizes() && key.sizes() == value.sizes()),
      "expected `query`/`key`/`value` shapes to match");
  TORCH_CHECK(
      qkv_weight.dim() == 2,
      "expected 2-D `qkv_weight`, got ",
      qkv_weight.dim(),
      "-D tensor");
  TORCH_CHECK(
      D * 3 == qkv_weight.sizes()[0],
      "expected `qkv_weight` first dim to be 3x embed_dim");
  TORCH_CHECK(
      D == qkv_weight.sizes()[1],
      "expected `qkv_weight` second dim to be embed_Dim");
  TORCH_CHECK(
      qkv_bias.dim() == 1,
      "expected 1-D `qkv_bias`, got ",
      qkv_bias.dim(),
      "-D tensor");
  TORCH_CHECK(
      qkv_bias.sizes()[0] == 3 * D,
      "expected `qkv_bias` first dim and first dim of query to be equal");
  TORCH_CHECK(
      D % num_head == 0, "`embed_dim` must divide evenly by `num_heads`");

#ifndef NDEBUG
  const auto B = query.is_nested()
      ? native::get_nested_tensor_impl(query)->get_nested_sizes().size(0)
      : query.sizes()[0];
  auto T = query.is_nested() ? 0 : query.sizes()[1];

#endif
  const auto dim_per_head = D / num_head;
  if ((query.is_same(key) && key.is_same(value)) && dim_per_head % 8 == 0 &&
      !need_weights) {
    // We have not done linear projection yet but the input for SDP
    // Is expected to be 4 dimensional. We "cheaply" create view tensors
    // That will then be used for checking hot path conditions with
    // select_sd_backend
    auto q =
        query.view({query.size(0), -1, num_head, dim_per_head}).transpose(1, 2);
    auto k =
        key.view({key.size(0), -1, num_head, dim_per_head}).transpose(1, 2);
    auto v =
        value.view({value.size(0), -1, num_head, dim_per_head}).transpose(1, 2);

    sdp::sdp_params kernel_params{q, k, v, mask, 0.0, false, false};
    auto backend =
        static_cast<sdp::SDPBackend>(XPUNativeFunctions::_fused_sdp_choice(
            q, k, v, mask, 0.0, false, {}, false));

    // strides from packed projection for nested tensors when seq_len is 1 will
    // be and will trigger a contiguous call in the kernel, so we prevent this
    bool no_seq_len_1_nested = query.is_nested()
        ? check_for_seq_len_1_nested_tensor(kernel_params, false)
        : true;
    // The API for transformer_encoder is a mask of shape (Batch_Size,
    // Seq_len_q) For mem-eff attention this will cause the expand call to error
    // For now I am going to turn of that path not have to deal with all the
    // annoying Mask type shape grossness
    if (!mask.has_value() && no_seq_len_1_nested &&
        (backend == sdp::SDPBackend::flash_attention ||
         backend == sdp::SDPBackend::efficient_attention)) {
      auto x = at::linear(query, qkv_weight, qkv_bias);
      auto chunks = x.chunk(3, -1);
      auto x_size_0 = x.size(0);

      chunks[0] = (chunks[0].view({x_size_0, -1, num_head, dim_per_head}))
                      .transpose(1, 2);
      chunks[1] = (chunks[1].view({x_size_0, -1, num_head, dim_per_head}))
                      .transpose(1, 2);
      chunks[2] = (chunks[2].view({x_size_0, -1, num_head, dim_per_head}))
                      .transpose(1, 2);
      auto y = at::scaled_dot_product_attention(
          chunks[0], chunks[1], chunks[2], mask, 0.0, false, std::nullopt);

      auto past_sdp = y.transpose(1, 2).reshape({x_size_0, -1, embed_dim});
      return std::make_tuple(
          at::linear(past_sdp, proj_weight, proj_bias), Tensor());
    }
    // Returned math or error lets not use it
  }

  // shape: [B, T, 3 x D]
  auto qkv = native::qkv_projection(query, key, value, embed_dim, qkv_weight);

  if (!qkv.is_nested() && qkv.numel() == 0) {
    if (query.is_nested()) {
      return std::make_tuple(Tensor(), Tensor());
    }
    return std::make_tuple(at::empty_like(query), Tensor());
  }

#ifndef NDEBUG
  if (!query.is_nested() || !qkv.is_nested()) {
    if (query.is_nested()) {
      T = qkv.size(1);
    }
    native::debug_assert_shape(__LINE__, qkv, {B, T, 3 * D});
  }
#endif

#ifdef DEBUG_PRINT_EACH_STEP
  if (!qkv.is_nested()) {
    std::cerr << "qkv: " << qkv << std::endl;
  }
#endif
  // shape: 3 x [B, num_head, T, dim_per_head]
  auto q_k_v =
      XPUNativeFunctions::_transform_bias_rescale_qkv(qkv, qkv_bias, num_head);
  qkv = Tensor(); // Not used any more, allow free
  auto& q = std::get<0>(q_k_v);
  const auto& k = std::get<1>(q_k_v);
  const auto& v = std::get<2>(q_k_v);
#ifndef NDEBUG
  native::debug_assert_shape(__LINE__, q, {B, num_head, T, dim_per_head});
  native::debug_assert_shape(__LINE__, k, {B, num_head, T, dim_per_head});
  native::debug_assert_shape(__LINE__, v, {B, num_head, T, dim_per_head});
#endif
#ifdef DEBUG_PRINT_EACH_STEP
  std::cerr << "q: " << q << std::endl;
  std::cerr << "k: " << k << std::endl;
  std::cerr << "v: " << v << std::endl;
#endif

  // shape: [B, num_head, T, T]
  auto qkt = native::bmm_nt(q, k);
  // q & k are dead but cannot be freed because they were packed with v
#ifndef NDEBUG
  native::debug_assert_shape(__LINE__, qkt, {B, num_head, T, T});
#endif
#ifdef DEBUG_PRINT_EACH_STEP
  std::cerr << "qkt: " << qkt << std::endl;
#endif

  // shape: [B, num_head, T, T]
  // TODO: long-term, have a kernel that works with
  // NestedTensor directly if there is no mask passed
  qkt = native::masked_softmax(qkt, mask, query, mask_type);
#ifdef DEBUG_PRINT_EACH_STEP
  std::cerr << "qkt after softmax: " << qkt << std::endl;
#endif

  // shape: [B, num_head, T, dim_per_head]
  // reuse storage for q; we're done with it
  auto attn_ctx = native::bmm_nn(q, qkt, v);
  // qkv is not dead; we just reused storage for q!
  if (!need_weights) {
    qkt = Tensor();
  }
#ifndef NDEBUG
  native::debug_assert_shape(
      __LINE__, attn_ctx, {B, num_head, T, dim_per_head});
#endif
#ifdef DEBUG_PRINT_EACH_STEP
  std::cerr << "attn_ctx: " << attn_ctx << std::endl;
#endif

  // shape: [B, T, D]
  // Fuse transform_0213 inside
  auto proj = native::transform0213_gemm_nt_bias(
      attn_ctx, proj_weight, proj_bias, query);
#ifndef NDEBUG
  native::debug_assert_shape(__LINE__, proj, {B, T, D});
#endif
  if (need_weights && average_attn_weights) {
    // weights are not needed for full transformer, so don't worry too
    // much about performance -- we implement this just to make use
    // cases that don't disable need_weights still get some speedup.
    qkt = qkt.sum(1);
    qkt /= num_head;
  }
  return std::make_tuple(std::move(proj), std::move(qkt));
}

} // namespace at