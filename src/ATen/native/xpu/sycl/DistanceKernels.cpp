/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/AccumulateType.h>
#include <ATen/OpMathType.h>
#include <ATen/native/xpu/sycl/BatchKernel.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/sum.h>
#include <comm/SYCLContext.h>

#include <ATen/native/xpu/sycl/DistanceKernels.h>

namespace at::native::xpu {

template <typename scalar_t>
static double device_sqrt(scalar_t val) {
  return sycl::sqrt(val);
};

template <typename scalar_t>
class Dists {
 public:
  static scalar_t sign(scalar_t val) {
    return (0 < val) - (val < 0);
  }
};

// Zero norm
template <typename scalar_t>
struct DistsZero {
  static void inc(scalar_t& agg, const scalar_t diff, const scalar_t p) {
    if (diff != diff) { // NaN
      agg = diff;
    } else if (diff != 0.0) {
      agg += 1.0;
    }
  }
  static scalar_t finish(const scalar_t agg, const scalar_t p) {
    return agg;
  }
  static void agg(scalar_t& update, const scalar_t other) {
    update += other;
  }
};

// One norm
template <typename scalar_t>
struct DistsOne {
  static void inc(scalar_t& agg, const scalar_t diff, const scalar_t p) {
    agg += diff;
  }
  static scalar_t finish(const scalar_t agg, const scalar_t p) {
    return agg;
  }
  static void agg(scalar_t& update, const scalar_t other) {
    update += other;
  }
  static scalar_t backward(
      const scalar_t diff,
      const scalar_t grad,
      const scalar_t dist,
      const scalar_t p) {
    return grad * Dists<scalar_t>::sign(diff);
  }
};

// Special case backward when p is less than two
template <typename scalar_t>
struct DistsLtTwo {
  static scalar_t backward(
      const scalar_t diff,
      const scalar_t grad,
      const scalar_t dist,
      const scalar_t p) {
    using opmath_t = at::opmath_type<scalar_t>;
    return (dist == 0.0f || (diff == 0.0f && p < 1.f))
        ? static_cast<scalar_t>(0)
        : static_cast<scalar_t>(
              Dists<scalar_t>::sign(diff) *
              sycl::pow(
                  sycl::fabs(static_cast<opmath_t>(diff)),
                  static_cast<opmath_t>(p - 1)) *
              grad /
              sycl::pow(
                  static_cast<opmath_t>(dist), static_cast<opmath_t>(p - 1)));
  }
};

// Two norm
template <typename scalar_t>
struct DistsTwo {
  static void inc(scalar_t& agg, const scalar_t diff, const scalar_t p) {
    agg += diff * diff;
  }
  static scalar_t finish(const scalar_t agg, const scalar_t p) {
    return device_sqrt<scalar_t>(agg);
  }
  static void agg(scalar_t& update, const scalar_t other) {
    update += other;
  }
  static scalar_t backward(
      const scalar_t diff,
      const scalar_t grad,
      const scalar_t dist,
      const scalar_t p) {
    return dist == 0.0f ? static_cast<scalar_t>(0) : grad * diff / dist;
  }
};

// General p norm
template <typename scalar_t>
struct DistsP {
  static void inc(scalar_t& agg, const scalar_t diff, const scalar_t p) {
    using opmath_t = at::opmath_type<scalar_t>;
    agg += static_cast<scalar_t>(
        sycl::pow(static_cast<opmath_t>(diff), static_cast<opmath_t>(p)));
  }
  static scalar_t finish(const scalar_t agg, const scalar_t p) {
    using opmath_t = at::opmath_type<scalar_t>;
    return static_cast<scalar_t>(
        sycl::pow(static_cast<opmath_t>(agg), static_cast<opmath_t>(1.0f / p)));
  }
  static void agg(scalar_t& update, const scalar_t other) {
    update += other;
  }

  static scalar_t backward(
      const scalar_t diff,
      const scalar_t grad,
      const scalar_t dist,
      const scalar_t p) {
    using opmath_t = at::opmath_type<scalar_t>;
    return dist == 0.0f
        ? static_cast<scalar_t>(0)
        : static_cast<scalar_t>(
              diff *
              sycl::pow(
                  sycl::fabs(static_cast<opmath_t>(diff)),
                  static_cast<opmath_t>(p - 2)) *
              grad /
              sycl::pow(
                  static_cast<opmath_t>(dist), static_cast<opmath_t>(p - 1)));
  }
};

// Inf norm
template <typename scalar_t>
struct DistsInf {
  static void inc(scalar_t& agg, const scalar_t diff, const scalar_t p) {
    if (diff > agg) {
      agg = diff;
    }
  }
  static scalar_t finish(const scalar_t agg, const scalar_t p) {
    return agg;
  }
  static void agg(scalar_t& update, const scalar_t other) {
    if (other > update) {
      update = other;
    }
  }
  static scalar_t backward(
      const scalar_t diff,
      const scalar_t grad,
      const scalar_t dist,
      const scalar_t p) {
    return grad * Dists<scalar_t>::sign(diff) * (sycl::fabs(diff) == dist);
  }
};

template <int SG_SIZE, typename scalar_t, typename F, typename nd_item>
scalar_t subgroup_reduce_agg_without_broadcast_impl(
    nd_item item,
    scalar_t value) {
  const auto sg = item.get_sub_group();

#pragma unroll
  for (int offset = (SG_SIZE >> 1); offset > 0; offset >>= 1) {
    F::agg(value, sycl::shift_group_left(sg, value, offset));
  }
  return value;
}

template <typename scalar_t, typename F, typename nd_item>
scalar_t subgroup_reduce_agg_without_broadcast(
    nd_item item,
    scalar_t value,
    const int sg_size) {
  scalar_t ret;
  switch (sg_size) {
    case 8:
      ret = subgroup_reduce_agg_without_broadcast_impl<8, scalar_t, F, nd_item>(
          item, value);
      break;
    case 16:
      ret =
          subgroup_reduce_agg_without_broadcast_impl<16, scalar_t, F, nd_item>(
              item, value);
      break;
    case 32:
      ret =
          subgroup_reduce_agg_without_broadcast_impl<32, scalar_t, F, nd_item>(
              item, value);
      break;
    case 64:
      ret =
          subgroup_reduce_agg_without_broadcast_impl<64, scalar_t, F, nd_item>(
              item, value);
      break;
    default:
      SYCL_KERNEL_ASSERT(false);
  }
  return ret;
}

template <
    typename scalar_t,
    typename F,
    typename nd_item,
    typename local_shared>
static inline scalar_t group_reduce_agg_without_broadcast(
    scalar_t agg,
    nd_item item,
    const local_shared& local_shared_mem) {
  const auto sg = item.get_sub_group();
  const int sg_size = sg.get_local_linear_range();
  const int lane_id = sg.get_local_linear_id();
  const int sg_id = sg.get_group_linear_id();
  const int local_id = item.get_local_linear_id();
  int num_active_sg = sg.get_group_linear_range();

  // num of active sgs >= sg_size
  do {
    agg = subgroup_reduce_agg_without_broadcast<scalar_t, F, nd_item>(
        item, agg, sg_size);
    if (num_active_sg == 1)
      return agg;
    sycl::group_barrier(item.get_group());
    if (0 == lane_id) {
      local_shared_mem[sg_id] = agg;
    }
    sycl::group_barrier(item.get_group());
    agg =
        local_id < num_active_sg ? local_shared_mem[local_id] : (scalar_t)0.0f;
    if (num_active_sg > sg_size)
      num_active_sg = (num_active_sg + sg_size - 1) / sg_size;
  } while (num_active_sg > sg_size);

  // num of active sgs < sg_size
  sycl::group_barrier(item.get_group());
  if (0 == sg_id) {
    agg = subgroup_reduce_agg_without_broadcast<scalar_t, F, nd_item>(
        item, agg, sg_size);
  }

  return agg;
}

std::tuple<Tensor, Tensor> _euclidean_dist_backward(
    const Tensor& grad,
    const Tensor& x1,
    const Tensor& x2,
    const Tensor& res) {
  if (!grad.defined()) {
    return std::tuple<Tensor, Tensor>(Tensor(), Tensor());
  }
  // handle case at 0 where we return a subgradient containing 0
  Tensor ratio = grad / res;
  ratio.masked_fill_(res == 0, 0);
  return std::tuple<Tensor, Tensor>{
      x1 * ratio.sum(-1, true) - ratio.matmul(x2),
      x2 * ratio.sum(-2, false).unsqueeze(-1) - ratio.mT().matmul(x1)};
}

template <typename scalar_t, typename F, int p_type, typename accscalar_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void cdist_forward_kernel(
    const int64_t r1,
    const int64_t r2,
    const int64_t m,
    const int64_t r_size,
    const int64_t l1_size,
    const int64_t l2_size,
    accscalar_t p_val,
    scalar_t* out_data,
    const scalar_t* x1_data,
    const scalar_t* x2_data) {
  auto item_id = syclext::this_work_item::get_nd_item<1>();
  auto* shared =
      reinterpret_cast<scalar_t*>(syclexp::get_work_group_scratch_memory());

  const int64_t group_id = item_id.get_group_linear_id();
  const int64_t local_id = item_id.get_local_linear_id();
  const int64_t l = group_id / r_size;
  const int64_t k = group_id % r_size;
  const int64_t i = k / r2;
  const int64_t j = k % r2;
  const size_t stride = item_id.get_local_range().size();

  const scalar_t* const start = x1_data + l * l1_size + i * m;
  const scalar_t* const end = start + m;
  const scalar_t* a = start + local_id;
  const scalar_t* b = x2_data + l * l2_size + j * m + local_id;

  scalar_t agg = 0.0f;
  for (; a < end; a += stride, b += stride) {
    F::inc(
        agg,
        sycl::fabs(static_cast<scalar_t>(*a) - static_cast<scalar_t>(*b)),
        p_val);
  }
  agg = group_reduce_agg_without_broadcast<scalar_t, F>(agg, item_id, shared);
  if (local_id == 0) {
    out_data[group_id] = F::finish(agg, p_val);
  }
}

template <typename scalar_t, typename F, int p_type>
static void launch_cdist_forward_kernel(
    Tensor& result,
    const Tensor& x1,
    const Tensor& x2,
    const double p,
    const int64_t r1,
    const int64_t r2,
    const int64_t m,
    const int64_t r_size,
    const int64_t l1_size,
    const int64_t l2_size) {
  const auto ngroups = result.numel();
  auto wgroup_size = 32;
  using accscalar_t = acc_type_device<scalar_t, kXPU>;
  auto p_val = static_cast<accscalar_t>(p);
  auto out_data = result.mutable_data_ptr<scalar_t>();
  auto x1_data = x1.const_data_ptr<scalar_t>();
  auto x2_data = x2.const_data_ptr<scalar_t>();

  auto& queue = getCurrentSYCLQueue();
  sycl_kernel_submit<cdist_forward_kernel<scalar_t, F, p_type, accscalar_t>>(
      ngroups * wgroup_size,
      wgroup_size,
      queue,
      wgroup_size * sizeof(scalar_t),
      r1,
      r2,
      m,
      r_size,
      l1_size,
      l2_size,
      p_val,
      out_data,
      x1_data,
      x2_data);
}

void cdist_kernel(
    Tensor& result,
    const Tensor& x1_expanded,
    const Tensor& x2_expanded,
    double p) {
  const int64_t r1 = x1_expanded.size(-2);
  const int64_t r2 = x2_expanded.size(-2);
  const int64_t m = x1_expanded.size(-1);

  AT_DISPATCH_FLOATING_TYPES(x1_expanded.scalar_type(), "cdist_xpu", [&] {
    if (p == 0.0) {
      launch_cdist_forward_kernel<scalar_t, DistsZero<scalar_t>, 0>(
          result,
          x1_expanded,
          x2_expanded,
          p,
          r1,
          r2,
          m,
          r1 * r2,
          r1 * m,
          r2 * m);
    } else if (p == 1.0) {
      launch_cdist_forward_kernel<scalar_t, DistsOne<scalar_t>, 1>(
          result,
          x1_expanded,
          x2_expanded,
          p,
          r1,
          r2,
          m,
          r1 * r2,
          r1 * m,
          r2 * m);
    } else if (p == 2.0) {
      launch_cdist_forward_kernel<scalar_t, DistsTwo<scalar_t>, 2>(
          result,
          x1_expanded,
          x2_expanded,
          p,
          r1,
          r2,
          m,
          r1 * r2,
          r1 * m,
          r2 * m);
    } else if (std::isinf(p)) {
      launch_cdist_forward_kernel<scalar_t, DistsInf<scalar_t>, 3>(
          result,
          x1_expanded,
          x2_expanded,
          p,
          r1,
          r2,
          m,
          r1 * r2,
          r1 * m,
          r2 * m);
    } else {
      launch_cdist_forward_kernel<scalar_t, DistsP<scalar_t>, 4>(
          result,
          x1_expanded,
          x2_expanded,
          p,
          r1,
          r2,
          m,
          r1 * r2,
          r1 * m,
          r2 * m);
    }
  });
}

template <typename scalar_t, typename F, int p_type, typename accscalar_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<3>))
void cdist_backward_kernel_impl_func(
    const int64_t r1,
    const int64_t r2,
    const int64_t m,
    const int64_t count,
    const int64_t r_size,
    const int64_t l1_size,
    const int64_t l2_size,
    const int group_size_x,
    const int group_size_y,
    const int group_num_x,
    accscalar_t p_val,
    const int group_num_z,
    scalar_t* buff_data,
    const scalar_t* grad_data,
    const scalar_t* dist_data,
    const scalar_t* x1_data,
    const scalar_t* x2_data) {
  auto item = syclext::this_work_item::get_nd_item<3>();

  const int y =
      (item.get_group(1) * group_num_z + item.get_group(2)) * group_size_y +
      item.get_local_id(1);
  const int init = item.get_group(0) * group_size_x + item.get_local_id(0);
  if (y >= count || init >= m) {
    return;
  }

  const int l = y / r_size;
  const int k = y % r_size;
  const int stride = group_size_x * group_num_x;
  const int l_size = r_size * m;

  int64_t i = k / r2;
  int64_t j = k % r2;

  const scalar_t grad_k = grad_data[y];
  const scalar_t dist_k = dist_data[y];

  const scalar_t* const start = x1_data + l * l1_size + i * m;
  const scalar_t* const end = start + m;
  const scalar_t* self_i = start + init;
  const scalar_t* self_j = x2_data + l * l2_size + j * m + init;

  scalar_t* buff_i = buff_data + l * l_size + (r1 * j + i) * m + init;

  for (; self_i < end; self_i += stride, self_j += stride, buff_i += stride) {
    const scalar_t res = F::backward(
        static_cast<scalar_t>(*self_i) - static_cast<scalar_t>(*self_j),
        grad_k,
        dist_k,
        p_val);
    *buff_i = res;
  }
}

template <typename scalar_t, typename F, int p_type>
static void cdist_backward_kernel_impl(
    Tensor& buffer,
    const Tensor& grad,
    const Tensor& x1,
    const Tensor& x2,
    const Tensor& dist,
    int64_t gs,
    const double p,
    const int64_t r1,
    const int64_t r2,
    const int64_t m,
    const int64_t count,
    const int64_t r_size,
    const int64_t l1_size,
    const int64_t l2_size) {
  auto wgroup_size = syclGpuHWThreadsPerEU() * syclMaxSubGroupSize();
  const int group_size_x = 256 > wgroup_size ? wgroup_size : 256;
  const int group_size_y = wgroup_size / group_size_x;
  const int group_num_x = (m + group_size_x * 32 - 1) / (group_size_x * 32);

  using accscalar_t = acc_type_device<scalar_t, kXPU>;
  auto p_val = static_cast<accscalar_t>(p);

  const int64_t group_num_temp = (count + group_size_y - 1) / group_size_y;

  const int group_num_y = (group_num_temp - 1) / 65535 + 1;
  const int group_num_z = (group_num_temp - 1) / group_num_y + 1;

  sycl::range<3> global_range(
      group_size_x * group_num_x, group_size_y * group_num_y, 1 * group_num_z);
  sycl::range<3> local_range(group_size_x, group_size_y, 1);
  sycl::nd_range<3> work_load(global_range, local_range);

  auto buff_data = buffer.mutable_data_ptr<scalar_t>();
  auto grad_data = grad.const_data_ptr<scalar_t>();
  auto dist_data = dist.const_data_ptr<scalar_t>();
  auto x1_data = x1.const_data_ptr<scalar_t>();
  auto x2_data = x2.const_data_ptr<scalar_t>();

  sycl_kernel_submit<
      cdist_backward_kernel_impl_func<scalar_t, F, p_type, accscalar_t>,
      3>(
      global_range,
      local_range,
      getCurrentSYCLQueue(),
      0,
      r1,
      r2,
      m,
      count,
      r_size,
      l1_size,
      l2_size,
      group_size_x,
      group_size_y,
      group_num_x,
      p_val,
      group_num_z,
      buff_data,
      grad_data,
      dist_data,
      x1_data,
      x2_data);
}

void cdist_backward_kernel(
    Tensor& result,
    const Tensor& grad,
    const Tensor& x1,
    const Tensor& x2,
    const double p,
    const Tensor& cdist) {
  if (p == 0.0 || grad.numel() == 0 || x1.numel() == 0 || x2.numel() == 0) {
    result.fill_(0);
    return;
  }

  const int64_t r1 = x1.size(-2);
  const int64_t r2 = x2.size(-2);
  const int64_t m = x1.size(-1);

  const int64_t count = cdist.numel();
  const int64_t gs = 1;

  int64_t batch = result.size(0);

  if (2.0 == p && (r1 > 25 || r2 > 25)) {
    std::tuple<Tensor, Tensor> edist_tuple;
    edist_tuple = _euclidean_dist_backward(grad, x1, x2, cdist);
    result = std::get<0>(edist_tuple);
    return;
  }

  Tensor buffer = (x1.dim() > 2)
      ? at::empty({batch, r2, r1, m}, result.options())
      : at::empty({r2, r1, m}, result.options());

  AT_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16, x1.scalar_type(), "cdist_backward_xpu", [&] {
        if (p == 1.0) {
          cdist_backward_kernel_impl<scalar_t, DistsOne<scalar_t>, 0>(
              buffer,
              grad,
              x1,
              x2,
              cdist,
              gs,
              p,
              r1,
              r2,
              m,
              count,
              r1 * r2,
              r1 * m,
              r2 * m);
        } else if (p < 2.0) {
          cdist_backward_kernel_impl<scalar_t, DistsLtTwo<scalar_t>, 1>(
              buffer,
              grad,
              x1,
              x2,
              cdist,
              gs,
              p,
              r1,
              r2,
              m,
              count,
              r1 * r2,
              r1 * m,
              r2 * m);
        } else if (p == 2.0) {
          cdist_backward_kernel_impl<scalar_t, DistsTwo<scalar_t>, 2>(
              buffer,
              grad,
              x1,
              x2,
              cdist,
              gs,
              p,
              r1,
              r2,
              m,
              count,
              r1 * r2,
              r1 * m,
              r2 * m);
        } else if (std::isinf(p)) {
          cdist_backward_kernel_impl<scalar_t, DistsInf<scalar_t>, 3>(
              buffer,
              grad,
              x1,
              x2,
              cdist,
              gs,
              p,
              r1,
              r2,
              m,
              count,
              r1 * r2,
              r1 * m,
              r2 * m);
        } else {
          cdist_backward_kernel_impl<scalar_t, DistsP<scalar_t>, 4>(
              buffer,
              grad,
              x1,
              x2,
              cdist,
              gs,
              p,
              r1,
              r2,
              m,
              count,
              r1 * r2,
              r1 * m,
              r2 * m);
        }
      });
  if (x1.dim() > 2) {
    at::sum_out(result, buffer, 1);
  } else {
    at::sum_out(result, buffer, 0);
  }
}

template <typename scalar_t, typename F, typename accscalar_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void pdist_kernel(
    const int64_t n,
    const int64_t m,
    accscalar_t p_val,
    const double n2_val,
    const double n2_squared_minus_1_val,
    scalar_t* out_data,
    const scalar_t* in_data) {
  auto item_id = syclext::this_work_item::get_nd_item<1>();
  auto* shared =
      reinterpret_cast<scalar_t*>(syclexp::get_work_group_scratch_memory());

  const size_t k = item_id.get_group_linear_id();
  const size_t stride = item_id.get_local_range().size();

  int64_t i = static_cast<int64_t>(
      (n2_val - device_sqrt<double>(n2_squared_minus_1_val - 2 * k)));
  int64_t j = k - n * i + i * (i + 1) / 2 + i + 1;

  const scalar_t* const start = in_data + i * m;
  const scalar_t* const end = start + m;
  const scalar_t* a = start + item_id.get_local_linear_id();
  const scalar_t* b = in_data + j * m + item_id.get_local_linear_id();
  scalar_t agg = 0.0f;
  for (; a < end; a += stride, b += stride) {
    F::inc(
        agg,
        sycl::fabs(static_cast<scalar_t>(*a) - static_cast<scalar_t>(*b)),
        p_val);
  }

  agg = group_reduce_agg_without_broadcast<scalar_t, F>(agg, item_id, shared);
  if (item_id.get_local_linear_id() == 0) {
    out_data[k] = F::finish(agg, p_val);
  }
}

template <typename scalar_t, typename F>
static void pdist_kernel_impl(
    Tensor& result,
    const Tensor& self,
    const int64_t n,
    const int64_t m,
    const double p,
    const double n2,
    const double n2_squared_minus_1) {
  const auto ngroups = result.numel();
  using accscalar_t = acc_type_device<scalar_t, kXPU>;
  auto min_sg_size = syclMinSubGroupSize();
  auto wgroup_size =
      syclMaxWorkGroupSize<pdist_kernel<scalar_t, F, accscalar_t>>();
  while (wgroup_size >> 1 >= m && wgroup_size >> 1 >= 32 /* sg_size */) {
    wgroup_size >>= 1;
  }

  auto p_val = static_cast<accscalar_t>(p);

  auto out_data = result.mutable_data_ptr<scalar_t>();
  auto in_data = self.const_data_ptr<scalar_t>();

  auto& queue = getCurrentSYCLQueue();
  sycl_kernel_submit<pdist_kernel<scalar_t, F, accscalar_t>>(
      ngroups * wgroup_size,
      wgroup_size,
      queue,
      (wgroup_size / min_sg_size) * sizeof(scalar_t),
      n,
      m,
      p_val,
      n2,
      n2_squared_minus_1,
      out_data,
      in_data);
}

void pdist_forward_kernel(Tensor& result, const Tensor& self, double p) {
  int64_t n = self.size(0);
  int64_t m = self.size(1);
  const double n2 = n - .5;
  const double n2_squared_minus_1 = n2 * n2 - 1;

  AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "pdist_xpu", [&] {
    if (p == 0.0) {
      pdist_kernel_impl<scalar_t, DistsZero<scalar_t>>(
          result, self, n, m, p, n2, n2_squared_minus_1);
    } else if (p == 1.0) {
      pdist_kernel_impl<scalar_t, DistsOne<scalar_t>>(
          result, self, n, m, p, n2, n2_squared_minus_1);
    } else if (p == 2.0) {
      pdist_kernel_impl<scalar_t, DistsTwo<scalar_t>>(
          result, self, n, m, p, n2, n2_squared_minus_1);
    } else if (std::isinf(p)) {
      pdist_kernel_impl<scalar_t, DistsInf<scalar_t>>(
          result, self, n, m, p, n2, n2_squared_minus_1);
    } else {
      pdist_kernel_impl<scalar_t, DistsP<scalar_t>>(
          result, self, n, m, p, n2, n2_squared_minus_1);
    }
  });
}

template <typename scalar_t, typename F, typename accscalar_t = double>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<2>))
void pdist_backward_kernel_func(
    scalar_t* out_ptr,
    const scalar_t* grad_ptr,
    const scalar_t* in_ptr,
    const scalar_t* dist_ptr,
    int64_t gs,
    const int64_t n,
    const int64_t m,
    const int64_t combs,
    const scalar_t p_val,
    const accscalar_t n2_val,
    const accscalar_t n2_squared_minus_1_val) {
  auto item = syclext::this_work_item::get_nd_item<2>();

  const int64_t k =
      item.get_group(1) * item.get_local_range(1) + item.get_local_id(1);
  const int init =
      item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
  const int stride = item.get_local_range(0) * item.get_group_range(0);

  if (k >= combs) {
    return;
  }

  // select row i, j depending on k
  int64_t i = static_cast<int64_t>(
      (n2_val - device_sqrt<accscalar_t>(n2_squared_minus_1_val - 2 * k)));
  int64_t j = k - n * i + i * (i + 1) / 2 + i + 1;
  int64_t ib = j - i - 1;
  int64_t jb = n - 2 - i;

  const scalar_t grad_k = grad_ptr[k * gs];
  const scalar_t dist_k = dist_ptr[k];

  const scalar_t* const start = in_ptr + i * m;
  const scalar_t* const end = start + m;
  const scalar_t* self_i = start + init;
  const scalar_t* self_j = in_ptr + j * m + init;
  scalar_t* buff_i = out_ptr + (ib * n + i) * m + init;
  scalar_t* buff_j = out_ptr + (jb * n + j) * m + init;

  for (; self_i < end;
       self_i += stride, self_j += stride, buff_i += stride, buff_j += stride) {
    const scalar_t res = F::backward(*self_i - *self_j, grad_k, dist_k, p_val);
    *buff_i = res;
    *buff_j = -res;
  }
}

void pdist_backward_kernel(
    Tensor& result,
    const Tensor& grad,
    const Tensor& self,
    const double p,
    const Tensor& dist) {
  if (p == 0.0 || grad.numel() == 0 || self.numel() == 0) {
    result.fill_(0);
    return;
  }
  const int64_t n = result.size(0);
  const int64_t m = self.size(1);
  const double n2 = n - .5;
  const double n2_squared_minus_1 = n2 * n2 - 1;

  Tensor buffer =
      at::empty({n - 1, result.size(0), result.size(1)}, result.options());

  const int group_size_x = 16;
  const int group_size_y = 64;
  const int ng_x = (dist.numel() + group_size_x - 1) / group_size_x;
  const int ng_y = (m + group_size_y * 8 - 1) / (group_size_y * 8);

  sycl::range<2> global_range(group_size_x * ng_x, group_size_y * ng_y);
  sycl::range<2> local_range(group_size_x, group_size_y);

  AT_DISPATCH_FLOATING_TYPES(result.scalar_type(), "pdist_backward_xpu", [&] {
    auto buffer_ptr = buffer.mutable_data_ptr<scalar_t>();
    auto grad_ptr = grad.const_data_ptr<scalar_t>();
    auto self_ptr = self.const_data_ptr<scalar_t>();
    auto dist_ptr = dist.const_data_ptr<scalar_t>();
    auto p_val = static_cast<scalar_t>(p);
    if (p == 1.0) {
      sycl_kernel_submit<
          pdist_backward_kernel_func<scalar_t, DistsOne<scalar_t>>,
          2>(
          global_range,
          local_range,
          getCurrentSYCLQueue(),
          0,
          buffer_ptr,
          grad_ptr,
          self_ptr,
          dist_ptr,
          grad.stride(0),
          n,
          m,
          dist.numel(),
          p_val,
          n2,
          n2_squared_minus_1);
    } else if (p < 2.0) {
      sycl_kernel_submit<
          pdist_backward_kernel_func<scalar_t, DistsLtTwo<scalar_t>>,
          2>(
          global_range,
          local_range,
          getCurrentSYCLQueue(),
          0,
          buffer_ptr,
          grad_ptr,
          self_ptr,
          dist_ptr,
          grad.stride(0),
          n,
          m,
          dist.numel(),
          p_val,
          n2,
          n2_squared_minus_1);
    } else if (p == 2.0) {
      sycl_kernel_submit<
          pdist_backward_kernel_func<scalar_t, DistsTwo<scalar_t>>,
          2>(
          global_range,
          local_range,
          getCurrentSYCLQueue(),
          0,
          buffer_ptr,
          grad_ptr,
          self_ptr,
          dist_ptr,
          grad.stride(0),
          n,
          m,
          dist.numel(),
          p_val,
          n2,
          n2_squared_minus_1);
    } else if (std::isinf(p)) {
      sycl_kernel_submit<
          pdist_backward_kernel_func<scalar_t, DistsInf<scalar_t>>,
          2>(
          global_range,
          local_range,
          getCurrentSYCLQueue(),
          0,
          buffer_ptr,
          grad_ptr,
          self_ptr,
          dist_ptr,
          grad.stride(0),
          n,
          m,
          dist.numel(),
          p_val,
          n2,
          n2_squared_minus_1);
    } else {
      sycl_kernel_submit<
          pdist_backward_kernel_func<scalar_t, DistsP<scalar_t>>,
          2>(
          global_range,
          local_range,
          getCurrentSYCLQueue(),
          0,
          buffer_ptr,
          grad_ptr,
          self_ptr,
          dist_ptr,
          grad.stride(0),
          n,
          m,
          dist.numel(),
          p_val,
          n2,
          n2_squared_minus_1);
    }
  });

  at::sum_out(result, buffer, 0);
}

} // namespace at::native::xpu
