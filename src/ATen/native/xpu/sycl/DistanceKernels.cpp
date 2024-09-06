#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <comm/SYCLContext.h>

#include <ATen/native/xpu/sycl/DistanceKernels.h>

namespace at::native::xpu {
template <typename scalar_t>
static double device_sqrt(scalar_t val) {
  return std::sqrt(val);
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
    agg += diff != 0.0f;
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
    agg += static_cast<scalar_t>(std::pow(static_cast<scalar_t>(diff), p));
  }
  static scalar_t finish(const scalar_t agg, const scalar_t p) {
    return static_cast<scalar_t>(
        std::pow(static_cast<scalar_t>(agg), 1.0f / p));
  }
  static void agg(scalar_t& update, const scalar_t other) {
    update += other;
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
    return grad * Dists<scalar_t>::sign(diff) * (std::abs(diff) == dist);
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
    item.barrier(sycl_local_fence);
    if (0 == lane_id) {
      local_shared_mem[sg_id] = agg;
    }
    item.barrier(sycl_local_fence);
    agg =
        local_id < num_active_sg ? local_shared_mem[local_id] : (scalar_t)0.0f;
    if (num_active_sg > sg_size)
      num_active_sg = (num_active_sg + sg_size - 1) / sg_size;
  } while (num_active_sg > sg_size);

  // num of active sgs < sg_size
  item.barrier(sycl_local_fence);
  if (0 == sg_id) {
    agg = subgroup_reduce_agg_without_broadcast<scalar_t, F, nd_item>(
        item, agg, sg_size);
  }

  return agg;
}

template <typename scalar_t, typename F, int p_type, typename accscalar_t>
struct CdistForwardKernelFunctor : public __SYCL_KER_CONFIG_CONVENTION__ {
  void operator()(sycl::nd_item<1> item_id) const {
    auto out_ptr = out_data_;
    auto x1_ptr = x1_data_;
    auto x2_ptr = x2_data_;

    const int64_t group_id = item_id.get_group_linear_id();
    const int64_t local_id = item_id.get_local_linear_id();
    const int64_t l = group_id / r_size_;
    const int64_t k = group_id % r_size_;
    const int64_t i = k / r2_;
    const int64_t j = k % r2_;
    const size_t stride = item_id.get_local_range().size();

    scalar_t* start = x1_ptr + l * l1_size_ + i * m_;
    scalar_t* end = start + m_;
    scalar_t* a = start + local_id;
    scalar_t* b = x2_ptr + l * l2_size_ + j * m_ + local_id;

    scalar_t agg = 0.0f;
    for (; a < end; a += stride, b += stride) {
      F::inc(
          agg,
          std::abs(static_cast<scalar_t>(*a) - static_cast<scalar_t>(*b)),
          p_val_);
    }
    agg =
        group_reduce_agg_without_broadcast<scalar_t, F>(agg, item_id, shared_);
    if (local_id == 0) {
      out_ptr[group_id] = F::finish(agg, p_val_);
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    shared_ = sycl_local_acc_t<scalar_t>(wgroup_size_, cgh);
  }

  CdistForwardKernelFunctor(
      const int64_t r1,
      const int64_t r2,
      const int64_t m,
      const int64_t r_size,
      const int64_t l1_size,
      const int64_t l2_size,
      accscalar_t p_val,
      scalar_t* out_data,
      scalar_t* x1_data,
      scalar_t* x2_data,
      const int64_t wgroup_size)
      : r1_(r1),
        r2_(r2),
        m_(m),
        r_size_(r_size),
        l1_size_(l1_size),
        l2_size_(l2_size),
        p_val_(p_val),
        out_data_(out_data),
        x1_data_(x1_data),
        x2_data_(x2_data),
        wgroup_size_(wgroup_size) {}

 private:
  const int64_t r1_;
  const int64_t r2_;
  const int64_t m_;
  const int64_t r_size_;
  const int64_t l1_size_;
  const int64_t l2_size_;
  accscalar_t p_val_;
  scalar_t* out_data_;
  scalar_t* x1_data_;
  scalar_t* x2_data_;
  sycl_local_acc_t<scalar_t, 1> shared_;
  const int64_t wgroup_size_;
};

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
  auto out_data = result.data_ptr<scalar_t>();
  auto x1_data = x1.data_ptr<scalar_t>();
  auto x2_data = x2.data_ptr<scalar_t>();

  CdistForwardKernelFunctor<scalar_t, F, p_type, accscalar_t> kfn(
      r1,
      r2,
      m,
      r_size,
      l1_size,
      l2_size,
      p_val,
      out_data,
      x1_data,
      x2_data,
      wgroup_size);
  auto& queue = getCurrentSYCLQueue();
  sycl_kernel_submit(ngroups * wgroup_size, wgroup_size, queue, kfn);
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

} // namespace at::native::xpu
