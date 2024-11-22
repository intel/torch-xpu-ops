#include <ATen/AccumulateType.h>
#include <ATen/native/xpu/sycl/BatchKernel.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/sum.h>
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

// Special case backward when p is less than two
template <typename scalar_t>
struct DistsLtTwo {
  static scalar_t backward(
      const scalar_t diff,
      const scalar_t grad,
      const scalar_t dist,
      const scalar_t p) {
    return (dist == 0.0f || (diff == 0.0f && p < 1.f))
        ? static_cast<scalar_t>(0)
        : static_cast<scalar_t>(
              Dists<scalar_t>::sign(diff) * std::pow(std::abs(diff), p - 1) *
              grad / std::pow(dist, p - 1));
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

  static scalar_t backward(
      const scalar_t diff,
      const scalar_t grad,
      const scalar_t dist,
      const scalar_t p) {
    return dist == 0.0f ? static_cast<scalar_t>(0)
                        : static_cast<scalar_t>(
                              diff * std::pow(std::abs(diff), p - 2) * grad /
                              std::pow(dist, p - 1));
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

    const scalar_t* const start = x1_ptr + l * l1_size_ + i * m_;
    const scalar_t* const end = start + m_;
    const scalar_t* a = start + local_id;
    const scalar_t* b = x2_ptr + l * l2_size_ + j * m_ + local_id;

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
      const scalar_t* x1_data,
      const scalar_t* x2_data,
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
  const scalar_t* x1_data_;
  const scalar_t* x2_data_;
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
  auto out_data = result.mutable_data_ptr<scalar_t>();
  auto x1_data = x1.const_data_ptr<scalar_t>();
  auto x2_data = x2.const_data_ptr<scalar_t>();

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

template <typename scalar_t, typename F, int p_type, typename accscalar_t>
struct CdistBackwardKernelImplFunctor {
  void operator()(sycl::nd_item<3> item) const {
    auto buff_ptr = buff_data_;
    auto grad_ptr = grad_data_;
    auto dist_ptr = dist_data_;
    auto x1_ptr = x1_data_;
    auto x2_ptr = x2_data_;

    const int y =
        (item.get_group(1) * group_num_z_ + item.get_group(2)) * group_size_y_ +
        item.get_local_id(1);
    const int init = item.get_group(0) * group_size_x_ + item.get_local_id(0);
    if (y >= count_ || init >= m_) {
      return;
    }

    const int l = y / r_size_;
    const int k = y % r_size_;
    const int stride = group_size_x_ * group_num_x_;
    const int l_size = r_size_ * m_;

    int64_t i = k / r2_;
    int64_t j = k % r2_;

    const scalar_t grad_k = grad_ptr[y];
    const scalar_t dist_k = dist_ptr[y];

    const scalar_t* const start = x1_ptr + l * l1_size_ + i * m_;
    const scalar_t* const end = start + m_;
    const scalar_t* self_i = start + init;
    const scalar_t* self_j = x2_ptr + l * l2_size_ + j * m_ + init;

    scalar_t* buff_i = buff_ptr + l * l_size + (r1_ * j + i) * m_ + init;

    for (; self_i < end; self_i += stride, self_j += stride, buff_i += stride) {
      const scalar_t res = F::backward(
          static_cast<scalar_t>(*self_i) - static_cast<scalar_t>(*self_j),
          grad_k,
          dist_k,
          p_val_);
      *buff_i = res;
    }
  }
  CdistBackwardKernelImplFunctor(
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
      const scalar_t* x2_data)
      : r1_(r1),
        r2_(r2),
        m_(m),
        count_(count),
        r_size_(r_size),
        l1_size_(l1_size),
        l2_size_(l2_size),
        group_size_x_(group_size_x),
        group_size_y_(group_size_y),
        group_num_x_(group_num_x),
        p_val_(p_val),
        group_num_z_(group_num_z),
        buff_data_(buff_data),
        grad_data_(grad_data),
        dist_data_(dist_data),
        x1_data_(x1_data),
        x2_data_(x2_data) {}

 private:
  const int64_t r1_;
  const int64_t r2_;
  const int64_t m_;
  const int64_t count_;
  const int64_t r_size_;
  const int64_t l1_size_;
  const int64_t l2_size_;
  const int group_size_x_;
  const int group_size_y_;
  const int group_num_x_;
  accscalar_t p_val_;
  const int group_num_z_;
  scalar_t* buff_data_;
  const scalar_t* grad_data_;
  const scalar_t* dist_data_;
  const scalar_t* x1_data_;
  const scalar_t* x2_data_;
};

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

  CdistBackwardKernelImplFunctor<scalar_t, F, p_type, accscalar_t> kfn(
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
  sycl_kernel_submit(global_range, local_range, getCurrentSYCLQueue(), kfn);
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
          cdist_backward_kernel_impl<scalar_t, DistsTwo<scalar_t>, 1>(
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
struct PdistKernelFunctor : public __SYCL_KER_CONFIG_CONVENTION__ {
  void operator()(sycl::nd_item<1> item_id) const {
    auto out_ptr = out_data_;
    auto in_ptr = in_data_;

    const size_t k = item_id.get_group_linear_id();
    const size_t stride = item_id.get_local_range().size();

    int64_t i = static_cast<int64_t>(
        (n2_val_ - device_sqrt<accscalar_t>(n2_squared_minus_1_val_ - 2 * k)));
    int64_t j = k - n_ * i + i * (i + 1) / 2 + i + 1;

    const scalar_t* const start = in_ptr + i * m_;
    const scalar_t* const end = start + m_;
    const scalar_t* a = start + item_id.get_local_linear_id();
    const scalar_t* b = in_ptr + j * m_ + item_id.get_local_linear_id();
    scalar_t agg = 0.0f;
    for (; a < end; a += stride, b += stride) {
      F::inc(
          agg,
          std::abs(static_cast<scalar_t>(*a) - static_cast<scalar_t>(*b)),
          p_val_);
    }

    agg =
        group_reduce_agg_without_broadcast<scalar_t, F>(agg, item_id, shared_);
    if (item_id.get_local_linear_id() == 0) {
      out_ptr[k] = F::finish(agg, p_val_);
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    // Create the local shared memory for reducing
    shared_ = sycl_local_acc_t<scalar_t>(wgroup_size_, cgh);
  }

  PdistKernelFunctor(
      const int64_t n,
      const int64_t m,
      accscalar_t p_val,
      accscalar_t n2_val,
      accscalar_t n2_squared_minus_1_val,
      scalar_t* out_data,
      const scalar_t* in_data,
      const int64_t wgroup_size)
      : n_(n),
        m_(m),
        p_val_(p_val),
        n2_val_(n2_val),
        n2_squared_minus_1_val_(n2_squared_minus_1_val),
        out_data_(out_data),
        in_data_(in_data),
        wgroup_size_(wgroup_size) {}

 private:
  const int64_t n_;
  const int64_t m_;
  accscalar_t p_val_;
  accscalar_t n2_val_;
  accscalar_t n2_squared_minus_1_val_;
  scalar_t* out_data_;
  const scalar_t* in_data_;
  sycl_local_acc_t<scalar_t, 1> shared_;
  const int64_t wgroup_size_;
};

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
  using KernelClass = PdistKernelFunctor<scalar_t, F, accscalar_t>;
  auto min_sg_size = syclMinSubGroupSize();
  auto wgroup_size = syclMaxWorkGroupSize<KernelClass>();
  while (wgroup_size >> 1 >= m && wgroup_size >> 1 >= 32 /* sg_size */) {
    wgroup_size >>= 1;
  }

  auto p_val = static_cast<accscalar_t>(p);
  auto n2_val = static_cast<accscalar_t>(n2);
  auto n2_squared_minus_1_val = static_cast<accscalar_t>(n2_squared_minus_1);

  auto out_data = result.mutable_data_ptr<scalar_t>();
  auto in_data = self.const_data_ptr<scalar_t>();

  auto kfn = KernelClass(
      n,
      m,
      p_val,
      n2_val,
      n2_squared_minus_1_val,
      out_data,
      in_data,
      wgroup_size / min_sg_size);
  auto& queue = getCurrentSYCLQueue();
  sycl_kernel_submit(ngroups * wgroup_size, wgroup_size, queue, kfn);
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
struct PdistBackwardKernelFunctor {
  void operator()(sycl::nd_item<2> item) const {
    const int64_t k =
        item.get_group(1) * item.get_local_range(1) + item.get_local_id(1);
    const int init =
        item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
    const int stride = item.get_local_range(0) * item.get_group_range(0);

    if (k >= combs_) {
      return;
    }

    // select row i, j depending on k
    int64_t i = static_cast<int64_t>(
        (n2_val_ - device_sqrt<accscalar_t>(n2_squared_minus_1_val_ - 2 * k)));
    int64_t j = k - n_ * i + i * (i + 1) / 2 + i + 1;
    int64_t ib = j - i - 1;
    int64_t jb = n_ - 2 - i;

    const scalar_t grad_k = grad_ptr_[k * gs_];
    const scalar_t dist_k = dist_ptr_[k];

    const scalar_t* const start = in_ptr_ + i * m_;
    const scalar_t* const end = start + m_;
    const scalar_t* self_i = start + init;
    const scalar_t* self_j = in_ptr_ + j * m_ + init;
    scalar_t* buff_i = out_ptr_ + (ib * n_ + i) * m_ + init;
    scalar_t* buff_j = out_ptr_ + (jb * n_ + j) * m_ + init;

    for (; self_i < end; self_i += stride,
                         self_j += stride,
                         buff_i += stride,
                         buff_j += stride) {
      const scalar_t res =
          F::backward(*self_i - *self_j, grad_k, dist_k, p_val_);
      *buff_i = res;
      *buff_j = -res;
    }
  }
  PdistBackwardKernelFunctor(
      scalar_t* buffer,
      const scalar_t* grad,
      const scalar_t* self,
      const scalar_t* dist,
      int64_t gs,
      const int64_t n,
      const int64_t m,
      const int64_t combs,
      const scalar_t p,
      const accscalar_t n2,
      const accscalar_t n2_squared_minus_1)
      : out_ptr_(buffer),
        grad_ptr_(grad),
        in_ptr_(self),
        dist_ptr_(dist),
        gs_(gs),
        n_(n),
        m_(m),
        combs_(combs),
        p_val_(p),
        n2_val_(n2),
        n2_squared_minus_1_val_(n2_squared_minus_1) {}

 private:
  scalar_t* out_ptr_;
  const scalar_t* grad_ptr_;
  const scalar_t* in_ptr_;
  const scalar_t* dist_ptr_;
  int64_t gs_;
  const int64_t n_;
  const int64_t m_;
  const int64_t combs_;
  const scalar_t p_val_;
  const accscalar_t n2_val_;
  const accscalar_t n2_squared_minus_1_val_;
};

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
    if (p == 1.0) {
      auto caller = PdistBackwardKernelFunctor<scalar_t, DistsOne<scalar_t>>(
          buffer_ptr,
          grad_ptr,
          self_ptr,
          dist_ptr,
          grad.stride(0),
          n,
          m,
          dist.numel(),
          p,
          n2,
          n2_squared_minus_1);
      sycl_kernel_submit(
          global_range, local_range, getCurrentSYCLQueue(), caller);
    } else if (p < 2.0) {
      auto caller = PdistBackwardKernelFunctor<scalar_t, DistsLtTwo<scalar_t>>(
          buffer_ptr,
          grad_ptr,
          self_ptr,
          dist_ptr,
          grad.stride(0),
          n,
          m,
          dist.numel(),
          p,
          n2,
          n2_squared_minus_1);
      sycl_kernel_submit(
          global_range, local_range, getCurrentSYCLQueue(), caller);
    } else if (p == 2.0) {
      auto caller = PdistBackwardKernelFunctor<scalar_t, DistsTwo<scalar_t>>(
          buffer_ptr,
          grad_ptr,
          self_ptr,
          dist_ptr,
          grad.stride(0),
          n,
          m,
          dist.numel(),
          p,
          n2,
          n2_squared_minus_1);
      sycl_kernel_submit(
          global_range, local_range, getCurrentSYCLQueue(), caller);
    } else if (std::isinf(p)) {
      auto caller = PdistBackwardKernelFunctor<scalar_t, DistsInf<scalar_t>>(
          buffer_ptr,
          grad_ptr,
          self_ptr,
          dist_ptr,
          grad.stride(0),
          n,
          m,
          dist.numel(),
          p,
          n2,
          n2_squared_minus_1);
      sycl_kernel_submit(
          global_range, local_range, getCurrentSYCLQueue(), caller);
    } else {
      auto caller = PdistBackwardKernelFunctor<scalar_t, DistsP<scalar_t>>(
          buffer_ptr,
          grad_ptr,
          self_ptr,
          dist_ptr,
          grad.stride(0),
          n,
          m,
          dist.numel(),
          p,
          n2,
          n2_squared_minus_1);
      sycl_kernel_submit(
          global_range, local_range, getCurrentSYCLQueue(), caller);
    }
  });

  at::sum_out(result, buffer, 0);
}

} // namespace at::native::xpu
