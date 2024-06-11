#pragma clang diagnostic push
#pragma GCC diagnostic push
// Avoid SYCL compiler return-type error
#pragma clang diagnostic ignored "-Wreturn-type"
#pragma GCC diagnostic ignored "-Wreturn-type"

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include "ATen/Dispatch.h"
#include <ATen/ExpandUtils.h>
#include <comm/Runtime.h>
#include <comm/SYCLContext.h>
#include <comm/SYCLHelpers.h>
#include <comm/TensorInfo.h>


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
    return grad * Dists<scalar_t>::sign(diff) *
        (std::abs(diff) == dist);
  }
};

template <int SG_SIZE, typename scalar_t, typename F, typename nd_item>
scalar_t subgroup_reduce_agg_impl(nd_item item, scalar_t value) {
  const auto sg = item.get_sub_group();

#pragma unroll
  for (int offset = (SG_SIZE >> 1); offset > 0; offset >>= 1) {
    F::agg(value, sg.shuffle_down(value, offset));
  }
  return value;
}

template <typename scalar_t, typename F, typename nd_item>
scalar_t subgroup_reduce_agg(nd_item item, scalar_t value, const int sg_size) {
  scalar_t ret;
  switch (sg_size) {
    case 8:
      ret = subgroup_reduce_agg_impl<8, scalar_t, F, nd_item>(item, value);
      break;
    case 16:
      ret = subgroup_reduce_agg_impl<16, scalar_t, F, nd_item>(item, value);
      break;
    case 32:
      ret = subgroup_reduce_agg_impl<32, scalar_t, F, nd_item>(item, value);
      break;
    case 64:
      ret = subgroup_reduce_agg_impl<64, scalar_t, F, nd_item>(item, value);
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
static inline scalar_t reduce_agg(
    scalar_t agg,
    nd_item item,
    const local_shared& local_shared_mem) {
  const auto sg = item.get_sub_group();
  const int sg_size = sg.get_local_range()[0];

  const int group_size = item.get_local_range(0);
  const int sg_num = group_size / sg_size;

  const int local_id = item.get_local_id(0);
  const int lane_id = local_id % sg_size;
  const int sg_id = local_id / sg_size;
  agg = subgroup_reduce_agg<scalar_t, F, nd_item>(item, agg, sg_size);
  item.barrier(sycl_local_fence);
  if (0 == lane_id) {
    local_shared_mem[sg_id] = agg;
  }
  item.barrier(sycl_local_fence);
  agg = (local_id < sg_num) ? local_shared_mem[lane_id] : (scalar_t)0.0f;
  if (0 == sg_id) {
    agg = subgroup_reduce_agg<scalar_t, F, nd_item>(item, agg, sg_size);
  }

  return agg;
}

template <typename scalar_t, typename F, int p_type, typename accscalar_t>
struct CdistForwardKernelImplFunctor  : public __SYCL_KER_CONFIG_CONVENTION__ {
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
          std::abs(
              static_cast<scalar_t>(*a) - static_cast<scalar_t>(*b)),
          p_val_);
    }
    agg = reduce_agg<scalar_t, F>(agg, item_id, shared_);
    if (local_id == 0) {
      out_ptr[group_id] = F::finish(agg, p_val_);
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    shared_ = sycl_local_acc_t<scalar_t>(wgroup_size_, cgh);
  }

  CdistForwardKernelImplFunctor(
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
        wgroup_size_(wgroup_size)
 {}

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
static void cdist_forward_kernel_impl(
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
  using accscalar_t = acc_type<scalar_t,true>;
  auto p_val = static_cast<accscalar_t>(p);
  auto out_data = result.data_ptr<scalar_t>();
  auto x1_data = x1.data_ptr<scalar_t>();
  auto x2_data = x2.data_ptr<scalar_t>();

  CdistForwardKernelImplFunctor<scalar_t, F, p_type, accscalar_t> kfn(
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
  auto& sycl_queue = getCurrentSYCLQueue();
  sycl_kernel_submit(ngroups * wgroup_size, wgroup_size, sycl_queue, kfn);
}

void cdist_kernel_impl(Tensor& result, const Tensor& x1_expanded, const Tensor& x2_expanded, double p){
  const int64_t r1 = x1_expanded.size(-2);
  const int64_t r2 = x2_expanded.size(-2);
  const int64_t m = x1_expanded.size(-1);

  AT_DISPATCH_FLOATING_TYPES(
        x1_expanded.scalar_type(),
        "cdist_forward_sycl",
        [&] {
          if (p == 0.0) {
            cdist_forward_kernel_impl<scalar_t, DistsZero<scalar_t>, 0>(
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
            cdist_forward_kernel_impl<scalar_t, DistsOne<scalar_t>, 1>(
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
            cdist_forward_kernel_impl<scalar_t, DistsTwo<scalar_t>, 2>(
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
            cdist_forward_kernel_impl<scalar_t, DistsInf<scalar_t>, 3>(
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
            cdist_forward_kernel_impl<scalar_t, DistsP<scalar_t>, 4>(
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

Tensor cdist_impl(
    const Tensor& x1,
    const Tensor& x2,
    const double p,
    c10::optional<int64_t> compute_mode) {
  TORCH_CHECK(at::isFloatingType(x1.scalar_type()), "cdist only supports floating-point dtypes, X1 got: ", x1.scalar_type());
  auto device1 = x1.device().type();
  TORCH_CHECK(at::isFloatingType(x2.scalar_type()), "cdist only supports floating-point dtypes, X2 got: ", x2.scalar_type());
  auto device2 = x2.device().type();
  TORCH_CHECK(p >= 0, "cdist only supports non-negative p values");
  TORCH_CHECK(device1 == device2, "X1 and X2 must have the same device type. X1: ", device1, " X2: ", device2);
  // TODO: This is bad; this test should apply universally
  TORCH_CHECK(!x1.is_xpu() || x1.get_device() == x2.get_device(), "device of X1 (", x1.get_device(), ") must match device of X2 (", x2.get_device(), ")");

  SymInt c1 = x1.sym_size(-1);
  SymInt c2 = x2.sym_size(-1);
  // 0 - default value. If p = 2 and r1 > 25 or r2 > 25 (these values are based on performance metrics),
  // it will try to compute distance using matrix multiplication approach
  // 1 - force to use matrix multiplication for p = 2
  // 2 - do not use matrix multiplication for p = 2
  int64_t mode = compute_mode.value_or(0);
  TORCH_CHECK(mode >= 0 && mode <= 2, "possible modes: 0, 1, 2, but was: ", mode);
  SymInt r1 = x1.size(-2);
  SymInt r2 = x2.size(-2);
  if (!(p == 2 && (mode == 1 || (mode == 0 && (r1 > 25 || r2 > 25))))) {
    TORCH_CHECK(device1 == kCPU || device1 == kXPU, "cdist only supports CPU and XPU devices, X1 got: ", device1);
    TORCH_CHECK(device2 == kCPU || device2 == kXPU, "cdist only supports CPU and XPU devices, X2 got: ", device2);
  }
  int64_t dim1 = x1.dim();
  int64_t dim2 = x2.dim();
  SymIntArrayRef batch_tensor1(x1.sym_sizes().data(), dim1 - 2);
  SymIntArrayRef batch_tensor2(x2.sym_sizes().data(), dim2 - 2);
  std::vector<SymInt> expand_batch_portion =
      at::infer_size_symint(batch_tensor1, batch_tensor2);
  std::vector<SymInt> x1_expand_size(expand_batch_portion);
  x1_expand_size.insert(x1_expand_size.end(), {r1, c1});
  std::vector<SymInt> x2_expand_size(expand_batch_portion);
  x2_expand_size.insert(x2_expand_size.end(), {r2, c2});

  const SymInt expand_batch_product = c10::multiply_integers(expand_batch_portion);
  std::vector<SymInt> x1_view{expand_batch_product, r1, c1};
  std::vector<SymInt> x2_view{expand_batch_product, r2, c2};

  Tensor x1_expanded = x1.expand_symint(x1_expand_size).contiguous().view_symint(x1_view);
  Tensor x2_expanded = x2.expand_symint(x2_expand_size).contiguous().view_symint(x2_view);

  std::vector<SymInt> output_shape(std::move(expand_batch_portion));
  output_shape.insert(output_shape.end(), {r1, r2});

  Tensor result;
  if (r1 == 0 || r2 == 0 || expand_batch_product == 0) {
    result = at::empty_symint(output_shape, x1.options());
  } else if (c1 == 0) {
    result = at::zeros_symint(output_shape, x1.options());
  } else if (p == 2 && (mode == 1 || (mode == 0 && (r1 > 25 || r2 > 25)))) {
    Tensor dist = (expand_batch_product == 1)
        ? at::_euclidean_dist(x1, x2)
        : at::_euclidean_dist(x1_expanded, x2_expanded);
    result = dist.view_symint(output_shape);
  } else {
    result = at::empty_symint(output_shape, x1.options());
    cdist_kernel_impl(result, x1_expanded, x2_expanded, p);
  }
  return result;
}
} // namespace at::native::xpu


#pragma GCC diagnostic pop
#pragma clang diagnostic pop