#pragma once

#include <ATen/ceil_div.h>
#include <ATen/native/xpu/sycl/MemoryAccess.h>
#include <ATen/native/xpu/sycl/MemoryAccessUtils.h>
#include <ATen/native/xpu/sycl/SortingKernels.h>
#include <ATen/ops/full.h>
#include <ATen/ops/full_like.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/ones_like.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like.h>
#include <ATen/record_function.h>
#include <comm/SYCLContext.h>
#include <comm/SYCLHelpers.h>
#include <comm/TensorOptions.h>
#include <cstdint>
#include <functional>

namespace at::native::xpu::pstl {

using namespace at::xpu;

template <typename Predicate>
struct Not2Pred {
  template <typename T>
  bool operator()(const T x, const T y) const {
    return !bool(pred(x, y));
  }
  Not2Pred(Predicate pred) : pred(pred) {}

 private:
  Predicate pred;
};

struct IdentityPred {
  template <typename T>
  T operator()(T x) const {
    return x;
  }
};

struct PlusPred {
  template <typename T>
  T operator()(T x, T y) const {
    return x + y;
  }
};

struct PlusOrPred {
  template <typename T>
  T operator()(T a, T b) const {
    return b ? b : static_cast<T>(a + b);
  }
};

template <typename scalar_t>
struct GTFunctor {
  bool operator()(scalar_t a, scalar_t b) const {
    return std::greater<scalar_t>()(a, b);
  }
};

template <typename scalar_t>
struct LSFunctor {
  bool operator()(scalar_t a, scalar_t b) const {
    return std::less<scalar_t>()(a, b);
  }
};

template <class T, class InputIt>
struct GetItemFunctor : public __SYCL_KER_CONFIG_CONVENTION__ {
  void operator()(sycl::item<1> item_id) const {
    d_first_[item_id] = first_[item_id];
  }
  GetItemFunctor(InputIt first, T index, InputIt d_first)
      : first_(first), index_(index), d_first_(d_first) {}

 private:
  InputIt first_;
  T index_;
  InputIt d_first_;
};

template <class T, class InputIt>
static inline T get_item(InputIt first, InputIt last, T index) {
  RECORD_FUNCTION("get_item_xpu", {});
  const auto N = std::distance(first, last);
  auto& q = getCurrentSYCLQueue();

  T res = -1;
  if (index >= N)
    return res;

  auto options = map_options<T>();
  Tensor d_tensor = at::empty({N}, options);
  T* d_tensor_ptr = d_tensor.data_ptr<T>();

  GetItemFunctor<T, InputIt> kfn1(first, index, d_tensor_ptr);
  sycl_kernel_submit(sycl::range<1>(N), q, kfn1);
  res = d_tensor[index].template item<T>();

  return res;
}

template <int scan_type, class InputIt, class OutputIt, class T>
struct KSScanKernelFunctor : public __SYCL_KER_CONFIG_CONVENTION__ {
  void operator()(sycl::nd_item<1> item_id) const {
    auto local_id = item_id.get_local_linear_id();

    // initialize local_input
    auto cur_init = init_;
    if (scan_type == 1) {
      local_scan_[local_id] = c10::load(&first_[local_id]);
    } else {
      if (local_id > 0)
        local_scan_[local_id] = c10::load(&first_[local_id - 1]);
      else
        local_scan_[local_id] = 0;
    }
    if (local_id == 0)
      local_scan_[local_id] += cur_init;
    item_id.barrier(sycl_local_fence);

    // body of KS algo
    for (auto __k = 1; __k < N_; __k <<= 1) {
      auto tmp = (local_id >= __k) ? local_scan_[local_id - __k] : 0;
      item_id.barrier(sycl_local_fence);
      local_scan_[local_id] += tmp;
      item_id.barrier(sycl_local_fence);
    }

    // flush result into dst
    d_first_[local_id] = local_scan_[local_id];
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    local_scan_ = sycl_local_acc_t<T>(N_, cgh);
  }

  KSScanKernelFunctor(InputIt first, T init, int64_t N, OutputIt d_first)
      : first_(first), init_(init), N_(N), d_first_(d_first), local_scan_() {}

 private:
  InputIt first_;
  T init_;
  int64_t N_;
  OutputIt d_first_;
  sycl_local_acc_t<T> local_scan_;
};

template <int scan_type, class InputIt, class OutputIt, class T>
struct KSScanWithCarrierKernelFunctor : public __SYCL_KER_CONFIG_CONVENTION__ {
  void operator()(sycl::nd_item<1> item_id) const {
    auto local_id = item_id.get_local_linear_id();
    auto global_id = item_id.get_global_linear_id();
    auto group_id = item_id.get_group_linear_id();

    // initialize local_input
    auto cur_init = (group_id == 0 ? init_ : 0);
    if (global_id < N_) {
      if (scan_type == 1) {
        local_scan_[local_id] = c10::load(&first_[global_id]);
      } else {
        if (local_id > 0)
          local_scan_[local_id] = c10::load(&first_[global_id - 1]);
        else
          local_scan_[local_id] = 0;
      }
      if (local_id == 0)
        local_scan_[local_id] += cur_init;
      if (local_id == wgroup_size_ - 1) {
        carry_ptr_[group_id] = c10::load(&first_[global_id]);
      }
    }
    item_id.barrier(sycl_local_fence);

    // body of KS algo
    for (auto __k = 1; __k < wgroup_size_; __k <<= 1) {
      auto tmp = (local_id >= __k) ? local_scan_[local_id - __k] : 0;
      item_id.barrier(sycl_local_fence);
      local_scan_[local_id] += tmp;
      item_id.barrier(sycl_local_fence);
    }

    // flush result into dst
    if (global_id < N_) {
      d_first_[global_id] = local_scan_[local_id];
    }
    if (local_id == wgroup_size_ - 1) {
      if (scan_type == 1)
        carry_ptr_[group_id] = local_scan_[local_id];
      else
        carry_ptr_[group_id] += local_scan_[local_id];
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    local_scan_ = sycl_local_acc_t<T>(wgroup_size_, cgh);
  }

  KSScanWithCarrierKernelFunctor(
      InputIt first,
      T init,
      int64_t N,
      T* carry_ptr,
      int64_t wgroup_size,
      OutputIt d_first)
      : first_(first),
        init_(init),
        N_(N),
        carry_ptr_(carry_ptr),
        wgroup_size_(wgroup_size),
        d_first_(d_first),
        local_scan_() {}

 private:
  InputIt first_;
  T init_;
  int64_t N_;
  T* carry_ptr_;
  int64_t wgroup_size_;
  OutputIt d_first_;
  sycl_local_acc_t<T> local_scan_;
};

template <class OutputIt, class T>
struct ScanAccumulateKernelFunctor : public __SYCL_KER_CONFIG_CONVENTION__ {
  void operator()(sycl::nd_item<1> item_id) const {
    auto local_id = item_id.get_local_linear_id();
    auto global_id = item_id.get_global_linear_id();
    auto group_id = item_id.get_group_linear_id();

    if (local_id == 0)
      local_carry_[0] = carry_ptr_[group_id];
    item_id.barrier(sycl_local_fence);

    if (global_id < N_) {
      d_first_[global_id] += local_carry_[0];
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    local_carry_ = sycl_local_acc_t<T>(1, cgh);
    return;
  }

  ScanAccumulateKernelFunctor(OutputIt d_first, T* carry_ptr, int64_t N)
      : local_carry_(), d_first_(d_first), carry_ptr_(carry_ptr), N_(N) {}

 private:
  sycl_local_acc_t<T> local_carry_;
  OutputIt d_first_;
  T* carry_ptr_;
  int64_t N_;
};

template <int scan_type, class InputIt, class OutputIt, class T>
static inline OutputIt _scan_kernel(
    InputIt first,
    InputIt last,
    OutputIt d_first,
    T init) {
  using KSScanKernel = KSScanKernelFunctor<scan_type, InputIt, OutputIt, T>;
  using KSScanWithCarrierKernel =
      KSScanWithCarrierKernelFunctor<scan_type, InputIt, OutputIt, T>;

  const auto N = std::distance(first, last);
  auto& q = getCurrentSYCLQueue();
  const auto kss_wgroup_size = syclMaxWorkGroupSize<KSScanKernel>();

  auto options = map_options<T>();

  if (N <= kss_wgroup_size) {
    // Kogge-Stone addr algorithm;
    KSScanKernel kfn1(first, init, N, d_first);
    sycl_kernel_submit(sycl::range<1>(N), sycl::range<1>(N), q, kfn1);

    return d_first + N;
  }

  const auto kssc_wgroup_size = syclMaxWorkGroupSize<KSScanWithCarrierKernel>();
  auto ngroups = (N + kssc_wgroup_size - 1) / kssc_wgroup_size;
  Tensor carry = at::empty({ngroups}, options);
  T* carry_ptr = carry.data_ptr<T>();

  // 1. do exclusive_scan on each workgroups
  KSScanWithCarrierKernel kfn2(
      first, init, N, carry_ptr, kssc_wgroup_size, d_first);
  sycl_kernel_submit(
      sycl::range<1>(ngroups * kssc_wgroup_size),
      sycl::range<1>(kssc_wgroup_size),
      q,
      kfn2);

  // 2. recursion for carry
  _scan_kernel<0>(carry_ptr, carry_ptr + ngroups, carry_ptr, (T)0);

  // 3. reduce among all work groups and flush data to dst
  ScanAccumulateKernelFunctor<OutputIt, T> kfn3(d_first, carry_ptr, N);

  const auto sa_wgroup_size = syclMaxWorkGroupSize(kfn3);
  ngroups = (N + sa_wgroup_size - 1) / sa_wgroup_size;

  sycl_kernel_submit(
      sycl::range<1>(ngroups * sa_wgroup_size),
      sycl::range<1>(sa_wgroup_size),
      q,
      kfn3);

  return d_first + N;
}

template <typename T, class InputIt, class OutputIt>
static inline OutputIt exclusive_scan(
    InputIt first,
    InputIt last,
    OutputIt d_first,
    T init) {
  RECORD_FUNCTION("exclusive_scan_xpu", {});
  return _scan_kernel<0>(first, last, d_first, init);
}

template <typename T, class InputIt, class OutputIt>
static inline OutputIt inclusive_scan(
    InputIt first,
    InputIt last,
    OutputIt d_first,
    T init) {
  RECORD_FUNCTION("inclusive_scan_xpu", {});
  return _scan_kernel<1>(first, last, d_first, init);
}

template <typename index_t, class InputIt, class OutputIt, class UnaryPredicate>
struct PredictKernelFunctor {
  void operator()(sycl::item<1> item_id) const {
    if (first_) {
      gmask_ptr_[item_id] =
          static_cast<index_t>(static_cast<bool>(pred_(first_[item_id])));
    } else {
      gmask_ptr_[item_id] =
          static_cast<index_t>(static_cast<bool>(pred_(item_id)));
    }
  }
  PredictKernelFunctor(InputIt first, UnaryPredicate pred, index_t* gmask_ptr)
      : first_(first), pred_(pred), gmask_ptr_(gmask_ptr) {}

 private:
  InputIt first_;
  UnaryPredicate pred_;
  index_t* gmask_ptr_;
};

template <typename T, class InputIt, class OutputIt, class BinaryPredicate>
struct InclusiveScanIfKernelFunctor {
  void operator()(sycl::item<1> item_id) const {
    d_first_[item_id] = static_cast<T>(first_[item_id]);

    if (mask_ptr_[item_id] == 0) {
      for (int64_t _k = 1; item_id - _k >= 0; _k++) {
        auto tmp = first_[item_id - _k];
        d_first_[item_id] = static_cast<T>(p_(d_first_[item_id], tmp));
        if (mask_ptr_[item_id - _k] != 0)
          break;
      }
    }
  }

  InclusiveScanIfKernelFunctor(
      InputIt first,
      InputIt mask_ptr,
      OutputIt d_first,
      BinaryPredicate p)
      : first_(first), mask_ptr_(mask_ptr), d_first_(d_first), p_(p) {}

 private:
  InputIt first_;
  InputIt mask_ptr_;
  OutputIt d_first_;
  BinaryPredicate p_;
};

template <typename T, class InputIt, class OutputIt, class BinaryPredicate>
OutputIt inclusive_scan_if(
    InputIt first,
    InputIt last,
    InputIt mask_ptr,
    OutputIt d_first,
    BinaryPredicate p) {
  RECORD_FUNCTION("inclusive_scan_if_xpu", {});
  const auto N = std::distance(first, last);
  auto& q = getCurrentSYCLQueue();

  InclusiveScanIfKernelFunctor<T, InputIt, OutputIt, BinaryPredicate> ifn(
      first, mask_ptr, d_first, p);
  sycl_kernel_submit(sycl::range<1>(N), q, ifn);

  return d_first;
}

template <typename index_t, class InputIt, class OutputIt>
struct ReverseCopyKernelFunctor {
  void operator()(sycl::item<1> item_id) const {
    d_first_[N_ - tpos_ptr_[item_id]] = first_[item_id];
  }

  ReverseCopyKernelFunctor(
      InputIt first,
      OutputIt d_first,
      index_t* tpos_ptr,
      index_t N)
      : first_(first), d_first_(d_first), tpos_ptr_(tpos_ptr), N_(N) {}

 private:
  InputIt first_;
  OutputIt d_first_;
  index_t* tpos_ptr_;
  index_t N_;
};

template <typename index_t, class InputIt, class OutputIt>
static inline OutputIt reverse_copy(
    InputIt first,
    InputIt last,
    OutputIt d_first) {
  RECORD_FUNCTION("copy_xpu", {});
  const auto N = std::distance(first, last);
  auto& q = getCurrentSYCLQueue();

  auto index_options = map_options<index_t>();

  Tensor global_mask = at::ones({N}, index_options);
  Tensor target_pos = at::empty({N}, index_options);
  index_t* gmask_ptr = global_mask.data_ptr<index_t>();
  index_t* tpos_ptr = target_pos.data_ptr<index_t>();

  inclusive_scan(gmask_ptr, gmask_ptr + N, tpos_ptr, static_cast<index_t>(0));

  // copy selected data into dst
  ReverseCopyKernelFunctor<index_t, InputIt, OutputIt> kfn(
      first, d_first, tpos_ptr, static_cast<index_t>(N));
  sycl_kernel_submit(sycl::range<1>(N), q, kfn);

  index_t M = target_pos[N - 1].template item<index_t>();
  return d_first + M;
}

template <typename index_t, class InputIt, class OutputIt>
struct CopyIfKernelFunctor {
  void operator()(sycl::item<1> item_id) const {
    if (gmask_ptr_[item_id] != 0) {
      if (first_) {
        d_first_[tpos_ptr_[item_id] - /*inclusive shift*/ 1] = first_[item_id];
      } else {
        d_first_[tpos_ptr_[item_id] - /*inclusive shift*/ 1] = item_id;
      }
    }
  }
  CopyIfKernelFunctor(
      InputIt first,
      OutputIt d_first,
      index_t* gmask_ptr,
      index_t* tpos_ptr)
      : first_(first),
        d_first_(d_first),
        gmask_ptr_(gmask_ptr),
        tpos_ptr_(tpos_ptr) {}

 private:
  InputIt first_;
  OutputIt d_first_;
  index_t* gmask_ptr_;
  index_t* tpos_ptr_;
};

template <typename index_t, class InputIt, class OutputIt, class UnaryPredicate>
static inline OutputIt copy_if(
    InputIt first,
    InputIt last,
    OutputIt d_first,
    UnaryPredicate pred) {
  RECORD_FUNCTION("copy_if_xpu", {});
  const auto N = std::distance(first, last);
  auto& q = getCurrentSYCLQueue();

  auto index_options = map_options<index_t>();

  Tensor global_mask = at::empty({N}, index_options);
  Tensor target_pos = at::empty({N}, index_options);
  index_t* gmask_ptr = global_mask.data_ptr<index_t>();
  index_t* tpos_ptr = target_pos.data_ptr<index_t>();

  // 1. get mask for `if` positions
  PredictKernelFunctor<index_t, InputIt, OutputIt, UnaryPredicate> kfn1(
      first, pred, gmask_ptr);
  sycl_kernel_submit(sycl::range<1>(N), q, kfn1);

  // 2. get target positions(with shift -1) using inclusive_scan
  inclusive_scan(gmask_ptr, gmask_ptr + N, tpos_ptr, static_cast<index_t>(0));

  // 3. copy selected data into dst
  CopyIfKernelFunctor<index_t, InputIt, OutputIt> kfn2(
      first, d_first, gmask_ptr, tpos_ptr);
  sycl_kernel_submit(sycl::range<1>(N), q, kfn2);

  index_t M = target_pos[N - 1].template item<index_t>();
  return d_first + M;
}

template <
    typename output_t,
    class InputIt,
    class OutputIt,
    class UnaryOperation>
struct TransformUnaryKernelFunctor {
  void operator()(sycl::item<1> item_id) const {
    d_first_[item_id] = static_cast<output_t>(unary_op_(first1_[item_id]));
  }
  TransformUnaryKernelFunctor(
      InputIt first1,
      OutputIt d_first,
      UnaryOperation unary_op)
      : first1_(first1), d_first_(d_first), unary_op_(unary_op) {}

 private:
  InputIt first1_;
  OutputIt d_first_;
  UnaryOperation unary_op_;
};

template <
    typename output_t,
    class InputIt,
    class OutputIt,
    class UnaryOperation>
static inline OutputIt transform(
    InputIt first1,
    InputIt last1,
    OutputIt d_first,
    UnaryOperation unary_op) {
  RECORD_FUNCTION("transform_unary_xpu", {});
  const auto N = std::distance(first1, last1);
  auto& q = getCurrentSYCLQueue();

  TransformUnaryKernelFunctor<output_t, InputIt, OutputIt, UnaryOperation> kfn(
      first1, d_first, unary_op);
  sycl_kernel_submit(sycl::range<1>(N), q, kfn);

  return d_first + N;
}

template <
    typename output_t,
    class InputIt1,
    class InputIt2,
    class OutputIt,
    class BinaryOperation>
struct TransformBinaryKernelFunctor {
  void operator()(sycl::item<1> item_id) const {
    d_first_[item_id] =
        static_cast<output_t>(binary_op_(first1_[item_id], first2_[item_id]));
  }
  TransformBinaryKernelFunctor(
      InputIt1 first1,
      InputIt2 first2,
      OutputIt d_first,
      BinaryOperation binary_op)
      : first1_(first1),
        first2_(first2),
        d_first_(d_first),
        binary_op_(binary_op) {}

 private:
  InputIt1 first1_;
  InputIt2 first2_;
  OutputIt d_first_;
  BinaryOperation binary_op_;
};

template <
    typename output_t,
    class InputIt1,
    class InputIt2,
    class OutputIt,
    class BinaryOperation>
static inline OutputIt transform(
    InputIt1 first1,
    InputIt1 last1,
    InputIt2 first2,
    OutputIt d_first,
    BinaryOperation binary_op) {
  RECORD_FUNCTION("transform_binary_xpu", {});
  const auto N = std::distance(first1, last1);
  auto& q = getCurrentSYCLQueue();

  TransformBinaryKernelFunctor<
      output_t,
      InputIt1,
      InputIt2,
      OutputIt,
      BinaryOperation>
      kfn(first1, first2, d_first, binary_op);
  sycl_kernel_submit(sycl::range<1>(N), q, kfn);

  return d_first + N;
}

template <
    typename output_t,
    class InputIt1,
    class InputIt2,
    class OutputIt,
    class BinaryOperation>
struct TransformFirstTrueKernelFunctor {
  void operator()(sycl::item<1> item_id) const {
    first1_[0] = 1;
    d_first_[item_id] =
        static_cast<output_t>(binary_op_(first1_[item_id], first2_[item_id]));
  }
  TransformFirstTrueKernelFunctor(
      InputIt1 first1,
      InputIt2 first2,
      OutputIt d_first,
      BinaryOperation binary_op)
      : first1_(first1),
        first2_(first2),
        d_first_(d_first),
        binary_op_(binary_op) {}

 private:
  InputIt1 first1_;
  InputIt2 first2_;
  OutputIt d_first_;
  BinaryOperation binary_op_;
};

template <
    typename output_t,
    class InputIt1,
    class InputIt2,
    class OutputIt,
    class BinaryOperation>
static inline OutputIt transform_first_true(
    InputIt1 first1,
    InputIt1 last1,
    InputIt2 first2,
    OutputIt d_first,
    BinaryOperation binary_op) {
  RECORD_FUNCTION("transform_first_true", {});
  const auto N = std::distance(first1, last1);
  auto& q = getCurrentSYCLQueue();

  TransformFirstTrueKernelFunctor<
      output_t,
      InputIt1,
      InputIt2,
      OutputIt,
      BinaryOperation>
      kfn(first1, first2, d_first, binary_op);
  sycl_kernel_submit(sycl::range<1>(N), q, kfn);

  return d_first + N;
}

template <class T, class ForwardIt>
struct ItoAKernelFunctor {
  void operator()(sycl::item<1> item_id) const {
    first_[item_id] = value_ + static_cast<T>(item_id);
  }
  ItoAKernelFunctor(ForwardIt first, T value) : first_(first), value_(value) {}

 private:
  ForwardIt first_;
  T value_;
};

template <class T, class ForwardIt>
static inline void itoa(ForwardIt first, ForwardIt last, T value) {
  RECORD_FUNCTION("itoa_xpu", {});
  const auto N = std::distance(first, last);
  auto& q = getCurrentSYCLQueue();

  ItoAKernelFunctor<T, ForwardIt> kfn(first, value);
  sycl_kernel_submit(sycl::range<1>(N), q, kfn);
}

template <class T, class InputIt1, class InputIt2, class OutputIt>
struct PiecewiseKernelFunctor {
  void operator()(sycl::item<1> item_id) const {
    auto start = cnt_start_first[item_id];
    auto end = cnt_end_first[item_id];
    pstl::inclusive_scan(
        first + static_cast<T>(start),
        first + static_cast<T>(start) + static_cast<T>(end) + 1,
        d_first,
        static_cast<T>(0));
  }

  PiecewiseKernelFunctor(
      InputIt1 first,
      InputIt1 last,
      InputIt1 cnt_start_first,
      InputIt1 cnt_end_first,
      InputIt2 flag_first,
      OutputIt d_first)
      : first(first),
        last(last),
        cnt_start_first(cnt_start_first),
        cnt_end_first(cnt_end_first),
        flag_first(flag_first),
        d_first(d_first) {}

 private:
  InputIt1 first;
  InputIt1 last;
  InputIt1 cnt_start_first;
  InputIt1 cnt_end_first;
  InputIt2 flag_first;
  OutputIt d_first;
};

template <class T, class InputIt1, class InputIt2, class OutputIt>
static inline void piecewise_sum(
    InputIt1 first,
    InputIt1 last,
    InputIt1 cnt_start_first,
    InputIt1 cnt_end_first,
    InputIt2 flag_first,
    OutputIt d_first) {
  RECORD_FUNCTION("piecewise_sum_xpu", {});
  const auto N = std::distance(first, last);
  auto& q = getCurrentSYCLQueue();

  PiecewiseKernelFunctor<T, InputIt1, InputIt2, OutputIt> kfn(
      first, last, cnt_start_first, cnt_end_first, flag_first, d_first);
  sycl_kernel_submit(sycl::range<1>(N), q, kfn);
}

template <typename index_t, class ForwardIt, class BinaryPredicate>
struct ExclusiveAdjacentDifferenceKernelFunctor {
  void operator()(sycl::item<1> item_id) const {
    if (item_id > 0)
      gmask_ptr_[item_id] = static_cast<index_t>(
          static_cast<bool>(!p_(first_[item_id - 1], first_[item_id])));
    else
      gmask_ptr_[item_id] = static_cast<index_t>(1); // Exclude first_[0]
  }
  ExclusiveAdjacentDifferenceKernelFunctor(
      ForwardIt first,
      index_t* gmask_ptr,
      BinaryPredicate p)
      : first_(first), gmask_ptr_(gmask_ptr), p_(p) {}

 private:
  ForwardIt first_;
  index_t* gmask_ptr_;
  BinaryPredicate p_;
};

template <typename T, typename index_t, class ForwardIt, class BinaryPredicate>
struct ExclusiveCopyIfKernelFunctor {
  void operator()(sycl::item<1> item_id) const {
    if (gmask_ptr_[item_id] != 0)
      scratchpad_ptr_[tpos_ptr_[item_id]] = first_[item_id];
  }
  ExclusiveCopyIfKernelFunctor(
      ForwardIt first,
      index_t* gmask_ptr,
      index_t* tpos_ptr,
      T* scratchpad_ptr)
      : first_(first),
        gmask_ptr_(gmask_ptr),
        tpos_ptr_(tpos_ptr),
        scratchpad_ptr_(scratchpad_ptr) {}

 private:
  ForwardIt first_;
  index_t* gmask_ptr_;
  index_t* tpos_ptr_;
  T* scratchpad_ptr_;
};

template <typename T, class ForwardIt>
struct SimpleCopyKernelFunctor {
  void operator()(sycl::item<1> item_id) const {
    first_[item_id] = scratchpad_ptr_[item_id];
  }
  SimpleCopyKernelFunctor(ForwardIt first, T* scratchpad_ptr)
      : first_(first), scratchpad_ptr_(scratchpad_ptr) {}

 private:
  ForwardIt first_;
  T* scratchpad_ptr_;
};

template <typename T, typename index_t, class ForwardIt, class BinaryPredicate>
ForwardIt unique(ForwardIt first, ForwardIt last, BinaryPredicate p) {
  RECORD_FUNCTION("unique_kernel_xpu", {});
  const auto N = std::distance(first, last);
  auto& q = getCurrentSYCLQueue();

  auto options = map_options<T>();
  auto index_options = map_options<index_t>();

  Tensor global_mask = at::empty({N}, index_options);
  Tensor target_pos = at::empty({N}, index_options);
  index_t* gmask_ptr = global_mask.data_ptr<index_t>();
  index_t* tpos_ptr = target_pos.data_ptr<index_t>();

  // 1. get mask for `if` positions
  ExclusiveAdjacentDifferenceKernelFunctor<index_t, ForwardIt, BinaryPredicate>
      kfn1(first, gmask_ptr, p);
  sycl_kernel_submit(sycl::range<1>(N), q, kfn1);

  // 2. get target positions with exclusive_scan
  exclusive_scan(gmask_ptr, gmask_ptr + N, tpos_ptr, static_cast<index_t>(0));

  // 3. copy selected data into dst
  Tensor scratchpad = at::empty({N}, options);
  T* scratchpad_ptr = scratchpad.data_ptr<T>();

  ExclusiveCopyIfKernelFunctor<T, index_t, ForwardIt, BinaryPredicate> kfn2(
      first, gmask_ptr, tpos_ptr, scratchpad_ptr);
  sycl_kernel_submit(sycl::range<1>(N), q, kfn2);

  index_t M = global_mask[N - 1].template item<index_t>() +
      target_pos[N - 1].template item<index_t>();

  SimpleCopyKernelFunctor<T, ForwardIt> kfn3(first, scratchpad_ptr);
  sycl_kernel_submit(sycl::range<1>(M), q, kfn3);

  return first + M;
}

template <
    typename T,
    typename zT,
    typename index_t,
    class ForwardIt,
    class ZipForwardIt,
    class BinaryPredicate>
struct ExclusiveCopyIfWithZipKernelFunctor {
  void operator()(sycl::item<1> item_id) const {
    if (gmask_ptr_[item_id] != 0) {
      scratchpad_ptr_[tpos_ptr_[item_id]] = first_[item_id];
      z_scratchpad_ptr_[tpos_ptr_[item_id]] = z_first_[item_id];
    }
  }
  ExclusiveCopyIfWithZipKernelFunctor(
      ForwardIt first,
      ZipForwardIt z_first,
      index_t* gmask_ptr,
      index_t* tpos_ptr,
      T* scratchpad_ptr,
      zT* z_scratchpad_ptr)
      : first_(first),
        z_first_(z_first),
        gmask_ptr_(gmask_ptr),
        tpos_ptr_(tpos_ptr),
        scratchpad_ptr_(scratchpad_ptr),
        z_scratchpad_ptr_(z_scratchpad_ptr) {}

 private:
  ForwardIt first_;
  ZipForwardIt z_first_;
  index_t* gmask_ptr_;
  index_t* tpos_ptr_;
  T* scratchpad_ptr_;
  zT* z_scratchpad_ptr_;
};

template <typename T, typename zT, class ForwardIt, class ZipForwardIt>
struct SimpleCopyWithZipKernelFunctor {
  void operator()(sycl::item<1> item_id) const {
    first_[item_id] = scratchpad_ptr_[item_id];
    z_first_[item_id] = z_scratchpad_ptr_[item_id];
  }
  SimpleCopyWithZipKernelFunctor(
      ForwardIt first,
      ZipForwardIt z_first,
      T* scratchpad_ptr,
      zT* z_scratchpad_ptr)
      : first_(first),
        z_first_(z_first),
        scratchpad_ptr_(scratchpad_ptr),
        z_scratchpad_ptr_(z_scratchpad_ptr) {}

 private:
  ForwardIt first_;
  ZipForwardIt z_first_;
  T* scratchpad_ptr_;
  zT* z_scratchpad_ptr_;
};

template <
    typename T,
    typename zT,
    typename index_t,
    class ForwardIt,
    class ZipForwardIt,
    class BinaryPredicate>
std::tuple<ForwardIt, ZipForwardIt> unique_with_zip(
    ForwardIt first,
    ForwardIt last,
    ZipForwardIt z_first,
    BinaryPredicate p) {
  RECORD_FUNCTION("unique_with_zip_xpu", {});
  const auto N = std::distance(first, last);
  auto& q = getCurrentSYCLQueue();

  auto options = map_options<T>();
  auto z_options = map_options<zT>();
  auto index_options = map_options<index_t>();

  Tensor global_mask = at::empty({N}, index_options);
  Tensor target_pos = at::empty({N}, index_options);
  index_t* gmask_ptr = global_mask.data_ptr<index_t>();
  index_t* tpos_ptr = target_pos.data_ptr<index_t>();

  // 1. get mask for `if` positions
  ExclusiveAdjacentDifferenceKernelFunctor<index_t, ForwardIt, BinaryPredicate>
      kfn1(first, gmask_ptr, p);
  sycl_kernel_submit(sycl::range<1>(N), q, kfn1);

  // 2. get target positions with exclusive_scan
  exclusive_scan(gmask_ptr, gmask_ptr + N, tpos_ptr, static_cast<index_t>(0));

  // 3. copy selected data into dst
  Tensor scratchpad = at::empty({N}, options);
  Tensor z_scratchpad = at::empty({N}, z_options);
  T* scratchpad_ptr = scratchpad.data_ptr<T>();
  zT* z_scratchpad_ptr = z_scratchpad.data_ptr<zT>();

  ExclusiveCopyIfWithZipKernelFunctor<
      T,
      zT,
      index_t,
      ForwardIt,
      ZipForwardIt,
      BinaryPredicate>
      kfn2(
          first,
          z_first,
          gmask_ptr,
          tpos_ptr,
          scratchpad_ptr,
          z_scratchpad_ptr);
  sycl_kernel_submit(sycl::range<1>(N), q, kfn2);

  index_t M = global_mask[N - 1].template item<index_t>() +
      target_pos[N - 1].template item<index_t>();

  SimpleCopyWithZipKernelFunctor<T, zT, ForwardIt, ZipForwardIt> kfn3(
      first, z_first, scratchpad_ptr, z_scratchpad_ptr);
  sycl_kernel_submit(sycl::range<1>(M), q, kfn3);

  return std::make_tuple<ForwardIt, ZipForwardIt>(first + M, z_first + M);
}

template <
    typename output_t,
    class InputIt,
    class OutputIt,
    class BinaryOperation>
struct AdjacentDifferenceKernelFunctor {
  void operator()(sycl::item<1> item_id) const {
    if (item_id > 0)
      adiff_[item_id] =
          static_cast<output_t>(op_(first_[item_id - 1], first_[item_id]));
    else
      adiff_[item_id] = static_cast<output_t>(first_[item_id]);
  }
  AdjacentDifferenceKernelFunctor(
      InputIt first,
      BinaryOperation op,
      OutputIt adiff)
      : first_(first), op_(op), adiff_(adiff) {}

 private:
  InputIt first_;
  BinaryOperation op_;
  OutputIt adiff_;
};

template <
    typename output_t,
    class InputIt,
    class OutputIt,
    class BinaryOperation>
OutputIt adjacent_difference(
    InputIt first,
    InputIt last,
    OutputIt d_first,
    BinaryOperation op) {
  RECORD_FUNCTION("adjacent_difference", {});
  const auto N = std::distance(first, last);
  auto& q = getCurrentSYCLQueue();

  Tensor scratchpad;
  OutputIt adiff = d_first;
  bool is_inplace = (void*)first == (void*)d_first ? true : false;
  if (is_inplace) {
    scratchpad = at::empty({N}, map_options<output_t>());
    adiff = scratchpad.data_ptr<output_t>();
  }

  AdjacentDifferenceKernelFunctor<output_t, InputIt, OutputIt, BinaryOperation>
      kfn1(first, op, adiff);
  sycl_kernel_submit(sycl::range<1>(N), q, kfn1);

  if (is_inplace) {
    SimpleCopyKernelFunctor<output_t, OutputIt> kfn2(d_first, adiff);
    sycl_kernel_submit(sycl::range<1>(N), q, kfn2);
  }

  return d_first + N;
}

struct AdjacentDifferenceFunctor {
  template <typename T>
  auto operator()(T l, T r) const {
    return r - l;
  }
};

template <typename output_t, class InputIt, class OutputIt>
OutputIt adjacent_difference(InputIt first, InputIt last, OutputIt d_first) {
  auto fn = AdjacentDifferenceFunctor();
  return adjacent_difference<output_t>(first, last, d_first, fn);
}

template <typename output_t, typename index_t, class OutputIt>
struct IndexCopyKernelFunctor {
  void operator()(sycl::item<1> item_id) const {
    d_first_[item_id] = range_ptr_[tpos_ptr_[item_id]];
  }
  IndexCopyKernelFunctor(
      OutputIt d_first,
      output_t* range_ptr,
      index_t* tpos_ptr)
      : d_first_(d_first), range_ptr_(range_ptr), tpos_ptr_(tpos_ptr) {}

 private:
  OutputIt d_first_;
  output_t* range_ptr_;
  index_t* tpos_ptr_;
};

template <typename output_t, typename index_t>
struct CountBySegmentCopyIfKernelFunctor {
  auto operator()(output_t a) const {
    return gmask_ptr_[a] != 0;
  }
  CountBySegmentCopyIfKernelFunctor(index_t* gmask_ptr)
      : gmask_ptr_(gmask_ptr) {}

 private:
  index_t* gmask_ptr_;
};

template <
    typename input_t,
    typename output_t,
    typename index_t,
    class InputIt,
    class OutputIt,
    class BinaryPredicate>
OutputIt count_by_segment(
    InputIt first,
    InputIt last,
    OutputIt d_first,
    BinaryPredicate p) {
  RECORD_FUNCTION("count_by_segment_xpu", {});
  const auto N = std::distance(first, last);
  auto& q = getCurrentSYCLQueue();

  auto options = map_options<output_t>();
  auto index_options = map_options<index_t>();

  Tensor global_mask = at::empty({N}, index_options);
  Tensor target_pos = at::empty({N}, index_options);
  index_t* gmask_ptr = global_mask.data_ptr<index_t>();
  index_t* tpos_ptr = target_pos.data_ptr<index_t>();

  // 1. get mask for `if` positions
  ExclusiveAdjacentDifferenceKernelFunctor<index_t, InputIt, BinaryPredicate>
      kfn1(first, gmask_ptr, p);
  sycl_kernel_submit(sycl::range<1>(N), q, kfn1);

  // 2. get target positions with inclusive_scan
  inclusive_scan(gmask_ptr, gmask_ptr + N, tpos_ptr, static_cast<index_t>(0));

  // 3. calculate counts for each unique point
  Tensor range = at::empty({N + 1}, options);
  output_t* range_ptr = range.data_ptr<output_t>();
  auto range_begin = range_ptr;
  itoa(range_begin, range_begin + N + 1, (output_t)0);
  Tensor picked_range = at::empty({N + 1}, options);
  output_t* picked_range_ptr = picked_range.data_ptr<output_t>();
  auto picked_range_begin = picked_range_ptr;
  auto picked_range_end = picked_range_begin;
  auto fn = CountBySegmentCopyIfKernelFunctor<output_t, index_t>(gmask_ptr);
  picked_range_end =
      copy_if<index_t>(range_begin, range_begin + N, picked_range_begin, fn);
  auto num_out = std::distance(picked_range_begin, picked_range_end);
  picked_range[num_out] = N;
  // notice: the temp tensor `range` will be re-used to store the result of
  // adjacent_difference
  adjacent_difference<index_t>(
      picked_range_begin + 1, picked_range_begin + num_out + 1, range_begin);

  // 4. flush range to every elements of counts
  IndexCopyKernelFunctor<output_t, index_t, OutputIt> kfn2(
      d_first, range_ptr, tpos_ptr);
  sycl_kernel_submit(sycl::range<1>(N), q, kfn2);

  return d_first + N;
}

template <typename T>
inline void swap_var(T& t1, T& t2) {
  T tmp = t1;
  t1 = t2;
  t2 = tmp;
}

template <typename KeyType, typename ValueType, typename CompFunc>
inline void compare_and_swap(
    KeyType& kA,
    ValueType& vA,
    KeyType& kB,
    ValueType& vB,
    bool dir,
    const CompFunc comp_t) {
  if (comp_t(kA, kB) == dir) {
    swap_var(kA, kB);
    swap_var(vA, vB);
  }
};

// bubble sort for the first round sorting
template <typename KeyType, typename ValueType, typename CompFunc>
inline void leaf_sort(
    const sycl::item<1>& item,
    KeyType* key,
    ValueType* val,
    size_t n,
    size_t sorted_sz,
    const CompFunc& comp_t) {
  auto start = item.get_linear_id() * n;
  auto end = std::min(start + n, sorted_sz);
  for (size_t i = start; i < end; ++i) {
    for (size_t j = start + 1; j < start + end - i; ++j) {
      // for stable sort, the condition should be:
      // if comp_t(key[j], key[j-1]), swap two elements;
      // so when key[j]==key[j-1], no swap.
      compare_and_swap(key[j], val[j], key[j - 1], val[j - 1], true, comp_t);
    }
  }
}

// lower_bound used in merge sort: pick up the elements in the sequence
// doesn't meet the compare situation with smallest index
template <typename KeyType, typename CompFunc>
inline size_t lower_bound(
    KeyType* in_data,
    size_t first,
    size_t last,
    const KeyType& key,
    const CompFunc& comp_t) {
  auto n = last - first;
  auto cur = n;
  size_t it;
  while (n > 0) {
    it = first;
    cur = n / 2;
    it += cur;
    if (comp_t(in_data[it], key)) {
      n -= cur + 1;
      first = ++it;
    } else {
      n = cur;
    }
  }
  return first;
}

template <class T, class ForwardIt, class InputIt, class OutputIt>
struct LowerBoundTenFunctor {
  void operator()(sycl::item<1> item_id) const {
    auto pilot = values_begin[item_id];
    auto N = std::distance(begin, end);
    auto cur = N;
    T first = 0;
    T it;
    while (N > 0) {
      it = first;
      cur = N / 2;
      it += cur;
      if (begin[it] < pilot) {
        N -= cur + 1;
        first = ++it;
      } else {
        N = cur;
      }
    }
    d_ptr[item_id] = first;
  }

  LowerBoundTenFunctor(
      ForwardIt begin,
      ForwardIt end,
      InputIt values_begin,
      OutputIt d_ptr)
      : begin(begin), end(end), values_begin(values_begin), d_ptr(d_ptr) {}

 private:
  ForwardIt begin;
  ForwardIt end;
  InputIt values_begin;
  OutputIt d_ptr;
};

template <class T, class ForwardIt, class InputIt, class OutputIt>
OutputIt lower_bound_tensor(
    ForwardIt begin,
    ForwardIt end,
    InputIt values_begin,
    InputIt values_end,
    OutputIt output) {
  RECORD_FUNCTION("lower_bound_tensor_xpu", {});
  // const auto N = std::distance(begin, end);
  const auto val_N = std::distance(values_begin, values_end);
  auto& q = getCurrentSYCLQueue();

  LowerBoundTenFunctor<T, ForwardIt, InputIt, OutputIt> kfn(
      begin, end, values_begin, output);
  sycl_kernel_submit(sycl::range<1>(val_N), q, kfn);

  return output + val_N;
}

template <typename KeyType, typename CompFunc>
inline size_t upper_bound(
    KeyType* in_data,
    size_t first,
    size_t last,
    const KeyType& key,
    const CompFunc& comp_t) {
  auto n = last - first;
  auto cur = n;
  size_t it;
  while (n > 0) {
    it = first;
    cur = n / 2;
    it += cur;
    if (!comp_t(key, in_data[it])) {
      n -= cur + 1;
      first = ++it;
    } else {
      n = cur;
    }
  }
  return first;
}

template <typename KeyType, typename ValueType, typename CompFunc>
inline void merge(
    const size_t offset,
    KeyType* in_key,
    ValueType* in_val,
    KeyType* out_key,
    ValueType* out_val,
    const size_t sq1_start,
    const size_t sq1_end,
    const size_t sq2_start,
    const size_t sq2_end,
    const size_t chunk_size,
    const CompFunc& comp_t) {
  const size_t chunk1_start = std::min((offset + sq1_start), sq1_end);
  const size_t chunk1_end = std::min((chunk1_start + chunk_size), sq1_end);
  const size_t chunk2_start = std::min((offset + sq2_start), sq2_end);
  const size_t chunk2_end = std::min((chunk2_start + chunk_size), sq2_end);

  const size_t chunk1_size = chunk1_end - chunk1_start;
  const size_t chunk2_size = chunk2_end - chunk2_start;

  size_t l_sq2_low_bound;
  size_t r_sq2_low_bound;
  size_t l_sq1_upper_bound;
  size_t r_sq1_upper_bound;
  if (!comp_t(in_key[sq2_start], in_key[sq1_end - 1])) {
    for (unsigned int i = 0; i < chunk1_size; ++i) {
      out_key[chunk1_start + i] = in_key[chunk1_start + i];
      out_val[chunk1_start + i] = in_val[chunk1_start + i];
    }

    for (unsigned int i = 0; i < chunk2_size; ++i) {
      out_key[chunk2_start + i] = in_key[chunk2_start + i];
      out_val[chunk2_start + i] = in_val[chunk2_start + i];
    }
  } else if (!comp_t(in_key[sq1_start], in_key[sq2_end - 1])) {
    auto out1_offset = sq2_end - sq2_start + chunk1_start;
    auto out2_offset = sq1_start + chunk2_start - sq2_start;
    for (unsigned int i = 0; i < chunk1_size; ++i) {
      out_key[out1_offset + i] = in_key[chunk1_start + i];
      out_val[out1_offset + i] = in_val[chunk1_start + i];
    }

    for (unsigned int i = 0; i < chunk2_size; ++i) {
      out_key[out2_offset + i] = in_key[chunk2_start + i];
      out_val[out2_offset + i] = in_val[chunk2_start + i];
    }
  } else {
    // Process 1st sequence
    if (chunk1_start < chunk1_end) {
      const auto chunk1_l_item = in_key[chunk1_start];
      l_sq2_low_bound =
          lower_bound(in_key, sq2_start, sq2_end, chunk1_l_item, comp_t);
      const auto l_shift1 = chunk1_start - sq1_start;
      const auto l_shift2 = l_sq2_low_bound - sq2_start;
      out_key[sq1_start + l_shift1 + l_shift2] = chunk1_l_item;
      out_val[sq1_start + l_shift1 + l_shift2] = in_val[chunk1_start];
      if (chunk1_end - chunk1_start > 1) {
        const auto chunk1_r_item = in_key[chunk1_end - 1];
        r_sq2_low_bound = lower_bound(
            in_key, l_sq2_low_bound, sq2_end, chunk1_r_item, comp_t);
        const auto r_shift1 = chunk1_end - 1 - sq1_start;
        const auto r_shift2 = r_sq2_low_bound - sq2_start;
        out_key[sq1_start + r_shift1 + r_shift2] = chunk1_r_item;
        out_val[sq1_start + r_shift1 + r_shift2] = in_val[chunk1_end - 1];
      }
      for (auto idx = chunk1_start + 1; idx < chunk1_end - 1; ++idx) {
        const auto inter_item_1 = in_key[idx];
        l_sq2_low_bound = lower_bound(
            in_key, l_sq2_low_bound, r_sq2_low_bound, inter_item_1, comp_t);
        const auto shift1 = idx - sq1_start;
        const auto shift2 = l_sq2_low_bound - sq2_start;
        out_key[sq1_start + shift1 + shift2] = inter_item_1;
        out_val[sq1_start + shift1 + shift2] = in_val[idx];
      }
    }
    // Process 2nd sequence
    if (chunk2_start < chunk2_end) {
      const auto chunk2_l_item = in_key[chunk2_start];
      l_sq1_upper_bound =
          upper_bound(in_key, sq1_start, sq1_end, chunk2_l_item, comp_t);
      const auto l_shift1 = l_sq1_upper_bound - sq1_start;
      const auto l_shift2 = chunk2_start - sq2_start;
      out_key[sq1_start + l_shift1 + l_shift2] = chunk2_l_item;
      out_val[sq1_start + l_shift1 + l_shift2] = in_val[chunk2_start];
      if (chunk2_end - chunk2_start > 1) {
        const auto chunk2_r_item = in_key[chunk2_end - 1];
        r_sq1_upper_bound = upper_bound(
            in_key, l_sq1_upper_bound, sq1_end, chunk2_r_item, comp_t);
        const auto r_shift1 = r_sq1_upper_bound - sq1_start;
        const auto r_shift2 = chunk2_end - 1 - sq2_start;
        out_key[sq1_start + r_shift1 + r_shift2] = chunk2_r_item;
        out_val[sq1_start + r_shift1 + r_shift2] = in_val[chunk2_end - 1];
      }

      for (auto idx = chunk2_start + 1; idx < chunk2_end - 1; ++idx) {
        const auto inter_item_2 = in_key[idx];
        l_sq1_upper_bound = upper_bound(
            in_key, l_sq1_upper_bound, r_sq1_upper_bound, inter_item_2, comp_t);
        const auto shift1 = l_sq1_upper_bound - sq1_start;
        const auto shift2 = idx - sq2_start;
        out_key[sq1_start + shift1 + shift2] = inter_item_2;
        out_val[sq1_start + shift1 + shift2] = in_val[idx];
      }
    }
  }
}

template <
    int vec_size,
    typename KeyType,
    typename ValueType,
    typename key_vec_t,
    typename val_vec_t>
struct VecCopyKernelImplFunctor {
  void operator()(sycl::item<1> item) const {
    auto item_id = item.get_linear_id();
    int remaining = sort_sz_ - item_id * vec_size;
    if (remaining < vec_size) {
      for (int index = 0; index < remaining; index++) {
        auto offset = item_id * vec_size + index;
        key_[offset] = tmp_key_data_[offset];
        val_[offset] = tmp_val_data_[offset];
      }
    } else {
#pragma unroll
      for (int index = 0; index < vec_size; index++) {
        key_vec_ptr_[item_id][index] = tmp_key_vec_ptr_[item_id][index];
        val_vec_ptr_[item_id][index] = tmp_val_vec_ptr_[item_id][index];
      }
    }
  }
  VecCopyKernelImplFunctor(
      KeyType* key,
      KeyType* tmp_key_data,
      ValueType* val,
      ValueType* tmp_val_data,
      const size_t sort_sz,
      key_vec_t* key_vec_ptr,
      key_vec_t* tmp_key_vec_ptr,
      val_vec_t* val_vec_ptr,
      val_vec_t* tmp_val_vec_ptr)
      : key_(key),
        tmp_key_data_(tmp_key_data),
        val_(val),
        tmp_val_data_(tmp_val_data),
        sort_sz_(sort_sz),
        key_vec_ptr_(key_vec_ptr),
        tmp_key_vec_ptr_(tmp_key_vec_ptr),
        val_vec_ptr_(val_vec_ptr),
        tmp_val_vec_ptr_(tmp_val_vec_ptr) {}

 private:
  KeyType* key_;
  KeyType* tmp_key_data_;
  ValueType* val_;
  ValueType* tmp_val_data_;
  const size_t sort_sz_;
  key_vec_t* key_vec_ptr_;
  key_vec_t* tmp_key_vec_ptr_;
  val_vec_t* val_vec_ptr_;
  val_vec_t* tmp_val_vec_ptr_;
};

template <int vec_size, typename KeyType, typename ValueType>
void vec_copy_kernel_impl(
    KeyType* key,
    KeyType* tmp_key_data,
    ValueType* val,
    ValueType* tmp_val_data,
    const size_t sort_sz) {
  auto& q = getCurrentSYCLQueue();
  using key_vec_t = at::native::memory::aligned_vector<KeyType, vec_size>;
  using val_vec_t = at::native::memory::aligned_vector<ValueType, vec_size>;
  key_vec_t* key_vec_ptr = reinterpret_cast<key_vec_t*>(key);
  key_vec_t* tmp_key_vec_ptr = reinterpret_cast<key_vec_t*>(tmp_key_data);
  val_vec_t* val_vec_ptr = reinterpret_cast<val_vec_t*>(val);
  val_vec_t* tmp_val_vec_ptr = reinterpret_cast<val_vec_t*>(tmp_val_data);
  auto num_work_item = ceil_div(sort_sz, (size_t)vec_size);
  VecCopyKernelImplFunctor<vec_size, KeyType, ValueType, key_vec_t, val_vec_t>
      kfn(key,
          tmp_key_data,
          val,
          tmp_val_data,
          sort_sz,
          key_vec_ptr,
          tmp_key_vec_ptr,
          val_vec_ptr,
          tmp_val_vec_ptr);
  sycl_kernel_submit(sycl::range<1>(num_work_item), q, kfn);
}

template <typename KeyType, typename ValueType>
void copy_to_dst(
    KeyType* key,
    KeyType* tmp_key_data,
    ValueType* val,
    ValueType* tmp_val_data,
    const size_t sort_sz) {
  int vec_size_key = at::native::memory::can_vectorize_up_to<KeyType>(
      reinterpret_cast<char*>(key));
  auto vec_size_val = at::native::memory::can_vectorize_up_to<ValueType>(
      reinterpret_cast<char*>(val));
  auto vec_size = std::min(vec_size_key, vec_size_val);

#define VEC_COPY_KERNEL_IMPL(vec_size)                  \
  {                                                     \
    vec_copy_kernel_impl<vec_size, KeyType, ValueType>( \
        key, tmp_key_data, val, tmp_val_data, sort_sz); \
  }

  switch (vec_size) {
    case 8: {
      VEC_COPY_KERNEL_IMPL(8);
      break;
    }
    case 4: {
      VEC_COPY_KERNEL_IMPL(4);
      break;
    }
    case 2: {
      VEC_COPY_KERNEL_IMPL(2);
      break;
    }
    case 1: {
      VEC_COPY_KERNEL_IMPL(1);
      break;
    }
    default:
      VEC_COPY_KERNEL_IMPL(1);
  }
#undef VEC_COPY_KERNEL_IMPL
}

template <typename KeyType, typename ValueType, typename CompFunc>
struct LeafSortKernelFunctor {
  void operator()(sycl::item<1> item) const {
    leaf_sort<KeyType, ValueType>(item, key_, val_, leaf_, sort_sz_, comp_t_);
  }
  LeafSortKernelFunctor(
      KeyType* key,
      ValueType* val,
      const size_t leaf,
      const size_t sort_sz,
      const CompFunc comp_t)
      : key_(key), val_(val), leaf_(leaf), sort_sz_(sort_sz), comp_t_(comp_t) {}

 private:
  KeyType* key_;
  ValueType* val_;
  const size_t leaf_;
  const size_t sort_sz_;
  const CompFunc comp_t_;
};

template <typename KeyType, typename ValueType, typename CompFunc>
struct MergeSortKernelFunctor {
  void operator()(sycl::item<1> item) const {
    const size_t idx = item.get_linear_id();
    const size_t sq1_start =
        std::min(sorted_pair_ * ((idx * chunk_) / sorted_), sort_sz_);
    const size_t sq1_end = std::min(sq1_start + sorted_, sort_sz_);
    const size_t sq2_start = sq1_end;
    const size_t sq2_end = std::min(sq2_start + sorted_, sort_sz_);

    const size_t offset_in_sq = chunk_ * (idx % chunk_num_per_sorted_);

    if (!data_in_tmp_) {
      merge(
          offset_in_sq,
          key_,
          val_,
          tmp_key_data_,
          tmp_val_data_,
          sq1_start,
          sq1_end,
          sq2_start,
          sq2_end,
          chunk_,
          comp_t_);
    } else {
      merge(
          offset_in_sq,
          tmp_key_data_,
          tmp_val_data_,
          key_,
          val_,
          sq1_start,
          sq1_end,
          sq2_start,
          sq2_end,
          chunk_,
          comp_t_);
    }
  }
  MergeSortKernelFunctor(
      size_t sorted_pair,
      size_t chunk_num_per_sorted,
      size_t chunk,
      size_t sorted,
      KeyType* key,
      ValueType* val,
      KeyType* tmp_key_data,
      ValueType* tmp_val_data,
      const size_t sort_sz,
      const CompFunc comp_t,
      bool data_in_tmp)
      : sorted_pair_(sorted_pair),
        chunk_num_per_sorted_(chunk_num_per_sorted),
        chunk_(chunk),
        sorted_(sorted),
        key_(key),
        val_(val),
        tmp_key_data_(tmp_key_data),
        tmp_val_data_(tmp_val_data),
        sort_sz_(sort_sz),
        comp_t_(comp_t),
        data_in_tmp_(data_in_tmp) {}

 private:
  size_t sorted_pair_;
  size_t chunk_num_per_sorted_;
  size_t chunk_;
  size_t sorted_;
  KeyType* key_;
  ValueType* val_;
  KeyType* tmp_key_data_;
  ValueType* tmp_val_data_;
  const size_t sort_sz_;
  const CompFunc comp_t_;
  bool data_in_tmp_;
};

// merge sort: only for 1d (single batch) tensor sort
template <typename KeyType, typename ValueType, typename CompFunc>
void merge_sort(
    KeyType* key,
    ValueType* val,
    const size_t sort_sz,
    const CompFunc comp_t) {
  RECORD_FUNCTION("merge_sort", {});
  const size_t leaf = 4;
  const size_t optimal_chunk = 4;

  const size_t leaf_step = ((sort_sz - 1) / leaf) + 1;
  auto& q = getCurrentSYCLQueue();

  // 1, leaf sort
  LeafSortKernelFunctor<KeyType, ValueType, CompFunc> kfn1(
      key, val, leaf, sort_sz, comp_t);
  sycl_kernel_submit(sycl::range<1>(leaf_step), q, kfn1);

  auto key_options = map_options<KeyType>();
  auto val_options = map_options<ValueType>();
  Tensor tmp_key = at::empty({static_cast<long>(sort_sz)}, key_options);
  Tensor tmp_val = at::empty({static_cast<long>(sort_sz)}, val_options);
  auto tmp_key_data = tmp_key.data_ptr<KeyType>();
  auto tmp_val_data = tmp_val.data_ptr<ValueType>();

  bool data_in_tmp = false;

  size_t sorted = leaf;
  size_t chunk = std::min(leaf, optimal_chunk);

  while (sorted < sort_sz) {
    size_t sorted_pair = 2 * sorted;
    size_t chunk_num_per_sorted = sorted / chunk;
    size_t full_pairs = sort_sz / sorted_pair;
    size_t incomplete_pair = sort_sz - sorted_pair * full_pairs;
    size_t first_block_in_incomplete_pair =
        incomplete_pair > sorted ? sorted : incomplete_pair;
    size_t incomplete_last_chunk = first_block_in_incomplete_pair % chunk != 0;
    size_t incomplete_pair_steps =
        first_block_in_incomplete_pair / chunk + incomplete_last_chunk;
    size_t full_pair_steps = full_pairs * chunk_num_per_sorted;
    size_t steps = full_pair_steps + incomplete_pair_steps;

    MergeSortKernelFunctor<KeyType, ValueType, CompFunc> kfn2(
        sorted_pair,
        chunk_num_per_sorted,
        chunk,
        sorted,
        key,
        val,
        tmp_key_data,
        tmp_val_data,
        sort_sz,
        comp_t,
        data_in_tmp);
    sycl_kernel_submit(sycl::range<1>(steps), q, kfn2);

    data_in_tmp = !data_in_tmp;
    sorted = sorted_pair;
    if (chunk < optimal_chunk)
      chunk *= 2;
  }
  if (data_in_tmp) {
    copy_to_dst<KeyType, ValueType>(
        key, tmp_key_data, val, tmp_val_data, sort_sz);
  }
}

// xpu::pstl::sort for non-batched tensor sort case.
// we have two sort API: one for user defined compare function; one for
// descending/ascending
//
// sort (out_key, out_val, sort_sz, comp_t)
// out_key: result of sort, it is a copy of tensor to be sorted
// out_val: indices of sort, it is initialized by [0, 1, 2, ...]
// sort_sz: element number to be sorted
// comp_t: compare function defined by user

// sort (in_key, out_key, out_val, sort_sz, descending)
// in_key: input tensor to be sorted
// out_key: result of sort, it is a copy of tensor to be sorted
// out_val: indices of sort, it is initialized by [0, 1, 2, ...]
// sort_sz: element number to be sorted
// descending: True for descending, False for ascending.
template <typename KeyType, typename ValueType, typename CompFunc>
void sort(
    KeyType* out_key,
    ValueType* out_val,
    const int64_t sort_sz,
    const CompFunc comp_t) {
  RECORD_FUNCTION("pstl::sort", {});
  merge_sort<KeyType, ValueType>(out_key, out_val, sort_sz, comp_t);
}

template <typename KeyType, typename ValueType>
void sort(
    const KeyType* in_key,
    KeyType* out_key,
    ValueType* out_val,
    const int64_t sort_sz,
    bool descending) {
  RECORD_FUNCTION("pstl::sort", {});
  sort_pairs<KeyType, ValueType>(
      in_key, out_key, nullptr, out_val, sort_sz, descending);
}

template <class T, class ForwardIt>
struct IotaKernelFunctor {
  void operator()(sycl::item<1> item_id) const {
    first_[item_id] = value_ + static_cast<T>(item_id);
  }
  IotaKernelFunctor(ForwardIt first, T value) : first_(first), value_(value) {}

 private:
  ForwardIt first_;
  T value_;
};

template <class T, class ForwardIt>
static inline void iota(ForwardIt first, ForwardIt last, T value) {
  RECORD_FUNCTION("iota_xpu", {});
  const auto N = std::distance(first, last);

  IotaKernelFunctor<T, ForwardIt> kfn(first, value);
  sycl_kernel_submit(sycl::range<1>(N), getCurrentSYCLQueue(), kfn);
}

template <
    typename output_t,
    class InputIt1,
    class InputIt2,
    class InputIt3,
    class ForwardIt,
    typename UnaryFunction,
    typename Predicate>
struct UnaryTransformIfWithStencilFunctor {
  void operator()(sycl::item<1> item_id) const {
    if (pred(stencil[item_id]))
      result[map[item_id]] = static_cast<output_t>(unary_op(first[item_id]));
  }

  UnaryTransformIfWithStencilFunctor(
      InputIt1 first,
      InputIt2 stencil,
      InputIt3 map,
      ForwardIt result,
      UnaryFunction unary_op,
      Predicate pred)
      : first{first},
        stencil(stencil),
        map(map),
        result(result),
        unary_op(unary_op),
        pred(pred) {}

 private:
  InputIt1 first;
  InputIt2 stencil;
  InputIt3 map;
  ForwardIt result;
  UnaryFunction unary_op;
  Predicate pred;
};

template <
    typename output_t,
    class InputIt1,
    class InputIt2,
    class InputIt3,
    class ForwardIt,
    typename UnaryFunction,
    typename Predicate>
ForwardIt transform_if(
    InputIt1 first,
    InputIt1 last,
    InputIt2 stencil,
    InputIt3 map,
    ForwardIt result,
    UnaryFunction unary_op,
    Predicate pred) {
  RECORD_FUNCTION("unary_transform_if_with_stencil_xpu", {});
  const auto N = std::distance(first, last);
  auto& q = getCurrentSYCLQueue();

  UnaryTransformIfWithStencilFunctor<
      output_t,
      InputIt1,
      InputIt2,
      InputIt3,
      ForwardIt,
      UnaryFunction,
      Predicate>
      kfn(first, stencil, map, result, unary_op, pred);
  sycl_kernel_submit(sycl::range<1>(N), q, kfn);

  return result + N;
}

template <
    class T,
    class InputIt1,
    class InputIt2,
    class InputIt3,
    class RandomAccessIt>
void scatter_if(
    InputIt1 first,
    InputIt1 last,
    InputIt2 map,
    InputIt3 stencil,
    RandomAccessIt output) {
  // default predicate is identity
  // typedef typename std::iterator_value<InputIterator3>::type StencilType;
  IdentityPred identity;
  scatter_if<T>(first, last, map, stencil, output, identity);
}

template <
    class T,
    class InputIt1,
    class InputIt2,
    class InputIt3,
    class RandomAccessIt,
    typename Predicate>
void scatter_if(
    InputIt1 first,
    InputIt1 last,
    InputIt2 map,
    InputIt3 stencil,
    RandomAccessIt output,
    Predicate pred) {
  // typedef typename std::iterator_value<InputIt1>::type InputType;
  // pstl::transform_if(
  //     first,
  //     last,
  //     stencil,
  //     thrust::make_permutation_iterator(output, map),
  //     std::identity<InputType>(),
  //     pred);
  IdentityPred identity;
  pstl::transform_if<T>(first, last, stencil, map, output, identity, pred);
}

template <
    class ValueType,
    class InputIt1,
    class InputIt2,
    class OutputIt1,
    class OutputIt2,
    typename BinaryPredicate,
    typename BinaryFunction>
OutputIt2 reduce_by_key(
    InputIt1 keys_first,
    InputIt1 keys_last,
    InputIt2 values_first,
    OutputIt1 keys_output,
    OutputIt2 values_output,
    BinaryPredicate binary_pred,
    BinaryFunction binary_op) {
  // typedef ValueType FlagType;

  if (keys_first == keys_last)
    return values_output;

  // input size
  auto N = std::distance(keys_first, keys_last);

  auto values_last = values_first + N;
  auto flag_options = map_options<ValueType>();

  // compute head flags
  Tensor head_flags = at::ones({N}, flag_options);
  auto head_flags_first = head_flags.data_ptr<ValueType>();
  Not2Pred<BinaryPredicate> not2(binary_pred);
  pstl::transform<ValueType>(
      keys_first, keys_last - 1, keys_first + 1, head_flags_first + 1, not2);

  // compute tail flags
  Tensor tail_flags = at::ones({N}, flag_options);
  auto tail_flags_first = tail_flags.data_ptr<ValueType>();
  pstl::transform<ValueType>(
      keys_first, keys_last - 1, keys_first + 1, tail_flags_first, not2);

  auto value_options = map_options<ValueType>();
  // scan the values by flag
  Tensor scanned_values = at::zeros({N}, value_options);
  auto scanned_values_first = scanned_values.data_ptr<ValueType>();

  Tensor scanned_tail_flags = at::ones({N}, flag_options);
  ValueType* scanned_tail_flags_first =
      scanned_tail_flags.data_ptr<ValueType>();

  pstl::inclusive_scan_if<int64_t>(
      values_first,
      values_last,
      head_flags_first,
      scanned_values_first,
      binary_op);

  pstl::exclusive_scan(
      tail_flags_first,
      tail_flags_first + N,
      scanned_tail_flags_first,
      static_cast<ValueType>(0));

  // number of unique keys
  int64_t new_sz =
      pstl::get_item<int64_t>(
          scanned_tail_flags_first, scanned_tail_flags_first + N, N - 1) +
      1;

  // scatter the keys and accumulated values
  pstl::scatter_if<ValueType>(
      keys_first,
      keys_last,
      scanned_tail_flags_first,
      head_flags_first,
      keys_output);

  pstl::scatter_if<ValueType>(
      scanned_values_first,
      scanned_values_first + N,
      scanned_tail_flags_first,
      tail_flags_first,
      values_output);

  return values_output + new_sz;
}

template <
    class ValueType,
    class InputIt1,
    class InputIt2,
    class OutputIt1,
    class OutputIt2>
OutputIt2 reduce_by_key(
    InputIt1 keys_first,
    InputIt1 keys_last,
    InputIt2 values_first,
    OutputIt1 keys_output,
    OutputIt2 values_output) {
  // use equal_to<ValueType> as default BinaryPredicate
  return reduce_by_key<ValueType>(
      keys_first,
      keys_last,
      values_first,
      keys_output,
      values_output,
      std::equal_to<ValueType>());
}

template <
    class ValueType,
    class InputIt1,
    class InputIt2,
    class OutputIt1,
    class OutputIt2,
    typename BinaryPredicate>
OutputIt2 reduce_by_key(
    InputIt1 keys_first,
    InputIt1 keys_last,
    InputIt2 values_first,
    OutputIt1 keys_output,
    OutputIt2 values_output,
    BinaryPredicate binary_pred) {
  // use plus<T> as default BinaryFunction

  // typedef int64_t FlagType;
  PlusPred plus;
  return reduce_by_key<ValueType>(
      keys_first,
      keys_last,
      values_first,
      keys_output,
      values_output,
      binary_pred,
      plus);
}

} // namespace at::native::xpu::pstl
