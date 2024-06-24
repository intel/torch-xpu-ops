#pragma once

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/core/Array.h>
#include <ATen/detail/FunctionTraits.h>
#include <ATen/native/xpu/sycl/MemoryAccess.h>
#include <comm/SYCLContext.h>
#include <comm/XPUMathCompat.h>

namespace at {
namespace native {
namespace xpu {

inline size_t get_group_reduce_group_size() {
  return syclMaxWorkGroupSize() / 2;
}

template <int DIM>
inline size_t get_local_linear_range(sycl::nd_item<DIM>& item) {
  size_t n = item.get_local_range(0);
#pragma unroll
  for (int i = 1; i < DIM; ++i) {
    n *= item.get_local_range(i);
  }
  return n;
}

template <typename T, int SIMD, int DIM>
inline T& SubgroupReduceSum(sycl::nd_item<DIM>& item, T& val) {
  auto sg = item.get_sub_group();
  auto sg_tid = sg.get_local_linear_id();
#pragma unroll
  for (int offset = 1; offset < SIMD; offset <<= 1) {
    T temp = sycl::shift_group_left(sg, val, offset);
    if (sg_tid < SIMD - offset) {
      val += temp;
    }
  }
  return val;
}

template <typename T, int SIMD, typename shared_t, int DIM>
inline T& GroupReduceSum(sycl::nd_item<DIM>& item, T& val, shared_t shared) {
  auto sg = item.get_sub_group();
  int sg_tid = sg.get_local_linear_id();
  int sg_id = sg.get_group_linear_id();
  int n_sg = get_local_linear_range<DIM>(item) / SIMD;
  val = SubgroupReduceSum<T, SIMD, DIM>(item, val);
  item.barrier(sycl_local_fence); // prevent races when GroupReduceSum are
                                  // called in a row.
  if (n_sg == 1) {
    return val;
  }
  if (sg_tid == 0) {
    shared[sg_id] = val;
  }
  item.barrier(sycl_local_fence);
  if (sg_id == 0) {
    for (int i = 1; i < n_sg; i++) {
      val += shared[i];
    }
  }
  return val;
}

template <typename T, class ReduceOp, int SIMD, int DIM>
inline T& SubgroupReduce(sycl::nd_item<DIM>& item, T& val, const ReduceOp& op) {
  auto sg = item.get_sub_group();
  auto sg_tid = sg.get_local_linear_id();
#pragma unroll
  for (int offset = 1; offset < SIMD; offset <<= 1) {
    T temp = sycl::shift_group_left(sg, val, offset);
    if (sg_tid < SIMD - offset) {
      val = op.combine(val, temp);
    }
  }
  return val;
}

template <typename T, class ReduceOp, int SIMD, typename shared_t, int DIM>
inline T& GroupReduce(
    sycl::nd_item<DIM>& item,
    T& val,
    const ReduceOp& op,
    shared_t shared) {
  auto sg = item.get_sub_group();
  int sg_tid = sg.get_local_linear_id();
  int sg_id = sg.get_group_linear_id();
  int n_sg = get_local_linear_range<DIM>(item) / SIMD;
  val = SubgroupReduce<T, ReduceOp, SIMD, DIM>(item, val, op);
  item.barrier(sycl_local_fence); // prevent races when GroupReduce
                                  // are called in a row.
  if (n_sg == 1) {
    return val;
  }
  if (sg_tid == 0) {
    shared[sg_id] = val;
  }
  item.barrier(sycl_local_fence);
  if (sg_id == 0) {
    for (int i = 1; i < n_sg; i++) {
      val = op.combine(val, shared[i]);
    }
  }
  return val;
}

#define SIMD16 16
#define SIMD32 32

template <
    typename fn_simd_16,
    typename fn_simd_32,
    typename range_t,
    typename... args_t>
static inline void group_norm_kernel_simd_choice_and_launch(
    int simd,
    range_t global_range,
    range_t local_range,
    ::sycl::queue q,
    args_t... args) {
  switch (simd) {
    case 16: {
      auto fn = fn_simd_16(args...);
      sycl_kernel_submit(global_range, local_range, q, fn);
    } break;
    default: {
      auto fn = fn_simd_32(args...);
      sycl_kernel_submit(global_range, local_range, q, fn);
    } break;
  }
}

} // namespace xpu
} // namespace native
} // namespace at
