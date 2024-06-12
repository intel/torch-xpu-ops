#pragma once

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/core/Array.h>
#include <ATen/detail/FunctionTraits.h>
#include <aten/sycl/MemoryAccess.h>
#include <comm/SYCLContext.h>
#include <comm/XPUMathCompat.h>

namespace at {
namespace native {
namespace xpu {

template <typename T, class ReduceOp, int SIMD>
inline T SubgroupReduce(T val, const ReduceOp& op) {
#pragma unroll
  for (int offset = 1; offset < SIMD; offset <<= 1) {
    val = op.combine(val, op.shfl_down(val, offset));
  }
  return val;
}

template <typename T, class ReduceOp, int SIMD, typename shared_t>
inline T GroupReduce(
    sycl::nd_item<1>& item,
    T val,
    const ReduceOp& op,
    const T& identity_element,
    shared_t shared) {
  int tid = item.get_local_linear_id();
  int sg_tid = tid % SIMD;
  int sg_id = tid / SIMD;
  int sg_n = item.get_local_range(0) / SIMD;
  val = SubgroupReduce<T, ReduceOp, SIMD>(val, op);
  item.barrier(
      sycl::access::fence_space::local_space); // prevent races when GroupReduce
                                               // are called in a row.
  if (sg_tid == 0) {
    shared[sg_id] = val;
  }
  item.barrier(sycl::access::fence_space::local_space);
  val = (tid < sg_n) ? shared[sg_tid] : identity_element;
  if (sg_id == 0) {
    val = SubgroupReduce<T, ReduceOp, SIMD>(val, op);
  }
  return val;
}

} // namespace xpu
} // namespace native
} // namespace at
