/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#pragma once

#include <ATen/native/CanUse32BitIndexMath.h>
#include <ATen/native/Copy.h>
#include <ATen/native/xpu/sycl/IndexUtils.h>
#include <comm/SYCLContext.h>
#include <comm/TensorInfo.h>
#include <math.h>

//
// This file contains pointwise operation functions and kernels that
// work on both contiguous and non-contiguous tensor arguments of
// arbitrary (up to XPU_MAX_TENSORINFO_DIMS) dimensioned arguments without
// copying or temporary storage.
//

namespace at {
namespace native {
namespace xpu {

using namespace at::xpu::detail;

enum class TensorArgType { ReadWrite, ReadOnly };

// Rearrange dimensions for pointwise operations so that strides are in
// decreasing order as much as possible, so that kernels have better memory
// access patterns.

template <
    typename T1,
    typename IndexType,
    typename T2 = void,
    typename T3 = void,
    typename T4 = void>
inline void rearrangeDims(
    TensorInfo<T1, IndexType>* aInfo,
    TensorInfo<T2, IndexType>* bInfo = nullptr,
    TensorInfo<T3, IndexType>* cInfo = nullptr,
    TensorInfo<T4, IndexType>* dInfo = nullptr) {
  int numInfos = 1;
  int dims = aInfo->dims;
  IndexType* sizes[4] = {
      aInfo->sizes,
  };
  IndexType* strides[4] = {
      aInfo->strides,
  };

  if (bInfo != nullptr) {
    ++numInfos;
    if (bInfo->dims != dims)
      return;
    sizes[1] = bInfo->sizes;
    strides[1] = bInfo->strides;
  }

  if (cInfo != nullptr) {
    ++numInfos;
    if (cInfo->dims != dims)
      return;
    sizes[2] = cInfo->sizes;
    strides[2] = cInfo->strides;
  }

  if (dInfo != nullptr) {
    ++numInfos;
    if (dInfo->dims != dims)
      return;
    sizes[3] = dInfo->sizes;
    strides[3] = dInfo->strides;
  }

  // Bail out if sizes do not match: we are using "deprecated pointwise
  // behavior" among tensors of different shapes but same number of elements.
  for (int i = 1; i < numInfos; ++i) {
    for (int j = 0; j < dims; ++j) {
      if (sizes[i][j] != sizes[0][j])
        return;
    }
  }

  for (int i = 0; i < dims - 1; ++i) {
    // No need to consider dimensions of size 1.
    if (sizes[0][i] == 1)
      continue;

    for (int j = i + 1; j < dims; ++j) {
      if (sizes[0][j] == 1)
        continue;

      // Compare the relative sizes of strides between dim #i and dim #j.
      bool hasIncreasingStrides = false;
      bool hasDecreasingStrides = false;

      for (int k = 0; k < numInfos; k++) {
        IndexType stride_i = strides[k][i];
        IndexType stride_j = strides[k][j];
        if (stride_i < stride_j) {
          hasIncreasingStrides = true;
        } else if (stride_i > stride_j) {
          hasDecreasingStrides = true;
        }
      }

      if (hasIncreasingStrides && !hasDecreasingStrides) {
        for (int k = 0; k < numInfos; k++) {
          IndexType size = sizes[k][i];
          sizes[k][i] = sizes[k][j];
          sizes[k][j] = size;

          IndexType stride = strides[k][i];
          strides[k][i] = strides[k][j];
          strides[k][j] = stride;
        }
      }
    }
  }
}

template <
    typename Op,
    typename scalar1,
    typename scalar2,
    typename IndexType,
    int remaining_steps,
    typename... Offsets>
struct ApplyOp2 {
  inline static void apply(
      sycl::nd_item<1>& item,
      TensorInfo<scalar1, IndexType> a,
      TensorInfo<scalar2, IndexType> b,
      const Op& op,
      int64_t n,
      IndexType linearIndex,
      Offsets... aOffsets,
      Offsets... bOffsets) {
    // Convert `linearIndex` into an offset of `a`
    const IndexType aOffset = static_cast<int64_t>(sizeof...(Offsets)) < n
        ? IndexToOffset<scalar1, IndexType, -1>::get(linearIndex, a)
        : 0;

    // Convert `linearIndex` into an offset of `b`
    const IndexType bOffset = static_cast<int64_t>(sizeof...(Offsets)) < n
        ? IndexToOffset<scalar2, IndexType, -1>::get(linearIndex, b)
        : 0;

    ApplyOp2<
        Op,
        scalar1,
        scalar2,
        IndexType,
        remaining_steps - 1,
        const IndexType,
        Offsets...>::
        apply(
            item,
            a,
            b,
            op,
            n,
            linearIndex + 1,
            aOffsets...,
            aOffset,
            bOffsets...,
            bOffset);
  }
};

// Specialize `step=1` case (i.e., `remaining_steps=0` and `len(Offsets)=1`).
// We don't need to pass in how many elements need to processed in this case.
template <
    typename Op,
    typename scalar1,
    typename scalar2,
    typename IndexType,
    typename Offset>
struct ApplyOp2<Op, scalar1, scalar2, IndexType, 0, Offset> {
  inline static void apply(
      sycl::nd_item<1>& item,
      TensorInfo<scalar1, IndexType> a,
      TensorInfo<scalar2, IndexType> b,
      const Op& op,
      int /*n*/,
      IndexType /*linearIndex*/,
      Offset aOffset,
      Offset bOffset) {
    op(item, a.data[aOffset], b.data[bOffset]);
  }
};

template <
    typename Op,
    typename scalar1,
    typename scalar2,
    typename IndexType,
    typename... Offsets>
struct ApplyOp2<Op, scalar1, scalar2, IndexType, 0, Offsets...> {
  inline static void apply(
      sycl::nd_item<1>& item,
      TensorInfo<scalar1, IndexType> a,
      TensorInfo<scalar2, IndexType> b,
      const Op& op,
      int n,
      IndexType linearIndex,
      Offsets... aOffsets,
      Offsets... bOffsets) {
    op(item, n, a.data[aOffsets]..., b.data[bOffsets]...);
  }
};

template <
    typename Op,
    typename scalar1,
    typename scalar2,
    typename IndexType,
    int step>
struct PointwiseApply2Functor {
  void operator()(sycl::nd_item<1> item) const {
    for (IndexType linearIndex = (item.get_group(0) * item.get_local_range(0) +
                                  item.get_local_id(0)) *
             step;
         linearIndex < totalElements_;
         linearIndex +=
         item.get_group_range(0) * item.get_local_range(0) * step) {
      ApplyOp2<Op, scalar1, scalar2, IndexType, step>::apply(
          item,
          a_,
          b_,
          op_,
          std::min(step, static_cast<int>(totalElements_ - linearIndex)),
          linearIndex);
    }
  }
  PointwiseApply2Functor(
      TensorInfo<scalar1, IndexType> a,
      TensorInfo<scalar2, IndexType> b,
      IndexType totalElements,
      const Op op)
      : a_(a), b_(b), totalElements_(totalElements), op_(op) {}

 private:
  TensorInfo<scalar1, IndexType> a_;
  TensorInfo<scalar2, IndexType> b_;
  IndexType totalElements_;
  const Op op_;
};

template <int step = 1>
inline uint64_t get_apply_group_count(
    uint64_t total_elements,
    int threads_per_group) {
  uint64_t numel_per_thread =
      static_cast<uint64_t>(threads_per_group) * static_cast<uint64_t>(step);
  uint64_t num_groups =
      (total_elements + numel_per_thread - 1) / numel_per_thread;
  uint64_t estimated_max_groups_per_tile =
      syclMaxWorkItemsPerTile() / threads_per_group;
  if (num_groups > estimated_max_groups_per_tile)
    num_groups = estimated_max_groups_per_tile;
  return num_groups;
}

template <
    typename scalar1,
    typename scalar2,
    int step,
    typename Op,
    int threads_per_group>
inline bool tensor_apply2(
    at::TensorBase& a,
    at::TensorBase& b,
    const Op op,
    TensorArgType aType = TensorArgType::ReadWrite,
    TensorArgType bType = TensorArgType::ReadOnly) {
  TORCH_CHECK(
      a.device().is_xpu() && b.device().is_xpu(),
      "tensor_apply2: Expected tensors to have XPU DeviceType, but got "
      "tensors with type ",
      a.device().type(),
      " and ",
      b.device().type());
  int64_t totalElements = a.numel();

  if (totalElements != b.numel()) {
    return false;
  }

  if (a.dim() > XPU_MAX_TENSORINFO_DIMS || b.dim() > XPU_MAX_TENSORINFO_DIMS) {
    return false;
  }

  if (a.numel() == 0) {
    // Empty tensor; do nothing
    return true;
  }

  int64_t group_count =
      get_apply_group_count<step>(totalElements, threads_per_group);

  /*
  Expands readable/writable tensors whose indices may be "overlapped."
  This ensures that each element of the tensor is operated on once and only
  once.
  */
  TensorBase oldA;
  TensorBase oldB;

  if (aType == TensorArgType::ReadWrite &&
      at::xpu::detail::maybeOverlappingIndices(a)) {
    // Must perform in contiguous space
    oldA = std::exchange(a, a.contiguous());
  }
  if (bType == TensorArgType::ReadWrite &&
      at::xpu::detail::maybeOverlappingIndices(b)) {
    // Must perform in contiguous space
    oldB = std::exchange(b, b.contiguous());
  }

  if (canUse32BitIndexMath(a) && canUse32BitIndexMath(b)) {
    TensorInfo<scalar1, unsigned int> aInfo =
        getTensorInfo<scalar1, unsigned int>(a);

    TensorInfo<scalar2, unsigned int> bInfo =
        getTensorInfo<scalar2, unsigned int>(b);
    rearrangeDims(&aInfo, &bInfo);
    aInfo.collapseDims();
    bInfo.collapseDims();

    using index_t = unsigned int;
    auto fn = PointwiseApply2Functor<Op, scalar1, scalar2, index_t, step>(
        aInfo, bInfo, static_cast<index_t>(totalElements), op);
    sycl_kernel_submit(
        group_count * threads_per_group,
        threads_per_group,
        getCurrentSYCLQueue(),
        fn);
  } else {
    TensorInfo<scalar1, uint64_t> aInfo = getTensorInfo<scalar1, uint64_t>(a);

    TensorInfo<scalar2, uint64_t> bInfo = getTensorInfo<scalar2, uint64_t>(b);
    rearrangeDims(&aInfo, &bInfo);
    aInfo.collapseDims();
    bInfo.collapseDims();

    using index_t = uint64_t;
    auto fn = PointwiseApply2Functor<Op, scalar1, scalar2, index_t, step>(
        aInfo, bInfo, static_cast<index_t>(totalElements), op);
    sycl_kernel_submit(
        group_count * threads_per_group,
        threads_per_group,
        getCurrentSYCLQueue(),
        fn);
  }

  if (oldA.defined()) {
    at::native::copy_ignoring_overlaps(oldA, a);
  }

  if (oldB.defined()) {
    at::native::copy_ignoring_overlaps(oldB, b);
  }

  return true;
}

/* Provides default step = 1 to tensor_apply2. */
template <
    typename scalar1,
    typename scalar2,
    typename Op,
    int max_threads_per_group>
inline bool tensor_apply2(
    at::TensorBase& a,
    at::TensorBase& b,
    const Op op,
    TensorArgType aType = TensorArgType::ReadWrite,
    TensorArgType bType = TensorArgType::ReadOnly) {
  return tensor_apply2<scalar1, scalar2, 1, Op, max_threads_per_group>(
      a, b, op, aType, bType);
}

} // namespace xpu
} // namespace native
} // namespace at
