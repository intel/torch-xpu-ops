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

#include <ATen/native/xpu/sycl/BatchKernel.h>
#include <ATen/native/xpu/sycl/IndexKernelUtils.h>
#include <ATen/native/xpu/sycl/Loops.h>
#include <ATen/native/xpu/sycl/MemoryAccess.h>
#include <comm/TensorInfo.h>

using namespace at::xpu::detail;
using namespace at::xpu;

namespace at::native::xpu {

template <int N>
struct alignas(N) OpaqueType {
  char data[N];
};

// Pretend that the scalar tensor is in fact a one-element vector.
template <typename T, typename IndexType>
TensorInfo<T, IndexType> tensorInfoIfScalar(TensorInfo<T, IndexType> ti) {
  if (ti.dims == 0) {
    ti.dims = 1;
    ti.sizes[0] = 1;
    ti.strides[0] = 1;
  }
  return ti;
}

template <class SrcInfo, class DstInfo, class IdxInfo, class FuncType>
class IndexKernelConfig : public BatchKernelConfig {
 public:
  using ValType = typename DstInfo::scalar_t;
  using IdxType = typename IdxInfo::scalar_t;

  IndexKernelConfig() = delete;
  IndexKernelConfig(
      SrcInfo& sinfo,
      DstInfo& dinfo,
      IdxInfo& iinfo,
      ValType alpha,
      int64_t index_num,
      int64_t indexing_dimension_size,
      bool indexing_dst,
      bool problem_inner,
      FuncType func,
      int64_t batch,
      int64_t problem,
      int64_t stride,
      int64_t problem_batch,
      bool problem_along_x)
      : BatchKernelConfig(
            batch,
            problem,
            stride,
            problem_batch,
            problem_along_x,
            Policy::pSegment,
            syclMaxWorkItemsPerSubSlice()),
        sinfo_(sinfo),
        dinfo_(dinfo),
        iinfo_(iinfo),
        alpha_(alpha),
        index_num_(index_num),
        indexing_dimension_size_(indexing_dimension_size),
        indexing_dst_(indexing_dst),
        problem_inner_(problem_inner),
        func_(func) {}

  template <class TarInfo>
  static inline void indexing_problem_mapping(
      TarInfo& tinfo,
      IdxInfo& iinfo,
      int dim,
      int64_t index_num,
      int64_t indexing_dimension_size,
      int64_t& batch,
      int64_t& problem,
      int64_t& stride,
      int64_t& problem_batch,
      bool& problem_along_x,
      bool& problem_inner) {
    int64_t outer = tinfo.outerSize(dim);
    int64_t inner = tinfo.innerSize(dim);

    if (inner == 1) {
      problem = outer;
      stride = indexing_dimension_size;
      batch = 1;
      problem_batch = index_num;
      problem_along_x = tinfo.strides[dim] == 1 ? false : true;
      problem_inner = false;
    } else if (outer == 1) {
      problem = inner;
      stride = 1;
      batch = indexing_dimension_size;
      problem_batch = index_num;
      problem_along_x = tinfo.strides[tinfo.dims - 1] == 1 ? true : false;
      problem_inner = true;
    } else {
      problem = inner;
      stride = 1;
      batch = outer * indexing_dimension_size;
      problem_batch = outer * index_num;
      problem_along_x = tinfo.strides[tinfo.dims - 1] == 1 ? true : false;
      problem_inner = true;
    }
    return;
  }

  template <class KernelClass>
  static IndexKernelConfig<SrcInfo, DstInfo, IdxInfo, FuncType> make_config(
      SrcInfo& src_info,
      DstInfo& dst_info,
      IdxInfo& index_info,
      ValType alpha,
      int64_t dim,
      bool indexing_dst,
      FuncType func) {
    int64_t index_num = index_info.sizes[0];
    int64_t indexing_dimension_size;

    bool problem_along_x, problem_inner;
    int64_t batch, problem, stride, problem_batch;

    TORCH_CHECK(
        indexing_dst || src_info.data != nullptr,
        "Indexing kernel backbone does not support null src ...");

    if (indexing_dst) {
      indexing_dimension_size = dst_info.sizes[dim];
      indexing_problem_mapping(
          dst_info,
          index_info,
          dim,
          index_num,
          indexing_dimension_size,
          batch,
          problem,
          stride,
          problem_batch,
          problem_along_x,
          problem_inner);
    } else {
      indexing_dimension_size = src_info.sizes[dim];
      indexing_problem_mapping(
          src_info,
          index_info,
          dim,
          index_num,
          indexing_dimension_size,
          batch,
          problem,
          stride,
          problem_batch,
          problem_along_x,
          problem_inner);
    }

    IndexKernelConfig<SrcInfo, DstInfo, IdxInfo, FuncType> cfg = {
        src_info,
        dst_info,
        index_info,
        alpha,
        index_num,
        indexing_dimension_size,
        indexing_dst,
        problem_inner,
        func,
        batch,
        problem,
        stride,
        problem_batch,
        problem_along_x};

    cfg.template build<KernelClass>();
    return cfg;
  }

 public:
  SrcInfo sinfo_; // sinfo_.data could be nullptr, while indexing along dst.
  DstInfo dinfo_;
  IdxInfo iinfo_;
  ValType alpha_;
  int64_t index_num_;
  int64_t indexing_dimension_size_;
  bool indexing_dst_;
  bool problem_inner_;
  FuncType func_;
};

template <
    class IdxConfig,
    bool TrivialOffCal = false,
    bool KnownProblemInner = false>
class IndexKernel {
 public:
  using ValType = typename IdxConfig::ValType;
  using IdxType = typename IdxConfig::IdxType;

  IndexKernel() = delete;
  IndexKernel(IdxConfig& cfg) : cfg_(cfg) {}

  void init_global_batch_info(
      BatchKernelConfig::ItemDesc& id,
      int64_t& idx_logical_off,
      int64_t& glb_batch_group,
      int64_t& glb_batch_group_loc_off) const {
    idx_logical_off = id.glb_batch % cfg_.index_num_;
    int64_t idx_off;
    if constexpr (TrivialOffCal) {
      idx_off = idx_logical_off;
    } else {
      idx_off = IndexToOffset<IdxType, int64_t, -1>::get(
          idx_logical_off, cfg_.iinfo_);
    }
    glb_batch_group = id.glb_batch / cfg_.index_num_;
    glb_batch_group_loc_off = cfg_.iinfo_.data[idx_off];
    glb_batch_group_loc_off = glb_batch_group_loc_off >= 0
        ? glb_batch_group_loc_off
        : cfg_.indexing_dimension_size_ + glb_batch_group_loc_off;
  }

  int64_t inline indexing_logical_off(
      BatchKernelConfig::ItemDesc& id,
      int64_t glb_batch_group,
      int64_t glb_batch_group_loc_off) const {
    int64_t si, pi, bi;
    int64_t glb_batch_group_glb_off =
        glb_batch_group * cfg_.indexing_dimension_size_ +
        glb_batch_group_loc_off;
    auto stride = cfg_.stride_;
    if constexpr (KnownProblemInner) {
      si = 0;
      pi = id.glb_problem;
      bi = glb_batch_group_glb_off;
      return (pi + bi * cfg_.problem_) * stride;
    } else {
      if (cfg_.problem_inner_) {
        si = 0;
        pi = id.glb_problem;
        bi = glb_batch_group_glb_off;
        return (pi + bi * cfg_.problem_) * stride;
      } else {
        si = glb_batch_group_glb_off;
        pi = id.glb_problem;
        bi = 0;
        return si + pi * stride;
      }
    }
  }

  int64_t inline fixing_logical_off(
      BatchKernelConfig::ItemDesc& id,
      int64_t glb_batch_group,
      int64_t idx_logical_off) const {
    int64_t si, pi, bi, stride;
    int64_t glb_batch_group_glb_off =
        glb_batch_group * cfg_.index_num_ + idx_logical_off;
    if constexpr (KnownProblemInner) {
      si = 0;
      stride = 1;
      pi = id.glb_problem;
      bi = glb_batch_group_glb_off;
      return pi + bi * cfg_.problem_;
    } else {
      if (cfg_.problem_inner_) {
        si = 0;
        stride = 1;
        pi = id.glb_problem;
        bi = glb_batch_group_glb_off;
        return pi + bi * cfg_.problem_;
      } else {
        bi = 0;
        si = glb_batch_group_glb_off;
        pi = id.glb_problem;
        stride = cfg_.index_num_;
        return si + pi * stride;
      }
    }
  }

  void operator()(sycl::nd_item<2> item) const {
    auto id = cfg_.get_item_desc(item);

    if (id.glb_problem >= cfg_.problem_ ||
        id.glb_batch >= cfg_.problem_batch_) {
      return;
    }

    // Indexing kernels have three operands,
    // 1. index operand
    // 2. operand indexing on
    // 3. operand has fixing size as index (optional)
    int64_t idx_logical_off, glb_batch_group, glb_batch_group_loc_off;
    int64_t glb_indexing_logical_off, glb_fixing_logical_off;
    int64_t dst_off, src_off;

    init_global_batch_info(
        id, idx_logical_off, glb_batch_group, glb_batch_group_loc_off);

    glb_indexing_logical_off =
        indexing_logical_off(id, glb_batch_group, glb_batch_group_loc_off);

    if (cfg_.sinfo_.data != nullptr && cfg_.dinfo_.data != nullptr) {
      glb_fixing_logical_off =
          fixing_logical_off(id, glb_batch_group, idx_logical_off);
    }

    if constexpr (TrivialOffCal) {
      if (cfg_.indexing_dst_) {
        dst_off = glb_indexing_logical_off;
        if (cfg_.sinfo_.data != nullptr) {
          src_off = glb_fixing_logical_off;
        }
      } else {
        src_off = glb_indexing_logical_off;
        dst_off = glb_fixing_logical_off;
      }
    } else {
      if (cfg_.indexing_dst_) {
        // index_copy, index_add, index_fill
        dst_off = IndexToOffset<ValType, int64_t, -1>::get(
            glb_indexing_logical_off, cfg_.dinfo_);
        if (cfg_.sinfo_.data != nullptr) {
          src_off = IndexToOffset<const ValType, int64_t, -1>::get(
              glb_fixing_logical_off, cfg_.sinfo_);
        }
      } else {
        // index_select
        src_off = IndexToOffset<const ValType, int64_t, -1>::get(
            glb_indexing_logical_off, cfg_.sinfo_);
        dst_off = IndexToOffset<ValType, int64_t, -1>::get(
            glb_fixing_logical_off, cfg_.dinfo_);
      }
    }
    cfg_.func_(
        cfg_.dinfo_.data,
        cfg_.sinfo_.data,
        dst_off,
        src_off,
        glb_batch_group_loc_off,
        cfg_.alpha_);
  }

 private:
  IdxConfig cfg_;
};

template <
    class IdxConfig,
    bool TrivialOffCal = false,
    bool KnownProblemInner = false>
static inline void launch_index_kernel(IdxConfig& cfg) {
  auto& queue = getCurrentSYCLQueue();
  IndexKernel<IdxConfig, TrivialOffCal, KnownProblemInner> idx_ker(cfg);
  sycl_kernel_submit(cfg.global_size(), cfg.group_size(), queue, idx_ker);
}

template <typename func_t, typename index_buf_type>
struct SmallIndexKernelFunctor : public __SYCL_KER_CONFIG_CONVENTION__ {
  void operator()(sycl::nd_item<1> item_id) const {
    auto local_id = item_id.get_local_id(0);
    auto group_id = item_id.get_group(0);

    // construct a indices_size table on SLM
    for (int64_t local_index = local_id; local_index < indices_size_;
         local_index += wgroup_size_) {
      int64_t offset = 0;
      for (size_t i = 0; i < num_indices_; i++) {
        // handle int32 index tensor according to the indice_size_bytes.
        // we didn't use template parametor to avoid too many kernels' creation
        // with numbers of input datatypes.
        if (indice_size_bytes_ == 4) {
          int32_t index =
              *(int32_t*)(index_ptrs_[i] + local_index * indice_size_bytes_);
          SYCL_KERNEL_ASSERT(
              index >= -sizes_[i] && index < sizes_[i] &&
              "index out of bounds");
          if (index < 0) {
            index += sizes_[i];
          }
          offset += index * strides_[i];
        } else {
          int64_t index =
              *(int64_t*)(index_ptrs_[i] + local_index * indice_size_bytes_);
          SYCL_KERNEL_ASSERT(
              index >= -sizes_[i] && index < sizes_[i] &&
              "index out of bounds");
          if (index < 0) {
            index += sizes_[i];
          }
          offset += index * strides_[i];
        }
      }
      local_offset_[local_index] = offset;
    }

    // calculate the number of workloads on each group
    auto group_linear_id = group_id * group_numel_;
    auto group_numel_range = group_numel_;
    if (group_num_tail_ && group_id >= group_num_) {
      group_linear_id = group_num_ * group_numel_ +
          (group_id - group_num_) * group_numel_tail_;
      group_numel_range = group_numel_tail_;
    }
    auto out_ptr = out_data_;
    auto in_ptr = in_data_;
    sycl::group_barrier(item_id.get_group());

    // compute the in/out/indices offsets and perform memory copy
    for (int64_t local_index = local_id; local_index < group_numel_range;
         local_index += wgroup_size_) {
      auto linear_id = group_linear_id + local_index;
      auto out_offset = linear_id * element_size_bytes_;
      auto src_linear_id = linear_id / indices_size_;
      int64_t in_offset = 0;
      for (int i = num_non_indices_ - 1; i > 0; --i) {
        in_offset += (src_linear_id % src_sizes_[i]) * src_strides_[i];
        src_linear_id /= src_sizes_[i];
      }
      in_offset += src_linear_id * src_strides0_;

      auto offset = local_offset_[local_index % indices_size_];
      f_(out_ptr + out_offset, in_ptr + in_offset, offset);
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    local_offset_ = sycl_local_acc_t<int64_t, 1>(indices_size_, cgh);
  }

  SmallIndexKernelFunctor() = default;

  SmallIndexKernelFunctor(
      const func_t f,
      int64_t indices_size,
      int64_t group_num_tail,
      int64_t group_num,
      int64_t group_numel,
      int64_t group_numel_tail,
      int64_t wgroup_size,
      size_t num_non_indices,
      at::detail::Array<int64_t, XPU_MAX_TENSORINFO_DIMS> src_sizes,
      at::detail::Array<int64_t, XPU_MAX_TENSORINFO_DIMS> src_strides,
      int64_t src_strides0,
      size_t num_indices,
      at::detail::Array<int64_t, XPU_MAX_TENSORINFO_DIMS> sizes,
      at::detail::Array<int64_t, XPU_MAX_TENSORINFO_DIMS> strides,
      int64_t element_size_bytes,
      int64_t indice_size_bytes,
      char* out_data,
      char* in_data,
      at::detail::Array<index_buf_type, XPU_MAX_TENSORINFO_DIMS> index_ptrs)
      : f_(f),
        indices_size_(indices_size),
        group_num_tail_(group_num_tail),
        group_num_(group_num),
        group_numel_(group_numel),
        group_numel_tail_(group_numel_tail),
        wgroup_size_(wgroup_size),
        num_non_indices_(num_non_indices),
        src_sizes_(src_sizes),
        src_strides_(src_strides),
        src_strides0_(src_strides0),
        num_indices_(num_indices),
        sizes_(sizes),
        strides_(strides),
        element_size_bytes_(element_size_bytes),
        indice_size_bytes_(indice_size_bytes),
        out_data_(out_data),
        in_data_(in_data),
        index_ptrs_(index_ptrs),
        local_offset_() {}

 private:
  const func_t f_;
  int64_t indices_size_;
  int64_t group_num_tail_;
  int64_t group_num_;
  int64_t group_numel_;
  int64_t group_numel_tail_;
  int64_t wgroup_size_;
  size_t num_non_indices_;
  at::detail::Array<int64_t, XPU_MAX_TENSORINFO_DIMS> src_sizes_;
  at::detail::Array<int64_t, XPU_MAX_TENSORINFO_DIMS> src_strides_;
  int64_t src_strides0_;
  size_t num_indices_;
  at::detail::Array<int64_t, XPU_MAX_TENSORINFO_DIMS> sizes_;
  at::detail::Array<int64_t, XPU_MAX_TENSORINFO_DIMS> strides_;
  int64_t element_size_bytes_;
  int64_t indice_size_bytes_;
  char* out_data_;
  char* in_data_;
  at::detail::Array<index_buf_type, XPU_MAX_TENSORINFO_DIMS> index_ptrs_;
  sycl_local_acc_t<int64_t, 1> local_offset_;
};

// SYCL suggests: it’s possible (and even desirable) to oversubscribe tasks to
// device;
constexpr int OVER_SUBSCRIBE_DSS_FACTOR = 16;

template <typename func_t>
void small_index_kernel(
    TensorIteratorBase& iter,
    IntArrayRef index_size,
    IntArrayRef index_stride,
    IntArrayRef non_index_size,
    IntArrayRef non_index_stride,
    const func_t f) {
  using index_buf_type = char*;
  using KernelClass = SmallIndexKernelFunctor<func_t, index_buf_type>;

  auto numel = iter.numel();
  auto indices_size = iter.tensor(2).size(-1);
  auto& queue = getCurrentSYCLQueue();
  int64_t max_group_num = syclMaxDSSNum() * OVER_SUBSCRIBE_DSS_FACTOR;

  auto total_index_iter = numel / indices_size;
  max_group_num = std::min(int64_t(total_index_iter / 2), max_group_num);

  // process the tail
  auto group_index_iter =
      (total_index_iter + max_group_num - 1) / max_group_num;
  auto group_num_tail = group_index_iter * max_group_num - total_index_iter;
  auto group_num = max_group_num - group_num_tail;
  auto group_numel = group_index_iter * indices_size;
  auto group_numel_tail = (group_index_iter - 1) * indices_size;

  auto wgroup_size = syclMaxWorkGroupSize<KernelClass>();
  wgroup_size = std::min(decltype(wgroup_size)(group_numel), wgroup_size);
  auto global_size = max_group_num * wgroup_size;

  size_t num_non_indices = non_index_size.size();
  at::detail::Array<int64_t, XPU_MAX_TENSORINFO_DIMS> src_sizes(0);
  at::detail::Array<int64_t, XPU_MAX_TENSORINFO_DIMS> src_strides(0);
  for (size_t i = 0; i < num_non_indices; ++i) {
    src_sizes[i] = non_index_size[i];
    src_strides[i] = non_index_stride[i];
  }
  auto src_strides0 = non_index_stride[0];

  size_t num_indices = index_size.size();
  at::detail::Array<int64_t, XPU_MAX_TENSORINFO_DIMS> sizes(0);
  at::detail::Array<int64_t, XPU_MAX_TENSORINFO_DIMS> strides(0);
  for (size_t i = 0; i < num_indices; i++) {
    sizes[i] = index_size[i];
    strides[i] = index_stride[i];
  }

  int64_t element_size_bytes = iter.tensor(1).element_size();
  int64_t indice_size_bytes = iter.tensor(2).element_size();

  auto out_data = (char*)iter.data_ptr(0);
  auto in_data = (char*)iter.data_ptr(1);
  at::detail::Array<index_buf_type, XPU_MAX_TENSORINFO_DIMS> index_ptrs;
  for (size_t i = 0; i < num_indices; i++) {
    index_ptrs[i] = (char*)iter.data_ptr(i + 2);
  }

  KernelClass kfn(
      f,
      indices_size,
      group_num_tail,
      group_num,
      group_numel,
      group_numel_tail,
      wgroup_size,
      num_non_indices,
      src_sizes,
      src_strides,
      src_strides0,
      num_indices,
      sizes,
      strides,
      element_size_bytes,
      indice_size_bytes,
      out_data,
      in_data,
      index_ptrs);
  sycl_kernel_submit(
      sycl::range<1>(global_size), sycl::range<1>(wgroup_size), queue, kfn);
}

template <
    typename func_t,
    typename index_buf_type,
    typename OffsetCalculatorType>
struct IndexKernelFunctor {
  void operator()(sycl::item<1> item_id) const {
    auto linear_idx = item_id.get_linear_id();
    auto offsets = offset_calc_.get(linear_idx);
    auto out_ptr = out_data_ + offsets[0];
    auto in_ptr = in_data_ + offsets[1];
    int64_t offset = 0;
    // #pragma unroll
    for (size_t i = 0; i < num_indices_; i++) {
      // handle int32 index tensor according to the indice_size_bytes.
      // we didn't use template parametor to avoid too many kernels' creation
      // with numbers of input datatypes.
      if (indice_size_bytes_ == 4) {
        int32_t index = *(int32_t*)(index_ptrs_[i] + offsets[2]);
        SYCL_KERNEL_ASSERT(
            index >= -sizes_[i] && index < sizes_[i] && "index out of bounds");
        if (index < 0) {
          index += sizes_[i];
        }
        offset += index * strides_[i];
      } else {
        int64_t index = *(int64_t*)(index_ptrs_[i] + offsets[2]);
        SYCL_KERNEL_ASSERT(
            index >= -sizes_[i] && index < sizes_[i] && "index out of bounds");
        if (index < 0) {
          index += sizes_[i];
        }
        offset += index * strides_[i];
      }
    }
    f_(out_ptr, in_ptr, offset);
  }
  IndexKernelFunctor(
      const func_t f,
      OffsetCalculatorType offset_calc,
      int64_t indice_size_bytes,
      char* out_data,
      char* in_data,
      size_t num_indices,
      at::detail::Array<index_buf_type, XPU_MAX_TENSORINFO_DIMS> index_ptrs,
      at::detail::Array<int64_t, XPU_MAX_TENSORINFO_DIMS> sizes,
      at::detail::Array<int64_t, XPU_MAX_TENSORINFO_DIMS> strides)
      : f_(f),
        offset_calc_(offset_calc),
        indice_size_bytes_(indice_size_bytes),
        out_data_(out_data),
        in_data_(in_data),
        num_indices_(num_indices),
        index_ptrs_(index_ptrs),
        sizes_(sizes),
        strides_(strides) {}

 private:
  const func_t f_;
  OffsetCalculatorType offset_calc_;
  int64_t indice_size_bytes_;
  char* out_data_;
  char* in_data_;
  size_t num_indices_;
  at::detail::Array<index_buf_type, XPU_MAX_TENSORINFO_DIMS> index_ptrs_;
  at::detail::Array<int64_t, XPU_MAX_TENSORINFO_DIMS> sizes_;
  at::detail::Array<int64_t, XPU_MAX_TENSORINFO_DIMS> strides_;
};

template <typename func_t>
void index_kernel_impl(
    TensorIteratorBase& iter,
    IntArrayRef index_size,
    IntArrayRef index_stride,
    const func_t f) {
  size_t num_indices = index_size.size();
  auto numel = iter.numel();
  at::detail::Array<int64_t, XPU_MAX_TENSORINFO_DIMS> sizes(0);
  at::detail::Array<int64_t, XPU_MAX_TENSORINFO_DIMS> strides(0);
  for (size_t i = 0; i < num_indices; i++) {
    sizes[i] = index_size[i];
    strides[i] = index_stride[i];
  }

  int64_t indice_size_bytes = iter.tensor(2).element_size();

  auto& queue = getCurrentSYCLQueue();
  auto out_data = (char*)iter.data_ptr(0);
  auto in_data = (char*)iter.data_ptr(1);
  using index_buf_type = decltype((char*)iter.data_ptr(0));
  at::detail::Array<index_buf_type, XPU_MAX_TENSORINFO_DIMS> index_ptrs;
  for (size_t i = 0; i < num_indices; i++) {
    index_ptrs[i] = (char*)iter.data_ptr(i + 2);
  }

  auto offset_calc = make_offset_calculator<3>(iter);
  IndexKernelFunctor<func_t, index_buf_type, decltype(offset_calc)> kfn(
      f,
      offset_calc,
      indice_size_bytes,
      out_data,
      in_data,
      num_indices,
      index_ptrs,
      sizes,
      strides);
  sycl_kernel_submit(sycl::range<1>(numel), queue, kfn);
}

template <typename func_t>
void _index_kernel(
    TensorIteratorBase& iter,
    IntArrayRef index_size,
    IntArrayRef index_stride,
    IntArrayRef non_index_size,
    IntArrayRef non_index_stride,
    const func_t f,
    const bool is_gather_like) {
  auto numel = iter.numel();

  if (numel == 0) {
    return;
  }

  size_t num_indices = index_size.size();
  TORCH_INTERNAL_ASSERT(num_indices == index_stride.size());
  TORCH_INTERNAL_ASSERT(
      num_indices == static_cast<size_t>(iter.ntensors()) - 2);
  TORCH_INTERNAL_ASSERT(num_indices <= XPU_MAX_TENSORINFO_DIMS);

  // the small_index_kernel is applied for last several
  // successive dims indexing of an input tensor Taking 3-dims tensor input
  // (input.shape=[x,y,z]) for example: input[:,:,idx] or input[:,idx1,idx2]
  // when input tensor satisfies the following conditions, the
  // small_index_kernel path will be selected: 1.there are common indices
  // such as input[:,:,idx] and input[:,idx1,idx2] instead of
  //   input[idx0,idx1,idx2], input[idx0,idx1,:], input[idx0,:,idx2],
  //   input[idx0,:,:], input[:,idx1,:]
  // 2.the common indices numel should larger than 2 times of the
  // syclMaxComputeUnitSize (then we can get memory access benifit) 3.the
  // workloads in each group should larger than the maximum number of
  // workitem (ensure all the workitem activate) 4.the indices_table size
  // should satisfied the SLM limit condition

  // check whether the current case satisfying the condition 1
  // for 3-dims input:
  // Taking input[idx0,:,idx2] for example, the indices_sizes=[sz,1,sz]
  // While the satified case is input[:,idx1,idx2], indices_sizes=[1,sz,sz]
  bool small_index = non_index_size.size() != 0 && iter.tensor(1).dim() == 3 &&
      non_index_size.size() + index_size.size() == 3;
  auto indices_sizes = iter.tensor(2).sizes();
  for (size_t i = 1; i < iter.tensor(2).dim(); ++i) {
    if (indices_sizes[i - 1] > indices_sizes[i]) {
      small_index = false;
      break;
    }
  }
  if (small_index) {
    using index_buf_type = char*;
    using KernelClass = SmallIndexKernelFunctor<func_t, index_buf_type>;

    int64_t max_group_num = syclMaxDSSNum();
    auto wgroup_size = syclMaxWorkGroupSize<KernelClass>();
    auto indices_size = iter.tensor(2).size(-1);
    auto total_index_iter = numel / indices_size;
    auto local_index = numel / max_group_num;

    // the max_local_mem_size = 65536B (64KB)
    // TODO: Is this right?
    auto max_local_mem_size = syclLocalMemSize();
    auto indice_table_size = indices_size * sizeof(int64_t);

    // check whether the current case satisfying conditions 2,3,4
    small_index =
        (total_index_iter > 2 * max_group_num && local_index > wgroup_size &&
         indice_table_size < max_local_mem_size * 0.5);
    if (small_index) {
      small_index_kernel<func_t>(
          iter, index_size, index_stride, non_index_size, non_index_stride, f);
      return;
    }
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      _index_kernel(
          sub_iter,
          index_size,
          index_stride,
          IntArrayRef{},
          IntArrayRef{},
          f,
          is_gather_like);
    }
    return;
  }
  char* const out_ptr = static_cast<char*>(iter.data_ptr(0));
  char* const in_ptr = static_cast<char*>(iter.data_ptr(1));

  if (is_gather_like && num_indices == 1) {
    const size_t element_size = iter.element_size(0);
    constexpr size_t alignment = 16;
    if (at::native::xpu::fast_gather_kernel_eligible<alignment>(
            iter, out_ptr, in_ptr, index_stride[0], element_size)) {
      auto slice_size = iter.shape()[0] * element_size;
      auto num_ind = iter.shape()[1];
      auto ind_dim_size = index_size[0];
      auto inp_stride_bytes = index_stride[0];
      auto out_stride_bytes = iter.strides(0)[1];
      if (iter.numel() == 0)
        return;
      at::native::xpu::vectorized_gather_kernel_launch<alignment, int64_t>(
          out_ptr,
          in_ptr,
          (int64_t*)iter.data_ptr(2),
          num_ind,
          slice_size,
          ind_dim_size,
          inp_stride_bytes,
          out_stride_bytes,
          /*allow_neg_indices*/ true);
      return;
    }
  }
  index_kernel_impl<func_t>(iter, index_size, index_stride, f);
}

template <typename scalar_t, typename accscalar_t>
struct IndexPutDeterministicKernelFunctor {
  void operator()(sycl::nd_item<2> item) const {
    auto id = cfg_.get_item_desc(item);

    if (id.glb_batch >= cfg_.problem_batch_ || id.glb_problem >= cfg_.problem_)
      return;

    int64_t idx = sorted_indices_[id.glb_batch];
    if (id.glb_batch != 0 && idx == sorted_indices_[id.glb_batch - 1])
      return;

    int64_t pi_ = id.glb_problem;
    int64_t si_ = pi_ % stride_;
    int64_t bi_ = pi_ / stride_;
    int64_t s_gid = si_ + idx * stride_ + bi_ * stride_before_;
    int64_t v_stride = si_ + bi_ * v_stride_before_;

    accscalar_t acc;
    if (accumulate_)
      acc = c10::load(&self_[s_gid]);
    for (int64_t inner_idx = id.glb_batch;
         inner_idx < cfg_.problem_batch_ && sorted_indices_[inner_idx] == idx;
         inner_idx++) {
      int64_t idx_orig = indices_[inner_idx];
      int64_t v_gid = idx_orig * stride_ + v_stride;
      if (accumulate_) {
        acc += (accscalar_t)c10::load(&value_[v_gid]);
      } else {
        self_[s_gid] = c10::load(&value_[v_gid]);
        break;
      }
    }
    if (accumulate_)
      self_[s_gid] = acc;
  }

  IndexPutDeterministicKernelFunctor(
      int64_t* sorted_indices,
      int64_t* indices,
      const scalar_t* value,
      scalar_t* self,
      int64_t stride,
      int64_t stride_before,
      bool accumulate,
      int64_t v_stride_before,
      BatchKernelConfig cfg)
      : sorted_indices_(sorted_indices),
        indices_(indices),
        value_(value),
        self_(self),
        stride_(stride),
        stride_before_(stride_before),
        accumulate_(accumulate),
        v_stride_before_(v_stride_before),
        cfg_(cfg) {}

 private:
  int64_t* sorted_indices_;
  int64_t* indices_;
  const scalar_t* value_;
  scalar_t* self_;
  int64_t stride_;
  int64_t stride_before_;
  bool accumulate_;
  int64_t v_stride_before_;
  BatchKernelConfig cfg_;
};

template <typename scalar_t, typename accscalar_t>
void launch_index_put_deterministic_kernel(
    int64_t* sorted_indices,
    int64_t* indices,
    const scalar_t* value,
    scalar_t* self,
    int64_t numel,
    int64_t stride,
    int64_t stride_before,
    int64_t outer_dim,
    bool accumulate) {
  if (outer_dim * stride == 0 || numel == 0) {
    return;
  }
  int64_t v_stride_before = numel * stride;
  using KernelClass = IndexPutDeterministicKernelFunctor<scalar_t, accscalar_t>;
  BatchKernelConfig cfg = BatchKernelConfig::make_config<KernelClass>(
      /* num of indices */ numel,
      /* num of elements to put per indices */ outer_dim * stride,
      1,
      numel,
      true,
      {BatchKernelConfig::Policy::pSegment,
       BatchKernelConfig::Policy::pAggressiveSplit});
  KernelClass kfn(
      sorted_indices,
      indices,
      value,
      self,
      stride,
      stride_before,
      accumulate,
      v_stride_before,
      cfg);

  sycl_kernel_submit(
      cfg.global_size(), cfg.group_size(), getCurrentSYCLQueue(), kfn);
}

template <int vt, typename func_t>
struct IndexElementwiseKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    int wg_sz = item.get_local_range(0);
    auto tid = item.get_local_id(0);
    auto nv = wg_sz * vt;
    auto idx = nv * item.get_group(0) + tid;
#pragma unroll
    for (int i = 0; i < vt; i++) {
      if (idx < N_) {
        f_(idx);
        idx += wg_sz;
      }
    }
  }

  IndexElementwiseKernelFunctor(const int64_t N, const func_t f)
      : N_(N), f_(f) {}

 private:
  const int64_t N_;
  const func_t f_;
};

template <int vt, typename func_t>
static void launch_index_group_stride_kernel(const int64_t N, const func_t& f) {
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  int wg_sz = syclMaxWorkItemsPerSubSlice();
  int num_wg = (N + wg_sz * vt - 1) / (wg_sz * vt);
  auto ker = IndexElementwiseKernelFunctor<vt, func_t>(N, f);
  sycl_kernel_submit(wg_sz * num_wg, wg_sz, getCurrentSYCLQueue(), ker);
}

#define TAKE_PUT_UNROLL_SZIE 4

template <int vt, typename func_t>
struct TakePutKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    const auto tid = item.get_local_id(0);
    const auto nt = item.get_local_range(0);
    const auto nv = nt * vt;
    auto idx = nv * item.get_group(0) + tid;
#pragma unroll
    for (int i = 0; i < vt; i++) {
      if (idx < N_) {
        f_(idx);
        idx += nt;
      }
    }
  }

  TakePutKernelFunctor(const int64_t N, const func_t f) : N_(N), f_(f) {}

 private:
  const int64_t N_;
  const func_t f_;
};


#define SKIP_SORTED_INDICES 32
#define SUBGROUP_SIZE 32

template <typename scalar_t, int SZ>
struct indexing_backward_kernel : public __SYCL_KER_CONFIG_CONVENTION__ {
  void operator()(sycl::nd_item<3> item) const {
    using opmath_t = at::opmath_type<scalar_t>;
    auto sg = item.get_sub_group();

    int smem_offset = item.get_local_id(1) * SUBGROUP_SIZE;
    int laneIdx = item.get_local_id(2) % SUBGROUP_SIZE;
    int64_t grad_row = 0;

    for (int64_t z = item.get_group(0); z < outer_dim;
         z += item.get_group_range(0)) {
      // Init duplicates every time we compute a new set of entries:
      smem_dups_cache[smem_offset + laneIdx] = 0;
      sycl::group_barrier(sg);

      int64_t base_idx =
          item.get_group(2) * item.get_local_range(1) * SUBGROUP_SIZE +
          item.get_local_id(1) * SUBGROUP_SIZE;
      int64_t idx = base_idx + laneIdx;

      if (idx < numel) {
        int64_t crnt_sorted_idx = sorted_indices[idx];

        if (idx == 0 || crnt_sorted_idx != sorted_indices[idx - 1]) {
          // Determine the number of duplicates in advance:
          int64_t num_duplicates = 1;

          // Lookahead in case there is a large number of duplicates. Once that
          // is done, handle the tail.
          while ((idx + num_duplicates + SKIP_SORTED_INDICES - 1) < numel) {
            if (sorted_indices
                    [idx + num_duplicates + SKIP_SORTED_INDICES - 1] !=
                crnt_sorted_idx)
              break;
            num_duplicates += SKIP_SORTED_INDICES;
          }
          while (((idx + num_duplicates) < numel) &&
                 (sorted_indices[idx + num_duplicates] == crnt_sorted_idx)) {
            num_duplicates++;
          }

          smem_dups_cache[smem_offset + laneIdx] = num_duplicates;
        }
      }

      sycl::group_barrier(sg);

      // All lanes in the warp are still active here. Use them all to reduce
      // duplicates when large number of duplicates are present:
      for (int subwarp = 0; subwarp < SUBGROUP_SIZE; subwarp++) {
        // All lanes read the shared memory entry for number of duplicates
        int64_t new_num_duplicates = smem_dups_cache[smem_offset + subwarp];

        // Check if the original sub-warp had duplicates to eliminate, if not
        // skip.
        if (new_num_duplicates == 0)
          continue;

        // There are duplicates that need eliminating:
        int64_t new_idx = base_idx + subwarp;
        int64_t new_crnt_sorted_idx = sorted_indices[new_idx];
        const int64_t new_weight_row =
            new_crnt_sorted_idx * stride + z * stride_before;

        if (!accumulate) {
          const int64_t grad_row =
              ((int64_t)indices[new_idx + new_num_duplicates - 1]) * stride +
              z * numel * stride;
          int64_t feature_dim = item.get_group(1) * item.get_local_range(2) +
              item.get_local_id(2);
          while (feature_dim < stride) {
            grad_weight[new_weight_row + feature_dim] =
                grad_output[grad_row + feature_dim];
            feature_dim += item.get_group_range(1) * item.get_local_range(2);
          }
          continue;
        }

        for (int dup = 0; dup < new_num_duplicates; dup++) {
          const int64_t grad_row =
              ((int64_t)indices[new_idx + dup]) * stride + z * numel * stride;

          // All lanes do the same thing up to here.
          int64_t feature_dim = item.get_group(1) * item.get_local_range(2) +
              item.get_local_id(2);

          // Each lane has a different feature_dim.
          while (feature_dim < stride) {
            grad_weight[new_weight_row + feature_dim] +=
                grad_output[grad_row + feature_dim];
            feature_dim += item.get_group_range(1) * item.get_local_range(2);
          }
        }
      }
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    smem_dups_cache = sycl_local_acc_t<int64_t>(smem_size, cgh);
  }

  indexing_backward_kernel(
      const int64_t* sorted_indices,
      const int64_t* indices,
      const scalar_t* grad_output,
      scalar_t* grad_weight,
      int64_t numel,
      int64_t stride,
      int64_t stride_before,
      int64_t outer_dim,
      bool accumulate,
      int64_t smem_size)
      : sorted_indices(sorted_indices),
        indices(indices),
        grad_output(grad_output),
        grad_weight(grad_weight),
        numel(numel),
        stride(stride),
        stride_before(stride_before),
        outer_dim(outer_dim),
        accumulate(accumulate),
        smem_size(smem_size) {}

 private:
  const int64_t* sorted_indices;
  const int64_t* indices;
  const scalar_t* grad_output;
  scalar_t* grad_weight;
  int64_t numel;
  int64_t stride;
  int64_t stride_before;
  int64_t outer_dim;
  bool accumulate;
  int64_t smem_size;
  sycl_local_acc_t<int64_t> smem_dups_cache;
};

template <typename scalar_t>
struct indexing_backward_kernel_stride_1
    : public __SYCL_KER_CONFIG_CONVENTION__ {
  void operator()(sycl::nd_item<3> item) const {
    using opmath_t = at::opmath_type<scalar_t>;
    auto sg = item.get_sub_group();

    int laneIdx = item.get_local_id(2) % SUBGROUP_SIZE;

    const opmath_t scale = (opmath_t)1.0;
    int64_t grad_row = 0;

    // Each warp gets a different section of the share memory allocation:
    int smem_offset = item.get_local_id(1) * SUBGROUP_SIZE;

    // Number of values processed by each thread (grain size)
    for (int64_t z = item.get_group(0); z < outer_dim;
         z += item.get_group_range(0)) {
      // Init duplicates every time we compute a new set of entries:
      smem_dups_cache[smem_offset + laneIdx] = 0;

      int64_t base_idx =
          item.get_group(2) * item.get_local_range(1) * SUBGROUP_SIZE +
          item.get_local_id(1) * SUBGROUP_SIZE;
      int64_t idx = base_idx + laneIdx;

      // Each lane calculates the number of duplicates:
      if (idx < numel) {
        int64_t crnt_sorted_idx = sorted_indices[idx];

        if (idx == 0 || crnt_sorted_idx != sorted_indices[idx - 1]) {
          // Determine the number of duplicates in advance:
          int64_t num_duplicates = 1;

          // Lookahead in case there is a large number of duplicates. Once that
          // is done, handle the tail.
          while ((idx + num_duplicates + SKIP_SORTED_INDICES - 1) < numel) {
            if (sorted_indices
                    [idx + num_duplicates + SKIP_SORTED_INDICES - 1] !=
                crnt_sorted_idx)
              break;
            num_duplicates += SKIP_SORTED_INDICES;
          }
          while (((idx + num_duplicates) < numel) &&
                 (sorted_indices[idx + num_duplicates] == crnt_sorted_idx)) {
            num_duplicates++;
          }

          if (!accumulate) {
            const int64_t weight_row =
                crnt_sorted_idx * stride + z * stride_before;
            grad_row = ((int64_t)indices[idx + num_duplicates - 1]) * stride +
                z * numel * stride;
            grad_weight[weight_row] = static_cast<scalar_t>(
                static_cast<opmath_t>(grad_output[grad_row]) * scale);
            continue;
          }

          // Each lane sequentially handles the duplicate elimination:
          if (num_duplicates < SUBGROUP_SIZE) {
            opmath_t gradient = (opmath_t)0.0;
            const int64_t weight_row =
                crnt_sorted_idx * stride + z * stride_before;
            for (int64_t i = 0; i < num_duplicates; ++i) {
              grad_row =
                  ((int64_t)indices[idx + i]) * stride + z * numel * stride;
              gradient += static_cast<opmath_t>(grad_output[grad_row]) * scale;
            }

            grad_weight[weight_row] = static_cast<scalar_t>(
                static_cast<opmath_t>(grad_weight[weight_row]) + gradient);
          } else {
            // Add duplicate to the cache:
            smem_dups_cache[smem_offset + laneIdx] = num_duplicates;
          }
        }
      }

      sycl::group_barrier(sg);

      // All lanes in the warp are still active here. Use them all to reduce
      // duplicates when large number of duplicates are present:
      for (int subwarp = 0; subwarp < SUBGROUP_SIZE; subwarp++) {
        // All lanes read the shared memory entry for number of duplicates
        int64_t new_num_duplicates = smem_dups_cache[smem_offset + subwarp];

        // Check if the original sub-warp had duplicates to eliminate, if not
        // skip.
        if (new_num_duplicates == 0)
          continue;

        // There are duplicates that need eliminating:
        int64_t new_idx = base_idx + subwarp;
        int64_t new_crnt_sorted_idx = sorted_indices[new_idx];
        const int64_t new_weight_row =
            new_crnt_sorted_idx * stride + z * stride_before;

        // Result of the reduction will be in this variable:
        opmath_t gradient = (opmath_t)0.0;

        int64_t num_warp_passes = new_num_duplicates / SUBGROUP_SIZE;
        // Parallel reduction across the array of duplicates using all the lanes
        // in the warp:
        for (int64_t i = 0; i < num_warp_passes; ++i) {
          grad_row = ((int64_t)indices[new_idx + i * SUBGROUP_SIZE + laneIdx]) *
                  stride +
              z * numel * stride;
          gradient += static_cast<opmath_t>(grad_output[grad_row]) * scale;
        }

        // Reduce across the lanes of the warp:
        sycl::group_barrier(sg);

        for (int offset = SUBGROUP_SIZE / 2; offset > 0; offset /= 2) {
          gradient += sycl::shift_group_left(sg, gradient, offset);
        }

        if (laneIdx == 0) {
          for (int64_t i = num_warp_passes * SUBGROUP_SIZE;
               i < new_num_duplicates;
               ++i) {
            grad_row =
                ((int64_t)indices[new_idx + i]) * stride + z * numel * stride;
            gradient += static_cast<opmath_t>(grad_output[grad_row]) * scale;
          }

          grad_weight[new_weight_row] = static_cast<scalar_t>(
              static_cast<opmath_t>(grad_weight[new_weight_row]) + gradient);
        }
      }
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    smem_dups_cache = sycl_local_acc_t<int64_t>(smem_size, cgh);
  }

  indexing_backward_kernel_stride_1(
      const int64_t* sorted_indices,
      const int64_t* indices,
      const scalar_t* grad_output,
      scalar_t* grad_weight,
      int64_t numel,
      int64_t stride,
      int64_t stride_before,
      int64_t outer_dim,
      bool accumulate,
      int64_t smem_size)
      : sorted_indices(sorted_indices),
        indices(indices),
        grad_output(grad_output),
        grad_weight(grad_weight),
        numel(numel),
        stride(stride),
        stride_before(stride_before),
        outer_dim(outer_dim),
        accumulate(accumulate),
        smem_size(smem_size) {}

 private:
  const int64_t* sorted_indices;
  const int64_t* indices;
  const scalar_t* grad_output;
  scalar_t* grad_weight;
  int64_t numel;
  int64_t stride;
  int64_t stride_before;
  int64_t outer_dim;
  bool accumulate;
  int64_t smem_size;
  sycl_local_acc_t<int64_t> smem_dups_cache;
};

template <typename scalar_t>
struct indexing_backward_kernel_small_stride {
  void operator()(sycl::nd_item<3> item) const {
    using opmath_t = at::opmath_type<scalar_t>;

    // Number of values processed by each thread (grain size)
    for (int64_t z = item.get_group(0); z < outer_dim;
         z += item.get_group_range(0)) {
      int64_t idx =
          item.get_group(2) * item.get_local_range(1) + item.get_local_id(1);
      int64_t tidx = item.get_local_id(2);
      int64_t crnt_sorted_idx = sorted_indices[idx];

      if ((idx < numel) && (tidx < stride) &&
          (idx == 0 || crnt_sorted_idx != sorted_indices[idx - 1])) {
        // Determine the number of duplicates in advance
        int64_t num_duplicates = 1;
        while (((idx + num_duplicates) < numel) &&
               (sorted_indices[idx + num_duplicates] == crnt_sorted_idx)) {
          num_duplicates++;
        }

        // Continue computing weights
        const int64_t weight_row = crnt_sorted_idx * stride + z * stride_before;
        int64_t grad_row = 0;
        const opmath_t scale = (opmath_t)1.0;

        if (!accumulate) {
          grad_row = ((int64_t)indices[idx + num_duplicates - 1]) * stride +
              z * numel * stride;
          grad_weight[weight_row + tidx] = static_cast<scalar_t>(
              static_cast<opmath_t>(grad_output[grad_row + tidx]) * scale);
        } else {
          opmath_t gradient = (opmath_t)0.0;
          for (int64_t i = 0; i < num_duplicates; ++i) {
            grad_row =
                ((int64_t)indices[idx + i]) * stride + z * numel * stride;
            gradient +=
                static_cast<opmath_t>(grad_output[grad_row + tidx]) * scale;
          }

          grad_weight[weight_row + tidx] = static_cast<scalar_t>(
              static_cast<opmath_t>(grad_weight[weight_row + tidx]) + gradient);
        }
      }
    }
  }

  indexing_backward_kernel_small_stride(
      const int64_t* sorted_indices,
      const int64_t* indices,
      const scalar_t* grad_output,
      scalar_t* grad_weight,
      int64_t numel,
      int64_t stride,
      int64_t stride_before,
      int64_t outer_dim,
      bool accumulate)
      : sorted_indices(sorted_indices),
        indices(indices),
        grad_output(grad_output),
        grad_weight(grad_weight),
        numel(numel),
        stride(stride),
        stride_before(stride_before),
        outer_dim(outer_dim),
        accumulate(accumulate) {}

 private:
  const int64_t* sorted_indices;
  const int64_t* indices;
  const scalar_t* grad_output;
  scalar_t* grad_weight;
  int64_t numel;
  int64_t stride;
  int64_t stride_before;
  int64_t outer_dim;
  bool accumulate;
};

#undef SKIP_SORTED_INDICES
#undef SUBGROUP_SIZE

} // namespace at::native::xpu
