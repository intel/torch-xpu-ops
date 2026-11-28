/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#pragma once

#include <algorithm>
#include <bit>
#include <functional>
#include <type_traits>

#include <ATen/OpMathType.h>
#include <ATen/ceil_div.h>
#include <ATen/native/Math.h>
#include <ATen/native/Resize.h>
#include <ATen/native/xpu/sycl/KernelUtils.h>
#include <ATen/xpu/XPUContext.h>
#include <comm/SYCLContext.h>
#include <comm/SYCLHelpers.h>
#include <comm/TensorInfo.h>
#include <comm/TensorOptions.h>

namespace at::native::xpu {
using namespace at::xpu::detail;
using namespace at::xpu;

enum ScanType {
  EXCLUSIVE_TYPE = 0,
  INCLUSIVE_TYPE = 1,
};

template <typename scalar_t, typename idx_t, typename BinaryOperation>
void binary_op_update(
    const scalar_t lhs,
    scalar_t& rhs,
    const idx_t lhs_idx,
    idx_t& rhs_idx,
    BinaryOperation binary_op) {
  if (!at::_isnan(rhs) && (at::_isnan(lhs) || !binary_op(rhs, lhs))) {
    rhs = lhs;
    rhs_idx = lhs_idx;
  }
}

// group x scan by using up down sweep algorithm(call uds for short)
template <
    class LSConfig,
    class T,
    class IndicesT,
    class BinaryFunction,
    bool TrivialOffCal = false>
void inline group_x_scan_by_uds_for_loop_scan_with_indices(
    sycl::nd_item<2> item,
    T& pre_max_carr,
    IndicesT& pre_idx_carr,
    int64_t base_off_batch,
    int64_t base_off_problem,
    sycl::local_ptr<T> slm,
    sycl::local_ptr<IndicesT> slm_idx,
    LSConfig cfg) {
  using InputInfo = typename LSConfig::InputInfoType;
  using OutputInfo = typename LSConfig::OutputInfoType;
  using IndicesInfo = typename LSConfig::IndicesInfoType;

  int64_t glb_ldr_off_0, glb_ldr_off_1, glb_str_off_0, glb_str_off_1,
      glb_ldr_logical_off_0, glb_ldr_logical_off_1, glb_str_logical_off_0,
      glb_str_logical_off_1, glb_idx_off_0, glb_idx_off_1;

  const auto sub_group = item.get_sub_group();
  const auto sub_group_size = sub_group.get_local_range()[0];

  const auto lix = item.get_local_id(1);
  const auto liy = item.get_local_id(0);
  const auto rx = item.get_local_range(1);

  uint32_t ix0 = base_off_problem + lix;
  uint32_t ix1 = base_off_problem + rx + lix;
  uint32_t glb0 = base_off_batch * cfg.problem_ + ix0;
  uint32_t glb1 = base_off_batch * cfg.problem_ + ix1;
  if constexpr (TrivialOffCal) {
    glb_ldr_off_0 = glb0;
    glb_ldr_off_1 = glb1;
    glb_str_off_0 = glb0;
    glb_str_off_1 = glb1;
    glb_idx_off_0 = glb0;
    glb_idx_off_1 = glb1;
  } else {
    glb_ldr_logical_off_0 = glb0;
    glb_ldr_off_0 =
        IndexToOffset<typename InputInfo::scalar_t, int64_t, -1>::get(
            glb_ldr_logical_off_0, cfg.input_);

    glb_ldr_logical_off_1 = glb1;
    glb_ldr_off_1 =
        IndexToOffset<typename InputInfo::scalar_t, int64_t, -1>::get(
            glb_ldr_logical_off_1, cfg.input_);

    glb_str_logical_off_0 = glb0;
    glb_str_off_0 =
        IndexToOffset<typename OutputInfo::scalar_t, int64_t, -1>::get(
            glb_str_logical_off_0, cfg.output_);

    glb_str_logical_off_1 = glb1;
    glb_str_off_1 =
        IndexToOffset<typename OutputInfo::scalar_t, int64_t, -1>::get(
            glb_str_logical_off_1, cfg.output_);

    glb_idx_off_0 =
        IndexToOffset<typename IndicesInfo::scalar_t, int64_t, -1>::get(
            glb0, cfg.indices_);

    glb_idx_off_1 =
        IndexToOffset<typename IndicesInfo::scalar_t, int64_t, -1>::get(
            glb1, cfg.indices_);
  }
  // TODO: opti for bank conflict elemination
  // Read data from global memory to shared local memory
  // Each work item load 2 elements from global device memory to shared local
  // memory
  if (base_off_batch < cfg.batch_) {
    if (ix0 < cfg.problem_) {
      slm[liy * rx * 2 + lix] = c10::load(cfg.input_.data + glb_ldr_off_0);
      slm_idx[liy * rx * 2 + lix] = ix0;
    } else {
      slm[liy * rx * 2 + lix] = cfg.init_;
    }
    if (ix1 < cfg.problem_) {
      slm[liy * rx * 2 + rx + lix] = c10::load(cfg.input_.data + glb_ldr_off_1);
      slm_idx[liy * rx * 2 + rx + lix] = ix1;
    } else {
      slm[liy * rx * 2 + rx + lix] = cfg.init_;
    }

    // Add the total value of all previous work groups to the first value of
    // this work group.
    if (0 == lix) {
      auto offset = liy * rx * 2 + lix;
      binary_op_update(
          pre_max_carr, slm[offset], pre_idx_carr, slm_idx[offset], cfg.func_);
    }
  }
  sycl::group_barrier(item.get_group());

  // Parallel reduction (Up-sweep)
  for (uint32_t s = rx, d = 1; s >= 1; s >>= 1, d <<= 1) {
    if (base_off_batch < cfg.batch_ && lix < s) {
      uint32_t offset = liy * rx * 2 + (2 * lix + 1) * d - 1;
      binary_op_update(
          slm[offset],
          slm[offset + d],
          slm_idx[offset],
          slm_idx[offset + d],
          cfg.func_);
    }
    if (sub_group_size != cfg.wg_range_x_) {
      sycl::group_barrier(item.get_group());
    }
  }

  // Down-sweep
  for (uint32_t s = 2, d = rx / 2; d >= 1; s <<= 1, d >>= 1) {
    if (base_off_batch < cfg.batch_ && lix < s - 1) {
      uint32_t offset = liy * rx * 2 + 2 * (lix + 1) * d - 1;
      binary_op_update(
          slm[offset],
          slm[offset + d],
          slm_idx[offset],
          slm_idx[offset + d],
          cfg.func_);
    }
    if (sub_group_size != cfg.wg_range_x_) {
      sycl::group_barrier(item.get_group());
    }
  }

  // Write back from shared local memory to global memory
  if (base_off_batch < cfg.batch_) {
    if (ix0 < cfg.problem_) {
      cfg.output_.data[glb_str_off_0] = slm[liy * rx * 2 + lix];
      cfg.indices_.data[glb_idx_off_0] = slm_idx[liy * rx * 2 + lix];
    }
    if (ix1 < cfg.problem_) {
      cfg.output_.data[glb_str_off_1] = slm[liy * rx * 2 + rx + lix];
      cfg.indices_.data[glb_idx_off_1] = slm_idx[liy * rx * 2 + rx + lix];
    }
  }

  pre_max_carr = slm[liy * rx * 2 + 2 * rx - 1];
  pre_idx_carr = slm_idx[liy * rx * 2 + 2 * rx - 1];
}

template <
    class InputInfo,
    class OutputInfo,
    class IndicesInfo,
    typename T,
    class BinaryFunction>
class LoopScanConfig {
 public:
  using arg_t = T;
  using func_t = BinaryFunction;
  using InputInfoType = InputInfo;
  using OutputInfoType = OutputInfo;
  using IndicesInfoType = IndicesInfo;

  LoopScanConfig() = default;

  LoopScanConfig(
      InputInfo input_info,
      OutputInfo output_info,
      IndicesInfo indices_info,
      size_t batch,
      size_t problem,
      T init,
      ScanType type,
      BinaryFunction func)
      : input_(input_info),
        output_(output_info),
        indices_(indices_info),
        batch_(batch),
        problem_(problem),
        stride_(1),
        init_(init),
        type_(type),
        func_(func),
        glb_range_x_(0),
        glb_range_y_(0),
        wg_range_x_(std::min<size_t>(32, std::bit_ceil(problem))),
        wg_range_y_(0) {
    size_t wg_size = at::xpu::getDeviceMaxSubGroupSize() *
        at::xpu::getDeviceEUCountPerXeCore();
    wg_range_y_ = wg_size / wg_range_x_;
    const size_t target_global_size = at::xpu::getDeviceMaxWorkItems();
    const size_t max_work_group_num = target_global_size / wg_size;
    const size_t wg_number =
        std::min(max_work_group_num, at::ceil_div(batch_, wg_range_y_));
    glb_range_x_ = wg_range_x_;
    glb_range_y_ = wg_range_y_ * wg_number;

    // For up down sweep algorithm, each work-item handle two elements.
    // This means that one work group would handle 2 times of work group size
    // elements.
    loops_batch = at::ceil_div(batch_, glb_range_y_);
    loops_problem = at::ceil_div(problem_, wg_range_x_ * 2);
  }

  static LoopScanConfig<InputInfo, OutputInfo, IndicesInfo, T, BinaryFunction>
  make_config(
      InputInfo& input_info,
      OutputInfo& output_info,
      IndicesInfo& indices_info,
      int scan_dim,
      T init,
      ScanType type,
      BinaryFunction func) {
    size_t batch = input_info.outerSize(scan_dim);
    size_t problem = input_info.sizes[scan_dim];
    return {
        input_info,
        output_info,
        indices_info,
        batch,
        problem,
        init,
        type,
        func};
  }

  sycl::range<2> global_size() const {
    return {glb_range_y_, glb_range_x_};
  }

  sycl::range<2> group_size() const {
    return {wg_range_y_, wg_range_x_};
  }

  void set_type(ScanType other) {
    type_ = other;
  }

  struct item_desc {
    /* parallel batch, not tensor batch */ size_t glb_batch;
    /* current global assignment id */ size_t glb_problem;
  };

  item_desc get_item_desc(sycl::nd_item<2> item) const {
    auto giy = item.get_global_id(0);
    auto gix = item.get_global_id(1);

    return {giy, gix};
  }

 public:
  InputInfo input_;
  OutputInfo output_;
  IndicesInfo indices_;
  size_t batch_;
  size_t problem_;
  int64_t stride_;
  T init_;
  int loops_batch;
  int loops_problem;
  ScanType type_;
  BinaryFunction func_;
  size_t glb_range_x_;
  size_t glb_range_y_;
  size_t wg_range_x_;
  size_t wg_range_y_;
};

template <typename LSConfig_, bool TrivialOffCal = false>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::sub_group_size<32>))
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
inline void loop_scan_kernel_func(LSConfig_ cfg_) {
  using LSConfig = LSConfig_;
  using T = typename LSConfig::arg_t;

  static_assert(TrivialOffCal);
  auto item = syclext::this_work_item::get_nd_item<1>();
  const auto sg = item.get_sub_group();
  const auto lane = sg.get_local_id()[0];
  constexpr int64_t sg_size = 32;

  XPU_KERNEL_LOOP_TYPE(item, i, int64_t(cfg_.batch_) * sg_size, int64_t) {
    const int64_t batch = i / sg_size;
    T carry = cfg_.init_;
    for (int64_t base = 0; base < cfg_.problem_; base += sg_size) {
      const int64_t ix = base + lane;
      T val = cfg_.init_;
      if (ix < cfg_.problem_) {
        val = static_cast<T>(cfg_.input_.data[batch * cfg_.problem_ + ix]);
      }

      for (uint32_t offset = 1; offset < sg_size; offset <<= 1) {
        T tmp = sycl::shift_group_right(sg, val, offset);
        if (lane >= offset) {
          val = cfg_.func_(tmp, val);
        }
      }
      val = cfg_.func_(carry, val);

      if (ix < cfg_.problem_) {
        cfg_.output_.data[batch * cfg_.problem_ + ix] =
            static_cast<typename LSConfig::OutputInfoType::scalar_t>(val);
      }

      carry = sycl::group_broadcast(sg, val, sg_size - 1);
    }
  }
}

template <typename LSConfig_, bool TrivialOffCal = false>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<2>))
inline void loop_scan_with_indices_kernel_func(LSConfig_ cfg_) {
  using LSConfig = LSConfig_;
  using T = typename LSConfig::arg_t;
  using IndicesT = typename LSConfig::IndicesInfoType::scalar_t;
  using BinaryFunction = typename LSConfig::func_t;

  auto item = syclext::this_work_item::get_nd_item<2>();

  T* slm_ = static_cast<T*>(syclexp::get_work_group_scratch_memory());
  IndicesT* slm_idx_ =
      (IndicesT*)(slm_ + cfg_.wg_range_x_ * cfg_.wg_range_y_ * 2);

  const int loops_batch = cfg_.loops_batch;
  const int loops_problem = cfg_.loops_problem;
  const auto group_size_x = cfg_.wg_range_x_;

  for (int k = 0,
           base_off_batch_group = item.get_group(0) * item.get_local_range(0);
       k < loops_batch && base_off_batch_group < cfg_.batch_;
       k++, base_off_batch_group += cfg_.glb_range_y_) {
    T pre_max_carr = cfg_.init_;
    IndicesT pre_idx_carr = 0;
    int64_t base_off_batch = k * cfg_.glb_range_y_ + item.get_global_id(0);
    for (int i = 0; i < loops_problem; ++i) {
      // calculate base addr offset for each loop
      int64_t base_off_problem = i * group_size_x * 2;
      group_x_scan_by_uds_for_loop_scan_with_indices<
          LSConfig,
          T,
          IndicesT,
          BinaryFunction,
          TrivialOffCal>(
          item,
          pre_max_carr,
          pre_idx_carr,
          base_off_batch,
          base_off_problem,
          slm_,
          slm_idx_,
          cfg_);
    }
  }
}

template <typename LSConfig, bool TrivialOffCal = false>
static inline void launch_loop_scan(const LSConfig& cfg) {
  auto& queue = getCurrentSYCLQueue();

  constexpr int64_t sg_size = 32;
  constexpr int64_t wg_size = 256;
  const int64_t group_num =
      xpuKernelLoopGroupRange(int64_t(cfg.batch_) * sg_size, wg_size);
  sycl_kernel_submit<loop_scan_kernel_func<LSConfig, TrivialOffCal>>(
      sycl::range<1>(group_num * wg_size),
      sycl::range<1>(wg_size),
      queue,
      0,
      cfg);
}

template <typename LSConfig, bool TrivialOffCal = false>
static inline void launch_loop_scan_with_indices(const LSConfig& cfg) {
  auto& queue = getCurrentSYCLQueue();
  int num_elments = cfg.wg_range_x_ * cfg.wg_range_y_ * 2;
  int slm_size = num_elments * sizeof(typename LSConfig::arg_t) +
      num_elments * sizeof(typename LSConfig::IndicesInfoType::scalar_t);
  sycl_kernel_submit<
      loop_scan_with_indices_kernel_func<LSConfig, TrivialOffCal>>(
      cfg.global_size(), cfg.group_size(), queue, slm_size, cfg);
}

template <class T, class BinaryFunction>
T group_x_scan(
    sycl::nd_item<2> item,
    T value,
    T* slm,
    T init,
    BinaryFunction func) {
  const auto lix = item.get_local_id(1);
  const auto liy = item.get_local_id(0);
  const auto rx = item.get_local_range(1);

  slm[liy * rx + lix] = value;
  for (size_t offset = 1; offset < rx; offset <<= 1) {
    sycl::group_barrier(item.get_group());
    if (lix >= offset)
      value = func(slm[liy * rx + (lix - offset)], slm[liy * rx + lix]);
    sycl::group_barrier(item.get_group());

    if (lix >= offset)
      slm[liy * rx + lix] = value;
  }

  return value;
}

template <class T, class IndicesT, class BinaryFunction>
void group_x_scan_with_indices(
    sycl::nd_item<2> item,
    T& value,
    IndicesT& idx,
    sycl::local_ptr<T> slm,
    sycl::local_ptr<IndicesT> slm_idx,
    T init,
    BinaryFunction func) {
  const auto lix = item.get_local_id(1);
  const auto liy = item.get_local_id(0);
  const auto rx = item.get_local_range(1);

  slm[liy * rx + lix] = value;
  slm_idx[liy * rx + lix] = idx;
  for (int offset = 1; offset < rx; offset <<= 1) {
    sycl::group_barrier(item.get_group());
    if (lix >= offset) {
      binary_op_update(
          slm[liy * rx + (lix - offset)],
          value,
          slm_idx[liy * rx + (lix - offset)],
          idx,
          func);
    }
    sycl::group_barrier(item.get_group());

    if (lix >= offset) {
      slm[liy * rx + lix] = value;
      slm_idx[liy * rx + lix] = idx;
    }
  }
}

template <class T, class BinaryFunction>
T group_y_scan(sycl::nd_item<2> item, T value, T* temp, BinaryFunction func) {
  const auto lix = item.get_local_id(1);
  const auto liy = item.get_local_id(0);
  const auto rx = item.get_local_range(1);
  const auto ry = item.get_local_range(0);

  temp[liy * rx + lix] = value;
  for (size_t offset = 1; offset < ry; offset <<= 1) {
    sycl::group_barrier(item.get_group());
    if (liy >= offset)
      value = func(temp[(liy - offset) * rx + lix], temp[liy * rx + lix]);
    sycl::group_barrier(item.get_group());

    if (liy >= offset)
      temp[liy * rx + lix] = value;
  }

  return value;
}

template <class T, class IndicesT, class BinaryFunction>
void group_y_scan_with_indices(
    sycl::nd_item<2> item,
    T& value,
    IndicesT& idx,
    sycl::local_ptr<T> temp,
    sycl::local_ptr<IndicesT> temp_idx,
    BinaryFunction func) {
  const auto lix = item.get_local_id(1);
  const auto liy = item.get_local_id(0);
  const auto rx = item.get_local_range(1);
  const auto ry = item.get_local_range(0);

  temp[liy * rx + lix] = value;
  temp_idx[liy * rx + lix] = idx;
  for (int offset = 1; offset < ry; offset <<= 1) {
    sycl::group_barrier(item.get_group());
    if (liy >= offset) {
      binary_op_update(
          temp[(liy - offset) * rx + lix],
          value,
          temp_idx[(liy - offset) * rx + lix],
          idx,
          func);
    }
    sycl::group_barrier(item.get_group());

    if (liy >= offset) {
      temp[liy * rx + lix] = value;
      temp_idx[liy * rx + lix] = idx;
    }
  }
}

template <typename T>
struct TrivialTensorPtr {
  using scalar_t = T;
  T* data;
};

template <bool TrivialOffCal, bool ConstPtr, typename Info>
inline auto to_scan_tensor_info(const Info& info) {
  if constexpr (TrivialOffCal) {
    using ElemT = typename Info::scalar_t;
    using PtrT =
        std::conditional_t<ConstPtr, const std::remove_const_t<ElemT>, ElemT>;
    return TrivialTensorPtr<PtrT>{info.data};
  } else {
    return info;
  }
}

struct ScanItemDesc {
  size_t chunk;
  size_t chunk_size;
  size_t chunk_off;
  size_t chunk_num;
  size_t glb_batch;
  size_t glb_problem;
};

struct ScanLaunchConfig {
  size_t wg_range_x;
  size_t wg_range_y;
  size_t glb_range_x;
  size_t glb_range_y;
  int64_t problem_wg_range;
  int64_t problem_glb_range;

  sycl::range<2> global_size() const {
    return {glb_range_y, glb_range_x};
  }

  sycl::range<2> group_size() const {
    return {wg_range_y, wg_range_x};
  }
};

template <auto* kptr>
static inline ScanLaunchConfig make_scan_launch_config(
    int64_t problem,
    int64_t problem_batch,
    bool problem_along_x) {
  size_t sg_size = at::xpu::getDeviceMaxSubGroupSize();
  size_t wg_size = at::xpu::getKernelMaxWorkGroupSize<kptr>();
  size_t wg_range_x = sg_size;
  size_t wg_range_y = wg_size / wg_range_x;

  size_t range_bound_x =
      static_cast<size_t>(problem_along_x ? problem : problem_batch);
  size_t range_bound_y =
      static_cast<size_t>(problem_along_x ? problem_batch : problem);

  wg_range_y = std::min<size_t>(
      wg_range_y, std::bit_ceil(static_cast<uint64_t>(range_bound_y)));
  wg_range_x = std::max<size_t>(
      std::min<size_t>(
          wg_size / wg_range_y,
          std::bit_ceil(static_cast<uint64_t>(range_bound_x))),
      sg_size);
  wg_range_y = std::min<size_t>(
      wg_size / wg_range_x,
      std::bit_ceil(static_cast<uint64_t>(range_bound_y)));

  size_t glb_range_x = at::ceil_div(range_bound_x, wg_range_x) * wg_range_x;
  size_t glb_range_y = at::ceil_div(range_bound_y, wg_range_y) * wg_range_y;

  int64_t problem_wg_range =
      static_cast<int64_t>(problem_along_x ? wg_range_x : wg_range_y);
  int64_t problem_glb_range =
      static_cast<int64_t>(problem_along_x ? glb_range_x : glb_range_y);
  return {
      wg_range_x,
      wg_range_y,
      glb_range_x,
      glb_range_y,
      problem_wg_range,
      problem_glb_range};
}

inline ScanItemDesc get_scan_item_desc(
    sycl::nd_item<2> item,
    bool problem_along_x) {
  auto lix = item.get_local_id(1);
  auto liy = item.get_local_id(0);
  auto lrx = item.get_local_range(1);
  auto lry = item.get_local_range(0);
  auto wgrx = item.get_group_range(1);
  auto wgry = item.get_group_range(0);
  auto gix = item.get_global_id(1);
  auto giy = item.get_global_id(0);
  auto gx = item.get_group(1);
  auto gy = item.get_group(0);

  if (problem_along_x) {
    return {gx, lrx, lix, wgrx, giy, gix};
  } else {
    return {gy, lry, liy, wgry, gix, giy};
  }
}

template <class InputInfo, class OutputInfo, typename T, class BinaryFunction>
struct SegmentScanParam {
  using arg_t = T;
  using func_t = BinaryFunction;
  using InputInfoType = InputInfo;
  using OutputInfoType = OutputInfo;

  InputInfo iinfo_;
  OutputInfo oinfo_;
  T* carrier_;
  int64_t problem_;
  int64_t stride_;
  int64_t problem_batch_;
  T init_;
  ScanType type_;
  BinaryFunction func_;
  bool problem_along_x_;
};

template <
    class InputInfo,
    class OutputInfo,
    class IndicesInfo,
    typename T,
    class BinaryFunction>
struct SegmentScanWithIndicesParam
    : SegmentScanParam<InputInfo, OutputInfo, T, BinaryFunction> {
  using IndicesInfoType = IndicesInfo;
  using IndicesT = typename IndicesInfo::scalar_t;

  IndicesInfo idxinfo_;
  IndicesT* carrier_idx_;
};

template <
    class SSConfig_,
    bool TrivialOffCal = false,
    bool TrivialIdxCal = false>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<2>))
inline void segment_scan_kernel(SSConfig_ cfg_) {
  auto item = syclext::this_work_item::get_nd_item<2>();

  using SSConfig = SSConfig_;
  using T = typename SSConfig::arg_t;
  using BinaryFunction = typename SSConfig::func_t;
  using InputInfo = typename SSConfig::InputInfoType;
  using OutputInfo = typename SSConfig::OutputInfoType;

  // Use SYCL work group static/scratch memory
  // instead of SYCL local accessor for SLM usage.
  T* slm_ = static_cast<T*>(syclexp::get_work_group_scratch_memory());

  auto id = get_scan_item_desc(item, cfg_.problem_along_x_);
  int64_t si, pi, bi, glb_ldr_off, glb_str_off, glb_str_off_0,
      glb_ldr_logical_off, glb_str_logical_off, crr_off;

  int64_t e = cfg_.type_ == INCLUSIVE_TYPE ? 0 : 1;
  if constexpr (TrivialIdxCal) {
    glb_ldr_logical_off = item.get_global_linear_id();
    glb_str_logical_off = glb_ldr_logical_off + e;
    crr_off = id.chunk;
  } else {
    si = id.glb_batch % cfg_.stride_;
    bi = id.glb_batch / cfg_.stride_;
    pi = id.chunk * id.chunk_size + id.chunk_off;
    glb_ldr_logical_off =
        si + pi * cfg_.stride_ + bi * cfg_.problem_ * cfg_.stride_;
    glb_str_logical_off =
        si + (pi + e) * cfg_.stride_ + bi * cfg_.problem_ * cfg_.stride_;
    crr_off = si + id.chunk * cfg_.stride_ + bi * id.chunk_num * cfg_.stride_;
  }

  if constexpr (TrivialOffCal) {
    glb_ldr_off = glb_ldr_logical_off;
    glb_str_off = glb_str_logical_off;
    glb_str_off_0 = glb_ldr_logical_off;
  } else {
    glb_ldr_off = IndexToOffset<typename InputInfo::scalar_t, int64_t, -1>::get(
        glb_ldr_logical_off, cfg_.iinfo_);
    glb_str_off =
        IndexToOffset<typename OutputInfo::scalar_t, int64_t, -1>::get(
            glb_str_logical_off, cfg_.oinfo_);
    glb_str_off_0 =
        IndexToOffset<typename OutputInfo::scalar_t, int64_t, -1>::get(
            glb_ldr_logical_off, cfg_.oinfo_);
  }
  T value = cfg_.init_;
  if (id.glb_problem < cfg_.problem_ && id.glb_batch < cfg_.problem_batch_) {
    value = c10::load(cfg_.iinfo_.data + glb_ldr_off);
  }

  if (cfg_.problem_along_x_) {
    // so far assign all work items along problem dimension
    // sg_shuffle benefits reduce on the dimension
    value = group_x_scan<T, BinaryFunction>(
        item, value, slm_, cfg_.init_, cfg_.func_);
  } else {
    // parallel prefix reduce
    value = group_y_scan<T, BinaryFunction>(item, value, slm_, cfg_.func_);
  }

  if (id.glb_batch < cfg_.problem_batch_) {
    if (cfg_.type_ == INCLUSIVE_TYPE) {
      if (id.glb_problem < cfg_.problem_) {
        cfg_.oinfo_.data[glb_str_off] = value;
      }
    } else {
      if (id.glb_problem < cfg_.problem_ - 1 &&
          id.chunk_off < id.chunk_size - 1) {
        cfg_.oinfo_.data[glb_str_off] = value;
      }
      if (id.glb_problem < cfg_.problem_ && id.chunk_off == 0) {
        cfg_.oinfo_.data[glb_str_off_0] = cfg_.init_;
      }
    }

    if (cfg_.carrier_ != nullptr && id.chunk_off == id.chunk_size - 1) {
      cfg_.carrier_[crr_off] = value;
    }
  }
}

template <
    class SSConfig_,
    bool TrivialOffCal = false,
    bool TrivialIdxCal = false,
    bool is_idx_carried = false>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<2>))
inline void segment_scan_with_indices_kernel(SSConfig_ cfg_) {
  using SSConfig = SSConfig_;
  using T = typename SSConfig::arg_t;
  using BinaryFunction = typename SSConfig::func_t;
  using InputInfo = typename SSConfig::InputInfoType;
  using OutputInfo = typename SSConfig::OutputInfoType;
  using IndicesInfo = typename SSConfig::IndicesInfoType;
  using IndicesT = typename SSConfig::IndicesT;

  auto item = syclext::this_work_item::get_nd_item<2>();

  T* slm_ = static_cast<T*>(syclexp::get_work_group_scratch_memory());
  IndicesT* slm_idx_ =
      (IndicesT*)(slm_ + item.get_local_range(0) * item.get_local_range(1));

  auto id = get_scan_item_desc(item, cfg_.problem_along_x_);
  int64_t si, pi, bi, glb_ldr_off, glb_str_off, glb_str_off_0, glb_idx_off,
      glb_idx_off_0, glb_ldr_logical_off, glb_str_logical_off, crr_off,
      glb_idx_logical_off;

  int64_t e = cfg_.type_ == INCLUSIVE_TYPE ? 0 : 1;
  if constexpr (TrivialIdxCal) {
    glb_ldr_logical_off = item.get_global_linear_id();
    glb_str_logical_off = glb_ldr_logical_off + e;
    glb_idx_logical_off = glb_ldr_logical_off + e;
    pi = id.chunk * id.chunk_size + id.chunk_off;
    crr_off = id.chunk;
  } else {
    si = id.glb_batch % cfg_.stride_;
    bi = id.glb_batch / cfg_.stride_;
    pi = id.chunk * id.chunk_size + id.chunk_off;
    glb_ldr_logical_off =
        si + pi * cfg_.stride_ + bi * cfg_.problem_ * cfg_.stride_;
    glb_str_logical_off =
        si + (pi + e) * cfg_.stride_ + bi * cfg_.problem_ * cfg_.stride_;
    glb_idx_logical_off =
        si + (pi + e) * cfg_.stride_ + bi * cfg_.problem_ * cfg_.stride_;
    crr_off = si + id.chunk * cfg_.stride_ + bi * id.chunk_num * cfg_.stride_;
  }

  if constexpr (TrivialOffCal) {
    glb_ldr_off = glb_ldr_logical_off;
    glb_str_off = glb_str_logical_off;
    glb_str_off_0 = glb_ldr_logical_off;
    glb_idx_off = glb_idx_logical_off;
    glb_idx_off_0 = glb_ldr_logical_off;
  } else {
    glb_ldr_off = IndexToOffset<typename InputInfo::scalar_t, int64_t, -1>::get(
        glb_ldr_logical_off, cfg_.iinfo_);
    glb_str_off =
        IndexToOffset<typename OutputInfo::scalar_t, int64_t, -1>::get(
            glb_str_logical_off, cfg_.oinfo_);
    glb_str_off_0 =
        IndexToOffset<typename OutputInfo::scalar_t, int64_t, -1>::get(
            glb_ldr_logical_off, cfg_.oinfo_);
    glb_idx_off =
        IndexToOffset<typename IndicesInfo::scalar_t, int64_t, -1>::get(
            glb_idx_logical_off, cfg_.idxinfo_);
    glb_idx_off_0 =
        IndexToOffset<typename IndicesInfo::scalar_t, int64_t, -1>::get(
            glb_ldr_logical_off, cfg_.oinfo_);
  }
  T value = cfg_.init_;
  IndicesT idx = pi;
  if (id.glb_problem < cfg_.problem_ && id.glb_batch < cfg_.problem_batch_) {
    value = c10::load(cfg_.iinfo_.data + glb_ldr_off);
    if constexpr (is_idx_carried) {
      idx = cfg_.idxinfo_.data[glb_ldr_off];
    }
  }

  if (cfg_.problem_along_x_) {
    // so far assign all work items along problem dimension
    // sg_shuffle benefits reduce on the dimension
    group_x_scan_with_indices<T, IndicesT, BinaryFunction>(
        item, value, idx, slm_, slm_idx_, cfg_.init_, cfg_.func_);
  } else {
    // parallel prefix reduce
    group_y_scan_with_indices<T, IndicesT, BinaryFunction>(
        item, value, idx, slm_, slm_idx_, cfg_.func_);
  }

  if (id.glb_batch < cfg_.problem_batch_) {
    if (cfg_.type_ == INCLUSIVE_TYPE) {
      if (id.glb_problem < cfg_.problem_) {
        cfg_.oinfo_.data[glb_str_off] = value;
        cfg_.idxinfo_.data[glb_idx_off] = idx;
      }
    } else {
      if (id.glb_problem < cfg_.problem_ - 1 &&
          id.chunk_off < id.chunk_size - 1) {
        cfg_.oinfo_.data[glb_str_off] = value;
        cfg_.idxinfo_.data[glb_idx_off] = idx;
      }
      if (id.glb_problem < cfg_.problem_ && id.chunk_off == 0) {
        cfg_.oinfo_.data[glb_str_off_0] = cfg_.init_;
        cfg_.idxinfo_.data[glb_idx_off_0] = pi;
      }
    }

    if (cfg_.carrier_ != nullptr && cfg_.carrier_idx_ != nullptr &&
        id.chunk_off == id.chunk_size - 1) {
      cfg_.carrier_[crr_off] = value;
      cfg_.carrier_idx_[crr_off] = idx;
    }
  }
}

template <
    typename ScanParam,
    bool TrivialOffCal = false,
    bool TrivialIdxCal = false>
static inline void launch_segment_scan(
    const ScanLaunchConfig& launch_cfg,
    const ScanParam& param) {
  auto& queue = getCurrentSYCLQueue();
  // free function
  int slm_size = launch_cfg.wg_range_x * launch_cfg.wg_range_y *
      sizeof(typename ScanParam::arg_t);
  sycl_kernel_submit<
      segment_scan_kernel<ScanParam, TrivialOffCal, TrivialIdxCal>>(
      launch_cfg.global_size(),
      launch_cfg.group_size(),
      queue,
      slm_size,
      param);
}

template <
    typename ScanParam,
    bool TrivialOffCal = false,
    bool TrivialIdxCal = false,
    bool is_idx_carried = false>
static inline void launch_segment_scan_with_indices(
    const ScanLaunchConfig& launch_cfg,
    const ScanParam& param) {
  auto& queue = getCurrentSYCLQueue();
  int num_elments = launch_cfg.wg_range_x * launch_cfg.wg_range_y;
  int slm_size = num_elments * sizeof(typename ScanParam::arg_t) +
      num_elments * sizeof(typename ScanParam::IndicesT);
  sycl_kernel_submit<segment_scan_with_indices_kernel<
      ScanParam,
      TrivialOffCal,
      TrivialIdxCal,
      is_idx_carried>>(
      launch_cfg.global_size(),
      launch_cfg.group_size(),
      queue,
      slm_size,
      param);
}

template <typename OutputT, typename T, class BinaryFunction>
struct AccumulateCarrierParam {
  OutputT* out_;
  const T* carrier_;
  int64_t problem_;
  int64_t stride_;
  int64_t problem_batch_;
  BinaryFunction func_;
  bool problem_along_x_;
};

template <typename OutputT, typename T, typename IndicesT, class BinaryFunction>
struct AccumulateCarrierWithIndicesParam
    : AccumulateCarrierParam<OutputT, T, BinaryFunction> {
  IndicesT* idx_;
  const IndicesT* carrier_idx_;
};

template <class ACParam, bool TrivialIdxCal = false>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<2>))
inline void accumulate_carrier_kernel(ACParam cfg) {
  auto item = syclext::this_work_item::get_nd_item<2>();

  auto id = get_scan_item_desc(item, cfg.problem_along_x_);
  int64_t si, pi, bi, glb_off, crr_off;
  if constexpr (TrivialIdxCal) {
    glb_off = item.get_global_linear_id();
    crr_off = id.chunk;
  } else {
    si = id.glb_batch % cfg.stride_;
    bi = id.glb_batch / cfg.stride_;
    pi = id.chunk * id.chunk_size + id.chunk_off;
    glb_off = si + pi * cfg.stride_ + bi * cfg.problem_ * cfg.stride_;
    crr_off = si + id.chunk * cfg.stride_ + bi * id.chunk_num * cfg.stride_;
  }
  if (id.glb_problem < cfg.problem_ && id.glb_batch < cfg.problem_batch_) {
    cfg.out_[glb_off] = cfg.func_(cfg.out_[glb_off], cfg.carrier_[crr_off]);
  }
}

template <class ACParam, bool TrivialIdxCal = false>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<2>))
inline void accumulate_carrier_with_indices_kernel(ACParam cfg_) {
  auto item = syclext::this_work_item::get_nd_item<2>();

  auto id = get_scan_item_desc(item, cfg_.problem_along_x_);
  int64_t si, pi, bi, glb_off, crr_off;
  if constexpr (TrivialIdxCal) {
    glb_off = item.get_global_linear_id();
    crr_off = id.chunk;
  } else {
    si = id.glb_batch % cfg_.stride_;
    bi = id.glb_batch / cfg_.stride_;
    pi = id.chunk * id.chunk_size + id.chunk_off;
    glb_off = si + pi * cfg_.stride_ + bi * cfg_.problem_ * cfg_.stride_;
    crr_off = si + id.chunk * cfg_.stride_ + bi * id.chunk_num * cfg_.stride_;
  }
  if (id.glb_problem < cfg_.problem_ && id.glb_batch < cfg_.problem_batch_) {
    binary_op_update(
        cfg_.carrier_[crr_off],
        cfg_.out_[glb_off],
        cfg_.carrier_idx_[crr_off],
        cfg_.idx_[glb_off],
        cfg_.func_);
  }
}

static inline bool dispatch_to_loop_scan_kernel(
    const int64_t problem,
    const int64_t stride,
    const int64_t batch) {
  // stride > 1 scenario
  if (stride > 1) {
    // If stride > 1, we use batch scan anyway.
    return false;
  }

  // 1 == stride scenario
  if (batch > 128 && problem < 16384 /*1024 * 16*/) {
    // Only if batch > 128, and problem is not so big, we use loop
    // scan kernel to avoid so many global memory access operations.
    // If batch is so small, or problem is so big, we use batch scan
    // to increase work group number to increase device coverage.
    return true;
  }

  return false;
}

template <class ScanParam, bool TrivialIdxCal = false>
static inline void accumulate_carrier(
    const ScanLaunchConfig& launch_cfg,
    const ScanParam& param) {
  TORCH_CHECK(
      param.carrier_ != nullptr, "scan: nullptr carrier in accumulation ...");
  auto& queue = getCurrentSYCLQueue();

  using OutputT = typename ScanParam::OutputInfoType::scalar_t;
  using T = typename ScanParam::arg_t;
  using BinaryFunction = typename ScanParam::func_t;
  using ACParam = AccumulateCarrierParam<OutputT, T, BinaryFunction>;
  ACParam ac_param{
      param.oinfo_.data,
      param.carrier_,
      param.problem_,
      param.stride_,
      param.problem_batch_,
      param.func_,
      param.problem_along_x_};
  sycl_kernel_submit<accumulate_carrier_kernel<ACParam, TrivialIdxCal>>(
      launch_cfg.global_size(), launch_cfg.group_size(), queue, 0, ac_param);
}

template <class ScanParam, bool TrivialIdxCal = false>
static inline void accumulate_carrier_with_indices(
    const ScanLaunchConfig& launch_cfg,
    const ScanParam& param) {
  TORCH_CHECK(
      param.carrier_ != nullptr, "scan: nullptr carrier in accumulation ...");
  TORCH_CHECK(
      param.carrier_idx_ != nullptr,
      "scan_with_indices: nullptr carrier in accumulation ...");
  auto& queue = getCurrentSYCLQueue();

  using OutputT = typename ScanParam::OutputInfoType::scalar_t;
  using T = typename ScanParam::arg_t;
  using IndicesT = typename ScanParam::IndicesT;
  using BinaryFunction = typename ScanParam::func_t;
  using ACParam =
      AccumulateCarrierWithIndicesParam<OutputT, T, IndicesT, BinaryFunction>;
  ACParam ac_param{
      {param.oinfo_.data,
       param.carrier_,
       param.problem_,
       param.stride_,
       param.problem_batch_,
       param.func_,
       param.problem_along_x_},
      param.idxinfo_.data,
      param.carrier_idx_};
  sycl_kernel_submit<
      accumulate_carrier_with_indices_kernel<ACParam, TrivialIdxCal>>(
      launch_cfg.global_size(), launch_cfg.group_size(), queue, 0, ac_param);
}

template <
    ScanType Type,
    bool TrivialOffCal,
    typename T,
    class InputInfo,
    class OutputInfo,
    class BinaryFunction>
static inline void loop_scan_kernel(
    InputInfo& input_info,
    OutputInfo& output_info,
    int dim_after_collapse,
    T init,
    BinaryFunction func) {
  auto cfg =
      LoopScanConfig<InputInfo, OutputInfo, OutputInfo, T, BinaryFunction>::
          make_config(
              input_info,
              output_info,
              output_info,
              dim_after_collapse,
              init,
              Type,
              func);
  TORCH_CHECK(1 == cfg.stride_);
  launch_loop_scan<decltype(cfg), TrivialOffCal>(cfg);

  return;
}

template <
    ScanType Type,
    bool TrivialOffCal,
    typename T,
    class InputInfo,
    class OutputInfo,
    class IndicesInfo,
    class BinaryFunction>
static inline void loop_scan_kernel_with_indices(
    InputInfo& input_info,
    OutputInfo& output_info,
    IndicesInfo& indices_info,
    int dim_after_collapse,
    T init,
    BinaryFunction func) {
  auto cfg =
      LoopScanConfig<InputInfo, OutputInfo, IndicesInfo, T, BinaryFunction>::
          make_config(
              input_info,
              output_info,
              indices_info,
              dim_after_collapse,
              init,
              Type,
              func);
  TORCH_CHECK(1 == cfg.stride_);
  launch_loop_scan_with_indices<decltype(cfg), TrivialOffCal>(cfg);

  return;
}

template <
    ScanType Type,
    bool TrivialOffCal,
    bool TrivialIdxCal,
    typename T,
    class InputInfo,
    class OutputInfo,
    class BinaryFunction>
static inline void _segment_scan_kernel(
    InputInfo& input_info,
    OutputInfo& output_info,
    int dim_after_collapse,
    T init,
    BinaryFunction func) {
  int64_t batch = input_info.outerSize(dim_after_collapse);
  int64_t stride = input_info.innerSize(dim_after_collapse);
  int64_t problem = input_info.sizes[dim_after_collapse];
  int64_t problem_batch = batch * stride;
  bool problem_along_x = input_info.strides[dim_after_collapse] == 1;
  auto scan_input_info = to_scan_tensor_info<TrivialOffCal, true>(input_info);
  auto scan_output_info =
      to_scan_tensor_info<TrivialOffCal, false>(output_info);
  using ScanInputInfo = decltype(scan_input_info);
  using ScanOutputInfo = decltype(scan_output_info);
  using ScanParam =
      SegmentScanParam<ScanInputInfo, ScanOutputInfo, T, BinaryFunction>;
  auto launch_cfg = make_scan_launch_config<
      segment_scan_kernel<ScanParam, TrivialOffCal, TrivialIdxCal>>(
      problem, problem_batch, problem_along_x);
  ScanParam param{
      scan_input_info,
      scan_output_info,
      nullptr,
      problem,
      stride,
      problem_batch,
      init,
      Type,
      func,
      problem_along_x};
  // 0. recursive convergence
  if (problem <= launch_cfg.problem_wg_range) {
    launch_segment_scan<ScanParam, TrivialOffCal, TrivialIdxCal>(
        launch_cfg, param);
    return;
  }

  // 1. inclusive scan in each chunk
  Tensor carrier_holder = at::empty(
      {batch,
       launch_cfg.problem_glb_range / launch_cfg.problem_wg_range,
       stride},
      map_options<T>());
  TensorInfo<T, int64_t> carrier_info =
      getTensorInfo<T, int64_t>(carrier_holder);
  param.carrier_ = carrier_info.data;
  launch_segment_scan<ScanParam, TrivialOffCal, TrivialIdxCal>(
      launch_cfg, param);

  // 2. recursion for carrier
  _segment_scan_kernel<EXCLUSIVE_TYPE, TrivialOffCal, TrivialIdxCal>(
      carrier_info, carrier_info, 1, init, func);

  // 3. accumulate among all chunk
  accumulate_carrier<ScanParam, TrivialIdxCal>(launch_cfg, param);

  return;
}

template <
    ScanType Type,
    bool TrivialOffCal,
    bool TrivialIdxCal,
    bool is_idx_carried,
    typename T,
    class InputInfo,
    class OutputInfo,
    class IndicesInfo,
    class BinaryFunction>
static inline void _segment_scan_kernel_with_indices(
    InputInfo& input_info,
    OutputInfo& output_info,
    IndicesInfo& indices_info,
    int dim_after_collapse,
    T init,
    BinaryFunction func) {
  using IndicesT = typename IndicesInfo::scalar_t;
  int64_t batch = input_info.outerSize(dim_after_collapse);
  int64_t stride = input_info.innerSize(dim_after_collapse);
  int64_t problem = input_info.sizes[dim_after_collapse];
  int64_t problem_batch = batch * stride;
  bool problem_along_x = input_info.strides[dim_after_collapse] == 1;
  auto scan_input_info = to_scan_tensor_info<TrivialOffCal, true>(input_info);
  auto scan_output_info =
      to_scan_tensor_info<TrivialOffCal, false>(output_info);
  auto scan_indices_info =
      to_scan_tensor_info<TrivialOffCal, false>(indices_info);
  using ScanInputInfo = decltype(scan_input_info);
  using ScanOutputInfo = decltype(scan_output_info);
  using ScanIndicesInfo = decltype(scan_indices_info);
  using ScanParam = SegmentScanWithIndicesParam<
      ScanInputInfo,
      ScanOutputInfo,
      ScanIndicesInfo,
      T,
      BinaryFunction>;
  auto launch_cfg = make_scan_launch_config<segment_scan_with_indices_kernel<
      ScanParam,
      TrivialOffCal,
      TrivialIdxCal,
      is_idx_carried>>(problem, problem_batch, problem_along_x);
  ScanParam param{
      {scan_input_info,
       scan_output_info,
       nullptr,
       problem,
       stride,
       problem_batch,
       init,
       Type,
       func,
       problem_along_x},
      scan_indices_info,
      nullptr};
  // 0. recursive convergence
  if (problem <= launch_cfg.problem_wg_range) {
    launch_segment_scan_with_indices<
        ScanParam,
        TrivialOffCal,
        TrivialIdxCal,
        is_idx_carried>(launch_cfg, param);
    return;
  }

  // 1. inclusive scan in each chunk
  Tensor carrier_holder = at::empty(
      {batch,
       launch_cfg.problem_glb_range / launch_cfg.problem_wg_range,
       stride},
      map_options<T>());
  TensorInfo<T, int64_t> carrier_info =
      getTensorInfo<T, int64_t>(carrier_holder);
  param.carrier_ = carrier_info.data;

  Tensor carrier_idx_holder = at::empty(
      {batch,
       launch_cfg.problem_glb_range / launch_cfg.problem_wg_range,
       stride},
      map_options<IndicesT>());
  TensorInfo<IndicesT, int64_t> carrier_idx_info =
      getTensorInfo<IndicesT, int64_t>(carrier_idx_holder);
  param.carrier_idx_ = carrier_idx_info.data;
  launch_segment_scan_with_indices<
      ScanParam,
      TrivialOffCal,
      TrivialIdxCal,
      is_idx_carried>(launch_cfg, param);

  // 2. recursion for carrier
  _segment_scan_kernel_with_indices<
      EXCLUSIVE_TYPE,
      TrivialOffCal,
      TrivialIdxCal,
      true>(carrier_info, carrier_info, carrier_idx_info, 1, init, func);

  // 3. accumulate among all chunk
  accumulate_carrier_with_indices<ScanParam, TrivialIdxCal>(launch_cfg, param);

  return;
}

template <
    ScanType Type,
    typename scalar_t,
    typename oscalar_t,
    class BinaryFunction>
void scan(
    const Tensor& self,
    const Tensor& input,
    int dimension,
    scalar_t init,
    BinaryFunction func) {
  if (self.sizes() != input.sizes()) {
    at::native::resize_output(self, input.sizes());
  }
  if (input.dim() == 0) {
    self.fill_(input);
    return;
  } else if (input.numel() == 0) {
    self.zero_();
    return;
  }
  auto input_ = input.contiguous();
  dimension = maybe_wrap_dim(dimension, input_.dim());
  TORCH_CHECK(
      dimension >= 0 && dimension < input_.dim(),
      "dimension ",
      dimension,
      " out of range");

  TORCH_INTERNAL_ASSERT(self.is_contiguous());

  TensorInfo<const scalar_t, int64_t> input_info =
      getTensorInfo<const scalar_t, int64_t>(input_);
  int dim_after_collapse = input_info.collapseDims(dimension);

  TensorInfo<oscalar_t, int64_t> output_info =
      getTensorInfo<oscalar_t, int64_t>(self);
  output_info.collapseDims(dimension);

  int64_t batch = input_info.outerSize(dim_after_collapse);
  int64_t stride = input_info.innerSize(dim_after_collapse);
  int64_t problem = input_info.sizes[dim_after_collapse];

  if (dispatch_to_loop_scan_kernel(problem, stride, batch)) {
    using value_t = std::remove_const_t<scalar_t>;
    if constexpr (std::is_same_v<BinaryFunction, std::plus<value_t>>) {
      // Sum in opmath precision; std::plus<value_t> would round each partial
      // sum to value_t.
      using acc_t = at::opmath_type<value_t>;
      loop_scan_kernel<Type, true>(
          input_info,
          output_info,
          dim_after_collapse,
          static_cast<acc_t>(init),
          std::plus<acc_t>());
    } else {
      loop_scan_kernel<Type, true>(
          input_info, output_info, dim_after_collapse, init, func);
    }
  } else {
    if (batch == 1 && stride == 1) {
      _segment_scan_kernel<Type, true, true>(
          input_info, output_info, dim_after_collapse, init, func);
    } else {
      _segment_scan_kernel<Type, true, false>(
          input_info, output_info, dim_after_collapse, init, func);
    }
  }
}

template <
    ScanType Type,
    typename scalar_t,
    typename oscalar_t,
    typename iscalar_t,
    class BinaryFunction>
void scan_with_indices(
    const Tensor& self_,
    const Tensor& values,
    const Tensor& indices,
    int dimension,
    scalar_t init,
    BinaryFunction func) {
  auto self = self_.contiguous();
  TORCH_INTERNAL_ASSERT(values.is_contiguous() && indices.is_contiguous());

  // Empty input: values/indices are empty too, nothing to write.
  if (self.numel() == 0) {
    return;
  }

  dimension = maybe_wrap_dim(dimension, self.dim());
  TORCH_CHECK(
      dimension >= 0 && dimension < self.dim(),
      "dimension ",
      dimension,
      " out of range");

  TensorInfo<const scalar_t, int64_t> input_info =
      getTensorInfo<const scalar_t, int64_t>(self);
  int dim_after_collapse = input_info.collapseDims(dimension);

  TensorInfo<oscalar_t, int64_t> output_info =
      getTensorInfo<oscalar_t, int64_t>(values);
  output_info.collapseDims(dimension);

  TensorInfo<iscalar_t, int64_t> indices_info =
      getTensorInfo<iscalar_t, int64_t>(indices);

  int64_t batch = input_info.outerSize(dim_after_collapse);
  int64_t stride = input_info.innerSize(dim_after_collapse);
  int64_t problem = input_info.sizes[dim_after_collapse];

  if (dispatch_to_loop_scan_kernel(problem, stride, batch)) {
    loop_scan_kernel_with_indices<Type, true>(
        input_info, output_info, indices_info, dim_after_collapse, init, func);
  } else {
    if (batch == 1 && stride == 1) {
      _segment_scan_kernel_with_indices<Type, true, true, false>(
          input_info,
          output_info,
          indices_info,
          dim_after_collapse,
          init,
          func);
    } else {
      _segment_scan_kernel_with_indices<Type, true, false, false>(
          input_info,
          output_info,
          indices_info,
          dim_after_collapse,
          init,
          func);
    }
  }
}

} // namespace at::native::xpu
