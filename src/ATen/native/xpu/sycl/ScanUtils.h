#pragma once

#include <ATen/native/Resize.h>
#include <ATen/native/xpu/sycl/BatchKernel.h>
#include <comm/SYCLContext.h>
#include <comm/TensorInfo.h>
#include <comm/TensorOptions.h>

namespace at::native::xpu {
using namespace at::xpu::detail;
using namespace at::xpu;
template <typename T>
inline T CeilDiv(T a, T b) {
  return (a + b - 1) / b;
}

typedef enum {
  EXCLUSIVE_TYPE = 0,
  INCLUSIVE_TYPE = 1,
} ScanType;

// group x scan by using up down sweep algorithm(call uds for short)
template <
    class LSConfig,
    class T,
    class BinaryFunction,
    bool TrivialOffCal = false>
T inline group_x_scan_by_uds_for_loop_scan(
    sycl::nd_item<2> item,
    const T pre_max_carr,
    size_t base_off_batch,
    size_t base_off_problem,
    sycl::local_ptr<T> slm,
    LSConfig cfg) {
  using InputInfo = typename LSConfig::InputInfoType;
  using OutputInfo = typename LSConfig::OutputInfoType;

  size_t glb_ldr_off_0, glb_ldr_off_1, glb_str_off_0, glb_str_off_1,
      glb_ldr_logical_off_0, glb_ldr_logical_off_1, glb_str_logical_off_0,
      glb_str_logical_off_1;

  const auto sub_group = item.get_sub_group();
  const auto sub_group_size = sub_group.get_local_range()[0];

  typename LSConfig::item_desc id = cfg.get_item_desc(item);

  const auto lix = item.get_local_id(1);
  const auto liy = item.get_local_id(0);
  const auto rx = item.get_local_range(1);

  size_t ix0 = base_off_problem + lix;
  size_t ix1 = base_off_problem + rx + lix;
  size_t glb0 = base_off_batch * cfg.problem_ + ix0;
  size_t glb1 = base_off_batch * cfg.problem_ + ix1;
  if constexpr (TrivialOffCal) {
    glb_ldr_off_0 = glb0;
    glb_ldr_off_1 = glb1;
    glb_str_off_0 = glb0;
    glb_str_off_1 = glb1;
  } else {
    glb_ldr_logical_off_0 = glb0;
    glb_ldr_off_0 = IndexToOffset<typename InputInfo::scalar_t, int64_t>::get(
        glb_ldr_logical_off_0,
        cfg.input_,
        IndexToOffset<typename InputInfo::scalar_t, int64_t>::
            NON_STRICT_CONTIGUOUS);

    glb_ldr_logical_off_1 = glb1;
    glb_ldr_off_1 = IndexToOffset<typename InputInfo::scalar_t, int64_t>::get(
        glb_ldr_logical_off_1,
        cfg.input_,
        IndexToOffset<typename InputInfo::scalar_t, int64_t>::
            NON_STRICT_CONTIGUOUS);

    glb_str_logical_off_0 = glb0;
    glb_str_off_0 = IndexToOffset<typename OutputInfo::scalar_t, int64_t>::get(
        glb_str_logical_off_0,
        cfg.output_,
        IndexToOffset<typename OutputInfo::scalar_t, int64_t>::
            NON_STRICT_CONTIGUOUS);

    glb_str_logical_off_1 = glb1;
    glb_str_off_1 = IndexToOffset<typename OutputInfo::scalar_t, int64_t>::get(
        glb_str_logical_off_1,
        cfg.output_,
        IndexToOffset<typename OutputInfo::scalar_t, int64_t>::
            NON_STRICT_CONTIGUOUS);
  }
  // TODO: opti for bank conflict elemination
  // Read data from global memory to shared local memory
  // Each work item load 2 elements from global device memory to shared local
  // memory
  if (base_off_batch < cfg.batch_) {
    if (ix0 < cfg.problem_) {
      slm[liy * rx * 2 + lix] = cfg.input_.data[glb_ldr_off_0];
    } else {
      slm[liy * rx * 2 + lix] = cfg.init_;
    }

    if (ix1 < cfg.problem_) {
      slm[liy * rx * 2 + rx + lix] = cfg.input_.data[glb_ldr_off_1];
    } else {
      slm[liy * rx * 2 + rx + lix] = cfg.init_;
    }

    // Add the total value of all previous work groups to the first value of
    // this work group.
    if (0 == lix) {
      slm[liy * rx * 2 + lix] =
          cfg.func_(slm[liy * rx * 2 + lix], pre_max_carr);
    }
  }
  item.barrier(sycl::access::fence_space::local_space);

  // Parallel reduction (Up-sweep)
  for (uint32_t s = rx, d = 1; s >= 1; s >>= 1, d <<= 1) {
    if (base_off_batch < cfg.batch_ && lix < s) {
      uint32_t offset = liy * rx * 2 + (2 * lix + 1) * d - 1;
      slm[offset + d] = cfg.func_(slm[offset], slm[offset + d]);
    }
    if (sub_group_size != cfg.wg_range_x_) {
      item.barrier(sycl::access::fence_space::local_space);
    }
  }

  // Down-sweep
  for (uint32_t s = 2, d = rx / 2; d >= 1; s <<= 1, d >>= 1) {
    if (base_off_batch < cfg.batch_ && lix < s - 1) {
      uint32_t offset = liy * rx * 2 + 2 * (lix + 1) * d - 1;
      slm[offset + d] = cfg.func_(slm[offset], slm[offset + d]);
    }
    if (sub_group_size != cfg.wg_range_x_) {
      item.barrier(sycl::access::fence_space::local_space);
    }
  }

  // Write back from shared local memory to global memory
  if (base_off_batch < cfg.batch_) {
    if (ix0 < cfg.problem_) {
      cfg.output_.data[glb_str_off_0] = slm[liy * rx * 2 + lix];
    }
    if (ix1 < cfg.problem_) {
      cfg.output_.data[glb_str_off_1] = slm[liy * rx * 2 + rx + lix];
    }
  }

  // each work item would return current max carr
  return slm[liy * rx * 2 + 2 * rx - 1];
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
  using IndicesInfoType = OutputInfo;

  LoopScanConfig() {}

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
        wg_range_x_(0),
        wg_range_y_(0) {
    size_t wg_size = syclMaxWorkItemsPerEU();
    wg_range_x_ = 32;
    while (problem_ <= wg_range_x_ >> 1) {
      wg_range_x_ = wg_range_x_ >> 1;
    }
    wg_range_y_ = wg_size / wg_range_x_;
    const auto target_global_size = syclMaxWorkItemsPerTile();
    ;
    const size_t max_work_group_num = target_global_size / wg_size;
    const size_t wg_number =
        std::min(max_work_group_num, CeilDiv(batch_, wg_range_y_));
    glb_range_x_ = wg_range_x_;
    glb_range_y_ = wg_range_y_ * wg_number;

    // For up down sweep algorithm, each work-item handle two elements.
    // This means that one work group would handle 2 times of work group size
    // elements.
    loops_batch = (batch_ + glb_range_y_ - 1) / glb_range_y_;
    loops_problem = (problem_ + (wg_range_x_ * 2) - 1) / (wg_range_x_ * 2);
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
class LoopScanKernel : public __SYCL_KER_CONFIG_CONVENTION__ {
  using LSConfig = LSConfig_;
  using T = typename LSConfig::arg_t;
  using BinaryFunction = typename LSConfig::func_t;

 public:
  LoopScanKernel(const LSConfig& cfg) : cfg_(cfg), slm_(), max_carr_() {}

  void operator()(sycl::nd_item<2> item) const {
    const int loops_batch = cfg_.loops_batch;
    const int loops_problem = cfg_.loops_problem;
    const auto group_size_x = cfg_.wg_range_x_;
    const auto liy = item.get_local_id(0);

    for (int k = 0,
             base_off_batch_group = item.get_group(0) * item.get_local_range(0);
         k < loops_batch && base_off_batch_group < cfg_.batch_;
         k++, base_off_batch_group += cfg_.glb_range_y_) {
      max_carr_[liy] = cfg_.init_;
      int64_t base_off_batch = k * cfg_.glb_range_y_ + item.get_global_id(0);
      for (int i = 0; i < loops_problem; ++i) {
        // calculate base addr offset for each loop
        int64_t base_off_problem = i * group_size_x * 2;
        max_carr_[liy] = group_x_scan_by_uds_for_loop_scan<
            LSConfig,
            T,
            BinaryFunction,
            TrivialOffCal>(
            item, max_carr_[liy], base_off_batch, base_off_problem, slm_, cfg_);
      }
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    int slm_size = cfg_.wg_range_x_ * cfg_.wg_range_y_ * 2;
    int carr_size = cfg_.wg_range_y_;
    slm_ = sycl::local_accessor<typename LSConfig::arg_t>(slm_size, cgh);
    max_carr_ = sycl::local_accessor<typename LSConfig::arg_t>(carr_size, cgh);
  }

 private:
  LSConfig cfg_;
  sycl::local_accessor<T> slm_;
  sycl::local_accessor<T> max_carr_;
};

template <typename LSConfig, bool TrivialOffCal = false>
static inline void launch_loop_scan(const LSConfig& cfg) {
  auto& queue = getCurrentSYCLQueue();

  LoopScanKernel<LSConfig, TrivialOffCal> kfn(cfg);

  sycl_kernel_submit(cfg.global_size(), cfg.group_size(), queue, kfn);
}

template <class T, class BinaryFunction>
T group_x_scan(
    sycl::nd_item<2> item,
    T value,
    sycl::local_ptr<T> slm,
    T init,
    BinaryFunction func) {
  const auto lix = item.get_local_id(1);
  const auto liy = item.get_local_id(0);
  const auto rx = item.get_local_range(1);

  slm[liy * rx + lix] = value;
  for (size_t offset = 1; offset < rx; offset <<= 1) {
    item.barrier(sycl::access::fence_space::local_space);
    if (lix >= offset)
      value = func(slm[liy * rx + (lix - offset)], slm[liy * rx + lix]);
    item.barrier(sycl::access::fence_space::local_space);

    if (lix >= offset)
      slm[liy * rx + lix] = value;
  }

  return value;
}

template <class T, class BinaryFunction>
T group_y_scan(
    sycl::nd_item<2> item,
    T value,
    sycl::local_ptr<T> temp,
    BinaryFunction func) {
  const auto lix = item.get_local_id(1);
  const auto liy = item.get_local_id(0);
  const auto rx = item.get_local_range(1);
  const auto ry = item.get_local_range(0);

  temp[liy * rx + lix] = value;
  for (size_t offset = 1; offset < ry; offset <<= 1) {
    item.barrier(sycl::access::fence_space::local_space);
    if (liy >= offset)
      value = func(temp[(liy - offset) * rx + lix], temp[liy * rx + lix]);
    item.barrier(sycl::access::fence_space::local_space);

    if (liy >= offset)
      temp[liy * rx + lix] = value;
  }

  return value;
}

template <
    class InputInfo,
    class OutputInfo,
    class IndicesInfo,
    typename T,
    class BinaryFunction>
class SegmentScanConfig : public BatchKernelConfig {
 public:
  using arg_t = T;
  using func_t = BinaryFunction;
  using InputInfoType = InputInfo;
  using OutputInfoType = OutputInfo;
  using IndicesInfoType = IndicesInfo;

  SegmentScanConfig() {}

  SegmentScanConfig(
      InputInfo input_info,
      OutputInfo output_info,
      IndicesInfo indices_info,
      int64_t batch,
      int64_t problem,
      int64_t stride,
      bool problem_along_x,
      T init,
      ScanType type,
      BinaryFunction func)
      : BatchKernelConfig(
            batch,
            problem,
            stride,
            batch * stride,
            problem_along_x),
        iinfo_(input_info),
        oinfo_(output_info),
        idxinfo_(indices_info),
        init_(init),
        type_(type),
        func_(func),
        carrier_(nullptr),
        carrier_idx_(nullptr) {}

  template <class KernelClass>
  static SegmentScanConfig<
      InputInfo,
      OutputInfo,
      IndicesInfo,
      T,
      BinaryFunction>
  make_config(
      InputInfo& input_info,
      OutputInfo& output_info,
      IndicesInfo& indices_info,
      int scan_dim,
      T init,
      ScanType type,
      BinaryFunction func) {
    int64_t batch = input_info.outerSize(scan_dim);
    int64_t stride = input_info.innerSize(scan_dim);
    int64_t problem = input_info.sizes[scan_dim];
    bool problem_along_x = input_info.strides[scan_dim] == 1 ? true : false;

    SegmentScanConfig<InputInfo, OutputInfo, IndicesInfo, T, BinaryFunction>
        cfg = {
            input_info,
            output_info,
            indices_info,
            batch,
            problem,
            stride,
            problem_along_x,
            init,
            type,
            func};

    cfg.template build<KernelClass>();
    return cfg;
  }

  int64_t carrier_size() {
    return problem_glb_range_ / problem_wg_range_ * batch_ * stride_;
  }

  void set_carrier(T* other) {
    carrier_ = other;
  }

  void set_carrier_idx(int64_t* other) {
    carrier_idx_ = other;
  }

  void set_type(ScanType other) {
    type_ = other;
  }

 public:
  InputInfo iinfo_;
  OutputInfo oinfo_;
  IndicesInfo idxinfo_;
  T init_;
  ScanType type_;
  BinaryFunction func_;
  /* contiguous temp buffer */ T* carrier_;
  /* contiguous temp buffer */ int64_t* carrier_idx_;
};

template <
    class SSConfig_,
    bool TrivialOffCal = false,
    bool TrivialIdxCal = false>
class SegmentScanKernel : public __SYCL_KER_CONFIG_CONVENTION__ {
 public:
  using SSConfig = SSConfig_;
  using T = typename SSConfig::arg_t;
  using BinaryFunction = typename SSConfig::func_t;
  using InputInfo = typename SSConfig::InputInfoType;
  using OutputInfo = typename SSConfig::OutputInfoType;

  SegmentScanKernel(const SSConfig& cfg) : cfg(cfg), slm() {}

 public:
  void operator()(sycl::nd_item<2> item) const {
    auto id = cfg.get_item_desc(item);
    int64_t si, pi, bi, glb_ldr_off, glb_str_off, glb_str_off_0,
        glb_ldr_logical_off, glb_str_logical_off, crr_off;

    int64_t e = cfg.type_ == INCLUSIVE_TYPE ? 0 : 1;
    if constexpr (TrivialIdxCal) {
      glb_ldr_logical_off = item.get_global_linear_id();
      glb_str_logical_off = glb_ldr_logical_off + e;
      crr_off = id.chunk;
    } else {
      si = id.glb_batch % cfg.stride_;
      bi = id.glb_batch / cfg.stride_;
      pi = id.chunk * id.chunk_size + id.chunk_off;
      glb_ldr_logical_off =
          si + pi * cfg.stride_ + bi * cfg.problem_ * cfg.stride_;
      glb_str_logical_off =
          si + (pi + e) * cfg.stride_ + bi * cfg.problem_ * cfg.stride_;
      crr_off = si + id.chunk * cfg.stride_ + bi * id.chunk_num * cfg.stride_;
    }

    if constexpr (TrivialOffCal) {
      glb_ldr_off = glb_ldr_logical_off;
      glb_str_off = glb_str_logical_off;
      glb_str_off_0 = glb_ldr_logical_off;
    } else {
      glb_ldr_off = IndexToOffset<typename InputInfo::scalar_t, int64_t>::get(
          glb_ldr_logical_off,
          cfg.iinfo_,
          IndexToOffset<typename InputInfo::scalar_t, int64_t>::
              NON_STRICT_CONTIGUOUS);
      glb_str_off = IndexToOffset<typename OutputInfo::scalar_t, int64_t>::get(
          glb_str_logical_off,
          cfg.oinfo_,
          IndexToOffset<typename InputInfo::scalar_t, int64_t>::
              NON_STRICT_CONTIGUOUS);
      glb_str_off_0 =
          IndexToOffset<typename OutputInfo::scalar_t, int64_t>::get(
              glb_ldr_logical_off,
              cfg.oinfo_,
              IndexToOffset<typename InputInfo::scalar_t, int64_t>::
                  NON_STRICT_CONTIGUOUS);
    }
    T value = cfg.init_;
    if (id.glb_problem < cfg.problem_ && id.glb_batch < cfg.problem_batch_) {
      value = cfg.iinfo_.data[glb_ldr_off];
    }

    if (cfg.problem_along_x_) {
      // so far assign all work items along problem dimension
      // sg_shuffle benefits reduce on the dimension
      value = group_x_scan<T, BinaryFunction>(
          item, value, slm, cfg.init_, cfg.func_);
    } else {
      // parallel prefix reduce
      value = group_y_scan<T, BinaryFunction>(item, value, slm, cfg.func_);
    }

    if (id.glb_batch < cfg.problem_batch_) {
      if (cfg.type_ == INCLUSIVE_TYPE) {
        if (id.glb_problem < cfg.problem_) {
          cfg.oinfo_.data[glb_str_off] = value;
        }
      } else {
        if (id.glb_problem < cfg.problem_ - 1 &&
            id.chunk_off < id.chunk_size - 1) {
          cfg.oinfo_.data[glb_str_off] = value;
        }
        if (id.glb_problem < cfg.problem_ && id.chunk_off == 0) {
          cfg.oinfo_.data[glb_str_off_0] = cfg.init_;
        }
      }

      if (cfg.carrier_ != nullptr && id.chunk_off == id.chunk_size - 1) {
        cfg.carrier_[crr_off] = value;
      }
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    int slm_size = cfg.wg_range_x_ * cfg.wg_range_y_;
    slm = sycl::local_accessor<T>(slm_size, cgh);
  }

 private:
  SSConfig cfg;
  sycl::local_accessor<T> slm;
};

template <
    typename SSConfig,
    bool TrivialOffCal = false,
    bool TrivialIdxCal = false>
static inline void launch_segment_scan(const SSConfig& cfg) {
  auto& queue = getCurrentSYCLQueue();
  SegmentScanKernel<SSConfig, TrivialOffCal, TrivialIdxCal> kfn(cfg);
  sycl_kernel_submit(cfg.global_size(), cfg.group_size(), queue, kfn);
}

template <class SSConfig, bool TrivialIdxCal = false>
struct AccumulateCarrierKernelFunctor {
  void operator()(sycl::nd_item<2> item) const {
    auto id = cfg.get_item_desc(item);
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
      cfg.oinfo_.data[glb_off] =
          cfg.func_(cfg.oinfo_.data[glb_off], cfg.carrier_[crr_off]);
    }
  }
  AccumulateCarrierKernelFunctor(const SSConfig cfg_) : cfg(cfg_) {}

 private:
  const SSConfig cfg;
};

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
    // to increase work group mumber to increase device coverage.
    return true;
  }

  return false;
}

template <class SSConfig, bool TrivialIdxCal = false>
static inline void accumulate_carrier(const SSConfig& cfg) {
  TORCH_CHECK(
      cfg.carrier_ != nullptr, "scan: nullptr carrier in accumulation ...");
  auto& queue = getCurrentSYCLQueue();

  AccumulateCarrierKernelFunctor<SSConfig, TrivialIdxCal> kfn(cfg);

  sycl_kernel_submit(cfg.global_size(), cfg.group_size(), queue, kfn);
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
  using SSConfig = SegmentScanConfig<
      InputInfo,
      OutputInfo,
      OutputInfo /*not used*/,
      T,
      BinaryFunction>;
  using KernelClass = SegmentScanKernel<SSConfig, TrivialOffCal, TrivialIdxCal>;

  auto cfg = SegmentScanConfig<
      InputInfo,
      OutputInfo,
      OutputInfo /*not used*/,
      T,
      BinaryFunction>::
      template make_config<KernelClass>(
          input_info,
          output_info,
          output_info /*not used*/,
          dim_after_collapse,
          init,
          Type,
          func);
  // 0. recursive convergence
  if (cfg.problem_ <= cfg.problem_wg_range_) {
    cfg.set_carrier(nullptr);
    launch_segment_scan<decltype(cfg), TrivialOffCal, TrivialIdxCal>(cfg);
    return;
  }

  // 1. inclusive scan in each chunk
  Tensor carrier_holder = at::empty(
      {cfg.batch_, cfg.problem_glb_range_ / cfg.problem_wg_range_, cfg.stride_},
      map_options<T>());
  TensorInfo<T, int64_t> carrier_info =
      getTensorInfo<T, int64_t>(carrier_holder);
  cfg.set_carrier(carrier_info.data);
  launch_segment_scan<decltype(cfg), TrivialOffCal, TrivialIdxCal>(cfg);

  // 2. recursion for carrier
  _segment_scan_kernel<EXCLUSIVE_TYPE, TrivialOffCal, TrivialIdxCal>(
      carrier_info, carrier_info, 1, init, func);

  // 3. accumulate among all chunk
  accumulate_carrier<decltype(cfg), TrivialIdxCal>(cfg);

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

  TensorInfo<scalar_t, int64_t> input_info =
      getTensorInfo<scalar_t, int64_t>(input_);
  int dim_after_collapse = input_info.collapseDims(dimension);

  TensorInfo<oscalar_t, int64_t> output_info =
      getTensorInfo<oscalar_t, int64_t>(self);
  output_info.collapseDims(dimension);

  int64_t batch = input_info.outerSize(dim_after_collapse);
  int64_t stride = input_info.innerSize(dim_after_collapse);
  int64_t problem = input_info.sizes[dim_after_collapse];

  if (dispatch_to_loop_scan_kernel(problem, stride, batch)) {
    loop_scan_kernel<Type, true>(
        input_info, output_info, dim_after_collapse, init, func);
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

} // namespace at::native::xpu
