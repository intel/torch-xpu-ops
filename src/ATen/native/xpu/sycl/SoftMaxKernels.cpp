#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/CanUse32BitIndexMath.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/Loops.h>
#include <ATen/xpu/XPUContext.h>
#include <comm/DeviceProperties.h>
#include <comm/SYCLContext.h>

using namespace xpu::sycl;

namespace at {
namespace native {
namespace xpu {

namespace impl {

#define MIN_WG_NUM 32768
#define SIMD32 32
#define SIMD16 16

template <
    int SIMD,
    typename accscalar_t,
    typename reduce_op,
    typename item_t,
    typename local_shared_t>
static inline void softmax_group_reduce(
    item_t item,
    int lid_row,
    int sub_group_num,
    accscalar_t& val,
    accscalar_t init,
    const local_shared_t& local_data,
    reduce_op bin_op) {
  auto sg = item.get_sub_group();

  // dynamic get SIMD width result in big performance drop
  // uint32_t SIMD = sg.get_local_range()[0];
#pragma unroll
  for (int i = 1; i < SIMD; i <<= 1) {
    val = bin_op(val, static_cast<accscalar_t>(sg.shuffle_down(val, i)));
  }
  if (sub_group_num == 1) {
    val = sycl::group_broadcast(sg, val, 0);
    return;
  }
  uint32_t sg_local_id = sg.get_local_linear_id();
  uint32_t sg_id = sg.get_group_linear_id();
  // reduce internal each subgroup, each subgroup will generate one result
  // there are WGroupSize/subGroupSize elements after this step
  int idx = sg_id - (lid_row * sub_group_num);
  if (sg_local_id == 0) {
    local_data[lid_row][idx] = val;
  }
  item.barrier(sycl_local_fence);

  // use one subgroup to reduce WGroupSize/subGroupSize elements
  // into the final result
  if (idx == 0) {
    val = init;
    if (sg_local_id < sub_group_num) {
      val = accscalar_t(local_data[lid_row][sg_local_id]);
    }
    for (int i = sg_local_id + SIMD; i < sub_group_num; i += SIMD) {
      val = bin_op(val, static_cast<accscalar_t>(local_data[lid_row][i]));
    }
#pragma unroll
    for (int i = 1; i < SIMD; i <<= 1) {
      val = bin_op(val, static_cast<accscalar_t>(sg.shuffle_down(val, i)));
      if (i >= ((sub_group_num + 1) >> 1))
        break;
    }

    // the 0th WI (the 0th WI in the 0th sub_group) generate the final result
    if (sg_local_id == 0) {
      local_data[lid_row][0] = val;
    }
  }

  item.barrier(sycl_local_fence);
  val = local_data[lid_row][0];
}

template <
    int vec_size,
    typename accscalar_t,
    typename reduce_op,
    typename item_t,
    typename local_shared_t>
static inline void softmax_group_reduce_spatial(
    item_t item,
    accscalar_t input[vec_size],
    const local_shared_t& local_data,
    int block_row,
    reduce_op bin_op) {
  auto local_row_id = item.get_local_id(1);
  auto local_col_id = item.get_local_id(2);

#pragma unroll(vec_size)
  for (int j = 0; j < vec_size; ++j) {
    local_data[local_row_id][local_col_id][j] = input[j];
  }
  item.barrier(sycl_local_fence);

  int k = 1;
  while (k < block_row) {
    if (local_row_id % (k << 1) == 0 && local_row_id + k < block_row)
#pragma unroll(vec_size)
      for (int j = 0; j < vec_size; ++j) {
        local_data[local_row_id][local_col_id][j] = bin_op(
            local_data[local_row_id][local_col_id][j],
            local_data[local_row_id + k][local_col_id][j]);
      }
    k *= 2;
    item.barrier(sycl_local_fence);
  }
}

template <int SIMD, int vec_size, int NUM>
static inline void get_wgroup_size(
    uint64_t dim_size,
    int outer_size,
    int& sub_group_num,
    int& range,
    int& global_size_row,
    int& local_size_row,
    int& local_size_col) {
  auto dev_id = getDeviceIndexOfCurrentQueue();
  int maxWGSize = syclMaxWorkGroupSize(dev_id);

  int local_size = (dim_size + NUM * vec_size - 1) / (NUM * vec_size);
  local_size = std::min(local_size, maxWGSize);
  // select the local_size_col to cover the dim_size
  sub_group_num = (local_size + SIMD - 1) / SIMD;
  local_size_col = sub_group_num * SIMD;
  // if one workitem [NUM][vec_size] can cover the dim_size number of elements
  // local_size_col will be 1
  if (dim_size <= vec_size * NUM) {
    local_size_col = 1;
    local_size_row = SIMD;
    global_size_row = (outer_size + local_size_row - 1) / local_size_row;
    return;
  }

  // if outer_size is too large and local_size_col is small,
  // then use one workgroup to handle multi rows (dim_size)
  local_size_row = 1;
  global_size_row = outer_size;
  while ((global_size_row >> 1) > MIN_WG_NUM &&
         (local_size_row << 1) * local_size_col <= maxWGSize &&
         !(global_size_row % 2)) {
    global_size_row = global_size_row >> 1;
    local_size_row = local_size_row << 1;
  }

  // compute the reduce range
  range = SIMD;
  while (sub_group_num <= (range >> 1)) {
    range = range >> 1;
  }
}

// this method help to divide the computation resource for spatial_softmax
template <int vec_size>
static inline void get_wgroup_size_spatial(
    int bs,
    int dim_size,
    int inner_size,
    int& GroupSize,
    int& GroupRow) {
  auto dev_id = getDeviceIndexOfCurrentQueue();
  int maxWGSize = syclMaxWorkGroupSize(dev_id);
  int total_resource = syclMaxWorkItemsPerTile(dev_id);

  // set the GroupSize smaller to ensure larger group number
  // smaller GroupSize is friendly to the tail case
  GroupSize = int((inner_size + vec_size - 1) / vec_size);
  GroupSize = std::min(GroupSize, SIMD32);
  auto local_group_num = (inner_size + GroupSize - 1) / GroupSize;

  // enlarge the GroupRow to occupy all the computation resource
  GroupRow = 1;
  while (bs * GroupRow * local_group_num * GroupSize <
         total_resource * vec_size) {
    GroupRow = GroupRow << 1;
    if (GroupRow * SIMD32 == maxWGSize)
      break;
  }
  GroupRow = std::min(GroupRow, int(dim_size));
}

template <
    int INNER_LOOP,
    int vec_size,
    int SIMD,
    typename scalar_t,
    typename accscalar_t,
    typename IndexType,
    bool LogSoftMax,
    int outer_loop,
    bool is_masked,
    typename calc_t,
    typename vec_t>
struct DispatchSoftmaxForwardKernelFunctor
    : public __SYCL_KER_CONFIG_CONVENTION__ {
  [[intel::reqd_sub_group_size(SIMD)]] void operator()(
      sycl::nd_item<1> item) const {
    if (local_size_ == 1 && item.get_global_id(0) >= outer_size_)
      return;

    uint32_t lid_row = 0;
    uint32_t lid_col = item.get_local_id(0);
    uint32_t group_offset = item.get_group(0) * dim_size_;
    if (local_size_row_ != 1) {
      lid_row = item.get_local_id(0) / local_size_;
      lid_col = item.get_local_id(0) % local_size_;
      group_offset =
          (item.get_group(0) * local_size_row_ + lid_row) * dim_size_;
    }
    vec_t reg_in[outer_loop];
    vec_t reg_mask[outer_loop];
    auto lid_offset = lid_col * vec_size;
    auto local_stride = local_size_ * vec_size;

    // load data and get max value
    accscalar_t max_value = std::numeric_limits<accscalar_t>::lowest();
#pragma unroll(outer_loop)
    for (int i = 0; i < outer_loop; ++i) {
      auto index = i * local_stride + lid_offset;
      if (index >= dim_size_)
        break;

      reg_in[i] = *(reinterpret_cast<vec_t*>(in_data_ + group_offset + index));
      if constexpr (is_masked) {
        auto vec_offset = group_offset + index;
#pragma unroll(vec_size)
        for (int j = 0; j < vec_size; ++j) {
          auto linear_idx = vec_offset + j;
          auto mask_offset = input_calc_.get(linear_idx)[1];
          reg_mask[i][j] = mask_data_[mask_offset];
        }
      }
#pragma unroll(vec_size)
      for (int j = 0; j < vec_size; ++j) {
        if constexpr (is_masked) {
          if (reg_mask[i][j]) {
            reg_in[i][j] = neginf_;
          }
        }
        max_value = std::max(max_value, accscalar_t(reg_in[i][j]));
      }
    }
    if (local_size_ > 1) {
      softmax_group_reduce<SIMD, accscalar_t>(
          item,
          lid_row,
          sub_group_num_,
          max_value,
          std::numeric_limits<accscalar_t>::lowest(),
          local_max_,
          [](accscalar_t a, accscalar_t b) { return std::max(a, b); });
    }

    // get sum value
    accscalar_t sum_value = 0;
#pragma unroll(outer_loop)
    for (int i = 0;
         i < outer_loop && ((i * local_stride + lid_offset) < dim_size_);
         ++i) {
#pragma unroll(vec_size)
      for (int j = 0; j < vec_size; ++j) {
        sum_value += ::exp(reg_in[i][j] - max_value);
      }
    }
    if (local_size_ > 1) {
      softmax_group_reduce<SIMD, accscalar_t>(
          item,
          lid_row,
          sub_group_num_,
          sum_value,
          accscalar_t(0),
          local_sum_,
          [](accscalar_t a, accscalar_t b) { return a + b; });
    }
    if constexpr (LogSoftMax)
      sum_value = ::log(sum_value);
    else if (sum_value != 0)
      sum_value = accscalar_t(1) / sum_value;

      // update result
#pragma unroll(outer_loop)
    for (int i = 0; i < outer_loop; ++i) {
      auto index = i * local_stride + lid_offset;
      if (index >= dim_size_)
        break;

#pragma unroll(vec_size)
      for (int j = 0; j < vec_size; ++j) {
        if constexpr (LogSoftMax) {
          reg_in[i][j] =
              static_cast<scalar_t>(reg_in[i][j] - max_value - sum_value);
        } else if (sum_value == 0) {
          reg_in[i][j] = nan_;
        } else {
          reg_in[i][j] = static_cast<scalar_t>(
              ::exp(reg_in[i][j] - max_value) * sum_value);
        }
      }
      *(reinterpret_cast<vec_t*>(out_data_ + group_offset + index)) = reg_in[i];
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    local_max_ = sycl_local_acc_t<accscalar_t, 2>(
        sycl::range<2>{(size_t)local_size_row_, (size_t)sub_group_num_}, cgh);
    local_sum_ = sycl_local_acc_t<accscalar_t, 2>(
        sycl::range<2>{(size_t)local_size_row_, (size_t)sub_group_num_}, cgh);
  }

  DispatchSoftmaxForwardKernelFunctor(
      scalar_t* in_data,
      scalar_t* out_data,
      int dim_size,
      int outer_size,
      bool* mask_data,
      calc_t input_calc,
      int sub_group_num,
      int global_size_row,
      int local_size_row,
      int range,
      int local_size,
      scalar_t neginf,
      scalar_t nan)
      : in_data_(in_data),
        out_data_(out_data),
        dim_size_(dim_size),
        outer_size_(outer_size),
        mask_data_(mask_data),
        input_calc_(input_calc),
        sub_group_num_(sub_group_num),
        global_size_row_(global_size_row),
        local_size_row_(local_size_row),
        range_(range),
        local_size_(local_size),
        neginf_(neginf),
        nan_(nan) {}

 private:
  scalar_t* in_data_;
  scalar_t* out_data_;
  int dim_size_;
  int outer_size_;
  bool* mask_data_;
  calc_t input_calc_;
  int sub_group_num_;
  int global_size_row_;
  int local_size_row_;
  int range_;
  int local_size_;
  scalar_t neginf_;
  scalar_t nan_;
  sycl_local_acc_t<accscalar_t, 2> local_max_;
  sycl_local_acc_t<accscalar_t, 2> local_sum_;
};

// replace std::nullptr_t to avoid kernel name in std namespace
struct DummyFunctor {};

template <
    int INNER_LOOP,
    int vec_size,
    int SIMD,
    typename scalar_t,
    typename accscalar_t,
    typename IndexType,
    bool LogSoftMax,
    int outer_loop,
    bool is_masked = false,
    typename calc_t = decltype(nullptr)>
void dispatch_softmax_forward_kernel(
    scalar_t* in_data,
    scalar_t* out_data,
    int dim_size,
    int outer_size,
    bool* mask_data = nullptr,
    calc_t input_calc = nullptr) {
  using vec_t = at::native::memory::aligned_vector<scalar_t, vec_size>;
  auto& queue = getCurrentSYCLQueue();

  int sub_group_num, global_size_row, local_size_row, range, local_size;
  get_wgroup_size<SIMD, vec_size, outer_loop>(
      dim_size,
      outer_size,
      sub_group_num,
      range,
      global_size_row,
      local_size_row,
      local_size);
  int64_t local_range{local_size_row * local_size};
  int64_t global_range{global_size_row * local_size_row * local_size};
  scalar_t neginf = -std::numeric_limits<scalar_t>::infinity();
  scalar_t nan = std::numeric_limits<accscalar_t>::quiet_NaN();

  if constexpr (is_masked) {
    auto caller = DispatchSoftmaxForwardKernelFunctor<
        INNER_LOOP,
        vec_size,
        SIMD,
        scalar_t,
        accscalar_t,
        IndexType,
        LogSoftMax,
        outer_loop,
        is_masked,
        calc_t,
        vec_t>(
        in_data,
        out_data,
        dim_size,
        outer_size,
        mask_data,
        input_calc,
        sub_group_num,
        global_size_row,
        local_size_row,
        range,
        local_size,
        neginf,
        nan);
    sycl_kernel_submit(global_range, local_range, queue, caller);
  } else {
    DummyFunctor dummy;
    auto caller = DispatchSoftmaxForwardKernelFunctor<
        INNER_LOOP,
        vec_size,
        SIMD,
        scalar_t,
        accscalar_t,
        IndexType,
        LogSoftMax,
        outer_loop,
        is_masked,
        DummyFunctor,
        vec_t>(
        in_data,
        out_data,
        dim_size,
        outer_size,
        mask_data,
        dummy,
        sub_group_num,
        global_size_row,
        local_size_row,
        range,
        local_size,
        neginf,
        nan);
    sycl_kernel_submit(global_range, local_range, queue, caller);
  }
}

template <
    int vec_size,
    typename scalar_t,
    typename accscalar_t,
    typename IndexType,
    bool LogSoftMax,
    typename vec_t,
    int align_bytes>
struct SoftmaxForwardKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    IndexType local_id = item.get_local_id(0);
    auto group_offset = item.get_group(0) * dim_size_;
    int start =
        ((uint64_t)(in_data_ + group_offset)) % align_bytes / sizeof(scalar_t);
    IndexType loops_end = (dim_size_ + start + vec_size - 1) / vec_size;

    // get max value
    auto max_value = std::numeric_limits<accscalar_t>::lowest();
    for (int i = local_id; i < loops_end; i += local_size_) {
      vec_t in_val = *(reinterpret_cast<vec_t*>(
          in_data_ + group_offset - start + i * vec_size));
#pragma unroll(vec_size)
      for (IndexType j = 0; j < vec_size; ++j) {
        IndexType linear_idx = i * vec_size + j - start;
        if (linear_idx >= 0 && linear_idx < dim_size_) {
          scalar_t in_value = in_val[j];
          max_value = std::max(accscalar_t(in_value), max_value);
        }
      }
    }
    max_value = sycl::reduce_over_group(
        item.get_group(), max_value, sycl::maximum<accscalar_t>());

    // get sum value
    auto sum_value = accscalar_t(0);
    for (IndexType i = local_id; i < loops_end; i += local_size_) {
      vec_t in_val = *(reinterpret_cast<vec_t*>(
          in_data_ + group_offset - start + i * vec_size));
#pragma unroll(vec_size)
      for (int j = 0; j < vec_size; ++j) {
        IndexType linear_idx = i * vec_size + j - start;
        if (linear_idx >= 0 && linear_idx < dim_size_)
          sum_value += ::exp(accscalar_t(in_val[j]) - max_value);
      }
    }
    sum_value = sycl::reduce_over_group(
        item.get_group(), sum_value, sycl::plus<accscalar_t>());
    if (LogSoftMax)
      sum_value = ::log(sum_value);
    else
      sum_value = accscalar_t(1) / sum_value;

    // update result
    for (IndexType i = local_id; i < loops_end; i += local_size_) {
      auto remaining = dim_size_ + start - i * vec_size;
      if ((start > 0 && i == 0) || (remaining < vec_size)) {
#pragma unroll(vec_size)
        for (int j = 0; j < vec_size; ++j) {
          IndexType linear_idx = i * vec_size + j - start;
          if (linear_idx >= 0 && linear_idx < dim_size_) {
            if (LogSoftMax)
              out_data_[group_offset + linear_idx] = static_cast<scalar_t>(
                  in_data_[group_offset + linear_idx] - max_value - sum_value);
            else
              out_data_[group_offset + linear_idx] = static_cast<scalar_t>(
                  ::exp(in_data_[group_offset + linear_idx] - max_value) *
                  sum_value);
          }
        }
      } else {
        vec_t in_val = *(reinterpret_cast<vec_t*>(
            in_data_ + group_offset - start + i * vec_size));
#pragma unroll(vec_size)
        for (int j = 0; j < vec_size; ++j) {
          if (LogSoftMax)
            in_val[j] =
                static_cast<scalar_t>(in_val[j] - max_value - sum_value);
          else
            in_val[j] =
                static_cast<scalar_t>(::exp(in_val[j] - max_value) * sum_value);
        }
        *(reinterpret_cast<vec_t*>(
            out_data_ + group_offset - start + i * vec_size)) = in_val;
      }
    }
  }
  SoftmaxForwardKernelFunctor(
      scalar_t* in_data,
      scalar_t* out_data,
      int dim_size,
      int outer_size,
      int local_size)
      : in_data_(in_data),
        out_data_(out_data),
        dim_size_(dim_size),
        outer_size_(outer_size),
        local_size_(local_size) {}

 private:
  scalar_t* in_data_;
  scalar_t* out_data_;
  int dim_size_;
  int outer_size_;
  int local_size_;
};

template <
    int vec_size,
    typename scalar_t,
    typename accscalar_t,
    typename IndexType,
    bool LogSoftMax>
void softmax_forward_kernel(
    scalar_t* in_data,
    scalar_t* out_data,
    int dim_size,
    int outer_size) {
  using vec_t = at::native::memory::aligned_vector<scalar_t, vec_size>;
  constexpr int align_bytes = alignof(vec_t);
  auto& queue = getCurrentSYCLQueue();
  auto dev_id = getDeviceIndexOfCurrentQueue();
  int local_size = std::min(
      (dim_size + vec_size - 1) / vec_size, int(syclMaxWorkGroupSize(dev_id)));

  int64_t local_range{local_size};
  int64_t global_range{local_size * outer_size};
  auto ker = SoftmaxForwardKernelFunctor<
      vec_size,
      scalar_t,
      accscalar_t,
      IndexType,
      LogSoftMax,
      vec_t,
      align_bytes>(in_data, out_data, dim_size, outer_size, local_size);
  sycl_kernel_submit(global_range, local_range, queue, ker);
}

template <
    int vec_size,
    typename scalar_t,
    typename accscalar_t,
    typename IndexType,
    bool LogSoftMax,
    typename vec_t>
struct SpatialSoftmaxForwardKernelFunctor
    : public __SYCL_KER_CONFIG_CONVENTION__ {
  void operator()(sycl::nd_item<3> item) const {
    IndexType global_col = item.get_global_id(2);
    IndexType local_row_id = item.get_local_id(1);
    IndexType local_col_id = item.get_local_id(2);

    auto group_offset = item.get_global_id(0) * dim_size_ * inner_size_;

    // get max value
    accscalar_t max_value[vec_size];
    auto offset = local_row_id * inner_size_ + global_col * vec_size;
    vec_t value = *(reinterpret_cast<vec_t*>(in_data_ + group_offset + offset));
#pragma unroll(vec_size)
    for (int j = 0; j < vec_size; ++j) {
      max_value[j] = accscalar_t(value[j]);
    }
    for (int i = local_row_id + block_row_; i < dim_size_; i += block_row_) {
      offset = i * inner_size_ + global_col * vec_size;
      value = *(reinterpret_cast<vec_t*>(in_data_ + group_offset + offset));
#pragma unroll(vec_size)
      for (int j = 0; j < vec_size; ++j) {
        max_value[j] = std::max(max_value[j], accscalar_t(value[j]));
      }
    }
    if (block_row_ > 1) {
      softmax_group_reduce_spatial<vec_size, accscalar_t>(
          item,
          max_value,
          local_data_,
          block_row_,
          [](accscalar_t a, accscalar_t b) { return std::max(a, b); });
#pragma unroll(vec_size)
      for (int j = 0; j < vec_size; ++j) {
        max_value[j] = local_data_[0][local_col_id][j];
      }
      item.barrier(sycl_local_fence);
    }

    // get sum value
    accscalar_t sum_value[vec_size];
    offset = local_row_id * inner_size_ + global_col * vec_size;
    value = *(reinterpret_cast<vec_t*>(in_data_ + group_offset + offset));
#pragma unroll(vec_size)
    for (int j = 0; j < vec_size; ++j) {
      sum_value[j] = ::exp(value[j] - max_value[j]);
    }
    for (int i = local_row_id + block_row_; i < dim_size_; i += block_row_) {
      offset = i * inner_size_ + global_col * vec_size;
      value = *(reinterpret_cast<vec_t*>(in_data_ + group_offset + offset));
#pragma unroll(vec_size)
      for (int j = 0; j < vec_size; ++j) {
        sum_value[j] += ::exp(value[j] - max_value[j]);
      }
    }
    if (block_row_ > 1) {
      softmax_group_reduce_spatial<vec_size, accscalar_t>(
          item,
          sum_value,
          local_data_,
          block_row_,
          [](accscalar_t a, accscalar_t b) { return a + b; });
#pragma unroll(vec_size)
      for (int j = 0; j < vec_size; ++j) {
        if (LogSoftMax)
          sum_value[j] = ::log(local_data_[0][local_col_id][j]);
        else
          sum_value[j] = accscalar_t(1) / local_data_[0][local_col_id][j];
      }
    } else {
#pragma unroll(vec_size)
      for (int j = 0; j < vec_size; ++j) {
        if (LogSoftMax)
          sum_value[j] = ::log(sum_value[j]);
        else
          sum_value[j] = accscalar_t(1) / sum_value[j];
      }
    }

    // update result
    if (global_col * vec_size < inner_size_) {
      for (int i = local_row_id; i < dim_size_; i += block_row_) {
        auto offset = i * inner_size_ + global_col * vec_size;
        vec_t in_val =
            *(reinterpret_cast<vec_t*>(in_data_ + group_offset + offset));
#pragma unroll(vec_size)
        for (int j = 0; j < vec_size; ++j) {
          if (LogSoftMax)
            in_val[j] =
                static_cast<scalar_t>(in_val[j] - max_value[j] - sum_value[j]);
          else
            in_val[j] = static_cast<scalar_t>(
                ::exp(in_val[j] - max_value[j]) * sum_value[j]);
        }
        *(reinterpret_cast<vec_t*>(out_data_ + group_offset + offset)) = in_val;
      }
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    local_data_ = sycl_local_acc_t<accscalar_t, 3>(
        sycl::range<3>{
            (size_t)block_row_, (size_t)local_size_, (size_t)vec_size},
        cgh);
  }

  SpatialSoftmaxForwardKernelFunctor(
      scalar_t* in_data,
      scalar_t* out_data,
      int dim_size,
      int inner_size,
      int outer_size,
      int local_size,
      int block_row,
      int group_num)
      : in_data_(in_data),
        out_data_(out_data),
        dim_size_(dim_size),
        inner_size_(inner_size),
        outer_size_(outer_size),
        local_size_(local_size),
        block_row_(block_row),
        group_num_(group_num) {}

 private:
  scalar_t* in_data_;
  scalar_t* out_data_;
  int dim_size_;
  int inner_size_;
  int outer_size_;
  int local_size_;
  int block_row_;
  int group_num_;
  sycl_local_acc_t<accscalar_t, 3> local_data_;
};

template <
    int vec_size,
    typename scalar_t,
    typename accscalar_t,
    typename IndexType,
    bool LogSoftMax>
void spatial_softmax_forward(
    scalar_t* in_data,
    scalar_t* out_data,
    int dim_size,
    int inner_size,
    int outer_size) {
  using vec_t = at::native::memory::aligned_vector<scalar_t, vec_size>;
  auto& queue = getCurrentSYCLQueue();

  int local_size, block_row;
  get_wgroup_size_spatial<vec_size>(
      outer_size, dim_size, inner_size, local_size, block_row);
  int group_num =
      (inner_size + local_size * vec_size - 1) / (local_size * vec_size);
  sycl::range<3> global_range{
      (size_t)outer_size, (size_t)block_row, (size_t)(group_num * local_size)};
  sycl::range<3> local_range{(size_t)1, (size_t)block_row, (size_t)local_size};

  auto caller = SpatialSoftmaxForwardKernelFunctor<
      vec_size,
      scalar_t,
      accscalar_t,
      IndexType,
      LogSoftMax,
      vec_t>(
      in_data,
      out_data,
      dim_size,
      inner_size,
      outer_size,
      local_size,
      block_row,
      group_num);
  sycl_kernel_submit(global_range, local_range, queue, caller);
}

template <
    int INNER_LOOP,
    int vec_size,
    int SIMD,
    typename scalar_t,
    typename accscalar_t,
    typename IndexType,
    bool LogSoftMax,
    bool is_masked,
    typename calc_t,
    typename vec_t,
    int NUM>
struct DispatchSoftmaxBackwardKernelFunctor
    : public __SYCL_KER_CONFIG_CONVENTION__ {
  [[intel::reqd_sub_group_size(SIMD)]] void operator()(
      sycl::nd_item<1> item) const {
    if (local_size_ == 1 && item.get_global_id(0) >= outer_size_)
      return;

    uint32_t lid_row = item.get_local_id(0) / local_size_;
    uint32_t lid_col = item.get_local_id(0) % local_size_;
    uint32_t group_offset =
        (item.get_group(0) * local_size_row_ + lid_row) * dim_size_;

    // load data and get max value
    accscalar_t sum_value = accscalar_t(0);
    vec_t reg_out[NUM];
    vec_t reg_gradout[NUM];
#pragma unroll(NUM)
    for (int i = 0; i < NUM; ++i) {
      auto index = (lid_col + i * local_size_) * vec_size;
      if (index >= dim_size_)
        break;

      reg_out[i] = *(reinterpret_cast<vec_t*>(output_ + group_offset + index));
      reg_gradout[i] =
          *(reinterpret_cast<vec_t*>(gradOutput_ + group_offset + index));
      if constexpr (is_masked) {
        auto vec_offset = group_offset + index;
#pragma unroll(vec_size)
        for (int j = 0; j < vec_size; ++j) {
          auto linear_idx = vec_offset + j;
          auto mask_offset = input_calc_.get(linear_idx)[1];
          if (mask_data_[mask_offset]) {
            reg_out[i][j] = scalar_t(0);
          }
        }
      }

#pragma unroll(vec_size)
      for (int j = 0; j < vec_size; ++j) {
        if (LogSoftMax) {
          sum_value += reg_gradout[i][j];
        } else {
          sum_value += reg_out[i][j] * reg_gradout[i][j];
        }
      }
    }
    if (local_size_ > 1) {
      softmax_group_reduce<SIMD, accscalar_t>(
          item,
          lid_row,
          sub_group_num_,
          sum_value,
          accscalar_t(0),
          local_sum_,
          [](accscalar_t a, accscalar_t b) { return a + b; });
    }
    // update result
#pragma unroll(NUM)
    for (int i = 0; i < NUM; ++i) {
      auto index = (lid_col + i * local_size_) * vec_size;
      if (index >= dim_size_)
        break;

#pragma unroll(vec_size)
      for (int j = 0; j < vec_size; ++j) {
        if (LogSoftMax) {
          reg_out[i][j] = static_cast<scalar_t>(
              reg_gradout[i][j] - ::exp(reg_out[i][j]) * sum_value);
        } else {
          reg_out[i][j] = static_cast<scalar_t>(
              reg_out[i][j] * (reg_gradout[i][j] - sum_value));
        }
      }
      *(reinterpret_cast<vec_t*>(gradInput_ + group_offset + index)) =
          reg_out[i];
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    local_sum_ = sycl_local_acc_t<accscalar_t, 2>(
        sycl::range<2>{(size_t)local_size_row_, (size_t)sub_group_num_}, cgh);
  }

  DispatchSoftmaxBackwardKernelFunctor(
      scalar_t* gradInput,
      scalar_t* output,
      scalar_t* gradOutput,
      int dim_size,
      int outer_size,
      bool* mask_data,
      calc_t input_calc,
      int sub_group_num,
      int global_size_row,
      int local_size_row,
      int range,
      int local_size)
      : gradInput_(gradInput),
        output_(output),
        gradOutput_(gradOutput),
        dim_size_(dim_size),
        outer_size_(outer_size),
        mask_data_(mask_data),
        input_calc_(input_calc),
        sub_group_num_(sub_group_num),
        global_size_row_(global_size_row),
        local_size_row_(local_size_row),
        range_(range),
        local_size_(local_size) {}

 private:
  scalar_t* gradInput_;
  scalar_t* output_;
  scalar_t* gradOutput_;
  int dim_size_;
  int outer_size_;
  bool* mask_data_;
  calc_t input_calc_;
  int sub_group_num_;
  int global_size_row_;
  int local_size_row_;
  int range_;
  int local_size_;
  sycl_local_acc_t<accscalar_t, 2> local_sum_;
};

template <
    int INNER_LOOP,
    int vec_size,
    int SIMD,
    typename scalar_t,
    typename accscalar_t,
    typename IndexType,
    bool LogSoftMax,
    bool is_masked = false,
    typename calc_t = decltype(nullptr)>
void dispatch_softmax_backward_kernel(
    scalar_t* gradInput,
    scalar_t* output,
    scalar_t* gradOutput,
    int dim_size,
    int outer_size,
    bool* mask_data = nullptr,
    calc_t input_calc = nullptr) {
  using vec_t = at::native::memory::aligned_vector<scalar_t, vec_size>;
  auto& queue = getCurrentSYCLQueue();
  constexpr int NUM = INNER_LOOP / vec_size * (SIMD32 / SIMD);
  int sub_group_num, global_size_row, local_size_row, range, local_size;
  get_wgroup_size<SIMD, vec_size, NUM>(
      dim_size,
      outer_size,
      sub_group_num,
      range,
      global_size_row,
      local_size_row,
      local_size);
  int64_t local_range{local_size_row * local_size};
  int64_t global_range{global_size_row * local_size_row * local_size};

  if constexpr (is_masked) {
    auto caller = DispatchSoftmaxBackwardKernelFunctor<
        INNER_LOOP,
        vec_size,
        SIMD,
        scalar_t,
        accscalar_t,
        IndexType,
        LogSoftMax,
        is_masked,
        calc_t,
        vec_t,
        NUM>(
        gradInput,
        output,
        gradOutput,
        dim_size,
        outer_size,
        mask_data,
        input_calc,
        sub_group_num,
        global_size_row,
        local_size_row,
        range,
        local_size);
    sycl_kernel_submit(global_range, local_range, queue, caller);
  } else {
    DummyFunctor dummy;
    auto caller = DispatchSoftmaxBackwardKernelFunctor<
        INNER_LOOP,
        vec_size,
        SIMD,
        scalar_t,
        accscalar_t,
        IndexType,
        LogSoftMax,
        is_masked,
        DummyFunctor,
        vec_t,
        NUM>(
        gradInput,
        output,
        gradOutput,
        dim_size,
        outer_size,
        mask_data,
        dummy,
        sub_group_num,
        global_size_row,
        local_size_row,
        range,
        local_size);
    sycl_kernel_submit(global_range, local_range, queue, caller);
  }
}

template <
    int vec_size,
    typename scalar_t,
    typename accscalar_t,
    bool LogSoftMax,
    typename vec_t,
    int align_bytes>
struct SoftmaxBackwardKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    int local_id = item.get_local_id(0);
    auto group_offset = item.get_group(0) * dim_size_;
    int start =
        ((uint64_t)(output_ + group_offset)) % align_bytes / sizeof(scalar_t);
    int loops_end = (dim_size_ + start + vec_size - 1) / vec_size;

    vec_t* vec_gradin_data_ptr =
        reinterpret_cast<vec_t*>(gradInput_ + group_offset - start);
    const vec_t* vec_out_data_ptr =
        reinterpret_cast<const vec_t*>(output_ + group_offset - start);
    const vec_t* vec_gradout_data_ptr =
        reinterpret_cast<const vec_t*>(gradOutput_ + group_offset - start);

    // get sum value
    auto sum_value = accscalar_t(0);
    for (int i = local_id; i < loops_end; i += local_size_) {
      auto gradout_val = vec_gradout_data_ptr[i];
      if (LogSoftMax) {
#pragma unroll(vec_size)
        for (int j = 0; j < vec_size; ++j) {
          int64_t linear_idx = i * vec_size + j - start;
          if (linear_idx >= 0 && linear_idx < dim_size_) {
            sum_value += gradout_val[j];
          }
        }
      } else {
        vec_t out_val = vec_out_data_ptr[i];
#pragma unroll(vec_size)
        for (int j = 0; j < vec_size; ++j) {
          int64_t linear_idx = i * vec_size + j - start;
          if (linear_idx >= 0 && linear_idx < dim_size_) {
            sum_value += out_val[j] * gradout_val[j];
          }
        }
      }
    }
    sum_value = sycl::reduce_over_group(
        item.get_group(), sum_value, sycl::plus<accscalar_t>());

    // update result
    for (int i = local_id; i < loops_end; i += local_size_) {
      // handle the head and tail
      auto remaining = dim_size_ + start - i * vec_size;
      if ((start > 0 && i == 0) || (remaining < vec_size)) {
#pragma unroll(vec_size)
        for (int j = 0; j < vec_size; ++j) {
          auto linear_idx = i * vec_size + j - start;
          if (linear_idx >= 0 && linear_idx < dim_size_) {
            auto offset = group_offset + linear_idx;
            if (LogSoftMax) {
              gradInput_[offset] =
                  gradOutput_[offset] - ::exp(output_[offset]) * sum_value;
            } else {
              gradInput_[offset] =
                  output_[offset] * (gradOutput_[offset] - sum_value);
            }
          }
        }
      } else {
        vec_t grad_val = vec_gradout_data_ptr[i];
        vec_t out_val = vec_out_data_ptr[i];
#pragma unroll(vec_size)
        for (int j = 0; j < vec_size; ++j) {
          if (LogSoftMax) {
            out_val[j] = grad_val[j] - ::exp(out_val[j]) * sum_value;
          } else {
            out_val[j] = out_val[j] * (grad_val[j] - sum_value);
          }
        }
        vec_gradin_data_ptr[i] = out_val;
      }
    }
  }
  SoftmaxBackwardKernelFunctor(
      scalar_t* gradInput,
      const scalar_t* output,
      const scalar_t* gradOutput,
      int dim_size,
      int outer_size,
      int local_size)
      : gradInput_(gradInput),
        output_(output),
        gradOutput_(gradOutput),
        dim_size_(dim_size),
        outer_size_(outer_size),
        local_size_(local_size) {}

 private:
  scalar_t* gradInput_;
  const scalar_t* output_;
  const scalar_t* gradOutput_;
  int dim_size_;
  int outer_size_;
  int local_size_;
};

template <
    int vec_size,
    typename scalar_t,
    typename accscalar_t,
    bool LogSoftMax>
void softmax_backward_kernel(
    scalar_t* gradInput,
    const scalar_t* output,
    const scalar_t* gradOutput,
    int dim_size,
    int outer_size) {
  using vec_t = at::native::memory::aligned_vector<scalar_t, vec_size>;
  constexpr int align_bytes = alignof(vec_t);
  auto& queue = getCurrentSYCLQueue();

  auto dev_id = getDeviceIndexOfCurrentQueue();
  int local_size = std::min(
      (dim_size + vec_size - 1) / vec_size, int(syclMaxWorkGroupSize(dev_id)));
  int64_t local_range{local_size};
  int64_t global_range{local_size * outer_size};

  auto caller = SoftmaxBackwardKernelFunctor<
      vec_size,
      scalar_t,
      accscalar_t,
      LogSoftMax,
      vec_t,
      align_bytes>(
      gradInput, output, gradOutput, dim_size, outer_size, local_size);

  sycl_kernel_submit(global_range, local_range, queue, caller);
}

template <
    int vec_size,
    typename scalar_t,
    typename accscalar_t,
    bool LogSoftMax,
    typename vec_t>
struct SpatialSoftmaxBackwardKernelFunctor
    : public __SYCL_KER_CONFIG_CONVENTION__ {
  void operator()(sycl::nd_item<3> item) const {
    auto global_col = item.get_global_id(2);
    auto local_row_id = item.get_local_id(1);
    auto local_col_id = item.get_local_id(2);

    auto group_offset = item.get_global_id(0) * dim_size_ * inner_size_;
    auto gradin_ptr = gradInput_ + group_offset;
    auto out_ptr = output_ + group_offset;
    auto gradout_ptr = gradOutput_ + group_offset;

    // get sum value
    accscalar_t sum_value[vec_size];
#pragma unroll(vec_size)
    for (int j = 0; j < vec_size; ++j)
      sum_value[j] = accscalar_t(0);

    for (int i = local_row_id; i < dim_size_; i += block_row_) {
      auto offset = i * inner_size_ + global_col * vec_size;
      vec_t gradout_val =
          *(reinterpret_cast<const vec_t*>(gradout_ptr + offset));
      if (LogSoftMax) {
#pragma unroll(vec_size)
        for (int j = 0; j < vec_size; ++j)
          sum_value[j] += gradout_val[j];
      } else {
        vec_t out_val = *(reinterpret_cast<const vec_t*>(out_ptr + offset));
#pragma unroll(vec_size)
        for (int j = 0; j < vec_size; ++j)
          sum_value[j] += accscalar_t(gradout_val[j]) * out_val[j];
      }
    }
    if (block_row_ > 1) {
      softmax_group_reduce_spatial<vec_size, accscalar_t>(
          item,
          sum_value,
          local_data_,
          block_row_,
          [](accscalar_t a, accscalar_t b) { return a + b; });
#pragma unroll(vec_size)
      for (int j = 0; j < vec_size; ++j) {
        sum_value[j] = local_data_[0][local_col_id][j];
      }
    }

    // update result
    if (global_col * vec_size < inner_size_) {
      for (int i = local_row_id; i < dim_size_; i += block_row_) {
        auto offset = i * inner_size_ + global_col * vec_size;
        vec_t out_val = *(reinterpret_cast<const vec_t*>(out_ptr + offset));
        vec_t gradout_val =
            *(reinterpret_cast<const vec_t*>(gradout_ptr + offset));
#pragma unroll(vec_size)
        for (int j = 0; j < vec_size; ++j) {
          if (LogSoftMax) {
            out_val[j] = static_cast<scalar_t>(
                gradout_val[j] - ::exp(out_val[j]) * sum_value[j]);
          } else {
            out_val[j] = static_cast<scalar_t>(
                out_val[j] * (gradout_val[j] - sum_value[j]));
          }
        }
        *(reinterpret_cast<vec_t*>(gradin_ptr + offset)) = out_val;
      }
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    local_data_ = sycl_local_acc_t<accscalar_t, 3>(
        sycl::range<3>{
            (size_t)block_row_, (size_t)local_size_, (size_t)vec_size},
        cgh);
  }

  SpatialSoftmaxBackwardKernelFunctor(
      scalar_t* gradInput,
      const scalar_t* output,
      const scalar_t* gradOutput,
      int dim_size,
      int inner_size,
      int outer_size,
      int local_size,
      int block_row)
      : gradInput_(gradInput),
        output_(output),
        gradOutput_(gradOutput),
        dim_size_(dim_size),
        inner_size_(inner_size),
        outer_size_(outer_size),
        local_size_(local_size),
        block_row_(block_row) {}

 private:
  scalar_t* gradInput_;
  const scalar_t* output_;
  const scalar_t* gradOutput_;
  int dim_size_;
  int inner_size_;
  int outer_size_;
  int local_size_;
  int block_row_;
  sycl_local_acc_t<accscalar_t, 3> local_data_;
};

template <
    int vec_size,
    typename scalar_t,
    typename accscalar_t,
    bool LogSoftMax>
void spatial_softmax_backward_kernel(
    scalar_t* gradInput,
    const scalar_t* output,
    const scalar_t* gradOutput,
    int dim_size,
    int inner_size,
    int outer_size) {
  using vec_t = at::native::memory::aligned_vector<scalar_t, vec_size>;
  auto& queue = getCurrentSYCLQueue();

  int local_size, block_row;
  get_wgroup_size_spatial<vec_size>(
      outer_size, dim_size, inner_size, local_size, block_row);
  int group_num =
      (inner_size + local_size * vec_size - 1) / (local_size * vec_size);
  sycl::range<3> global_range{
      (size_t)outer_size, (size_t)block_row, (size_t)(group_num * local_size)};
  sycl::range<3> local_range{(size_t)1, (size_t)block_row, (size_t)local_size};

  auto caller = SpatialSoftmaxBackwardKernelFunctor<
      vec_size,
      scalar_t,
      accscalar_t,
      LogSoftMax,
      vec_t>(
      gradInput,
      output,
      gradOutput,
      dim_size,
      inner_size,
      outer_size,
      local_size,
      block_row);
  sycl_kernel_submit(global_range, local_range, queue, caller);
}

template <typename scalar_t, typename accscalar_t, bool LogSoftMax>
void spatial_softmax_forward(Tensor& output, Tensor& input, int dim) {
  auto inner_size = input.stride(dim);
  auto dim_size = input.size(dim);
  auto outer_size = input.numel() / (inner_size * dim_size);

  constexpr int float4_size = sizeof(float) * 4;
  constexpr int max_vec_size = float4_size / sizeof(scalar_t);
  constexpr int INNER_LOOP = max_vec_size * 2;

  // decide vec_size: max_vec_size or 1
  using vec_t = at::native::memory::aligned_vector<scalar_t, max_vec_size>;
  constexpr int align_bytes = alignof(vec_t);
  int input_start =
      ((uint64_t)input.data_ptr()) % align_bytes / sizeof(scalar_t);
  int output_start =
      ((uint64_t)output.data_ptr()) % align_bytes / sizeof(scalar_t);

  // decide indexing range: uint32_t (4GB) or uint64_t (>4GB)
  bool can_use_32bit_index =
      canUse32BitIndexMath(input) && canUse32BitIndexMath(output);

  // decide SIMD: SIMD32 or SIMD16
  auto* dev_prop =
      at::xpu::getDeviceProperties(at::xpu::getDeviceIndexOfCurrentQueue());
  auto sub_group_size = dev_prop->sub_group_sizes;
  int SIMD = sub_group_size[1];
  if (SIMD == SIMD32) {
    if (dim_size < SIMD16 * INNER_LOOP)
      SIMD = SIMD16;
  }

#define DISPATCH_SOFTMAX_FORWARD_IMPL(vec_size, SIMD, outer_loop) \
  {                                                               \
    dispatch_softmax_forward_kernel<                              \
        INNER_LOOP,                                               \
        vec_size,                                                 \
        SIMD,                                                     \
        scalar_t,                                                 \
        accscalar_t,                                              \
        uint32_t,                                                 \
        LogSoftMax,                                               \
        outer_loop>(                                              \
        input.data_ptr<scalar_t>(),                               \
        output.data_ptr<scalar_t>(),                              \
        dim_size,                                                 \
        outer_size);                                              \
  }

#define SOFTMAX_FORWARD_IMPL(vec_size, IndexType) \
  {                                               \
    softmax_forward_kernel<                       \
        vec_size,                                 \
        scalar_t,                                 \
        accscalar_t,                              \
        IndexType,                                \
        LogSoftMax>(                              \
        input.data_ptr<scalar_t>(),               \
        output.data_ptr<scalar_t>(),              \
        dim_size,                                 \
        outer_size);                              \
  }

#define SPATIAL_SOFTMAX_FORWARD_IMPL(vec_size, IndexType) \
  {                                                       \
    spatial_softmax_forward<                              \
        vec_size,                                         \
        scalar_t,                                         \
        accscalar_t,                                      \
        IndexType,                                        \
        LogSoftMax>(                                      \
        input.data_ptr<scalar_t>(),                       \
        output.data_ptr<scalar_t>(),                      \
        dim_size,                                         \
        inner_size,                                       \
        outer_size);                                      \
  }

  if (inner_size == 1) {
    // if the element number is smaller than max_work_group_size * INNER_LOOP,
    // the fast path (dispatch_softmax_forward) will be selected.
    // otherwise, the general path (softmax_forward_kernel) will be selected.
    auto dev_id = getDeviceIndexOfCurrentQueue();
    int max_group_size = syclMaxWorkGroupSize(dev_id);

    if (can_use_32bit_index && max_group_size * INNER_LOOP >= dim_size) {
      // it assumes vec_size * outer_loop * work_group_size >= dim_size

      if (SIMD == SIMD32) {
        // Ensure input/output tensor are aligned with max_vec_size
        if (input_start == 0 && output_start == 0 &&
            dim_size % max_vec_size == 0) {
          constexpr int outer_loop = INNER_LOOP / max_vec_size;
          DISPATCH_SOFTMAX_FORWARD_IMPL(
              /*vec_size*/ max_vec_size, /*SIMD*/ SIMD32, outer_loop);
        } else {
          constexpr int outer_loop = INNER_LOOP;
          DISPATCH_SOFTMAX_FORWARD_IMPL(
              /*vec_size*/ 1, /*SIMD*/ SIMD32, outer_loop);
        }
      } else {
        if (input_start == 0 && output_start == 0 &&
            dim_size % max_vec_size == 0) {
          if (max_vec_size >= 4 && dim_size <= 4 * SIMD) {
            // if vec_size >= 4 and dim_size <= 4 * SIMD, take smaller vec_size
            // and 1 outer_loop
            constexpr int outer_loop = 1;
            DISPATCH_SOFTMAX_FORWARD_IMPL(
                /*vec_size*/ 4, /*SIMD*/ SIMD16, outer_loop);
          } else if (dim_size <= max_vec_size * SIMD) {
            // if dim_size <= max_vec_size * SIMD , take 1 outer_loop
            constexpr int outer_loop = 1;
            DISPATCH_SOFTMAX_FORWARD_IMPL(
                /*vec_size*/ max_vec_size, /*SIMD*/ SIMD16, outer_loop);
          } else {
            // SIMD16 will use less register numbers than SIMD32
            // if the SIMD = SIMD16, then outer_loop will be enlarged 2x
            constexpr int outer_loop = INNER_LOOP / max_vec_size * 2;
            DISPATCH_SOFTMAX_FORWARD_IMPL(
                /*vec_size*/ max_vec_size, /*SIMD*/ SIMD16, outer_loop);
          }
        } else {
          constexpr int outer_loop = INNER_LOOP * 2;
          DISPATCH_SOFTMAX_FORWARD_IMPL(
              /*vec_size*/ 1, /*SIMD*/ SIMD16, outer_loop);
        }
      }
    } else {
      if (can_use_32bit_index) {
        // the start psition of tensor pointer should be the same
        // the kernel can handle the non-aligned status
        if (input_start == output_start) {
          SOFTMAX_FORWARD_IMPL(
              /*vec_size*/ max_vec_size, /*IndexType*/ uint32_t);
        } else {
          SOFTMAX_FORWARD_IMPL(/*vec_size*/ 1, /*IndexType*/ uint32_t);
        }
      } else {
        if (input_start == output_start) {
          SOFTMAX_FORWARD_IMPL(
              /*vec_size*/ max_vec_size, /*IndexType*/ uint64_t);
        } else {
          SOFTMAX_FORWARD_IMPL(/*vec_size*/ 1, /*IndexType*/ uint64_t);
        }
      }
    }
  } else {
    if (can_use_32bit_index) {
      if (input_start == output_start && inner_size % max_vec_size == 0) {
        SPATIAL_SOFTMAX_FORWARD_IMPL(
            /*vec_size*/ max_vec_size, /*IndexType*/ uint32_t);
      } else {
        SPATIAL_SOFTMAX_FORWARD_IMPL(/*vec_size*/ 1, /*IndexType*/ uint32_t);
      }
    } else {
      if (input_start == output_start && inner_size % max_vec_size == 0) {
        SPATIAL_SOFTMAX_FORWARD_IMPL(
            /*vec_size*/ max_vec_size, /*IndexType*/ uint64_t);
      } else {
        SPATIAL_SOFTMAX_FORWARD_IMPL(/*vec_size*/ 1, /*IndexType*/ uint64_t);
      }
    }
  }
#undef DISPATCH_SOFTMAX_FORWARD_IMPL
#undef SOFTMAX_FORWARD_IMPL
#undef SPATIAL_SOFTMAX_FORWARD_IMPL
}

template <typename scalar_t, typename accscalar_t, bool LogSoftMax>
void spatial_softmax_backward(
    Tensor& gradInput,
    Tensor& output,
    Tensor& gradOutput,
    int dim) {
  auto inner_size = output.stride(dim);
  auto dim_size = output.size(dim);
  auto outer_size = output.numel() / (dim_size * inner_size);

  constexpr int float4_size = sizeof(float) * 4;
  constexpr int max_vec_size = float4_size / sizeof(scalar_t);
  constexpr int INNER_LOOP = max_vec_size;

  // decide vec_size: max_vec_size or 1
  using vec_t = at::native::memory::aligned_vector<scalar_t, max_vec_size>;
  constexpr int align_bytes = alignof(vec_t);
  int gradin_start =
      ((uint64_t)gradInput.data_ptr()) % align_bytes / sizeof(scalar_t);
  int output_start =
      ((uint64_t)output.data_ptr()) % align_bytes / sizeof(scalar_t);
  int gradoutput_start =
      ((uint64_t)gradOutput.data_ptr()) % align_bytes / sizeof(scalar_t);

  // decide indexing range: uint32_t (4GB) or uint64_t (>4GB)
  bool can_use_32bit_index = canUse32BitIndexMath(gradInput) &&
      canUse32BitIndexMath(output) && canUse32BitIndexMath(gradOutput);

  // decide SIMD: SIMD32 or SIMD16
  auto* dev_prop =
      at::xpu::getDeviceProperties(at::xpu::getDeviceIndexOfCurrentQueue());
  auto sub_group_size = dev_prop->sub_group_sizes;
  int SIMD = sub_group_size[1];
  if (SIMD == SIMD32) {
    if (dim_size < SIMD16 * max_vec_size)
      SIMD = SIMD16;
  }

#define DISPATCH_SOFTMAX_BACKWARD_IMPL(vec_size, SIMD) \
  {                                                    \
    dispatch_softmax_backward_kernel<                  \
        INNER_LOOP,                                    \
        vec_size,                                      \
        SIMD,                                          \
        scalar_t,                                      \
        accscalar_t,                                   \
        uint32_t,                                      \
        LogSoftMax>(                                   \
        gradInput.data_ptr<scalar_t>(),                \
        output.data_ptr<scalar_t>(),                   \
        gradOutput.data_ptr<scalar_t>(),               \
        dim_size,                                      \
        outer_size);                                   \
  }

#define SOFTMAX_BACKWARD_IMPL(vec_size, IndexType)                      \
  softmax_backward_kernel<vec_size, scalar_t, accscalar_t, LogSoftMax>( \
      gradInput.data_ptr<scalar_t>(),                                   \
      output.data_ptr<scalar_t>(),                                      \
      gradOutput.data_ptr<scalar_t>(),                                  \
      dim_size,                                                         \
      outer_size);

#define SPATIAL_SOFTMAX_BACKWARD_IMPL(vec_size, IndexType) \
  spatial_softmax_backward_kernel<                         \
      vec_size,                                            \
      scalar_t,                                            \
      accscalar_t,                                         \
      LogSoftMax>(                                         \
      gradInput.data_ptr<scalar_t>(),                      \
      output.data_ptr<scalar_t>(),                         \
      gradOutput.data_ptr<scalar_t>(),                     \
      dim_size,                                            \
      inner_size,                                          \
      outer_size);

  if (inner_size == 1) {
    auto dev_id = getDeviceIndexOfCurrentQueue();
    int max_group_size = syclMaxWorkGroupSize(dev_id);
    // if the element number is smaller than max_work_group_size * INNER_LOOP
    // / 2, (2 indicates reading two tensors: output and gradOutput) the fast
    // path (dispatch_softmax_backward) will be selected. otherwise, the
    // general path (softmax_backward_kernel) will be selected.
    if (can_use_32bit_index && max_group_size * INNER_LOOP >= dim_size) {
      if (SIMD == SIMD32) {
        if (gradin_start == 0 && output_start == 0 && gradoutput_start == 0 &&
            dim_size % max_vec_size == 0) {
          DISPATCH_SOFTMAX_BACKWARD_IMPL(
              /*vec_size*/ max_vec_size, /*SIMD*/ SIMD32);
        } else {
          DISPATCH_SOFTMAX_BACKWARD_IMPL(/*vec_size*/ 1, /*SIMD*/ SIMD32);
        }
      } else {
        if (gradin_start == 0 && output_start == 0 && gradoutput_start == 0 &&
            dim_size % max_vec_size == 0) {
          DISPATCH_SOFTMAX_BACKWARD_IMPL(
              /*vec_size*/ max_vec_size, /*SIMD*/ SIMD16);
        } else {
          DISPATCH_SOFTMAX_BACKWARD_IMPL(/*vec_size*/ 1, /*SIMD*/ SIMD16);
        }
      }
    } else {
      if (can_use_32bit_index) {
        if (gradin_start == output_start && gradin_start == gradoutput_start) {
          SOFTMAX_BACKWARD_IMPL(
              /*vec_size*/ max_vec_size, /*IndexType*/ uint32_t);
        } else {
          SOFTMAX_BACKWARD_IMPL(/*vec_size*/ 1, /*IndexType*/ uint32_t);
        }
      } else {
        if (gradin_start == output_start && gradin_start == gradoutput_start) {
          SOFTMAX_BACKWARD_IMPL(
              /*vec_size*/ max_vec_size, /*IndexType*/ uint64_t);
        } else {
          SOFTMAX_BACKWARD_IMPL(/*vec_size*/ 1, /*IndexType*/ uint64_t);
        }
      }
    }
  } else {
    if (can_use_32bit_index) {
      if (gradin_start == output_start && gradin_start == gradoutput_start &&
          inner_size % max_vec_size == 0) {
        SPATIAL_SOFTMAX_BACKWARD_IMPL(
            /*vec_size*/ max_vec_size, /*IndexType*/ uint32_t);
      } else {
        SPATIAL_SOFTMAX_BACKWARD_IMPL(/*vec_size*/ 1, /*IndexType*/ uint32_t);
      }
    } else {
      if (gradin_start == output_start && gradin_start == gradoutput_start &&
          inner_size % max_vec_size == 0) {
        SPATIAL_SOFTMAX_BACKWARD_IMPL(
            /*vec_size*/ max_vec_size, /*IndexType*/ uint64_t);
      } else {
        SPATIAL_SOFTMAX_BACKWARD_IMPL(1, uint64_t);
      }
    }
  }
#undef DISPATCH_SOFTMAX_BACKWARD_IMPL
#undef SOFTMAX_BACKWARD_IMPL
#undef SPATIAL_SOFTMAX_BACKWARD_IMPL
}

#undef MIN_WG_NUM
#undef SIMD16
#undef SIMD32
} // namespace impl

template <bool LogSoftMax>
Tensor& host_softmax(
    const Tensor& input_,
    const int64_t dim_,
    const bool half_to_float,
    Tensor& output) {
  AT_ASSERTM(
      !half_to_float,
      "softmax with half to float conversion is not supported on XPU");
  TORCH_CHECK(
      input_.is_contiguous(),
      "** host_softmax only supports contiguous input tensor");
  if (!output.defined()) {
    output = at::native::empty_like(input_);
  }
  Tensor input = input_;
  if (input.dim() == 0)
    input = input.view(1);
  int64_t dim = maybe_wrap_dim(dim_, input.dim());
  TORCH_CHECK(
      dim >= 0 && dim < input.dim(),
      "** sycl dim must be non-negative and less than input dimensions");

  if (input.numel() > 0) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::BFloat16,
        at::ScalarType::Half,
        input.scalar_type(),
        "host_softmax",
        [&] {
          using accscalar_t = acc_type<scalar_t, true>;
          impl::spatial_softmax_forward<scalar_t, accscalar_t, LogSoftMax>(
              output, input, dim);
        });
  }
  return output;
}

template <bool LogSoftMax>
Tensor& host_softmax_backward(
    const Tensor& grad_,
    const Tensor& output_,
    int64_t dim_,
    bool half_to_float,
    Tensor& gI) {
  AT_ASSERTM(
      !half_to_float,
      "softmax with half to float conversion is not supported on XPU");
  TORCH_CHECK(
      grad_.is_contiguous(),
      "** host_softmax_backward only supports contiguous grad tensor");
  TORCH_CHECK(
      output_.is_contiguous(),
      "** host_softmax_backward only supports contiguous output tensor");

  int64_t dim = maybe_wrap_dim(dim_, grad_.dim());
  if (!gI.defined()) {
    gI = at::empty_like(grad_);
  }

  if (output_.numel() == 0) {
    return gI;
  }

  Tensor grad = grad_;
  if (grad.dim() == 0)
    grad = grad.view(1);
  TORCH_CHECK(
      dim >= 0 && dim < grad.dim(),
      "dim must be non-negative and less than input dimensions");
  Tensor output = output_;
  if (output.dim() == 0)
    output = output.view(1);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      grad.scalar_type(),
      "host_softmax_backward",
      [&] {
        using accscalar_t = acc_type<scalar_t, true>;
        impl::spatial_softmax_backward<scalar_t, accscalar_t, LogSoftMax>(
            gI, output, grad, dim);
      });
  return gI;
}

Tensor& _softmax_kernel(
    const Tensor& input,
    const int64_t dim,
    const bool half_to_float,
    Tensor& output) {
  return host_softmax<false>(input.contiguous(), dim, half_to_float, output);
}

Tensor& _log_softmax_kernel(
    const Tensor& input,
    const int64_t dim,
    const bool half_to_float,
    Tensor& output) {
  return host_softmax<true>(input.contiguous(), dim, half_to_float, output);
}

Tensor& _softmax_backward_kernel(
    const Tensor& grad,
    const Tensor& output,
    int64_t dim,
    bool half_to_float,
    Tensor& grad_input) {
  return host_softmax_backward<false>(
      grad.contiguous(), output.contiguous(), dim, half_to_float, grad_input);
}

Tensor& _log_softmax_backward_kernel(
    const Tensor& grad,
    const Tensor& output,
    int64_t dim,
    bool half_to_float,
    Tensor& grad_input) {
  return host_softmax_backward<true>(
      grad.contiguous(), output.contiguous(), dim, half_to_float, grad_input);
}

} // namespace xpu
} // namespace native
} // namespace at
