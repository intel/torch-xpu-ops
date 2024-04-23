#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <aten/sycl/Loops.h>
#include <comm/DeviceProperties.h>
#include <comm/SYCLContext.h>

using namespace xpu::sycl;

namespace at {
namespace native {
namespace xpu {

namespace impl {

#define MIN_WG_NUM 32768
#define SIMD32 32

template <
    int SIMD,
    typename accscalar_t,
    typename reduce_op,
    typename nd_item_id,
    typename local_shared>
static inline void group_reduce(
    nd_item_id item_id,
    int lid_row,
    int sub_group_num,
    accscalar_t& val,
    accscalar_t init,
    const local_shared& local_data,
    reduce_op bin_op) {
  auto sg = item_id.get_sub_group();

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
  item_id.barrier(sycl_local_fence);

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

  item_id.barrier(sycl_local_fence);
  val = local_data[lid_row][0];
}

template <
    int vec_size,
    typename accscalar_t,
    typename reduce_op,
    typename nd_item_id,
    typename local_shared>
static inline void group_reduce_spatial(
    nd_item_id item_id,
    accscalar_t input[vec_size],
    const local_shared& local_data,
    int block_row,
    reduce_op bin_op) {
  auto local_row_id = item_id.get_local_id(1);
  auto local_col_id = item_id.get_local_id(2);

#pragma unroll(vec_size)
  for (int j = 0; j < vec_size; ++j) {
    local_data[local_row_id][local_col_id][j] = input[j];
  }
  item_id.barrier(sycl_local_fence);

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
    item_id.barrier(sycl_local_fence);
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
struct DispatchSoftmaxForwardKernelFunctor {
  [[intel::reqd_sub_group_size(SIMD)]] void operator()(
      sycl::nd_item<1> item_id) const {
    if (local_size == 1 && item_id.get_global_id(0) >= outer_size)
      return;

    uint32_t lid_row = 0;
    uint32_t lid_col = item_id.get_local_id(0);
    uint32_t group_offset = item_id.get_group(0) * dim_size;
    if (local_size_row != 1) {
      lid_row = item_id.get_local_id(0) / local_size;
      lid_col = item_id.get_local_id(0) % local_size;
      group_offset =
          (item_id.get_group(0) * local_size_row + lid_row) * dim_size;
    }
    vec_t reg_in[outer_loop];
    vec_t reg_mask[outer_loop];
    auto lid_offset = lid_col * vec_size;
    auto local_stride = local_size * vec_size;

    // load data and get max value
    accscalar_t max_value = std::numeric_limits<accscalar_t>::lowest();
#pragma unroll(outer_loop)
    for (int i = 0; i < outer_loop; ++i) {
      auto index = i * local_stride + lid_offset;
      if (index >= dim_size)
        break;

      reg_in[i] = *(reinterpret_cast<vec_t*>(in_data + group_offset + index));
      if constexpr (is_masked) {
        auto vec_offset = group_offset + index;
#pragma unroll(vec_size)
        for (int j = 0; j < vec_size; ++j) {
          auto linear_idx = vec_offset + j;
          auto mask_offset = input_calc.get(linear_idx)[1];
          reg_mask[i][j] = mask_data[mask_offset];
        }
      }
#pragma unroll(vec_size)
      for (int j = 0; j < vec_size; ++j) {
        if constexpr (is_masked) {
          if (reg_mask[i][j]) {
            reg_in[i][j] = neginf;
          }
        }
        max_value = max(max_value, accscalar_t(reg_in[i][j]));
      }
    }
    if (local_size > 1) {
      group_reduce<SIMD, accscalar_t>(
          item_id,
          lid_row,
          sub_group_num,
          max_value,
          std::numeric_limits<accscalar_t>::lowest(),
          local_max,
          [](accscalar_t a, accscalar_t b) { return max(a, b); });
    }

    // get sum value
    accscalar_t sum_value = 0;
#pragma unroll(outer_loop)
    for (int i = 0;
         i < outer_loop && ((i * local_stride + lid_offset) < dim_size);
         ++i) {
#pragma unroll(vec_size)
      for (int j = 0; j < vec_size; ++j) {
        sum_value += ::exp(reg_in[i][j] - max_value);
      }
    }
    if (local_size > 1) {
      group_reduce<SIMD, accscalar_t>(
          item_id,
          lid_row,
          sub_group_num,
          sum_value,
          accscalar_t(0),
          local_sum,
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
      if (index >= dim_size)
        break;

#pragma unroll(vec_size)
      for (int j = 0; j < vec_size; ++j) {
        if constexpr (LogSoftMax) {
          reg_in[i][j] =
              static_cast<scalar_t>(reg_in[i][j] - max_value - sum_value);
        } else if (sum_value == 0) {
          reg_in[i][j] = nan;
        } else {
          reg_in[i][j] = static_cast<scalar_t>(
              ::exp(reg_in[i][j] - max_value) * sum_value);
        }
      }
      *(reinterpret_cast<vec_t*>(out_data + group_offset + index)) = reg_in[i];
    }
  }
  DispatchSoftmaxForwardKernelFunctor(
      scalar_t* in_data_,
      scalar_t* out_data_,
      int dim_size_,
      int outer_size_,
      bool* mask_data_,
      calc_t input_calc_,
      int sub_group_num_,
      int global_size_row_,
      int local_size_row_,
      int range_,
      int local_size_,
      scalar_t neginf_,
      scalar_t nan_,
      sycl::local_accessor<accscalar_t, 2> local_max_,
      sycl::local_accessor<accscalar_t, 2> local_sum_)
      : in_data(in_data_),
        out_data(out_data_),
        dim_size(dim_size_),
        outer_size(outer_size_),
        mask_data(mask_data_),
        input_calc(input_calc_),
        sub_group_num(sub_group_num_),
        global_size_row(global_size_row_),
        local_size_row(local_size_row_),
        range(range_),
        local_size(local_size_),
        neginf(neginf_),
        nan(nan_),
        local_max(local_max_),
        local_sum(local_sum_) {}

 private:
  scalar_t* in_data;
  scalar_t* out_data;
  int dim_size;
  int outer_size;
  bool* mask_data;
  calc_t input_calc;
  int sub_group_num;
  int global_size_row;
  int local_size_row;
  int range;
  int local_size;
  scalar_t neginf;
  scalar_t nan;
  sycl::local_accessor<accscalar_t, 2> local_max;
  sycl::local_accessor<accscalar_t, 2> local_sum;
};

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
struct DispatchSoftmaxForwardKernelFunctorCreator {
  DispatchSoftmaxForwardKernelFunctorCreator(
      scalar_t* in_data_,
      scalar_t* out_data_,
      int dim_size_,
      int outer_size_,
      bool* mask_data_,
      calc_t input_calc_,
      int sub_group_num_,
      int global_size_row_,
      int local_size_row_,
      int range_,
      int local_size_,
      scalar_t neginf_,
      scalar_t nan_)
      : in_data(in_data_),
        out_data(out_data_),
        dim_size(dim_size_),
        outer_size(outer_size_),
        mask_data(mask_data_),
        input_calc(input_calc_),
        sub_group_num(sub_group_num_),
        global_size_row(global_size_row_),
        local_size_row(local_size_row_),
        range(range_),
        local_size(local_size_),
        neginf(neginf_),
        nan(nan_) {}

  auto operator()(::sycl::handler& cgh) const {
    auto local_max = sycl::local_accessor<accscalar_t, 2>(
        sycl::range<2>{local_size_row, sub_group_num}, cgh);
    auto local_sum = sycl::local_accessor<accscalar_t, 2>(
        sycl::range<2>{local_size_row, sub_group_num}, cgh);
    return DispatchSoftmaxForwardKernelFunctor<
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
        nan,
        local_max,
        local_sum);
  }

 private:
  scalar_t* in_data;
  scalar_t* out_data;
  int dim_size;
  int outer_size;
  bool* mask_data;
  calc_t input_calc;
  int sub_group_num;
  int global_size_row;
  int local_size_row;
  int range;
  int local_size;
  scalar_t neginf;
  scalar_t nan;
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
    auto creator = DispatchSoftmaxForwardKernelFunctorCreator<
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
    sycl_kernel_submit(global_range, local_range, queue, creator);
  } else {
    DummyFunctor dummy;
    auto creator = DispatchSoftmaxForwardKernelFunctorCreator<
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
    sycl_kernel_submit(global_range, local_range, queue, creator);
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
  void operator()(sycl::nd_item<1> item_id) const {
    IndexType local_id = item_id.get_local_id(0);
    auto group_offset = item_id.get_group(0) * dim_size;
    int start =
        ((uint64_t)(in_data + group_offset)) % align_bytes / sizeof(scalar_t);
    IndexType loops_end = (dim_size + start + vec_size - 1) / vec_size;

    // get max value
    auto max_value = std::numeric_limits<accscalar_t>::lowest();
    for (int i = local_id; i < loops_end; i += local_size) {
      vec_t in_val = *(reinterpret_cast<vec_t*>(
          in_data + group_offset - start + i * vec_size));
#pragma unroll(vec_size)
      for (IndexType j = 0; j < vec_size; ++j) {
        IndexType linear_idx = i * vec_size + j - start;
        if (linear_idx >= 0 && linear_idx < dim_size) {
          scalar_t in_value = in_val[j];
          max_value = max(accscalar_t(in_value), max_value);
        }
      }
    }
    max_value = sycl::reduce_over_group(
        item_id.get_group(), max_value, sycl::maximum<accscalar_t>());

    // get sum value
    auto sum_value = accscalar_t(0);
    for (IndexType i = local_id; i < loops_end; i += local_size) {
      vec_t in_val = *(reinterpret_cast<vec_t*>(
          in_data + group_offset - start + i * vec_size));
#pragma unroll(vec_size)
      for (int j = 0; j < vec_size; ++j) {
        IndexType linear_idx = i * vec_size + j - start;
        if (linear_idx >= 0 && linear_idx < dim_size)
          sum_value += ::exp(accscalar_t(in_val[j]) - max_value);
      }
    }
    sum_value = sycl::reduce_over_group(
        item_id.get_group(), sum_value, sycl::plus<accscalar_t>());
    if (LogSoftMax)
      sum_value = ::log(sum_value);
    else
      sum_value = accscalar_t(1) / sum_value;

    // update result
    for (IndexType i = local_id; i < loops_end; i += local_size) {
      auto remaining = dim_size + start - i * vec_size;
      if ((start > 0 && i == 0) || (remaining < vec_size)) {
#pragma unroll(vec_size)
        for (int j = 0; j < vec_size; ++j) {
          IndexType linear_idx = i * vec_size + j - start;
          if (linear_idx >= 0 && linear_idx < dim_size) {
            if (LogSoftMax)
              out_data[group_offset + linear_idx] = static_cast<scalar_t>(
                  in_data[group_offset + linear_idx] - max_value - sum_value);
            else
              out_data[group_offset + linear_idx] = static_cast<scalar_t>(
                  ::exp(in_data[group_offset + linear_idx] - max_value) *
                  sum_value);
          }
        }
      } else {
        vec_t in_val = *(reinterpret_cast<vec_t*>(
            in_data + group_offset - start + i * vec_size));
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
            out_data + group_offset - start + i * vec_size)) = in_val;
      }
    }
  }
  SoftmaxForwardKernelFunctor(
      scalar_t* in_data_,
      scalar_t* out_data_,
      int dim_size_,
      int outer_size_,
      int local_size_)
      : in_data(in_data_),
        out_data(out_data_),
        dim_size(dim_size_),
        outer_size(outer_size_),
        local_size(local_size_) {}

 private:
  scalar_t* in_data;
  scalar_t* out_data;
  int dim_size;
  int outer_size;
  int local_size;
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
    int INNER_LOOP,
    int vec_size,
    int SIMD,
    typename scalar_t,
    typename accscalar_t,
    typename IndexType,
    bool LogSoftMax,
    int outer_loop,
    typename inp_offset_calc_t,
    typename vec_t>
struct DispatchSoftmaxForwardAddKernelFunctor {
  [[intel::reqd_sub_group_size(SIMD)]] void operator()(
      sycl::nd_item<1> item_id) const {
    if (local_size == 1 && item_id.get_global_id(0) >= outer_size)
      return;

    uint32_t lid_row = 0;
    uint32_t lid_col = item_id.get_local_id(0);
    uint32_t group_offset = item_id.get_group(0) * dim_size;
    if (local_size_row != 1) {
      lid_row = item_id.get_local_id(0) / local_size;
      lid_col = item_id.get_local_id(0) % local_size;
      group_offset =
          (item_id.get_group(0) * local_size_row + lid_row) * dim_size;
    }
    vec_t reg_in[outer_loop];
    vec_t reg_tmp;
    auto lid_offset = lid_col * vec_size;
    auto local_stride = local_size * vec_size;
    // load data and get max value
    accscalar_t max_value = std::numeric_limits<accscalar_t>::lowest();
#pragma unroll(outer_loop)
    for (int i = 0; i < outer_loop; ++i) {
      auto index = i * local_stride + lid_offset;
      if (index >= dim_size)
        break;

      auto group_batch_offset = group_offset + index;
#pragma unroll(vec_size)
      for (int j = 0; j < vec_size; ++j) {
        auto linear_offset = group_batch_offset + j;
        scalar_t input_value = in_data[input_calc.get(linear_offset)[0]];
        scalar_t other_value = other_data[input_calc.get(linear_offset)[1]];
        reg_in[i][j] = input_value + alpha * other_value;
      }

#pragma unroll(vec_size)
      for (int j = 0; j < vec_size; ++j) {
        max_value = max(max_value, accscalar_t(reg_in[i][j]));
      }
    }
    if (local_size > 1) {
      group_reduce<SIMD, accscalar_t>(
          item_id,
          lid_row,
          sub_group_num,
          max_value,
          std::numeric_limits<accscalar_t>::lowest(),
          local_max,
          [](accscalar_t a, accscalar_t b) { return max(a, b); });
    }

    // get sum value
    accscalar_t sum_value = 0;
#pragma unroll(outer_loop)
    for (int i = 0;
         i < outer_loop && ((i * local_stride + lid_offset) < dim_size);
         ++i) {
#pragma unroll(vec_size)
      for (int j = 0; j < vec_size; ++j) {
        sum_value += ::exp(reg_in[i][j] - max_value);
      }
    }
    if (local_size > 1) {
      group_reduce<SIMD, accscalar_t>(
          item_id,
          lid_row,
          sub_group_num,
          sum_value,
          accscalar_t(0),
          local_sum,
          [](accscalar_t a, accscalar_t b) { return a + b; });
    }
    if constexpr (LogSoftMax)
      sum_value = ::log(sum_value);
    else
      sum_value = accscalar_t(1) / sum_value;

      // update result
#pragma unroll(outer_loop)
    for (int i = 0; i < outer_loop; ++i) {
      auto index = i * local_stride + lid_offset;
      if (index >= dim_size)
        break;

#pragma unroll(vec_size)
      for (int j = 0; j < vec_size; ++j) {
        if constexpr (LogSoftMax) {
          reg_in[i][j] =
              static_cast<scalar_t>(reg_in[i][j] - max_value - sum_value);
        } else {
          reg_in[i][j] = static_cast<scalar_t>(
              ::exp(reg_in[i][j] - max_value) * sum_value);
        }
      }
      *(reinterpret_cast<vec_t*>(out_data + group_offset + index)) = reg_in[i];
    }
  }
  DispatchSoftmaxForwardAddKernelFunctor(
      scalar_t* in_data_,
      scalar_t* other_data_,
      scalar_t* out_data_,
      int dim_size_,
      scalar_t alpha_,
      int outer_size_,
      int other_outer_size_,
      inp_offset_calc_t input_calc_,
      int sub_group_num_,
      int global_size_row_,
      int local_size_row_,
      int range_,
      int local_size_,
      int other_offset_,
      sycl::local_accessor<accscalar_t, 2> local_max_,
      sycl::local_accessor<accscalar_t, 2> local_sum_)
      : in_data(in_data_),
        other_data(other_data_),
        out_data(out_data_),
        dim_size(dim_size_),
        alpha(alpha_),
        outer_size(outer_size_),
        other_outer_size(other_outer_size_),
        input_calc(input_calc_),
        sub_group_num(sub_group_num_),
        global_size_row(global_size_row_),
        local_size_row(local_size_row_),
        range(range_),
        local_size(local_size_),
        other_offset(other_offset_),
        local_max(local_max_),
        local_sum(local_sum_) {}

 private:
  scalar_t* in_data;
  scalar_t* other_data;
  scalar_t* out_data;
  int dim_size;
  scalar_t alpha;
  int outer_size;
  int other_outer_size;
  inp_offset_calc_t input_calc;
  int sub_group_num;
  int global_size_row;
  int local_size_row;
  int range;
  int local_size;
  int other_offset;
  sycl::local_accessor<accscalar_t, 2> local_max;
  sycl::local_accessor<accscalar_t, 2> local_sum;
};

template <
    int INNER_LOOP,
    int vec_size,
    int SIMD,
    typename scalar_t,
    typename accscalar_t,
    typename IndexType,
    bool LogSoftMax,
    int outer_loop,
    typename inp_offset_calc_t,
    typename vec_t>
struct DispatchSoftmaxForwardAddKernelFunctorCreator {
  DispatchSoftmaxForwardAddKernelFunctorCreator(
      scalar_t* in_data_,
      scalar_t* other_data_,
      scalar_t* out_data_,
      int dim_size_,
      scalar_t alpha_,
      int outer_size_,
      int other_outer_size_,
      inp_offset_calc_t input_calc_,
      int sub_group_num_,
      int global_size_row_,
      int local_size_row_,
      int range_,
      int local_size_,
      int other_offset_)
      : in_data(in_data_),
        other_data(other_data_),
        out_data(out_data_),
        dim_size(dim_size_),
        alpha(alpha_),
        outer_size(outer_size_),
        other_outer_size(other_outer_size_),
        input_calc(input_calc_),
        sub_group_num(sub_group_num_),
        global_size_row(global_size_row_),
        local_size_row(local_size_row_),
        range(range_),
        local_size(local_size_),
        other_offset(other_offset_) {}
  auto operator()(::sycl::handler& cgh) const {
    auto local_max = sycl::local_accessor<accscalar_t, 2>(
        sycl::range<2>{local_size_row, sub_group_num}, cgh);
    auto local_sum = sycl::local_accessor<accscalar_t, 2>(
        sycl::range<2>{local_size_row, sub_group_num}, cgh);

    return DispatchSoftmaxForwardAddKernelFunctor<
        INNER_LOOP,
        vec_size,
        SIMD,
        scalar_t,
        accscalar_t,
        IndexType,
        LogSoftMax,
        outer_loop,
        inp_offset_calc_t,
        vec_t>(
        in_data,
        other_data,
        out_data,
        dim_size,
        alpha,
        outer_size,
        other_outer_size,
        input_calc,
        sub_group_num,
        global_size_row,
        local_size_row,
        range,
        local_size,
        other_offset,
        local_max,
        local_sum);
  }

 private:
  scalar_t* in_data;
  scalar_t* other_data;
  scalar_t* out_data;
  int dim_size;
  scalar_t alpha;
  int outer_size;
  int other_outer_size;
  inp_offset_calc_t input_calc;
  int sub_group_num;
  int global_size_row;
  int local_size_row;
  int range;
  int local_size;
  int other_offset;
};

template <
    int INNER_LOOP,
    int vec_size,
    int SIMD,
    typename scalar_t,
    typename accscalar_t,
    typename IndexType,
    bool LogSoftMax,
    int outer_loop,
    typename inp_offset_calc_t>
void dispatch_softmax_forward_add_kernel(
    scalar_t* in_data,
    scalar_t* other_data,
    scalar_t* out_data,
    int dim_size,
    scalar_t alpha,
    int outer_size,
    int other_outer_size,
    inp_offset_calc_t input_calc) {
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
  auto other_offset = other_outer_size * dim_size;
  auto creator = DispatchSoftmaxForwardAddKernelFunctorCreator<
      INNER_LOOP,
      vec_size,
      SIMD,
      scalar_t,
      accscalar_t,
      IndexType,
      LogSoftMax,
      outer_loop,
      inp_offset_calc_t,
      vec_t>(
      in_data,
      other_data,
      out_data,
      dim_size,
      alpha,
      outer_size,
      other_outer_size,
      input_calc,
      sub_group_num,
      global_size_row,
      local_size_row,
      range,
      local_size,
      other_offset);
  sycl_kernel_submit(global_range, local_range, queue, creator);
}

template <
    int vec_size,
    typename scalar_t,
    typename accscalar_t,
    typename IndexType,
    bool LogSoftMax,
    typename vec_t>
struct SpatialSoftmaxForwardKernelFunctor {
  void operator()(sycl::nd_item<3> item_id) const {
    IndexType global_col = item_id.get_global_id(2);
    IndexType local_row_id = item_id.get_local_id(1);
    IndexType local_col_id = item_id.get_local_id(2);

    auto group_offset = item_id.get_global_id(0) * dim_size * inner_size;
    auto out_ptr = out_data + group_offset;

    // get max value
    accscalar_t max_value[vec_size];
    auto offset = local_row_id * inner_size + global_col * vec_size;
    vec_t value = *(reinterpret_cast<vec_t*>(in_data + group_offset + offset));
#pragma unroll(vec_size)
    for (int j = 0; j < vec_size; ++j) {
      max_value[j] = accscalar_t(value[j]);
    }
    for (int i = local_row_id + block_row; i < dim_size; i += block_row) {
      offset = i * inner_size + global_col * vec_size;
      value = *(reinterpret_cast<vec_t*>(in_data + group_offset + offset));
#pragma unroll(vec_size)
      for (int j = 0; j < vec_size; ++j) {
        max_value[j] = max(max_value[j], accscalar_t(value[j]));
      }
    }
    if (block_row > 1) {
      group_reduce_spatial<vec_size, accscalar_t>(
          item_id,
          max_value,
          local_data,
          block_row,
          [](accscalar_t a, accscalar_t b) { return max(a, b); });
#pragma unroll(vec_size)
      for (int j = 0; j < vec_size; ++j) {
        max_value[j] = local_data[0][local_col_id][j];
      }
      item_id.barrier();
    }

    // get sum value
    accscalar_t sum_value[vec_size];
    offset = local_row_id * inner_size + global_col * vec_size;
    value = *(reinterpret_cast<vec_t*>(in_data + group_offset + offset));
#pragma unroll(vec_size)
    for (int j = 0; j < vec_size; ++j) {
      sum_value[j] = ::exp(value[j] - max_value[j]);
    }
    for (int i = local_row_id + block_row; i < dim_size; i += block_row) {
      offset = i * inner_size + global_col * vec_size;
      value = *(reinterpret_cast<vec_t*>(in_data + group_offset + offset));
#pragma unroll(vec_size)
      for (int j = 0; j < vec_size; ++j) {
        sum_value[j] += ::exp(value[j] - max_value[j]);
      }
    }
    if (block_row > 1) {
      group_reduce_spatial<vec_size, accscalar_t>(
          item_id,
          sum_value,
          local_data,
          block_row,
          [](accscalar_t a, accscalar_t b) { return a + b; });
#pragma unroll(vec_size)
      for (int j = 0; j < vec_size; ++j) {
        if (LogSoftMax)
          sum_value[j] = ::log(local_data[0][local_col_id][j]);
        else
          sum_value[j] = accscalar_t(1) / local_data[0][local_col_id][j];
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
    if (global_col * vec_size < inner_size) {
      for (int i = local_row_id; i < dim_size; i += block_row) {
        auto offset = i * inner_size + global_col * vec_size;
        vec_t in_val =
            *(reinterpret_cast<vec_t*>(in_data + group_offset + offset));
#pragma unroll(vec_size)
        for (int j = 0; j < vec_size; ++j) {
          if (LogSoftMax)
            in_val[j] =
                static_cast<scalar_t>(in_val[j] - max_value[j] - sum_value[j]);
          else
            in_val[j] = static_cast<scalar_t>(
                ::exp(in_val[j] - max_value[j]) * sum_value[j]);
        }
        *(reinterpret_cast<vec_t*>(out_data + group_offset + offset)) = in_val;
      }
    }
  }
  SpatialSoftmaxForwardKernelFunctor(
      scalar_t* in_data_,
      scalar_t* out_data_,
      int dim_size_,
      int inner_size_,
      int outer_size_,
      int local_size_,
      int block_row_,
      int group_num_,
      sycl::local_accessor<accscalar_t, 3> local_data_)
      : in_data(in_data_),
        out_data(out_data_),
        dim_size(dim_size_),
        inner_size(inner_size_),
        outer_size(outer_size_),
        local_size(local_size_),
        block_row(block_row_),
        group_num(group_num_),
        local_data(local_data_) {}

 private:
  scalar_t* in_data;
  scalar_t* out_data;
  int dim_size;
  int inner_size;
  int outer_size;
  int local_size;
  int block_row;
  int group_num;
  sycl::local_accessor<accscalar_t, 3> local_data;
};

template <
    int vec_size,
    typename scalar_t,
    typename accscalar_t,
    typename IndexType,
    bool LogSoftMax,
    typename vec_t>
struct SpatialSoftmaxForwardKernelFunctorCreator {
  SpatialSoftmaxForwardKernelFunctorCreator(
      scalar_t* in_data_,
      scalar_t* out_data_,
      int dim_size_,
      int inner_size_,
      int outer_size_,
      int local_size_,
      int block_row_,
      int group_num_)
      : in_data(in_data_),
        out_data(out_data_),
        dim_size(dim_size_),
        inner_size(inner_size_),
        outer_size(outer_size_),
        local_size(local_size_),
        block_row(block_row_),
        group_num(group_num_) {}

  auto operator()(::sycl::handler& cgh) const {
    auto local_data = sycl::local_accessor<accscalar_t, 3>(
        sycl::range<3>{block_row, local_size, vec_size}, cgh);
    return SpatialSoftmaxForwardKernelFunctor<
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
        group_num,
        local_data);
  }

 private:
  scalar_t* in_data;
  scalar_t* out_data;
  int dim_size;
  int inner_size;
  int outer_size;
  int local_size;
  int block_row;
  int group_num;
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

  auto caller = SpatialSoftmaxForwardKernelFunctorCreator<
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
struct DispatchSoftmaxBackwardKernelFunctor {
  [[intel::reqd_sub_group_size(SIMD)]] void operator()(
      sycl::nd_item<1> item_id) const {
    if (local_size == 1 && item_id.get_global_id(0) >= outer_size)
      return;

    uint32_t lid_row = item_id.get_local_id(0) / local_size;
    uint32_t lid_col = item_id.get_local_id(0) % local_size;
    uint32_t group_offset =
        (item_id.get_group(0) * local_size_row + lid_row) * dim_size;

    // load data and get max value
    accscalar_t sum_value = accscalar_t(0);
    vec_t reg_out[NUM];
    vec_t reg_gradout[NUM];
#pragma unroll(NUM)
    for (int i = 0; i < NUM; ++i) {
      auto index = (lid_col + i * local_size) * vec_size;
      if (index >= dim_size)
        break;

      reg_out[i] = *(reinterpret_cast<vec_t*>(output + group_offset + index));
      reg_gradout[i] =
          *(reinterpret_cast<vec_t*>(gradOutput + group_offset + index));
      if constexpr (is_masked) {
        auto vec_offset = group_offset + index;
#pragma unroll(vec_size)
        for (int j = 0; j < vec_size; ++j) {
          auto linear_idx = vec_offset + j;
          auto mask_offset = input_calc.get(linear_idx)[1];
          if (mask_data[mask_offset]) {
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
    if (local_size > 1) {
      group_reduce<SIMD, accscalar_t>(
          item_id,
          lid_row,
          sub_group_num,
          sum_value,
          accscalar_t(0),
          local_sum,
          [](accscalar_t a, accscalar_t b) { return a + b; });
    }
    // update result
#pragma unroll(NUM)
    for (int i = 0; i < NUM; ++i) {
      auto index = (lid_col + i * local_size) * vec_size;
      if (index >= dim_size)
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
      *(reinterpret_cast<vec_t*>(gradInput + group_offset + index)) =
          reg_out[i];
    }
  }
  DispatchSoftmaxBackwardKernelFunctor(
      scalar_t* gradInput_,
      scalar_t* output_,
      scalar_t* gradOutput_,
      int dim_size_,
      int outer_size_,
      bool* mask_data_,
      calc_t input_calc_,
      int sub_group_num_,
      int global_size_row_,
      int local_size_row_,
      int range_,
      int local_size_,
      sycl::local_accessor<accscalar_t, 2> local_sum_)
      : gradInput(gradInput_),
        output(output_),
        gradOutput(gradOutput_),
        dim_size(dim_size_),
        outer_size(outer_size_),
        mask_data(mask_data_),
        input_calc(input_calc_),
        sub_group_num(sub_group_num_),
        global_size_row(global_size_row_),
        local_size_row(local_size_row_),
        range(range_),
        local_size(local_size_),
        local_sum(local_sum_) {}

 private:
  scalar_t* gradInput;
  scalar_t* output;
  scalar_t* gradOutput;
  int dim_size;
  int outer_size;
  bool* mask_data;
  calc_t input_calc;
  int sub_group_num;
  int global_size_row;
  int local_size_row;
  int range;
  int local_size;
  sycl::local_accessor<accscalar_t, 2> local_sum;
};

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
struct DispatchSoftmaxBackwardKernelFunctorCreator {
  DispatchSoftmaxBackwardKernelFunctorCreator(
      scalar_t* gradInput_,
      scalar_t* output_,
      scalar_t* gradOutput_,
      int dim_size_,
      int outer_size_,
      bool* mask_data_,
      calc_t input_calc_,
      int sub_group_num_,
      int global_size_row_,
      int local_size_row_,
      int range_,
      int local_size_)
      : gradInput(gradInput_),
        output(output_),
        gradOutput(gradOutput_),
        dim_size(dim_size_),
        outer_size(outer_size_),
        mask_data(mask_data_),
        input_calc(input_calc_),
        sub_group_num(sub_group_num_),
        global_size_row(global_size_row_),
        local_size_row(local_size_row_),
        range(range_),
        local_size(local_size_) {}

  auto operator()(::sycl::handler& cgh) const {
    auto local_sum = sycl::local_accessor<accscalar_t, 2>(
        sycl::range<2>{local_size_row, sub_group_num}, cgh);
    return DispatchSoftmaxBackwardKernelFunctor<
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
        local_size,
        local_sum);
  }

 private:
  scalar_t* gradInput;
  scalar_t* output;
  scalar_t* gradOutput;
  int dim_size;
  int outer_size;
  bool* mask_data;
  calc_t input_calc;
  int sub_group_num;
  int global_size_row;
  int local_size_row;
  int range;
  int local_size;
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
    auto creator = DispatchSoftmaxBackwardKernelFunctorCreator<
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
    sycl_kernel_submit(global_range, local_range, queue, creator);
  } else {
    DummyFunctor dummy;
    auto creator = DispatchSoftmaxBackwardKernelFunctorCreator<
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
    sycl_kernel_submit(global_range, local_range, queue, creator);
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
  void operator()(sycl::nd_item<1> item_id) const {
    int local_id = item_id.get_local_id(0);
    auto group_offset = item_id.get_group(0) * dim_size;
    int start =
        ((uint64_t)(output + group_offset)) % align_bytes / sizeof(scalar_t);
    int loops_end = (dim_size + start + vec_size - 1) / vec_size;

    vec_t* vec_gradin_data_ptr =
        reinterpret_cast<vec_t*>(gradInput + group_offset - start);
    const vec_t* vec_out_data_ptr =
        reinterpret_cast<const vec_t*>(output + group_offset - start);
    const vec_t* vec_gradout_data_ptr =
        reinterpret_cast<const vec_t*>(gradOutput + group_offset - start);

    // get sum value
    auto sum_value = accscalar_t(0);
    for (int i = local_id; i < loops_end; i += local_size) {
      auto gradout_val = vec_gradout_data_ptr[i];
      if (LogSoftMax) {
#pragma unroll(vec_size)
        for (int j = 0; j < vec_size; ++j) {
          int64_t linear_idx = i * vec_size + j - start;
          if (linear_idx >= 0 && linear_idx < dim_size) {
            sum_value += gradout_val[j];
          }
        }
      } else {
        vec_t out_val = vec_out_data_ptr[i];
#pragma unroll(vec_size)
        for (int j = 0; j < vec_size; ++j) {
          int64_t linear_idx = i * vec_size + j - start;
          if (linear_idx >= 0 && linear_idx < dim_size) {
            sum_value += out_val[j] * gradout_val[j];
          }
        }
      }
    }
    sum_value = sycl::reduce_over_group(
        item_id.get_group(), sum_value, sycl::plus<accscalar_t>());

    // update result
    for (int i = local_id; i < loops_end; i += local_size) {
      // handle the head and tail
      auto remaining = dim_size + start - i * vec_size;
      if ((start > 0 && i == 0) || (remaining < vec_size)) {
#pragma unroll(vec_size)
        for (int j = 0; j < vec_size; ++j) {
          auto linear_idx = i * vec_size + j - start;
          if (linear_idx >= 0 && linear_idx < dim_size) {
            auto offset = group_offset + linear_idx;
            if (LogSoftMax) {
              gradInput[offset] =
                  gradOutput[offset] - ::exp(output[offset]) * sum_value;
            } else {
              gradInput[offset] =
                  output[offset] * (gradOutput[offset] - sum_value);
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
      scalar_t* gradInput_,
      const scalar_t* output_,
      const scalar_t* gradOutput_,
      int dim_size_,
      int outer_size_,
      int local_size_)
      : gradInput(gradInput_),
        output(output_),
        gradOutput(gradOutput_),
        dim_size(dim_size_),
        outer_size(outer_size_),
        local_size(local_size_) {}

 private:
  scalar_t* gradInput;
  const scalar_t* output;
  const scalar_t* gradOutput;
  int dim_size;
  int outer_size;
  int local_size;
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
struct SpatialSoftmaxBackwardKernelFunctor {
  void operator()(sycl::nd_item<3> item_id) const {
    auto global_col = item_id.get_global_id(2);
    auto local_row_id = item_id.get_local_id(1);
    auto local_col_id = item_id.get_local_id(2);

    auto group_offset = item_id.get_global_id(0) * dim_size * inner_size;
    auto gradin_ptr = gradInput + group_offset;
    auto out_ptr = output + group_offset;
    auto gradout_ptr = gradOutput + group_offset;

    // get sum value
    accscalar_t sum_value[vec_size];
#pragma unroll(vec_size)
    for (int j = 0; j < vec_size; ++j)
      sum_value[j] = accscalar_t(0);

    for (int i = local_row_id; i < dim_size; i += block_row) {
      auto offset = i * inner_size + global_col * vec_size;
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
    if (block_row > 1) {
      group_reduce_spatial<vec_size, accscalar_t>(
          item_id,
          sum_value,
          local_data,
          block_row,
          [](accscalar_t a, accscalar_t b) { return a + b; });
#pragma unroll(vec_size)
      for (int j = 0; j < vec_size; ++j) {
        sum_value[j] = local_data[0][local_col_id][j];
      }
    }

    // update result
    if (global_col * vec_size < inner_size) {
      for (int i = local_row_id; i < dim_size; i += block_row) {
        auto offset = i * inner_size + global_col * vec_size;
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
  SpatialSoftmaxBackwardKernelFunctor(
      scalar_t* gradInput_,
      const scalar_t* output_,
      const scalar_t* gradOutput_,
      int dim_size_,
      int inner_size_,
      int outer_size_,
      int local_size_,
      int block_row_,
      sycl::local_accessor<accscalar_t, 3> local_data_)
      : gradInput(gradInput_),
        output(output_),
        gradOutput(gradOutput_),
        dim_size(dim_size_),
        inner_size(inner_size_),
        outer_size(outer_size_),
        local_size(local_size_),
        block_row(block_row_),
        local_data(local_data_) {}

 private:
  scalar_t* gradInput;
  const scalar_t* output;
  const scalar_t* gradOutput;
  int dim_size;
  int inner_size;
  int outer_size;
  int local_size;
  int block_row;
  sycl::local_accessor<accscalar_t, 3> local_data;
};

template <
    int vec_size,
    typename scalar_t,
    typename accscalar_t,
    bool LogSoftMax,
    typename vec_t>
struct SpatialSoftmaxBackwardKernelFunctorCreator {
  SpatialSoftmaxBackwardKernelFunctorCreator(
      scalar_t* gradInput_,
      const scalar_t* output_,
      const scalar_t* gradOutput_,
      int dim_size_,
      int inner_size_,
      int outer_size_,
      int local_size_,
      int block_row_)
      : gradInput(gradInput_),
        output(output_),
        gradOutput(gradOutput_),
        dim_size(dim_size_),
        inner_size(inner_size_),
        outer_size(outer_size_),
        local_size(local_size_),
        block_row(block_row_) {}
  auto operator()(::sycl::handler& cgh) const {
    auto local_data = sycl::local_accessor<accscalar_t, 3>(
        sycl::range<3>{block_row, local_size, vec_size}, cgh);
    return SpatialSoftmaxBackwardKernelFunctor<
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
  }

 private:
  scalar_t* gradInput;
  const scalar_t* output;
  const scalar_t* gradOutput;
  int dim_size;
  int inner_size;
  int outer_size;
  int local_size;
  int block_row;
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

  auto creator = SpatialSoftmaxBackwardKernelFunctorCreator<
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
  sycl_kernel_submit(global_range, local_range, queue, creator);
}

} // namespace impl

} // namespace xpu
} // namespace native
} // namespace at
