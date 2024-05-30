#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/CanUse32BitIndexMath.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/xpu/XPUContext.h>
#include <aten/sycl/Loops.h>
#include <aten/sycl/Reduce.h>
#include <comm/SYCLContext.h>

namespace at {
namespace native {
namespace xpu {

#define SIMD32 32
#define SIMD16 16

inline bool batch_norm_use_channels_last_kernels(const at::Tensor& self) {
  return (
      self.is_contiguous(at::MemoryFormat::ChannelsLast) ||
      self.is_contiguous(at::MemoryFormat::ChannelsLast3d) ||
      (self.is_contiguous() && self.strides()[1] == 1));
}

struct InvStd {
  template <typename T>
  inline T operator()(T var, double epsilon) const {
    T invstd = 0.0f;
    if (var != static_cast<T>(0.0f) || epsilon != static_cast<T>(0.0f)) {
      invstd = static_cast<T>(1.0f) / std::sqrt(var + static_cast<T>(epsilon));
    }
    return invstd;
  }
};

struct Var {
  template <typename T>
  inline T operator()(T var, double epsilon) const {
    return var;
  }
};

static int get_num_threads(int nElem, int max_size) {
  int threadSizes[6] = {16, 32, 64, 128, 256, max_size};
  for (int i = 0; i < 6; ++i) {
    if (nElem <= threadSizes[i]) {
      return threadSizes[i];
    }
  }
  return max_size;
}

int get_prefer_wg_size(unsigned int nHw, int simd) {
  if (nHw < simd)
    return simd;
  auto size_problem = get_num_threads(nHw, simd * simd);
  auto wg_size = syclMaxWorkGroupSize();
  return std::min(int64_t(size_problem), wg_size);
}

int get_prefer_simd(int numPlane, int nHw) {
  // decide SIMD: SIMD32 or SIMD16

  auto dev_id = at::xpu::getDeviceIndexOfCurrentQueue();

  auto* dev_prop = at::xpu::getDeviceProperties(dev_id);
  auto sub_group_size = dev_prop->sub_group_sizes;
  int simd = sub_group_size[1];
  if (simd <= SIMD16)
    return simd;

  // if max supported simd >16
  if (nHw <= SIMD16)
    return SIMD16;
  if (simd >= SIMD32 && nHw <= SIMD32)
    return SIMD32;

  int64_t target_tile_size = syclMaxWorkItemsPerTile(dev_id);
  // for work group barrier perf
  int64_t wg_size = syclMaxWorkItemsPerEU(dev_id);
  if (simd == SIMD32) {
    // when setting wg_size 256 can achieve high occupancy, use SIMD16
    if (wg_size * numPlane >= target_tile_size)
      return SIMD16;
    // for latency case
    if (nHw <= 1024 && numPlane > 128 && SIMD16 * SIMD16 >= wg_size) {
      return SIMD16;
    }
  }
  return simd;
}

template <
    int SIMD,
    typename VarTransform,
    typename input_scalar_t,
    typename stat_scalar_t,
    typename stat_accscalar_t>
struct BatchNormCollectStatisticsKernelFunctor
    : public __SYCL_KER_CONFIG_CONVENTION__ {
  [[intel::reqd_sub_group_size(SIMD)]] void operator()(
      sycl::nd_item<2> item) const {
    int plane = item.get_group(0);
    int tid = item.get_local_linear_id();
    auto sg = item.get_sub_group();
    auto sg_lid = sg.get_local_linear_id();
    auto sg_id = sg.get_group_linear_id();

    // Compute the mean and variance across (batch, x/y/z)
    // this uses the Welford (in the for loop)/parallel algorithm (to sum
    // across the group)
    // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_Online_algorithm
    // and the parallel algorithm on the same page.
    // We use two shuffles to reduce across the entire group.
    // https://devblogs.nvidia.com/faster-parallel-reductions-kepler/ has a
    // description.

    // first the reductions each thread does separately
    stat_accscalar_t avg = 0;
    stat_accscalar_t var_n = 0;
    int n = 0;
    for (int batch = item.get_local_id(0); batch < N_;
         batch += item.get_local_range(0)) {
      for (int x = item.get_local_id(1); x < Hw_;
           x += item.get_local_range(1)) {
        auto offset = batch * batch_stride_ + plane * Hw_ + x;
        stat_accscalar_t v = input_[offset];
        stat_accscalar_t d1 = v - avg;
        n++;
        avg += d1 / n;
        var_n += d1 * (v - avg);
      }
    }

    // first warpSum to get one value per thread to
    // one value per warp
#pragma unroll
    for (int i = 1; i < SIMD; i <<= 1) {
      stat_accscalar_t o_avg = sg.shuffle_xor(avg, i);
      int o_n = sg.shuffle_xor(n, i);
      stat_accscalar_t factor = 1.0 / fmaxf(1.0, n + o_n);
      var_n += sg.shuffle_xor(var_n, i) +
          (avg - o_avg) * (avg - o_avg) * n * o_n * factor;
      avg = (n * avg + o_n * o_avg) * factor;
      n += o_n;
    }

    // this writes each warps item into shared memory

    if (sg_lid == 0) {
      shared_n_[sg_id] = n;
      shared_avg_var_[sg_id * 2] = avg;
      shared_avg_var_[sg_id * 2 + 1] = var_n;
    }
    item.barrier(sycl_local_fence);
    // now have a second warpSum to reduce the intermediate values
    // from shared memory to a single number. The very first
    // thread writes it to shared memory.

    if (tid < sg_num_) {
      n = shared_n_[tid];
      avg = shared_avg_var_[2 * tid];
      var_n = shared_avg_var_[2 * tid + 1];
    } else {
      n = 0;
      avg = stat_accscalar_t(0);
      var_n = stat_accscalar_t(0);
    }
#pragma unroll
    for (int i = 1; i < SIMD; i <<= 1) {
      stat_accscalar_t o_avg = sg.shuffle_xor(avg, i);
      int o_n = sg.shuffle_xor(n, i);
      stat_accscalar_t factor = 1.0f / fmaxf(1.0f, n + o_n);
      var_n += sg.shuffle_xor(var_n, i) +
          (avg - o_avg) * (avg - o_avg) * n * o_n * factor;
      avg = (n * avg + o_n * o_avg) * factor;
      n += o_n;
    }

    // Save the mean, variance, and moving averages
    if (tid == 0) {
      if (save_mean_ != nullptr) {
        save_mean_[plane] = avg;
      }
      if (save_transformed_var_ != nullptr) {
        save_transformed_var_[plane] =
            VarTransform{}(var_n / (N_ * Hw_), epsilon_);
      }
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    shared_n_ = sycl_local_acc_t<stat_accscalar_t, 1>(
        sycl::range<1>{(size_t)sg_num_}, cgh);
    shared_avg_var_ = sycl_local_acc_t<stat_accscalar_t, 1>(
        sycl::range<1>{(size_t)sg_num_ * 2 + 2}, cgh);
  }

  BatchNormCollectStatisticsKernelFunctor(
      int N,
      int numPlane,
      int Hw,
      const input_scalar_t* input,
      const stat_accscalar_t epsilon,
      const stat_accscalar_t momentum,
      stat_accscalar_t* save_mean,
      stat_accscalar_t* save_transformed_var,
      int64_t sg_num,
      int batch_stride)
      : N_(N),
        numPlane_(numPlane),
        Hw_(Hw),
        input_(input),
        epsilon_(epsilon),
        momentum_(momentum),
        save_mean_(save_mean),
        save_transformed_var_(save_transformed_var),
        sg_num_(sg_num),
        batch_stride_(batch_stride) {}

 private:
  int N_;
  int numPlane_;
  int Hw_;
  const input_scalar_t* input_;
  const stat_accscalar_t epsilon_;
  const stat_accscalar_t momentum_;
  stat_accscalar_t* save_mean_;
  stat_accscalar_t* save_transformed_var_;
  int64_t sg_num_;
  int batch_stride_;
  sycl_local_acc_t<stat_accscalar_t, 1> shared_n_;
  sycl_local_acc_t<stat_accscalar_t, 1> shared_avg_var_;
};

template <
    int SIMD,
    typename VarTransform,
    typename input_scalar_t,
    typename stat_scalar_t,
    typename stat_accscalar_t,
    typename index_t>
void batch_norm_collect_statistics_kernel(
    int N,
    int numPlane,
    int Hw,
    const input_scalar_t* input,
    const stat_accscalar_t epsilon,
    const stat_accscalar_t momentum,
    stat_accscalar_t* save_mean,
    stat_accscalar_t* save_transformed_var) {
  auto& queue = getCurrentSYCLQueue();
  int64_t wg_size = get_prefer_wg_size(N * Hw, SIMD);
  int64_t work_group_size_x = get_num_threads(Hw, wg_size);
  int64_t work_group_size_y = std::max(int64_t(1), wg_size / work_group_size_x);
  work_group_size_y = std::min(int64_t(N), work_group_size_y);
  int64_t sg_num = work_group_size_x * work_group_size_y / SIMD;
  auto batch_stride = numPlane * Hw;
  auto caller = BatchNormCollectStatisticsKernelFunctor<
      SIMD,
      VarTransform,
      input_scalar_t,
      stat_scalar_t,
      stat_accscalar_t>(
      N,
      numPlane,
      Hw,
      input,
      epsilon,
      momentum,
      save_mean,
      save_transformed_var,
      sg_num,
      batch_stride);
  sycl_kernel_submit(
      sycl::range<2>(
          (size_t)numPlane * work_group_size_y, (size_t)work_group_size_x),
      sycl::range<2>((size_t)work_group_size_y, (size_t)work_group_size_x),
      queue,
      caller);
}

template <typename scalar_t, typename index_t, typename VarTransform>
void batch_norm_stats_template(
    const Tensor& out_mean,
    const Tensor& out_invstd,
    const Tensor& input_,
    double epsilon) {
  using accscalar_t = at::acc_type<scalar_t, true>;
  int64_t n_input = input_.size(1);
  Tensor dummy_mean_;
  Tensor dummy_var_;
  auto input_reshaped = input_.reshape(
      {input_.size(0),
       input_.size(1),
       -1}); // internally we merge the feature dimensions
  int N = input_reshaped.size(0);
  int C = input_reshaped.size(1);
  int Hw = input_reshaped.size(2);

  at::native::resize_output(out_mean, {n_input});
  at::native::resize_output(out_invstd, {n_input});
  TORCH_INTERNAL_ASSERT(
      out_invstd.dim() == 1 && out_invstd.is_contiguous() &&
      out_invstd.sizes()[0]);
  TORCH_INTERNAL_ASSERT(
      out_mean.dim() == 1 && out_mean.is_contiguous() && out_mean.sizes()[0]);

  auto input_ptr = input_reshaped.data_ptr<scalar_t>();
  auto mean_ptr = out_mean.data_ptr<accscalar_t>();
  auto invstd_ptr = out_invstd.data_ptr<accscalar_t>();
  int simd = get_prefer_simd(C, N * Hw);
  if (simd == SIMD32) {
    batch_norm_collect_statistics_kernel<
        SIMD32,
        VarTransform,
        scalar_t,
        scalar_t,
        accscalar_t,
        index_t>(N, C, Hw, input_ptr, epsilon, 0.0, mean_ptr, invstd_ptr);
  } else {
    batch_norm_collect_statistics_kernel<
        SIMD16,
        VarTransform,
        scalar_t,
        scalar_t,
        accscalar_t,
        index_t>(N, C, Hw, input_ptr, epsilon, 0.0, mean_ptr, invstd_ptr);
  }
}

template <typename scalar_t>
int inline get_nhwc_suggest_vec_size(
    const Tensor input,
    int reduction_size,
    int channels) {
  if (!batch_norm_use_channels_last_kernels(input))
    return 1;
  // no need to vectorize if channels < 16
  if (channels < 16)
    return 1;
  // if small reduction size, make no vectorization for higher occupancy
  if (reduction_size < 8 * syclMaxWorkGroupSize())
    return 1;

  // just to load/store data
  auto func = [](scalar_t a) { return a + static_cast<scalar_t>(1.0f); };
  at::detail::Array<char*, 1> data;
  data[0] = (char*)input.data_ptr();

  int vec_size = memory::can_vectorize_up_to<decltype(func)>(data);

  // for resnet50 shape, bf16 type, vec 4 have better performance
  if (vec_size == 8 && reduction_size == 256 * 56 * 56 &&
      (channels == 128 || channels == 256))
    return 4;

  if (channels % vec_size != 0)
    return 1;
  return vec_size;
}

template <
    typename scalar_t,
    typename accscalar_t,
    typename vec_t,
    typename vec_y,
    int vec_size,
    bool two_pass_reduce>
struct BatchNormReduceSumChannelsLastKernelFunctor
    : public __SYCL_KER_CONFIG_CONVENTION__ {
  void operator()(sycl::nd_item<2> item) const {
    // int plane = item.get_group(0);
    // int tid = item.get_local_linear_id();
    auto sg = item.get_sub_group();

    // offset along m dimension
    int m_offset = item.get_global_id(0);
    int c_offset_base = item.get_global_id(1) * vec_size;

    int thread_idx_y = item.get_local_id(0);
    // int thread_idx_x = item.get_local_id(1);
    int group_idx_y = item.get_group(0);
    // int group_idx_x = item.get_group(1);

    int address_base = m_offset * stride_ + c_offset_base;
    int inner_loop_stride = global_range_y_;
    int address_increment = inner_loop_stride * stride_;

    accscalar_t x_sum[vec_size] = {0.0f};
    accscalar_t x_sq_sum[vec_size] = {0.0f};
    // thread reduction
    for (int i = 0; i < loop_count_; i++) {
      vec_t x_math_vec = *(reinterpret_cast<vec_t*>(input_ptr_ + address_base));
#pragma unroll
      for (int j = 0; j < vec_size; j++) {
        auto c_offset = c_offset_base + j;

        if (c_offset < stride_ && m_offset < reduction_size_) {
          // scalar_t arr = input_ptr_[address_base + j];
          auto x_math = x_math_vec[j];
          x_sum[j] += x_math;
          x_sq_sum[j] += x_math * x_math;
        }
      }
      m_offset += inner_loop_stride;
      address_base += address_increment;
    }

#pragma unroll
    for (int j = 0; j < vec_size; j++) {
      vec_y value;
      value[0] = x_sum[j];
      value[1] = x_sq_sum[j];

      value = group_y_reduce(
          item, shared_, value, [](accscalar_t a, accscalar_t b) {
            return a + b;
          });

      x_sum[j] = value[0];
      x_sq_sum[j] = value[1];

      item.barrier(sycl_local_fence);
    }

#pragma unroll
    for (int j = 0; j < vec_size; j++) {
      auto c_offset = c_offset_base + j;
      // global_reduciton
      if (thread_idx_y == 0 && c_offset < stride_) {
        if constexpr (two_pass_reduce) {
          // write to temp[c][group_idx_y]
          // int offset = c_offset * group_num_y_ + group_idx_y;
          temp_sum_ptr_[c_offset * group_num_y_ + group_idx_y] = x_sum[j];
          temp_sum_sq_ptr_[c_offset * group_num_y_ + group_idx_y] = x_sq_sum[j];
        } else {
          out_mean_ptr_[c_offset] = x_sum[j];
          out_invstd_ptr_[c_offset] = x_sq_sum[j];
        }
      }
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    shared_ = sycl_local_acc_t<vec_y, 1>(sycl::range<1>{(size_t)wg_size_}, cgh);
  }

  BatchNormReduceSumChannelsLastKernelFunctor(
      const int reduction_size,
      const int stride,
      int global_range_y,
      int local_range_y,
      int group_num_x,
      int group_num_y,
      accscalar_t* temp_sum_ptr,
      accscalar_t* temp_sum_sq_ptr,
      int wg_size,
      scalar_t* input_ptr,
      accscalar_t* out_mean_ptr,
      accscalar_t* out_invstd_ptr,
      int loop_count)
      : reduction_size_(reduction_size),
        stride_(stride),
        global_range_y_(global_range_y),
        local_range_y_(local_range_y),
        group_num_x_(group_num_x),
        group_num_y_(group_num_y),
        temp_sum_ptr_(temp_sum_ptr),
        temp_sum_sq_ptr_(temp_sum_sq_ptr),
        wg_size_(wg_size),
        input_ptr_(input_ptr),
        out_mean_ptr_(out_mean_ptr),
        out_invstd_ptr_(out_invstd_ptr),
        loop_count_(loop_count) {}

 private:
  const int reduction_size_;
  const int stride_;
  int global_range_y_;
  int local_range_y_;
  int group_num_x_;
  int group_num_y_;
  accscalar_t* temp_sum_ptr_;
  accscalar_t* temp_sum_sq_ptr_;
  int wg_size_;
  scalar_t* input_ptr_;
  accscalar_t* out_mean_ptr_;
  accscalar_t* out_invstd_ptr_;
  int loop_count_;
  sycl_local_acc_t<vec_y, 1> shared_;
};

template <typename accscalar_t>
struct BatchNormReduceSumChannelsLastTwoPassKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    auto local_id = item.get_local_linear_id();
    // auto global_id = item.get_global_linear_id();
    auto c_offset = item.get_group_linear_id();

    accscalar_t temp_sum_val = 0.0f;
    accscalar_t temp_sum_sq_val = 0.0f;
    for (int i = local_id; i < group_num_y_; i += wg_size_) {
      int offset = c_offset * group_num_y_ + i;
      temp_sum_val += temp_sum_ptr_[offset];
      temp_sum_sq_val += temp_sum_sq_ptr_[offset];
    }
    auto total_sum = sycl::reduce_over_group(
        item.get_group(), temp_sum_val, sycl::plus<accscalar_t>());
    auto total_sum_sq = sycl::reduce_over_group(
        item.get_group(), temp_sum_sq_val, sycl::plus<accscalar_t>());
    if (local_id == 0) {
      out_mean_ptr_[c_offset] = total_sum;
      out_invstd_ptr_[c_offset] = total_sum_sq;
    }
  }
  BatchNormReduceSumChannelsLastTwoPassKernelFunctor(
      int group_num_y,
      accscalar_t* temp_sum_ptr,
      accscalar_t* temp_sum_sq_ptr,
      int wg_size,
      accscalar_t* out_mean_ptr,
      accscalar_t* out_invstd_ptr)
      : group_num_y_(group_num_y),
        temp_sum_ptr_(temp_sum_ptr),
        temp_sum_sq_ptr_(temp_sum_sq_ptr),
        wg_size_(wg_size),
        out_mean_ptr_(out_mean_ptr),
        out_invstd_ptr_(out_invstd_ptr) {}

 private:
  int group_num_y_;
  accscalar_t* temp_sum_ptr_;
  accscalar_t* temp_sum_sq_ptr_;
  int wg_size_;
  accscalar_t* out_mean_ptr_;
  accscalar_t* out_invstd_ptr_;
};

inline int div_up(int a, int b) {
  return (a + b - 1) / b;
}

std::tuple<sycl::range<2>, sycl::range<2>> flexible_launch_configs(
    const int reduction,
    const int stride,
    const bool coop_flag = false,
    const int loops_per_item = 1) {
  int wg_size = syclMaxWorkItemsPerEU();
  int group_x = std::min(last_pow2(stride), 32);
  int group_y =
      std::min(last_pow2(div_up(reduction, loops_per_item)), wg_size / group_x);
  if (group_x * group_y != wg_size) {
    group_x = std::min(last_pow2(stride), wg_size / group_y);
  }

  int grid_x = div_up(stride, group_x);
  //  int grid_y = std::min(div_up(reduction, group_y * loops_per_item), 1024);
  int grid_y = std::min(
      div_up(reduction, group_y * loops_per_item),
      int(syclMaxWorkItemsPerTile()) / (grid_x * group_x) / (group_y));
  grid_y = std::max(grid_y, 1);

  if (coop_flag) {
    // it's not worth having a grid reduction if the reduction dimension is not
    // big enough
    grid_y = grid_y < 8 ? 1 : grid_y;
  }

  sycl::range<2> local_range(group_y, group_x);
  sycl::range<2> global_range(grid_y * group_y, grid_x * group_x);

  return std::make_tuple(global_range, local_range);
}

// sum x and x^2 in channels
template <
    typename scalar_t,
    typename accscalar_t,
    int vec_size,
    bool two_pass_reduce>
void batch_norm_reduce_sum_channels_last_kernel(
    const Tensor input,
    Tensor& out_mean,
    Tensor& out_invstd,
    const int reduction_size,
    const int stride) {
  sycl::range<2> global_range(1, 1), local_range(1, 1);
  std::tie(global_range, local_range) =
      flexible_launch_configs(reduction_size, stride / vec_size, true);
  using vec_t = memory::aligned_vector<scalar_t, vec_size>;
  auto& queue = getCurrentSYCLQueue();
  auto global_range_y = global_range[0];
  auto local_range_y = local_range[0];
  int group_num_x = global_range[1] / local_range[1];
  int group_num_y = global_range[0] / local_range[0];
  Tensor temp_sum, temp_sum_sq;
  accscalar_t* temp_sum_ptr = nullptr;
  accscalar_t* temp_sum_sq_ptr = nullptr;
  if constexpr (two_pass_reduce) {
    out_mean.zero_();
    out_invstd.zero_();
    temp_sum = at::empty({group_num_y * stride}, out_mean.options());
    temp_sum_sq = at::empty({group_num_y * stride}, out_mean.options());
    temp_sum_ptr = temp_sum.data_ptr<accscalar_t>();
    temp_sum_sq_ptr = temp_sum_sq.data_ptr<accscalar_t>();
  }
  int wg_size = local_range[0] * local_range[1];

  auto input_ptr = input.data_ptr<scalar_t>();
  auto out_mean_ptr = out_mean.data_ptr<accscalar_t>();
  auto out_invstd_ptr = out_invstd.data_ptr<accscalar_t>();

  int loop_count = 1 + (reduction_size - 1) / (global_range_y);
  using vec_y = at::detail::Array<accscalar_t, 2>;

  auto caller = BatchNormReduceSumChannelsLastKernelFunctor<
      scalar_t,
      accscalar_t,
      vec_t,
      vec_y,
      vec_size,
      two_pass_reduce>(
      reduction_size,
      stride,
      global_range_y,
      local_range_y,
      group_num_x,
      group_num_y,
      temp_sum_ptr,
      temp_sum_sq_ptr,
      wg_size,
      input_ptr,
      out_mean_ptr,
      out_invstd_ptr,
      loop_count);
  sycl_kernel_submit(global_range, local_range, queue, caller);

  // reduce temp sum
  if constexpr (two_pass_reduce) {
    int wg_size = std::min(group_num_y, int(syclMaxWorkItemsPerEU()));
    auto caller =
        BatchNormReduceSumChannelsLastTwoPassKernelFunctor<accscalar_t>(
            group_num_y,
            temp_sum_ptr,
            temp_sum_sq_ptr,
            wg_size,
            out_mean_ptr,
            out_invstd_ptr);
    sycl_kernel_submit(
        (size_t)stride * wg_size, (size_t)wg_size, queue, caller);
  }
}

template <typename VarTransform, typename scalar_t, typename stat_accscalar_t>
struct BatchNormUpdateMeanVarKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    auto c_offset = item.get_global_linear_id();
    if (c_offset < channel_num_) {
      scalar_t mean = mean_[c_offset] * factor_;

      mean_[c_offset] = mean;
      var_[c_offset] =
          VarTransform{}(var_[c_offset] * factor_ - mean * mean, epsilon_);
    }
  }
  BatchNormUpdateMeanVarKernelFunctor(
      scalar_t* mean,
      scalar_t* var,
      int channel_num,
      scalar_t factor,
      stat_accscalar_t epsilon)
      : mean_(mean),
        var_(var),
        channel_num_(channel_num),
        factor_(factor),
        epsilon_(epsilon) {}

 private:
  scalar_t* mean_;
  scalar_t* var_;
  int channel_num_;
  scalar_t factor_;
  stat_accscalar_t epsilon_;
};

template <typename VarTransform, typename scalar_t, typename stat_accscalar_t>
void batch_norm_update_mean_var_kernel(
    scalar_t* mean_,
    scalar_t* var_,
    int channel_num,
    scalar_t factor,
    stat_accscalar_t epsilon) {
  auto& queue = getCurrentSYCLQueue();
  int64_t wg_size = std::min(
      int64_t(channel_num),
      syclMaxWorkItemsPerEU()); // for work group barrier

  sycl::range<1> local_range(wg_size);
  sycl::range<1> global_range((channel_num + wg_size - 1) / wg_size * wg_size);

  auto caller = BatchNormUpdateMeanVarKernelFunctor<
      VarTransform,
      scalar_t,
      stat_accscalar_t>(mean_, var_, channel_num, factor, epsilon);

  sycl_kernel_submit(global_range, local_range, queue, caller);
}

template <typename scalar_t, typename VarTransform>
void batch_norm_stats_channels_last_template(
    Tensor& out_mean,
    Tensor& out_invstd,
    const Tensor& input,
    double epsilon) {
  using accscalar_t = acc_type<scalar_t, true>;

  const auto stride = input.sizes()[1];
  const auto reduction_size = input.numel() / stride;

  at::native::resize_output(out_mean, {stride});
  at::native::resize_output(out_invstd, {stride});
  TORCH_INTERNAL_ASSERT(
      out_invstd.dim() == 1 && out_invstd.is_contiguous() &&
      out_invstd.sizes()[0]);
  TORCH_INTERNAL_ASSERT(
      out_mean.dim() == 1 && out_mean.is_contiguous() && out_mean.sizes()[0]);

  int suggest_vec_size =
      get_nhwc_suggest_vec_size<scalar_t>(input, reduction_size, stride);

#define DISPATCH_REDUCE_2_PASS_IMPL(vec_size)                       \
  {                                                                 \
    batch_norm_reduce_sum_channels_last_kernel<                     \
        scalar_t,                                                   \
        accscalar_t,                                                \
        vec_size,                                                   \
        true>(input, out_mean, out_invstd, reduction_size, stride); \
  }

#define DISPATCH_REDUCE_IMPL(vec_size)                               \
  {                                                                  \
    batch_norm_reduce_sum_channels_last_kernel<                      \
        scalar_t,                                                    \
        accscalar_t,                                                 \
        vec_size,                                                    \
        false>(input, out_mean, out_invstd, reduction_size, stride); \
  }
  sycl::range<2> global_range(1, 1), local_range(1, 1);

  switch (suggest_vec_size) {
    case 8: {
      constexpr int vec_size = 8;

      std::tie(global_range, local_range) =
          flexible_launch_configs(reduction_size, stride / vec_size, true);
      int group_num_y = global_range[0] / local_range[0];
      if (group_num_y > 1) {
        DISPATCH_REDUCE_2_PASS_IMPL(vec_size);
      } else {
        DISPATCH_REDUCE_IMPL(vec_size);
      }
      break;
    }
    case 4: {
      constexpr int vec_size = 4;

      std::tie(global_range, local_range) =
          flexible_launch_configs(reduction_size, stride / vec_size, true);
      int group_num_y = global_range[0] / local_range[0];
      if (group_num_y > 1) {
        DISPATCH_REDUCE_2_PASS_IMPL(vec_size);
      } else {
        DISPATCH_REDUCE_IMPL(vec_size);
      }
      break;
    }
    default: {
      constexpr int vec_size = 1;

      std::tie(global_range, local_range) =
          flexible_launch_configs(reduction_size, stride / vec_size, true);
      int group_num_y = global_range[0] / local_range[0];
      if (group_num_y > 1) {
        DISPATCH_REDUCE_2_PASS_IMPL(vec_size);
      } else {
        DISPATCH_REDUCE_IMPL(vec_size);
      }
    }
  }

  auto out_mean_ptr = out_mean.data_ptr<accscalar_t>();
  auto out_invstd_ptr = out_invstd.data_ptr<accscalar_t>();
  const auto factor = static_cast<accscalar_t>(1.0f / reduction_size);
  batch_norm_update_mean_var_kernel<VarTransform>(
      out_mean_ptr, out_invstd_ptr, stride, factor, epsilon);
#undef DISPATCH_REDUCE_2_PASS_IMPL
#undef DISPATCH_REDUCE_IMPL
}

std::tuple<Tensor, Tensor> batch_norm_stats_kernel(
    const Tensor& self,
    double epsilon) {
  auto options =
      self.options().dtype(at::toAccumulateType(self.scalar_type(), true));
  auto n_channels = self.size(1);
  auto save_mean = at::empty({n_channels}, options);
  auto save_invstd = at::empty({n_channels}, options);

  bool use_channels_last_kernel = batch_norm_use_channels_last_kernels(self);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "batch_norm_stats_xpu",
      [&] {
        if (canUse32BitIndexMath(self)) {
          if (use_channels_last_kernel) {
            batch_norm_stats_channels_last_template<scalar_t, InvStd>(
                save_mean, save_invstd, self, epsilon);
          } else {
            batch_norm_stats_template<scalar_t, int32_t, InvStd>(
                save_mean, save_invstd, self, epsilon);
          }
        } else {
          batch_norm_stats_template<scalar_t, int64_t, InvStd>(
              save_mean, save_invstd, self, epsilon);
        }
      });
  return std::tuple<Tensor, Tensor>(save_mean, save_invstd);
}

} // namespace xpu
} // namespace native
} // namespace at
