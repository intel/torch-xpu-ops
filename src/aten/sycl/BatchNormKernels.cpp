#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/CanUse32BitIndexMath.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/xpu/XPUContext.h>
#include <aten/Resize.h>
#include <aten/sycl/Loops.h>
#include <aten/sycl/Reduce.h>
#include <comm/SYCLContext.h>
#include <comm/XPUMathCompat.h>

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

// ========================== batch_norm_stats ==========================

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

// ========================== batch_norm_elemt ==========================

ScalarType first_type() {
  return ScalarType::Undefined;
}

template <typename... Args>
ScalarType first_type(const Tensor& arg, const Args&... parameters) {
  return arg.defined() ? arg.scalar_type() : first_type(parameters...);
}

// A transform is mixed type if the parameters are higher precision than the
// input
template <typename... Args>
bool is_mixed_type(const Tensor& input, const Args&... parameters) {
  const auto parameter_type = first_type(parameters...);
  return (
      (parameter_type != ScalarType::Undefined) &&
      (parameter_type != input.scalar_type()));
}

enum class Impl {
  Contiguous,
  ChannelsLast,
  General,
};

inline Impl batch_norm_choose_impl(const Tensor& self) {
  if (!canUse32BitIndexMath(self)) {
    return Impl::General;
  }

  if (self.is_contiguous()) {
    return self.strides()[1] == 1 ? Impl::ChannelsLast : Impl::Contiguous;
  }

  if (self.is_contiguous(at::MemoryFormat::ChannelsLast)) {
    return Impl::ChannelsLast;
  }

  return Impl::General;
}

template <
    typename input_scalar_t,
    typename stat_scalar_t,
    typename stat_accscalar_t,
    bool train,
    typename index_t>
struct BatchNormTransformInputKernelFunctor {
  void operator()(sycl::nd_item<2> item) const {
    auto group_idx_x = item.get_group().get_group_id(1);
    index_t plane = group_idx_x;

    if (plane >= numPlane_) {
      return;
    }

    stat_accscalar_t gamma = weight_ptr_ != nullptr
        ? static_cast<stat_accscalar_t>(weight_ptr_[plane])
        : static_cast<stat_accscalar_t>(1);
    stat_accscalar_t beta = bias_ptr_ != nullptr
        ? static_cast<stat_accscalar_t>(bias_ptr_[plane])
        : static_cast<stat_accscalar_t>(0);

    stat_accscalar_t mean = static_cast<stat_accscalar_t>(mean_ptr_[plane]);
    stat_accscalar_t invstd;
    if constexpr (train) {
      invstd = var_or_invstd_ptr_[plane];
    } else {
      invstd = static_cast<stat_accscalar_t>(1) /
          std::sqrt(
                   static_cast<stat_accscalar_t>(var_or_invstd_ptr_[plane]) +
                   epsilon_);
    }

    index_t bstep = item.get_global_range(0);
    for (index_t batch = item.get_global_id(0); batch < bs_; batch += bstep) {
      auto batch_offset = batch * numPlane_ * fs_ + plane * fs_;
      for (index_t feature = item.get_local_id(1); feature < fs_;
           feature += item.get_local_range(1)) {
        output_ptr_[batch_offset + feature] = static_cast<input_scalar_t>(
            gamma * (input_ptr_[batch_offset + feature] - mean) * invstd +
            beta);
      }
    }
  }

  BatchNormTransformInputKernelFunctor(
      stat_accscalar_t epsilon,
      int numPlane,
      int64_t target_tile_size,
      int64_t wg_size,
      int bs,
      int fs,
      int weight_size,
      int bias_size,
      int tf,
      int tb,
      input_scalar_t* input_ptr,
      input_scalar_t* output_ptr,
      stat_scalar_t* weight_ptr,
      stat_scalar_t* bias_ptr,
      stat_accscalar_t* mean_ptr,
      stat_accscalar_t* var_or_invstd_ptr)
      : epsilon_(epsilon),
        numPlane_(numPlane),
        target_tile_size_(target_tile_size),
        wg_size_(wg_size),
        bs_(bs),
        fs_(fs),
        weight_size_(weight_size),
        bias_size_(bias_size),
        tf_(tf),
        tb_(tb),
        input_ptr_(input_ptr),
        output_ptr_(output_ptr),
        weight_ptr_(weight_ptr),
        bias_ptr_(bias_ptr),
        mean_ptr_(mean_ptr),
        var_or_invstd_ptr_(var_or_invstd_ptr) {}

 private:
  stat_accscalar_t epsilon_;
  int numPlane_;
  int64_t target_tile_size_;
  int64_t wg_size_;
  int bs_;
  int fs_;
  int weight_size_;
  int bias_size_;
  int tf_;
  int tb_;
  input_scalar_t* input_ptr_;
  input_scalar_t* output_ptr_;
  stat_scalar_t* weight_ptr_;
  stat_scalar_t* bias_ptr_;
  stat_accscalar_t* mean_ptr_;
  stat_accscalar_t* var_or_invstd_ptr_;
};

template <
    typename input_scalar_t,
    typename stat_scalar_t,
    typename stat_accscalar_t,
    bool train,
    typename index_t>
void batch_norm_transform_input_kernel(
    const Tensor input,
    Tensor& output,
    const Tensor& mean_,
    const Tensor& var_or_invstd,
    const Tensor& weight,
    const Tensor& bias,
    stat_accscalar_t epsilon) {
  auto& queue = getCurrentSYCLQueue();
  int numPlane = input.size(1);
  int64_t target_tile_size = syclMaxWorkItemsPerTile();
  int64_t wg_size = syclMaxWorkItemsPerEU(); // for work group barrier
  if (wg_size * numPlane < target_tile_size) {
    wg_size = syclMaxWorkGroupSize(); // for higher occupancy
  }

  int bs = input.size(0);
  int fs = input.size(2);
  int weight_size = weight.size(0);
  int bias_size = bias.size(0);

  int tf = get_num_threads(fs, wg_size);
  int tb = std::max<int>(wg_size / tf, 1);
  sycl::range<2> local_range(tb, tf);
  sycl::range<2> global_range((bs + tb - 1) / tb * tb, numPlane * tf);

  auto input_ptr = input.data_ptr<input_scalar_t>();
  auto output_ptr = output.data_ptr<input_scalar_t>();
  auto weight_ptr =
      weight.defined() ? weight.data_ptr<stat_scalar_t>() : nullptr;
  auto bias_ptr = bias.defined() ? bias.data_ptr<stat_scalar_t>() : nullptr;
  auto mean_ptr = mean_.data_ptr<stat_accscalar_t>();
  auto var_or_invstd_ptr = var_or_invstd.data_ptr<stat_accscalar_t>();

  auto caller = BatchNormTransformInputKernelFunctor<
      input_scalar_t,
      stat_scalar_t,
      stat_accscalar_t,
      train,
      index_t>(
      epsilon,
      numPlane,
      target_tile_size,
      wg_size,
      bs,
      fs,
      weight_size,
      bias_size,
      tf,
      tb,
      input_ptr,
      output_ptr,
      weight_ptr,
      bias_ptr,
      mean_ptr,
      var_or_invstd_ptr);

  sycl_kernel_submit(global_range, local_range, queue, caller);
}

template <typename input_scalar_t, typename stat_scalar_t, typename index_t>
void batch_norm_elemt_template(
    Tensor& output_,
    const Tensor& input_,
    const Tensor& weight_,
    const Tensor& bias_,
    const Tensor& mean_,
    const Tensor& invstd_) {
  using stat_accscalar_t = acc_type<stat_scalar_t, true>;
  auto input_reshaped = input_.reshape(
      {input_.size(0),
       input_.size(1),
       -1}); // internally we merge the feature dimensions
  auto output_reshaped = output_.view({input_.size(0), input_.size(1), -1});

  // NOTE: We use transform_input_kernel in training mode, which ignores
  // epsilon
  const double dummy_epsilon = 1e-5;

  batch_norm_transform_input_kernel<
      input_scalar_t,
      stat_scalar_t,
      stat_accscalar_t,
      true,
      index_t>(
      input_reshaped,
      output_reshaped,
      mean_,
      invstd_,
      weight_,
      bias_,
      dummy_epsilon);
}

template <typename scalar_t, typename acc_t>
struct BatchNormElementwiseLoopsFunctor {
  scalar_t operator()(
      scalar_t input,
      acc_t weight,
      acc_t bias,
      acc_t mean,
      acc_t invstd) const {
    return ((input - mean) * invstd) * weight + bias;
  }
};

template <
    typename scalar_t,
    typename accscalar_t,
    typename layerscalar_t,
    int vec_size,
    typename vec_t,
    typename vec_s_t>
struct BatchNormTransformInputChannelsLastKernelFunctor {
  void operator()(sycl::nd_item<2> item) const {
    // auto group_idx_x = item.get_group().get_group_id(1);

    // int inner_loop_stride = item.get_global_range(0);
    // offset along m dimension
    int m_offset = item.get_global_id(0);
    int c_offset_base = item.get_global_id(1) * vec_size;

    if (c_offset_base >= stride_ || m_offset >= reduction_size_) {
      return;
    }

    vec_s_t m_c = *(reinterpret_cast<vec_s_t*>(mean_ptr_ + c_offset_base));
    vec_s_t inv_vec =
        *(reinterpret_cast<vec_s_t*>(inv_std_ptr_ + c_offset_base));
    vec_s_t w_c;
    vec_s_t s_c;
#pragma unroll
    for (int j = 0; j < vec_size; j++) {
      if (weight_ptr_ != nullptr) {
        w_c[j] = static_cast<accscalar_t>(weight_ptr_[c_offset_base + j]) *
            inv_vec[j];
      } else {
        w_c[j] = (inv_vec[j]);
      }
      if (shift_ptr_ != nullptr) {
        s_c[j] = shift_ptr_[c_offset_base + j];
      } else {
        s_c[j] = static_cast<accscalar_t>(0.0f);
      }
    }

    int address_base = m_offset * stride_ + c_offset_base;
    int address_increment = item.get_global_range(0) * stride_;

    vec_t output_vec;
    for (; address_base < total_num_; address_base += address_increment) {
      vec_t x_math_vec = *(reinterpret_cast<vec_t*>(input_ptr_ + address_base));
#pragma unroll
      for (int j = 0; j < vec_size; j++) {
        // auto c_offset = c_offset_base + j;

        output_vec[j] =
            w_c[j] * (static_cast<accscalar_t>(x_math_vec[j]) - m_c[j]) +
            s_c[j];
      }
      *(reinterpret_cast<vec_t*>(output_ptr_ + address_base)) = output_vec;
    }
  }
  BatchNormTransformInputChannelsLastKernelFunctor(
      scalar_t* input_ptr,
      const scalar_t* z_ptr,
      accscalar_t* mean_ptr,
      accscalar_t* inv_std_ptr,
      const layerscalar_t* weight_ptr,
      const layerscalar_t* shift_ptr,
      scalar_t* output_ptr,
      const int reduction_size,
      const int stride,
      const bool fuse_relu,
      int64_t total_num)
      : input_ptr_(input_ptr),
        z_ptr_(z_ptr),
        mean_ptr_(mean_ptr),
        inv_std_ptr_(inv_std_ptr),
        weight_ptr_(weight_ptr),
        shift_ptr_(shift_ptr),
        output_ptr_(output_ptr),
        reduction_size_(reduction_size),
        stride_(stride),
        fuse_relu_(fuse_relu),
        total_num_(total_num) {}

 private:
  scalar_t* input_ptr_;
  const scalar_t* z_ptr_;
  accscalar_t* mean_ptr_;
  accscalar_t* inv_std_ptr_;
  const layerscalar_t* weight_ptr_;
  const layerscalar_t* shift_ptr_;
  scalar_t* output_ptr_;
  const int reduction_size_;
  const int stride_;
  const bool fuse_relu_;
  int64_t total_num_;
};

template <
    typename scalar_t,
    typename accscalar_t,
    typename layerscalar_t,
    int vec_size>
void batch_norm_transform_input_channels_last_kernel(
    scalar_t* input_ptr,
    const scalar_t* z_ptr,
    accscalar_t* mean_ptr,
    accscalar_t* inv_std_ptr,
    const layerscalar_t* weight_ptr,
    const layerscalar_t* shift_ptr,
    scalar_t* output_ptr,
    const int reduction_size,
    const int stride,
    const bool fuse_relu) {
  // tensor dimension (m,c)
  // loop along m dimension
  int64_t total_num = reduction_size * stride;
  using vec_t = memory::aligned_vector<scalar_t, vec_size>;
  using vec_s_t = memory::aligned_vector<accscalar_t, vec_size>;
  auto& queue = getCurrentSYCLQueue();
  sycl::range<2> global_range(1, 1), local_range(1, 1);
  std::tie(global_range, local_range) =
      flexible_launch_configs(reduction_size, stride / vec_size);

  auto caller = BatchNormTransformInputChannelsLastKernelFunctor<
      scalar_t,
      accscalar_t,
      layerscalar_t,
      vec_size,
      vec_t,
      vec_s_t>(
      input_ptr,
      z_ptr,
      mean_ptr,
      inv_std_ptr,
      weight_ptr,
      shift_ptr,
      output_ptr,
      reduction_size,
      stride,
      fuse_relu,
      total_num);

  sycl_kernel_submit(global_range, local_range, queue, caller);
}

void batch_norm_elemt_channels_last_template(
    Tensor& output,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& shift, // bias of BN
    const at::Tensor& mean,
    const at::Tensor& inv_std,
    const at::optional<at::Tensor>& z = c10::nullopt, // bias after BN
    const bool fuse_relu = false) {
  const auto second_dtype = weight.defined()
      ? weight.scalar_type()
      : (shift.defined() ? shift.scalar_type() : input.scalar_type());
  const auto stride = input.sizes()[1];
  const auto reduction_size = input.numel() / stride;

#define DISPATCH_TRANSFORM_INPUT_IMPL(vec_size)                   \
  {                                                               \
    batch_norm_transform_input_channels_last_kernel<              \
        scalar_t,                                                 \
        accscalar_t,                                              \
        scalar_t,                                                 \
        vec_size>(                                                \
        input.data_ptr<scalar_t>(),                               \
        z.has_value() ? z.value().data_ptr<scalar_t>() : nullptr, \
        mean.data_ptr<accscalar_t>(),                             \
        inv_std.data_ptr<accscalar_t>(),                          \
        weight.defined() ? weight.data_ptr<scalar_t>() : nullptr, \
        shift.defined() ? shift.data_ptr<scalar_t>() : nullptr,   \
        output.data_ptr<scalar_t>(),                              \
        reduction_size,                                           \
        stride,                                                   \
        fuse_relu);                                               \
  }

#define DISPATCH_TRANSFORM_ACC_INPUT_IMPL(vec_size)                  \
  {                                                                  \
    batch_norm_transform_input_channels_last_kernel<                 \
        scalar_t,                                                    \
        accscalar_t,                                                 \
        accscalar_t,                                                 \
        vec_size>(                                                   \
        input.data_ptr<scalar_t>(),                                  \
        z.has_value() ? z.value().data_ptr<scalar_t>() : nullptr,    \
        mean.data_ptr<accscalar_t>(),                                \
        inv_std.data_ptr<accscalar_t>(),                             \
        weight.defined() ? weight.data_ptr<accscalar_t>() : nullptr, \
        shift.defined() ? shift.data_ptr<accscalar_t>() : nullptr,   \
        output.data_ptr<scalar_t>(),                                 \
        reduction_size,                                              \
        stride,                                                      \
        fuse_relu);                                                  \
  }

  if (input.scalar_type() != second_dtype) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf, kBFloat16, input.scalar_type(), "batchnorm_forward", [&] {
          using accscalar_t = acc_type<scalar_t, true>;
          int suggest_vec_size = get_nhwc_suggest_vec_size<scalar_t>(
              input, reduction_size, stride);
          switch (suggest_vec_size) {
            case 8: {
              DISPATCH_TRANSFORM_ACC_INPUT_IMPL(8);
              break;
            }
            case 4: {
              DISPATCH_TRANSFORM_ACC_INPUT_IMPL(4);
              break;
            }
            default:
              DISPATCH_TRANSFORM_ACC_INPUT_IMPL(1);
          }
        });
  } else {
    if (weight.defined()) {
      TORCH_CHECK(
          input.scalar_type() == weight.scalar_type(),
          "batchnorm_forward: input.scalar_type() ",
          input.scalar_type(),
          " is not supported with weight.scalar_type() ",
          weight.scalar_type());
    }
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf, kBFloat16, input.scalar_type(), "batchnorm_forward", [&] {
          using accscalar_t = acc_type<scalar_t, true>;
          int suggest_vec_size = get_nhwc_suggest_vec_size<scalar_t>(
              input, reduction_size, stride);
          switch (suggest_vec_size) {
            case 8: {
              DISPATCH_TRANSFORM_INPUT_IMPL(8);
              break;
            }
            case 4: {
              DISPATCH_TRANSFORM_INPUT_IMPL(4);
              break;
            }
            default:
              DISPATCH_TRANSFORM_INPUT_IMPL(1);
          }
        });
  }
#undef DISPATCH_TRANSFORM_INPUT_IMPL
#undef DISPATCH_TRANSFORM_ACC_INPUT_IMPL
}

void batch_norm_elemt_kernel(
    Tensor& out,
    const Tensor& self,
    const c10::optional<Tensor>& weight_opt,
    const c10::optional<Tensor>& bias_opt,
    const Tensor& mean_,
    const Tensor& invstd_) {
  switch (batch_norm_choose_impl(self)) {
    case Impl::Contiguous: {
      c10::MaybeOwned<Tensor> weight =
          at::borrow_from_optional_tensor(weight_opt);
      c10::MaybeOwned<Tensor> bias = at::borrow_from_optional_tensor(bias_opt);
      at::native::resize_output(out, self.sizes());
      AT_DISPATCH_FLOATING_TYPES_AND2(
          kBFloat16,
          kHalf,
          self.scalar_type(),
          "batch_norm_elementwise_xpu",
          [&] {
            using accscalar_t = acc_type<scalar_t, true>;
            const bool mixed_type = is_mixed_type(self, *weight, *bias);
            if (mixed_type) {
              batch_norm_elemt_template<scalar_t, accscalar_t, int32_t>(
                  out, self, *weight, *bias, mean_, invstd_);
            } else {
              batch_norm_elemt_template<scalar_t, scalar_t, int32_t>(
                  out, self, *weight, *bias, mean_, invstd_);
            }
          });
      return;
    }
    case Impl::ChannelsLast: {
      auto weight = at::borrow_from_optional_tensor(weight_opt);
      auto bias = at::borrow_from_optional_tensor(bias_opt);

      if (resize_output_check(out, self.sizes())) {
        resize_impl_xpu_(
            out.unsafeGetTensorImpl(), self.sizes(), self.strides());
      }
      if ((out.strides() == self.strides()) &&
          (!weight->defined() || weight->is_contiguous()) &&
          (!bias->defined() || bias->is_contiguous()) &&
          (!mean_.defined() || mean_.is_contiguous()) &&
          (!invstd_.defined() || invstd_.is_contiguous())) {
        batch_norm_elemt_channels_last_template(
            out, self, *weight, *bias, mean_, invstd_);
        return;
      }
      [[fallthrough]];
    }
    case Impl::General: {
      const int64_t ndim = self.dim();
      DimVector sizes(ndim, 1), strides(ndim, 0);
      // Helper to convert 1d tensors to an nd tensor that broadcasts with
      // input All elements go into the channel dimension
      auto as_nd = [&](const Tensor& t) {
        TORCH_INTERNAL_ASSERT(t.defined() && t.dim() == 1);
        sizes[1] = t.sizes()[0];
        strides[1] = t.strides()[0];
        return t.as_strided(sizes, strides);
      };

      auto weight = weight_opt.has_value() && weight_opt->defined()
          ? as_nd(*weight_opt)
          : at::scalar_tensor(1, mean_.options());
      auto bias = bias_opt.has_value() && bias_opt->defined()
          ? as_nd(*bias_opt)
          : at::scalar_tensor(0, mean_.options());
      auto mean = as_nd(mean_);
      auto invstd = as_nd(invstd_);

      auto iter = TensorIteratorConfig()
                      .add_output(out)
                      .add_input(self)
                      .add_input(weight)
                      .add_input(bias)
                      .add_input(mean)
                      .add_input(invstd)
                      .check_all_same_dtype(false)
                      .promote_inputs_to_common_dtype(false)
                      .build();

      AT_DISPATCH_FLOATING_TYPES_AND2(
          kBFloat16,
          kHalf,
          self.scalar_type(),
          "batch_norm_elementwise_xpu",
          [&] {
            using acc_t = acc_type<scalar_t, true>;
            auto f = BatchNormElementwiseLoopsFunctor<scalar_t, acc_t>();
            gpu_kernel(iter, f);
          });
      return;
    }
  }
}

// ========================== batch_norm_gather_stats ==========================

template <typename scalar_t, typename accscalar_t, typename index_t>
struct BatchNormGatherStatsReductionKernelFunctor
    : public __SYCL_KER_CONFIG_CONVENTION__ {
  void operator()(sycl::nd_item<2> item) const {
    int global_row = item.get_global_id(0);
    int global_col = item.get_global_id(1);
    int group_row = item.get_group(0);
    // int group_col = item.get_group(1);
    int local_row = item.get_local_id(0);
    int local_col = item.get_local_id(1);

    int global_len = len_;
    int local_len = len_;

    if (global_row < global_len && global_col < feature_sz_) {
      save_mean_local_mem_[local_row][local_col] =
          save_mean_in_[global_row * feature_sz_ + global_col];
      counts_local_mem_[local_row][local_col] = counts_in_[global_row];
      accscalar_t v =
          1.0f / save_invstd_in_[global_row * feature_sz_ + global_col];
      var_n_local_mem_[local_row][local_col] =
          (v * v - epsilon_) * counts_local_mem_[local_row][local_col];
    }

    // Do a tree reduction on work-items in work-group
    for (int i = wgroup_size_batch_dim_ / 2; i > 0; i >>= 1) {
      item.barrier(sycl::access::fence_space::local_space);
      if (local_row < i && global_row < global_len &&
          global_row + i < global_len && local_row < local_len &&
          local_row + i < local_len && global_col < feature_sz_) {
        index_t n_1 = counts_local_mem_[local_row + i][local_col];
        index_t n_0 = counts_local_mem_[local_row][local_col];
        accscalar_t m_1 = save_mean_local_mem_[local_row + i][local_col];
        accscalar_t m_0 = save_mean_local_mem_[local_row][local_col];
        accscalar_t v_1 = var_n_local_mem_[local_row + i][local_col];
        accscalar_t v = std::sqrt(v_1 / n_1 + epsilon_);
        v = (v * v - epsilon_) * n_1;
        accscalar_t factor = 1.0f / (n_0 + n_1);

        var_n_local_mem_[local_row][local_col] +=
            v + (m_0 - m_1) * (m_0 - m_1) * n_0 * n_1 * factor;
        save_mean_local_mem_[local_row][local_col] =
            n_0 * factor * m_0 + n_1 * factor * m_1;
        counts_local_mem_[local_row][local_col] += n_1;
      }
      local_len = i;
      i = i + (i % 2 && i != 1);
    }

    if (local_row == 0 && global_col < feature_sz_) {
      save_mean_out_[group_row * feature_sz_ + global_col] =
          save_mean_local_mem_[0][local_col];
      save_invstd_out_[group_row * feature_sz_ + global_col] =
          static_cast<accscalar_t>(1.0f) /
          std::sqrt(
              var_n_local_mem_[0][local_col] / counts_local_mem_[0][local_col] +
              epsilon_);
      counts_out_[group_row] = counts_local_mem_[0][0];
    }

    if (n_wgroups_batch_dim_ == 1 && local_row == 0) {
      if (running_mean_ != nullptr) {
        running_mean_[global_col] = static_cast<scalar_t>(
            (1 - momentum_) * running_mean_[global_col] +
            momentum_ * save_mean_local_mem_[0][global_col]);
      }
      if (running_var_ != nullptr) {
        running_var_[global_col] = static_cast<scalar_t>(
            (1 - momentum_) * running_var_[global_col] +
            momentum_ *
                (var_n_local_mem_[0][global_col] /
                 (counts_local_mem_[0][global_col] - 1)));
      }
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    save_mean_local_mem_ = sycl_local_acc_t<accscalar_t, 2>(
        sycl::range<2>{
            (size_t)wgroup_size_batch_dim_, (size_t)wgroup_size_feature_dim_},
        cgh);
    var_n_local_mem_ = sycl_local_acc_t<accscalar_t, 2>(
        sycl::range<2>{
            (size_t)wgroup_size_batch_dim_, (size_t)wgroup_size_feature_dim_},
        cgh);
    counts_local_mem_ = sycl_local_acc_t<accscalar_t, 2>(
        sycl::range<2>{
            (size_t)wgroup_size_batch_dim_, (size_t)wgroup_size_feature_dim_},
        cgh);
  }

  BatchNormGatherStatsReductionKernelFunctor(
      accscalar_t* save_mean_in,
      accscalar_t* save_mean_out,
      accscalar_t* save_invstd_in,
      accscalar_t* save_invstd_out,
      scalar_t* counts_in,
      scalar_t* counts_out,
      scalar_t* running_mean,
      scalar_t* running_var,
      int len,
      int n_wgroups_batch_dim,
      int wgroup_size_batch_dim,
      int n_wgroups_feature_dim,
      int wgroup_size_feature_dim,
      int feature_sz,
      float momentum,
      float epsilon)
      : save_mean_in_(save_mean_in),
        save_mean_out_(save_mean_out),
        save_invstd_in_(save_invstd_in),
        save_invstd_out_(save_invstd_out),
        counts_in_(counts_in),
        counts_out_(counts_out),
        running_mean_(running_mean),
        running_var_(running_var),
        len_(len),
        n_wgroups_batch_dim_(n_wgroups_batch_dim),
        wgroup_size_batch_dim_(wgroup_size_batch_dim),
        n_wgroups_feature_dim_(n_wgroups_feature_dim),
        wgroup_size_feature_dim_(wgroup_size_feature_dim),
        feature_sz_(feature_sz),
        momentum_(momentum),
        epsilon_(epsilon) {}

 private:
  accscalar_t* save_mean_in_;
  accscalar_t* save_mean_out_;
  accscalar_t* save_invstd_in_;
  accscalar_t* save_invstd_out_;
  scalar_t* counts_in_;
  scalar_t* counts_out_;
  scalar_t* running_mean_;
  scalar_t* running_var_;
  int len_;
  int n_wgroups_batch_dim_;
  int wgroup_size_batch_dim_;
  int n_wgroups_feature_dim_;
  int wgroup_size_feature_dim_;
  int feature_sz_;
  float momentum_;
  float epsilon_;
  sycl_local_acc_t<accscalar_t, 2> save_mean_local_mem_;
  sycl_local_acc_t<accscalar_t, 2> var_n_local_mem_;
  sycl_local_acc_t<accscalar_t, 2> counts_local_mem_;
};

// instead of having one thread (work-item) reduce one column,
// split the column into chunks (work-groups) for each thread to
// parallely compute a partial result for each chunk
// until the number of chunks is 1
template <typename scalar_t, typename accscalar_t, typename index_t>
void batch_norm_gather_stats_reduction(
    sycl::queue& q,
    accscalar_t* save_mean_in,
    accscalar_t* save_mean_out,
    accscalar_t* save_invstd_in,
    accscalar_t* save_invstd_out,
    scalar_t* counts_in,
    scalar_t* counts_out,
    scalar_t* running_mean,
    scalar_t* running_var,
    int len,
    int n_wgroups_batch_dim,
    int wgroup_size_batch_dim,
    int n_wgroups_feature_dim,
    int wgroup_size_feature_dim,
    int feature_sz,
    float momentum_,
    float epsilon_) {
  sycl::range<2> global_range{
      (size_t)n_wgroups_batch_dim * wgroup_size_batch_dim,
      (size_t)n_wgroups_feature_dim * wgroup_size_feature_dim};
  sycl::range<2> local_range{
      (size_t)wgroup_size_batch_dim, (size_t)wgroup_size_feature_dim};

  auto caller = BatchNormGatherStatsReductionKernelFunctor<
      scalar_t,
      accscalar_t,
      index_t>(
      save_mean_in,
      save_mean_out,
      save_invstd_in,
      save_invstd_out,
      counts_in,
      counts_out,
      running_mean,
      running_var,
      len,
      n_wgroups_batch_dim,
      wgroup_size_batch_dim,
      n_wgroups_feature_dim,
      wgroup_size_feature_dim,
      feature_sz,
      momentum_,
      epsilon_);

  sycl_kernel_submit(global_range, local_range, q, caller);
}

template <typename scalar_t, typename accscalar_t, typename index_t>
std::tuple<Tensor, Tensor> batch_norm_gather_stats_xpu_template(
    const Tensor& mean_,
    const Tensor& invstd_,
    const Tensor& running_mean_,
    const Tensor& running_var_,
    double momentum,
    double epsilon,
    const Tensor& counts_) {
  int feature_sz = mean_.size(1);
  int batch_sz = mean_.size(0);

  auto input_options = mean_.options();
  if (mean_.scalar_type() == at::ScalarType::Half ||
      mean_.scalar_type() == at::ScalarType::BFloat16) {
    input_options = input_options.dtype(ScalarType::Float);
  }

  auto counts_options = counts_.options();
  if (counts_.scalar_type() == at::ScalarType::Half ||
      counts_.scalar_type() == at::ScalarType::BFloat16) {
    counts_options = counts_options.dtype(ScalarType::Float);
  }

  auto running_mean =
      running_mean_.defined() ? running_mean_.data_ptr<scalar_t>() : nullptr;
  auto running_var =
      running_var_.defined() ? running_var_.data_ptr<scalar_t>() : nullptr;

  // // Avoid double issues in ATSM
  // float momentum_ = momentum;
  // float epsilon_ = epsilon;
  auto& queue = getCurrentSYCLQueue();
  int max_work_items_per_eu = syclMaxWorkItemsPerEU();

  int wgroup_size_feature_dim = std::min(feature_sz, 32);
  int n_wgroups_feature_dim =
      (feature_sz + wgroup_size_feature_dim - 1) / wgroup_size_feature_dim;

  int len = batch_sz;

  accscalar_t* save_mean_prev = mean_.data_ptr<accscalar_t>();
  accscalar_t* save_invstd_prev = invstd_.data_ptr<accscalar_t>();
  scalar_t* counts_prev = counts_.data_ptr<scalar_t>();

  Tensor save_mean_tmp;
  Tensor save_invstd_tmp;
  Tensor counts_tmp;

  if (len == 1) {
    int wgroup_size_batch_dim = 2;
    int n_wgroups_batch_dim = 1;

    save_mean_tmp = at::empty({feature_sz}, input_options);
    accscalar_t* save_mean_curr = save_mean_tmp.data_ptr<accscalar_t>();

    save_invstd_tmp = at::empty({feature_sz}, input_options);
    accscalar_t* save_invstd_curr = save_invstd_tmp.data_ptr<accscalar_t>();

    counts_tmp = at::empty({1}, counts_options);
    scalar_t* counts_curr = counts_tmp.data_ptr<scalar_t>();

    batch_norm_gather_stats_reduction<scalar_t, accscalar_t, index_t>(
        queue,
        save_mean_prev,
        save_mean_curr,
        save_invstd_prev,
        save_invstd_curr,
        counts_prev,
        counts_curr,
        running_mean,
        running_var,
        len,
        n_wgroups_batch_dim,
        wgroup_size_batch_dim,
        n_wgroups_feature_dim,
        wgroup_size_feature_dim,
        feature_sz,
        (float)momentum,
        (float)epsilon);

    return std::make_tuple(save_mean_tmp, save_invstd_tmp);
  }

  while (len != 1) {
    int wgroup_size = std::min(max_work_items_per_eu, len + (len % 2));
    int wgroup_size_batch_dim = (wgroup_size_feature_dim == 1)
        ? wgroup_size
        : ((wgroup_size * wgroup_size_feature_dim <= max_work_items_per_eu)
               ? wgroup_size
               : max_work_items_per_eu /
                   (wgroup_size_feature_dim + (wgroup_size_feature_dim % 2)));
    wgroup_size_batch_dim = wgroup_size_batch_dim + (wgroup_size_batch_dim % 2);
    int n_wgroups_batch_dim =
        len / wgroup_size_batch_dim + (len % wgroup_size_batch_dim != 0);

    save_mean_tmp =
        at::empty({n_wgroups_batch_dim * feature_sz}, input_options);
    accscalar_t* save_mean_curr = save_mean_tmp.data_ptr<accscalar_t>();

    save_invstd_tmp =
        at::empty({n_wgroups_batch_dim * feature_sz}, input_options);
    accscalar_t* save_invstd_curr = save_invstd_tmp.data_ptr<accscalar_t>();

    counts_tmp = at::empty({n_wgroups_batch_dim}, counts_options);
    scalar_t* counts_curr = counts_tmp.data_ptr<scalar_t>();

    batch_norm_gather_stats_reduction<scalar_t, accscalar_t, index_t>(
        queue,
        save_mean_prev,
        save_mean_curr,
        save_invstd_prev,
        save_invstd_curr,
        counts_prev,
        counts_curr,
        running_mean,
        running_var,
        len,
        n_wgroups_batch_dim,
        wgroup_size_batch_dim,
        n_wgroups_feature_dim,
        wgroup_size_feature_dim,
        feature_sz,
        (float)momentum,
        (float)epsilon);

    save_mean_prev = save_mean_curr;
    save_invstd_prev = save_invstd_curr;
    counts_prev = counts_curr;

    len = n_wgroups_batch_dim;
  }

  return std::make_tuple(save_mean_tmp, save_invstd_tmp);
}

std::tuple<Tensor, Tensor> batch_norm_gather_stats_with_counts(
    const Tensor& self,
    const Tensor& mean,
    const Tensor& invstd,
    const Tensor& running_mean /* optional */,
    const Tensor& running_var /* optional */,
    double momentum,
    double epsilon,
    const Tensor& counts) {
  auto scalar_type =
      running_mean.defined() ? running_mean.scalar_type() : self.scalar_type();
  return AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      scalar_type,
      "batch_norm_update_stats_xpu",
      [&] {
        using accscalar_t = acc_type<scalar_t, true>;
        if (canUse32BitIndexMath(self)) {
          return batch_norm_gather_stats_xpu_template<
              scalar_t,
              accscalar_t,
              int32_t>(
              mean,
              invstd,
              running_mean,
              running_var,
              momentum,
              epsilon,
              counts);
        } else {
          return batch_norm_gather_stats_xpu_template<
              scalar_t,
              accscalar_t,
              int64_t>(
              mean,
              invstd,
              running_mean,
              running_var,
              momentum,
              epsilon,
              counts);
        }
      });
}

// accepting input(self) here to determine template data types, since
// running_mean/running_var are optional
std::tuple<Tensor, Tensor> batch_norm_gather_stats_kernel(
    const Tensor& self,
    const Tensor& mean,
    const Tensor& invstd,
    const c10::optional<Tensor>& running_mean_opt,
    const c10::optional<Tensor>& running_var_opt,
    double momentum,
    double epsilon,
    int64_t count) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> running_mean_maybe_owned =
      at::borrow_from_optional_tensor(running_mean_opt);
  const Tensor& running_mean = *running_mean_maybe_owned;
  const Tensor& running_var =
      c10::value_or_else(running_var_opt, [] { return Tensor(); });

  Tensor counts_ = at::empty(
      mean.size(0),
      self.options().dtype(
          running_mean.defined() ? running_mean.dtype() : self.dtype()));
  counts_.fill_(count);
  return batch_norm_gather_stats_with_counts(
      self,
      mean,
      invstd,
      running_mean,
      running_var,
      momentum,
      epsilon,
      counts_);
}

std::tuple<Tensor, Tensor> batch_norm_gather_stats_with_counts_kernel(
    const Tensor& self,
    const Tensor& mean,
    const Tensor& invstd,
    const c10::optional<Tensor>& running_mean_opt /* optional */,
    const c10::optional<Tensor>& running_var_opt /* optional */,
    double momentum,
    double epsilon,
    const Tensor& counts) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> running_mean_maybe_owned =
      at::borrow_from_optional_tensor(running_mean_opt);
  const Tensor& running_mean = *running_mean_maybe_owned;
  const Tensor& running_var =
      c10::value_or_else(running_var_opt, [] { return Tensor(); });

  return batch_norm_gather_stats_with_counts(
      self, mean, invstd, running_mean, running_var, momentum, epsilon, counts);
}

} // namespace xpu
} // namespace native
} // namespace at
