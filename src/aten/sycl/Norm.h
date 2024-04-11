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

using namespace at::native::memory;
using namespace at::xpu;

constexpr int SIMD = 16;

template <
    typename accscalar_t,
    typename reduce_op,
    typename nd_item_id,
    typename local_shared>
static inline void norm_group_reduce(
    nd_item_id item_id,
    int sub_group_num,
    accscalar_t& mean,
    accscalar_t& rstd,
    const local_shared& local_mean,
    const local_shared& local_rstd,
    reduce_op bin_op) {
  auto sg = item_id.get_sub_group();

  // dynamic get SIMD width result in big performance drop
  // uint32_t SIMD = sg.get_local_range()[0];
#pragma unroll
  for (int i = 1; i < SIMD; i <<= 1) {
    mean = bin_op(mean, static_cast<accscalar_t>(sg.shuffle_down(mean, i)));
    rstd = bin_op(rstd, static_cast<accscalar_t>(sg.shuffle_down(rstd, i)));
  }
  if (sub_group_num == 1) {
    mean = sycl::group_broadcast(sg, mean, 0);
    rstd = sycl::group_broadcast(sg, rstd, 0);
    return;
  }

  uint32_t sg_local_id = sg.get_local_linear_id();
  uint32_t sg_id = sg.get_group_linear_id();
  // reduce internal each subgroup, each subgroup will generate one result
  // there are WGroupSize/subGroupSize elements after this step
  int idx = sg_id;
  if (sg_local_id == 0) {
    local_mean[sg_id] = mean;
    local_rstd[sg_id] = rstd;
  }
  item_id.barrier(sycl_local_fence);

  // use one subgroup to reduce WGroupSize/subGroupSize elements
  // into the final result
  if (idx == 0) {
    mean = 0;
    rstd = 0;
    if ((int)sg_local_id < sub_group_num) {
      mean = accscalar_t(local_mean[sg_local_id]);
      rstd = accscalar_t(local_rstd[sg_local_id]);
    }
    for (int i = sg_local_id + SIMD; i < sub_group_num; i += SIMD) {
      mean = bin_op(mean, static_cast<accscalar_t>(local_mean[i]));
      rstd = bin_op(rstd, static_cast<accscalar_t>(local_rstd[i]));
    }
#pragma unroll
    for (int i = 1; i < SIMD; i <<= 1) {
      mean = bin_op(mean, static_cast<accscalar_t>(sg.shuffle_down(mean, i)));
      rstd = bin_op(rstd, static_cast<accscalar_t>(sg.shuffle_down(rstd, i)));
      if (i >= ((sub_group_num + 1) >> 1))
        break;
    }

    // the 0th WI (the 0th WI in the 0th sub_group) generate the final result
    if (sg_local_id == 0) {
      local_mean[0] = mean;
      local_rstd[0] = rstd;
    }
  }
  item_id.barrier(sycl_local_fence);

  mean = local_mean[0];
  rstd = local_rstd[0];
}

template <
    typename accscalar_t,
    typename index_t,
    bool one_moment,
    typename reduce_op,
    typename nd_item_id,
    typename local_shared,
    typename local_shared_bool>
static void norm_global_reduce(
    nd_item_id item_id,
    int workgroup_num_foreach,
    int workgroup_size,
    int sub_group_num,
    accscalar_t& sum1,
    accscalar_t& sum2,
    accscalar_t* scratchpad_ptr,
    int* semaphores_ptr,
    const local_shared& local_mean,
    const local_shared& local_rstd,
    const local_shared_bool& last_workgroup,
    reduce_op bin_op) {
  index_t local_id = item_id.get_local_id(2);
  index_t group_id = item_id.get_group(0);
  index_t group_id_foreach = item_id.get_group(1);

  if (local_id == 0) {
    if constexpr (one_moment) {
      auto idx = group_id * workgroup_num_foreach + group_id_foreach;
      scratchpad_ptr[idx] = sum1;
    } else {
      auto idx = group_id * workgroup_num_foreach * 2 + group_id_foreach;
      scratchpad_ptr[idx] = sum1;
      scratchpad_ptr[workgroup_num_foreach + idx] = sum2;
    }
  }
  item_id.barrier(sycl_global_fence);

  if (local_id == 0) {
    sycl_atomic_ref_rlx_dev_global_t<int> count(semaphores_ptr[group_id]);
    int prev_groups_finished = count.fetch_add(1);
    last_workgroup[0] = (prev_groups_finished == workgroup_num_foreach - 1);
  }
  item_id.barrier(sycl_local_fence);

  // use the last workgroup for reduction
  if (last_workgroup[0]) {
    if constexpr (one_moment) {
      sum1 = accscalar_t(0);
      for (int i = local_id; i < workgroup_num_foreach; i += workgroup_size) {
        auto idx = group_id * workgroup_num_foreach + i;
        sum1 = bin_op(sum1, scratchpad_ptr[idx]);
      }
      sum1 = sycl::reduce_over_group(
          item_id.get_group(), sum1, sycl::plus<accscalar_t>());
    } else {
      sum1 = accscalar_t(0);
      sum2 = accscalar_t(0);
      for (int i = local_id; i < workgroup_num_foreach; i += workgroup_size) {
        auto idx = group_id * workgroup_num_foreach * 2 + i;
        sum1 = bin_op(sum1, scratchpad_ptr[idx]);
        sum2 = bin_op(sum2, scratchpad_ptr[workgroup_num_foreach + idx]);
      }
      norm_group_reduce<accscalar_t>(
          item_id, sub_group_num, sum1, sum2, local_mean, local_rstd, bin_op);
    }
  }
}

template <
    int vec_size,
    typename accscalar_t,
    typename reduce_op,
    typename nd_item_id,
    typename local_shared>
static inline void norm_group_reduce_row(
    nd_item_id item_id,
    accscalar_t input1[vec_size],
    accscalar_t input2[vec_size],
    const local_shared& local_data1,
    const local_shared& local_data2,
    int block_row,
    reduce_op bin_op) {
  auto local_row_id = item_id.get_local_id(1);
  auto local_col_id = item_id.get_local_id(2);

#pragma unroll(vec_size)
  for (int j = 0; j < vec_size; ++j) {
    local_data1[local_row_id][local_col_id][j] = input1[j];
    local_data2[local_row_id][local_col_id][j] = input2[j];
  }
  item_id.barrier(sycl_local_fence);

  int k = 1;
  while (k < block_row) {
    if (local_row_id % (k << 1) == 0 && local_row_id + k < block_row)
#pragma unroll(vec_size)
      for (int j = 0; j < vec_size; ++j) {
        local_data1[local_row_id][local_col_id][j] = bin_op(
            local_data1[local_row_id][local_col_id][j],
            local_data1[local_row_id + k][local_col_id][j]);
        local_data2[local_row_id][local_col_id][j] = bin_op(
            local_data2[local_row_id][local_col_id][j],
            local_data2[local_row_id + k][local_col_id][j]);
      }
    k *= 2;
    item_id.barrier(sycl_local_fence);
  }
}

class NormConfig {
 public:
  NormConfig(int Batch, int Plane, int problem_dim, int element_size_bytes)
      : Batch(Batch),
        Plane(Plane),
        problem_dim(problem_dim),
        element_size_bytes(element_size_bytes) {
    semaphores_ptr = nullptr;
    scratchpad_ptr = nullptr;
    sub_group_num_global = 1;

    get_max_vec_size();
    if (problem_dim == 1) {
      get_workgroup_size();
      WGPlane = (Plane + workgroup_num_foreach - 1) / workgroup_num_foreach;
    } else {
      get_workgroup_size_row();
    }
  }

  int Batch;
  int Plane;
  int WGPlane;
  int problem_dim;
  int element_size_bytes;
  int max_vec_size;

  int block_row;
  int workgroup_num;
  int workgroup_num_foreach;
  int workgroup_size;
  int sub_group_num;

  int* semaphores_ptr;
  void* scratchpad_ptr;
  int sub_group_num_global;

  template <typename scalar_t>
  void init_global_reduce(
      const Tensor& X,
      Tensor& semaphores,
      Tensor& scratchpad) {
    if (workgroup_num_foreach > 1) {
      int semaphores_size = workgroup_num;
      semaphores = at::zeros(semaphores_size, X.options().dtype(kInt));
      const auto kAccType =
          (X.scalar_type() == kHalf || X.scalar_type() == kBFloat16)
          ? kFloat
          : X.scalar_type();
      int scratchpad_size =
          2 * Batch * workgroup_num_foreach * sizeof(acc_type<scalar_t, true>);
      scratchpad = at::zeros(scratchpad_size, X.options().dtype(kAccType));
      semaphores_ptr = semaphores.data_ptr<int>();
      scratchpad_ptr = scratchpad.data_ptr();
      sub_group_num_global = (workgroup_num_foreach + SIMD - 1) / SIMD;
    }
  }

  void get_max_vec_size() {
    auto dev_id = getDeviceIndexOfCurrentQueue();
    int total_resource = syclMaxWorkItemsPerTile(dev_id);

    constexpr int float4_size = sizeof(float) * 4;
    max_vec_size = float4_size / element_size_bytes;
    while ((max_vec_size >> 1) * total_resource >= (Batch * Plane) &&
           (max_vec_size >> 1) >= 1) {
      max_vec_size = max_vec_size >> 1;
    }
  }

  // get resource size for Reduce problem [Batch, Plane]
  // the reduce is performed on Plane dimension
  void get_workgroup_size() {
    auto dev_id = getDeviceIndexOfCurrentQueue();
    int max_workgroup_size = syclMaxWorkGroupSize(dev_id);
    int total_resource = syclMaxWorkItemsPerTile(dev_id);
    workgroup_num = total_resource / max_workgroup_size;
    int max_workgroup_num_foreach = 1;
    workgroup_size = max_workgroup_size;

    // To keep high occupancy, we should activate at least workgroup_num number
    // of WG if Batch is larger than workgroup_num, use only one WG to process
    // Plane elements if Batch is smaller than workgroup_num, use
    // workgroup_num_foreach to process Plan elements
    while (workgroup_num > Batch) {
      workgroup_num = workgroup_num >> 1;
      max_workgroup_num_foreach = max_workgroup_num_foreach << 1;
    }
    workgroup_num_foreach = (Plane + workgroup_size * max_vec_size - 1) /
        (workgroup_size * max_vec_size);
    workgroup_num_foreach =
        std::min(workgroup_num_foreach, max_workgroup_num_foreach);
    // Reduce will waste the EU resource, then
    // minimize the workgroup_size and maximize the workgroup_num
    while (workgroup_num << 1 <= Batch && (workgroup_size >> 1) >= SIMD) {
      workgroup_num = workgroup_num << 1;
      workgroup_size = workgroup_size >> 1;
    }

    // Workgroup_num should larger or equal to Batch
    workgroup_num = std::max(workgroup_num, int(Batch));
    // At least one subgroup for reduce
    sub_group_num = (workgroup_size + SIMD - 1) / SIMD;
  }

  void get_workgroup_size_row() {
    // enlarge the occupancy, compute the least workgroup_num
    auto dev_id = getDeviceIndexOfCurrentQueue();
    int max_workgroup_size = syclMaxWorkGroupSize(dev_id);
    int total_resource = syclMaxWorkItemsPerTile(dev_id);
    workgroup_num = total_resource / max_workgroup_size;

    int max_block_row = max_workgroup_size / SIMD;
    block_row = 1;
    while ((block_row << 2) <= Batch && (block_row << 1) <= max_block_row) {
      block_row = block_row << 1;
    }
    workgroup_size = max_workgroup_size / block_row;

    // maximize the vec_size
    constexpr int float4_size = sizeof(float) * 4;
    max_vec_size = float4_size / element_size_bytes;
    while ((max_vec_size >> 1) * workgroup_num * workgroup_size >= Plane &&
           (max_vec_size >> 1) >= 1) {
      max_vec_size = max_vec_size >> 1;
    }

    // maximize the workgroup_size, and minimize the block_row
    while ((workgroup_size >> 1) * workgroup_num * max_vec_size > Plane &&
           (workgroup_size >> 1) >= SIMD) {
      workgroup_size = workgroup_size >> 1;
    }
    while ((workgroup_size << 1) * workgroup_num * max_vec_size <= Plane &&
           (workgroup_size << 1) <= max_workgroup_size) {
      workgroup_size = workgroup_size << 1;
    }
    block_row = max_workgroup_size / workgroup_size;

    workgroup_num = (Plane + workgroup_size * max_vec_size - 1) /
        (workgroup_size * max_vec_size);
  }
};

template <
    typename scalar_t,
    typename mean_t,
    typename weight_t,
    bool one_moment = false>
class NormForward {
 public:
  using accscalar_t = acc_type<scalar_t, true>;
  NormForward() = delete;
  NormForward(
      scalar_t* X_data,
      scalar_t* Y_data,
      mean_t* mean_data,
      mean_t* var_data,
      weight_t* gamma_data,
      weight_t* beta_data,
      accscalar_t eps)
      : X_data(X_data),
        Y_data(Y_data),
        mean_data(mean_data),
        var_data(var_data),
        gamma_data(gamma_data),
        beta_data(beta_data),
        eps(eps) {}

  int get_rowwise_reduce_vec_size(int Plane, int vec_size) {
    vec_size = std::min(
        vec_size,
        can_vectorize_up_to<scalar_t>(reinterpret_cast<char*>(X_data)));

    while (Plane % vec_size != 0) {
      vec_size = vec_size >> 1;
    }
    return vec_size;
  }

  int get_update_vec_size(int Plane, int vec_size) {
    vec_size = std::min(
        vec_size,
        can_vectorize_up_to<scalar_t>(reinterpret_cast<char*>(X_data)));
    vec_size = std::min(
        vec_size,
        can_vectorize_up_to<scalar_t>(reinterpret_cast<char*>(Y_data)));
    if (gamma_data) {
      vec_size = std::min(
          vec_size,
          can_vectorize_up_to<weight_t>(reinterpret_cast<char*>(gamma_data)));
    }
    if (beta_data) {
      vec_size = std::min(
          vec_size,
          can_vectorize_up_to<weight_t>(reinterpret_cast<char*>(gamma_data)));
    }

    while (Plane % vec_size != 0) {
      vec_size = vec_size >> 1;
    }
    return vec_size;
  }

  int get_eltwise_update_vec_size(int vec_size) {
    vec_size = std::min(
        vec_size,
        can_vectorize_up_to<scalar_t>(reinterpret_cast<char*>(X_data)));
    vec_size = std::min(
        vec_size,
        can_vectorize_up_to<scalar_t>(reinterpret_cast<char*>(Y_data)));
    return vec_size;
  }

  template <
      int vec_size,
      typename vec_t,
      typename weight_vec_t,
      typename index_t,
      typename nd_item_id>
  void reduce_combine(
      nd_item_id item_id,
      const NormConfig& cfg,
      accscalar_t& sum1,
      accscalar_t& sum2) const {
    auto group_id = item_id.get_group(0);
    auto group_id_foreach = item_id.get_group(1);
    auto local_id = item_id.get_local_id(2);
    index_t group_offset = group_id * cfg.Plane;

    for (index_t j = local_id * vec_size; j < (index_t)cfg.WGPlane;
         j += cfg.workgroup_size * vec_size) {
      index_t plane_offset = group_id_foreach * cfg.WGPlane + j;
      if (plane_offset < (index_t)cfg.Plane) {
        vec_t value =
            *(reinterpret_cast<vec_t*>(X_data + group_offset + plane_offset));
        for (int v = 0; v < vec_size; ++v) {
          sum1 += static_cast<accscalar_t>(value[v]);
          sum2 += static_cast<accscalar_t>(value[v]) *
              static_cast<accscalar_t>(value[v]);
        }
      }
    }
  }

  template <typename nd_item_id>
  void reduce_project(
      nd_item_id item_id,
      accscalar_t sum1,
      accscalar_t sum2,
      const NormConfig& cfg) const {
    auto group_id = item_id.get_group(0);
    accscalar_t scale = static_cast<accscalar_t>(cfg.Plane);
    sum2 = (sum2 - sum1 * sum1 / scale) / scale;
    sum1 = sum1 / scale;
    mean_data[group_id] = static_cast<mean_t>(sum1);
    var_data[group_id] = static_cast<mean_t>(c10::xpu::compat::rsqrt(
        sum2 < 0 ? 0 : sum2 + static_cast<accscalar_t>(eps)));
  }

 public:
  scalar_t* X_data;
  scalar_t* Y_data;
  mean_t* mean_data;
  mean_t* var_data;
  weight_t* gamma_data;
  weight_t* beta_data;
  accscalar_t eps;
};

template <
    typename scalar_t,
    typename mean_t,
    typename weight_t,
    bool one_moment = false>
class NormBackward {
 public:
  using accscalar_t = acc_type<scalar_t, true>;
  NormBackward(
      scalar_t* X_data,
      scalar_t* dY_data,
      scalar_t* dX_data,
      mean_t* mean_data,
      mean_t* var_data,
      weight_t* gamma_data,
      accscalar_t* a_data,
      accscalar_t* b_data)
      : X_data(X_data),
        dY_data(dY_data),
        dX_data(dX_data),
        mean_data(mean_data),
        var_data(var_data),
        gamma_data(gamma_data),
        a_data(a_data),
        b_data(b_data) {}

  scalar_t* X_data;
  scalar_t* dY_data;
  scalar_t* dX_data;
  mean_t* mean_data;
  mean_t* var_data;
  weight_t* gamma_data;
  accscalar_t* a_data;
  accscalar_t* b_data;

  int get_rowwise_reduce_vec_size(int Plane, int vec_size) {
    vec_size = std::min(
        vec_size,
        can_vectorize_up_to<scalar_t>(reinterpret_cast<char*>(X_data)));
    vec_size = std::min(
        vec_size,
        can_vectorize_up_to<scalar_t>(reinterpret_cast<char*>(dY_data)));
    if (gamma_data) {
      vec_size = std::min(
          vec_size,
          can_vectorize_up_to<weight_t>(reinterpret_cast<char*>(gamma_data)));
    }

    while (Plane % vec_size != 0) {
      vec_size = vec_size >> 1;
    }
    return vec_size;
  }

  int get_update_vec_size(int Plane, int vec_size) {
    vec_size = std::min(
        vec_size,
        can_vectorize_up_to<scalar_t>(reinterpret_cast<char*>(X_data)));
    vec_size = std::min(
        vec_size,
        can_vectorize_up_to<scalar_t>(reinterpret_cast<char*>(dY_data)));
    vec_size = std::min(
        vec_size,
        can_vectorize_up_to<scalar_t>(reinterpret_cast<char*>(dX_data)));
    if (gamma_data) {
      vec_size = std::min(
          vec_size,
          can_vectorize_up_to<weight_t>(reinterpret_cast<char*>(gamma_data)));
    }

    while (Plane % vec_size != 0) {
      vec_size = vec_size >> 1;
    }
    return vec_size;
  }

  int get_eltwise_update_vec_size(int vec_size) {
    vec_size = std::min(
        vec_size,
        can_vectorize_up_to<scalar_t>(reinterpret_cast<char*>(X_data)));
    vec_size = std::min(
        vec_size,
        can_vectorize_up_to<scalar_t>(reinterpret_cast<char*>(dY_data)));
    vec_size = std::min(
        vec_size,
        can_vectorize_up_to<scalar_t>(reinterpret_cast<char*>(dX_data)));
    return vec_size;
  }

  template <typename nd_item_id>
  void reduce_project(
      nd_item_id item_id,
      accscalar_t sum1,
      accscalar_t sum2,
      const NormConfig& cfg) const {
    auto group_id = item_id.get_group(0);
    a_data[group_id] = sum1;
    b_data[group_id] = sum2;
  };
};

template <
    typename scalar_t,
    typename mean_t,
    typename weight_t,
    typename index_t,
    typename accscalar_t,
    typename vec_t,
    typename weight_vec_t,
    int vec_size,
    template <typename, typename, typename>
    class Norm,
    bool one_moment = false>
struct FusedNormKernelFunctor {
  [[intel::reqd_sub_group_size(SIMD)]] void operator()(
      sycl::nd_item<3> item_id) const {
    accscalar_t sum1 = 0;
    accscalar_t sum2 = 0;
    norm.template reduce_combine<vec_size, vec_t, weight_vec_t, index_t>(
        item_id, cfg, sum1, sum2);

    if constexpr (one_moment) {
      sum1 = sycl::reduce_over_group(
          item_id.get_group(), sum1, sycl::plus<accscalar_t>());
    } else {
      norm_group_reduce<accscalar_t>(
          item_id,
          cfg.sub_group_num,
          sum1,
          sum2,
          local_sum1,
          local_sum2,
          [](accscalar_t a, accscalar_t b) { return a + b; });
    }
    norm.template update<vec_size, index_t, vec_t, weight_vec_t>(
        item_id, cfg, sum1, sum2);
  }
  FusedNormKernelFunctor(
      sycl_local_acc_t<accscalar_t> local_sum1_,
      sycl_local_acc_t<accscalar_t> local_sum2_,
      Norm<scalar_t, mean_t, weight_t> norm_,
      NormConfig cfg_)
      : local_sum1(local_sum1_),
        local_sum2(local_sum2_),
        norm(norm_),
        cfg(cfg_) {}

 private:
  sycl_local_acc_t<accscalar_t> local_sum1;
  sycl_local_acc_t<accscalar_t> local_sum2;
  Norm<scalar_t, mean_t, weight_t> norm;
  const NormConfig cfg;
};

template <
    typename scalar_t,
    typename mean_t,
    typename weight_t,
    typename index_t,
    typename accscalar_t,
    typename vec_t,
    typename weight_vec_t,
    int vec_size,
    template <typename, typename, typename>
    class Norm,
    bool one_moment = false>
struct FusedNormKernelFunctorCreator {
  FusedNormKernelFunctorCreator(
      Norm<scalar_t, mean_t, weight_t> norm,
      NormConfig cfg,
      size_t slm_sz)
      : norm_(norm), cfg_(cfg), slm_sz_(slm_sz) {}

  auto operator()(::sycl::handler& cgh) const {
    sycl_local_acc_t<accscalar_t> local_sum1(slm_sz_, cgh);
    sycl_local_acc_t<accscalar_t> local_sum2(slm_sz_, cgh);
    return FusedNormKernelFunctor<
        scalar_t,
        mean_t,
        weight_t,
        index_t,
        accscalar_t,
        vec_t,
        weight_vec_t,
        vec_size,
        Norm,
        one_moment>(local_sum1, local_sum2, norm_, cfg_);
  }

 private:
  Norm<scalar_t, mean_t, weight_t> norm_;
  NormConfig cfg_;
  size_t slm_sz_;
};

template <
    typename scalar_t,
    typename mean_t,
    typename weight_t,
    typename index_t,
    int vec_size,
    template <typename, typename, typename>
    class Norm,
    bool one_moment = false>
void fused_norm_kernel(
    Norm<scalar_t, mean_t, weight_t>& norm,
    const NormConfig& cfg) {
  using accscalar_t = acc_type<scalar_t, true>;
  using vec_t = aligned_vector<scalar_t, vec_size>;
  using weight_vec_t = aligned_vector<weight_t, vec_size>;
  sycl::range<3> local_range{
      1, (size_t)cfg.workgroup_num_foreach, (size_t)cfg.workgroup_size};
  sycl::range<3> global_range{
      (size_t)cfg.workgroup_num,
      (size_t)cfg.workgroup_num_foreach,
      (size_t)cfg.workgroup_size};

  auto creator = FusedNormKernelFunctorCreator<
      scalar_t,
      mean_t,
      weight_t,
      index_t,
      accscalar_t,
      vec_t,
      weight_vec_t,
      vec_size,
      Norm,
      one_moment>(norm, cfg, (size_t)cfg.sub_group_num);
  sycl_kernel_submit<typename function_traits<decltype(creator)>::result_type>(
      global_range, local_range, getCurrentSYCLQueue(), creator);
}

template <
    typename scalar_t,
    typename mean_t,
    typename weight_t,
    template <typename, typename, typename>
    class Norm,
    bool one_moment = false>
void launch_vectorized_fused_norm_kernel(
    Norm<scalar_t, mean_t, weight_t>& norm,
    const NormConfig& config,
    bool can_use_32bit_index) {
  int vec_size = norm.get_update_vec_size(config.WGPlane, config.max_vec_size);
#define vectorized_fused_norm_kernel(vec_size) \
  {                                            \
    if (can_use_32bit_index) {                 \
      fused_norm_kernel<                       \
          scalar_t,                            \
          mean_t,                              \
          weight_t,                            \
          uint32_t,                            \
          vec_size,                            \
          Norm,                                \
          one_moment>(norm, config);           \
    } else {                                   \
      fused_norm_kernel<                       \
          scalar_t,                            \
          mean_t,                              \
          weight_t,                            \
          uint32_t,                            \
          vec_size,                            \
          Norm,                                \
          one_moment>(norm, config);           \
    }                                          \
    break;                                     \
  }

  switch (vec_size) {
    case 8: {
      vectorized_fused_norm_kernel(8);
    }
    case 4: {
      vectorized_fused_norm_kernel(4);
    }
    case 2: {
      vectorized_fused_norm_kernel(2);
    }
    default: {
      vectorized_fused_norm_kernel(1);
    }
  }
}

template <
    typename scalar_t,
    typename mean_t,
    typename weight_t,
    typename index_t,
    typename accscalar_t,
    typename vec_t,
    typename weight_vec_t,
    int vec_size,
    template <typename, typename, typename>
    class Norm,
    bool one_moment = false>
struct RowwiseMomentsSYCLKernelFunctor {
  [[intel::reqd_sub_group_size(SIMD)]] void operator()(
      sycl::nd_item<3> item_id) const {
    index_t local_id = item_id.get_local_id(2);

    accscalar_t sum1 = 0;
    accscalar_t sum2 = 0;
    norm.template reduce_combine<vec_size, vec_t, weight_vec_t, index_t>(
        item_id, cfg, sum1, sum2);
    if constexpr (one_moment) {
      sum1 = sycl::reduce_over_group(
          item_id.get_group(), sum1, sycl::plus<accscalar_t>());
    } else {
      norm_group_reduce<accscalar_t>(
          item_id,
          cfg.sub_group_num,
          sum1,
          sum2,
          local_sum1,
          local_sum2,
          [](accscalar_t a, accscalar_t b) { return a + b; });
    }
    if (cfg.workgroup_num_foreach > 1) {
      norm_global_reduce<accscalar_t, index_t, one_moment>(
          item_id,
          cfg.workgroup_num_foreach,
          cfg.workgroup_size,
          cfg.sub_group_num_global,
          sum1,
          sum2,
          static_cast<accscalar_t*>(cfg.scratchpad_ptr),
          cfg.semaphores_ptr,
          local_sum1,
          local_sum2,
          last_workgroup,
          [](accscalar_t a, accscalar_t b) { return a + b; });
      if (last_workgroup[0] && local_id == 0) {
        norm.template reduce_project(item_id, sum1, sum2, cfg);
      }
    } else {
      if (local_id == 0) {
        norm.template reduce_project(item_id, sum1, sum2, cfg);
      }
    }
  }
  RowwiseMomentsSYCLKernelFunctor(
      sycl_local_acc_t<accscalar_t> local_sum1_,
      sycl_local_acc_t<accscalar_t> local_sum2_,
      Norm<scalar_t, mean_t, weight_t> norm_,
      NormConfig cfg_,
      sycl_local_acc_t<bool> last_workgroup_)
      : local_sum1(local_sum1_),
        local_sum2(local_sum2_),
        norm(norm_),
        cfg(cfg_),
        last_workgroup(last_workgroup_) {}

 private:
  sycl_local_acc_t<accscalar_t> local_sum1;
  sycl_local_acc_t<accscalar_t> local_sum2;
  Norm<scalar_t, mean_t, weight_t> norm;
  const NormConfig cfg;
  sycl_local_acc_t<bool> last_workgroup;
};

template <
    typename scalar_t,
    typename mean_t,
    typename weight_t,
    typename index_t,
    typename accscalar_t,
    typename vec_t,
    typename weight_vec_t,
    int vec_size,
    template <typename, typename, typename>
    class Norm,
    bool one_moment = false>
struct RowwiseMomentsSYCLKernelFunctorCreator {
  auto operator()(::sycl::handler& cgh) const {
    sycl_local_acc_t<accscalar_t> local_sum1(
        slm_size_, cgh); // cfg.sub_group_num
    sycl_local_acc_t<accscalar_t> local_sum2(slm_size_, cgh);
    sycl_local_acc_t<bool> last_workgroup(1, cgh);
    return RowwiseMomentsSYCLKernelFunctor<
        scalar_t,
        mean_t,
        weight_t,
        index_t,
        accscalar_t,
        vec_t,
        weight_vec_t,
        vec_size,
        Norm,
        one_moment>(local_sum1, local_sum2, norm_, cfg_, last_workgroup);
  }
  RowwiseMomentsSYCLKernelFunctorCreator(
      Norm<scalar_t, mean_t, weight_t> norm,
      NormConfig cfg,
      size_t slm_size)
      : norm_(norm), cfg_(cfg), slm_size_(slm_size) {}

 private:
  Norm<scalar_t, mean_t, weight_t> norm_;
  NormConfig cfg_;
  size_t slm_size_;
};

template <
    typename scalar_t,
    typename mean_t,
    typename weight_t,
    typename index_t,
    int vec_size,
    template <typename, typename, typename>
    class Norm,
    bool one_moment = false>
void RowwiseMomentsSYCLKernel(
    Norm<scalar_t, mean_t, weight_t>& norm,
    NormConfig& cfg) {
  using accscalar_t = acc_type<scalar_t, true>;
  using vec_t = aligned_vector<scalar_t, vec_size>;
  using weight_vec_t = aligned_vector<weight_t, vec_size>;

  sycl::range<3> local_range{1, 1, (size_t)cfg.workgroup_size};
  sycl::range<3> global_range{
      (size_t)cfg.workgroup_num,
      (size_t)cfg.workgroup_num_foreach,
      (size_t)cfg.workgroup_size};
  auto creator = RowwiseMomentsSYCLKernelFunctorCreator<
      scalar_t,
      mean_t,
      weight_t,
      index_t,
      accscalar_t,
      vec_t,
      weight_vec_t,
      vec_size,
      Norm,
      one_moment>(norm, cfg, (size_t)cfg.sub_group_num);
  sycl_kernel_submit<typename function_traits<decltype(creator)>::result_type>(
      global_range, local_range, getCurrentSYCLQueue(), creator);
}

template <
    typename scalar_t,
    typename mean_t,
    typename weight_t,
    template <typename, typename, typename>
    class Norm,
    bool one_moment = false>
void RowwiseMomentsSYCLKernelImpl(
    Norm<scalar_t, mean_t, weight_t>& norm,
    NormConfig& config,
    bool can_use_32bit_index) {
  int vec_size =
      norm.get_rowwise_reduce_vec_size(config.WGPlane, config.max_vec_size);
#define VecRowwiseMomentsSYCLKernel(vec_size) \
  {                                           \
    if (can_use_32bit_index) {                \
      RowwiseMomentsSYCLKernel<               \
          scalar_t,                           \
          mean_t,                             \
          weight_t,                           \
          uint32_t,                           \
          vec_size,                           \
          Norm,                               \
          one_moment>(norm, config);          \
    } else {                                  \
      RowwiseMomentsSYCLKernel<               \
          scalar_t,                           \
          mean_t,                             \
          weight_t,                           \
          uint64_t,                           \
          vec_size,                           \
          Norm,                               \
          one_moment>(norm, config);          \
    }                                         \
    break;                                    \
  }
  switch (vec_size) {
    case 8: {
      VecRowwiseMomentsSYCLKernel(8);
    }
    case 4: {
      VecRowwiseMomentsSYCLKernel(4);
    }
    case 2: {
      VecRowwiseMomentsSYCLKernel(2);
    }
    default: {
      VecRowwiseMomentsSYCLKernel(1);
    }
  }
}

template <
    typename scalar_t,
    typename mean_t,
    typename weight_t,
    typename index_t,
    int vec_size,
    template <typename, typename, typename>
    class Norm,
    typename vec_t,
    typename weight_vec_t>
struct NormUpdateKernelFunctor {
  void operator()(sycl::nd_item<3> item_id) const {
    norm.template update<vec_size, index_t, vec_t, weight_vec_t>(item_id, cfg);
  }
  NormUpdateKernelFunctor(
      Norm<scalar_t, mean_t, weight_t> norm_,
      NormConfig cfg_)
      : norm(norm_), cfg(cfg_) {}

 private:
  Norm<scalar_t, mean_t, weight_t> norm;
  NormConfig cfg;
};

template <
    typename scalar_t,
    typename mean_t,
    typename weight_t,
    typename index_t,
    int vec_size,
    template <typename, typename, typename>
    class Norm,
    bool one_moment = false>
void NormUpdateKernel(
    Norm<scalar_t, mean_t, weight_t>& norm,
    const NormConfig& cfg) {
  // input: [M][N]
  // gamma, beta: [M]
  // mean, var: [N]
  using vec_t = aligned_vector<scalar_t, vec_size>;
  using weight_vec_t = aligned_vector<weight_t, vec_size>;

  sycl::range<3> local_range{1, 1, (size_t)cfg.workgroup_size};
  sycl::range<3> global_range{
      (size_t)cfg.workgroup_num,
      (size_t)cfg.workgroup_num_foreach,
      (size_t)cfg.workgroup_size};

  auto caller = NormUpdateKernelFunctor<
      scalar_t,
      mean_t,
      weight_t,
      index_t,
      vec_size,
      Norm,
      vec_t,
      weight_vec_t>(norm, cfg);
  sycl_kernel_submit(global_range, local_range, getCurrentSYCLQueue(), caller);
}

template <
    typename scalar_t,
    typename mean_t,
    typename weight_t,
    template <typename, typename, typename>
    class Norm,
    bool one_moment = false>
void NormUpdateKernelImpl(
    Norm<scalar_t, mean_t, weight_t>& norm,
    const NormConfig& config,
    bool can_use_32bit_index) {
  int vec_size = norm.get_update_vec_size(config.WGPlane, config.max_vec_size);

#define VecNormUpdateKernel(vec_size) \
  {                                   \
    if (can_use_32bit_index) {        \
      NormUpdateKernel<               \
          scalar_t,                   \
          mean_t,                     \
          weight_t,                   \
          uint32_t,                   \
          vec_size,                   \
          Norm,                       \
          one_moment>(norm, config);  \
    } else {                          \
      NormUpdateKernel<               \
          scalar_t,                   \
          mean_t,                     \
          weight_t,                   \
          uint64_t,                   \
          vec_size,                   \
          Norm,                       \
          one_moment>(norm, config);  \
    }                                 \
    break;                            \
  }

  switch (vec_size) {
    case 8: {
      VecNormUpdateKernel(8);
    }
    case 4: {
      VecNormUpdateKernel(4);
    }
    case 2: {
      VecNormUpdateKernel(2);
    }
    case 1: {
      VecNormUpdateKernel(1);
    }
  }
}

template <
    typename scalar_t,
    typename mean_t,
    typename weight_t,
    typename index_t,
    int vec_size,
    template <typename, typename, typename>
    class Norm,
    typename vec_t>
struct NormEltwiseUpdateKernelFunctor {
  void operator()(sycl::nd_item<1> item_id) const {
    index_t local_id = item_id.get_global_linear_id();
    for (index_t i = local_id; i < loops_end; i += total_threads) {
      norm.template eltwise_update<vec_size, index_t, vec_t>(i);
    }
  }
  NormEltwiseUpdateKernelFunctor(
      Norm<scalar_t, mean_t, weight_t> norm_,
      index_t loops_end_,
      int total_threads_)
      : norm(norm_), loops_end(loops_end_), total_threads(total_threads_) {}

 private:
  Norm<scalar_t, mean_t, weight_t> norm;
  index_t loops_end;
  int total_threads;
};

template <
    typename scalar_t,
    typename mean_t,
    typename weight_t,
    typename index_t,
    int vec_size,
    template <typename, typename, typename>
    class Norm,
    bool one_moment = false>
void NormEltwiseUpdateKernel(Norm<scalar_t, mean_t, weight_t>& norm) {
  using vec_t = aligned_vector<scalar_t, vec_size>;
  int total_threads = syclMaxWorkItemsPerTile();
  auto workgroup_size = syclMaxWorkGroupSize();
  index_t loops_end = (norm.numel + vec_size - 1) / vec_size;

  auto caller = NormEltwiseUpdateKernelFunctor<
      scalar_t,
      mean_t,
      weight_t,
      index_t,
      vec_size,
      Norm,
      vec_t>(norm, loops_end, total_threads);
  sycl_kernel_submit(
      total_threads, workgroup_size, getCurrentSYCLQueue(), caller);
}

template <
    typename scalar_t,
    typename mean_t,
    typename weight_t,
    template <typename, typename, typename>
    class Norm,
    bool one_moment = false>
void NormEltwiseUpdateKernelImpl(
    Norm<scalar_t, mean_t, weight_t>& norm,
    const NormConfig& cfg,
    bool can_use_32bit_index) {
  int vec_size = norm.get_eltwise_update_vec_size(cfg.max_vec_size);
#define VecNormEltwiseUpdateKernel(vec_size) \
  {                                          \
    if (can_use_32bit_index) {               \
      NormEltwiseUpdateKernel<               \
          scalar_t,                          \
          mean_t,                            \
          weight_t,                          \
          uint32_t,                          \
          vec_size,                          \
          Norm>(norm);                       \
    } else {                                 \
      NormEltwiseUpdateKernel<               \
          scalar_t,                          \
          mean_t,                            \
          weight_t,                          \
          uint64_t,                          \
          vec_size,                          \
          Norm>(norm);                       \
    }                                        \
    break;                                   \
  }

  switch (vec_size) {
    case 8: {
      VecNormEltwiseUpdateKernel(8);
    }
    case 4: {
      VecNormEltwiseUpdateKernel(4);
    }
    case 2: {
      VecNormEltwiseUpdateKernel(2);
    }
    case 1: {
      VecNormEltwiseUpdateKernel(1);
    }
  }
}

} // namespace xpu
} // namespace native
} // namespace at
