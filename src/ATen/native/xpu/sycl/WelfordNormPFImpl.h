#pragma once

#include <ATen/native/Resize.h>
#include <ATen/native/xpu/sycl/MemoryAccess.h>
#include <comm/SYCLContext.h>
#include <comm/XPUMathCompat.h>

namespace at::native::xpu {

namespace impl {

// returns floor(log2(n))
inline int last_pow2(int n) {
  n |= (n >> 1);
  n |= (n >> 2);
  n |= (n >> 4);
  n |= (n >> 8);
  n |= (n >> 16);
  return std::max(1, n - (n >> 1));
}

template <typename T>
inline T divup(T a, T b) {
  return (a + b - 1) / b;
}

std::tuple<int, int, int, int> get_adaptive_config(
    const int problem_size,
    const int batch_size,
    const int vec_size,
    const int max_block_size,
    int loops_per_thread = 8,
    int coop_th = 8) {
  loops_per_thread /=
      vec_size; // Ensure the number of instructions is normalized
  int threads_along_batch = last_pow2(batch_size / vec_size);
  int threads_along_problem = last_pow2(divup(problem_size, loops_per_thread));

  int block_size_x = std::min(threads_along_batch, 32);
  int block_size_y =
      std::min(threads_along_problem, max_block_size / block_size_x);
  if (block_size_x * block_size_y != max_block_size) {
    block_size_x = std::min(threads_along_batch, max_block_size / block_size_y);
  }

  int max_threads_gpu = syclMaxWorkItemsPerTile();
  int nblock_x = divup(batch_size, block_size_x * vec_size);
  int nblock_y = std::min(
      divup(problem_size, block_size_y * loops_per_thread),
      max_threads_gpu / (nblock_x * block_size_x) / (block_size_y));
  nblock_y = std::max(nblock_y, 1);

  // it's not worth having reduction between blocks if the reduction
  // dimension is not big enough
  coop_th /= vec_size;
  nblock_y = nblock_y < coop_th ? 1 : nblock_y;

  return std::make_tuple(block_size_y, block_size_x, nblock_y, nblock_x);
}

template <typename T, typename C>
inline void welford_merge(
    C& count,
    T& mean,
    T& m2n,
    const C& count_new,
    const T& mean_new,
    const T& m2n_new) {
  T factor = T(1.0) / std::max(1, (count + count_new));
  T delta0 = mean - mean_new;
  mean = (mean_new * count_new + mean * count) * factor;
  m2n += m2n_new + delta0 * delta0 * count_new * count * factor;
  count += count_new;
}

} // namespace impl

template <
    typename VarTransform,
    typename scalar_t,
    typename acc_t,
    int VEC_SIZE>
struct WelfordNormPFKernel : public __SYCL_KER_CONFIG_CONVENTION__ {
  using vec_t = memory::aligned_vector<scalar_t, VEC_SIZE>;
  using acc_vec_t = memory::aligned_vector<acc_t, VEC_SIZE>;
  using int_vec_t = memory::aligned_vector<int, VEC_SIZE>;

  void operator()(sycl::nd_item<2> item) const {
    // init welford counters
    acc_vec_t mean;
    acc_vec_t m2n;
    int_vec_t count;
#pragma unroll
    for (int v = 0; v < VEC_SIZE; ++v) {
      mean[v] = acc_t(0);
      m2n[v] = acc_t(0);
      count[v] = int(0);
    }

    int bx = item.get_group(1); // along batch dim
    int by = item.get_group(0); // along problem dim
    int batch_vec_offset = item.get_global_id(1) * VEC_SIZE;
    int num_cooperative_blocks = item.get_group_range(0);
    int inner_loop_stride = item.get_local_range(0) * num_cooperative_blocks;

    if (batch_vec_offset < batch_size_) {
      for (int p_offset = item.get_global_id(0); p_offset < problem_size_;
           p_offset += inner_loop_stride) {
        int address_vec_base = p_offset * batch_size_ + batch_vec_offset;
        auto input_vec = *reinterpret_cast<vec_t*>(
            const_cast<scalar_t*>(&input_[address_vec_base]));
#pragma unroll
        for (int v = 0; v < VEC_SIZE; ++v) {
          auto x = input_vec[v];
          count[v]++;
          acc_t delta0 = x - mean[v];
          mean[v] += delta0 / count[v];
          acc_t delta1 = x - mean[v];
          m2n[v] += delta0 * delta1;
        }
      }
    }

    welford_vertical_merge(
        item, count, mean, m2n, shmem_count_, shmem_mean_, shmem_m2n_);

    // welford vertical merge
    if (num_cooperative_blocks > 1) {
      acc_t* staging_mean = staging_data_;
      acc_t* staging_m2n = &staging_data_[batch_size_ * num_cooperative_blocks];
      int* staging_count = reinterpret_cast<int*>(
          &staging_m2n[batch_size_ * num_cooperative_blocks]);
      int address_vec_base = batch_vec_offset + by * batch_size_;

      // write data to staging_data;
      if (item.get_local_id(0) == 0 && batch_vec_offset < batch_size_) {
        *reinterpret_cast<acc_vec_t*>(&staging_mean[address_vec_base]) = mean;
        *reinterpret_cast<acc_vec_t*>(&staging_m2n[address_vec_base]) = m2n;
        *reinterpret_cast<int_vec_t*>(&staging_count[address_vec_base]) = count;
      }
      item.barrier(sycl_local_fence);

      // mark block done
      if (item.get_local_linear_id() == 0) {
        sycl_atomic_ref_rlx_dev_global_t<int> atomic_count(semaphores_[bx]);
        int old = atomic_count.fetch_add(
            1, sycl_mem_odr_acq_rel
            /* , default memory scope is device */);
        is_last_block_done_[0] = (old == (num_cooperative_blocks - 1));
      }
      item.barrier(sycl_local_fence);

      // check that all data is now available in global memory
      if (is_last_block_done_[0]) {
#pragma unroll
        for (int v = 0; v < VEC_SIZE; ++v) {
          mean[v] = acc_t(0);
          m2n[v] = acc_t(0);
          count[v] = int(0);
        }

        for (int y = item.get_local_id(0); y < num_cooperative_blocks;
             y += item.get_local_range(0)) {
          if (batch_vec_offset < batch_size_) {
            address_vec_base = y * batch_size_ + batch_vec_offset;
            auto mean_new =
                *reinterpret_cast<acc_vec_t*>(&staging_mean[address_vec_base]);
            auto m2n_new =
                *reinterpret_cast<acc_vec_t*>(&staging_m2n[address_vec_base]);
            auto count_new =
                *reinterpret_cast<int_vec_t*>(&staging_count[address_vec_base]);
#pragma unroll
            for (int v = 0; v < VEC_SIZE; ++v) {
              impl::welford_merge(
                  count[v],
                  mean[v],
                  m2n[v],
                  count_new[v],
                  mean_new[v],
                  m2n_new[v]);
            }
          }
        }
        welford_vertical_merge(
            item, count, mean, m2n, shmem_count_, shmem_mean_, shmem_m2n_);
      }
    }

    if (item.get_local_id(0) == 0 &&
        (num_cooperative_blocks == 1 || is_last_block_done_[0]) &&
        batch_vec_offset < batch_size_) {
      acc_vec_t invstd_vec;
#pragma unroll
      for (int v = 0; v < VEC_SIZE; ++v) {
        invstd_vec[v] = VarTransform{}(m2n[v] / count[v], epsilon_);
      }
      *reinterpret_cast<acc_vec_t*>(&save_mean_[batch_vec_offset]) = mean;
      *reinterpret_cast<acc_vec_t*>(&save_invstd_[batch_vec_offset]) =
          invstd_vec;

      if (running_mean_ != nullptr) {
        auto running_mean_vec =
            *reinterpret_cast<vec_t*>(&running_mean_[batch_vec_offset]);
#pragma unroll
        for (int v = 0; v < VEC_SIZE; ++v) {
          running_mean_vec[v] =
              mean[v] * momentum_ + (1 - momentum_) * running_mean_vec[v];
        }
        *reinterpret_cast<vec_t*>(&running_mean_[batch_vec_offset]) =
            running_mean_vec;
      }

      if (running_var_ != nullptr) {
        auto running_var_vec =
            *reinterpret_cast<vec_t*>(&running_var_[batch_vec_offset]);
#pragma unroll
        for (int v = 0; v < VEC_SIZE; ++v) {
          auto unbiased_var = m2n[v] / (count[v] - 1);
          running_var_vec[v] =
              unbiased_var * momentum_ + (1 - momentum_) * running_var_vec[v];
        }
        *reinterpret_cast<vec_t*>(&running_var_[batch_vec_offset]) =
            running_var_vec;
      }
    }
  }

  template <typename CACC, typename TACC>
  inline void welford_vertical_merge(
      sycl::nd_item<2>& item,
      int_vec_t& count,
      acc_vec_t& mean,
      acc_vec_t& m2n,
      CACC& shmem_count,
      TACC& shmem_mean,
      TACC& shmem_m2n) const {
    // write to shared memory
    auto address_base = item.get_local_linear_id();
#pragma unroll
    for (int offset = item.get_local_range(0) / 2; offset > 0; offset >>= 1) {
      if (item.get_local_id(0) < offset * 2) {
        shmem_mean[address_base] = mean;
        shmem_m2n[address_base] = m2n;
        shmem_count[address_base] = count;
      }
      item.barrier(sycl_local_fence);
      if (item.get_local_id(0) < offset &&
          item.get_local_id(0) + offset < item.get_local_range(0)) {
        auto address = address_base + offset * item.get_local_range(1);
        // read shared memory back to register for reduction
        auto count_new = shmem_count[address];
        auto mean_new = shmem_mean[address];
        auto m2n_new = shmem_m2n[address];
#pragma unroll
        for (int v = 0; v < VEC_SIZE; ++v) {
          impl::welford_merge(
              count[v], mean[v], m2n[v], count_new[v], mean_new[v], m2n_new[v]);
        }
      }
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    auto local_size = block_size_x_ * block_size_y_;
    shmem_mean_ = sycl_local_acc_t<acc_vec_t>(sycl::range<1>(local_size), cgh);
    shmem_m2n_ = sycl_local_acc_t<acc_vec_t>(sycl::range<1>(local_size), cgh);
    shmem_count_ = sycl_local_acc_t<int_vec_t>(sycl::range<1>(local_size), cgh);
    is_last_block_done_ = sycl_local_acc_t<bool>(sycl::range<1>(1), cgh);
  }

  void init() {
    using KernelT =
        WelfordNormPFKernel<VarTransform, scalar_t, acc_t, VEC_SIZE>;
    auto max_group_size = syclMaxWorkGroupSize<KernelT>();
    std::tie(block_size_y_, block_size_x_, nblocks_y_, nblocks_x_) =
        impl::get_adaptive_config(
            problem_size_, batch_size_, VEC_SIZE, max_group_size);
  }

  static bool valid(
      int batch_size,
      int problem_size,
      const scalar_t* input,
      acc_t* save_mean,
      acc_t* save_invstd) {
    if (VEC_SIZE <= 1)
      return true;
    bool valid = sizeof(scalar_t) <= 4;
    valid = valid && (batch_size % VEC_SIZE == 0);
    valid = valid &&
        (memory::can_vectorize_up_to<scalar_t>((char*)input) >= VEC_SIZE);
    valid = valid &&
        (memory::can_vectorize_up_to<acc_t>((char*)save_mean) >= VEC_SIZE);
    valid = valid &&
        (memory::can_vectorize_up_to<acc_t>((char*)save_invstd) >= VEC_SIZE);
    return valid;
  }

  sycl::range<2> local_range() const {
    return sycl::range<2>(block_size_y_, block_size_x_);
  }

  sycl::range<2> global_range() const {
    return sycl::range<2>(
        block_size_y_ * nblocks_y_, block_size_x_ * nblocks_x_);
  }

  int staging_size() const {
    return nblocks_y_ * batch_size_ * 4;
  }

  int semaphores_size() const {
    return nblocks_x_;
  }

  bool set_staging_data_check(acc_t* staging_data) {
    staging_data_ = staging_data;
    return (
        (staging_data == nullptr) ||
        (memory::can_vectorize_up_to<acc_t>((char*)staging_data) >= VEC_SIZE));
  }

  void set_semaphores(int* semaphores) {
    semaphores_ = semaphores;
  }

  void set_running_mean_var(
      scalar_t* running_mean,
      scalar_t* running_var,
      acc_t momentum) {
    running_mean_ = running_mean;
    running_var_ = running_var;
    momentum_ = momentum;
  }

  int num_cooperative_blocks() const {
    return nblocks_y_;
  }

  WelfordNormPFKernel(
      const scalar_t* input,
      int batch_size,
      int problem_size,
      acc_t epsilon,
      acc_t* save_mean,
      acc_t* save_invstd)
      : input_(input),
        batch_size_(batch_size),
        problem_size_(problem_size),
        epsilon_(epsilon),
        save_mean_(save_mean),
        save_invstd_(save_invstd),
        staging_data_(nullptr),
        semaphores_(nullptr),
        running_mean_(nullptr),
        running_var_(nullptr) {}

 private:
  const scalar_t* input_;
  int batch_size_;
  int problem_size_;
  acc_t epsilon_;
  acc_t* save_mean_;
  acc_t* save_invstd_;
  acc_t* staging_data_;
  int* semaphores_;

  scalar_t* running_mean_;
  scalar_t* running_var_;
  acc_t momentum_;

  size_t block_size_y_;
  size_t block_size_x_;
  size_t nblocks_y_;
  size_t nblocks_x_;

  sycl_local_acc_t<acc_vec_t> shmem_mean_;
  sycl_local_acc_t<acc_vec_t> shmem_m2n_;
  sycl_local_acc_t<int_vec_t> shmem_count_;
  sycl_local_acc_t<bool> is_last_block_done_;
};

} // namespace at::native::xpu
