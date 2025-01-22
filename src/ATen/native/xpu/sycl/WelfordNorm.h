#pragma once

#include <ATen/native/xpu/sycl/MemoryAccess.h>
#include <comm/SYCLContext.h>

namespace at::native::xpu {

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

template <
    typename VarTransform,
    typename scalar_t,
    typename acc_t,
    int VEC_SIZE = 4,
    int SG_SIZE = 32,
    int NUM_SG = 8>
struct WelfordBatchNormStatChannelsLastVecKernelFunctor
    : public __SYCL_KER_CONFIG_CONVENTION__ {
  enum {
    LOCAL_SIZE = SG_SIZE * NUM_SG,
  };
  using vec_t = memory::aligned_vector<scalar_t, VEC_SIZE>;
  using acc_vec_t = memory::aligned_vector<acc_t, VEC_SIZE>;
  using int_vec_t = memory::aligned_vector<int, VEC_SIZE>;

  [[intel::reqd_sub_group_size(SG_SIZE)]] void operator()(
      sycl::nd_item<2> item) const {
    //  init private counter
    acc_vec_t x_mean;
    acc_vec_t m_2_n;
    int_vec_t count;
#pragma unroll
    for (int v = 0; v < VEC_SIZE; ++v) {
      x_mean[v] = acc_t(0);
      m_2_n[v] = acc_t(0);
      count[v] = int(0);
    }

    int lx = item.get_local_id(1);
    int gx = item.get_group(1);
    int gy = item.get_group(0);
    int num_cooperative_groups = item.get_group_range(0);
    auto c_vec_offset = gx * VEC_SIZE;

    for (int row_wg = gy * LOCAL_SIZE; row_wg < reduction_size_;
         row_wg += num_cooperative_groups * LOCAL_SIZE) {
      int vec_offset = (row_wg + lx) * n_channels_ + c_vec_offset;
      auto input_vec =
          *reinterpret_cast<vec_t*>(const_cast<scalar_t*>(&input_[vec_offset]));
#pragma unroll
      for (int v = 0; v < VEC_SIZE; ++v) {
        auto x = input_vec[v];
        count[v]++;
        acc_t delta0 = x - x_mean[v];
        x_mean[v] += delta0 / count[v];
        acc_t delta1 = x - x_mean[v];
        m_2_n[v] += delta0 * delta1;
      }
    }
    shmem_mean_[lx] = x_mean;
    shmem_m2n_[lx] = m_2_n;
    shmem_count_[lx] = count;
    item.barrier(sycl_local_fence);

    // sub-group welford merge
    if (lx < SG_SIZE) {
#pragma unroll
      for (int sg_id = 1; sg_id < NUM_SG; ++sg_id) {
        auto idx = lx + sg_id * SG_SIZE;
#pragma unroll
        for (int v = 0; v < VEC_SIZE; ++v) {
          welford_merge(
              count[v],
              x_mean[v],
              m_2_n[v],
              shmem_count_[idx][v],
              shmem_mean_[idx][v],
              shmem_m2n_[idx][v]);
        }
      }
    }
    shmem_mean_[lx] = x_mean;
    shmem_m2n_[lx] = m_2_n;
    shmem_count_[lx] = count;
    item.barrier(sycl_local_fence);
#pragma unroll
    for (int i = 1; i < SG_SIZE; i++) {
#pragma unroll
      for (int v = 0; v < VEC_SIZE; ++v) {
        welford_merge(
            count[v],
            x_mean[v],
            m_2_n[v],
            shmem_count_[i][v],
            shmem_mean_[i][v],
            shmem_m2n_[i][v]);
      }
    }

    // welford vertical merge
    if (num_cooperative_groups > 1) {
      acc_t* staging_mean = staging_data_;
      acc_t* staging_m2n = &staging_data_[n_channels_ * num_cooperative_groups];
      int* staging_count = reinterpret_cast<int*>(
          &staging_m2n[n_channels_ * num_cooperative_groups]);
      int address_vec_base = c_vec_offset + gy * n_channels_;

      // write data to staging_data;
      if (lx == 0) {
        *reinterpret_cast<acc_vec_t*>(&staging_mean[address_vec_base]) = x_mean;
        *reinterpret_cast<acc_vec_t*>(&staging_m2n[address_vec_base]) = m_2_n;
        *reinterpret_cast<int_vec_t*>(&staging_count[address_vec_base]) = count;
      }
      item.barrier(sycl_local_fence);

      // mark group done
      if (item.get_local_linear_id() == 0) {
        sycl_atomic_ref_rlx_dev_global_t<int> atomic_count(semaphores_[gx]);
        int old = atomic_count.fetch_add(
            1, sycl_mem_odr_acq_rel
            /* , default memory scope is device */);
        is_last_group_done_[0] = (old == (num_cooperative_groups - 1));
      }
      item.barrier(sycl_local_fence);

      // check that all data is now available in global memory
      if (is_last_group_done_[0]) {
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
          x_mean[i] = acc_t(0);
          m_2_n[i] = acc_t(0);
          count[i] = int(0);
        }

        for (int y = lx; y < num_cooperative_groups; y += LOCAL_SIZE) {
          address_vec_base = y * n_channels_ + c_vec_offset;
          auto mean_new =
              *reinterpret_cast<acc_vec_t*>(&staging_mean[address_vec_base]);
          auto m2n_new =
              *reinterpret_cast<acc_vec_t*>(&staging_m2n[address_vec_base]);
          auto count_new =
              *reinterpret_cast<int_vec_t*>(&staging_count[address_vec_base]);
#pragma unroll
          for (int v = 0; v < VEC_SIZE; ++v) {
            welford_merge(
                count[v],
                x_mean[v],
                m_2_n[v],
                count_new[v],
                mean_new[v],
                m2n_new[v]);
          }
        }

        shmem_mean_[lx] = x_mean;
        shmem_m2n_[lx] = m_2_n;
        shmem_count_[lx] = count;
        item.barrier(sycl_local_fence);

        // sub-group welford merge
        if (lx < SG_SIZE) {
#pragma unroll
          for (int sg_id = 1; sg_id < NUM_SG; ++sg_id) {
            auto idx = lx + sg_id * SG_SIZE;
#pragma unroll
            for (int v = 0; v < VEC_SIZE; ++v) {
              welford_merge(
                  count[v],
                  x_mean[v],
                  m_2_n[v],
                  shmem_count_[idx][v],
                  shmem_mean_[idx][v],
                  shmem_m2n_[idx][v]);
            }
          }
        }
        shmem_mean_[lx] = x_mean;
        shmem_m2n_[lx] = m_2_n;
        shmem_count_[lx] = count;
        item.barrier(sycl_local_fence);
#pragma unroll
        for (int i = 1; i < SG_SIZE; i++) {
#pragma unroll
          for (int v = 0; v < VEC_SIZE; ++v) {
            welford_merge(
                count[v],
                x_mean[v],
                m_2_n[v],
                shmem_count_[i][v],
                shmem_mean_[i][v],
                shmem_m2n_[i][v]);
          }
        }

        if (lx == 0) {
          *reinterpret_cast<acc_vec_t*>(&save_mean_[c_vec_offset]) = x_mean;
          acc_vec_t invstd_vec;
#pragma unroll
          for (int i = 0; i < VEC_SIZE; ++i) {
            invstd_vec[i] = VarTransform{}(m_2_n[i] / count[i], 1e-5);
          }
          *reinterpret_cast<acc_vec_t*>(&save_invstd_[c_vec_offset]) =
              invstd_vec;
        }
      }
    } else {
      if (gy == 0 && lx == 0) {
        *reinterpret_cast<acc_vec_t*>(&save_mean_[c_vec_offset]) = x_mean;
        acc_vec_t invstd_vec;
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
          invstd_vec[i] = VarTransform{}(m_2_n[i] / count[i], 1e-5);
        }
        *reinterpret_cast<acc_vec_t*>(&save_invstd_[c_vec_offset]) = invstd_vec;
      }
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    shmem_mean_ = sycl_local_acc_t<acc_vec_t>(sycl::range<1>(LOCAL_SIZE), cgh);
    shmem_m2n_ = sycl_local_acc_t<acc_vec_t>(sycl::range<1>(LOCAL_SIZE), cgh);
    shmem_count_ = sycl_local_acc_t<int_vec_t>(sycl::range<1>(LOCAL_SIZE), cgh);
    is_last_group_done_ = sycl_local_acc_t<bool>(sycl::range<1>(1), cgh);
  }

  WelfordBatchNormStatChannelsLastVecKernelFunctor(
      const scalar_t* input,
      acc_t* save_mean,
      acc_t* save_invstd,
      int reduction_size,
      int n_channels,
      acc_t* staging_data,
      int* semaphores)
      : input_(input),
        save_mean_(save_mean),
        save_invstd_(save_invstd),
        reduction_size_(reduction_size),
        n_channels_(n_channels),
        staging_data_(staging_data),
        semaphores_(semaphores) {
    auto gy = reduction_size / LOCAL_SIZE;
    if (gy % 8 == 0) {
      num_cooperative_groups_ = gy / 8;
    } else if (gy % 4 == 0) {
      num_cooperative_groups_ = gy / 4;
    } else if (gy % 2 == 0) {
      num_cooperative_groups_ = gy / 2;
    } else {
      num_cooperative_groups_ = gy;
    }
  }

  static bool valid(
      int reduction_size,
      int n_channels,
      const scalar_t* input,
      acc_t* save_mean,
      acc_t* save_invstd) {
    bool valid = true;
    valid = valid && (n_channels % VEC_SIZE == 0);
    valid = valid && (reduction_size % LOCAL_SIZE == 0);
    valid = valid && (reduction_size / n_channels >= 32);
    valid = valid &&
        (memory::can_vectorize_up_to<scalar_t>((char*)input) >= VEC_SIZE);
    valid = valid &&
        (memory::can_vectorize_up_to<acc_t>((char*)save_mean) >= VEC_SIZE);
    valid = valid &&
        (memory::can_vectorize_up_to<acc_t>((char*)save_invstd) >= VEC_SIZE);
    return valid;
  }

  sycl::range<2> local_range() const {
    return sycl::range<2>(1, LOCAL_SIZE);
  }

  sycl::range<2> global_range() const {
    return sycl::range<2>(
        num_cooperative_groups_, n_channels_ / VEC_SIZE * LOCAL_SIZE);
  }

  int staging_size() const {
    return num_cooperative_groups_ * n_channels_ * 4;
  }

  int semaphores_size() const {
    return n_channels_ / VEC_SIZE;
  }

  void set_staging_data(acc_t* staging_data) {
    staging_data_ = staging_data;
  }

  void set_semaphores(int* semaphores) {
    semaphores_ = semaphores;
  }

  int num_cooperative_groups() const {
    return num_cooperative_groups_;
  }

 private:
  const scalar_t* input_;
  acc_t* save_mean_;
  acc_t* save_invstd_;
  int reduction_size_;
  int n_channels_;
  acc_t* staging_data_;
  int* semaphores_;
  size_t num_cooperative_groups_;
  sycl_local_acc_t<acc_vec_t> shmem_mean_;
  sycl_local_acc_t<acc_vec_t> shmem_m2n_;
  sycl_local_acc_t<int_vec_t> shmem_count_;
  sycl_local_acc_t<bool> is_last_group_done_;
};

} // namespace at::native::xpu
