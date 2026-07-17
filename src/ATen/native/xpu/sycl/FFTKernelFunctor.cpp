/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/WrapDimUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/SpectralOpsUtils.h>
#include <ATen/native/xpu/sycl/OffsetCalculator.h>
#include <comm/SYCLContext.h>
#include <comm/TensorInfo.h>

#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>

namespace at {
namespace native {
namespace xpu {

namespace impl {

namespace syclexp = sycl::ext::oneapi::experimental;

// Process-lifetime in-memory cache of JIT-compiled FFT kernel bundles.
// This cache stores the executable bundles keyed by the exact build
// parameters that determine the binary, identical repeated
// FFTs reuse a compiled bundle instead of recompiling.
//
// Bundles are owned here via shared_ptr for the lifetime of the process; the
// fft_descriptor holds a shared reference rather than owning them. The set of
// distinct keys is tiny (supported sizes x dims x directions x precisions per
// device), so no eviction is needed for now.
class FftKernelBundleCache {
 public:
  using bundle_t = sycl::kernel_bundle<sycl::bundle_state::executable>;

  static FftKernelBundleCache& instance() {
    static FftKernelBundleCache cache;
    return cache;
  }

  std::shared_ptr<bundle_t> get(const std::string& key) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = cache_.find(key);
    return it == cache_.end() ? nullptr : it->second;
  }

  void put(const std::string& key, std::shared_ptr<bundle_t> bundle) {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_.emplace(key, std::move(bundle));
  }

  // NOTE: the critical section is guarded by std::lock_guard (RAII).
  // It acquires the mutex on construction and releases it in its destructor when
  // `lock` goes out of scope at the end of get()/put().
  // Adding a manual unlock() would cause a double-unlock bug.
 private:
  std::mutex mutex_;
  std::unordered_map<std::string, std::shared_ptr<bundle_t>> cache_;
};

struct fft_descriptor {
  std::vector<std::int64_t> fft_len;
  std::vector<std::int64_t> fwd_strides;
  std::vector<std::int64_t> bwd_strides;
  std::int64_t batch;
  std::int64_t fwd_dist;
  std::int64_t bwd_dist;
  std::vector<std::vector<std::int64_t>> facts;
  std::int64_t slm_size[3] = {0, 0, 0};
  double fwd_scale;
  double bwd_scale;
  int which_dir; // 0 for forward, 1 for backward, 2 for both
  bool external_workspace = false;
  std::int64_t external_workspace_size = 0;
  std::int64_t twidl_table_size[3] = {0, 0, 0};
  const void* twidl_table[3][2] = {{nullptr}}; // [dimension][direction]
  std::array<std::array<size_t, 3>, 3> local_work_size{
      {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}}};
  std::array<std::array<size_t, 3>, 3> global_work_size{
      {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}}};
  // Shared references to cached executable bundles; ownership stays with
  // FftKernelBundleCache. [dimension][direction]
  std::shared_ptr<sycl::kernel_bundle<sycl::bundle_state::executable>>
      exe_bundle[3][2];
  sycl::queue* queue = nullptr; // non-owning, for resource cleanup

  fft_descriptor() = default;

  fft_descriptor(const fft_descriptor&) = delete;
  fft_descriptor& operator=(const fft_descriptor&) = delete;
  fft_descriptor(fft_descriptor&&) = delete;
  fft_descriptor& operator=(fft_descriptor&&) = delete;

  ~fft_descriptor() {
    release();
  }

 private:
  void release() {
    for (int dim = 0; dim < 3; ++dim) {
      for (int dir = 0; dir < 2; ++dir) {
        if (twidl_table[dim][dir] && !external_workspace && queue) {
          sycl::free(const_cast<void*>(twidl_table[dim][dir]), *queue);
          twidl_table[dim][dir] = nullptr;
        }
        // exe_bundle holds shared references into FftKernelBundleCache; the
        // cache retains ownership, so we just drop our reference here.
        exe_bundle[dim][dir].reset();
      }
    }
    queue = nullptr;
  }
};

// Sort transform dimensions by input layout, for best performance
// exclude_last is for onesided transforms where the last dimension cannot be
// reordered
static DimVector _sort_dims(
    const Tensor& self,
    IntArrayRef dim,
    bool exclude_last = false) {
  DimVector sorted_dims(dim.begin(), dim.end());
  auto self_strides = self.strides();
  std::sort(
      sorted_dims.begin(),
      sorted_dims.end() - exclude_last,
      [&](int64_t a, int64_t b) { return self_strides[a] > self_strides[b]; });
  return sorted_dims;
}

double _dft_scale(
    IntArrayRef dim,
    IntArrayRef norm_sizes,
    int64_t normalization) {
  const auto norm = static_cast<fft_norm_mode>(normalization);
  double double_scale = 1.0;

  if (norm == fft_norm_mode::none) {
    return double_scale;
  }

  int64_t signal_numel = 1;
  for (const int64_t& d : dim) {
    signal_numel *= norm_sizes[d];
  }
  if (norm == fft_norm_mode::by_root_n) {
    double_scale = 1.0 / std::sqrt(signal_numel);
  } else {
    double_scale = 1.0 / static_cast<double>(signal_numel);
  }

  return double_scale;
}

const Tensor& _fft_apply_normalization(
    const Tensor& self,
    int64_t normalization,
    IntArrayRef norm_sizes,
    IntArrayRef dims) {
  auto scale = _dft_scale(dims, norm_sizes, normalization);
  return (scale == 1.0) ? self : self.mul_(scale);
}

// TODO: Remove this work-around in future.
Tensor promote_fft_input(const Tensor& input) {
  if (input.scalar_type() == ScalarType::Half)
    return input.to(ScalarType::Float);
  if (input.scalar_type() == ScalarType::ComplexHalf)
    return input.to(ScalarType::ComplexFloat);
  return input;
}

std::string kernel_src = R"""(

#include <sycl/sycl.hpp>

#ifdef DFT_DOUBLE_PRECISION
    #define DFT_FTYPE double
#else
    #define DFT_FTYPE float
#endif

inline sycl::marray<DFT_FTYPE, 2> swapReIm(sycl::marray<DFT_FTYPE, 2> &tmp)
{
    constexpr DFT_FTYPE exp_sign = (DIR_VAL==0) ? 1.0 : -1.0;
    auto Re = tmp[0];
    auto Im = tmp[1];
    sycl::marray<DFT_FTYPE, 2> res;
    res[0] = exp_sign * Im;
    res[1] = -1 * exp_sign * Re;
    return res;
}

inline void twidl_mult_inplace(sycl::marray<DFT_FTYPE, 2> &var, const DFT_FTYPE &twr,
                               const DFT_FTYPE &twi)
{
    constexpr DFT_FTYPE exp_sign = (DIR_VAL==0) ? 1.0 : -1.0;
    auto Re = var[0];
    auto Im = var[1];
    var[0] = Re * twr + exp_sign * twi * Im;
    var[1] = Im * twr - exp_sign * twi * Re;
}

inline void fft_2(sycl::marray<DFT_FTYPE, 2> *in, sycl::marray<DFT_FTYPE, 2> *out, int is = 1, int id = 0, int os = 1, int od = 0)
{
    auto x1 = in[id+0*is];
    auto x2 = in[id+1*is];
    auto y1 = x1 + x2;
    auto y2 = x1 - x2;
    out[od+0*os] = y1;
    out[od+1*os] = y2;
}

inline void fft_3(sycl::marray<DFT_FTYPE, 2> *in, sycl::marray<DFT_FTYPE, 2> *out, int is = 1, int id = 0, int os = 1, int od = 0)
{
     sycl::marray<DFT_FTYPE, 2> x0, x1, x2;
     sycl::marray<DFT_FTYPE, 2> y0, y1, y2;
     x0 = in[id+0*is];
     x1 = in[id+1*is];
     x2 = in[id+2*is];

     y0 = x0;
     y1 = x0;
     y2 = x0;
     sycl::marray<DFT_FTYPE, 2> x1px2     = x1+x2;
     sycl::marray<DFT_FTYPE, 2> wre_x1px2 = sycl::cospi((DFT_FTYPE)1.0/3.0)*x1px2;
     sycl::marray<DFT_FTYPE, 2> wim_x1mx2 = sycl::sinpi((DFT_FTYPE)1.0/3.0)*(x1-x2);
     sycl::marray<DFT_FTYPE, 2> pm_swp_wim_x1mx2 = swapReIm(wim_x1mx2);
     y0 += x1px2;
     y1 += -wre_x1px2 + pm_swp_wim_x1mx2;
     y2 += -wre_x1px2 - pm_swp_wim_x1mx2;

     out[od+0*os] = y0;
     out[od+1*os] = y1;
     out[od+2*os] = y2;
}

inline void fft_4(sycl::marray<DFT_FTYPE, 2> *in, sycl::marray<DFT_FTYPE, 2> *out, int is = 1, int id = 0, int os = 1, int od = 0)
{
    sycl::marray<DFT_FTYPE, 2> tmp0 = in[id+0*is] + in[id+2*is];
    sycl::marray<DFT_FTYPE, 2> tmp1 = in[id+1*is] + in[id+3*is];
    sycl::marray<DFT_FTYPE, 2> tmp2 = in[id+0*is] - in[id+2*is];
    sycl::marray<DFT_FTYPE, 2> tmp3 = in[id+1*is] - in[id+3*is];
    tmp3 = swapReIm(tmp3);
    out[od+0*os] = tmp0 + tmp1;
    out[od+1*os] = tmp2 + tmp3;
    out[od+2*os] = tmp0 - tmp1;
    out[od+3*os] = tmp2 - tmp3;
}

inline void fft_6(sycl::marray<DFT_FTYPE, 2> *in, sycl::marray<DFT_FTYPE, 2> *out, int is = 1, int id = 0, int os = 1, int od = 0) {
    fft_2(in, out, 3*is, id+0*is, 1*os, od+0*2*os);
    fft_2(in, out, 3*is, id+1*is, 1*os, od+1*2*os);
    fft_2(in, out, 3*is, id+2*is, 1*os, od+2*2*os);

    twidl_mult_inplace(out[od+3*os], sycl::cospi((DFT_FTYPE)1.0/3.0), sycl::sinpi((DFT_FTYPE)1.0/3.0));
    twidl_mult_inplace(out[od+5*os], -sycl::cospi((DFT_FTYPE)1.0/3.0), sycl::sinpi((DFT_FTYPE)1.0/3.0));

    fft_3(out, out, 2*os, od+0*os, 2*os, od+0*os);
    fft_3(out, out, 2*os, od+1*os, 2*os, od+1*os);
}

inline void fft_8(sycl::marray<DFT_FTYPE, 2> *in, sycl::marray<DFT_FTYPE, 2> *out, int is = 1, int id = 0, int os = 1, int od = 0)
{
    fft_2(in, out, 4*is, id+0*is, 1*os, od+0*2*os);
    fft_2(in, out, 4*is, id+1*is, 1*os, od+1*2*os);
    fft_2(in, out, 4*is, id+2*is, 1*os, od+2*2*os);
    fft_2(in, out, 4*is, id+3*is, 1*os, od+3*2*os);

    twidl_mult_inplace(out[od+3*os], sycl::cospi((DFT_FTYPE)1.0/4.0), sycl::sinpi((DFT_FTYPE)1.0/4.0));
    twidl_mult_inplace(out[od+5*os], (DFT_FTYPE)0.0, (DFT_FTYPE)1.0);
    twidl_mult_inplace(out[od+7*os], -sycl::cospi((DFT_FTYPE)1.0/4.0), sycl::sinpi((DFT_FTYPE)1.0/4.0));

    fft_4(out, out, 2*os, od+0*os, 2*os, od+0*os);
    fft_4(out, out, 2*os, od+1*os, 2*os, od+1*os);
}

#define FFT_FN(N, FACT0, FACT1)                                                      \
    _Pragma("unroll")                                                                \
    for(auto i=0;i<FACT1;++i) {                                                      \
        fft_##FACT0(in, y, FACT1, i, 1, i*FACT0);                                    \
    }                                                                                \
                                                                                     \
    _Pragma("unroll")                                                                \
    for(auto i=1;i<FACT0;++i) {                                                      \
        _Pragma("unroll")                                                            \
        for(auto j=1;j<FACT1;++j) {                                                  \
            const DFT_FTYPE theta = (((DFT_FTYPE)(2.0)) * i * j) / ((DFT_FTYPE)(N)); \
            const DFT_FTYPE cos_val = sycl::cospi(theta);                            \
            const DFT_FTYPE sin_val = sycl::sinpi(theta);                            \
            twidl_mult_inplace(y[i + FACT0*j], cos_val, sin_val);                    \
        }                                                                            \
    }                                                                                \
                                                                                     \
    _Pragma("unroll")                                                                \
    for(auto i=0;i<FACT0;++i) {                                                      \
        fft_##FACT1(y, in, FACT0, i, FACT0, i);                                      \
    }

inline void fft_16(sycl::marray<DFT_FTYPE, 2> *in, sycl::marray<DFT_FTYPE, 2> *out, int is = 1, int id = 0, int os = 1, int od = 0)
{
    sycl::marray<DFT_FTYPE, 2> y[16];
    FFT_FN(16, 4, 4);
}

inline void fft_24(sycl::marray<DFT_FTYPE, 2> *in, sycl::marray<DFT_FTYPE, 2> *out, int is = 1, int id = 0, int os = 1, int od = 0)
{
    sycl::marray<DFT_FTYPE, 2> y[24];
    FFT_FN(24, 6, 4);
}

inline void fft_32(sycl::marray<DFT_FTYPE, 2> *in, sycl::marray<DFT_FTYPE, 2> *out, int is = 1, int id = 0, int os = 1, int od = 0)
{
    sycl::marray<DFT_FTYPE, 2> y[32];
    FFT_FN(32, 4, 8);
}

inline void read_input(const DFT_FTYPE *in, sycl::marray<DFT_FTYPE, 2> *inReg, int group_id,
                       int local_id, int rows, int stride_to_next_row, int batch, int idist) {
    size_t offset = group_id * idist + local_id * DIST_TO_NEXT_THREAD;

    if(local_id < FACT1) {
        #pragma unroll
        for(int i=0;i<rows;++i) {
            const DFT_FTYPE *base = in + i * stride_to_next_row * DIST_TO_NEXT_THREAD * 2;
            inReg[i][0] = base[offset * 2];
            inReg[i][1] = base[offset * 2 + 1];
        }
    }
}

inline void write_output(DFT_FTYPE *out, sycl::marray<DFT_FTYPE, 2> *inReg, int group_id, int local_id,
                         int rows, int stride_to_next_row, int batch, int odist, int fact1s_idx, int threads_in_group) {
    size_t offset = group_id * odist + fact1s_idx * threads_in_group * DIST_TO_NEXT_THREAD + local_id * DIST_TO_NEXT_THREAD;

    #pragma unroll
    for(int i=0;i<rows;++i) {
        DFT_FTYPE *base = out + i * stride_to_next_row * DIST_TO_NEXT_THREAD * 2;
        base[offset * 2] = inReg[i][0];
        base[offset * 2 + 1] = inReg[i][1];
    }
}

inline void write_to_slm_nontransposed(sycl::marray<DFT_FTYPE, 2> *inReg, sycl::local_accessor<DFT_FTYPE, 1> slm,
                                       int local_id, int rows, int stride_to_next_row) {
    if(local_id < FACT1) {
        #pragma unroll
        for(int i=0;i<rows;++i) {
            slm[i*stride_to_next_row*2 + local_id*2 + 0] = (inReg[i])[0];
            slm[i*stride_to_next_row*2 + local_id*2 + 1] = (inReg[i])[1];
        }
    }
}

inline void read_from_slm_transposed(sycl::marray<DFT_FTYPE, 2> *inReg, sycl::local_accessor<DFT_FTYPE, 1> slm,
                                     int local_id, int rows, int stride_to_next_row, int fact1s_idx, int threads_in_group) {
    size_t offset = fact1s_idx * stride_to_next_row * threads_in_group * 2 + local_id * stride_to_next_row * 2;

    #pragma unroll
    for(int i=0;i<rows;++i) {
        inReg[i][0] = slm[offset + i*2 + 0];
        inReg[i][1] = slm[offset + i*2 + 1];
    }
}

inline void twiddle_mult(sycl::marray<DFT_FTYPE, 2> *inReg, const DFT_FTYPE *twidl, int threads_in_group,
                         int group_id, int local_id, int rows, int cols) {
    const int col =  local_id;
    sycl::marray<DFT_FTYPE, 2> twidl_val[REGSIZE];
    #pragma unroll
    for(int i=0;i<rows;++i) {
        twidl_val[i][0] = twidl[i*cols*2 + col*2];
        twidl_val[i][1] = twidl[i*cols*2 + col*2 + 1];
    }

    #pragma unroll
    for(int i=0;i<rows;++i) {
        const DFT_FTYPE cos_val = twidl_val[i][0];
        const DFT_FTYPE sin_val = twidl_val[i][1];
        twidl_mult_inplace(inReg[i], cos_val, sin_val);
    }
}

extern "C"
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((sycl::ext::oneapi::experimental::nd_range_kernel<3>))
void KERNEL_NAME(const DFT_FTYPE *in, DFT_FTYPE *out, sycl::local_accessor<DFT_FTYPE, 1> slm, const DFT_FTYPE *twidl) {
    sycl::nd_item<3> it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int group_id = it.get_group(0);
    int inner_batch = it.get_group(1);
    int outer_batch = it.get_group(2);
    int local_id = it.get_local_id(0);
    int threads_in_group = it.get_local_range(0);
    sycl::marray<DFT_FTYPE, 2> inReg[REGSIZE];

    const DFT_FTYPE *in_batch = in + outer_batch * OUTER_BATCH_FWD_DIST * 2 + inner_batch * INNER_BATCH_FWD_DIST * 2;
    DFT_FTYPE *out_batch = out + outer_batch * OUTER_BATCH_BWD_DIST * 2 + inner_batch * INNER_BATCH_BWD_DIST * 2;
    read_input(in_batch, inReg, group_id, local_id, FACT0, FACT1, BATCH, FWD_DIST);

    FFT_FACT0(inReg, inReg);

    twiddle_mult(inReg, twidl, threads_in_group, group_id, local_id, FACT0, FACT1);
    write_to_slm_nontransposed(inReg, slm, local_id, FACT0, FACT1);
    it.barrier(sycl::access::fence_space::local_space);

    #pragma unroll
    for(int i=0;i<NUM_FACT1S;++i) {
        read_from_slm_transposed(inReg, slm, local_id, FACT1, FACT1, i, threads_in_group);
        FFT_FACT1(inReg, inReg);
        write_output(out_batch, inReg, group_id, local_id, FACT1, FACT0, BATCH, BWD_DIST, i, threads_in_group);
    }
}

)""";

constexpr int supported_sizes[][3] = {{512, 32, 16}, {768, 32, 24}};

template <typename T>
struct TwiddleTableKernel2FactsFunctor {
  void operator()(sycl::item<2> item) const {
    const int row = item.get_id(0);
    const int col = item.get_id(1);
    const T theta = (2.0 * row * col) / (fact0_ * fact1_);
    T* cosPtr = twidl_buf_ + row * fact1_ * 2 + col * 2;
    T* sinPtr = twidl_buf_ + row * fact1_ * 2 + col * 2 + 1;
    *cosPtr = scale_ * sycl::cospi(theta);
    *sinPtr = scale_ * sycl::sinpi(theta);
  }

  TwiddleTableKernel2FactsFunctor(
      std::int64_t fact0,
      std::int64_t fact1,
      T* twidl_buf,
      T scale)
      : fact0_(fact0), fact1_(fact1), twidl_buf_(twidl_buf), scale_(scale) {}

 private:
  std::int64_t fact0_;
  std::int64_t fact1_;
  T* twidl_buf_;
  T scale_;
};

template <typename T>
void calculate_twiddle_factors(sycl::queue& q, fft_descriptor& desc) {
  for (size_t dim = 0; dim < desc.fft_len.size(); ++dim) {
    auto fact0 = desc.facts[dim][0];
    auto fact1 = desc.facts[dim][1];
    for (int i = 0; i < 2; ++i) {
      if (desc.which_dir != 2 && i != desc.which_dir)
        continue;

      T scale = static_cast<T>(1.0);
      if (dim == 0)
        scale = (i == 0) ? static_cast<T>(desc.fwd_scale)
                         : static_cast<T>(desc.bwd_scale);

      T* twidl_buf = nullptr;
      if (!desc.external_workspace) {
        twidl_buf = (T*)malloc_device(2 * fact0 * fact1 * sizeof(T), q);
        if (twidl_buf == nullptr) {
          throw std::runtime_error(
              "Failed to allocate device memory for twiddle factors");
        }
        desc.twidl_table[dim][i] = static_cast<const void*>(twidl_buf);
      } else {
        twidl_buf = (T*)desc.twidl_table[dim][i];
      }

      auto ker =
          TwiddleTableKernel2FactsFunctor<T>(fact0, fact1, twidl_buf, scale);
      q.submit([&](sycl::handler& h) {
         h.parallel_for(sycl::range<2>(fact0, fact1), ker);
       }).wait();
    }
  }
}

template <typename T>
void commit(sycl::queue& q, fft_descriptor& desc) {
  if (!q.get_device().is_gpu()) {
    throw std::runtime_error("Device is not a GPU");
  }
  desc.queue = &q;

  if (desc.fft_len.size() < 1 || desc.fft_len.size() > 3) {
    throw std::runtime_error("Unsupported number of dimensions");
  }
  std::reverse(desc.fft_len.begin(), desc.fft_len.end());

  desc.fwd_strides.erase(desc.fwd_strides.begin());
  desc.bwd_strides.erase(desc.bwd_strides.begin());
  if (desc.fwd_strides.size() > 0 && desc.bwd_strides.size() > 0) {
    std::reverse(desc.fwd_strides.begin(), desc.fwd_strides.end());
    std::reverse(desc.bwd_strides.begin(), desc.bwd_strides.end());
  } else {
    // Default packed layout strides.
    desc.fwd_strides = {1, desc.fft_len[0]};
    desc.bwd_strides = {1, desc.fft_len[0]};
  }

  // The source bundle is only needed on a cache miss; create it lazily so a
  // fully-cached commit() skips create_kernel_bundle_from_source() as well.
  std::optional<sycl::kernel_bundle<sycl::bundle_state::ext_oneapi_source>>
      src_bundle;

  for (size_t dim = 0; dim < desc.fft_len.size(); ++dim) {
    for (int i = 0; i < sizeof(supported_sizes) / sizeof(supported_sizes[0]);
         ++i) {
      if (desc.fft_len[dim] == supported_sizes[i][0]) {
        desc.facts.push_back({supported_sizes[i][1], supported_sizes[i][2]});
        break;
      }
    }

    auto fact0 = desc.facts[dim][0];
    auto fact1 = desc.facts[dim][1];
    auto slm_size = fact0 * fact1 * 2;
    if (slm_size * static_cast<std::int64_t>(sizeof(T)) >
        q.get_device().get_info<sycl::info::device::local_mem_size>()) {
      throw std::runtime_error("Required SLM size exceeds device limits");
    }
    desc.slm_size[dim] = slm_size;
    desc.twidl_table_size[dim] = fact0 * fact1 * 2;
    desc.external_workspace_size += desc.twidl_table_size[dim] * sizeof(T) *
        ((desc.which_dir == 2) ? 2 : 1);

    if (fact0 == 0 || fact1 == 0) {
      throw std::runtime_error("Unsupported FFT length");
    }

    desc.local_work_size[dim][0] = fact1;
    if (fact0 % fact1 != 0)
      desc.local_work_size[dim][0] = std::max(fact0, fact1);
    desc.global_work_size[dim][0] = desc.local_work_size[dim][0] * desc.batch;

    // Kernel build parameters
    std::string kernel_prec;
    std::string kernel_name = "dft_2_facts_kernel_" + std::to_string(dim);
    int num_regs = std::max(fact0, fact1);
    int num_fact1s = (fact0 % fact1 == 0) ? (fact0 / fact1) : 1;
    int dir_val = 0; // 0 for forward, 1 for backward
    std::string fact0_fn = "fft_" + std::to_string(fact0);
    std::string fact1_fn = "fft_" + std::to_string(fact1);

    auto fwd_dist = desc.fwd_dist;
    auto bwd_dist = desc.bwd_dist;
    auto batch = desc.batch;
    auto inner_batch_fwd_dist = 0;
    auto inner_batch_bwd_dist = 0;
    auto outer_batch_fwd_dist = 0;
    auto outer_batch_bwd_dist = 0;
    auto dist_to_next_thread = desc.fwd_strides[dim];
    if (desc.fft_len.size() > 1) {
      if (dim == 0) {
        fwd_dist = desc.fwd_strides[0] * desc.fft_len[0];
        bwd_dist = desc.bwd_strides[0] * desc.fft_len[0];
        batch = desc.fft_len[1] * desc.batch;
        if (desc.fft_len.size() == 3)
          batch *= desc.fft_len[2];
        desc.global_work_size[dim][0] =
            desc.local_work_size[dim][0] * desc.batch * desc.fft_len[1];
      } else if (dim == 1) {
        fwd_dist = desc.fwd_strides[0];
        bwd_dist = desc.bwd_strides[0];
        batch = desc.fft_len[0];
        inner_batch_fwd_dist = desc.fwd_strides[1] * desc.fft_len[1];
        inner_batch_bwd_dist = desc.bwd_strides[1] * desc.fft_len[1];
        dist_to_next_thread = desc.fwd_strides[1];
        desc.global_work_size[dim][0] =
            desc.local_work_size[dim][0] * desc.fft_len[0];
        desc.global_work_size[dim][1] = desc.batch;
        if (desc.fft_len.size() == 3) {
          outer_batch_fwd_dist = desc.fwd_dist;
          outer_batch_bwd_dist = desc.bwd_dist;
          desc.global_work_size[dim][1] = desc.fft_len[2];
          desc.global_work_size[dim][2] = desc.fft_len[2] * desc.batch;
        }
      } else {
        fwd_dist = desc.fwd_strides[0];
        bwd_dist = desc.bwd_strides[0];
        batch = desc.fft_len[0];
        inner_batch_fwd_dist = desc.fwd_strides[1];
        inner_batch_bwd_dist = desc.bwd_strides[1];
        outer_batch_fwd_dist = desc.fwd_dist;
        outer_batch_bwd_dist = desc.bwd_dist;
        dist_to_next_thread = desc.fwd_strides[2];
        desc.global_work_size[dim][1] = desc.fft_len[1] * desc.fft_len[2];
        desc.global_work_size[dim][2] = desc.batch;
      }
    }

    for (dir_val = 0; dir_val < 2; ++dir_val) {
      if (desc.which_dir != 2 && dir_val != desc.which_dir)
        continue;
      auto kernel_name_dir = kernel_name + ((dir_val == 0) ? "_fwd" : "_bwd");
      char scale_buf[64];
      std::snprintf(
          scale_buf,
          sizeof(scale_buf),
          "%a",
          dir_val == 0   ? desc.fwd_scale
              : dim == 0 ? desc.bwd_scale
                         : 1.0);
      std::vector<std::string> fft_build_opts = {
          "-DKERNEL_NAME=" + kernel_name_dir,
          "-DREGSIZE=" + std::to_string(num_regs),
          "-DFACT0=" + std::to_string(fact0),
          "-DFACT1=" + std::to_string(fact1),
          "-DBATCH=" + std::to_string(batch),
          "-DFWD_DIST=" + std::to_string(fwd_dist),
          "-DBWD_DIST=" + std::to_string(bwd_dist),
          "-DNUM_FACT1S=" + std::to_string(num_fact1s),
          "-DDIR_VAL=" + std::to_string(dir_val),
          "-DSCALE=" + std::string(scale_buf),
          "-DFFT_FACT0=" + fact0_fn,
          "-DFFT_FACT1=" + fact1_fn,
          "-DDIST_TO_NEXT_THREAD=" + std::to_string(dist_to_next_thread),
          "-DOUTER_BATCH_FWD_DIST=" + std::to_string(outer_batch_fwd_dist),
          "-DINNER_BATCH_FWD_DIST=" + std::to_string(inner_batch_fwd_dist),
          "-DOUTER_BATCH_BWD_DIST=" + std::to_string(outer_batch_bwd_dist),
          "-DINNER_BATCH_BWD_DIST=" + std::to_string(inner_batch_bwd_dist)};
      if constexpr (std::is_same_v<T, float>) {
        // no define needed - that's default
      } else if constexpr (std::is_same_v<T, double>) {
        fft_build_opts.push_back("-DDFT_DOUBLE_PRECISION");
      } else {
        throw std::runtime_error("Unsupported data type");
      };

      // Build options fully determine the compiled binary, so use them (plus
      // the device index) as the cache key. A '\x1f' separator keeps distinct
      // option lists from colliding.
      std::string cache_key = std::to_string(at::xpu::current_device());
      for (const auto& opt : fft_build_opts) {
        cache_key += '\x1f';
        cache_key += opt;
      }

      auto& bundle_cache = FftKernelBundleCache::instance();
      auto cached = bundle_cache.get(cache_key);
      if (!cached) {
        if (!src_bundle) {
          src_bundle = syclexp::create_kernel_bundle_from_source(
              q.get_context(), syclexp::source_language::sycl, kernel_src);
        }
        auto exe_bundle = syclexp::build(
            *src_bundle,
            syclexp::properties{syclexp::build_options{fft_build_opts}});
        cached = std::make_shared<FftKernelBundleCache::bundle_t>(
            std::move(exe_bundle));
        bundle_cache.put(cache_key, cached);
      }
      desc.exe_bundle[dim][dir_val] = cached;
    }
  }
  if (!desc.external_workspace) {
    calculate_twiddle_factors<T>(q, desc);
  }
}

template <typename T>
void set_workspace(T* workspace, sycl::queue& q, fft_descriptor& desc) {
  if (workspace == nullptr) {
    throw std::runtime_error("Workspace pointer is null");
  }

  // Delete any previously allocated twiddle factor buffers if this overrides
  // workspace from internal to external.
  if (!desc.external_workspace && desc.queue) {
    for (int dim = 0; dim < 3; ++dim) {
      for (int dir = 0; dir < 2; ++dir) {
        if (desc.twidl_table[dim][dir]) {
          sycl::free(
              const_cast<void*>(desc.twidl_table[dim][dir]), *desc.queue);
          desc.twidl_table[dim][dir] = nullptr;
        }
      }
    }
  }

  desc.external_workspace = true;
  for (auto i = 0; i < desc.fft_len.size(); ++i) {
    for (int dir = 0; dir < 2; ++dir) {
      if (desc.which_dir != 2 && dir != desc.which_dir)
        continue;
      desc.twidl_table[i][dir] = static_cast<const void*>(workspace);
      workspace += desc.twidl_table_size[i];
    }
  }

  calculate_twiddle_factors<T>(q, desc);
}

template <typename T>
static sycl::event compute(
    sycl::queue& q,
    fft_descriptor& desc,
    const T* in,
    T* out,
    int dir) {
  const char* dir_suffix = (dir == 0) ? "_fwd" : "_bwd";
  sycl::event prev_ev;
  for (auto dim = 0; dim < desc.fft_len.size(); ++dim) {
    if (desc.external_workspace && desc.twidl_table[dim][dir] == nullptr) {
      throw std::runtime_error(
          "set_workspace must be called with a valid workspace before compute");
    }
    auto& bundle = *(desc.exe_bundle[dim][dir]);
    std::string kernel_name =
        "dft_2_facts_kernel_" + std::to_string(dim) + dir_suffix;
    auto kernel = bundle.ext_oneapi_get_kernel(kernel_name);

    sycl::range<3> global_range{
        desc.global_work_size[dim][0],
        desc.global_work_size[dim][1],
        desc.global_work_size[dim][2]};
    sycl::range<3> local_range{
        desc.local_work_size[dim][0],
        desc.local_work_size[dim][1],
        desc.local_work_size[dim][2]};
    sycl::nd_range<3> ndrange(global_range, local_range);

    const T* input = (dim == 0) ? in : out;
    prev_ev = q.submit([&](sycl::handler& h) {
      h.depends_on(prev_ev);
      h.set_arg(0, input);
      h.set_arg(1, out);
      h.set_arg(2, sycl::local_accessor<T, 1>(desc.slm_size[dim], h));
      h.set_arg(3, static_cast<const T*>(desc.twidl_table[dim][dir]));

      h.parallel_for(ndrange, kernel);
    });
  }
  return prev_ev;
}

// NOTE: This function does not handle multi-pass chunking  of
// higher dimensional FFTs like _fft_c2c_mkl does.
void _fft_with_size_sycl(
    Tensor& output,
    const Tensor& self,
    int64_t signal_ndim,
    bool complex_input,
    bool complex_output,
    bool inverse,
    IntArrayRef checked_signal_sizes,
    bool onesided) {
  Tensor input_ = self;
  // real/imag dimension must aligned when viewed as of complex type

  if (complex_input) {
    const auto strides = input_.strides();
    bool need_contiguous = strides.back() != 1;
    for (int64_t i = 0; !need_contiguous && i <= signal_ndim; i++) {
      need_contiguous |= (strides[i] % 2 != 0);
    }

    if (need_contiguous) {
      input_ = input_.contiguous();
    }
  }

  Tensor input = input_;
  auto& queue = at::xpu::getCurrentSYCLQueue();
  int64_t batch = checked_signal_sizes[0];
  std::vector<int64_t> mkl_signal_sizes(
      checked_signal_sizes.begin() + 1, checked_signal_sizes.end());

  auto istrides = input.strides();
  auto ostrides = output.strides();

  int64_t idist = istrides[0];
  int64_t odist = ostrides[0];

  std::vector<int64_t> input_strides(
      istrides.cbegin(), istrides.cbegin() + signal_ndim + 1),
      output_strides(ostrides.cbegin(), ostrides.cbegin() + signal_ndim + 1);
  input_strides[0] = 0;
  output_strides[0] = 0;

  fft_descriptor desc;
  desc.fft_len = mkl_signal_sizes;
  desc.batch = batch;
  desc.fwd_scale = 1.0;
  desc.bwd_scale = 1.0;
  desc.external_workspace = true;

  if (!inverse) {
    desc.which_dir = 0;

    desc.fwd_dist = idist;
    desc.bwd_dist = odist;

    desc.fwd_strides = input_strides;
    desc.bwd_strides = output_strides;
  } else {
    desc.which_dir = 1;

    desc.fwd_dist = odist;
    desc.bwd_dist = idist;

    desc.fwd_strides = output_strides;
    desc.bwd_strides = input_strides;
  }

  auto run = [&](auto type_tag) {
    using T = decltype(type_tag);
    commit<T>(queue, desc);

    // Obtain the size of workspace required after commit.
    int64_t workspaceSizeBytes = desc.external_workspace_size;

    // Allocate USM workspace and provide it to the descriptor.
    Tensor workspaceBuf = at::empty(
        {(long)(workspaceSizeBytes)},
        input.options().dtype(at::kChar),
        std::nullopt);
    set_workspace((T*)workspaceBuf.mutable_data_ptr(), queue, desc);

    auto in_data = (T*)input.const_data_ptr();
    auto out_data = (T*)output.mutable_data_ptr();
    compute(queue, desc, in_data, out_data, desc.which_dir);
  };

  if (input.scalar_type() == ScalarType::Float ||
      input.scalar_type() == ScalarType::ComplexFloat) {
    run(float{});
  } else {
    run(double{});
  }
  queue.throw_asynchronous();
}

// Execute a general fft operation (can be c2c, onesided r2c or onesided c2r)
Tensor& _exec_fft_sycl(
    Tensor& out,
    Tensor self,
    IntArrayRef out_sizes,
    IntArrayRef dim,
    bool onesided,
    bool forward) {
  const auto ndim = self.dim();
  const int64_t signal_ndim = dim.size();
  const auto batch_dims = ndim - signal_ndim;

  // Permute dimensions so batch dimensions come first, and in stride order
  // This maximizes data locality when collapsing to a single batch dimension
  DimVector dim_permute(ndim);
  std::iota(dim_permute.begin(), dim_permute.end(), int64_t{0});

  c10::SmallVector<bool, kDimVectorStaticSize> is_transformed_dim(ndim);
  for (const auto& d : dim) {
    is_transformed_dim[d] = true;
  }

  auto batch_end =
      std::partition(dim_permute.begin(), dim_permute.end(), [&](int64_t d) {
        return !is_transformed_dim[d];
      });

  auto self_strides = self.strides();
  std::sort(dim_permute.begin(), batch_end, [&](int64_t a, int64_t b) {
    return self_strides[a] > self_strides[b];
  });
  std::copy(dim.cbegin(), dim.cend(), batch_end);

  auto input = self.permute(dim_permute);

  // Collapse batch dimensions into a single dimension
  DimVector batched_sizes(signal_ndim + 1);
  batched_sizes[0] = -1;
  std::copy(
      input.sizes().cbegin() + batch_dims,
      input.sizes().cend(),
      batched_sizes.begin() + 1);
  input = input.reshape(batched_sizes);

  const auto in_sizes = input.sizes();
  const auto batch_size = in_sizes[0];
  DimVector signal_size(signal_ndim + 1);
  signal_size[0] = batch_size;

  for (const auto i : c10::irange(signal_ndim)) {
    auto in_size = in_sizes[i + 1];
    auto out_size = out_sizes[dim[i]];
    signal_size[i + 1] = std::max(in_size, out_size);
    TORCH_INTERNAL_ASSERT(
        in_size == signal_size[i + 1] ||
        in_size == (signal_size[i + 1] / 2) + 1);
    TORCH_INTERNAL_ASSERT(
        out_size == signal_size[i + 1] ||
        out_size == (signal_size[i + 1] / 2) + 1);
  }

  batched_sizes[0] = batch_size;
  DimVector batched_out_sizes(batched_sizes.begin(), batched_sizes.end());

  for (const auto i : c10::irange(dim.size())) {
    batched_out_sizes[i + 1] = out_sizes[dim[i]];
  }

  out.resize_(batched_out_sizes, MemoryFormat::Contiguous);

  // run the FFT
  _fft_with_size_sycl(
      out,
      input,
      signal_ndim,
      input.is_complex(),
      out.is_complex(),
      !forward,
      signal_size,
      onesided);

  // Inplace reshaping to original batch shape and inverting the dimension
  // permutation
  DimVector out_strides(ndim);
  int64_t batch_numel = 1;

  for (int64_t i = batch_dims - 1; i >= 0; --i) {
    out_strides[dim_permute[i]] = batch_numel * out.stride(0);
    batch_numel *= out_sizes[dim_permute[i]];
  }

  for (const auto i : c10::irange(batch_dims, ndim)) {
    out_strides[dim_permute[i]] = out.stride(1 + (i - batch_dims));
  }

  out.as_strided_(out_sizes, out_strides, out.storage_offset());

  return out;
}

} // namespace impl

template <typename index_t>
struct HermitianSymmetryOffsetCalculator {
  using offset_type = at::detail::Array<index_t, 1>;
  using dim_type = std::remove_cv_t<decltype(XPU_MAX_TENSORINFO_DIMS)>;

  dim_type dims;
  at::detail::IntDivider<index_t> sizes_[XPU_MAX_TENSORINFO_DIMS];
  index_t strides_[XPU_MAX_TENSORINFO_DIMS];
  uint32_t mirror_dim_; // bit mask
  static_assert(XPU_MAX_TENSORINFO_DIMS < 32, "Need a bigger mask type");

  HermitianSymmetryOffsetCalculator(
      IntArrayRef sizes,
      IntArrayRef strides,
      IntArrayRef dim,
      const int64_t element_size) {
    TORCH_INTERNAL_ASSERT(sizes.size() == strides.size());
    TORCH_INTERNAL_ASSERT(sizes.size() <= XPU_MAX_TENSORINFO_DIMS);
    dims = sizes.size();

    {
      dim_type i;
      for (i = 0; i < dims; ++i) {
        sizes_[i] = at::detail::IntDivider<index_t>(sizes[i]);
        strides_[i] = strides[i] / element_size;
      }
      for (; i < XPU_MAX_TENSORINFO_DIMS; ++i) {
        sizes_[i] = at::detail::IntDivider<index_t>(1);
        strides_[i] = 0;
      }
    }

    mirror_dim_ = 0;
    for (int64_t i = 0; i < dim.size(); ++i) {
      mirror_dim_ |= (uint32_t{1} << dim[i]);
    }
  }

  offset_type get(index_t linear_idx) const {
    index_t offset = 0;

    for (dim_type dim = 0; dim < dims; ++dim) {
      auto divmod = sizes_[dim].divmod(linear_idx);
      linear_idx = divmod.div;

      if ((mirror_dim_ & (uint32_t{1} << dim)) == 0) {
        offset += divmod.mod * strides_[dim];
      } else if (divmod.mod != 0) {
        offset += (sizes_[dim].divisor - divmod.mod) * strides_[dim];
      }
    }
    offset_type offsets;
    offsets[0] = offset;

    return offsets;
  }
};

template <typename scalar_t, typename inp_calc_t, typename out_calc_t>
struct FFTConjugateCopyKernelFunctor {
  void operator()(sycl::item<1> item_id) const {
    auto in_offset = ic_.get(item_id)[0];
    auto out_offset = oc_.get(item_id)[0];
    out_data_[out_offset] = std::conj(in_data_[in_offset]);
  }

  FFTConjugateCopyKernelFunctor(
      int64_t numel,
      scalar_t* out_data,
      const scalar_t* in_data,
      inp_calc_t ic,
      out_calc_t oc)
      : numel_(numel),
        out_data_(out_data),
        in_data_(in_data),
        ic_(ic),
        oc_(oc) {}

 private:
  int64_t numel_;
  scalar_t* out_data_;
  const scalar_t* in_data_;
  inp_calc_t ic_;
  out_calc_t oc_;
};

template <typename scalar_t, typename inp_calc_t, typename out_calc_t>
void _fft_conjugate_copy_kernel(
    int64_t numel,
    scalar_t* out_data,
    const scalar_t* in_data,
    inp_calc_t ic,
    out_calc_t oc) {
  auto& queue = at::xpu::getCurrentSYCLQueue();
  int thread_num = numel;

  auto ker = FFTConjugateCopyKernelFunctor<scalar_t, inp_calc_t, out_calc_t>(
      numel, out_data, in_data, ic, oc);

  sycl_kernel_submit(sycl::range<1>(thread_num), queue, ker);
}

void _fft_fill_with_conjugate_symmetry_xpu(
    ScalarType dtype,
    IntArrayRef mirror_dims,
    IntArrayRef signal_half_sizes,
    IntArrayRef in_strides,
    const void* in_data,
    IntArrayRef out_strides,
    void* out_data) {
  // Do the actual conjugate mirroring.
  auto* in_strides_ptr = in_strides.data();
  const int ndim = in_strides.size();
  const int64_t element_size = scalarTypeToTypeMeta(dtype).itemsize();

  OffsetCalculator<1, int64_t> input_offset_calculator(
      ndim, signal_half_sizes.data(), &in_strides_ptr, &element_size);
  HermitianSymmetryOffsetCalculator<int64_t> output_offset_calculator(
      signal_half_sizes, out_strides, mirror_dims, element_size);

  const auto numel = c10::multiply_integers(signal_half_sizes);
  AT_DISPATCH_COMPLEX_TYPES(dtype, "_fft_fill_with_conjugate_symmetry_", [&] {
    _fft_conjugate_copy_kernel(
        numel,
        static_cast<scalar_t*>(out_data),
        static_cast<const scalar_t*>(in_data),
        input_offset_calculator,
        output_offset_calculator);
  });
}

bool _is_fft_size_supported_sycl(const Tensor& orig_self, IntArrayRef dim) {
  if (dim.size() != 2) {
    return false;
  }

  if (!orig_self.is_complex()) {
    return false;
  }

  for (const auto& d : dim) {
    auto size = orig_self.sizes()[d];
    bool found = false;
    for (const auto& size_pair : impl::supported_sizes) {
      if (size == size_pair[0]) {
        found = true;
        break;
      }
    }
    if (!found) {
      return false;
    }
  }
  return true;
}

Tensor _fft_c2c_sycl(
    const Tensor& orig_self,
    IntArrayRef dim,
    int64_t normalization,
    bool forward) {
  if (dim.empty()) {
    return orig_self.clone();
  }
  auto self = impl::promote_fft_input(orig_self);

  auto sorted_dims = impl::_sort_dims(self, dim);
  auto out_sizes = self.sizes();
  auto out = at::empty(out_sizes, self.options());
  auto input_sizes = self.sizes();

  impl::_exec_fft_sycl(
      out,
      self,
      out_sizes,
      sorted_dims,
      /*onesided=*/false,
      forward);

  impl::_fft_apply_normalization(out, normalization, input_sizes, dim);

  if (orig_self.scalar_type() == ScalarType::ComplexHalf)
    return out.to(ScalarType::ComplexHalf);
  return out;
}

Tensor& _fft_c2c_sycl_out(
    const Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    bool forward,
    Tensor& out) {
  auto result = _fft_c2c_sycl(self, dim, normalization, forward);
  at::native::resize_output(out, result.sizes());
  out.copy_(result);
  return out;
}

} // namespace xpu
} // namespace native
} // namespace at
