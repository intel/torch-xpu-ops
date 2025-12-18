/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/AccumulateType.h>
#include <ATen/core/Array.h>
#include <ATen/native/xpu/sycl/BatchKernel.h>
#include <ATen/native/xpu/sycl/Reduce.h>
#include <comm/SYCLContext.h>
#include <comm/TensorInfo.h>
#include <comm/TensorOptions.h>
#include "comm/Runtime.h"

#include <ATen/native/xpu/sycl/WeightNormKernels.h>

namespace at::native::xpu {

template <typename T>
struct ReduceAdd {
  T operator()(const T a, const T b) const {
    return a + b;
  }
};

template <
    class ScalarTypeInfo,
    class AccTypeInfo,
    typename scalar_t,
    typename accscalar_t,
    typename vec_t>
struct WeightNormReduceKernelFunctor : public __SYCL_KER_CONFIG_CONVENTION__ {
  void operator()(sycl::nd_item<2> item) const {
    auto id = cfg_.get_item_desc(item);
    int64_t si = id.glb_batch % cfg_.stride_;
    int64_t bi = id.glb_batch / cfg_.stride_;
    int64_t ldr_pi = id.chunk * id.chunk_size + id.chunk_off;
    int64_t str_pi = id.chunk;
    int64_t ldr_lid =
        si + ldr_pi * cfg_.stride_ + bi * cfg_.problem_ * cfg_.stride_;
    int64_t ldr_off =
        at::xpu::detail::IndexToOffset<scalar_t, int64_t, -1>::get(
            ldr_lid, iinfo_);
    int64_t str_lid =
        si + str_pi * cfg_.stride_ + bi * id.chunk_num * cfg_.stride_;
    int64_t str_off =
        at::xpu::detail::IndexToOffset<accscalar_t, int64_t, -1>::get(
            str_lid, oinfo_);

    accscalar_t value = 0;
    if (id.glb_problem < cfg_.problem_ && id.glb_batch < cfg_.problem_batch_) {
      value = (accscalar_t)iinfo_.data[ldr_off];
      if (need_squre_)
        value *= value;
    }

    if (cfg_.problem_along_x_) {
      value = group_x_reduce(
          item, shared_, vec_t(value), ReduceAdd<accscalar_t>())[0];
    } else {
      value = group_y_reduce(
          item, shared_, vec_t(value), ReduceAdd<accscalar_t>())[0];
    }

    if (id.glb_problem < cfg_.problem_ && id.glb_batch < cfg_.problem_batch_) {
      if (id.chunk_off == 0) {
        oinfo_.data[str_off] = is_final_ ? sqrtf(value) : value;
      }
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    shared_ = sycl_local_acc_t<accscalar_t>(shared_memeory_size_, cgh);
  }
  WeightNormReduceKernelFunctor(
      ScalarTypeInfo iinfo,
      AccTypeInfo oinfo,
      BatchKernelConfig cfg,
      bool need_squre,
      bool is_final,
      int64_t shared_memeory_size)
      : iinfo_(iinfo),
        oinfo_(oinfo),
        cfg_(cfg),
        need_squre_(need_squre),
        is_final_(is_final),
        shared_memeory_size_(shared_memeory_size) {}

 private:
  ScalarTypeInfo iinfo_;
  AccTypeInfo oinfo_;
  BatchKernelConfig cfg_;
  bool need_squre_;
  bool is_final_;
  int64_t shared_memeory_size_;
  sycl_local_acc_t<accscalar_t> shared_;
};

template <class ScalarTypeInfo, class AccTypeInfo>
static inline void launch_weight_norm_reduce_kernel(
    ScalarTypeInfo& iinfo,
    AccTypeInfo& oinfo,
    BatchKernelConfig& cfg,
    bool need_squre,
    bool is_final) {
  using scalar_t = typename ScalarTypeInfo::scalar_t;
  using accscalar_t = typename AccTypeInfo::scalar_t;
  using vec_t = at::detail::Array<accscalar_t, 1>;

  WeightNormReduceKernelFunctor<
      ScalarTypeInfo,
      AccTypeInfo,
      scalar_t,
      accscalar_t,
      vec_t>
      kfn(iinfo, oinfo, cfg, need_squre, is_final, cfg.group_size().size());
  sycl_kernel_submit(
      cfg.global_size(), cfg.group_size(), getCurrentSYCLQueue(), kfn);
}

template <class ScalarTypeInfo, class AccTypeInfo>
static inline void weight_norm_reduce(
    ScalarTypeInfo& vinfo,
    AccTypeInfo& ninfo,
    int dim_after_collapse,
    bool need_square) {
  int64_t batch = vinfo.outerSize(dim_after_collapse);
  int64_t problem = vinfo.sizes[dim_after_collapse];
  int64_t stride = vinfo.innerSize(dim_after_collapse);
  bool problem_along_x = vinfo.strides[dim_after_collapse] == 1 ? true : false;
  using scalar_t = typename ScalarTypeInfo::scalar_t;
  using accscalar_t = typename AccTypeInfo::scalar_t;
  using vec_t = at::detail::Array<accscalar_t, 1>;
  using KernelClass = WeightNormReduceKernelFunctor<
      ScalarTypeInfo,
      AccTypeInfo,
      scalar_t,
      accscalar_t,
      vec_t>;
  BatchKernelConfig cfg = BatchKernelConfig::make_config<KernelClass>(
      batch, problem, stride, batch * stride, problem_along_x);

  if (cfg.problem_ <= cfg.problem_wg_range_) {
    launch_weight_norm_reduce_kernel(vinfo, ninfo, cfg, need_square, true);
    return;
  }

  Tensor carrier = at::empty(
      {cfg.batch_, cfg.problem_glb_range_ / cfg.problem_wg_range_, cfg.stride_},
      map_options<typename AccTypeInfo::scalar_t>());
  auto cinfo =
      at::xpu::detail::getTensorInfo<typename AccTypeInfo::scalar_t, int64_t>(
          carrier);
  launch_weight_norm_reduce_kernel(vinfo, cinfo, cfg, need_square, false);

  weight_norm_reduce(cinfo, ninfo, 1, false);
  return;
}

template <
    class ScalarTypeInfo,
    class AccTypeInfo,
    typename scalar_t,
    typename accscalar_t>
struct SegmentWeightNormKernelFunctor {
  void operator()(sycl::nd_item<2> item) const {
    auto id = cfg_.get_item_desc(item);
    int64_t si = id.glb_batch % cfg_.stride_;
    int64_t bi = id.glb_batch / cfg_.stride_;
    int64_t pi = id.chunk * id.chunk_size + id.chunk_off;
    int64_t w_lid = si + pi * cfg_.stride_ + bi * cfg_.problem_ * cfg_.stride_;
    int64_t n_lid = id.glb_batch;

    int64_t v_off = at::xpu::detail::IndexToOffset<scalar_t, int64_t, -1>::get(
        w_lid, vinfo_);
    int64_t w_off = at::xpu::detail::IndexToOffset<scalar_t, int64_t, -1>::get(
        w_lid, winfo_);
    int64_t g_off = at::xpu::detail::IndexToOffset<scalar_t, int64_t, -1>::get(
        n_lid, ginfo_);
    int64_t n_off =
        at::xpu::detail::IndexToOffset<accscalar_t, int64_t, -1>::get(
            n_lid, ninfo_);

    if (id.glb_problem < cfg_.problem_ && id.glb_batch < cfg_.problem_batch_) {
      winfo_.data[w_off] =
          (1.f / ninfo_.data[n_off]) * vinfo_.data[v_off] * ginfo_.data[g_off];
    }
  }
  SegmentWeightNormKernelFunctor(
      ScalarTypeInfo vinfo,
      ScalarTypeInfo ginfo,
      ScalarTypeInfo winfo,
      AccTypeInfo ninfo,
      BatchKernelConfig cfg)
      : vinfo_(vinfo), ginfo_(ginfo), winfo_(winfo), ninfo_(ninfo), cfg_(cfg) {}

 private:
  ScalarTypeInfo vinfo_;
  ScalarTypeInfo ginfo_;
  ScalarTypeInfo winfo_;
  AccTypeInfo ninfo_;
  BatchKernelConfig cfg_;
};

template <class ScalarTypeInfo, class AccTypeInfo>
static inline void segment_weight_norm(
    ScalarTypeInfo& vinfo,
    ScalarTypeInfo& ginfo,
    ScalarTypeInfo& winfo,
    AccTypeInfo& ninfo,
    int dim_after_collapse) {
  // segment reduce for statistics
  weight_norm_reduce(vinfo, ninfo, dim_after_collapse, true);

  // normalization
  int64_t batch = vinfo.outerSize(dim_after_collapse);
  int64_t problem = vinfo.sizes[dim_after_collapse];
  int64_t stride = vinfo.innerSize(dim_after_collapse);
  bool problem_along_x = vinfo.strides[dim_after_collapse] == 1 ? true : false;
  using scalar_t = typename ScalarTypeInfo::scalar_t;
  using accscalar_t = typename AccTypeInfo::scalar_t;

  using KernelClass = SegmentWeightNormKernelFunctor<
      ScalarTypeInfo,
      AccTypeInfo,
      scalar_t,
      accscalar_t>;
  BatchKernelConfig cfg = BatchKernelConfig::make_config<KernelClass>(
      batch, problem, stride, batch * stride, problem_along_x);

  KernelClass kfn(vinfo, ginfo, winfo, ninfo, cfg);
  sycl_kernel_submit(
      cfg.global_size(), cfg.group_size(), getCurrentSYCLQueue(), kfn);
}

template <
    class ScalarTypeInfo,
    class AccTypeInfo,
    typename scalar_t,
    typename accscalar_t,
    typename vec_t>
struct WeightNormKernelFunctor : public __SYCL_KER_CONFIG_CONVENTION__ {
  void operator()(sycl::nd_item<2> item) const {
    auto id = cfg_.get_item_desc(item);
    int64_t n_lid = id.glb_batch;

    int64_t g_off = at::xpu::detail::IndexToOffset<scalar_t, int64_t, -1>::get(
        n_lid, ginfo_);

    int64_t n_off =
        at::xpu::detail::IndexToOffset<accscalar_t, int64_t, -1>::get(
            n_lid, ninfo_);

    int64_t si = id.glb_batch % cfg_.stride_;
    int64_t bi = id.glb_batch / cfg_.stride_;
    int64_t pi = id.chunk_off;
    bi = si + bi * cfg_.problem_ * cfg_.stride_;

    accscalar_t value = 0;
    if (id.glb_batch < cfg_.problem_batch_) {
      for (int pi_ = pi; pi_ < cfg_.problem_; pi_ += cfg_.problem_wg_range_) {
        int64_t v_lid = bi + pi_ * cfg_.stride_;
        int64_t v_off =
            at::xpu::detail::IndexToOffset<scalar_t, int64_t, -1>::get(
                v_lid, vinfo_);

        accscalar_t v = (accscalar_t)vinfo_.data[v_off];
        value += v * v;
      }
    }

    if (cfg_.problem_along_x_) {
      value = group_x_reduce(
          item, shared_, vec_t(value), ReduceAdd<accscalar_t>())[0];
    } else {
      value = group_y_reduce(
          item, shared_, vec_t(value), ReduceAdd<accscalar_t>())[0];
    }

    int n_slid = (int)id.glb_batch % batch_wg_range_;
    if (id.glb_batch < cfg_.problem_batch_ && id.chunk_off == 0) {
      value = sqrtf(value);
      ninfo_.data[n_off] = value;
      shared_[n_slid] = value;
    }
    // Here using slm instead. If using ugm, need fence w/
    // order:acq_rel & scope:workgroup & space:global_mem.
    sycl::group_barrier(item.get_group());

    if (id.glb_batch < cfg_.problem_batch_) {
      for (int pi_ = pi; pi_ < cfg_.problem_; pi_ += cfg_.problem_wg_range_) {
        int64_t v_lid = bi + pi_ * cfg_.stride_;
        int64_t v_off =
            at::xpu::detail::IndexToOffset<scalar_t, int64_t, -1>::get(
                v_lid, vinfo_);
        int64_t w_off =
            at::xpu::detail::IndexToOffset<scalar_t, int64_t, -1>::get(
                v_lid, winfo_);

        winfo_.data[w_off] =
            (1.f / shared_[n_slid]) * vinfo_.data[v_off] * ginfo_.data[g_off];
      }
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    shared_ = sycl_local_acc_t<accscalar_t>(wg_size_, cgh);
  }

  WeightNormKernelFunctor(
      ScalarTypeInfo vinfo,
      ScalarTypeInfo ginfo,
      ScalarTypeInfo winfo,
      AccTypeInfo ninfo,
      BatchKernelConfig cfg,
      int wg_size,
      int batch_wg_range)
      : vinfo_(vinfo),
        ginfo_(ginfo),
        winfo_(winfo),
        ninfo_(ninfo),
        cfg_(cfg),
        wg_size_(wg_size),
        batch_wg_range_(batch_wg_range) {}

 private:
  ScalarTypeInfo vinfo_;
  ScalarTypeInfo ginfo_;
  ScalarTypeInfo winfo_;
  AccTypeInfo ninfo_;
  BatchKernelConfig cfg_;
  int wg_size_;
  int batch_wg_range_;
  sycl_local_acc_t<accscalar_t> shared_;
};

template <class ScalarTypeInfo, class AccTypeInfo>
static inline void weight_norm(
    ScalarTypeInfo& vinfo,
    ScalarTypeInfo& ginfo,
    ScalarTypeInfo& winfo,
    AccTypeInfo& ninfo,
    int dim_after_collapse) {
  int64_t batch = vinfo.outerSize(dim_after_collapse);
  int64_t problem = vinfo.sizes[dim_after_collapse];
  int64_t stride = vinfo.innerSize(dim_after_collapse);
  bool problem_along_x = vinfo.strides[dim_after_collapse] == 1 ? true : false;
  using scalar_t = typename ScalarTypeInfo::scalar_t;
  using accscalar_t = typename AccTypeInfo::scalar_t;
  using vec_t = at::detail::Array<accscalar_t, 1>;

  using KernelClass = WeightNormKernelFunctor<
      ScalarTypeInfo,
      AccTypeInfo,
      scalar_t,
      accscalar_t,
      vec_t>;
  BatchKernelConfig cfg = BatchKernelConfig::make_config<KernelClass>(
      batch,
      problem,
      stride,
      batch * stride,
      problem_along_x,
      BatchKernelConfig::Policy::pLoop);

  int wg_size = cfg.group_size().size();
  int batch_wg_range = wg_size / cfg.problem_wg_range_;
  KernelClass kfn(vinfo, ginfo, winfo, ninfo, cfg, wg_size, batch_wg_range);
  sycl_kernel_submit(
      cfg.global_size(), cfg.group_size(), getCurrentSYCLQueue(), kfn);

  return;
}

std::tuple<Tensor, Tensor> weight_norm_kernel(
    const Tensor& v,
    const Tensor& g,
    int64_t dim) {
  TORCH_INTERNAL_ASSERT(
      dim == 0 || dim == v.dim() - 1,
      "fused kernels can only be applied for first or last dim");

  at::ScalarType scalar_acc_t = (g.scalar_type() == at::ScalarType::Half ||
                                 g.scalar_type() == at::ScalarType::BFloat16)
      ? at::ScalarType::Float
      : g.scalar_type();
  auto norms = at::empty(
      g.sizes(), g.options().dtype(scalar_acc_t), g.suggest_memory_format());
  auto w = at::empty(v.sizes(), v.options(), v.suggest_memory_format());

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      v.scalar_type(),
      "aten::weight_norm",
      [&] {
        auto vinfo = at::xpu::detail::getTensorInfo<scalar_t, int64_t>(v);
        int dim_after_collapse = vinfo.collapseDims(dim);
        auto ginfo = at::xpu::detail::getTensorInfo<scalar_t, int64_t>(g);
        ginfo.collapseDims();

        auto winfo = at::xpu::detail::getTensorInfo<scalar_t, int64_t>(w);
        winfo.collapseDims(dim);
        using accscalar_t = acc_type<scalar_t, true>;
        auto ninfo =
            at::xpu::detail::getTensorInfo<accscalar_t, int64_t>(norms);
        ninfo.collapseDims();
        dim_after_collapse = 1 - dim_after_collapse; // remain dim

        int64_t batch = vinfo.outerSize(dim_after_collapse);
        int64_t problem = vinfo.sizes[dim_after_collapse];
        int64_t stride = vinfo.innerSize(dim_after_collapse);
        bool problem_along_x =
            vinfo.strides[dim_after_collapse] == 1 ? true : false;
        if (BatchKernelConfig::Policy::pSegment ==
            BatchKernelConfig::suggest_policy(
                batch, problem, stride, problem_along_x)) {
          segment_weight_norm(vinfo, ginfo, winfo, ninfo, dim_after_collapse);
        } else {
          weight_norm(vinfo, ginfo, winfo, ninfo, dim_after_collapse);
        }
      });

  return {w, norms};
}

template <
    bool is_first,
    class ScalarType1Info,
    class ScalarType2Info,
    class AccTypeInfo,
    typename scalar1_t,
    typename scalar2_t,
    typename accscalar_t,
    typename vec_t>
struct WeightNormBackwardReduceKernelFunctor
    : public __SYCL_KER_CONFIG_CONVENTION__ {
  void operator()(sycl::nd_item<2> item) const {
    auto id = cfg_.get_item_desc(item);
    int64_t si = id.glb_batch % cfg_.stride_;
    int64_t bi = id.glb_batch / cfg_.stride_;
    int64_t i_pi = id.chunk * id.chunk_size + id.chunk_off;
    int64_t o_pi = id.chunk;

    int64_t i_lid =
        si + i_pi * cfg_.stride_ + bi * cfg_.problem_ * cfg_.stride_;
    int64_t i1_off =
        at::xpu::detail::IndexToOffset<scalar1_t, int64_t, -1>::get(
            i_lid, i1info_);
    int64_t i2_off;
    if (is_first) {
      i2_off = at::xpu::detail::IndexToOffset<scalar2_t, int64_t, -1>::get(
          i_lid, i2info_);
    }

    int64_t o_lid = si + o_pi * cfg_.stride_ + bi * id.chunk_num * cfg_.stride_;
    int64_t o_off =
        at::xpu::detail::IndexToOffset<accscalar_t, int64_t, -1>::get(
            o_lid, oinfo_);

    accscalar_t value = 0;
    if (id.glb_problem < cfg_.problem_ && id.glb_batch < cfg_.problem_batch_) {
      if (is_first) {
        auto value1 = (accscalar_t)i1info_.data[i1_off];
        auto value2 = (accscalar_t)i2info_.data[i2_off];
        value = value1 * value2;
      } else {
        value = (accscalar_t)i1info_.data[i1_off];
      }
    }

    if (cfg_.problem_along_x_) {
      value = group_x_reduce(
          item, shared_, vec_t(value), ReduceAdd<accscalar_t>())[0];
    } else {
      value = group_y_reduce(
          item, shared_, vec_t(value), ReduceAdd<accscalar_t>())[0];
    }

    if (id.glb_problem < cfg_.problem_ && id.glb_batch < cfg_.problem_batch_) {
      if (id.chunk_off == 0) {
        oinfo_.data[o_off] = value;
      }
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    shared_ = sycl_local_acc_t<accscalar_t>(local_size_, cgh);
  }
  WeightNormBackwardReduceKernelFunctor(
      ScalarType1Info i1info,
      ScalarType2Info i2info,
      AccTypeInfo oinfo,
      BatchKernelConfig cfg,
      int64_t local_size)
      : i1info_(i1info),
        i2info_(i2info),
        oinfo_(oinfo),
        cfg_(cfg),
        local_size_(local_size) {}

 private:
  ScalarType1Info i1info_;
  ScalarType2Info i2info_;
  AccTypeInfo oinfo_;
  BatchKernelConfig cfg_;
  int64_t local_size_;
  sycl_local_acc_t<accscalar_t> shared_;
};

template <
    bool is_first,
    class ScalarType1Info,
    class ScalarType2Info,
    class AccTypeInfo>
static inline void launch_weight_norm_backward_reduce_kernel(
    ScalarType1Info& i1info,
    ScalarType2Info& i2info,
    AccTypeInfo& oinfo,
    BatchKernelConfig& cfg) {
  using scalar1_t = typename ScalarType1Info::scalar_t;
  using scalar2_t = typename ScalarType2Info::scalar_t;
  using accscalar_t = typename AccTypeInfo::scalar_t;
  using vec_t = at::detail::Array<accscalar_t, 1>;
  WeightNormBackwardReduceKernelFunctor<
      is_first,
      ScalarType1Info,
      ScalarType2Info,
      AccTypeInfo,
      scalar1_t,
      scalar2_t,
      accscalar_t,
      vec_t>
      kfn(i1info, i2info, oinfo, cfg, cfg.group_size().size());
  sycl_kernel_submit(
      cfg.global_size(), cfg.group_size(), getCurrentSYCLQueue(), kfn);
}

template <class ScalarType1Info, class ScalarType2Info, class AccTypeInfo>
static inline void weight_norm_backward_reduce(
    ScalarType1Info& vinfo,
    ScalarType2Info& gwinfo,
    AccTypeInfo& rinfo,
    int dim_after_collapse,
    bool is_first) {
  int64_t batch = vinfo.outerSize(dim_after_collapse);
  int64_t problem = vinfo.sizes[dim_after_collapse];
  int64_t stride = vinfo.innerSize(dim_after_collapse);
  bool problem_along_x = vinfo.strides[dim_after_collapse] == 1 ? true : false;

  using scalar1_t = typename ScalarType1Info::scalar_t;
  using scalar2_t = typename ScalarType2Info::scalar_t;
  using accscalar_t = typename AccTypeInfo::scalar_t;
  using vec_t = at::detail::Array<accscalar_t, 1>;
  using KernelClass = WeightNormBackwardReduceKernelFunctor<
      true,
      ScalarType1Info,
      ScalarType2Info,
      AccTypeInfo,
      scalar1_t,
      scalar2_t,
      accscalar_t,
      vec_t>;
  BatchKernelConfig cfg = BatchKernelConfig::make_config<KernelClass>(
      batch, problem, stride, batch * stride, problem_along_x);
  if (cfg.problem_ <= cfg.problem_wg_range_) {
    if (is_first) {
      launch_weight_norm_backward_reduce_kernel<true>(
          vinfo, gwinfo, rinfo, cfg);
    } else {
      launch_weight_norm_backward_reduce_kernel<false>(
          vinfo, gwinfo, rinfo, cfg);
    }
    return;
  }

  Tensor carrier = at::empty(
      {cfg.batch_, cfg.problem_glb_range_ / cfg.problem_wg_range_, cfg.stride_},
      map_options<typename AccTypeInfo::scalar_t>());
  auto cinfo =
      at::xpu::detail::getTensorInfo<typename AccTypeInfo::scalar_t, int64_t>(
          carrier);
  if (is_first) {
    launch_weight_norm_backward_reduce_kernel<true>(vinfo, gwinfo, cinfo, cfg);
  } else {
    launch_weight_norm_backward_reduce_kernel<false>(vinfo, gwinfo, cinfo, cfg);
  }

  weight_norm_backward_reduce(cinfo, gwinfo, rinfo, 1, false);
  return;
}

template <
    class ScalarTypeInfo,
    class AccTypeInfo,
    typename scalar_t,
    typename accscalar_t>
struct SegmentWeightNormBackwardKernelFunctor {
  void operator()(sycl::nd_item<2> item) const {
    auto id = cfg_.get_item_desc(item);

    int64_t si = id.glb_batch % cfg_.stride_;
    int64_t bi = id.glb_batch / cfg_.stride_;
    int64_t pi = id.chunk * id.chunk_size + id.chunk_off;

    int64_t gv_lid = si + pi * cfg_.stride_ + bi * cfg_.problem_ * cfg_.stride_;
    int64_t gg_lid = id.glb_batch;

    int64_t v_off = at::xpu::detail::IndexToOffset<scalar_t, int64_t, -1>::get(
        gv_lid, vinfo_);

    int64_t gw_off = at::xpu::detail::IndexToOffset<scalar_t, int64_t, -1>::get(
        gv_lid, gwinfo_);

    int64_t gv_off = at::xpu::detail::IndexToOffset<scalar_t, int64_t, -1>::get(
        gv_lid, gvinfo_);

    int64_t g_off = at::xpu::detail::IndexToOffset<scalar_t, int64_t, -1>::get(
        gg_lid, ginfo_);

    int64_t n_off =
        at::xpu::detail::IndexToOffset<accscalar_t, int64_t, -1>::get(
            gg_lid, ninfo_);

    int64_t r_off =
        at::xpu::detail::IndexToOffset<accscalar_t, int64_t, -1>::get(
            gg_lid, rinfo_);

    int64_t gg_off = at::xpu::detail::IndexToOffset<scalar_t, int64_t, -1>::get(
        gg_lid, gginfo_);

    if (id.glb_problem < cfg_.problem_ && id.glb_batch < cfg_.problem_batch_) {
      accscalar_t g = ginfo_.data[g_off];
      accscalar_t gw = gwinfo_.data[gw_off];
      accscalar_t v = vinfo_.data[v_off];
      accscalar_t n = 1.f / ninfo_.data[n_off];
      accscalar_t r = rinfo_.data[r_off];
      accscalar_t gg = r * n;
      accscalar_t n3 = n * n * n;
      accscalar_t gv = g * (n * gw - n3 * v * r);

      gvinfo_.data[gv_off] = static_cast<scalar_t>(gv);
      if (id.chunk == 0 && id.chunk_off == 0)
        gginfo_.data[gg_off] = static_cast<scalar_t>(gg);
    }
  }
  SegmentWeightNormBackwardKernelFunctor(
      ScalarTypeInfo vinfo,
      ScalarTypeInfo ginfo,
      ScalarTypeInfo gwinfo,
      AccTypeInfo ninfo,
      ScalarTypeInfo gvinfo,
      ScalarTypeInfo gginfo,
      AccTypeInfo rinfo,
      BatchKernelConfig cfg)
      : vinfo_(vinfo),
        ginfo_(ginfo),
        gwinfo_(gwinfo),
        ninfo_(ninfo),
        gvinfo_(gvinfo),
        gginfo_(gginfo),
        rinfo_(rinfo),
        cfg_(cfg) {}

 private:
  ScalarTypeInfo vinfo_;
  ScalarTypeInfo ginfo_;
  ScalarTypeInfo gwinfo_;
  AccTypeInfo ninfo_;
  ScalarTypeInfo gvinfo_;
  ScalarTypeInfo gginfo_;
  AccTypeInfo rinfo_;
  BatchKernelConfig cfg_;
};

template <class ScalarTypeInfo, class AccTypeInfo>
static inline void segment_weight_norm_backward(
    ScalarTypeInfo& vinfo,
    ScalarTypeInfo& ginfo,
    ScalarTypeInfo& gwinfo,
    AccTypeInfo& ninfo,
    ScalarTypeInfo& gvinfo,
    ScalarTypeInfo& gginfo,
    AccTypeInfo& rinfo,
    int dim_after_collapse) {
  // segment reduce
  weight_norm_backward_reduce(vinfo, gwinfo, rinfo, dim_after_collapse, true);

  // compute gradient
  int64_t batch = vinfo.outerSize(dim_after_collapse);
  int64_t problem = vinfo.sizes[dim_after_collapse];
  int64_t stride = vinfo.innerSize(dim_after_collapse);
  bool problem_along_x = vinfo.strides[dim_after_collapse] == 1 ? true : false;

  using scalar_t = typename ScalarTypeInfo::scalar_t;
  using accscalar_t = typename AccTypeInfo::scalar_t;
  using KernelClass = SegmentWeightNormBackwardKernelFunctor<
      ScalarTypeInfo,
      AccTypeInfo,
      scalar_t,
      accscalar_t>;
  BatchKernelConfig cfg = BatchKernelConfig::make_config<KernelClass>(
      batch, problem, stride, batch * stride, problem_along_x);

  KernelClass kfn(vinfo, ginfo, gwinfo, ninfo, gvinfo, gginfo, rinfo, cfg);
  sycl_kernel_submit(
      cfg.global_size(), cfg.group_size(), getCurrentSYCLQueue(), kfn);

  return;
}

template <
    class ScalarTypeInfo,
    class AccTypeInfo,
    typename scalar_t,
    typename accscalar_t,
    typename vec_t>
struct WeightNormBackwardKernelFunctor : public __SYCL_KER_CONFIG_CONVENTION__ {
  void operator()(sycl::nd_item<2> item) const {
    auto id = cfg_.get_item_desc(item);
    int64_t n_lid = id.glb_batch;
    int64_t g_off = at::xpu::detail::IndexToOffset<scalar_t, int64_t, -1>::get(
        n_lid, ginfo_);
    int64_t gg_off = at::xpu::detail::IndexToOffset<scalar_t, int64_t, -1>::get(
        n_lid, gginfo_);
    int64_t n_off =
        at::xpu::detail::IndexToOffset<accscalar_t, int64_t, -1>::get(
            n_lid, ninfo_);
    int64_t si = id.glb_batch % cfg_.stride_;
    int64_t bi = id.glb_batch / cfg_.stride_;
    int64_t pi = id.chunk_off;
    bi = si + bi * cfg_.problem_ * cfg_.stride_;

    accscalar_t value = 0;
    if (id.glb_batch < cfg_.problem_batch_) {
      for (int pi_ = pi; pi_ < cfg_.problem_; pi_ += cfg_.problem_wg_range_) {
        int64_t v_lid, v_off, gw_off;
        v_lid = bi + pi_ * cfg_.stride_;

        v_off = at::xpu::detail::IndexToOffset<scalar_t, int64_t, -1>::get(
            v_lid, vinfo_);

        gw_off = at::xpu::detail::IndexToOffset<scalar_t, int64_t, -1>::get(
            v_lid, gwinfo_);

        accscalar_t v = (accscalar_t)vinfo_.data[v_off];
        accscalar_t gw = (accscalar_t)gwinfo_.data[gw_off];
        value += v * gw;
      }
    }

    if (cfg_.problem_along_x_) {
      value = group_x_reduce(
          item, shared_, vec_t(value), ReduceAdd<accscalar_t>())[0];
    } else {
      value = group_y_reduce(
          item, shared_, vec_t(value), ReduceAdd<accscalar_t>())[0];
    }

    int n_slid = (int)id.glb_batch % batch_wg_range_;
    if (id.glb_batch < cfg_.problem_batch_ && id.chunk_off == 0) {
      shared_[n_slid] = value;
    }
    sycl::group_barrier(item.get_group());

    if (id.glb_batch < cfg_.problem_batch_) {
      for (int pi_ = pi; pi_ < cfg_.problem_; pi_ += cfg_.problem_wg_range_) {
        int64_t v_lid, v_off, gw_off, gv_off;
        v_lid = bi + pi_ * cfg_.stride_;

        v_off = at::xpu::detail::IndexToOffset<scalar_t, int64_t, -1>::get(
            v_lid, vinfo_);

        gw_off = at::xpu::detail::IndexToOffset<scalar_t, int64_t, -1>::get(
            v_lid, gwinfo_);

        gv_off = at::xpu::detail::IndexToOffset<scalar_t, int64_t, -1>::get(
            v_lid, gvinfo_);

        accscalar_t g = ginfo_.data[g_off];
        accscalar_t gw = gwinfo_.data[gw_off];
        accscalar_t v = vinfo_.data[v_off];
        accscalar_t n = 1.f / ninfo_.data[n_off];
        accscalar_t r = shared_[n_slid];
        accscalar_t gg = r * n;
        accscalar_t n3 = n * n * n;
        accscalar_t gv = g * (n * gw - n3 * v * r);

        gvinfo_.data[gv_off] = static_cast<scalar_t>(gv);
        if (id.chunk_off == 0)
          gginfo_.data[gg_off] = static_cast<scalar_t>(gg);
      }
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    shared_ = sycl_local_acc_t<accscalar_t>(wg_size_, cgh);
  }

  WeightNormBackwardKernelFunctor(
      ScalarTypeInfo vinfo,
      ScalarTypeInfo ginfo,
      ScalarTypeInfo gwinfo,
      AccTypeInfo ninfo,
      ScalarTypeInfo gvinfo,
      ScalarTypeInfo gginfo,
      BatchKernelConfig cfg,
      int wg_size,
      int batch_wg_range)
      : vinfo_(vinfo),
        ginfo_(ginfo),
        gwinfo_(gwinfo),
        ninfo_(ninfo),
        gvinfo_(gvinfo),
        gginfo_(gginfo),
        cfg_(cfg),
        wg_size_(wg_size),
        batch_wg_range_(batch_wg_range) {}

 private:
  ScalarTypeInfo vinfo_;
  ScalarTypeInfo ginfo_;
  ScalarTypeInfo gwinfo_;
  AccTypeInfo ninfo_;
  ScalarTypeInfo gvinfo_;
  ScalarTypeInfo gginfo_;
  BatchKernelConfig cfg_;
  int wg_size_;
  int batch_wg_range_;
  sycl_local_acc_t<accscalar_t> shared_;
};

template <class ScalarTypeInfo, class AccTypeInfo>
static inline void weight_norm_backward(
    ScalarTypeInfo& vinfo,
    ScalarTypeInfo& ginfo,
    ScalarTypeInfo& gwinfo,
    AccTypeInfo& ninfo,
    ScalarTypeInfo& gvinfo,
    ScalarTypeInfo& gginfo,
    int dim_after_collapse) {
  int64_t batch = vinfo.outerSize(dim_after_collapse);
  int64_t problem = vinfo.sizes[dim_after_collapse];
  int64_t stride = vinfo.innerSize(dim_after_collapse);
  bool problem_along_x = vinfo.strides[dim_after_collapse] == 1 ? true : false;

  using scalar_t = typename ScalarTypeInfo::scalar_t;
  using accscalar_t = typename AccTypeInfo::scalar_t;
  using vec_t = at::detail::Array<accscalar_t, 1>;
  using KernelClass = WeightNormBackwardKernelFunctor<
      ScalarTypeInfo,
      AccTypeInfo,
      scalar_t,
      accscalar_t,
      vec_t>;
  BatchKernelConfig cfg = BatchKernelConfig::make_config<KernelClass>(
      batch,
      problem,
      stride,
      batch * stride,
      problem_along_x,
      BatchKernelConfig::Policy::pLoop);
  int wg_size = cfg.group_size().size();
  int batch_wg_range = wg_size / cfg.problem_wg_range_;
  KernelClass kfn(
      vinfo,
      ginfo,
      gwinfo,
      ninfo,
      gvinfo,
      gginfo,
      cfg,
      wg_size,
      batch_wg_range);
  sycl_kernel_submit(
      cfg.global_size(), cfg.group_size(), getCurrentSYCLQueue(), kfn);
  return;
}

std::tuple<Tensor, Tensor> weight_norm_backward_kernel(
    const Tensor& grad_w,
    const Tensor& saved_v,
    const Tensor& saved_g,
    const Tensor& saved_norms,
    int64_t dim) {
  auto grad_v = at::empty_like(saved_v, c10::get_contiguous_memory_format());
  auto grad_g = at::empty_like(saved_g, c10::get_contiguous_memory_format());

  at::ScalarType scalar_acc_t =
      (saved_g.scalar_type() == at::ScalarType::Half ||
       saved_g.scalar_type() == at::ScalarType::BFloat16)
      ? at::ScalarType::Float
      : saved_g.scalar_type();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      saved_v.scalar_type(),
      "aten::weight_norm_backward",
      [&] {
        auto vinfo = at::xpu::detail::getTensorInfo<scalar_t, int64_t>(saved_v);
        int dim_after_collapse = vinfo.collapseDims(dim);

        auto ginfo = at::xpu::detail::getTensorInfo<scalar_t, int64_t>(saved_g);
        ginfo.collapseDims();

        auto gwinfo = at::xpu::detail::getTensorInfo<scalar_t, int64_t>(grad_w);
        gwinfo.collapseDims(dim);
        using accscalar_t = acc_type<scalar_t, true>;
        auto ninfo =
            at::xpu::detail::getTensorInfo<accscalar_t, int64_t>(saved_norms);
        ninfo.collapseDims();

        auto gvinfo = at::xpu::detail::getTensorInfo<scalar_t, int64_t>(grad_v);
        gvinfo.collapseDims(dim);

        auto gginfo = at::xpu::detail::getTensorInfo<scalar_t, int64_t>(grad_g);
        gginfo.collapseDims();

        dim_after_collapse = 1 - dim_after_collapse; // remain dim

        int64_t batch = vinfo.outerSize(dim_after_collapse);
        int64_t problem = vinfo.sizes[dim_after_collapse];
        int64_t stride = vinfo.innerSize(dim_after_collapse);
        bool problem_along_x =
            vinfo.strides[dim_after_collapse] == 1 ? true : false;
        if (BatchKernelConfig::Policy::pSegment ==
            BatchKernelConfig::suggest_policy(
                batch, problem, stride, problem_along_x)) {
          auto reduce = at::empty(
              saved_g.sizes(),
              saved_g.options().dtype(scalar_acc_t),
              c10::get_contiguous_memory_format());
          auto rinfo =
              at::xpu::detail::getTensorInfo<accscalar_t, int64_t>(reduce);
          rinfo.collapseDims();

          segment_weight_norm_backward(
              vinfo,
              ginfo,
              gwinfo,
              ninfo,
              gvinfo,
              gginfo,
              rinfo,
              dim_after_collapse);
        } else {
          weight_norm_backward(
              vinfo, ginfo, gwinfo, ninfo, gvinfo, gginfo, dim_after_collapse);
        }
      });

  return {grad_v, grad_g};
}

} // namespace at::native::xpu
