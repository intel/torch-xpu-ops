/*
 * Copyright 2020-2026 Intel Corporation
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
#include <comm/Runtime.h>
#include <comm/SYCLContext.h>
#include <comm/TensorInfo.h>
#include <comm/TensorOptions.h>

#include <ATen/native/xpu/sycl/WeightNormKernels.h>

#include <limits>

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
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<2>))
void weight_norm_reduce_kernel(
    ScalarTypeInfo iinfo,
    AccTypeInfo oinfo,
    BatchKernelConfig cfg,
    bool need_squre,
    bool is_final) {
  auto item = syclext::this_work_item::get_nd_item<2>();
  char* shared_ptr =
      static_cast<char*>(syclexp::get_work_group_scratch_memory());
  
  auto id = cfg.get_item_desc(item);
  int64_t si = id.glb_batch % cfg.stride_;
  int64_t bi = id.glb_batch / cfg.stride_;
  int64_t ldr_pi = id.chunk * id.chunk_size + id.chunk_off;
  int64_t str_pi = id.chunk;
  int64_t ldr_lid = si + ldr_pi * cfg.stride_ + bi * cfg.problem_ * cfg.stride_;
  int64_t ldr_off = at::xpu::detail::IndexToOffset<scalar_t, int64_t, -1>::get(
      ldr_lid, iinfo);
  int64_t str_lid = si + str_pi * cfg.stride_ + bi * id.chunk_num * cfg.stride_;
  int64_t str_off =
      at::xpu::detail::IndexToOffset<accscalar_t, int64_t, -1>::get(
          str_lid, oinfo);

  accscalar_t value = 0;
  if (id.glb_problem < cfg.problem_ && id.glb_batch < cfg.problem_batch_) {
    value = (accscalar_t)iinfo.data[ldr_off];
    if (need_squre)
      value *= value;
  }

  if (cfg.problem_along_x_) {
    value = group_x_reduce(
        item, shared_ptr, vec_t(value), ReduceAdd<accscalar_t>())[0];
  } else {
    value = group_y_reduce(
        item, shared_ptr, vec_t(value), ReduceAdd<accscalar_t>())[0];
  }

  if (id.glb_problem < cfg.problem_ && id.glb_batch < cfg.problem_batch_) {
    if (id.chunk_off == 0) {
      oinfo.data[str_off] = is_final ? std::sqrt(value) : value;
    }
  }
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
  // static constexpr: MSVC loses a plain constexpr local when it is used as
  // a non-type template argument from inside the launch lambda below
  // (C2065/C2326)
  static constexpr auto kptr = weight_norm_reduce_kernel<
      ScalarTypeInfo,
      AccTypeInfo,
      scalar_t,
      accscalar_t,
      vec_t>;
  BatchKernelConfig cfg = BatchKernelConfig::make_config<kptr>(
      batch,
      problem,
      stride,
      batch * stride,
      problem_along_x,
      {BatchKernelConfig::Policy::pSegment});

  int slm_sz = cfg.group_size().size() * sizeof(accscalar_t);
  auto launch = [&](auto& iinfo, auto& oinfo, bool is_final) {
    sycl_kernel_submit<kptr, 2>(
        cfg.global_size(),
        cfg.group_size(),
        getCurrentSYCLQueue(),
        slm_sz,
        iinfo,
        oinfo,
        cfg,
        need_square,
        is_final);
  };

  if (cfg.problem_ <= cfg.problem_wg_range_) {
    launch(vinfo, ninfo, true);
    return;
  }

  Tensor carrier = at::empty(
      {cfg.batch_, cfg.problem_glb_range_ / cfg.problem_wg_range_, cfg.stride_},
      map_options<typename AccTypeInfo::scalar_t>());
  auto cinfo =
      at::xpu::detail::getTensorInfo<typename AccTypeInfo::scalar_t, int64_t>(
          carrier);
  launch(vinfo, cinfo, false);

  weight_norm_reduce(cinfo, ninfo, 1, false);
  return;
}

template <
    class ScalarTypeInfo,
    class AccTypeInfo,
    typename scalar_t,
    typename accscalar_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<2>))
void segment_weight_norm_kernel(
    ScalarTypeInfo vinfo,
    ScalarTypeInfo ginfo,
    ScalarTypeInfo winfo,
    AccTypeInfo ninfo,
    BatchKernelConfig cfg) {
  auto item = syclext::this_work_item::get_nd_item<2>();
  auto id = cfg.get_item_desc(item);
  int64_t si = id.glb_batch % cfg.stride_;
  int64_t bi = id.glb_batch / cfg.stride_;
  int64_t pi = id.chunk * id.chunk_size + id.chunk_off;
  int64_t w_lid = si + pi * cfg.stride_ + bi * cfg.problem_ * cfg.stride_;
  int64_t n_lid = id.glb_batch;

  int64_t v_off =
      at::xpu::detail::IndexToOffset<scalar_t, int64_t, -1>::get(w_lid, vinfo);
  int64_t w_off =
      at::xpu::detail::IndexToOffset<scalar_t, int64_t, -1>::get(w_lid, winfo);
  int64_t g_off =
      at::xpu::detail::IndexToOffset<scalar_t, int64_t, -1>::get(n_lid, ginfo);
  int64_t n_off = at::xpu::detail::IndexToOffset<accscalar_t, int64_t, -1>::get(
      n_lid, ninfo);

  if (id.glb_problem < cfg.problem_ && id.glb_batch < cfg.problem_batch_) {
    winfo.data[w_off] = (accscalar_t(1) / ninfo.data[n_off]) *
        vinfo.data[v_off] * ginfo.data[g_off];
  }
}

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

  constexpr auto kptr = segment_weight_norm_kernel<
      ScalarTypeInfo,
      AccTypeInfo,
      scalar_t,
      accscalar_t>;
  BatchKernelConfig cfg = BatchKernelConfig::make_config<kptr>(
      batch,
      problem,
      stride,
      batch * stride,
      problem_along_x,
      {BatchKernelConfig::Policy::pSegment});

  sycl_kernel_submit<kptr, 2>(
      cfg.global_size(),
      cfg.group_size(),
      getCurrentSYCLQueue(),
      0,
      vinfo,
      ginfo,
      winfo,
      ninfo,
      cfg);
}

template <
    class ScalarTypeInfo,
    class AccTypeInfo,
    typename scalar_t,
    typename accscalar_t,
    typename vec_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<2>))
void weight_norm_combine_kernel(
    ScalarTypeInfo vinfo,
    ScalarTypeInfo ginfo,
    ScalarTypeInfo winfo,
    AccTypeInfo ninfo,
    BatchKernelConfig cfg,
    int batch_wg_range) {
  auto item = syclext::this_work_item::get_nd_item<2>();
  accscalar_t* shared_ = (accscalar_t*)syclexp::get_work_group_scratch_memory();

  auto id = cfg.get_item_desc(item);
  int64_t n_lid = id.glb_batch;

  int64_t g_off =
      at::xpu::detail::IndexToOffset<scalar_t, int64_t, -1>::get(n_lid, ginfo);

  int64_t n_off = at::xpu::detail::IndexToOffset<accscalar_t, int64_t, -1>::get(
      n_lid, ninfo);

  int64_t si = id.glb_batch % cfg.stride_;
  int64_t bi = id.glb_batch / cfg.stride_;
  int64_t pi = id.chunk_off;
  bi = si + bi * cfg.problem_ * cfg.stride_;

  accscalar_t value = 0;
  if (id.glb_batch < cfg.problem_batch_) {
    for (int pi_ = pi; pi_ < cfg.problem_; pi_ += cfg.problem_wg_range_) {
      int64_t v_lid = bi + pi_ * cfg.stride_;
      int64_t v_off =
          at::xpu::detail::IndexToOffset<scalar_t, int64_t, -1>::get(
              v_lid, vinfo);
      accscalar_t v = (accscalar_t)vinfo.data[v_off];
      value += v * v;
    }
  }

  char* shared_ptr = reinterpret_cast<char*>(shared_);
  if (cfg.problem_along_x_) {
    value = group_x_reduce(
        item, shared_ptr, vec_t(value), ReduceAdd<accscalar_t>())[0];
  } else {
    value = group_y_reduce(
        item, shared_ptr, vec_t(value), ReduceAdd<accscalar_t>())[0];
  }

  int n_slid = (int)id.glb_batch % batch_wg_range;
  if (id.glb_batch < cfg.problem_batch_ && id.chunk_off == 0) {
    value = std::sqrt(value);
    ninfo.data[n_off] = value;
    shared_[n_slid] = value;
  }
  // Here using slm instead. If using ugm, need fence w/
  // order:acq_rel & scope:workgroup & space:global_mem.
  sycl::group_barrier(item.get_group());

  if (id.glb_batch < cfg.problem_batch_) {
    for (int pi_ = pi; pi_ < cfg.problem_; pi_ += cfg.problem_wg_range_) {
      int64_t v_lid = bi + pi_ * cfg.stride_;
      int64_t v_off =
          at::xpu::detail::IndexToOffset<scalar_t, int64_t, -1>::get(
              v_lid, vinfo);
      int64_t w_off =
          at::xpu::detail::IndexToOffset<scalar_t, int64_t, -1>::get(
              v_lid, winfo);

      winfo.data[w_off] = (accscalar_t(1) / shared_[n_slid]) *
          vinfo.data[v_off] * ginfo.data[g_off];
    }
  }
}

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

  constexpr auto kptr = weight_norm_combine_kernel<
      ScalarTypeInfo,
      AccTypeInfo,
      scalar_t,
      accscalar_t,
      vec_t>;
  BatchKernelConfig cfg = BatchKernelConfig::make_config<kptr>(
      batch,
      problem,
      stride,
      batch * stride,
      problem_along_x,
      {BatchKernelConfig::Policy::pLoop});

  int wg_size = cfg.group_size().size();
  int batch_wg_range = wg_size / cfg.problem_wg_range_;
  int slm_sz = wg_size * sizeof(accscalar_t);
  sycl_kernel_submit<kptr, 2>(
      cfg.global_size(),
      cfg.group_size(),
      getCurrentSYCLQueue(),
      slm_sz,
      vinfo,
      ginfo,
      winfo,
      ninfo,
      cfg,
      batch_wg_range);

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

  // Empty v: w is empty, norm of an empty vector is 0 (matches CPU/CUDA).
  if (v.numel() == 0) {
    norms.zero_();
    return {w, norms};
  }

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
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<2>))
void weight_norm_backward_reduce_kernel(
    ScalarType1Info i1info,
    ScalarType2Info i2info,
    AccTypeInfo oinfo,
    BatchKernelConfig cfg) {
  auto item = syclext::this_work_item::get_nd_item<2>();
  char* shared_ptr = (char*)syclexp::get_work_group_scratch_memory();

  auto id = cfg.get_item_desc(item);
  int64_t si = id.glb_batch % cfg.stride_;
  int64_t bi = id.glb_batch / cfg.stride_;
  int64_t i_pi = id.chunk * id.chunk_size + id.chunk_off;
  int64_t o_pi = id.chunk;

  int64_t i_lid = si + i_pi * cfg.stride_ + bi * cfg.problem_ * cfg.stride_;
  int64_t i1_off = at::xpu::detail::IndexToOffset<scalar1_t, int64_t, -1>::get(
      i_lid, i1info);
  int64_t i2_off;
  if (is_first) {
    i2_off = at::xpu::detail::IndexToOffset<scalar2_t, int64_t, -1>::get(
        i_lid, i2info);
  }

  int64_t o_lid = si + o_pi * cfg.stride_ + bi * id.chunk_num * cfg.stride_;
  int64_t o_off = at::xpu::detail::IndexToOffset<accscalar_t, int64_t, -1>::get(
      o_lid, oinfo);

  accscalar_t value = 0;
  if (id.glb_problem < cfg.problem_ && id.glb_batch < cfg.problem_batch_) {
    if (is_first) {
      auto value1 = (accscalar_t)i1info.data[i1_off];
      auto value2 = (accscalar_t)i2info.data[i2_off];
      value = value1 * value2;
    } else {
      value = (accscalar_t)i1info.data[i1_off];
    }
  }

  if (cfg.problem_along_x_) {
    value = group_x_reduce(
        item, shared_ptr, vec_t(value), ReduceAdd<accscalar_t>())[0];
  } else {
    value = group_y_reduce(
        item, shared_ptr, vec_t(value), ReduceAdd<accscalar_t>())[0];
  }

  if (id.glb_problem < cfg.problem_ && id.glb_batch < cfg.problem_batch_) {
    if (id.chunk_off == 0) {
      oinfo.data[o_off] = value;
    }
  }
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
  // static constexpr: MSVC loses a plain constexpr local when it is used as
  // a non-type template argument from inside the launch lambda below
  // (C2065/C2326)
  static constexpr auto kptr_first = weight_norm_backward_reduce_kernel<
      true,
      ScalarType1Info,
      ScalarType2Info,
      AccTypeInfo,
      scalar1_t,
      scalar2_t,
      accscalar_t,
      vec_t>;
  static constexpr auto kptr_rest = weight_norm_backward_reduce_kernel<
      false,
      ScalarType1Info,
      ScalarType2Info,
      AccTypeInfo,
      scalar1_t,
      scalar2_t,
      accscalar_t,
      vec_t>;

  BatchKernelConfig cfg = is_first ? BatchKernelConfig::make_config<kptr_first>(
                                         batch,
                                         problem,
                                         stride,
                                         batch * stride,
                                         problem_along_x,
                                         {BatchKernelConfig::Policy::pSegment})
                                   : BatchKernelConfig::make_config<kptr_rest>(
                                         batch,
                                         problem,
                                         stride,
                                         batch * stride,
                                         problem_along_x,
                                         {BatchKernelConfig::Policy::pSegment});

  int slm_sz = cfg.group_size().size() * sizeof(accscalar_t);
  auto launch = [&](auto& iinfo, auto& oinfo) {
    if (is_first) {
      sycl_kernel_submit<kptr_first, 2>(
          cfg.global_size(),
          cfg.group_size(),
          getCurrentSYCLQueue(),
          slm_sz,
          iinfo,
          gwinfo,
          oinfo,
          cfg);
    } else {
      sycl_kernel_submit<kptr_rest, 2>(
          cfg.global_size(),
          cfg.group_size(),
          getCurrentSYCLQueue(),
          slm_sz,
          iinfo,
          gwinfo,
          oinfo,
          cfg);
    }
  };

  if (cfg.problem_ <= cfg.problem_wg_range_) {
    launch(vinfo, rinfo);
    return;
  }

  Tensor carrier = at::empty(
      {cfg.batch_, cfg.problem_glb_range_ / cfg.problem_wg_range_, cfg.stride_},
      map_options<typename AccTypeInfo::scalar_t>());
  auto cinfo =
      at::xpu::detail::getTensorInfo<typename AccTypeInfo::scalar_t, int64_t>(
          carrier);
  launch(vinfo, cinfo);

  weight_norm_backward_reduce(cinfo, gwinfo, rinfo, 1, false);
  return;
}

template <
    class ScalarTypeInfo,
    class AccTypeInfo,
    typename scalar_t,
    typename accscalar_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<2>))
void segment_weight_norm_backward_kernel(
    ScalarTypeInfo vinfo,
    ScalarTypeInfo ginfo,
    ScalarTypeInfo gwinfo,
    AccTypeInfo ninfo,
    ScalarTypeInfo gvinfo,
    ScalarTypeInfo gginfo,
    AccTypeInfo rinfo,
    BatchKernelConfig cfg) {
  auto item = syclext::this_work_item::get_nd_item<2>();
  auto id = cfg.get_item_desc(item);

  int64_t si = id.glb_batch % cfg.stride_;
  int64_t bi = id.glb_batch / cfg.stride_;
  int64_t pi = id.chunk * id.chunk_size + id.chunk_off;

  int64_t gv_lid = si + pi * cfg.stride_ + bi * cfg.problem_ * cfg.stride_;
  int64_t gg_lid = id.glb_batch;

  int64_t v_off =
      at::xpu::detail::IndexToOffset<scalar_t, int64_t, -1>::get(gv_lid, vinfo);

  int64_t gw_off = at::xpu::detail::IndexToOffset<scalar_t, int64_t, -1>::get(
      gv_lid, gwinfo);

  int64_t gv_off = at::xpu::detail::IndexToOffset<scalar_t, int64_t, -1>::get(
      gv_lid, gvinfo);

  int64_t g_off =
      at::xpu::detail::IndexToOffset<scalar_t, int64_t, -1>::get(gg_lid, ginfo);

  int64_t n_off = at::xpu::detail::IndexToOffset<accscalar_t, int64_t, -1>::get(
      gg_lid, ninfo);

  int64_t r_off = at::xpu::detail::IndexToOffset<accscalar_t, int64_t, -1>::get(
      gg_lid, rinfo);

  int64_t gg_off = at::xpu::detail::IndexToOffset<scalar_t, int64_t, -1>::get(
      gg_lid, gginfo);

  if (id.glb_problem < cfg.problem_ && id.glb_batch < cfg.problem_batch_) {
    accscalar_t g = ginfo.data[g_off];
    accscalar_t gw = gwinfo.data[gw_off];
    accscalar_t v = vinfo.data[v_off];
    accscalar_t n = accscalar_t(1) / ninfo.data[n_off];
    accscalar_t r = rinfo.data[r_off];
    accscalar_t gg = r * n;
    accscalar_t n3 = n * n * n;
    accscalar_t gv = g * (n * gw - n3 * v * r);

    gvinfo.data[gv_off] = static_cast<scalar_t>(gv);
    if (id.chunk == 0 && id.chunk_off == 0)
      gginfo.data[gg_off] = static_cast<scalar_t>(gg);
  }
}

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
  constexpr auto kptr = segment_weight_norm_backward_kernel<
      ScalarTypeInfo,
      AccTypeInfo,
      scalar_t,
      accscalar_t>;
  BatchKernelConfig cfg = BatchKernelConfig::make_config<kptr>(
      batch,
      problem,
      stride,
      batch * stride,
      problem_along_x,
      {BatchKernelConfig::Policy::pSegment});

  sycl_kernel_submit<kptr, 2>(
      cfg.global_size(),
      cfg.group_size(),
      getCurrentSYCLQueue(),
      0,
      vinfo,
      ginfo,
      gwinfo,
      ninfo,
      gvinfo,
      gginfo,
      rinfo,
      cfg);

  return;
}

template <
    class ScalarTypeInfo,
    class AccTypeInfo,
    typename scalar_t,
    typename accscalar_t,
    typename vec_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<2>))
void weight_norm_backward_combine_kernel(
    ScalarTypeInfo vinfo,
    ScalarTypeInfo ginfo,
    ScalarTypeInfo gwinfo,
    AccTypeInfo ninfo,
    ScalarTypeInfo gvinfo,
    ScalarTypeInfo gginfo,
    BatchKernelConfig cfg,
    int batch_wg_range) {
  auto item = syclext::this_work_item::get_nd_item<2>();
  accscalar_t* shared_ = (accscalar_t*)syclexp::get_work_group_scratch_memory();

  auto id = cfg.get_item_desc(item);
  int64_t n_lid = id.glb_batch;
  int64_t g_off =
      at::xpu::detail::IndexToOffset<scalar_t, int64_t, -1>::get(n_lid, ginfo);
  int64_t gg_off =
      at::xpu::detail::IndexToOffset<scalar_t, int64_t, -1>::get(n_lid, gginfo);
  int64_t n_off = at::xpu::detail::IndexToOffset<accscalar_t, int64_t, -1>::get(
      n_lid, ninfo);
  int64_t si = id.glb_batch % cfg.stride_;
  int64_t bi = id.glb_batch / cfg.stride_;
  int64_t pi = id.chunk_off;
  bi = si + bi * cfg.problem_ * cfg.stride_;

  accscalar_t value = 0;
  if (id.glb_batch < cfg.problem_batch_) {
    for (int pi_ = pi; pi_ < cfg.problem_; pi_ += cfg.problem_wg_range_) {
      int64_t v_lid, v_off, gw_off;
      v_lid = bi + pi_ * cfg.stride_;

      v_off = at::xpu::detail::IndexToOffset<scalar_t, int64_t, -1>::get(
          v_lid, vinfo);

      gw_off = at::xpu::detail::IndexToOffset<scalar_t, int64_t, -1>::get(
          v_lid, gwinfo);

      accscalar_t v = (accscalar_t)vinfo.data[v_off];
      accscalar_t gw = (accscalar_t)gwinfo.data[gw_off];
      value += v * gw;
    }
  }

  char* shared_ptr = reinterpret_cast<char*>(shared_);
  if (cfg.problem_along_x_) {
    value = group_x_reduce(
        item, shared_ptr, vec_t(value), ReduceAdd<accscalar_t>())[0];
  } else {
    value = group_y_reduce(
        item, shared_ptr, vec_t(value), ReduceAdd<accscalar_t>())[0];
  }

  int n_slid = (int)id.glb_batch % batch_wg_range;
  if (id.glb_batch < cfg.problem_batch_ && id.chunk_off == 0) {
    shared_[n_slid] = value;
  }
  sycl::group_barrier(item.get_group());

  if (id.glb_batch < cfg.problem_batch_) {
    for (int pi_ = pi; pi_ < cfg.problem_; pi_ += cfg.problem_wg_range_) {
      int64_t v_lid, v_off, gw_off, gv_off;
      v_lid = bi + pi_ * cfg.stride_;

      v_off = at::xpu::detail::IndexToOffset<scalar_t, int64_t, -1>::get(
          v_lid, vinfo);

      gw_off = at::xpu::detail::IndexToOffset<scalar_t, int64_t, -1>::get(
          v_lid, gwinfo);

      gv_off = at::xpu::detail::IndexToOffset<scalar_t, int64_t, -1>::get(
          v_lid, gvinfo);

      accscalar_t g = ginfo.data[g_off];
      accscalar_t gw = gwinfo.data[gw_off];
      accscalar_t v = vinfo.data[v_off];
      accscalar_t n = accscalar_t(1) / ninfo.data[n_off];
      accscalar_t r = shared_[n_slid];
      accscalar_t gg = r * n;
      accscalar_t n3 = n * n * n;
      accscalar_t gv = g * (n * gw - n3 * v * r);

      gvinfo.data[gv_off] = static_cast<scalar_t>(gv);
      if (id.chunk_off == 0)
        gginfo.data[gg_off] = static_cast<scalar_t>(gg);
    }
  }
}

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
  constexpr auto kptr = weight_norm_backward_combine_kernel<
      ScalarTypeInfo,
      AccTypeInfo,
      scalar_t,
      accscalar_t,
      vec_t>;
  BatchKernelConfig cfg = BatchKernelConfig::make_config<kptr>(
      batch,
      problem,
      stride,
      batch * stride,
      problem_along_x,
      {BatchKernelConfig::Policy::pLoop});
  int wg_size = cfg.group_size().size();
  int batch_wg_range = wg_size / cfg.problem_wg_range_;
  int slm_sz = wg_size * sizeof(accscalar_t);
  sycl_kernel_submit<kptr, 2>(
      cfg.global_size(),
      cfg.group_size(),
      getCurrentSYCLQueue(),
      slm_sz,
      vinfo,
      ginfo,
      gwinfo,
      ninfo,
      gvinfo,
      gginfo,
      cfg,
      batch_wg_range);
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

  // Empty saved_v: grad_v is empty, grad_g = 0 (matches CPU/CUDA).
  if (saved_v.numel() == 0) {
    grad_g.zero_();
    return {grad_v, grad_g};
  }

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
