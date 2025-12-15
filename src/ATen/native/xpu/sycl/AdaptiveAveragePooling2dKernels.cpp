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
#include <ATen/OpMathType.h>
#include <ATen/ceil_div.h>
#include <ATen/native/AdaptivePooling.h>
#include <ATen/native/Pool.h>
#include <ATen/native/xpu/sycl/LaunchUtils.h>
#include <ATen/native/xpu/sycl/MemoryAccess.h>
#include <comm/MemoryFormat.h>
#include <comm/xpu_aten.h>
#include <vector>

#include <ATen/native/xpu/sycl/AdaptiveAveragePooling2dKernels.h>

#define START_IND(a, b, c) ((int64_t)((a / b) * c + ((a % b) * c) / b))
#define END_IND(a, b, c) (1 + ((int64_t)(a + 1) * c - 1) / b)

#define START_IND_INT(a, b, c) ((a * c) / b)
#define END_IND_INT(a, b, c) (((a + 1) * c + b - 1) / b)

#define XPU_MAX_THREADS 1024
#define GROUP_STRIDE 2 // increasing group_stride to lower # of groups launched

namespace at::native::xpu {

using namespace at::xpu;

template <typename scalar_t, typename opmath_t>
struct AdaptiveAvgPool2dBwdKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    int64_t gi = item.get_global_linear_id();

    for (int64_t i = gi; i < numel_; i += global_range_) {
      int64_t _iw, _ih, _ic, _ib;

      _iw = i % iw_;
      _ih = i / iw_ % ih_;
      _ic = i / iw_ / ih_ % ic_;
      _ib = i / iw_ / ih_ / ic_;

      int64_t _oh0 = native::start_index(_ih, ih_, oh_);
      int64_t _oh1 = native::end_index(_ih, ih_, oh_);
      int64_t _ow0 = native::start_index(_iw, iw_, ow_);
      int64_t _ow1 = native::end_index(_iw, iw_, ow_);
      int64_t _ob = _ib;
      int64_t _oc = _ic;

      opmath_t gx = 0;
      opmath_t _ikh, _ikw;
      for (int _oh = _oh0; _oh < _oh1; _oh++) {
        _ikh = opmath_t(1.0) /
            (opmath_t)(native::end_index(_oh, oh_, ih_) - native::start_index(_oh, oh_, ih_));
        for (int _ow = _ow0; _ow < _ow1; _ow++) {
          _ikw = opmath_t(1.0) /
              (opmath_t)(native::end_index(_ow, ow_, iw_) - native::start_index(_ow, ow_, iw_));
          gx += gyacc_[_ob][_oc][_oh][_ow] * _ikh * _ikw;
        }
      }

      const auto store = [](PackedTensorAccessor64<scalar_t, 4> gxacc,
                            int64_t _ib,
                            int64_t _ic,
                            int64_t _ih,
                            int64_t _iw,
                            scalar_t res) { gxacc[_ib][_ic][_ih][_iw] = res; };
      store(gxacc_, _ib, _ic, _ih, _iw, (scalar_t)gx);
    }
  }

  AdaptiveAvgPool2dBwdKernelFunctor(
      PackedTensorAccessor64<const scalar_t, 4> gyacc,
      PackedTensorAccessor64<scalar_t, 4> gxacc)
      : gyacc_(gyacc), gxacc_(gxacc) {
    ib_ = gxacc_.size(0);
    ic_ = gxacc_.size(1);
    ih_ = gxacc_.size(2);
    iw_ = gxacc_.size(3);
    oh_ = gyacc_.size(2);
    ow_ = gyacc_.size(3);

    numel_ = ib_ * ic_ * ih_ * iw_;
    int total_item = std::min(numel_, syclMaxWorkItemsPerTile());
    local_range_ = syclMaxWorkItemsPerSubSlice();
    global_range_ = total_item < local_range_
        ? local_range_
        : (total_item / local_range_) * local_range_;
  }

  sycl::range<1> glb_range() {
    return sycl::range<1>(global_range_);
  }

  sycl::range<1> loc_range() {
    return sycl::range<1>(local_range_);
  }

 private:
  int ib_;
  int ic_;
  int ih_;
  int iw_;
  int oh_;
  int ow_;
  int64_t numel_;
  int global_range_;
  int local_range_;
  PackedTensorAccessor64<const scalar_t, 4> gyacc_;
  PackedTensorAccessor64<scalar_t, 4> gxacc_;
};

template <typename scalar_t, typename opmath_t>
struct AdaptiveAvgPool2dBwdSLMKernelFunctor
    : public __SYCL_KER_CONFIG_CONVENTION__ {
  void operator()(sycl::nd_item<1> item) const {
    int64_t gi = item.get_global_linear_id();
    int64_t li = item.get_local_id(0);

    // for-loop order: oh*ow->ih->iw
    // reuse oh*ow(oh0, oh1, ow0, ow1), ih(ikh), iw(ikw) in inner loop.
    for (int _ih = li; _ih < ih_; _ih += local_range_) {
      _oh0_cached_[_ih] = (int)native::start_index(_ih, ih_, oh_);
      _oh1_cached_[_ih] = (int)native::end_index(_ih, ih_, oh_);
    }
    for (int _iw = li; _iw < iw_; _iw += local_range_) {
      _ow0_cached_[_iw] = (int)native::start_index(_iw, iw_, ow_);
      _ow1_cached_[_iw] = (int)native::end_index(_iw, iw_, ow_);
    }
    for (int _oh = li; _oh < oh_; _oh += local_range_) {
      _ikh_cached_[_oh] = opmath_t(1.0) /
          (opmath_t)(native::end_index(_oh, oh_, ih_) -
                     native::start_index(_oh, oh_, ih_));
    }
    for (int _ow = li; _ow < ow_; _ow += local_range_) {
      _ikw_cached_[_ow] = opmath_t(1.0) /
          (opmath_t)(native::end_index(_ow, ow_, iw_) -
                     native::start_index(_ow, ow_, iw_));
    }

    item.barrier(sycl_local_fence);

    for (int64_t i = gi; i < numel_; i += global_range_) {
      int64_t _iw, _ih, _ic, _ib;

      _iw = i % iw_;
      _ih = i / iw_ % ih_;
      _ic = i / iw_ / ih_ % ic_;
      _ib = i / iw_ / ih_ / ic_;

      int64_t _oh0, _oh1, _ow0, _ow1;
      _oh0 = _oh0_cached_[_ih];
      _oh1 = _oh1_cached_[_ih];
      _ow0 = _ow0_cached_[_iw];
      _ow1 = _ow1_cached_[_iw];
      int64_t _ob = _ib;
      int64_t _oc = _ic;

      opmath_t gx = 0;
      opmath_t _ikh, _ikw;
      for (int _oh = _oh0; _oh < _oh1; _oh++) {
        _ikh = _ikh_cached_[_oh];
        for (int _ow = _ow0; _ow < _ow1; _ow++) {
          _ikw = _ikw_cached_[_ow];
          gx += gyacc_[_ob][_oc][_oh][_ow] * _ikh * _ikw;
        }
      }

      const auto store = [](PackedTensorAccessor64<scalar_t, 4> gxacc,
                            int64_t _ib,
                            int64_t _ic,
                            int64_t _ih,
                            int64_t _iw,
                            scalar_t res) { gxacc[_ib][_ic][_ih][_iw] = res; };
      store(gxacc_, _ib, _ic, _ih, _iw, (scalar_t)gx);
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    _oh0_cached_ = sycl_local_acc_t<int>(ih_, cgh);
    _oh1_cached_ = sycl_local_acc_t<int>(ih_, cgh);
    _ow0_cached_ = sycl_local_acc_t<int>(iw_, cgh);
    _ow1_cached_ = sycl_local_acc_t<int>(iw_, cgh);
    _ikh_cached_ = sycl_local_acc_t<opmath_t>(oh_, cgh);
    _ikw_cached_ = sycl_local_acc_t<opmath_t>(ow_, cgh);
  }

  AdaptiveAvgPool2dBwdSLMKernelFunctor(
      PackedTensorAccessor64<const scalar_t, 4> gyacc,
      PackedTensorAccessor64<scalar_t, 4> gxacc)
      : gyacc_(gyacc), gxacc_(gxacc) {
    ib_ = gxacc_.size(0);
    ic_ = gxacc_.size(1);
    ih_ = gxacc_.size(2);
    iw_ = gxacc_.size(3);
    oh_ = gyacc_.size(2);
    ow_ = gyacc_.size(3);

    numel_ = ib_ * ic_ * ih_ * iw_;
    int total_item = std::min(numel_, syclMaxWorkItemsPerTile());

    local_range_ = syclMaxWorkGroupSize(*this);
    global_range_ = total_item < local_range_
        ? local_range_
        : (total_item / local_range_) * local_range_;
  }

  sycl::range<1> glb_range() {
    return sycl::range<1>(global_range_);
  }

  sycl::range<1> loc_range() {
    return sycl::range<1>(local_range_);
  }

 private:
  int ib_;
  int ic_;
  int ih_;
  int iw_;
  int oh_;
  int ow_;
  int64_t numel_;
  int local_range_;
  int global_range_;
  PackedTensorAccessor64<const scalar_t, 4> gyacc_;
  PackedTensorAccessor64<scalar_t, 4> gxacc_;
  sycl_local_acc_t<int> _oh0_cached_;
  sycl_local_acc_t<int> _oh1_cached_;
  sycl_local_acc_t<int> _ow0_cached_;
  sycl_local_acc_t<int> _ow1_cached_;
  sycl_local_acc_t<opmath_t> _ikh_cached_;
  sycl_local_acc_t<opmath_t> _ikw_cached_;
};

template <typename index_t, typename scalar_t>
struct AdaptiveAvgPool2dBwdSLMChannelsLastKernelFunctor
    : public __SYCL_KER_CONFIG_CONVENTION__ {
  void operator()(sycl::nd_item<3> item) const {
    scalar_t* out_cached =
        (scalar_t*)out_cached_
            .template get_multi_ptr<sycl::access::decorated::no>()
            .get();
    // flattening cta for pre-computation & smem initialization;
    int thread_id = item.get_local_id(2) +
        item.get_local_range(2) *
            (item.get_local_id(1) +
             item.get_local_range(1) * item.get_local_id(0));
    // Precompute output start/end index per input index on width dimension;
    // Not doing this for height dimension, as that's our out-most loop.
    for (index_t i = thread_id; i < isizeW_; i += group_size_) {
      ostartW_cached_[i] = START_IND_INT(i, isizeW_, osizeW_);
      oendW_cached_[i] = END_IND_INT(i, isizeW_, osizeW_);
    }

    // Precompute pooling height/weight factor for each output element;
    // This is used to weight output gradient when accumulate them on input
    // gradient.
    for (index_t i = thread_id; i < osizeH_; i += group_size_) {
      r_kH_cached_[i] = scalar_t(1.0) /
          (END_IND_INT(i, osizeH_, isizeH_) -
           START_IND_INT(i, osizeH_, isizeH_));
    }
    for (index_t i = thread_id; i < osizeW_; i += group_size_) {
      r_kW_cached_[i] = scalar_t(1.0) /
          (END_IND_INT(i, osizeW_, isizeW_) -
           START_IND_INT(i, osizeW_, isizeW_));
    }

    // each cta handles a portion of a single slice on batch dimension;
    // we use get_group_range(2) to handle striding on C as well.
    int batch_id = item.get_group(2) % sizeB_;
    int channel_id = item.get_group(2) / sizeB_;
    int channel_offset =
        item.get_local_id(2) + channel_id * item.get_local_range(2);

    // use shared memory to store temporary output value. This is simply to
    // reduce register usage.
    for (index_t i = thread_id; i < kernel_size_C_ * item.get_local_range(2) *
             item.get_local_range(1) * item.get_local_range(0);
         i += group_size_) {
      out_cached[i] = scalar_t(0.0);
    }

    item.barrier(sycl_local_fence);

    auto gradInput = gradInput_ + batch_id * isizeH_ * isizeW_ * sizeC_;
    auto gradOutput = gradOutput_ + batch_id * ostrideB_;

    // split out_cached and exclusively it assigned to each thread;
    out_cached = &out_cached
                     [(item.get_local_id(0) * item.get_local_range(1) +
                       item.get_local_id(1)) *
                      item.get_local_range(2) * kernel_size_C_];

    // iterate on input H & W.
    // each cta handles a consecutive H & W section (TILE); Do NOT stride CTA on
    // tile so there's a better chance to hit L1 cache.
    index_t iH =
        (isizeH_ + item.get_group_range(0) - 1) / item.get_group_range(0);
    index_t iW =
        (isizeW_ + item.get_group_range(1) - 1) / item.get_group_range(1);
    index_t istartH = item.get_local_id(0) + item.get_group(0) * iH;
    index_t iendH = std::min(istartH + iH, isizeH_);
    index_t istartW = item.get_local_id(1) + item.get_group(1) * iW;
    index_t iendW = std::min(istartW + iW, isizeW_);

    // Stride for threads, each subgroup can reuse L1 as they go. So
    // theoretically better chance to survive cache eviction.
    for (index_t ih = istartH; ih < iendH; ih += item.get_local_range(0)) {
      index_t ostartH = START_IND_INT(ih, isizeH_, osizeH_);
      index_t oendH = END_IND_INT(ih, isizeH_, osizeH_);
      for (index_t iw = istartW; iw < iendW; iw += item.get_local_range(1)) {
        // loop on output: hierarchy h->w->c, so we could reuse weight factor f
        // because it remains the same for given oh & ow
        for (index_t oh = ostartH; oh < oendH; ++oh) {
          for (index_t ow = ostartW_cached_[iw]; ow < oendW_cached_[iw]; ++ow) {
            scalar_t f = r_kW_cached_[ow] * r_kH_cached_[oh];
            const scalar_t* ptr_gradOutput =
                gradOutput + oh * ostrideH_ + ow * ostrideW_;
            int cached_index = item.get_local_id(2);
            for (index_t c = channel_offset; c < sizeC_;
                 c += item.get_local_range(2) * kernel_stride_C_) {
              out_cached[cached_index] += ptr_gradOutput[c * ostrideC_] * f;
              cached_index += item.get_local_range(2);
            }
          }
        }
        scalar_t* ptr_gradInput = gradInput + (ih * isizeW_ + iw) * sizeC_;
        int cached_index = item.get_local_id(2);
        // write accumulated gradInput to global memory;
        for (index_t c = channel_offset; c < sizeC_;
             c += item.get_local_range(2) * kernel_stride_C_) {
          ptr_gradInput[c] = out_cached[cached_index];
          out_cached[cached_index] = scalar_t(0.0);
          cached_index += item.get_local_range(2);
        }
      }
    }
  }
  void sycl_ker_config_convention(sycl::handler& cgh) {
    ostartW_cached_ = sycl_local_acc_t<int>(isizeW_, cgh);
    oendW_cached_ = sycl_local_acc_t<int>(isizeW_, cgh);
    r_kW_cached_ = sycl_local_acc_t<scalar_t>(osizeW_, cgh);
    r_kH_cached_ = sycl_local_acc_t<scalar_t>(osizeH_, cgh);
    out_cached_ = sycl_local_acc_t<scalar_t>(kernel_size_C_ * group_size_, cgh);
  }

  AdaptiveAvgPool2dBwdSLMChannelsLastKernelFunctor(
      scalar_t* gradInput,
      const scalar_t* gradOutput,
      int sizeB,
      int sizeC,
      int isizeH,
      int isizeW,
      int osizeH,
      int osizeW,
      int kernel_stride_C,
      int kernel_size_C,
      index_t ostrideB,
      index_t ostrideC,
      index_t ostrideH,
      index_t ostrideW,
      size_t group_size)
      : gradInput_(gradInput),
        gradOutput_(gradOutput),
        sizeB_(sizeB),
        sizeC_(sizeC),
        isizeH_(isizeH),
        isizeW_(isizeW),
        osizeH_(osizeH),
        osizeW_(osizeW),
        kernel_stride_C_(kernel_stride_C),
        kernel_size_C_(kernel_size_C),
        ostrideB_(ostrideB),
        ostrideC_(ostrideC),
        ostrideH_(ostrideH),
        ostrideW_(ostrideW),
        group_size_(group_size) {}

 private:
  scalar_t* gradInput_;
  const scalar_t* gradOutput_;
  int sizeB_;
  int sizeC_;
  int isizeH_;
  int isizeW_;
  int osizeH_;
  int osizeW_;
  int kernel_stride_C_;
  int kernel_size_C_;
  index_t ostrideB_;
  index_t ostrideC_;
  index_t ostrideH_;
  index_t ostrideW_;
  size_t shmem_size_;
  size_t group_size_;
  sycl_local_acc_t<int> ostartW_cached_;
  sycl_local_acc_t<int> oendW_cached_;
  sycl_local_acc_t<scalar_t> r_kW_cached_;
  sycl_local_acc_t<scalar_t> r_kH_cached_;
  sycl_local_acc_t<scalar_t> out_cached_;
};

void adaptive_avg_pool2d_backward_kernel(
    Tensor& grad_input,
    const Tensor& grad_output_,
    const Tensor& input_) {
  Tensor input, grad_output;
  if (input_.ndimension() == 3) {
    input = input_.contiguous();
    grad_output = grad_output_.contiguous();
    grad_input = at::empty_like(input);
  } else {
    auto smf = input_.suggest_memory_format();
    input = input_.contiguous(smf);
    grad_output = grad_output_.contiguous(smf);
    grad_input = at::empty_like(input_, smf);
  }

  int osizeH = grad_output.size(-2);
  int osizeW = grad_output.size(-1);

  int sizeC = input.size(-3);
  int isizeH = input.size(-2);
  int isizeW = input.size(-1);

  bool is_3d = grad_output.ndimension() == 3;
  if (is_3d) {
    grad_output.resize_({1, sizeC, osizeH, osizeW});
    grad_input.resize_({1, sizeC, isizeH, isizeW});
  }

  int sizeB = input.size(0);

  int64_t ostrideB = grad_output.stride(0);
  int64_t ostrideC = grad_output.stride(1);
  int64_t ostrideH = grad_output.stride(2);
  int64_t ostrideW = grad_output.stride(3);

  if (is_smf_channels_last(grad_output)) {
    // preserve channels_last stride on input tensor;
    if (!grad_input.is_contiguous(at::MemoryFormat::ChannelsLast)) {
      grad_input.as_strided_(
          {sizeB, sizeC, isizeH, isizeW},
          {sizeC * isizeH * isizeW, 1, isizeW * sizeC, sizeC});
    }

    int max_threads =
        std::min<int>(syclMaxWorkItemsPerSubSlice(), XPU_MAX_THREADS);
    size_t sharedMemPerGroup = syclLocalMemSize();

    bool done = false;
    do {
      int group_x = std::max<int>(
          std::min<int>(lastPow2(sizeC), syclMaxSubGroupSize()), 1);
      int group_y = std::max<int>(
          std::min<int>(lastPow2(isizeW), max_threads / group_x), 1);
      int group_z = std::max<int>(
          std::min<int>(lastPow2(isizeH), max_threads / group_x / group_y), 1);
      group_x = std::max<int>(
          std::min<int>(lastPow2(sizeC), max_threads / group_y / group_z), 1);
      sycl::range<3> local_range{
          (size_t)group_z, (size_t)group_y, (size_t)group_x};

      int kernel_stride_C = ceil_div(sizeC, group_x * 4);
      int kernel_size_C = ceil_div(sizeC, group_x * kernel_stride_C);

      int range_x = sizeB * kernel_stride_C;

      int range_y = ceil_div(isizeW, group_y * GROUP_STRIDE);
      int range_z = ceil_div(isizeH, group_z * GROUP_STRIDE);

      sycl::range<3> global_range{
          (size_t)range_z * group_z,
          (size_t)range_y * group_y,
          (size_t)range_x * group_x};

      AT_ASSERT(input.numel() < std::numeric_limits<int32_t>::max());

      AT_DISPATCH_FLOATING_TYPES_AND2(
          kHalf,
          kBFloat16,
          input.scalar_type(),
          "adaptive_avg_pool2d_backward_nhwc_xpu",
          [&] {
            size_t shmem_size = (kernel_size_C * group_x * group_y * group_z +
                                 osizeH + osizeW) *
                    sizeof(scalar_t) +
                2 * isizeW * sizeof(int32_t);
            if (shmem_size <= sharedMemPerGroup) {
              AdaptiveAvgPool2dBwdSLMChannelsLastKernelFunctor<
                  int32_t,
                  scalar_t>
                  kfn(grad_input.mutable_data_ptr<scalar_t>(),
                      grad_output.const_data_ptr<scalar_t>(),
                      sizeB,
                      sizeC,
                      isizeH,
                      isizeW,
                      osizeH,
                      osizeW,
                      kernel_stride_C,
                      kernel_size_C,
                      ostrideB,
                      ostrideC,
                      ostrideH,
                      ostrideW,
                      group_x * group_y * group_z);
              sycl_kernel_submit(
                  global_range, local_range, getCurrentSYCLQueue(), kfn);
              done = true;
            } else {
              TORCH_WARN_ONCE(
                  "Requested shmem_size exceeds sharedMemPerBlock "
                  "limit! Reducing max_threads...");
              max_threads /= 2;
            }
          });
    } while (!done && max_threads);
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::BFloat16,
        at::ScalarType::Half,
        grad_output.scalar_type(),
        "adaptive_avg_pool2d_backward_xpu",
        [&]() {
          using opmath_t = at::opmath_type<scalar_t>;
          auto gyacc = grad_output.packed_accessor64<const scalar_t, 4>();
          auto gxacc = grad_input.packed_accessor64<scalar_t, 4>();

          int64_t ohw01_shared_size = ((isizeH + isizeW) * 2) * sizeof(int);
          int64_t ikhw_shared_size = (osizeH + osizeW) * sizeof(opmath_t);
          bool using_shared =
              syclLocalMemSize() >= ohw01_shared_size + ikhw_shared_size;

          auto& q = getCurrentSYCLQueue();
          if (using_shared) {
            AdaptiveAvgPool2dBwdSLMKernelFunctor<scalar_t, opmath_t> kfn(
                gyacc, gxacc);
            sycl_kernel_submit(kfn.glb_range(), kfn.loc_range(), q, kfn);
          } else {
            AdaptiveAvgPool2dBwdKernelFunctor<scalar_t, opmath_t> kfn(
                gyacc, gxacc);
            sycl_kernel_submit(kfn.glb_range(), kfn.loc_range(), q, kfn);
          }
        });
  }
  if (is_3d) {
    grad_output.resize_({sizeC, osizeH, osizeW});
    grad_input.resize_({sizeC, isizeH, isizeW});
  }
}

template <typename scalar_t, typename opmath_t, bool is_channels_last>
struct AdaptiveAvgPool2dKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    int64_t gi = item.get_global_linear_id();
    for (int64_t i = gi; i < numel_; i += global_range_) {
      int64_t _ow, _oh, _oc, _ob;
      if constexpr (is_channels_last) {
        _oc = i % oc_;
        _ow = i / oc_ % ow_;
        _oh = i / oc_ / ow_ % oh_;
        _ob = i / oc_ / ow_ / oh_;
      } else {
        _ow = i % ow_;
        _oh = i / ow_ % oh_;
        _oc = i / ow_ / oh_ % oc_;
        _ob = i / ow_ / oh_ / oc_;
      }

      int64_t _ih0 = native::start_index(_oh, oh_, ih_);
      int64_t _ih1 = native::end_index(_oh, oh_, ih_);
      int64_t _iw0 = native::start_index(_ow, ow_, iw_);
      int64_t _iw1 = native::end_index(_ow, ow_, iw_);
      int64_t kh = _ih1 - _ih0;
      int64_t kw = _iw1 - _iw0;
      int64_t _ib = _ob;
      int64_t _ic = _oc;

      opmath_t sum = static_cast<opmath_t>(0);
      for (int _ih = _ih0; _ih < _ih1; _ih++) {
        for (int _iw = _iw0; _iw < _iw1; _iw++) {
          sum += opmath_t(input_[_ib][_ic][_ih][_iw]);
        }
      }
      opmath_t avg = sum / kh / kw;

      const auto store = [](PackedTensorAccessor64<scalar_t, 4> oacc,
                            int64_t _ob,
                            int64_t _oc,
                            int64_t _oh,
                            int64_t _ow,
                            scalar_t res) { oacc[_ob][_oc][_oh][_ow] = res; };
      store(output_, _ob, _oc, _oh, _ow, avg);
    }
  }
  AdaptiveAvgPool2dKernelFunctor(
      int ih,
      int iw,
      int ob,
      int oc,
      int oh,
      int ow,
      int64_t numel,
      int global_range,
      PackedTensorAccessor64<const scalar_t, 4> input,
      PackedTensorAccessor64<scalar_t, 4> output)
      : ih_(ih),
        iw_(iw),
        ob_(ob),
        oc_(oc),
        oh_(oh),
        ow_(ow),
        numel_(numel),
        global_range_(global_range),
        input_(input),
        output_(output) {}

 private:
  int ih_;
  int iw_;
  int ob_;
  int oc_;
  int oh_;
  int ow_;
  int64_t numel_;
  int global_range_;
  PackedTensorAccessor64<const scalar_t, 4> input_;
  PackedTensorAccessor64<scalar_t, 4> output_;
};

template <typename scalar_t, typename opmath_t, typename vec_t, int vec_size>
struct AdaptiveAvgPool2dKernelFunctor_cl {
  void operator()(sycl::nd_item<1> item) const {
    int64_t index = item.get_global_linear_id();
    if (index < numel_) {
      int _ow, _oh, _oc, _ob;
      int oc_vec_ = oc_ / vec_size;

      _oc = index % oc_vec_;
      _ow = index / oc_vec_ % ow_;
      _oh = index / oc_vec_ / ow_ % oh_;
      _ob = index / oc_vec_ / ow_ / oh_;

      int64_t _ih0 = native::start_index(_oh, oh_, ih_);
      int64_t _ih1 = native::end_index(_oh, oh_, ih_);
      int64_t _iw0 = native::start_index(_ow, ow_, iw_);
      int64_t _iw1 = native::end_index(_ow, ow_, iw_);
      int64_t kh = _ih1 - _ih0;
      int64_t kw = _iw1 - _iw0;
      int64_t _ib = _ob;
      int64_t _ic = _oc;

      opmath_t sum[vec_size] = {static_cast<opmath_t>(0)};
      for (int _ih = _ih0; _ih < _ih1; _ih++) {
        for (int _iw = _iw0; _iw < _iw1; _iw++) {
          auto read = input_
              [_ic + _iw * oc_vec_ + _ih * oc_vec_ * iw_ +
               _ib * ih_ * iw_ * oc_vec_];
#pragma unroll
          for (int v = 0; v < vec_size; v++) {
            sum[v] += opmath_t(read[v]);
          }
        }
      }
#pragma unroll
      for (int v = 0; v < vec_size; v++) {
        sum[v] /= kh * kw;
      }
      vec_t output_value;
#pragma unroll
      for (int v = 0; v < vec_size; v++) {
        output_value[v] = static_cast<scalar_t>(sum[v]);
      }
      output_[index] = output_value;
    }
  }
  AdaptiveAvgPool2dKernelFunctor_cl(
      vec_t* output,
      const vec_t* input,
      int ih,
      int iw,
      int ob,
      int oc,
      int oh,
      int ow,
      int64_t numel)
      : output_(output),
        input_(input),
        ih_(ih),
        iw_(iw),
        ob_(ob),
        oc_(oc),
        oh_(oh),
        ow_(ow),
        numel_(numel) {}

 private:
  vec_t* output_;
  const vec_t* input_;
  int ih_;
  int iw_;
  int ob_;
  int oc_;
  int oh_;
  int ow_;
  int64_t numel_;
};

#define LAUNCH_AVGPOOL_CHANNEL_LAST_VEC(                                  \
    scalar_t,                                                             \
    opmath_t,                                                             \
    vec_size,                                                             \
    num_wg,                                                               \
    wg_size,                                                              \
    queue,                                                                \
    output,                                                               \
    input,                                                                \
    ih,                                                                   \
    iw,                                                                   \
    ob,                                                                   \
    oc,                                                                   \
    oh,                                                                   \
    ow,                                                                   \
    numel)                                                                \
  {                                                                       \
    using vec_t = memory::aligned_vector<scalar_t, vec_size>;             \
    vec_t* output_vec =                                                   \
        reinterpret_cast<vec_t*>(output.mutable_data_ptr<scalar_t>());    \
    const vec_t* input_vec =                                              \
        reinterpret_cast<const vec_t*>(input.const_data_ptr<scalar_t>()); \
    auto kfn = AdaptiveAvgPool2dKernelFunctor_cl<                         \
        scalar_t,                                                         \
        opmath_t,                                                         \
        vec_t,                                                            \
        vec_size>(output_vec, input_vec, ih, iw, ob, oc, oh, ow, numel);  \
    sycl_kernel_submit(num_wg* wg_size, wg_size, queue, kfn);             \
  }

template <typename scalar_t, typename opmath_t>
void launch_adaptive_avg_pool2d_kernel_cl(const Tensor& input, Tensor& output) {
  int ih = input.size(2);
  int iw = input.size(3);
  int ob = output.size(0);
  int oc = output.size(1);
  int oh = output.size(2);
  int ow = output.size(3);

  int64_t numel = ob * oc * oh * ow;
  int vec_size = 1;
  for (vec_size = std::min(
           8,
           memory::can_vectorize_up_to<scalar_t>(
               (char*)output.mutable_data_ptr<scalar_t>()));
       vec_size > 1;
       vec_size /= 2) {
    if (oc % vec_size != 0)
      continue;
    if (2 * numel / vec_size > syclMaxWorkItemsPerTile()) {
      numel /= vec_size;
      break;
    }
  }

  auto wg_size = syclDeviceMaxWorkGroupSize();
  int64_t num_wg = (numel + wg_size - 1) / wg_size;
  switch (vec_size) {
    case 8:
      LAUNCH_AVGPOOL_CHANNEL_LAST_VEC(
          scalar_t,
          opmath_t,
          8,
          num_wg,
          wg_size,
          at::xpu::getCurrentSYCLQueue(),
          output,
          input,
          ih,
          iw,
          ob,
          oc,
          oh,
          ow,
          numel);
      return;
    case 4:
      LAUNCH_AVGPOOL_CHANNEL_LAST_VEC(
          scalar_t,
          opmath_t,
          4,
          num_wg,
          wg_size,
          at::xpu::getCurrentSYCLQueue(),
          output,
          input,
          ih,
          iw,
          ob,
          oc,
          oh,
          ow,
          numel);
      return;
    case 2:
      LAUNCH_AVGPOOL_CHANNEL_LAST_VEC(
          scalar_t,
          opmath_t,
          2,
          num_wg,
          wg_size,
          at::xpu::getCurrentSYCLQueue(),
          output,
          input,
          ih,
          iw,
          ob,
          oc,
          oh,
          ow,
          numel);
      return;
    case 1:
      LAUNCH_AVGPOOL_CHANNEL_LAST_VEC(
          scalar_t,
          opmath_t,
          1,
          num_wg,
          wg_size,
          at::xpu::getCurrentSYCLQueue(),
          output,
          input,
          ih,
          iw,
          ob,
          oc,
          oh,
          ow,
          numel);
      return;
    default:
      TORCH_INTERNAL_ASSERT(false, "Unexpected vectorization size");
  }
}
#undef LAUNCH_AVGPOOL_CHANNEL_LAST_VEC

template <typename scalar_t, typename opmath_t, bool is_channels_last>
void launch_adaptive_avg_pool2d_kernel(
    PackedTensorAccessor64<const scalar_t, 4> input,
    PackedTensorAccessor64<scalar_t, 4> output) {
  int ih = input.size(2);
  int iw = input.size(3);
  int ob = output.size(0);
  int oc = output.size(1);
  int oh = output.size(2);
  int ow = output.size(3);

  int64_t numel = ob * oc * oh * ow;
  int total_item = std::min(numel, syclMaxWorkItemsPerTile());
  int local_range = syclMaxWorkItemsPerSubSlice();
  int global_range = total_item < local_range
      ? local_range
      : ((total_item + local_range - 1) / local_range) * local_range;
  auto caller =
      AdaptiveAvgPool2dKernelFunctor<scalar_t, opmath_t, is_channels_last>(
          ih, iw, ob, oc, oh, ow, numel, global_range, input, output);
  sycl_kernel_submit(
      sycl::range<1>(global_range),
      sycl::range<1>(local_range),
      getCurrentSYCLQueue(),
      caller);
}

void adaptive_avg_pool2d_kernel(
    Tensor& output,
    const Tensor& input,
    IntArrayRef output_size) {
  auto outputWidth = output_size[1];
  auto outputHeight = output_size[0];

  if (!input.is_quantized() && outputWidth == 1 && outputHeight == 1) {
    // in this case, adaptive pooling is just computing mean over hw
    // dimensions, which can be done more efficiently

    output = input.mean({-1, -2}, /* keepdim = */ true);
    if (input.suggest_memory_format() == at::MemoryFormat::ChannelsLast) {
      // assert ndim == 4, since ndim = 3 doesn't give channels_last
      const int n = input.size(0);
      const int c = input.size(1);
      output.as_strided_({n, c, 1, 1}, {c, 1, c, c});
    }
    return;
  }

  /* sizes */
  const int64_t nbatch = input.ndimension() == 4 ? input.size(-4) : 1;
  const auto nInputPlane = input.size(-3);
  const auto inputHeight = input.size(-2);
  const auto inputWidth = input.size(-1);
  Tensor input_;
  if (input.ndimension() == 3) {
    input_ = input.contiguous();
    output.resize_({nInputPlane, outputHeight, outputWidth});
  } else {
    auto smf = input.suggest_memory_format();
    input_ = input.contiguous(smf);
    output.resize_({nbatch, nInputPlane, outputHeight, outputWidth}, smf);
  }
  if (output.numel() == 0) {
    return;
  }
  int dH = std::floor((float)2 * inputHeight / outputHeight) -
      (inputHeight / outputHeight);
  int dW = std::floor((float)2 * inputWidth / outputWidth) -
      (inputWidth / outputWidth);
  std::vector<int64_t> stride_vec = {dH, dW};

  int kH = std::ceil((float)2 * inputHeight / outputHeight) -
      (inputHeight / outputHeight);
  int kW = std::ceil((float)2 * inputWidth / outputWidth) -
      (inputWidth / outputWidth);
  std::vector<int64_t> kernel_size_vec = {kH, kW};

  int padH = (dH * (outputHeight - 1) + kH - inputHeight) / 2;
  int padW = (dW * (outputWidth - 1) + kW - inputWidth) / 2;
  std::vector<int64_t> padding_vec = {padH, padW};

  bool is_3d = input_.ndimension() == 3;
  if (is_3d) {
    input_.resize_({1, nInputPlane, inputHeight, inputWidth});
    output.resize_({1, nInputPlane, outputHeight, outputWidth});
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      input_.scalar_type(),
      "adaptive_avg_pool2d_xpu",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        auto iacc = input_.packed_accessor64<const scalar_t, 4>();
        auto oacc = output.packed_accessor64<scalar_t, 4>();
        if (is_smf_channels_last(output)) {
          if (input_.is_contiguous(at::MemoryFormat::ChannelsLast)) {
            launch_adaptive_avg_pool2d_kernel_cl<scalar_t, opmath_t>(
                input_, output);
          } else {
            launch_adaptive_avg_pool2d_kernel<scalar_t, opmath_t, true>(
                iacc, oacc);
          }
        } else {
          launch_adaptive_avg_pool2d_kernel<scalar_t, opmath_t, false>(
              iacc, oacc);
        }
      });

  if (is_3d) {
    input_.resize_({nInputPlane, inputHeight, inputWidth});
    output.resize_({nInputPlane, outputHeight, outputWidth});
  }
}

} // namespace at::native::xpu
