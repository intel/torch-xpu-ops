#include <ATen/AccumulateType.h>
#include <ATen/OpMathType.h>
#include <ATen/native/AdaptivePooling.h>
#include <ATen/native/Pool.h>
#include <comm/MemoryFormat.h>
#include <comm/xpu_aten.h>
#include <vector>

#include <ATen/native/xpu/sycl/AdaptiveAveragePooling2dKernels.h>

namespace at::native::xpu {

using namespace at::xpu;

template <typename scalar_t, typename opmath_t, bool is_channels_last>
struct AdaptiveAvgPool2dBwdKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    int64_t gi = item.get_global_linear_id();

    for (int64_t i = gi; i < numel_; i += global_range_) {
      int64_t _iw, _ih, _ic, _ib;
      if constexpr (is_channels_last) {
        _ic = i % ic_;
        _iw = i / ic_ % iw_;
        _ih = i / ic_ / iw_ % ih_;
        _ib = i / ic_ / iw_ / ih_;
      } else {
        _iw = i % iw_;
        _ih = i / iw_ % ih_;
        _ic = i / iw_ / ih_ % ic_;
        _ib = i / iw_ / ih_ / ic_;
      }

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
    local_range_ = syclMaxWorkItemsPerEU();
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

template <typename scalar_t, typename opmath_t, bool is_channels_last>
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
      if constexpr (is_channels_last) {
        _ic = i % ic_;
        _iw = i / ic_ % iw_;
        _ih = i / ic_ / iw_ % ih_;
        _ib = i / ic_ / iw_ / ih_;
      } else {
        _iw = i % iw_;
        _ih = i / iw_ % ih_;
        _ic = i / iw_ / ih_ % ic_;
        _ib = i / iw_ / ih_ / ic_;
      }

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

  auto outputHeight = grad_output.size(-2);
  auto outputWidth = grad_output.size(-1);

  const auto nInputPlane = input.size(-3);
  const auto inputHeight = input.size(-2);
  const auto inputWidth = input.size(-1);

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

  bool is_3d = grad_output.ndimension() == 3;
  if (is_3d) {
    grad_output.resize_({1, nInputPlane, outputHeight, outputWidth});
    grad_input.resize_({1, nInputPlane, inputHeight, inputWidth});
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      grad_output.scalar_type(),
      "adaptive_avg_pool2d_backward_xpu",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        auto gyacc = grad_output.packed_accessor64<const scalar_t, 4>();
        auto gxacc = grad_input.packed_accessor64<scalar_t, 4>();

        int64_t ohw01_shared_size =
            ((inputHeight + inputWidth) * 2) * sizeof(int);
        int64_t ikhw_shared_size =
            (outputHeight + outputWidth) * sizeof(opmath_t);
        bool using_shared =
            syclLocalMemSize() >= ohw01_shared_size + ikhw_shared_size;

        auto& q = getCurrentSYCLQueue();
        if (is_smf_channels_last(grad_output)) {
          if (using_shared) {
            AdaptiveAvgPool2dBwdSLMKernelFunctor<scalar_t, opmath_t, true> kfn(
                gyacc, gxacc);
            sycl_kernel_submit(kfn.glb_range(), kfn.loc_range(), q, kfn);
          } else {
            AdaptiveAvgPool2dBwdKernelFunctor<scalar_t, opmath_t, true> kfn(
                gyacc, gxacc);
            sycl_kernel_submit(kfn.glb_range(), kfn.loc_range(), q, kfn);
          }
        } else {
          if (using_shared) {
            AdaptiveAvgPool2dBwdSLMKernelFunctor<scalar_t, opmath_t, false> kfn(
                gyacc, gxacc);
            sycl_kernel_submit(kfn.glb_range(), kfn.loc_range(), q, kfn);
          } else {
            AdaptiveAvgPool2dBwdKernelFunctor<scalar_t, opmath_t, false> kfn(
                gyacc, gxacc);
            sycl_kernel_submit(kfn.glb_range(), kfn.loc_range(), q, kfn);
          }
        }
      });

  if (is_3d) {
    grad_output.resize_({nInputPlane, outputHeight, outputWidth});
    grad_input.resize_({nInputPlane, inputHeight, inputWidth});
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
  int local_range = syclMaxWorkItemsPerEU();
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
          launch_adaptive_avg_pool2d_kernel<scalar_t, opmath_t, true>(
              iacc, oacc);
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
