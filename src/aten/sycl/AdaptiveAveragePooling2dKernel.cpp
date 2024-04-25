#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/AdaptivePooling.h>
#include <ATen/native/Pool.h>
#include <comm/MemoryFormat.h>
#include <vector>

namespace at::native::xpu {

using namespace at::xpu;

template <typename scalar_t, typename accscalar_t, bool is_channels_last>
struct AdaptiveAvgPool2dBwdKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    int64_t gi = item.get_global_linear_id();

    for (int64_t i = gi; i < numel; i += global_range) {
      int64_t _iw, _ih, _ic, _ib;
      if constexpr (is_channels_last) {
        _ic = i % ic;
        _iw = i / ic % iw;
        _ih = i / ic / iw % ih;
        _ib = i / ic / iw / ih;
      } else {
        _iw = i % iw;
        _ih = i / iw % ih;
        _ic = i / iw / ih % ic;
        _ib = i / iw / ih / ic;
      }

      int64_t _oh0 = native::start_index(_ih, ih, oh);
      int64_t _oh1 = native::end_index(_ih, ih, oh);
      int64_t _ow0 = native::start_index(_iw, iw, ow);
      int64_t _ow1 = native::end_index(_iw, iw, ow);
      int64_t _ob = _ib;
      int64_t _oc = _ic;

      accscalar_t gx = 0;
      accscalar_t _ikh, _ikw;
      for (int _oh = _oh0; _oh < _oh1; _oh++) {
        _ikh = accscalar_t(1.0) /
            (accscalar_t)(native::end_index(_oh, oh, ih) - native::start_index(_oh, oh, ih));
        for (int _ow = _ow0; _ow < _ow1; _ow++) {
          _ikw = accscalar_t(1.0) /
              (accscalar_t)(native::end_index(_ow, ow, iw) - native::start_index(_ow, ow, iw));
          gx += gyacc[_ob][_oc][_oh][_ow] * _ikh * _ikw;
        }
      }

      const auto store = [](PackedTensorAccessor64<scalar_t, 4> gxacc,
                            int64_t _ib,
                            int64_t _ic,
                            int64_t _ih,
                            int64_t _iw,
                            scalar_t res) { gxacc[_ib][_ic][_ih][_iw] = res; };
      store(gxacc, _ib, _ic, _ih, _iw, (scalar_t)gx);
    }
  }

  AdaptiveAvgPool2dBwdKernelFunctor(
      PackedTensorAccessor64<scalar_t, 4> gyacc_,
      PackedTensorAccessor64<scalar_t, 4> gxacc_)
      : gyacc(gyacc_), gxacc(gxacc_) {
    ib = gxacc.size(0);
    ic = gxacc.size(1);
    ih = gxacc.size(2);
    iw = gxacc.size(3);
    oh = gyacc.size(2);
    ow = gyacc.size(3);

    numel = ib * ic * ih * iw;
    int total_item = std::min(numel, syclMaxWorkItemsPerTile());
    local_range = syclMaxWorkItemsPerEU();
    global_range = total_item < local_range
        ? local_range
        : (total_item / local_range) * local_range;
  }

  sycl::range<1> glb_range() {
    return sycl::range<1>(global_range);
  }

  sycl::range<1> loc_range() {
    return sycl::range<1>(local_range);
  }

 private:
  int ib;
  int ic;
  int ih;
  int iw;
  int oh;
  int ow;
  int64_t numel;
  int global_range;
  int local_range;
  PackedTensorAccessor64<scalar_t, 4> gyacc;
  PackedTensorAccessor64<scalar_t, 4> gxacc;
};

template <typename scalar_t, typename accscalar_t, bool is_channels_last>
struct AdaptiveAvgPool2dBwdSLMKernelFunctor
    : public __SYCL_KER_CONFIG_CONVENTION__ {
  void operator()(sycl::nd_item<1> item) const {
    int64_t gi = item.get_global_linear_id();
    int64_t li = item.get_local_id(0);

    // for-loop order: oh*ow->ih->iw
    // reuse oh*ow(oh0, oh1, ow0, ow1), ih(ikh), iw(ikw) in inner loop.
    for (int _ih = li; _ih < ih; _ih += local_range) {
      _oh0_cached[_ih] = (int)native::start_index(_ih, ih, oh);
      _oh1_cached[_ih] = (int)native::end_index(_ih, ih, oh);
    }
    for (int _iw = li; _iw < iw; _iw += local_range) {
      _ow0_cached[_iw] = (int)native::start_index(_iw, iw, ow);
      _ow1_cached[_iw] = (int)native::end_index(_iw, iw, ow);
    }
    for (int _oh = li; _oh < oh; _oh += local_range) {
      _ikh_cached[_oh] = accscalar_t(1.0) /
          (accscalar_t)(native::end_index(_oh, oh, ih) -
                        native::start_index(_oh, oh, ih));
    }
    for (int _ow = li; _ow < ow; _ow += local_range) {
      _ikw_cached[_ow] = accscalar_t(1.0) /
          (accscalar_t)(native::end_index(_ow, ow, iw) -
                        native::start_index(_ow, ow, iw));
    }

    item.barrier(sycl_local_fence);

    for (int64_t i = gi; i < numel; i += global_range) {
      int64_t _iw, _ih, _ic, _ib;
      if constexpr (is_channels_last) {
        _ic = i % ic;
        _iw = i / ic % iw;
        _ih = i / ic / iw % ih;
        _ib = i / ic / iw / ih;
      } else {
        _iw = i % iw;
        _ih = i / iw % ih;
        _ic = i / iw / ih % ic;
        _ib = i / iw / ih / ic;
      }

      int64_t _oh0, _oh1, _ow0, _ow1;
      _oh0 = _oh0_cached[_ih];
      _oh1 = _oh1_cached[_ih];
      _ow0 = _ow0_cached[_iw];
      _ow1 = _ow1_cached[_iw];
      int64_t _ob = _ib;
      int64_t _oc = _ic;

      accscalar_t gx = 0;
      accscalar_t _ikh, _ikw;
      for (int _oh = _oh0; _oh < _oh1; _oh++) {
        _ikh = _ikh_cached[_oh];
        for (int _ow = _ow0; _ow < _ow1; _ow++) {
          _ikw = _ikw_cached[_ow];
          gx += gyacc[_ob][_oc][_oh][_ow] * _ikh * _ikw;
        }
      }

      const auto store = [](PackedTensorAccessor64<scalar_t, 4> gxacc,
                            int64_t _ib,
                            int64_t _ic,
                            int64_t _ih,
                            int64_t _iw,
                            scalar_t res) { gxacc[_ib][_ic][_ih][_iw] = res; };
      store(gxacc, _ib, _ic, _ih, _iw, (scalar_t)gx);
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    _oh0_cached = sycl_local_acc_t<int>(ih * sizeof(int), cgh);
    _oh1_cached = sycl_local_acc_t<int>(ih * sizeof(int), cgh);
    _ow0_cached = sycl_local_acc_t<int>(iw * sizeof(int), cgh);
    _ow1_cached = sycl_local_acc_t<int>(iw * sizeof(int), cgh);
    _ikh_cached = sycl_local_acc_t<accscalar_t>(oh * sizeof(accscalar_t), cgh);
    _ikw_cached = sycl_local_acc_t<accscalar_t>(ow * sizeof(accscalar_t), cgh);
  }

  AdaptiveAvgPool2dBwdSLMKernelFunctor(
      PackedTensorAccessor64<scalar_t, 4> gyacc_,
      PackedTensorAccessor64<scalar_t, 4> gxacc_)
      : gyacc(gyacc_), gxacc(gxacc_) {
    ib = gxacc.size(0);
    ic = gxacc.size(1);
    ih = gxacc.size(2);
    iw = gxacc.size(3);
    oh = gyacc.size(2);
    ow = gyacc.size(3);

    numel = ib * ic * ih * iw;
    int total_item = std::min(numel, syclMaxWorkItemsPerTile());

    local_range = syclMaxWorkGroupSize();
    global_range = total_item < local_range
        ? local_range
        : (total_item / local_range) * local_range;

    // trade-off occupancy and slm leverage
    ohw01_shared_size = ((iw + ih) * 2) * sizeof(int);
    ikhw_shared_size = (oh + ow) * sizeof(accscalar_t);
  }

  sycl::range<1> glb_range() {
    return sycl::range<1>(global_range);
  }

  sycl::range<1> loc_range() {
    return sycl::range<1>(local_range);
  }

 private:
  int ib;
  int ic;
  int ih;
  int iw;
  int oh;
  int ow;
  int64_t numel;
  int local_range;
  int global_range;
  PackedTensorAccessor64<scalar_t, 4> gyacc;
  PackedTensorAccessor64<scalar_t, 4> gxacc;
  int64_t ohw01_shared_size;
  int64_t ikhw_shared_size;
  sycl_local_acc_t<int> _oh0_cached;
  sycl_local_acc_t<int> _oh1_cached;
  sycl_local_acc_t<int> _ow0_cached;
  sycl_local_acc_t<int> _ow1_cached;
  sycl_local_acc_t<accscalar_t> _ikh_cached;
  sycl_local_acc_t<accscalar_t> _ikw_cached;
};

void adaptive_avg_pool2d_backward_out_template(
    Tensor& gradInput,
    const Tensor& gradOutput,
    const Tensor& input) {
  TORCH_CHECK(
      (input.ndimension() == 3 || input.ndimension() == 4),
      "non-empty 3D or 4D (batch mode) tensor expected for input");

  auto outputHeight = gradOutput.size(-2);
  auto outputWidth = gradOutput.size(-1);

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

  bool is_3d = gradOutput.ndimension() == 3;
  if (is_3d) {
    gradOutput.resize_({1, nInputPlane, outputHeight, outputWidth});
    gradInput.resize_({1, nInputPlane, inputHeight, inputWidth});
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      gradOutput.scalar_type(),
      "adaptive_avg_pool2d_backward_xpu",
      [&]() {
        using accscalar_t = acc_type<scalar_t, false>;
        auto gyacc = gradOutput.packed_accessor64<scalar_t, 4>();
        auto gxacc = gradInput.packed_accessor64<scalar_t, 4>();

        int64_t ohw01_shared_size =
            ((inputHeight + inputWidth) * 2) * sizeof(int);
        int64_t ikhw_shared_size =
            (outputHeight + outputWidth) * sizeof(accscalar_t);
        bool using_shared =
            syclLocalMemSize() >= ohw01_shared_size + ikhw_shared_size;

        auto& q = getCurrentSYCLQueue();
        if (is_smf_channels_last(gradOutput)) {
          if (using_shared) {
            AdaptiveAvgPool2dBwdSLMKernelFunctor<scalar_t, accscalar_t, true>
                kfn(gyacc, gxacc);
            sycl_kernel_submit(kfn.glb_range(), kfn.loc_range(), q, kfn);
          } else {
            AdaptiveAvgPool2dBwdKernelFunctor<scalar_t, accscalar_t, true> kfn(
                gyacc, gxacc);
            sycl_kernel_submit(kfn.glb_range(), kfn.loc_range(), q, kfn);
          }
        } else {
          if (using_shared) {
            AdaptiveAvgPool2dBwdSLMKernelFunctor<scalar_t, accscalar_t, false>
                kfn(gyacc, gxacc);
            sycl_kernel_submit(kfn.glb_range(), kfn.loc_range(), q, kfn);
          } else {
            AdaptiveAvgPool2dBwdKernelFunctor<scalar_t, accscalar_t, false> kfn(
                gyacc, gxacc);
            sycl_kernel_submit(kfn.glb_range(), kfn.loc_range(), q, kfn);
          }
        }
      });

  if (is_3d) {
    gradOutput.resize_({nInputPlane, outputHeight, outputWidth});
    gradInput.resize_({nInputPlane, inputHeight, inputWidth});
  }
}

} // namespace at::native::xpu
