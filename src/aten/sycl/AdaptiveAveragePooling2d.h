#include <ATen/ATen.h>
#include <oneDNN/oneDNN.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/Pool.h>
#include <vector>

namespace at::native::xpu {

using namespace dnnl;
using namespace xpu::dpcpp;
using namespace xpu::oneDNN;

template <
    typename scalar_t,
    typename accscalar_t,
    bool is_channels_last,
    bool using_shared>
class adaptive_avg_pool2d_backward_kernel {
 public:
  void operator()(
      PackedTensorAccessor64<scalar_t, 4> gyacc,
      PackedTensorAccessor64<scalar_t, 4> gxacc) {}
};

template <typename scalar_t, typename accscalar_t, bool is_channels_last>
struct AdaptiveAvgPool2dBackwardKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    int64_t gi = item.get_global_linear_id();
    int64_t li = item.get_local_id(0);

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
  AdaptiveAvgPool2dBackwardKernelFunctor(
      int ib_,
      int ic_,
      int ih_,
      int iw_,
      int oh_,
      int ow_,
      int64_t numel_,
      int global_range_,
      PackedTensorAccessor64<scalar_t, 4> gyacc_,
      PackedTensorAccessor64<scalar_t, 4> gxacc_)
      : ib(ib_),
        ic(ic_),
        ih(ih_),
        iw(iw_),
        oh(oh_),
        ow(ow_),
        numel(numel_),
        global_range(global_range_),
        gyacc(gyacc_),
        gxacc(gxacc_) {}

 private:
  int ib;
  int ic;
  int ih;
  int iw;
  int oh;
  int ow;
  int64_t numel;
  int global_range;
  PackedTensorAccessor64<scalar_t, 4> gyacc;
  PackedTensorAccessor64<scalar_t, 4> gxacc;
};

template <typename scalar_t, typename accscalar_t, bool is_channels_last>
class adaptive_avg_pool2d_backward_kernel<
    scalar_t,
    accscalar_t,
    is_channels_last,
    false> {
 public:
  void operator()(
      PackedTensorAccessor64<scalar_t, 4> gyacc,
      PackedTensorAccessor64<scalar_t, 4> gxacc) {
    int ib = gxacc.size(0);
    int ic = gxacc.size(1);
    int ih = gxacc.size(2);
    int iw = gxacc.size(3);
    int oh = gyacc.size(2);
    int ow = gyacc.size(3);

    int64_t numel = ib * ic * ih * iw;
    int total_item = std::min(numel, dpcppMaxWorkItemsPerTile());
    int local_range = dpcppMaxWorkItemsPerEU();
    int global_range = total_item < local_range
        ? local_range
        : (total_item / local_range) * local_range;

    auto cgf = DPCPP_Q_CGF(cgh) {
      AdaptiveAvgPool2dBackwardKernelFunctor<
          scalar_t,
          accscalar_t,
          is_channels_last>
          kfn(ib, ic, ih, iw, oh, ow, numel, global_range, gyacc, gxacc);

      cgh.parallel_for<decltype(kfn)>(
          sycl::nd_range<1>(
              sycl::range<1>(global_range), sycl::range<1>(local_range)),
          kfn);
    };

    DPCPP_Q_SUBMIT(dpcppGetCurrentQueue(), cgf);
  }
};

template <typename scalar_t, typename accscalar_t, bool is_channels_last>
struct AdaptiveAvgPool2dBackwardKernel2Functor {
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

    item.barrier(dpcpp_local_fence);

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
  AdaptiveAvgPool2dBackwardKernel2Functor(
      int ib_,
      int ic_,
      int ih_,
      int iw_,
      int oh_,
      int ow_,
      int64_t numel_,
      int local_range_,
      int global_range_,
      PackedTensorAccessor64<scalar_t, 4> gyacc_,
      PackedTensorAccessor64<scalar_t, 4> gxacc_,
      int64_t ohw01_shared_size_,
      int64_t ikhw_shared_size_,
      dpcpp_local_acc_t<int> _oh0_cached_,
      dpcpp_local_acc_t<int> _oh1_cached_,
      dpcpp_local_acc_t<int> _ow0_cached_,
      dpcpp_local_acc_t<int> _ow1_cached_,
      dpcpp_local_acc_t<accscalar_t> _ikh_cached_,
      dpcpp_local_acc_t<accscalar_t> _ikw_cached_)
      : ib(ib_),
        ic(ic_),
        ih(ih_),
        iw(iw_),
        oh(oh_),
        ow(ow_),
        numel(numel_),
        local_range(local_range_),
        global_range(global_range_),
        gyacc(gyacc_),
        gxacc(gxacc_),
        ohw01_shared_size(ohw01_shared_size_),
        ikhw_shared_size(ikhw_shared_size_),
        _oh0_cached(_oh0_cached_),
        _oh1_cached(_oh1_cached_),
        _ow0_cached(_ow0_cached_),
        _ow1_cached(_ow1_cached_),
        _ikh_cached(_ikh_cached_),
        _ikw_cached(_ikw_cached_) {}

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
  dpcpp_local_acc_t<int> _oh0_cached;
  dpcpp_local_acc_t<int> _oh1_cached;
  dpcpp_local_acc_t<int> _ow0_cached;
  dpcpp_local_acc_t<int> _ow1_cached;
  dpcpp_local_acc_t<accscalar_t> _ikh_cached;
  dpcpp_local_acc_t<accscalar_t> _ikw_cached;
};

template <typename scalar_t, typename accscalar_t, bool is_channels_last>
class adaptive_avg_pool2d_backward_kernel<
    scalar_t,
    accscalar_t,
    is_channels_last,
    true> {
 public:
  void operator()(
      PackedTensorAccessor64<scalar_t, 4> gyacc,
      PackedTensorAccessor64<scalar_t, 4> gxacc) {
    int ib = gxacc.size(0);
    int ic = gxacc.size(1);
    int ih = gxacc.size(2);
    int iw = gxacc.size(3);
    int oh = gyacc.size(2);
    int ow = gyacc.size(3);

    int64_t numel = ib * ic * ih * iw;
    int total_item = std::min(numel, dpcppMaxWorkItemsPerTile());

    // Not use dpcppMaxWorkItemsPerEU to improve shared local memory usage.
    // Size of local memory is fixed (ih/iw/oh/ow) in the case.
    // Using max work group size to make more work items share same local
    // memory.
    int local_range = dpcppMaxWorkGroupSize();
    int global_range = total_item < local_range
        ? local_range
        : (total_item / local_range) * local_range;

    // trade-off occupancy and slm leverage
    int64_t ohw01_shared_size = ((iw + ih) * 2) * sizeof(int);
    int64_t ikhw_shared_size = (oh + ow) * sizeof(accscalar_t);

    auto cgf = DPCPP_Q_CGF(cgh) {
      dpcpp_local_acc_t<int> _oh0_cached(ih * sizeof(int), cgh);
      dpcpp_local_acc_t<int> _oh1_cached(ih * sizeof(int), cgh);
      dpcpp_local_acc_t<int> _ow0_cached(iw * sizeof(int), cgh);
      dpcpp_local_acc_t<int> _ow1_cached(iw * sizeof(int), cgh);
      dpcpp_local_acc_t<accscalar_t> _ikh_cached(oh * sizeof(accscalar_t), cgh);
      dpcpp_local_acc_t<accscalar_t> _ikw_cached(ow * sizeof(accscalar_t), cgh);

      AdaptiveAvgPool2dBackwardKernel2Functor<
          scalar_t,
          accscalar_t,
          is_channels_last>
          kfn(ib,
              ic,
              ih,
              iw,
              oh,
              ow,
              numel,
              local_range,
              global_range,
              gyacc,
              gxacc,
              ohw01_shared_size,
              ikhw_shared_size,
              _oh0_cached,
              _oh1_cached,
              _ow0_cached,
              _ow1_cached,
              _ikh_cached,
              _ikw_cached);

      cgh.parallel_for<decltype(kfn)>(
          sycl::nd_range<1>(
              sycl::range<1>(global_range), sycl::range<1>(local_range)),
          kfn);
    };

    DPCPP_Q_SUBMIT(dpcppGetCurrentQueue(), cgf);
  }
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

  /* sizes */
  const int64_t nbatch = input.ndimension() == 4 ? input.size(-4) : 1;
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

  // per oneDNN definition, no dilation means dilation ratio is 0
  std::vector<int64_t> dilation_vec = {0, 0};
  if (xpu::oneDNN::is_valid_pooling(
          {inputHeight, inputWidth},
          {outputHeight, inputHeight},
          {kH, kW},
          {dH, dW},
          {padH, padW})) {
    xpu::oneDNN::pooling_backward<
        xpu::oneDNN::alg::pooling_avg_exclude_padding>(
        gradInput,
        gradOutput,
        input,
        nbatch,
        nInputPlane,
        0,
        inputHeight,
        inputWidth,
        0,
        outputHeight,
        outputWidth,
        stride_vec,
        kernel_size_vec,
        dilation_vec,
        padding_vec,
        padding_vec);
  } else {
    auto gradOutput_ = to_plain_if_needed(gradOutput);

    bool is_3d = gradOutput_.ndimension() == 3;
    if (is_3d) {
      gradOutput_.resize_({1, nInputPlane, outputHeight, outputWidth});
      gradInput.resize_({1, nInputPlane, inputHeight, inputWidth});
    }

    IPEX_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::BFloat16,
        at::ScalarType::Half,
        gradOutput_.scalar_type(),
        "aten::adaptive_avg_pool2d_backward",
        [&]() {
          using accscalar_t = acc_type<scalar_t>;
          auto gyacc = gradOutput_.packed_accessor64<scalar_t, 4>();
          auto gxacc = gradInput.packed_accessor64<scalar_t, 4>();

          int64_t ohw01_shared_size =
              ((inputHeight + inputWidth) * 2) * sizeof(int);
          int64_t ikhw_shared_size =
              (outputHeight + outputWidth) * sizeof(accscalar_t);
          bool using_shared =
              dpcppLocalMemSize() >= ohw01_shared_size + ikhw_shared_size;

          if (is_smf_channels_last(gradOutput)) {
            if (using_shared) {
              adaptive_avg_pool2d_backward_kernel<
                  scalar_t,
                  accscalar_t,
                  true,
                  true>()(gyacc, gxacc);
            } else {
              adaptive_avg_pool2d_backward_kernel<
                  scalar_t,
                  accscalar_t,
                  true,
                  false>()(gyacc, gxacc);
            }
          } else {
            if (using_shared) {
              adaptive_avg_pool2d_backward_kernel<
                  scalar_t,
                  accscalar_t,
                  false,
                  true>()(gyacc, gxacc);
            } else {
              adaptive_avg_pool2d_backward_kernel<
                  scalar_t,
                  accscalar_t,
                  false,
                  false>()(gyacc, gxacc);
            }
          }
        });

    if (is_3d) {
      gradOutput_.resize_({nInputPlane, outputHeight, outputWidth});
      gradInput.resize_({nInputPlane, inputHeight, inputWidth});
    }
  }
}

} // namespace at::native::xpu