#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/NumericUtils.h>
#include <ATen/OpMathType.h>
#include <ATen/TensorIterator.h>
#include <ATen/core/TensorAccessor.h>
#include <ATen/native/CanUse32BitIndexMath.h>
#include <c10/util/generic_math.h>
#include <comm/SYCLContext.h>
#include <comm/TensorInfo.h>

#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>

#include <ATen/native/xpu/sycl/Loops.h>
#include <ATen/native/xpu/sycl/RNNKernels.h>

namespace at::native::xpu {

using at::native::canUse32BitIndexMath;
using at::xpu::detail::getTensorInfo;
using at::xpu::detail::IndexToOffset;
using at::xpu::detail::TensorInfo;

std::tuple<int64_t, int64_t> rnn_get_launch_config(
    int64_t max_threads_per_group,
    int64_t numel) {
  int64_t num_groups =
      (numel + max_threads_per_group - 1) / max_threads_per_group;
  auto hw_max_groups = syclMaxWorkItemsPerTile() / max_threads_per_group;
  num_groups = num_groups > hw_max_groups ? hw_max_groups : num_groups;
  return std::make_tuple(num_groups, max_threads_per_group);
}

// Factor will be 3 for GRU and 4 for LSTM
void checkSizes(
    CheckedFrom c,
    const TensorArg& input_gates,
    const TensorArg& hidden_gates,
    const TensorArg& input_bias,
    const TensorArg& hidden_bias,
    int64_t factor,
    const TensorArg& prev_hidden) {
  checkDim(c, input_gates, 2);
  checkSameSize(c, input_gates, hidden_gates);
  int64_t gates_size = input_gates->size(1);

  if (input_bias->defined()) {
    checkDim(c, input_bias, 1);
    checkNumel(c, input_bias, gates_size);
    checkSameSize(c, input_bias, hidden_bias);
  }

  checkDim(c, prev_hidden, 2);
  checkNumel(c, prev_hidden, input_gates->size(0) * gates_size / factor);

  checkAllSameGPU(
      c, {input_gates, hidden_gates, input_bias, hidden_bias, prev_hidden});
}

bool allContiguous(at::TensorList tensors) {
  return std::all_of(tensors.begin(), tensors.end(), [](const at::Tensor& t) {
    return !t.defined() || t.is_contiguous();
  });
}

template <typename T, typename T2>
TensorInfo<T, T2> tryGetTensorInfo(const at::Tensor& t) {
  return t.defined() ? getTensorInfo<T, T2>(t) : TensorInfo<T, T2>{};
}

void collapseDims(){};
template <typename T, typename T2, typename... Args>
void collapseDims(TensorInfo<T, T2>& info, Args&... infos) {
  info.collapseDims();
  collapseDims(infos...);
}

#define DEVICE_LINEAR_GET(D_TENSOR, INDEX) \
  D_TENSOR.data[IndexToOffset<scalar_t, index_type>::get(INDEX, D_TENSOR)]

// Biases are always 1D
#define DEVICE_BIAS_GET(D_TENSOR, INDEX) \
  D_TENSOR.data[IndexToOffset<scalar_t, index_type>::get(INDEX, D_TENSOR)]

#define H2F(input) static_cast<accscalar_t>(input)
#define F2H(input) static_cast<scalar_t>(input)

template <typename T>
inline T sigmoid(T in) {
  T one = static_cast<T>(1.0);
  return one / (one + std::exp(-in));
}

template <typename scalar_t, typename accscalar_t, typename index_type>
struct LstmCellForwardFunctor {
  void operator()(sycl::nd_item<1> item) const {
    bool has_bias = bias1_.data != nullptr;

    for (index_type linearIndex = item.get_global_id(0);
         linearIndex < totalElements_;
         linearIndex += item.get_group_range(0) * item.get_local_range(0)) {
      index_type offset = (linearIndex / hsz_) * 4 * hsz_ + linearIndex % hsz_;

      scalar_t iig = DEVICE_LINEAR_GET(input_, offset + 0 * hsz_);
      scalar_t ifg = DEVICE_LINEAR_GET(input_, offset + 1 * hsz_);
      scalar_t icg = DEVICE_LINEAR_GET(input_, offset + 2 * hsz_);
      scalar_t iog = DEVICE_LINEAR_GET(input_, offset + 3 * hsz_);

      scalar_t hig = DEVICE_LINEAR_GET(hidden_, offset + 0 * hsz_);
      scalar_t hfg = DEVICE_LINEAR_GET(hidden_, offset + 1 * hsz_);
      scalar_t hcg = DEVICE_LINEAR_GET(hidden_, offset + 2 * hsz_);
      scalar_t hog = DEVICE_LINEAR_GET(hidden_, offset + 3 * hsz_);

      scalar_t* wig = &DEVICE_LINEAR_GET(workspace_, offset + 0 * hsz_);
      scalar_t* wfg = &DEVICE_LINEAR_GET(workspace_, offset + 1 * hsz_);
      scalar_t* wcg = &DEVICE_LINEAR_GET(workspace_, offset + 2 * hsz_);
      scalar_t* wog = &DEVICE_LINEAR_GET(workspace_, offset + 3 * hsz_);

      scalar_t cx = DEVICE_LINEAR_GET(_cx_, linearIndex);

      scalar_t* hy = &DEVICE_LINEAR_GET(_hy_, linearIndex);
      scalar_t* cy = &DEVICE_LINEAR_GET(_cy_, linearIndex);

      scalar_t b1i, b1f, b1c, b1o;
      scalar_t b2i, b2f, b2c, b2o;

      if (has_bias) {
        b1i = DEVICE_BIAS_GET(bias1_, linearIndex % hsz_ + 0 * hsz_);
        b1f = DEVICE_BIAS_GET(bias1_, linearIndex % hsz_ + 1 * hsz_);
        b1c = DEVICE_BIAS_GET(bias1_, linearIndex % hsz_ + 2 * hsz_);
        b1o = DEVICE_BIAS_GET(bias1_, linearIndex % hsz_ + 3 * hsz_);

        b2i = DEVICE_BIAS_GET(bias2_, linearIndex % hsz_ + 0 * hsz_);
        b2f = DEVICE_BIAS_GET(bias2_, linearIndex % hsz_ + 1 * hsz_);
        b2c = DEVICE_BIAS_GET(bias2_, linearIndex % hsz_ + 2 * hsz_);
        b2o = DEVICE_BIAS_GET(bias2_, linearIndex % hsz_ + 3 * hsz_);
      } else {
        b1i = F2H(0.0);
        b1f = F2H(0.0);
        b1c = F2H(0.0);
        b1o = F2H(0.0);
        b2i = F2H(0.0);
        b2f = F2H(0.0);
        b2c = F2H(0.0);
        b2o = F2H(0.0);
      }

      accscalar_t ig, fg, cg, og;
      accscalar_t f_hy, f_cy;

      ig = sigmoid(H2F(iig) + H2F(hig) + H2F(b1i) + H2F(b2i));
      fg = sigmoid(H2F(ifg) + H2F(hfg) + H2F(b1f) + H2F(b2f));
      cg = std::tanh(H2F(icg) + H2F(hcg) + H2F(b1c) + H2F(b2c));
      og = sigmoid(H2F(iog) + H2F(hog) + H2F(b1o) + H2F(b2o));

      f_cy = (fg * H2F(cx)) + (ig * cg);
      f_hy = og * std::tanh(f_cy);

      *hy = F2H(f_hy);
      *cy = F2H(f_cy);

      // SAVE FOR BACKWARDS
      // Also need cy and cx but can be saved easily in python
      *wig = F2H(ig);
      *wfg = F2H(fg);
      *wcg = F2H(cg);
      *wog = F2H(og);
    }
  }

  LstmCellForwardFunctor(
      TensorInfo<scalar_t, index_type> input,
      TensorInfo<scalar_t, index_type> hidden,
      TensorInfo<scalar_t, index_type> bias1,
      TensorInfo<scalar_t, index_type> bias2,
      TensorInfo<scalar_t, index_type> _cx,
      TensorInfo<scalar_t, index_type> _hy,
      TensorInfo<scalar_t, index_type> _cy,
      TensorInfo<scalar_t, index_type> workspace,
      index_type hsz,
      index_type totalElements)
      : input_(input),
        hidden_(hidden),
        bias1_(bias1),
        bias2_(bias2),
        _cx_(_cx),
        _hy_(_hy),
        _cy_(_cy),
        workspace_(workspace),
        hsz_(hsz),
        totalElements_(totalElements) {}

 private:
  TensorInfo<scalar_t, index_type> input_;
  TensorInfo<scalar_t, index_type> hidden_;
  TensorInfo<scalar_t, index_type> bias1_;
  TensorInfo<scalar_t, index_type> bias2_;
  TensorInfo<scalar_t, index_type> _cx_;
  TensorInfo<scalar_t, index_type> _hy_;
  TensorInfo<scalar_t, index_type> _cy_;
  TensorInfo<scalar_t, index_type> workspace_;
  index_type hsz_;
  index_type totalElements_;
};

template <typename scalar_t, typename accscalar_t, typename index_type>
struct LstmCellBackwardFunctor {
  void operator()(sycl::nd_item<1> item) const {
    bool has_gradoutput = gradoutput_.data != nullptr;
    bool has_gradoutputcell = gradoutputcell_.data != nullptr;

    for (index_type linearIndex = item.get_global_id(0);
         linearIndex < totalElements_;
         linearIndex += item.get_group_range(0) * item.get_local_range(0)) {
      index_type offset = (linearIndex / hsz_) * 4 * hsz_ + linearIndex % hsz_;

      scalar_t ig = DEVICE_LINEAR_GET(storage_, offset + 0 * hsz_);
      scalar_t fg = DEVICE_LINEAR_GET(storage_, offset + 1 * hsz_);
      scalar_t cg = DEVICE_LINEAR_GET(storage_, offset + 2 * hsz_);
      scalar_t og = DEVICE_LINEAR_GET(storage_, offset + 3 * hsz_);

      scalar_t* ih = &DEVICE_LINEAR_GET(gradInGates_, offset + 0 * hsz_);
      scalar_t* fh = &DEVICE_LINEAR_GET(gradInGates_, offset + 1 * hsz_);
      scalar_t* ch = &DEVICE_LINEAR_GET(gradInGates_, offset + 2 * hsz_);
      scalar_t* oh = &DEVICE_LINEAR_GET(gradInGates_, offset + 3 * hsz_);

      // will return hidden grads here
      scalar_t cx = DEVICE_LINEAR_GET(_cx_, linearIndex);
      scalar_t cy = DEVICE_LINEAR_GET(_cy_, linearIndex);

      scalar_t* gi = &DEVICE_LINEAR_GET(gradInputCx_, linearIndex);

      accscalar_t go = has_gradoutput
          ? H2F(DEVICE_LINEAR_GET(gradoutput_, linearIndex))
          : 0.f;
      accscalar_t goc = has_gradoutputcell
          ? H2F(DEVICE_LINEAR_GET(gradoutputcell_, linearIndex))
          : 0.f;

      accscalar_t gcx = std::tanh(H2F(cy));

      accscalar_t gog = go * gcx;
      gcx = go * H2F(og) * (1 - gcx * gcx) + goc;

      accscalar_t gig = gcx * H2F(cg);
      accscalar_t gfg = gcx * H2F(cx);
      accscalar_t gcg = gcx * H2F(ig);

      gcx = gcx * H2F(fg);

      gig = gig * (1 - H2F(ig)) * H2F(ig);
      gfg = gfg * (1 - H2F(fg)) * H2F(fg);
      gcg = gcg * (1 - H2F(cg) * H2F(cg));
      gog = gog * (1 - H2F(og)) * H2F(og);

      *ih = F2H(gig);
      *fh = F2H(gfg);
      *ch = F2H(gcg);
      *oh = F2H(gog);

      *gi = F2H(gcx);
    }
  }

  LstmCellBackwardFunctor(
      TensorInfo<scalar_t, index_type> storage,
      TensorInfo<scalar_t, index_type> gradInGates,
      TensorInfo<scalar_t, index_type> _cx,
      TensorInfo<scalar_t, index_type> _cy,
      TensorInfo<scalar_t, index_type> gradoutput,
      TensorInfo<scalar_t, index_type> gradoutputcell,
      TensorInfo<scalar_t, index_type> gradInputCx,
      index_type hsz,
      index_type totalElements)
      : storage_(storage),
        gradInGates_(gradInGates),
        _cx_(_cx),
        _cy_(_cy),
        gradoutput_(gradoutput),
        gradoutputcell_(gradoutputcell),
        gradInputCx_(gradInputCx),
        hsz_(hsz),
        totalElements_(totalElements) {}

 private:
  TensorInfo<scalar_t, index_type> storage_;
  TensorInfo<scalar_t, index_type> gradInGates_;
  TensorInfo<scalar_t, index_type> _cx_;
  TensorInfo<scalar_t, index_type> _cy_;
  TensorInfo<scalar_t, index_type> gradoutput_;
  TensorInfo<scalar_t, index_type> gradoutputcell_;
  TensorInfo<scalar_t, index_type> gradInputCx_;
  index_type hsz_;
  index_type totalElements_;
};

template <typename scalar_t, typename accscalar_t, typename index_type>
struct GruCellForwardFunctor {
  void operator()(sycl::nd_item<1> item) const {
    bool has_bias = Bias1_.data != nullptr;

    for (index_type linearIndex = item.get_global_id(0);
         linearIndex < totalElements_;
         linearIndex += item.get_group_range(0) * item.get_local_range(0)) {
      index_type offset = (linearIndex / hsz_) * 3 * hsz_ + linearIndex % hsz_;

      scalar_t ir = DEVICE_LINEAR_GET(Input_, offset + 0 * hsz_);
      scalar_t ii = DEVICE_LINEAR_GET(Input_, offset + 1 * hsz_);
      scalar_t in = DEVICE_LINEAR_GET(Input_, offset + 2 * hsz_);
      scalar_t hr = DEVICE_LINEAR_GET(Hidden_, offset + 0 * hsz_);
      scalar_t hi = DEVICE_LINEAR_GET(Hidden_, offset + 1 * hsz_);
      scalar_t hn = DEVICE_LINEAR_GET(Hidden_, offset + 2 * hsz_);

      scalar_t hx = DEVICE_LINEAR_GET(_hx_, linearIndex);
      scalar_t* hy = &DEVICE_LINEAR_GET(_hy_, linearIndex);

      scalar_t b1r, b1i, b1n, b2r, b2i, b2n;

      if (has_bias) {
        b1r = DEVICE_BIAS_GET(Bias1_, linearIndex % hsz_ + 0 * hsz_);
        b1i = DEVICE_BIAS_GET(Bias1_, linearIndex % hsz_ + 1 * hsz_);
        b1n = DEVICE_BIAS_GET(Bias1_, linearIndex % hsz_ + 2 * hsz_);

        b2r = DEVICE_BIAS_GET(Bias2_, linearIndex % hsz_ + 0 * hsz_);
        b2i = DEVICE_BIAS_GET(Bias2_, linearIndex % hsz_ + 1 * hsz_);
        b2n = DEVICE_BIAS_GET(Bias2_, linearIndex % hsz_ + 2 * hsz_);
      } else {
        b1r = F2H(0.0);
        b1i = F2H(0.0);
        b1n = F2H(0.0);
        b2r = F2H(0.0);
        b2i = F2H(0.0);
        b2n = F2H(0.0);
      }

      offset = (linearIndex / hsz_) * 5 * hsz_ + linearIndex % hsz_;

      accscalar_t rg, ig, ng;

      rg = sigmoid<accscalar_t>(H2F(ir) + H2F(hr) + H2F(b1r) + H2F(b2r));
      ig = sigmoid<accscalar_t>(H2F(ii) + H2F(hi) + H2F(b1i) + H2F(b2i));

      ng = H2F(in) + H2F(b1n) + rg * (H2F(hn) + H2F(b2n));
      ng = std::tanh(ng);
      *hy = F2H(ng + ig * (H2F(hx) - ng));

      // SAVE FOR BACKWARDS
      DEVICE_LINEAR_GET(storage_, offset + 0 * hsz_) = F2H(rg);
      DEVICE_LINEAR_GET(storage_, offset + 1 * hsz_) = F2H(ig);
      DEVICE_LINEAR_GET(storage_, offset + 2 * hsz_) = F2H(ng);
      DEVICE_LINEAR_GET(storage_, offset + 3 * hsz_) = hx;
      DEVICE_LINEAR_GET(storage_, offset + 4 * hsz_) = F2H(H2F(hn) + H2F(b2n));
    }
  }

  GruCellForwardFunctor(
      TensorInfo<scalar_t, index_type> Input,
      const TensorInfo<scalar_t, index_type> Hidden,
      const TensorInfo<scalar_t, index_type> Bias1,
      const TensorInfo<scalar_t, index_type> Bias2,
      const TensorInfo<scalar_t, index_type> _hx,
      const TensorInfo<scalar_t, index_type> _hy,
      const TensorInfo<scalar_t, index_type> storage,
      const index_type hsz,
      const index_type totalElements)
      : Input_(Input),
        Hidden_(Hidden),
        Bias1_(Bias1),
        Bias2_(Bias2),
        _hx_(_hx),
        _hy_(_hy),
        storage_(storage),
        hsz_(hsz),
        totalElements_(totalElements) {}

 private:
  TensorInfo<scalar_t, index_type> Input_;
  const TensorInfo<scalar_t, index_type> Hidden_;
  const TensorInfo<scalar_t, index_type> Bias1_;
  const TensorInfo<scalar_t, index_type> Bias2_;
  const TensorInfo<scalar_t, index_type> _hx_;
  const TensorInfo<scalar_t, index_type> _hy_;
  const TensorInfo<scalar_t, index_type> storage_;
  const index_type hsz_;
  const index_type totalElements_;
};

template <typename scalar_t, typename accscalar_t, typename index_type>
struct GruCellBackwardFunctor {
  void operator()(sycl::nd_item<1> item) const {
    for (index_type linearIndex = item.get_global_id(0);
         linearIndex < totalElements_;
         linearIndex += item.get_group_range(0) * item.get_local_range(0)) {
      index_type offset = (linearIndex / hsz_) * 5 * hsz_ + linearIndex % hsz_;

      scalar_t rg = DEVICE_LINEAR_GET(storage_, offset + 0 * hsz_);
      scalar_t ig = DEVICE_LINEAR_GET(storage_, offset + 1 * hsz_);
      scalar_t ng = DEVICE_LINEAR_GET(storage_, offset + 2 * hsz_);
      scalar_t hx = DEVICE_LINEAR_GET(storage_, offset + 3 * hsz_);
      scalar_t hn = DEVICE_LINEAR_GET(storage_, offset + 4 * hsz_);

      scalar_t go = DEVICE_LINEAR_GET(gradOutput_, linearIndex);

      offset = (linearIndex / hsz_) * 3 * hsz_ + linearIndex % hsz_;

      accscalar_t gig = H2F(go) * (H2F(hx) - H2F(ng)) * (1 - H2F(ig)) * H2F(ig);
      accscalar_t ghx = H2F(go) * H2F(ig);
      accscalar_t gin = H2F(go) * (1 - H2F(ig)) * (1 - H2F(ng) * H2F(ng));
      accscalar_t ghn = gin * H2F(rg);
      accscalar_t grg = gin * H2F(hn) * (1 - H2F(rg)) * H2F(rg);

      DEVICE_LINEAR_GET(gradInInput_, offset + 0 * hsz_) = F2H(grg);
      DEVICE_LINEAR_GET(gradInInput_, offset + 1 * hsz_) = F2H(gig);
      DEVICE_LINEAR_GET(gradInInput_, offset + 2 * hsz_) = F2H(gin);

      DEVICE_LINEAR_GET(gradInHidden_, offset + 0 * hsz_) = F2H(grg);
      DEVICE_LINEAR_GET(gradInHidden_, offset + 1 * hsz_) = F2H(gig);
      DEVICE_LINEAR_GET(gradInHidden_, offset + 2 * hsz_) = F2H(ghn);
      DEVICE_LINEAR_GET(gradInputHx_, linearIndex) = F2H(ghx);
    }
  }

  GruCellBackwardFunctor(
      TensorInfo<scalar_t, index_type> gradInInput,
      TensorInfo<scalar_t, index_type> gradInHidden,
      TensorInfo<scalar_t, index_type> gradOutput,
      TensorInfo<scalar_t, index_type> gradInputHx,
      TensorInfo<scalar_t, index_type> storage,
      index_type hsz,
      index_type totalElements)
      : gradInInput_(gradInInput),
        gradInHidden_(gradInHidden),
        gradOutput_(gradOutput),
        gradInputHx_(gradInputHx),
        storage_(storage),
        hsz_(hsz),
        totalElements_(totalElements) {}

 private:
  TensorInfo<scalar_t, index_type> gradInInput_;
  TensorInfo<scalar_t, index_type> gradInHidden_;
  TensorInfo<scalar_t, index_type> gradOutput_;
  TensorInfo<scalar_t, index_type> gradInputHx_;
  TensorInfo<scalar_t, index_type> storage_;
  index_type hsz_;
  index_type totalElements_;
};

#undef DEVICE_LINEAR_GET
#undef DEVICE_BIAS_GET
#undef H2F
#undef F2H

template <typename scalar_t, typename index_type>
void lstm_forward_impl(
    const Tensor& input_gates,
    const Tensor& hidden_gates,
    const Tensor& input_bias,
    const Tensor& hidden_bias,
    const Tensor& cx,
    const Tensor& hy,
    const Tensor& cy,
    const Tensor& workspace) {
  using accscalar_t = at::acc_type_device<scalar_t, kXPU>;

  int64_t numel = cx.numel();
  if (numel == 0)
    return;

  using KernelT = LstmCellForwardFunctor<scalar_t, accscalar_t, index_type>;
  auto max_wg_size = syclMaxWorkGroupSize<KernelT>();
  auto config = rnn_get_launch_config(max_wg_size, numel);
  auto nwg = std::get<0>(config);
  auto local_range = std::get<1>(config);

  auto input_gatesI = getTensorInfo<scalar_t, index_type>(input_gates);
  auto hidden_gatesI = getTensorInfo<scalar_t, index_type>(hidden_gates);
  auto input_biasI = tryGetTensorInfo<scalar_t, index_type>(input_bias);
  auto hidden_biasI = tryGetTensorInfo<scalar_t, index_type>(hidden_bias);
  auto cxI = getTensorInfo<scalar_t, index_type>(cx);
  auto hyI = getTensorInfo<scalar_t, index_type>(hy);
  auto cyI = getTensorInfo<scalar_t, index_type>(cy);
  auto workspaceI = getTensorInfo<scalar_t, index_type>(workspace);
  index_type hidden_size = cxI.sizes[cxI.dims - 1];

  if (allContiguous(
          {input_gates,
           hidden_gates,
           input_bias,
           hidden_bias,
           cx,
           hy,
           cy,
           workspace})) {
    collapseDims(
        input_gatesI,
        hidden_gatesI,
        input_biasI,
        hidden_biasI,
        cxI,
        hyI,
        cyI,
        workspaceI);
    KernelT kfn(
        input_gatesI,
        hidden_gatesI,
        input_biasI,
        hidden_biasI,
        cxI,
        hyI,
        cyI,
        workspaceI,
        hidden_size,
        numel);
    sycl_kernel_submit(
        nwg * local_range, local_range, getCurrentSYCLQueue(), kfn);
  } else {
    KernelT kfn(
        input_gatesI,
        hidden_gatesI,
        input_biasI,
        hidden_biasI,
        cxI,
        hyI,
        cyI,
        workspaceI,
        hidden_size,
        numel);
    sycl_kernel_submit(
        nwg * local_range, local_range, getCurrentSYCLQueue(), kfn);
  }
}

template <typename scalar_t, typename index_type>
void lstm_backward_impl(
    const Tensor& grad_hy,
    const Tensor& grad_cy,
    const Tensor& cx,
    const Tensor& cy,
    const Tensor& workspace,
    const Tensor& grad_gates,
    const Tensor& grad_cx) {
  using accscalar_t = at::acc_type_device<scalar_t, kXPU>;

  int64_t numel = cx.numel();
  if (numel == 0)
    return;

  using KernelT = LstmCellBackwardFunctor<scalar_t, accscalar_t, index_type>;
  auto max_wg_size = syclMaxWorkGroupSize<KernelT>();
  auto config = rnn_get_launch_config(max_wg_size, numel);
  auto nwg = std::get<0>(config);
  auto local_range = std::get<1>(config);

  auto grad_hyI = tryGetTensorInfo<scalar_t, index_type>(grad_hy);
  auto grad_cyI = tryGetTensorInfo<scalar_t, index_type>(grad_cy);
  auto cxI = getTensorInfo<scalar_t, index_type>(cx);
  auto cyI = getTensorInfo<scalar_t, index_type>(cy);
  auto workspaceI = getTensorInfo<scalar_t, index_type>(workspace);
  auto grad_gatesI = getTensorInfo<scalar_t, index_type>(grad_gates);
  auto grad_cxI = getTensorInfo<scalar_t, index_type>(grad_cx);
  index_type hidden_size = cxI.sizes[cxI.dims - 1];

  if (allContiguous(
          {grad_hy, grad_cy, cx, cy, workspace, grad_gates, grad_cx})) {
    collapseDims(
        grad_hyI, grad_cyI, cxI, cyI, workspaceI, grad_gatesI, grad_cxI);
    KernelT kfn(
        workspaceI,
        grad_gatesI,
        cxI,
        cyI,
        grad_hyI,
        grad_cyI,
        grad_cxI,
        hidden_size,
        numel);
    sycl_kernel_submit(
        nwg * local_range, local_range, getCurrentSYCLQueue(), kfn);
  } else {
    KernelT kfn(
        workspaceI,
        grad_gatesI,
        cxI,
        cyI,
        grad_hyI,
        grad_cyI,
        grad_cxI,
        hidden_size,
        numel);
    sycl_kernel_submit(
        nwg * local_range, local_range, getCurrentSYCLQueue(), kfn);
  }
}

template <typename scalar_t, typename index_type>
void gru_forward_impl(
    const Tensor& input_gates,
    const Tensor& hidden_gates,
    const Tensor& input_bias,
    const Tensor& hidden_bias,
    const Tensor& hx,
    const Tensor& hy,
    const Tensor& workspace) {
  using accscalar_t = at::acc_type_device<scalar_t, kXPU>;

  int64_t numel = hx.numel();
  if (numel == 0)
    return;

  using KernelT = GruCellForwardFunctor<scalar_t, accscalar_t, index_type>;
  auto max_wg_size = syclMaxWorkGroupSize<KernelT>();
  auto config = rnn_get_launch_config(max_wg_size, numel);
  auto nwg = std::get<0>(config);
  auto local_range = std::get<1>(config);

  auto input_gatesI = getTensorInfo<scalar_t, index_type>(input_gates);
  auto hidden_gatesI = getTensorInfo<scalar_t, index_type>(hidden_gates);
  auto input_biasI = tryGetTensorInfo<scalar_t, index_type>(input_bias);
  auto hidden_biasI = tryGetTensorInfo<scalar_t, index_type>(hidden_bias);
  auto hxI = getTensorInfo<scalar_t, index_type>(hx);
  auto hyI = getTensorInfo<scalar_t, index_type>(hy);
  auto workspaceI = getTensorInfo<scalar_t, index_type>(workspace);
  index_type hidden_size = hxI.sizes[hxI.dims - 1];

  if (allContiguous(
          {input_gates,
           hidden_gates,
           input_bias,
           hidden_bias,
           hx,
           hy,
           workspace})) {
    collapseDims(
        input_gatesI,
        hidden_gatesI,
        input_biasI,
        hidden_biasI,
        hxI,
        hyI,
        workspaceI);
    KernelT kfn(
        input_gatesI,
        hidden_gatesI,
        input_biasI,
        hidden_biasI,
        hxI,
        hyI,
        workspaceI,
        hidden_size,
        numel);
    sycl_kernel_submit(
        nwg * local_range, local_range, getCurrentSYCLQueue(), kfn);
  } else {
    KernelT kfn(
        input_gatesI,
        hidden_gatesI,
        input_biasI,
        hidden_biasI,
        hxI,
        hyI,
        workspaceI,
        hidden_size,
        numel);
    sycl_kernel_submit(
        nwg * local_range, local_range, getCurrentSYCLQueue(), kfn);
  }
}

template <typename scalar_t, typename index_type>
void gru_backward_impl(
    const Tensor& grad_hy,
    const Tensor& workspace,
    const Tensor& grad_input_gates,
    const Tensor& grad_hidden_gates,
    const Tensor& grad_hx) {
  using accscalar_t = at::acc_type_device<scalar_t, kXPU>;

  int64_t numel = grad_hy.numel();
  if (numel == 0)
    return;

  using KernelT = GruCellBackwardFunctor<scalar_t, accscalar_t, index_type>;
  auto max_wg_size = syclMaxWorkGroupSize<KernelT>();
  auto config = rnn_get_launch_config(max_wg_size, numel);
  auto nwg = std::get<0>(config);
  auto local_range = std::get<1>(config);

  auto grad_hyI = getTensorInfo<scalar_t, index_type>(grad_hy);
  auto workspaceI = getTensorInfo<scalar_t, index_type>(workspace);
  auto grad_input_gatesI =
      getTensorInfo<scalar_t, index_type>(grad_input_gates);
  auto grad_hidden_gatesI =
      getTensorInfo<scalar_t, index_type>(grad_hidden_gates);
  auto grad_hxI = getTensorInfo<scalar_t, index_type>(grad_hx);
  index_type hidden_size = grad_hyI.sizes[grad_hyI.dims - 1];

  if (allContiguous(
          {grad_hy, workspace, grad_input_gates, grad_hidden_gates, grad_hx})) {
    collapseDims(
        grad_hyI, workspaceI, grad_input_gatesI, grad_hidden_gatesI, grad_hxI);
    KernelT kfn(
        grad_input_gatesI,
        grad_hidden_gatesI,
        grad_hyI,
        grad_hxI,
        workspaceI,
        hidden_size,
        numel);
    sycl_kernel_submit(
        nwg * local_range, local_range, getCurrentSYCLQueue(), kfn);
  } else {
    KernelT kfn(
        grad_input_gatesI,
        grad_hidden_gatesI,
        grad_hyI,
        grad_hxI,
        workspaceI,
        hidden_size,
        numel);
    sycl_kernel_submit(
        nwg * local_range, local_range, getCurrentSYCLQueue(), kfn);
  }
}

// Note [64-bit index math check elision]
// It's enough to perform the check for 64-bit math on the largest tensor only.
// If 32-bit is enough for it, it will suffice for all other tensors too, and we
// can save some work using this trick.

std::tuple<Tensor, Tensor, Tensor> _thnn_fused_lstm_cell_kernel(
    const Tensor& input_gates,
    const Tensor& hidden_gates,
    const Tensor& cx,
    const std::optional<Tensor>& input_bias_opt,
    const std::optional<Tensor>& hidden_bias_opt) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> input_bias_maybe_owned =
      at::borrow_from_optional_tensor(input_bias_opt);
  const Tensor& input_bias = *input_bias_maybe_owned;
  const Tensor& hidden_bias = hidden_bias_opt.value_or(Tensor());

  checkSizes(
      "_thnn_fused_lstm_cell_xpu",
      {input_gates, "input_gates", 1},
      {hidden_gates, "hidden_gates", 2},
      {input_bias, "input_bias", 3},
      {hidden_bias, "hidden_bias", 4},
      /*factor=*/4,
      {cx, "prev_hidden", 5});

  auto workspace = at::empty_like(input_gates, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto hy = at::empty_like(cx, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto cy = at::empty_like(cx, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input_gates.scalar_type(),
      "_thnn_fused_lstm_cell_xpu",
      [&] {
        if (canUse32BitIndexMath(
                workspace)) { // See Note [64-bit index math check elision]
          lstm_forward_impl<scalar_t, int32_t>(
              input_gates,
              hidden_gates,
              input_bias,
              hidden_bias,
              cx,
              hy,
              cy,
              workspace);
        } else {
          lstm_forward_impl<scalar_t, int64_t>(
              input_gates,
              hidden_gates,
              input_bias,
              hidden_bias,
              cx,
              hy,
              cy,
              workspace);
        }
      });
  return std::make_tuple(std::move(hy), std::move(cy), std::move(workspace));
}

void checkLSTMBackwardSizes(
    const TensorArg& grad_hy,
    const TensorArg& grad_cy,
    const TensorArg& cx,
    const TensorArg& cy,
    const TensorArg& workspace) {
  CheckedFrom c = "fused_lstm_cell_backward";
  const TensorArg& defined_grad = grad_hy->defined() ? grad_hy : grad_cy;
  checkDim(c, defined_grad, 2);
  auto exp_size = defined_grad->sizes();
  if (grad_hy->defined()) {
    checkSize(c, grad_hy, exp_size);
  }
  if (grad_cy->defined()) {
    checkSize(c, grad_cy, exp_size);
  }
  checkSize(c, cx, exp_size);
  checkSize(c, cy, exp_size);
  checkDim(c, workspace, 2);
  checkNumel(c, workspace, exp_size[0] * exp_size[1] * 4);
}

std::tuple<Tensor, Tensor, Tensor> _thnn_fused_lstm_cell_backward_kernel(
    const std::optional<Tensor>& grad_hy_opt,
    const std::optional<Tensor>& grad_cy_opt,
    const Tensor& cx,
    const Tensor& cy,
    const Tensor& workspace,
    bool has_bias) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> grad_hy_maybe_owned =
      at::borrow_from_optional_tensor(grad_hy_opt);
  const Tensor& grad_hy = *grad_hy_maybe_owned;
  const Tensor& grad_cy = grad_cy_opt.value_or(Tensor());

  if (!grad_hy.defined() && !grad_cy.defined()) {
    return std::tuple<Tensor, Tensor, Tensor>();
  }
  checkLSTMBackwardSizes(
      {grad_hy, "grad_hy", 1},
      {grad_cy, "grad_cy", 2},
      {cx, "cx", 3},
      {cy, "cy", 4},
      {workspace, "workspace", 5});

  auto grad_gates = at::empty_like(workspace, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto grad_cx = at::empty_like(cx, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      workspace.scalar_type(),
      "_thnn_fused_lstm_cell_backward_xpu",
      [&] {
        if (canUse32BitIndexMath(
                workspace)) { // See Note [64-bit index math check elision]
          lstm_backward_impl<scalar_t, int32_t>(
              grad_hy, grad_cy, cx, cy, workspace, grad_gates, grad_cx);
        } else {
          lstm_backward_impl<scalar_t, int64_t>(
              grad_hy, grad_cy, cx, cy, workspace, grad_gates, grad_cx);
        }
      });

  auto grad_bias =
      has_bias ? grad_gates.sum(0, /*keepdim=*/false) : at::Tensor{};
  return std::make_tuple(
      std::move(grad_gates), std::move(grad_cx), std::move(grad_bias));
}

static constexpr int64_t GRU_WORKSPACE_MULTIPLIER = 5;

std::tuple<Tensor, Tensor> _thnn_fused_gru_cell_kernel(
    const Tensor& input_gates,
    const Tensor& hidden_gates,
    const Tensor& hx,
    const std::optional<Tensor>& input_bias_opt,
    const std::optional<Tensor>& hidden_bias_opt) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> input_bias_maybe_owned =
      at::borrow_from_optional_tensor(input_bias_opt);
  const Tensor& input_bias = *input_bias_maybe_owned;
  const Tensor& hidden_bias = hidden_bias_opt.value_or(Tensor());

  checkSizes(
      "_thnn_fused_gru_cell_xpu",
      {input_gates, "input_gates", 1},
      {hidden_gates, "hidden_gates", 2},
      {input_bias, "input_bias", 3},
      {hidden_bias, "hidden_bias", 4},
      /*factor=*/3,
      {hx, "prev_hidden", 5});

  auto workspace = at::empty(
      {hx.size(0), hx.size(1) * GRU_WORKSPACE_MULTIPLIER}, hx.options());
  auto hy = at::empty_like(hx, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input_gates.scalar_type(),
      "_thnn_fused_gru_cell_xpu",
      [&] {
        if (canUse32BitIndexMath(
                workspace)) { // See Note [64-bit index math check elision]
          gru_forward_impl<scalar_t, int32_t>(
              input_gates,
              hidden_gates,
              input_bias,
              hidden_bias,
              hx,
              hy,
              workspace);
        } else {
          gru_forward_impl<scalar_t, int64_t>(
              input_gates,
              hidden_gates,
              input_bias,
              hidden_bias,
              hx,
              hy,
              workspace);
        }
      });
  return std::make_tuple(std::move(hy), std::move(workspace));
}

void checkGRUBackwardSizes(
    const TensorArg& grad_hy,
    const TensorArg& workspace) {
  CheckedFrom c = "fused_gru_cell_backward";
  checkDim(c, grad_hy, 2);
  checkSize(
      c,
      workspace,
      {grad_hy->size(0), grad_hy->size(1) * GRU_WORKSPACE_MULTIPLIER});
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor>
_thnn_fused_gru_cell_backward_kernel(
    const Tensor& grad_hy,
    const Tensor& workspace,
    bool has_bias) {
  checkGRUBackwardSizes({grad_hy, "grad_hy", 1}, {workspace, "workspace", 2});

  int64_t hidden_size = workspace.size(1) / GRU_WORKSPACE_MULTIPLIER;
  auto grad_input_gates =
      at::empty({workspace.size(0), hidden_size * 3}, workspace.options());
  auto grad_hidden_gates =
      at::empty({workspace.size(0), hidden_size * 3}, workspace.options());
  auto grad_hx = at::empty_like(grad_hy, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      grad_hy.scalar_type(),
      "_thnn_fused_gru_cell_backward_xpu",
      [&] {
        if (canUse32BitIndexMath(
                workspace)) { // See Note [64-bit index math check elision]
          gru_backward_impl<scalar_t, int32_t>(
              grad_hy, workspace, grad_input_gates, grad_hidden_gates, grad_hx);
        } else {
          gru_backward_impl<scalar_t, int64_t>(
              grad_hy, workspace, grad_input_gates, grad_hidden_gates, grad_hx);
        }
      });

  at::Tensor grad_input_bias, grad_hidden_bias;
  if (has_bias) {
    grad_input_bias = grad_input_gates.sum(0, /*keepdim=*/false);
    grad_hidden_bias = grad_hidden_gates.sum(0, /*keepdim=*/false);
  }

  return std::make_tuple(
      std::move(grad_input_gates),
      std::move(grad_hidden_gates),
      std::move(grad_hx),
      std::move(grad_input_bias),
      std::move(grad_hidden_bias));
}

} // namespace at::native::xpu
