#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <cmath>

#include <ATen/native/quantized/sycl/FakeQuantizeCoreKernels.h>
#include <ATen/native/xpu/sycl/Loops.h>

namespace at::native::xpu {

template <typename scalar_t>
struct FakeQuantizeTensorCachemaskKernelFunctor {
  std::tuple<scalar_t, bool> operator()(scalar_t input_val) const {
    const auto qval = static_cast<int64_t>(
        std::nearbyint(input_val * inv_scale_) + zero_point_);
    return {// fake_quantized value
            (fminf(quant_max_, fmaxf(quant_min_, qval)) - zero_point_) * scale_,
            // mask for grad
            ((quant_min_ <= qval) && (qval <= quant_max_))};
  }
  FakeQuantizeTensorCachemaskKernelFunctor(
      float scale,
      float inv_scale,
      int64_t zero_point,
      int64_t quant_min,
      int64_t quant_max)
      : scale_(scale),
        inv_scale_(inv_scale),
        zero_point_(zero_point),
        quant_min_(quant_min),
        quant_max_(quant_max) {}

 private:
  float scale_;
  float inv_scale_;
  int64_t zero_point_;
  int64_t quant_min_;
  int64_t quant_max_;
};

void fake_quantize_tensor_cachemask_kernel(
    Tensor& output,
    Tensor& mask,
    const Tensor& input,
    float scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max) {
  float inv_scale = 1.0f / scale;
  auto iter = TensorIteratorConfig()
                  .check_all_same_dtype(false)
                  .add_output(output)
                  .add_output(mask)
                  .add_input(input)
                  .build();

  if (at::isReducedFloatingType(input.scalar_type())) {
    AT_DISPATCH_REDUCED_FLOATING_TYPES(
        input.scalar_type(),
        "fake_quantize_tensor_cachemask_kernel_types_xpu",
        [&] {
          auto fn = FakeQuantizeTensorCachemaskKernelFunctor<scalar_t>(
              scale, inv_scale, zero_point, quant_min, quant_max);
          gpu_kernel_multiple_outputs(iter, fn);
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(),
        "fake_quantize_tensor_cachemask_kernel_types_xpu",
        [&] {
          auto fn = FakeQuantizeTensorCachemaskKernelFunctor<scalar_t>(
              scale, inv_scale, zero_point, quant_min, quant_max);
          gpu_kernel_multiple_outputs(iter, fn);
        });
  }
}

template <typename scalar_t>
struct FakeQuantizeTensorCachemaskTensorQparamsFunctor {
  std::tuple<scalar_t, bool> operator()(scalar_t input_val) const {
    if (*fake_quant_on_ == 0) {
      return {input_val, 1};
    }
    float inv_scale = 1.0f / (*scale_ptr_);
    const auto qval = static_cast<int64_t>(
        std::nearbyint(input_val * inv_scale) + (*zp_ptr_));
    return {// fake_quantized value
            (std::min(quant_max_, std::max(quant_min_, qval)) - (*zp_ptr_)) *
                (*scale_ptr_),
            // mask for grad
            ((quant_min_ <= qval) && (qval <= quant_max_))};
  }
  FakeQuantizeTensorCachemaskTensorQparamsFunctor(
      int64_t quant_min,
      int64_t quant_max,
      float* scale_ptr,
      int32_t* zp_ptr,
      int64_t* fake_quant_on)
      : quant_min_(quant_min),
        quant_max_(quant_max),
        scale_ptr_(scale_ptr),
        zp_ptr_(zp_ptr),
        fake_quant_on_(fake_quant_on) {}

 private:
  int64_t quant_min_;
  int64_t quant_max_;
  float* scale_ptr_;
  int32_t* zp_ptr_;
  int64_t* fake_quant_on_;
};

void fake_quantize_tensor_cachemask_tensor_qparams_kernel(
    Tensor& output,
    Tensor& mask,
    const Tensor& input,
    const Tensor& scale,
    const Tensor& zero_point,
    const Tensor& fake_quant_enabled,
    int64_t quant_min,
    int64_t quant_max) {
  float* scale_ptr = scale.data_ptr<float>();
  int32_t* zp_ptr = zero_point.data_ptr<int32_t>();
  int64_t* fake_quant_on = fake_quant_enabled.data_ptr<int64_t>();
  auto iter = TensorIteratorConfig()
                  .check_all_same_dtype(false)
                  .add_output(output)
                  .add_output(mask)
                  .add_input(input)
                  .build();

  if (at::isReducedFloatingType(input.scalar_type())) {
    AT_DISPATCH_REDUCED_FLOATING_TYPES(
        input.scalar_type(),
        "fake_quantize_tensor_cachemask_kernel_types_xpu",
        [&] {
          auto fn = FakeQuantizeTensorCachemaskTensorQparamsFunctor<scalar_t>(
              quant_min, quant_max, scale_ptr, zp_ptr, fake_quant_on);
          gpu_kernel_multiple_outputs(iter, fn);
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(),
        "fake_quantize_tensor_cachemask_kernel_types_xpu",
        [&] {
          auto fn = FakeQuantizeTensorCachemaskTensorQparamsFunctor<scalar_t>(
              quant_min, quant_max, scale_ptr, zp_ptr, fake_quant_on);
          gpu_kernel_multiple_outputs(iter, fn);
        });
  }
}

struct FakeQuantizeGradLearnableTensorFunctor {
  std::tuple<float, float, float> operator()(float XInput, float dYInput)
      const {
    float dXOutput, dZeroPointOutput, dScaleOutput;
    int64_t xq = std::nearbyint(XInput * inv_scale_) + zero_point_;
    dXOutput = dYInput * (xq >= quant_min_ && xq <= quant_max_);
    float xfq = static_cast<float>(
        (std::max(std::min(xq, quant_max_), quant_min_) - zero_point_) *
        scale_);
    if (xq < quant_min_ || xq > quant_max_) {
      dZeroPointOutput = (dYInput) * (-1) * scale_ * grad_factor_;
      dScaleOutput = ((xq < quant_min_) ? (dYInput * dscale_small_)
                                        : (dYInput * dscale_big_)) *
          grad_factor_;
    } else {
      dZeroPointOutput = 0;
      dScaleOutput = (dYInput) * (xfq - (XInput)) * inv_scale_ * grad_factor_;
    }
    return {dXOutput, dScaleOutput, dZeroPointOutput};
  }

  FakeQuantizeGradLearnableTensorFunctor(
      float inv_scale,
      int64_t zero_point,
      int64_t quant_min,
      int64_t quant_max,
      float scale,
      float grad_factor,
      float dscale_small,
      float dscale_big)
      : inv_scale_(inv_scale),
        zero_point_(zero_point),
        quant_min_(quant_min),
        quant_max_(quant_max),
        scale_(scale),
        grad_factor_(grad_factor),
        dscale_small_(dscale_small),
        dscale_big_(dscale_big) {}

 private:
  float inv_scale_;
  int64_t zero_point_;
  int64_t quant_min_;
  int64_t quant_max_;
  float scale_;
  float grad_factor_;
  float dscale_small_;
  float dscale_big_;
};

void _fake_quantize_grad_learnable_tensor_kernel(
    TensorIterator& iter,
    float scale,
    float inv_scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max,
    float grad_factor) {
  float dscale_small = quant_min - zero_point;
  float dscale_big = quant_max - zero_point;
  FakeQuantizeGradLearnableTensorFunctor f(
      inv_scale,
      zero_point,
      quant_min,
      quant_max,
      scale,
      grad_factor,
      dscale_small,
      dscale_big);
  gpu_kernel_multiple_outputs(iter, f);
}

template <typename scalar_t>
inline long int lrint_impl(scalar_t num) {
  return static_cast<long int>(std::round(num));
}

template <typename SelfType, typename scalar_t>
struct FakeQuantPerChannelCachemaskHelperAFunctor {
  bool operator()(
      const SelfType input_val,
      const float scale,
      const scalar_t zero_point) const {
    const float inv_scale = 1.0f / scale;
    const auto qval = lrint_impl(input_val * inv_scale + zero_point);
    return ((quant_min_ <= qval) && (qval <= quant_max_));
  }
  FakeQuantPerChannelCachemaskHelperAFunctor(
      const int64_t quant_min,
      const int64_t quant_max)
      : quant_min_(quant_min), quant_max_(quant_max) {}

 private:
  const int64_t quant_min_;
  const int64_t quant_max_;
};

template <typename SelfType, typename scalar_t>
struct FakeQuantPerChannelCachemaskHelperBFunctor {
  SelfType operator()(
      const SelfType input_val,
      const float scale,
      const scalar_t zero_point) const {
    const float inv_scale = 1.0f / scale;
    const auto qval = lrint_impl(input_val * inv_scale + zero_point);
    const auto bounded_qval = fminf(quant_max_, fmaxf(quant_min_, qval));
    return (bounded_qval - zero_point) * scale;
  }
  FakeQuantPerChannelCachemaskHelperBFunctor(
      const int64_t quant_min,
      const int64_t quant_max)
      : quant_min_(quant_min), quant_max_(quant_max) {}

 private:
  const int64_t quant_min_;
  const int64_t quant_max_;
};

template <typename SelfType>
struct FakeQuantPerChannelCachemaskHelperCFunctor {
  bool operator()(
      const SelfType input_val,
      const float scale,
      const int64_t zero_point) const {
    const float inv_scale = 1.0f / scale;
    const auto qval =
        static_cast<int64_t>(std::nearbyint(input_val * inv_scale)) +
        zero_point;
    return ((quant_min_ <= qval) && (qval <= quant_max_));
  }
  FakeQuantPerChannelCachemaskHelperCFunctor(
      const int64_t quant_min,
      const int64_t quant_max)
      : quant_min_(quant_min), quant_max_(quant_max) {}

 private:
  const int64_t quant_min_;
  const int64_t quant_max_;
};

template <typename SelfType>
struct FakeQuantPerChannelCachemaskHelperDFunctor {
  SelfType operator()(
      const SelfType input_val,
      const float scale,
      const int64_t zero_point) const {
    const float inv_scale = 1.0f / scale;
    const auto qval =
        static_cast<int64_t>(std::nearbyint(input_val * inv_scale)) +
        zero_point;
    const auto bounded_qval = std::min(quant_max_, std::max(quant_min_, qval));
    return (bounded_qval - zero_point) * scale;
  }
  FakeQuantPerChannelCachemaskHelperDFunctor(
      const int64_t quant_min,
      const int64_t quant_max)
      : quant_min_(quant_min), quant_max_(quant_max) {}

 private:
  const int64_t quant_min_;
  const int64_t quant_max_;
};

template <typename SelfType>
void _fake_quant_per_channel_cachemask_xpu_helper(
    TensorIterator& iter,
    TensorIterator& iter_mask,
    const int64_t quant_min,
    const int64_t quant_max) {
  const auto& zero_point_dtype = iter.input_dtype(2);
  if (at::isFloatingType(zero_point_dtype)) {
    // When zero_point is float, quantize mirroring affine quantizer equation
    // Xq = Round(Xf * inv_scale + zero_point)
    // where zero_point is in float.
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        zero_point_dtype,
        "fake_quantize_channel_cachemask_xpu_mask_type_handling",
        [&] {
          // write mask
          auto fna =
              FakeQuantPerChannelCachemaskHelperAFunctor<SelfType, scalar_t>(
                  quant_min, quant_max);
          gpu_kernel(iter_mask, fna);

          // write fake_quant
          auto fnb =
              FakeQuantPerChannelCachemaskHelperBFunctor<SelfType, scalar_t>(
                  quant_min, quant_max);
          gpu_kernel(iter, fnb);
        });
  } else {
    // write mask
    auto fnc = FakeQuantPerChannelCachemaskHelperCFunctor<SelfType>(
        quant_min, quant_max);
    gpu_kernel(iter_mask, fnc);

    // write fake_quant
    auto fnd = FakeQuantPerChannelCachemaskHelperDFunctor<SelfType>(
        quant_min, quant_max);
    gpu_kernel(iter, fnd);
  }
}

void fake_quant_per_channel_cachemask_kernel(
    TensorIterator& iter,
    TensorIterator& iter_mask,
    int64_t quant_min,
    int64_t quant_max) {
  if (at::isReducedFloatingType(iter.dtype())) {
    AT_DISPATCH_REDUCED_FLOATING_TYPES(
        iter.dtype(), "fake_quantize_channel_cachemask_xpu_type_handling", [&] {
          _fake_quant_per_channel_cachemask_xpu_helper<scalar_t>(
              iter, iter_mask, quant_min, quant_max);
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        iter.dtype(), "fake_quantize_channel_cachemask_xpu_type_handling", [&] {
          _fake_quant_per_channel_cachemask_xpu_helper<scalar_t>(
              iter, iter_mask, quant_min, quant_max);
        });
  }
}

struct FakeQuantizeGradLearnableChannelFunctor {
  std::tuple<float, float, float> operator()(
      float x_input,
      float dy_input,
      float scale_input,
      float zero_point_input) const {
    float dx_output, dscale_output, dzero_point_output;
    float inv_scale = 1.0f / scale_input;
    float dscale_small = quant_min_ - zero_point_input;
    float dscale_big = quant_max_ - zero_point_input;
    // Calculate gradients for X.
    int64_t xqi = std::nearbyint(x_input * inv_scale) +
        static_cast<int64_t>(zero_point_input);
    dx_output = dy_input * (xqi >= quant_min_ && xqi <= quant_max_);
    // Calculate gradients for scale and zero point.
    float xfqi = static_cast<float>(
        (std::max(std::min(xqi, quant_max_), quant_min_) - zero_point_input) *
        scale_input);
    if (xqi < quant_min_ || xqi > quant_max_) {
      dzero_point_output = dy_input * (-1) * scale_input * grad_factor_;
      dscale_output = ((xqi < quant_min_) ? (dy_input * dscale_small)
                                          : (dy_input * dscale_big)) *
          grad_factor_;
    } else {
      dzero_point_output = 0;
      dscale_output = dy_input * (xfqi - x_input) * inv_scale * grad_factor_;
    }
    return {dx_output, dscale_output, dzero_point_output};
  }
  FakeQuantizeGradLearnableChannelFunctor(
      int64_t quant_min,
      int64_t quant_max,
      float grad_factor)
      : quant_min_(quant_min),
        quant_max_(quant_max),
        grad_factor_(grad_factor) {}

 private:
  int64_t quant_min_;
  int64_t quant_max_;
  float grad_factor_;
};

void _fake_quantize_grad_learnable_channel_kernel(
    TensorIterator& iter,
    int64_t quant_min,
    int64_t quant_max,
    float grad_factor) {
  FakeQuantizeGradLearnableChannelFunctor f(quant_min, quant_max, grad_factor);
  gpu_kernel_multiple_outputs(iter, f);
}

} // namespace at::native::xpu
