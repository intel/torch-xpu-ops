#pragma clang diagnostic push
#pragma GCC diagnostic push
// Avoid SYCL compiler return-type error
#pragma clang diagnostic ignored "-Wreturn-type"
#pragma GCC diagnostic ignored "-Wreturn-type"

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/ceil_div.h>
#include <ATen/native/Pool.h>
#include <ATen/native/utils/ParamUtils.h>

#include <ATen/native/xpu/sycl/Atomics.h>
#include <ATen/native/xpu/sycl/BatchKernel.h>
#include <ATen/native/xpu/sycl/DilatedMaxPool3d.h>
#include <ATen/native/xpu/sycl/NumericLimits.h>
#include <comm/Runtime.h>
#include <comm/SYCLHelpers.h>

namespace at::native::xpu {

template <typename scalar_t, bool channels_last_>
struct MaxPool3dKerenlFunctor {
  void operator()(sycl::nd_item<1> item) const {
    auto outputIndex = item.get_global_id(0);
    if (outputIndex < OutputSize_) {
      int64_t batch = 0;
      int64_t channel = 0;
      int64_t oTime = 0;
      int64_t oRow = 0;
      int64_t oColumn = 0;
      // used only for channels-first indexing
      int64_t slice = 0;
      batch = outputIndex / out_batch_stride_;
      if constexpr (!channels_last_) {
        // order: batch, channel, time
        oColumn = outputIndex % owidth_;
        oRow = outputIndex / owidth_ % oheight_;
        oTime = outputIndex / out_cf_d_stride_ % otime_;
        channel = outputIndex / out_cf_c_stride_ % features_;
        slice = outputIndex / out_cf_c_stride_;
      } else {
        channel = outputIndex % features_;
        oColumn = outputIndex / features_ % owidth_;
        oRow = outputIndex / out_cl_h_stride_ % oheight_;
        oTime = outputIndex / out_cl_d_stride_ % otime_;
      }

      int tStart = oTime * dT_ - pT_;
      int hStart = oRow * dH_ - pH_;
      int wStart = oColumn * dW_ - pW_;
      int tEnd = std::min(tStart + (kT_ - 1) * dilationT_ + 1, itime_);
      int hEnd = std::min(hStart + (kH_ - 1) * dilationH_ + 1, iheight_);
      int wEnd = std::min(wStart + (kW_ - 1) * dilationW_ + 1, iwidth_);

      while (tStart < 0)
        tStart += dilationT_;
      while (hStart < 0)
        hStart += dilationH_;
      while (wStart < 0)
        wStart += dilationW_;

      // maxIndex remains in "channels-first"/contiguous
      int64_t maxIndex;
      int64_t ioffset;

      if constexpr (!channels_last_) {
        ioffset = (int64_t)slice * in_cf_c_stride_;
      } else {
        ioffset = ((int64_t)batch * in_batch_stride_) + channel;
      }

      scalar_t max = at::numeric_limits<scalar_t>::lower_bound();

      for (int t = tStart; t < tEnd; t += dilationT_) {
        for (int h = hStart; h < hEnd; h += dilationH_) {
          for (int w = wStart; w < wEnd; w += dilationW_) {
            scalar_t val;
            int index = t * in_hw_stride_ + h * iwidth_ + w;
            if constexpr (!channels_last_) {
              val = inputData_[ioffset + index];
            } else {
              int64_t index_channels_last = index * features_;
              val = inputData_[ioffset + index_channels_last];
            }

            if ((max < val) || at::_isnan(val)) {
              max = val;
              maxIndex = index;
            }
          }
        }
      }

      int64_t out_index;
      if constexpr (!channels_last_) {
        out_index = (int64_t)slice * out_cf_c_stride_ +
            oTime * out_cf_d_stride_ + oRow * owidth_ + oColumn;
      } else {
        out_index = (int64_t)batch * out_batch_stride_ +
            oTime * out_cl_d_stride_ + oRow * out_cl_h_stride_ +
            oColumn * features_ + channel;
      }
      outputData_[out_index] = max;
      indicesData_[out_index] = maxIndex;
    }
  }
  MaxPool3dKerenlFunctor(
      const scalar_t* inputData,
      scalar_t* outputData,
      int64_t* indicesData,
      int features,
      int itime,
      int iheight,
      int iwidth,
      int obatch,
      int otime,
      int oheight,
      int owidth,
      int kT,
      int kH,
      int kW,
      int dT,
      int dH,
      int dW,
      int pT,
      int pH,
      int pW,
      int dilationT,
      int dilationH,
      int dilationW,
      int64_t OutputSize,
      int out_cf_d_stride,
      int out_cf_c_stride,
      int in_cf_d_stride,
      int in_cf_c_stride,
      int out_cl_h_stride,
      int out_cl_d_stride,
      int in_cl_h_stride,
      int in_cl_d_stride,
      int in_batch_stride,
      int out_batch_stride,
      int in_hw_stride)
      : inputData_(inputData),
        outputData_(outputData),
        indicesData_(indicesData),
        features_(features),
        itime_(itime),
        iheight_(iheight),
        iwidth_(iwidth),
        obatch_(obatch),
        otime_(otime),
        oheight_(oheight),
        owidth_(owidth),
        kT_(kT),
        kH_(kH),
        kW_(kW),
        dT_(dT),
        dH_(dH),
        dW_(dW),
        pT_(pT),
        pH_(pH),
        pW_(pW),
        dilationT_(dilationT),
        dilationH_(dilationH),
        dilationW_(dilationW),
        OutputSize_(OutputSize),
        out_cf_d_stride_(out_cf_d_stride),
        out_cf_c_stride_(out_cf_c_stride),
        in_cf_d_stride_(in_cf_d_stride),
        in_cf_c_stride_(in_cf_c_stride),
        out_cl_h_stride_(out_cl_h_stride),
        out_cl_d_stride_(out_cl_d_stride),
        in_cl_h_stride_(in_cl_h_stride),
        in_cl_d_stride_(in_cl_d_stride),
        in_batch_stride_(in_batch_stride),
        out_batch_stride_(out_batch_stride),
        in_hw_stride_(in_hw_stride) {}

 private:
  const scalar_t* inputData_;
  scalar_t* outputData_;
  int64_t* indicesData_;
  int features_;
  int itime_;
  int iheight_;
  int iwidth_;
  int obatch_;
  int otime_;
  int oheight_;
  int owidth_;
  int kT_;
  int kH_;
  int kW_;
  int dT_;
  int dH_;
  int dW_;
  int pT_;
  int pH_;
  int pW_;
  int dilationT_;
  int dilationH_;
  int dilationW_;
  int64_t OutputSize_;
  int out_cf_d_stride_;
  int out_cf_c_stride_;
  int in_cf_d_stride_;
  int in_cf_c_stride_;
  int out_cl_h_stride_;
  int out_cl_d_stride_;
  int in_cl_h_stride_;
  int in_cl_d_stride_;
  int in_batch_stride_;
  int out_batch_stride_;
  int in_hw_stride_;
};

template <typename scalar_t, bool channels_last>
void max_pool3d_with_indices_out_template(
    const scalar_t* inputData,
    scalar_t* outputData,
    int64_t* indicesData,
    int features,
    int itime,
    int iheight,
    int iwidth,
    int obatch,
    int otime,
    int oheight,
    int owidth,
    int kT,
    int kH,
    int kW,
    int dT,
    int dH,
    int dW,
    int pT,
    int pH,
    int pW,
    int dilationT,
    int dilationH,
    int dilationW) {
  int64_t OutputSize = obatch * features * otime * oheight * owidth;

  int out_cf_d_stride, out_cf_c_stride, in_cf_d_stride, in_cf_c_stride;
  int out_cl_h_stride, out_cl_d_stride, in_cl_h_stride, in_cl_d_stride;
  if constexpr (!channels_last) {
    out_cf_d_stride = owidth * oheight;
    out_cf_c_stride = otime * out_cf_d_stride;
    in_cf_d_stride = iwidth * iheight;
    in_cf_c_stride = itime * in_cf_d_stride;
  } else {
    out_cl_h_stride = owidth * features;
    out_cl_d_stride = oheight * out_cl_h_stride;
  }
  auto in_batch_stride = itime * iheight * iwidth * features;
  auto out_batch_stride = otime * oheight * owidth * features;
  auto in_hw_stride = iwidth * iheight;
  MaxPool3dKerenlFunctor<scalar_t, channels_last> kfn(
      inputData,
      outputData,
      indicesData,
      features,
      itime,
      iheight,
      iwidth,
      obatch,
      otime,
      oheight,
      owidth,
      kT,
      kH,
      kW,
      dT,
      dH,
      dW,
      pT,
      pH,
      pW,
      dilationT,
      dilationH,
      dilationW,
      OutputSize,
      out_cf_d_stride,
      out_cf_c_stride,
      in_cf_d_stride,
      in_cf_c_stride,
      out_cl_h_stride,
      out_cl_d_stride,
      in_cl_h_stride,
      in_cl_d_stride,
      in_batch_stride,
      out_batch_stride,
      in_hw_stride);
  int work_group_size = syclMaxWorkItemsPerSubSlice();
  auto global_range =
      (OutputSize + work_group_size - 1) / work_group_size * work_group_size;

  auto& queue = getCurrentSYCLQueue();

  sycl_kernel_submit(global_range, work_group_size, queue, kfn);
}

template <typename scalar_t, bool channels_last>
struct MaxPool3dBackwardKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    auto outputIndex = item.get_global_id(0);
    if (outputIndex < gradOutputSize_) {
      int batch = outputIndex / out_nbatch_stride_;
      if constexpr (channels_last) {
        int channel = outputIndex % features_;
        int64_t index = indicesData_[outputIndex];
        int64_t gradIn_offset =
            batch * in_nbatch_stride_ + channel + index * features_;
        atomicAdd(
            (sycl_global_ptr<scalar_t>)&gradInputData_[gradIn_offset],
            gradOutputData_[outputIndex]);
      } else {
        int channel = outputIndex / out_cf_channel_stride_ % features_;
        int64_t index = indicesData_[outputIndex];
        int64_t gradIn_offset =
            batch * in_nbatch_stride_ + channel * in_cf_channel_stride_ + index;
        atomicAdd(
            (sycl_global_ptr<scalar_t>)&gradInputData_[gradIn_offset],
            gradOutputData_[outputIndex]);
      }
    }
  }
  MaxPool3dBackwardKernelFunctor(
      scalar_t* gradInputData,
      const scalar_t* gradOutputData,
      const int64_t* indicesData,
      int features,
      int64_t gradOutputSize,
      int out_cf_channel_stride,
      int in_cf_channel_stride,
      int out_nbatch_stride,
      int in_nbatch_stride)
      : gradInputData_(gradInputData),
        gradOutputData_(gradOutputData),
        indicesData_(indicesData),
        features_(features),
        gradOutputSize_(gradOutputSize),
        out_cf_channel_stride_(out_cf_channel_stride),
        in_cf_channel_stride_(in_cf_channel_stride),
        out_nbatch_stride_(out_nbatch_stride),
        in_nbatch_stride_(in_nbatch_stride) {}

 private:
  scalar_t* gradInputData_;
  const scalar_t* gradOutputData_;
  const int64_t* indicesData_;
  int features_;
  int64_t gradOutputSize_;
  int out_cf_channel_stride_;
  int in_cf_channel_stride_;
  int out_nbatch_stride_;
  int in_nbatch_stride_;
};

template <typename scalar_t, bool channels_last>
void max_pool3d_with_indices_backward_template(
    scalar_t* gradInputData,
    const scalar_t* gradOutputData,
    const int64_t* indicesData,
    int features,
    int itime,
    int iheight,
    int iwidth,
    int obatch,
    int otime,
    int oheight,
    int owidth) {
  int64_t gradOutputSize = obatch * features * otime * oheight * owidth;

  auto out_cf_channel_stride = otime * oheight * owidth;
  auto in_cf_channel_stride = itime * iheight * iwidth;
  auto out_nbatch_stride = features * out_cf_channel_stride;
  auto in_nbatch_stride = features * in_cf_channel_stride;
  MaxPool3dBackwardKernelFunctor<scalar_t, channels_last> kfn(
      gradInputData,
      gradOutputData,
      indicesData,
      features,
      gradOutputSize,
      out_cf_channel_stride,
      in_cf_channel_stride,
      out_nbatch_stride,
      in_nbatch_stride);

  int work_group_size = syclMaxWorkItemsPerSubSlice();

  auto global_range =
      ((gradOutputSize - 1) / work_group_size + 1) * work_group_size;

  auto& queue = getCurrentSYCLQueue();

  sycl_kernel_submit(global_range, work_group_size, queue, kfn);
}

void max_pool3d_with_indices_kernel(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode,
    Tensor& output,
    Tensor& indices) {
  NoNamesGuard guard;

  TensorArg output_arg{output, "output", 1};
  TensorArg indices_arg{indices, "indices", 2};
  TensorArg input_arg{input, "input", 3};

  checkAllSameGPU(__func__, {output_arg, indices_arg, input_arg});

  TORCH_CHECK(
      kernel_size.size() == 1 || kernel_size.size() == 3,
      "max_pool3d: kernel_size must either be a single int, or a tuple of three ints")
  const int kT = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kH = kernel_size.size() == 1
      ? kT
      : safe_downcast<int, int64_t>(kernel_size[1]);
  const int kW = kernel_size.size() == 1
      ? kT
      : safe_downcast<int, int64_t>(kernel_size[2]);

  TORCH_CHECK(
      stride.size() == 0 || stride.size() == 1 || stride.size() == 3,
      "max_pool3d: stride must either be omitted, a single int, or a tuple of three ints")
  const int dT = stride.empty() ? kT : safe_downcast<int, int64_t>(stride[0]);
  const int dH = stride.empty() ? kH
      : stride.size() == 1      ? dT
                                : safe_downcast<int, int64_t>(stride[1]);
  const int dW = stride.empty() ? kW
      : stride.size() == 1      ? dT
                                : safe_downcast<int, int64_t>(stride[2]);

  TORCH_CHECK(
      padding.size() == 1 || padding.size() == 3,
      "max_pool3d: padding must either be a single int, or a tuple of three ints");
  const int pT = safe_downcast<int, int64_t>(padding[0]);
  const int pH =
      padding.size() == 1 ? pT : safe_downcast<int, int64_t>(padding[1]);
  const int pW =
      padding.size() == 1 ? pT : safe_downcast<int, int64_t>(padding[2]);

  TORCH_CHECK(
      dilation.size() == 1 || dilation.size() == 3,
      "max_pool3d: dilation must be either a single int, or a tuple of three ints");
  const int dilationT = safe_downcast<int, int64_t>(dilation[0]);
  const int dilationH = dilation.size() == 1
      ? dilationT
      : safe_downcast<int, int64_t>(dilation[1]);
  const int dilationW = dilation.size() == 1
      ? dilationT
      : safe_downcast<int, int64_t>(dilation[2]);

  const int64_t nbatch = input.ndimension() == 5 ? input.size(-5) : 1;
  const int64_t nslices = input.size(-4);
  const int64_t itime = input.size(-3);
  const int64_t iheight = input.size(-2);
  const int64_t iwidth = input.size(-1);

  const int64_t otime =
      pooling_output_shape<int64_t>(itime, kT, pT, dT, dilationT, ceil_mode);
  const int64_t oheight =
      pooling_output_shape<int64_t>(iheight, kH, pH, dH, dilationH, ceil_mode);
  const int64_t owidth =
      pooling_output_shape<int64_t>(iwidth, kW, pW, dW, dilationW, ceil_mode);

  pool3d_shape_check(
      input,
      nslices,
      kT,
      kH,
      kW,
      dT,
      dH,
      dW,
      pT,
      pH,
      pW,
      dilationT,
      dilationH,
      dilationW,
      itime,
      iheight,
      iwidth,
      otime,
      oheight,
      owidth,
      "max_pool3d_with_indices_kernel()");

  bool channels_last = input.ndimension() == 5 &&
      input.suggest_memory_format() == at::MemoryFormat::ChannelsLast3d;
  Tensor _input = input;
  if (input.ndimension() == 4) {
    Tensor input_channels_last_check = input.unsqueeze(0);
    channels_last = (!input_channels_last_check.is_contiguous()) &&
        input_channels_last_check.is_contiguous(
            at::MemoryFormat::ChannelsLast3d);
    if (!channels_last) {
      output.resize_({nslices, otime, oheight, owidth});
      indices.resize_({nslices, otime, oheight, owidth});
    } else {
      _input = input_channels_last_check;
      output.resize_(
          {1, nslices, otime, oheight, owidth},
          at::MemoryFormat::ChannelsLast3d);
      indices.resize_(
          {1, nslices, otime, oheight, owidth},
          at::MemoryFormat::ChannelsLast3d);
      output = output.squeeze(0);
      indices = indices.squeeze(0);
    }
  } else {
    if (!channels_last) {
      output.resize_({nbatch, nslices, otime, oheight, owidth});
      indices.resize_({nbatch, nslices, otime, oheight, owidth});
    } else {
      output.resize_(
          {nbatch, nslices, otime, oheight, owidth},
          at::MemoryFormat::ChannelsLast3d);
      indices.resize_(
          {nbatch, nslices, otime, oheight, owidth},
          at::MemoryFormat::ChannelsLast3d);
    }
  }

  if (input.numel() == 0) {
    return;
  }

  Tensor work_input;
  Tensor work_output = output;
  if (!channels_last) {
    work_input = input.contiguous();
  } else {
    work_input = _input.contiguous(at::MemoryFormat::ChannelsLast3d);
  }
  Tensor work_indices = indices;
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf, kBFloat16, input.scalar_type(), "max_pool3d_xpu", [&] {
        const scalar_t* input_data = work_input.const_data_ptr<scalar_t>();
        if (!channels_last) {
          max_pool3d_with_indices_out_template<scalar_t, false>(
              input_data,
              work_output.mutable_data_ptr<scalar_t>(),
              work_indices.mutable_data_ptr<int64_t>(),
              nslices, // features
              itime,
              iheight,
              iwidth,
              nbatch,
              otime,
              oheight,
              owidth,
              kT,
              kH,
              kW,
              dT,
              dH,
              dW,
              pT,
              pH,
              pW,
              dilationT,
              dilationH,
              dilationW);
        } else {
          max_pool3d_with_indices_out_template<scalar_t, true>(
              input_data,
              work_output.mutable_data_ptr<scalar_t>(),
              work_indices.mutable_data_ptr<int64_t>(),
              nslices, // features
              itime,
              iheight,
              iwidth,
              nbatch,
              otime,
              oheight,
              owidth,
              kT,
              kH,
              kW,
              dT,
              dH,
              dW,
              pT,
              pH,
              pW,
              dilationT,
              dilationH,
              dilationW);
        }
      });
}
void max_pool3d_with_indices_backward_kernel(
    Tensor& gradInput,
    const Tensor& gradOutput,
    const Tensor& input,
    const Tensor& indices,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  TensorArg gradInput_arg{gradInput, "gradInput", 1};
  TensorArg gradOutput_arg{gradOutput, "gradOutput", 2};
  TensorArg input_arg{input, "input", 3};
  TensorArg indices_arg{indices, "indices", 4};

  checkAllSameGPU(
      __func__, {gradInput_arg, gradOutput_arg, input_arg, indices_arg});

  TORCH_CHECK(
      kernel_size.size() == 1 || kernel_size.size() == 3,
      "max_pool3d: kernel_size must either be a single int, or a tuple of three ints")
  const int kT = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kH = kernel_size.size() == 1
      ? kT
      : safe_downcast<int, int64_t>(kernel_size[1]);
  const int kW = kernel_size.size() == 1
      ? kT
      : safe_downcast<int, int64_t>(kernel_size[2]);

  TORCH_CHECK(
      stride.size() == 0 || stride.size() == 1 || stride.size() == 3,
      "max_pool3d: stride must either be omitted, a single int, or a tuple of three ints")
  const int dT = stride.empty() ? kT : safe_downcast<int, int64_t>(stride[0]);
  const int dH = stride.empty() ? kH
      : stride.size() == 1      ? dT
                                : safe_downcast<int, int64_t>(stride[1]);
  const int dW = stride.empty() ? kW
      : stride.size() == 1      ? dT
                                : safe_downcast<int, int64_t>(stride[2]);

  TORCH_CHECK(
      padding.size() == 1 || padding.size() == 3,
      "max_pool3d: padding must either be a single int, or a tuple of three ints");
  const int pT = safe_downcast<int, int64_t>(padding[0]);
  const int pH =
      padding.size() == 1 ? pT : safe_downcast<int, int64_t>(padding[1]);
  const int pW =
      padding.size() == 1 ? pT : safe_downcast<int, int64_t>(padding[2]);

  TORCH_CHECK(
      dilation.size() == 1 || dilation.size() == 3,
      "max_pool3d: dilation must be either a single int, or a tuple of three ints");
  const int dilationT = safe_downcast<int, int64_t>(dilation[0]);
  const int dilationH = dilation.size() == 1
      ? dilationT
      : safe_downcast<int, int64_t>(dilation[1]);
  const int dilationW = dilation.size() == 1
      ? dilationT
      : safe_downcast<int, int64_t>(dilation[2]);

  TORCH_CHECK(
      (input.ndimension() == 4 || input.ndimension() == 5),
      "max_pool3d_with_indices_backward_kernel(): ",
      "Expected 4D or 5D input tensor, but got ",
      input.sizes());

  TORCH_CHECK(
      (gradOutput.ndimension() == 4 || gradOutput.ndimension() == 5),
      "max_pool3d_with_indices_backward_kernel(): ",
      "Expected 4D or 5D gradOutput tensor, but got ",
      gradOutput.sizes());

  // Resize and initialize result tensor.
  bool channels_last = input.ndimension() == 5 &&
      input.suggest_memory_format() == at::MemoryFormat::ChannelsLast3d;
  Tensor _input = input;
  if (input.ndimension() == 4) {
    Tensor input_channels_last_check = input.unsqueeze(0);
    channels_last = (!input_channels_last_check.is_contiguous()) &&
        input_channels_last_check.is_contiguous(
            at::MemoryFormat::ChannelsLast3d);
    if (channels_last) {
      _input = input_channels_last_check;
    }
  }
  if (!channels_last) {
    gradInput.resize_as_(input);
  } else {
    gradInput.resize_as_(_input, at::MemoryFormat::ChannelsLast3d);
  }
  gradInput.zero_();

  const int64_t nbatch = input.ndimension() == 5 ? input.size(-5) : 1;
  const int64_t nslices = input.size(-4);

  const int64_t otime = gradOutput.size(-3);
  const int64_t oheight = gradOutput.size(-2);
  const int64_t owidth = gradOutput.size(-1);

  const int64_t itime = gradInput.size(-3);
  const int64_t iheight = gradInput.size(-2);
  const int64_t iwidth = gradInput.size(-1);
  max_pool3d_backward_shape_check(
      input,
      gradOutput,
      indices,
      nslices,
      kT,
      kH,
      kW,
      dT,
      dH,
      dW,
      pT,
      pH,
      pW,
      dilationT,
      dilationH,
      dilationW,
      itime,
      iheight,
      iwidth,
      otime,
      oheight,
      owidth,
      "max_pool3d_with_indices_backward_kernel()");
  if (gradOutput.numel() == 0) {
    return;
  }

  Tensor work_grad_input = gradInput;
  Tensor work_grad_output;
  Tensor work_indices;
  if (!channels_last) {
    work_grad_output = gradOutput.contiguous();
    work_indices = indices.contiguous();
  } else {
    if (input.ndimension() == 4) {
      work_grad_output =
          gradOutput.unsqueeze(0).contiguous(at::MemoryFormat::ChannelsLast3d);
      work_indices =
          indices.unsqueeze(0).contiguous(at::MemoryFormat::ChannelsLast3d);
    } else {
      work_grad_output =
          gradOutput.contiguous(at::MemoryFormat::ChannelsLast3d);
      work_indices = indices.contiguous(at::MemoryFormat::ChannelsLast3d);
    }
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf,
      kBFloat16,
      input.scalar_type(),
      "max_pool3d_with_indices_backward_out_xpu",
      [&] {
        scalar_t* grad_input_data =
            work_grad_input.mutable_data_ptr<scalar_t>();
        if (!channels_last) {
          max_pool3d_with_indices_backward_template<scalar_t, false>(
              grad_input_data,
              work_grad_output.const_data_ptr<scalar_t>(),
              work_indices.const_data_ptr<int64_t>(),
              nslices,
              itime,
              iheight,
              iwidth,
              nbatch,
              otime,
              oheight,
              owidth);
        } else {
          max_pool3d_with_indices_backward_template<scalar_t, true>(
              grad_input_data,
              work_grad_output.const_data_ptr<scalar_t>(),
              work_indices.const_data_ptr<int64_t>(),
              nslices,
              itime,
              iheight,
              iwidth,
              nbatch,
              otime,
              oheight,
              owidth);
        }
      });
}

} // namespace at::native::xpu

#pragma GCC diagnostic pop
#pragma clang diagnostic pop
