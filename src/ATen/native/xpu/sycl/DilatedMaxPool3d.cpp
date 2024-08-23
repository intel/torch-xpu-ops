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
#include <ATen/native/xpu/sycl/NumericLimits.h>
#include <comm/Runtime.h>
#include <comm/SYCLHelpers.h>

namespace at::native::xpu {

inline int min(int a, int b) {
  return a <= b ? a : b;
}

template <typename scalar_t>
struct MaxPool3dWithIndicesOutFrameImplKernelFunctor {
  void operator()(sycl::nd_item<3> item) const {
    int oColumn = item.get_global_id()[2];
    int oRow = item.get_global_id()[1];
    int oFrame = 0;
    // used only for channels-first
    int64_t slice = 0;
    // used only for channels-last
    int batch = 0;
    int channel = 0;
    if (!channels_last_) {
      // order: batch, channel, time
      oFrame = (item.get_global_id()[0] + offsetZ_) % otime_;
      slice = (item.get_global_id()[0] + offsetZ_) / otime_;
    } else {
      // order: batch, time, channel
      channel = item.get_global_id()[0] % features_;
      slice = item.get_global_id()[0] / features_;
      batch = slice / otime_;
      oFrame = slice % otime_;
    }
    auto inputDataSlice = inputData_;
    if (oRow < oheight_ && oColumn < owidth_ && oFrame < otime_ &&
        channel < features_ && batch < obatch_) {
      int tStart = oFrame * dT_ - pT_;
      int hStart = oRow * dH_ - pH_;
      int wStart = oColumn * dW_ - pW_;
      int tEnd = min(tStart + (kT_ - 1) * dilationT_ + 1, itime_);
      int hEnd = min(hStart + (kH_ - 1) * dilationH_ + 1, iheight_);
      int wEnd = min(wStart + (kW_ - 1) * dilationW_ + 1, iwidth_);

      while (tStart < 0)
        tStart += dilationT_;
      while (hStart < 0)
        hStart += dilationH_;
      while (wStart < 0)
        wStart += dilationW_;

      // maxIndex remains in "channels-first"/contiguous
      int64_t maxIndex =
          tStart * iheight_ * iwidth_ + hStart * iwidth_ + wStart;

      if (!channels_last_) {
        inputDataSlice += (int64_t)slice * itime_ * iheight_ * iwidth_;
      } else {
        inputDataSlice +=
            ((int64_t)batch * itime_ * iheight_ * iwidth_ * features_) +
            channel;
      }

      scalar_t max = at::numeric_limits<scalar_t>::lower_bound();

      for (int t = tStart; t < tEnd; t += dilationT_) {
        for (int h = hStart; h < hEnd; h += dilationH_) {
          for (int w = wStart; w < wEnd; w += dilationW_) {
            scalar_t val;
            int index = t * iheight_ * iwidth_ + h * iwidth_ + w;
            if (!channels_last_) {
              val = inputDataSlice[index];
            } else {
              int64_t index_channels_last = index * features_;
              val = inputDataSlice[index_channels_last];
            }

            if ((max < val) || at::_isnan(val)) {
              max = val;
              maxIndex = index;
            }
          }
        }
      }

      int64_t out_index;
      if (!channels_last_) {
        out_index = (int64_t)slice * otime_ * oheight_ * owidth_ +
            oFrame * oheight_ * owidth_ + oRow * owidth_ + oColumn;
      } else {
        out_index = ((int64_t)batch * otime_ * oheight_ * owidth_ +
                     oFrame * oheight_ * owidth_ + oRow * owidth_ + oColumn) *
                features_ +
            channel;
      }
      outputData_[out_index] = max;
      indicesData_[out_index] = maxIndex;
    }
  }
  MaxPool3dWithIndicesOutFrameImplKernelFunctor(
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
      int offsetZ,
      bool channels_last)
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
        offsetZ_(offsetZ),
        channels_last_(channels_last) {}

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
  int offsetZ_;
  bool channels_last_;
};

template <typename scalar_t>
static void max_pool3d_with_indices_out_frame(
    const scalar_t* inputData,
    const Tensor& output,
    const Tensor& indices,
    int features,
    int64_t totalZ,
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
    bool channels_last) {
  int offsetZ = 0;
  while (totalZ > 0) {
    MaxPool3dWithIndicesOutFrameImplKernelFunctor<scalar_t> kfn(
        inputData,
        output.mutable_data_ptr<scalar_t>(),
        indices.mutable_data_ptr<int64_t>(),
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
        offsetZ,
        channels_last);

    // width size is fixed size = 32, height dim equals = syclMaxWorkGroupSize
    // / width_size
    int width_group_size = 32;
    int height_group_size = syclMaxWorkGroupSize(kfn) / width_group_size;
    int width_group_range = ceil_div<int>(owidth, width_group_size);
    int height_group_range = ceil_div<int>(oheight, height_group_size);

    int z_group_range = totalZ > 65535 ? 65535 : totalZ;
    auto& queue = getCurrentSYCLQueue();
    sycl_kernel_submit(
        sycl::range<3>{
            size_t(z_group_range),
            size_t(height_group_range * height_group_size),
            size_t(width_group_range * width_group_size),
        },
        sycl::range<3>{1, size_t(height_group_size), size_t(width_group_size)},
        queue,
        kfn);
    totalZ -= 65535;
    offsetZ += 65535;
  }
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
        const int64_t totalZ = otime * nslices * nbatch;

        max_pool3d_with_indices_out_frame(
            input_data,
            work_output,
            work_indices,
            nslices, // features
            totalZ,
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
            dilationW,
            channels_last);
      });
}

template <typename scalar_t>
struct MaxPool3dWithIndicesBackwardOutFrameImplKernelFunctor {
  void operator()(sycl::nd_item<3> item) const {
    int oColumn = item.get_global_id()[2];
    int oRow = item.get_global_id()[1];
    int oFrame = 0;
    // used only for channels-first
    int64_t slice = 0;
    // used only for channels-last
    int batch = 0;
    int channel = 0;
    if (!channels_last_) {
      // order: batch, channel, time
      oFrame = (item.get_global_id()[0] + offsetZ_) % otime_;
      slice = (item.get_global_id()[0] + offsetZ_) / otime_;
    } else {
      // order: batch, time, channel
      channel = item.get_global_id()[0] % features_;
      slice = item.get_global_id()[0] / features_;
      batch = slice / otime_;
      oFrame = slice % otime_;
    }

    if (oRow < oheight_ && oColumn < owidth_ && oFrame < otime_ &&
        batch < obatch_ && channel < features_) {
      int64_t out_index;
      if (!channels_last_) {
        out_index = (int64_t)slice * otime_ * oheight_ * owidth_ +
            oFrame * oheight_ * owidth_ + oRow * owidth_ + oColumn;
      } else {
        out_index = ((int64_t)batch * otime_ * oheight_ * owidth_ +
                     oFrame * oheight_ * owidth_ + oRow * owidth_ + oColumn) *
                features_ +
            channel;
      }
      int64_t maxIndex = indicesData_[out_index];
      if (maxIndex != -1) {
        if (!channels_last_) {
          atomicAdd(
              (sycl_global_ptr<scalar_t>)&gradInputData_
                  [(int64_t)slice * itime_ * iheight_ * iwidth_ + maxIndex],
              gradOutputData_[out_index]);
        } else {
          atomicAdd(
              (sycl_global_ptr<scalar_t>)&gradInputData_
                  [((int64_t)batch * itime_ * iheight_ * iwidth_ + maxIndex) *
                       features_ +
                   channel],
              gradOutputData_[out_index]);
        }
      }
    }
  }
  MaxPool3dWithIndicesBackwardOutFrameImplKernelFunctor(
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
      int owidth,
      int offsetZ,
      bool channels_last)
      : gradInputData_(gradInputData),
        gradOutputData_(gradOutputData),
        indicesData_(indicesData),
        features_(features),
        itime_(itime),
        iheight_(iheight),
        iwidth_(iwidth),
        obatch_(obatch),
        otime_(otime),
        oheight_(oheight),
        owidth_(owidth),
        offsetZ_(offsetZ),
        channels_last_(channels_last) {}

 private:
  scalar_t* gradInputData_;
  const scalar_t* gradOutputData_;
  const int64_t* indicesData_;
  int features_;
  int itime_;
  int iheight_;
  int iwidth_;
  int obatch_;
  int otime_;
  int oheight_;
  int owidth_;
  int offsetZ_;
  bool channels_last_;
};

template <typename scalar_t>
void max_pool3d_with_indices_backward_out_frame(
    scalar_t* gradInputData,
    const Tensor& gradOutput,
    const Tensor& indices,
    int features,
    int64_t totalZ,
    int itime,
    int iheight,
    int iwidth,
    int obatch,
    int otime,
    int oheight,
    int owidth,
    bool channels_last) {
  int offsetZ = 0;
  while (totalZ > 0) {
    MaxPool3dWithIndicesBackwardOutFrameImplKernelFunctor<scalar_t> kfn(
        gradInputData,
        gradOutput.const_data_ptr<scalar_t>(),
        indices.const_data_ptr<int64_t>(),
        features,
        itime,
        iheight,
        iwidth,
        obatch,
        otime,
        oheight,
        owidth,
        offsetZ,
        channels_last);

    // width size is fixed size = 32, height dim equals = syclMaxWorkGroupSize
    // / width_size
    int width_group_size = 32;
    int height_group_size = syclMaxWorkGroupSize(kfn) / width_group_size;
    int width_group_range = ceil_div<int>(owidth, width_group_size);
    int height_group_range = ceil_div<int>(oheight, height_group_size);

    int z_group_range = totalZ > 65535 ? 65535 : totalZ;
    auto& queue = getCurrentSYCLQueue();
    sycl_kernel_submit(
        sycl::range<3>{
            size_t(z_group_range),
            size_t(height_group_range * height_group_size),
            size_t(width_group_range * width_group_size),
        },
        sycl::range<3>{1, size_t(height_group_size), size_t(width_group_size)},
        queue,
        kfn);
    totalZ -= 65535;
    offsetZ += 65535;
  }
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
      "max_pool2d_with_indices_backward_out_cuda_template(): ",
      "Expected 4D or 5D input tensor, but got ",
      input.sizes());

  TORCH_CHECK(
      (gradOutput.ndimension() == 4 || gradOutput.ndimension() == 5),
      "max_pool2d_with_indices_backward_out_cuda_template(): ",
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
        const int64_t totalZ = otime * nslices * nbatch;
        scalar_t* grad_input_data =
            work_grad_input.mutable_data_ptr<scalar_t>();

        max_pool3d_with_indices_backward_out_frame(
            grad_input_data,
            work_grad_output,
            work_indices,
            nslices,
            totalZ,
            itime,
            iheight,
            iwidth,
            nbatch,
            otime,
            oheight,
            owidth,
            channels_last);
      });
}

} // namespace at::native::xpu

#pragma GCC diagnostic pop
#pragma clang diagnostic pop
