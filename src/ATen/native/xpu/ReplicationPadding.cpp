#include <ATen/Context.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/Padding.h>
#include <ATen/native/xpu/sycl/ReplicationPaddingKernels.h>
#include <ATen/xpu/XPUNativeFunctions.h>
#include <comm/RegisterUtils.h>

namespace at {

void replication_pad1d_meta(
    Tensor& output,
    const Tensor& input,
    IntArrayRef paddingSize) {
  TORCH_CHECK(paddingSize.size() == 2, "padding size is expected to be 2");

  int64_t dimw = 1;
  int64_t dimslices = 0;
  int64_t nbatch = 1;

  int64_t pad_l = paddingSize[0];
  int64_t pad_r = paddingSize[1];

  at::native::padding::check_valid_input<1>(input, paddingSize);

  if (input.ndimension() == 3) {
    nbatch = input.size(0);
    dimw++;
    dimslices++;
  }

  /* sizes */
  int64_t nslices = input.size(dimslices);
  int64_t iwidth = input.size(dimw);
  int64_t owidth = iwidth + pad_l + pad_r;

  TORCH_CHECK(owidth >= 1,
      "input (W: ", iwidth, ") is too small."
      " Calculated output W: ", owidth);

  if (output.defined()) {
    if (input.ndimension() == 2) {
      xpu::resize_out(output, {nslices, owidth}, {}, input.options());
    } else {
      xpu::resize_out(output, {nbatch, nslices, owidth}, {}, input.options());
    }
  } else {
    if (input.ndimension() == 2) {
      output = xpu::create_out({nslices, owidth}, {}, input.options());
    } else {
      output = xpu::create_out({nbatch, nslices, owidth}, {}, input.options());
    }
  }
}

void replication_pad1d_backward_meta(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef paddingSize) {
  int64_t dimw = 1;
  TORCH_CHECK(paddingSize.size() == 2, "padding size is expected to be 2");
  int64_t pad_l = paddingSize[0];
  int64_t pad_r = paddingSize[1];

  if (input.ndimension() == 3) {
    dimw++;
  }

  /* sizes */
  int64_t iwidth = input.size(dimw);
  int64_t owidth  = iwidth + pad_l + pad_r;

  TORCH_CHECK(owidth == grad_output.size(dimw),
      "grad_output width unexpected. Expected: ", owidth,
      " Got: ", grad_output.size(dimw));

  if (grad_input.defined()) {
    xpu::resize_out(grad_input, input.sizes(), {}, input.options());
  } else {
    grad_input = xpu::create_out(input.sizes(), {}, input.options());
  }
}

void replication_pad2d_meta(
    Tensor& output,
    const Tensor& input,
    IntArrayRef paddingSize) {
  TORCH_CHECK(paddingSize.size() == 4, "padding size is expected to be 4");
  int64_t pad_l = paddingSize[0];
  int64_t pad_r = paddingSize[1];
  int64_t pad_t = paddingSize[2];
  int64_t pad_b = paddingSize[3];
  int64_t dimw = 2;
  int64_t dimh = 1;
  int64_t dimslices = 0;
  int64_t nbatch = 1;

  at::native::padding::check_valid_input<2>(input, paddingSize);

  if (input.dim() == 4) {
    nbatch = input.size(0);
    dimw++;
    dimh++;
    dimslices++;
  }

  /* sizes */
  int64_t nslices = input.size(dimslices);
  int64_t iheight = input.size(dimh);
  int64_t iwidth = input.size(dimw);
  int64_t oheight = iheight + pad_t + pad_b;
  int64_t owidth  = iwidth + pad_l + pad_r;

  TORCH_CHECK(owidth >= 1 || oheight >= 1,
      "input (H: ", iheight, ", W: ", iwidth, " ) is too small."
      " Calculated output H: ", oheight, " W: ", owidth);

  if (output.defined()) {
    if (input.dim() == 3) {
      xpu::resize_out(
          output,
          {nslices, oheight, owidth}, {}, input.options());
    } else {
      xpu::resize_out(
          output, {nbatch, nslices, oheight, owidth}, {}, input.options());
    }
  } else {
    if (input.dim() == 3) {
      output = xpu::create_out(
          {nslices, oheight, owidth}, {}, input.options());
    } else {
      output = xpu::create_out(
          {nbatch, nslices, oheight, owidth}, {}, input.options());
    }
  }
}

void replication_pad3d_meta(
    Tensor& output,
    const Tensor& input,
    IntArrayRef paddingSize) {
  TORCH_CHECK(paddingSize.size() == 6, "padding size is expected to be 6");
  int64_t pleft = paddingSize[0];
  int64_t pright = paddingSize[1];
  int64_t ptop = paddingSize[2];
  int64_t pbottom = paddingSize[3];
  int64_t pfront = paddingSize[4];
  int64_t pback = paddingSize[5];
  int64_t dimw = 3;
  int64_t dimh = 2;
  int64_t dimd = 1;
  int64_t dimslices = 0;
  int64_t nbatch = 1;

  at::native::padding::check_valid_input<3>(input, paddingSize);

  if (input.dim() == 5) {
    nbatch = input.size(0);
    dimw++;
    dimh++;
    dimd++;
    dimslices++;
  }

  /* sizes */
  int64_t nslices = input.size(dimslices);
  int64_t idepth = input.size(dimd);
  int64_t iheight = input.size(dimh);
  int64_t iwidth = input.size(dimw);
  int64_t odepth = idepth + pfront + pback;
  int64_t oheight = iheight + ptop + pbottom;
  int64_t owidth  = iwidth + pleft + pright;

  TORCH_CHECK(owidth >= 1 || oheight >= 1 || odepth >= 1,
      "input (D: ", idepth, " H: ", iheight, ", W: ", iwidth,
      ") is too small."
      " Calculated output D: ", odepth, " H: ", oheight, " W: ", owidth);

  if (output.defined()) {
    if (input.dim() == 4) {
      xpu::resize_out(
          output,
          {nslices, odepth, oheight, owidth}, {}, input.options());
    } else {
      xpu::resize_out(
          output, {nbatch, nslices, odepth, oheight, owidth}, {}, input.options());
    }
  } else {
    if (input.dim() == 4) {
      output = xpu::create_out(
          {nslices, odepth, oheight, owidth}, {}, input.options());
    } else {
      output = xpu::create_out(
          {nbatch, nslices, odepth, oheight, owidth}, {}, input.options());
    }
  }
}

Tensor XPUNativeFunctions::replication_pad1d(
    const Tensor& input,
    IntArrayRef padding) {
  Tensor output;
  replication_pad1d_meta(output, input, padding);
  native::xpu::replication_pad1d_kernel(output, input, padding);
  return output;
}

Tensor& XPUNativeFunctions::replication_pad1d_out(
    const Tensor& input,
    IntArrayRef padding,
    Tensor& output) {
  replication_pad1d_meta(output, input, padding);
  native::xpu::replication_pad1d_kernel(output, input, padding);
  return output;
}

Tensor XPUNativeFunctions::replication_pad1d_backward(
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef padding) {
  Tensor grad_input;
  replication_pad1d_backward_meta(grad_input, grad_output, input, padding);
  native::xpu::replication_pad1d_backward_kernel(
      grad_input, grad_output, input, padding);
  return grad_input;
}

Tensor& XPUNativeFunctions::replication_pad1d_backward_out(
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef padding,
    Tensor& grad_input) {
  replication_pad1d_backward_meta(grad_input, grad_output, input, padding);
  native::xpu::replication_pad1d_backward_kernel(
      grad_input, grad_output, input, padding);
  return grad_input;
}

Tensor& XPUNativeFunctions::replication_pad2d_out(
    const Tensor& input,
    IntArrayRef padding,
    Tensor& output) {
  replication_pad2d_meta(output, input, padding);
  native::xpu::replication_pad2d_kernel(output, input, padding);
  return output;
}

Tensor XPUNativeFunctions::replication_pad2d(
    const Tensor& input,
    IntArrayRef padding) {
  Tensor output;
  replication_pad2d_meta(output, input, padding);
  native::xpu::replication_pad2d_kernel(output, input, padding);
  return output;
}

Tensor& XPUNativeFunctions::replication_pad2d_backward_out(
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef padding,
    Tensor& grad_input) {
  native::xpu::replication_pad2d_backward_kernel(
      grad_input, grad_output, input, padding);
  return grad_input;
}

Tensor XPUNativeFunctions::replication_pad2d_backward(
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef padding) {
  auto grad_input = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  native::xpu::replication_pad2d_backward_kernel(
      grad_input, grad_output, input, padding);
  return grad_input;
}

Tensor XPUNativeFunctions::replication_pad3d(
    const Tensor& input,
    IntArrayRef padding) {
  Tensor output;
  replication_pad3d_meta(output, input, padding);
  native::xpu::replication_pad3d_kernel(output, input, padding);
  return output;
}

Tensor& XPUNativeFunctions::replication_pad3d_out(
    const Tensor& input,
    IntArrayRef padding,
    Tensor& output) {
  replication_pad3d_meta(output, input, padding);
  native::xpu::replication_pad3d_kernel(output, input, padding);
  return output;
}

Tensor XPUNativeFunctions::replication_pad3d_backward(
    const Tensor& grad_output,
    const Tensor& input,
    at::IntArrayRef padding) {
  auto grad_input = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  native::xpu::replication_pad3d_backward_kernel(
      grad_input, grad_output, input, padding);
  return grad_input;
}

Tensor& XPUNativeFunctions::replication_pad3d_backward_out(
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef padding,
    Tensor& grad_input) {
  native::xpu::replication_pad3d_backward_kernel(
      grad_input, grad_output, input, padding);
  return grad_input;
}

} // namespace at