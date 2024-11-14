#include <ATen/Dispatch.h>
#include <comm/xpu_aten.h>

#include <ATen/native/im2col_shape_check.h>
#include <comm/Runtime.h>
#include <comm/SYCLHelpers.h>

#include <ATen/native/xpu/sycl/Im2ColKernel.h>

namespace at {
namespace native {
namespace xpu {

template <typename T>
struct Im2colKernelFunctor {
  void operator()(sycl::item<1> itemId) const {
    auto in_ptr = in_data;
    auto out_ptr = out_data;
    auto id = itemId.get_id(0);

    int64_t w_out = id % width_col;
    id /= width_col;

    int64_t h_out = id % height_col;
    int64_t channel_in = id / height_col;
    int64_t channel_out = channel_in * kernel_h * kernel_w;
    int64_t h_in = h_out * stride_h - pad_h;
    int64_t w_in = w_out * stride_w - pad_w;

    out_ptr += (channel_out * height_col + h_out) * width_col + w_out;
    in_ptr += (channel_in * height + h_in) * width + w_in;

    for (int64_t i = 0; i < kernel_h; ++i) {
      for (int64_t j = 0; j < kernel_w; ++j) {
        int64_t h = h_in + i * dilation_h;
        int64_t w = w_in + j * dilation_w;
        *out_ptr = (h >= 0 && w >= 0 && h < height && w < width)
            ? in_ptr[i * dilation_h * width + j * dilation_w]
            : static_cast<T>(0);
        ;
        out_ptr += height_col * width_col;
      }
    }
  }
  Im2colKernelFunctor(
      const T* in_data_,
      const int64_t channels_,
      const int64_t height_,
      const int64_t width_,
      const int64_t height_col_,
      const int64_t width_col_,
      const int64_t kernel_h_,
      const int64_t kernel_w_,
      const int64_t pad_h_,
      const int64_t pad_w_,
      const int64_t stride_h_,
      const int64_t stride_w_,
      const int64_t dilation_h_,
      const int64_t dilation_w_,
      T* out_data_)
      : in_data(in_data_),
        channels(channels_),
        height(height_),
        width(width_),
        height_col(height_col_),
        width_col(width_col_),
        kernel_h(kernel_h_),
        kernel_w(kernel_w_),
        pad_h(pad_h_),
        pad_w(pad_w_),
        stride_h(stride_h_),
        stride_w(stride_w_),
        dilation_h(dilation_h_),
        dilation_w(dilation_w_),
        out_data(out_data_) {}

 private:
  const T* in_data;
  const int64_t channels;
  const int64_t height;
  const int64_t width;
  const int64_t height_col;
  const int64_t width_col;
  const int64_t kernel_h;
  const int64_t kernel_w;
  const int64_t pad_h;
  const int64_t pad_w;
  const int64_t stride_h;
  const int64_t stride_w;
  const int64_t dilation_h;
  const int64_t dilation_w;
  T* out_data;
};

template <typename T>
static void im2col_kernel(
    const T* data_im,
    const int64_t channels,
    const int64_t height,
    const int64_t width,
    const int64_t output_height,
    const int64_t output_width,
    const int64_t kernel_h,
    const int64_t kernel_w,
    const int64_t pad_h,
    const int64_t pad_w,
    const int64_t stride_h,
    const int64_t stride_w,
    const int64_t dilation_h,
    const int64_t dilation_w,
    T* data_col) {
  const int64_t height_col = output_height;
  const int64_t width_col = output_width;
  auto& sycl_queue = at::xpu::getCurrentSYCLQueue();
  auto total_threads = channels * output_width * output_height;

  auto in_data = data_im;
  auto out_data = data_col;
  auto kfn = Im2colKernelFunctor<T>(
      in_data,
      channels,
      height,
      width,
      height_col,
      width_col,
      kernel_h,
      kernel_w,
      pad_h,
      pad_w,
      stride_h,
      stride_w,
      dilation_h,
      dilation_w,
      out_data);
  sycl_kernel_submit(::sycl::range<1>(total_threads), sycl_queue, kfn);
}

void im2col_kernel(
    Tensor& output,
    const Tensor& input_,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride) {
  TORCH_CHECK(
      kernel_size.size() == 2,
      "It is expected kernel_size equals to 2, but got size ",
      kernel_size.size());

  TORCH_CHECK(
      dilation.size() == 2,
      "It is expected dilation equals to 2, but got size ",
      dilation.size());

  TORCH_CHECK(
      padding.size() == 2,
      "It is expected padding equals to 2, but got size ",
      padding.size());

  TORCH_CHECK(
      stride.size() == 2,
      "It is expected stride equals to 2, but got size ",
      stride.size());

  auto kernel_height = kernel_size[0];
  auto kernel_width = kernel_size[1];
  auto dilation_height = dilation[0];
  auto dilation_width = dilation[1];
  auto pad_height = padding[0];
  auto pad_width = padding[1];
  auto stride_height = stride[0];
  auto stride_width = stride[1];

  im2col_shape_check(
      input_,
      Tensor(),
      kernel_height,
      kernel_width,
      dilation_height,
      dilation_width,
      pad_height,
      pad_width,
      stride_height,
      stride_width);

  Tensor input = input_.contiguous();

  bool batched_input = true;

  if (input.dim() == 3) {
    batched_input = false;
    input = input.view({1, input.size(0), input.size(1), input.size(2)});
  }

  auto batch_size = input.size(0);
  auto n_input_plane = input.size(1);
  auto input_height = input.size(2);
  auto input_width = input.size(3);

  auto output_height = (input_height + 2 * pad_height -
                        (dilation_height * (kernel_height - 1) + 1)) /
          stride_height +
      1;
  auto output_width = (input_width + 2 * pad_width -
                       (dilation_width * (kernel_width - 1) + 1)) /
          stride_width +
      1;
  auto n_output_plane = n_input_plane * kernel_width * kernel_height;
  auto output_length = output_height * output_width;

  output.resize_({batch_size, n_output_plane, output_length});
  output.zero_();

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND3(
      kHalf, kBFloat16, kBool, input.scalar_type(), "im2col_xpu", [&] {
        Tensor input_n;
        Tensor output_n;

        for (int64_t elt = 0; elt < batch_size; elt++) {
          input_n = input.select(0, elt);
          output_n = output.select(0, elt);

          im2col_kernel<scalar_t>(
              input_n.const_data_ptr<scalar_t>(),
              n_input_plane,
              input_height,
              input_width,
              output_height,
              output_width,
              kernel_height,
              kernel_width,
              pad_height,
              pad_width,
              stride_height,
              stride_width,
              dilation_height,
              dilation_width,
              output_n.data_ptr<scalar_t>());
        }

        if (!batched_input) {
          output.resize_({n_output_plane, output_length});
        }
      });
}

} // namespace xpu
} // namespace native
} // namespace at
