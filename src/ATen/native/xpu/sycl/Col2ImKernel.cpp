#include <ATen/Dispatch.h>
#include <comm/xpu_aten.h>

#include <ATen/native/im2col_shape_check.h>
#include <comm/Runtime.h>
#include <comm/SYCLHelpers.h>

#include <ATen/native/xpu/sycl/Col2ImKernel.h>

namespace at::native::xpu {

template <typename T>
struct Col2imKernelFunctor {
  void operator()(sycl::item<1> itemId) const {
    auto in_ptr = in_data;
    auto out_ptr = out_data;
    auto id = itemId.get_id(0);

    T val = static_cast<T>(0);
    const int64_t w_im = id % width + pad_w;
    const int64_t h_im = (id / width) % height + pad_h;
    const int64_t c_im = id / (width * height);
    int64_t kernel_extent_w = (kernel_w - 1) * dilation_w + 1;
    int64_t kernel_extent_h = (kernel_h - 1) * dilation_h + 1;
    // compute the start and end of the output
    const int64_t w_col_start =
        (w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) / stride_w + 1;
    const int64_t w_col_end = std::min(w_im / stride_w + 1, output_width);
    const int64_t h_col_start =
        (h_im < kernel_extent_h) ? 0 : (h_im - kernel_extent_h) / stride_h + 1;
    const int64_t h_col_end = std::min(h_im / stride_h + 1, output_height);

    for (int64_t h_col = h_col_start; h_col < h_col_end; h_col += 1) {
      for (int64_t w_col = w_col_start; w_col < w_col_end; w_col += 1) {
        int64_t h_k = (h_im - h_col * stride_h);
        int64_t w_k = (w_im - w_col * stride_w);
        if (h_k % dilation_h == 0 && w_k % dilation_w == 0) {
          h_k /= dilation_h;
          w_k /= dilation_w;
          int64_t data_col_index =
              (((c_im * kernel_h + h_k) * kernel_w + w_k) * output_height +
               h_col) *
                  output_width +
              w_col;
          val += in_ptr[data_col_index];
        }
      }
    }
    out_ptr[id] = static_cast<T>(val);
  }
  Col2imKernelFunctor(
      const T* in_data_,
      const int64_t channels_,
      const int64_t height_,
      const int64_t width_,
      const int64_t output_height_,
      const int64_t output_width_,
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
        output_height(output_height_),
        output_width(output_width_),
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
  const int64_t output_height;
  const int64_t output_width;
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
static void col2im_kernel(
    const T* data_col,
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
    T* data_im) {
  auto& sycl_queue = at::xpu::getCurrentSYCLQueue();
  auto total_threads = channels * width * height;

  auto in_data = data_col;
  auto out_data = data_im;
  auto kfn = Col2imKernelFunctor<T>(
      in_data,
      channels,
      height,
      width,
      output_height,
      output_width,
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

void col2im_kernel(
    Tensor& output,
    const Tensor& input_,
    IntArrayRef output_size,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride) {
  TORCH_CHECK(
      output_size.size() == 2,
      "It is expected output_size equals to 2, but got size ",
      output_size.size());

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

  auto output_height = output_size[0];
  auto output_width = output_size[1];
  auto kernel_height = kernel_size[0];
  auto kernel_width = kernel_size[1];
  auto dilation_height = dilation[0];
  auto dilation_width = dilation[1];
  auto pad_height = padding[0];
  auto pad_width = padding[1];
  auto stride_height = stride[0];
  auto stride_width = stride[1];

  col2im_shape_check(
      input_,
      Tensor(),
      output_height,
      output_width,
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
  if (input.dim() == 2) {
    batched_input = false;
    input = input.view({1, input.size(0), input.size(1)});
  }

  auto batch_size = input.size(0);
  auto n_input_plane = input.size(1);
  auto n_output_plane = n_input_plane / (kernel_width * kernel_height);

  output.resize_({batch_size, n_output_plane, output_height, output_width});
  output.zero_();

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND3(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      at::ScalarType::Bool,
      input.scalar_type(),
      "col2im_xpu",
      [&] {
        Tensor input_n = Tensor();
        Tensor output_n = Tensor();

        auto height_col = (output_height + 2 * pad_height -
                           (dilation_height * (kernel_height - 1) + 1)) /
                stride_height +
            1;
        auto width_col = (output_width + 2 * pad_width -
                          (dilation_width * (kernel_width - 1) + 1)) /
                stride_width +
            1;

        for (int64_t elt = 0; elt < batch_size; elt++) {
          input_n = input.select(0, elt);
          output_n = output.select(0, elt);

          col2im_kernel<scalar_t>(
              input_n.const_data_ptr<scalar_t>(),
              n_output_plane,
              output_height,
              output_width,
              height_col,
              width_col,
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
          output.resize_({n_output_plane, output_height, output_width});
        }
      });
}

} // namespace at::native::xpu
