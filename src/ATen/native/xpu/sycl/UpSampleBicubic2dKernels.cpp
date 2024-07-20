#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorUtils.h>
#include <ATen/ceil_div.h>
#include <ATen/native/xpu/UpSample.h>
#include <comm/SYCLContext.h>

namespace at::native::xpu {

template <typename scalar_t, typename accscalar_t>
struct UpsampleBicubic2dKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    auto idata = in_data_;
    auto odata = out_data_;
    int global_id = item.get_global_linear_id();
    const int nbatch = idata.size(0);
    const int channels = idata.size(1);
    const int input_height = idata.size(2);
    const int input_width = idata.size(3);
    const int output_height = odata.size(2);
    const int output_width = odata.size(3);
    if (global_id < output_height * output_width) {
      const int output_x = global_id % output_width;
      const int output_y = global_id / output_width;
      if (input_height == output_height && input_width == output_width) {
        for (int n = 0; n < nbatch; n++) {
          for (int c = 0; c < channels; c++) {
            auto val = idata[n][c][output_y][output_x];
            odata[n][c][output_y][output_x] = val;
          }
        }
        return;
      }

      // Interpolation kernel
      accscalar_t real_x = area_pixel_compute_source_index(
          width_scale_, output_x, align_corners_, /*cubic=*/true);
      int in_x = floorf(real_x);
      accscalar_t t_x = real_x - in_x;

      accscalar_t real_y = area_pixel_compute_source_index(
          height_scale_, output_y, align_corners_, /*cubic=*/true);
      int in_y = floorf(real_y);
      accscalar_t t_y = real_y - in_y;
      for (int n = 0; n < nbatch; n++) {
        for (int c = 0; c < channels; c++) {
          accscalar_t coefficients[4];
          for (int k = 0; k < 4; k++) {
            coefficients[k] = cubic_interp1d<scalar_t, accscalar_t>(
                upsample_get_value_bounded<scalar_t>(
                    idata,
                    n,
                    c,
                    input_width,
                    input_height,
                    in_x - 1,
                    in_y - 1 + k),
                upsample_get_value_bounded<scalar_t>(
                    idata,
                    n,
                    c,
                    input_width,
                    input_height,
                    in_x + 0,
                    in_y - 1 + k),
                upsample_get_value_bounded<scalar_t>(
                    idata,
                    n,
                    c,
                    input_width,
                    input_height,
                    in_x + 1,
                    in_y - 1 + k),
                upsample_get_value_bounded<scalar_t>(
                    idata,
                    n,
                    c,
                    input_width,
                    input_height,
                    in_x + 2,
                    in_y - 1 + k),
                t_x);
          }

          odata[n][c][output_y][output_x] =
              static_cast<scalar_t>(cubic_interp1d<scalar_t, accscalar_t>(
                  coefficients[0],
                  coefficients[1],
                  coefficients[2],
                  coefficients[3],
                  t_y));
        }
      }
    }
  }
  UpsampleBicubic2dKernelFunctor(
      PackedTensorAccessor64<scalar_t, 4> out_data,
      const PackedTensorAccessor64<const scalar_t, 4> in_data,
      int64_t onum,
      bool align_corners,
      const accscalar_t height_scale,
      const accscalar_t width_scale)
      : out_data_(out_data),
        in_data_(in_data),
        onum_(onum),
        align_corners_(align_corners),
        height_scale_(height_scale),
        width_scale_(width_scale) {}

 private:
  PackedTensorAccessor64<scalar_t, 4> out_data_;
  const PackedTensorAccessor64<const scalar_t, 4> in_data_;
  int64_t onum_;
  bool align_corners_;
  const accscalar_t height_scale_;
  const accscalar_t width_scale_;
};

template <typename scalar_t, typename accscalar_t>
static void upsample_bicubic2d_out_template(
    PackedTensorAccessor64<scalar_t, 4> odata,
    const PackedTensorAccessor64<const scalar_t, 4> idata,
    int64_t onum,
    bool align_corners,
    const accscalar_t height_scale,
    const accscalar_t width_scale) {
  UpsampleBicubic2dKernelFunctor<scalar_t, accscalar_t> kfn(
      odata, idata, onum, align_corners, height_scale, width_scale);

  int64_t wg_size = syclMaxWorkGroupSize(kfn);
  int64_t num_wg = at::ceil_div(onum, wg_size);
  auto queue = getCurrentSYCLQueue();

  sycl_kernel_submit(num_wg * wg_size, wg_size, queue, kfn);
}

void upsample_bicubic2d_kernel(
    Tensor& output,
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners,
    std::optional<double> scales_h,
    std::optional<double> scales_w) {
  TensorArg input_arg{input, "input", 1}, output_arg{output, "output", 2};
  checkAllSameGPU(__func__, {input_arg, output_arg});

  int output_height = output_size[0];
  int output_width = output_size[1];

  int input_height = input.size(2);
  int input_width = input.size(3);

  output.zero_();

  const int num_output_elements = output_height * output_width;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "upsample_bicubic2d_xpu",
      [&] {
        auto idata = input.packed_accessor64<const scalar_t, 4>();
        auto odata = output.packed_accessor64<scalar_t, 4>();

        // Get scaling factors
        using accscalar_t = at::acc_type<scalar_t, true>;
        const accscalar_t rheight = area_pixel_compute_scale<accscalar_t>(
            input_height, output_height, align_corners, scales_h);
        const accscalar_t rwidth = area_pixel_compute_scale<accscalar_t>(
            input_width, output_width, align_corners, scales_w);

        upsample_bicubic2d_out_template<scalar_t, accscalar_t>(
            odata, idata, num_output_elements, align_corners, rheight, rwidth);
      });
}

} // namespace at::native::xpu
