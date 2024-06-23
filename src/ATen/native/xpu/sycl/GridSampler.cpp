#include <ATen/ATen.h>
#include <ATen/OpMathType.h>
#include <ATen/native/CanUse32BitIndexMath.h>
#include <ATen/native/GridSamplerUtils.h>
#include <comm/SYCLContext.h>
#include <comm/TensorInfo.h>

#include "GridSampler.h"
#include "UpSample.h"

namespace at::native::xpu {

using namespace at::xpu::detail;

template <typename scalar_t, typename index_t>
struct GridSampler2dKernelFunctor {
  using opmath_t = at::opmath_type<scalar_t>;
  void operator()(sycl::nd_item<1> item_id) const {
    auto index = item_id.get_global_linear_id();
    if (index >= nthreads_)
      return;
    const index_t w = index % out_W_;
    const index_t h = (index / out_W_) % out_H_;
    const index_t n = index / (out_H_ * out_W_);
    const index_t grid_offset = n * grid_sN_ + h * grid_sH_ + w * grid_sW_;

    // get the corresponding input x, y co-ordinates from grid
    opmath_t x = grid_.data[grid_offset];
    opmath_t y = grid_.data[grid_offset + grid_sCoor_];

    opmath_t ix = at::native::xpu::grid_sampler_compute_source_index(
        x, inp_W_, padding_mode_, align_corners_);
    opmath_t iy = at::native::xpu::grid_sampler_compute_source_index(
        y, inp_H_, padding_mode_, align_corners_);

    if (interpolation_mode_ == GridSamplerInterpolation::Bilinear) {
      // get NE, NW, SE, SW pixel values from (x, y)
      index_t ix_nw = static_cast<index_t>(std::floor(ix));
      index_t iy_nw = static_cast<index_t>(std::floor(iy));
      index_t ix_ne = ix_nw + 1;
      index_t iy_ne = iy_nw;
      index_t ix_sw = ix_nw;
      index_t iy_sw = iy_nw + 1;
      index_t ix_se = ix_nw + 1;
      index_t iy_se = iy_nw + 1;

      // get surfaces to each neighbor:
      opmath_t nw = (ix_se - ix) * (iy_se - iy);
      opmath_t ne = (ix - ix_sw) * (iy_sw - iy);
      opmath_t sw = (ix_ne - ix) * (iy - iy_ne);
      opmath_t se = (ix - ix_nw) * (iy - iy_nw);

      // calculate bilinear weighted pixel value and set output pixel
      auto inp_ptr_NC = input_.data + n * inp_sN_;
      auto out_ptr_NCHW =
          output_.data + n * out_sN_ + h * out_sH_ + w * out_sW_;
      for (index_t c = 0; c < C_;
           ++c, inp_ptr_NC += inp_sC_, out_ptr_NCHW += out_sC_) {
        opmath_t out_acc = 0;
        if (within_bounds_2d(iy_nw, ix_nw, inp_H_, inp_W_)) {
          out_acc += inp_ptr_NC[iy_nw * inp_sH_ + ix_nw * inp_sW_] * nw;
        }
        if (within_bounds_2d(iy_ne, ix_ne, inp_H_, inp_W_)) {
          out_acc += inp_ptr_NC[iy_ne * inp_sH_ + ix_ne * inp_sW_] * ne;
        }
        if (within_bounds_2d(iy_sw, ix_sw, inp_H_, inp_W_)) {
          out_acc += inp_ptr_NC[iy_sw * inp_sH_ + ix_sw * inp_sW_] * sw;
        }
        if (within_bounds_2d(iy_se, ix_se, inp_H_, inp_W_)) {
          out_acc += inp_ptr_NC[iy_se * inp_sH_ + ix_se * inp_sW_] * se;
        }
        *out_ptr_NCHW = out_acc;
      }
    } else if (interpolation_mode_ == GridSamplerInterpolation::Nearest) {
      index_t ix_nearest = static_cast<index_t>(std::nearbyint(ix));
      index_t iy_nearest = static_cast<index_t>(std::nearbyint(iy));

      // assign nearest neighor pixel value to output pixel
      auto inp_ptr_NC = input_.data + n * inp_sN_;
      auto out_ptr_NCHW =
          output_.data + n * out_sN_ + h * out_sH_ + w * out_sW_;
      for (index_t c = 0; c < C_;
           ++c, inp_ptr_NC += inp_sC_, out_ptr_NCHW += out_sC_) {
        if (within_bounds_2d(iy_nearest, ix_nearest, inp_H_, inp_W_)) {
          *out_ptr_NCHW =
              inp_ptr_NC[iy_nearest * inp_sH_ + ix_nearest * inp_sW_];
        } else {
          *out_ptr_NCHW = static_cast<scalar_t>(0);
        }
      }
    } else if (interpolation_mode_ == GridSamplerInterpolation::Bicubic) {
      ix = grid_sampler_unnormalize(x, inp_W_, align_corners_);
      iy = grid_sampler_unnormalize(y, inp_H_, align_corners_);

      opmath_t ix_nw = std::floor(ix);
      opmath_t iy_nw = std::floor(iy);

      const opmath_t tx = ix - ix_nw;
      const opmath_t ty = iy - iy_nw;

      auto inp_ptr_NC = input_.data + n * inp_sN_;
      auto out_ptr_NCHW =
          output_.data + n * out_sN_ + h * out_sH_ + w * out_sW_;
      for (index_t c = 0; c < C_;
           ++c, inp_ptr_NC += inp_sC_, out_ptr_NCHW += out_sC_) {
        opmath_t coefficients[4];

#pragma unroll 4
        for (index_t i = 0; i < 4; ++i) {
          coefficients[i] = cubic_interp1d(
              at::native::xpu::get_value_bounded<scalar_t>(
                  inp_ptr_NC,
                  ix_nw - 1,
                  iy_nw - 1 + i,
                  inp_W_,
                  inp_H_,
                  inp_sW_,
                  inp_sH_,
                  padding_mode_,
                  align_corners_),
              at::native::xpu::get_value_bounded<scalar_t>(
                  inp_ptr_NC,
                  ix_nw + 0,
                  iy_nw - 1 + i,
                  inp_W_,
                  inp_H_,
                  inp_sW_,
                  inp_sH_,
                  padding_mode_,
                  align_corners_),
              at::native::xpu::get_value_bounded<scalar_t>(
                  inp_ptr_NC,
                  ix_nw + 1,
                  iy_nw - 1 + i,
                  inp_W_,
                  inp_H_,
                  inp_sW_,
                  inp_sH_,
                  padding_mode_,
                  align_corners_),
              at::native::xpu::get_value_bounded<scalar_t>(
                  inp_ptr_NC,
                  ix_nw + 2,
                  iy_nw - 1 + i,
                  inp_W_,
                  inp_H_,
                  inp_sW_,
                  inp_sH_,
                  padding_mode_,
                  align_corners_),
              tx);
        }

        *out_ptr_NCHW = cubic_interp1d(
            coefficients[0],
            coefficients[1],
            coefficients[2],
            coefficients[3],
            ty);
      }
    }
  }
  GridSampler2dKernelFunctor(
      const index_t nthreads,
      TensorInfo<scalar_t, index_t> input,
      TensorInfo<scalar_t, index_t> grid,
      TensorInfo<scalar_t, index_t> output,
      const GridSamplerInterpolation interpolation_mode,
      const GridSamplerPadding padding_mode,
      bool align_corners,
      index_t C,
      index_t inp_H,
      index_t inp_W,
      index_t out_H,
      index_t out_W,
      index_t inp_sN,
      index_t inp_sC,
      index_t inp_sH,
      index_t inp_sW,
      index_t grid_sN,
      index_t grid_sH,
      index_t grid_sW,
      index_t grid_sCoor,
      index_t out_sN,
      index_t out_sC,
      index_t out_sH,
      index_t out_sW)
      : nthreads_(nthreads),
        input_(input),
        grid_(grid),
        output_(output),
        interpolation_mode_(interpolation_mode),
        padding_mode_(padding_mode),
        align_corners_(align_corners),
        C_(C),
        inp_H_(inp_H),
        inp_W_(inp_W),
        out_H_(out_H),
        out_W_(out_W),
        inp_sN_(inp_sN),
        inp_sC_(inp_sC),
        inp_sH_(inp_sH),
        inp_sW_(inp_sW),
        grid_sN_(grid_sN),
        grid_sH_(grid_sH),
        grid_sW_(grid_sW),
        grid_sCoor_(grid_sCoor),
        out_sN_(out_sN),
        out_sC_(out_sC),
        out_sH_(out_sH),
        out_sW_(out_sW) {}

 private:
  const index_t nthreads_;
  TensorInfo<scalar_t, index_t> input_;
  TensorInfo<scalar_t, index_t> grid_;
  TensorInfo<scalar_t, index_t> output_;
  const GridSamplerInterpolation interpolation_mode_;
  const GridSamplerPadding padding_mode_;
  bool align_corners_;
  index_t C_;
  index_t inp_H_;
  index_t inp_W_;
  index_t out_H_;
  index_t out_W_;
  index_t inp_sN_;
  index_t inp_sC_;
  index_t inp_sH_;
  index_t inp_sW_;
  index_t grid_sN_;
  index_t grid_sH_;
  index_t grid_sW_;
  index_t grid_sCoor_;
  index_t out_sN_;
  index_t out_sC_;
  index_t out_sH_;
  index_t out_sW_;
};

template <typename scalar_t, typename index_t>
void grid_sampler_2d_forward_template(
    const index_t nthreads,
    TensorInfo<scalar_t, index_t> input,
    TensorInfo<scalar_t, index_t> grid,
    TensorInfo<scalar_t, index_t> output,
    const GridSamplerInterpolation interpolation_mode,
    const GridSamplerPadding padding_mode,
    bool align_corners) {
  auto& queue = getCurrentSYCLQueue();
  const auto wgroup_size = syclMaxWorkGroupSize();
  const auto ngroups = (nthreads + wgroup_size - 1) / wgroup_size;

  index_t C = input.sizes[1];
  index_t inp_H = input.sizes[2];
  index_t inp_W = input.sizes[3];
  index_t out_H = grid.sizes[1];
  index_t out_W = grid.sizes[2];
  index_t inp_sN = input.strides[0];
  index_t inp_sC = input.strides[1];
  index_t inp_sH = input.strides[2];
  index_t inp_sW = input.strides[3];
  index_t grid_sN = grid.strides[0];
  index_t grid_sH = grid.strides[1];
  index_t grid_sW = grid.strides[2];
  index_t grid_sCoor = grid.strides[3];
  index_t out_sN = output.strides[0];
  index_t out_sC = output.strides[1];
  index_t out_sH = output.strides[2];
  index_t out_sW = output.strides[3];

  GridSampler2dKernelFunctor<scalar_t, index_t> kfn(
      nthreads,
      input,
      grid,
      output,
      interpolation_mode,
      padding_mode,
      align_corners,
      C,
      inp_H,
      inp_W,
      out_H,
      out_W,
      inp_sN,
      inp_sC,
      inp_sH,
      inp_sW,
      grid_sN,
      grid_sH,
      grid_sW,
      grid_sCoor,
      out_sN,
      out_sC,
      out_sH,
      out_sW);
  sycl_kernel_submit(
      sycl::range<1>(ngroups * wgroup_size),
      sycl::range<1>(wgroup_size),
      queue,
      kfn);
}

Tensor grid_sampler_2d_kernel(
    const Tensor& input,
    const Tensor& grid,
    int64_t interpolation_mode,
    int64_t padding_mode,
    bool align_corners) {
  check_grid_sampler_common(input, grid);
  check_grid_sampler_2d(input, grid);
  auto N = input.size(0);
  auto C = input.size(1);
  auto H = grid.size(1);
  auto W = grid.size(2);
  auto output = at::empty({N, C, H, W}, input.options());
  int64_t count = N * H * W;
  if (count > 0) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::BFloat16,
        at::ScalarType::Half,
        input.scalar_type(),
        "grid_sampler_2d_xpu",
        [&] {
          if (canUse32BitIndexMath(input) && canUse32BitIndexMath(grid) &&
              canUse32BitIndexMath(output)) {
            grid_sampler_2d_forward_template<scalar_t>(
                static_cast<int>(count),
                getTensorInfo<scalar_t, int>(input),
                getTensorInfo<scalar_t, int>(grid),
                getTensorInfo<scalar_t, int>(output),
                static_cast<GridSamplerInterpolation>(interpolation_mode),
                static_cast<GridSamplerPadding>(padding_mode),
                align_corners);
          } else {
            grid_sampler_2d_forward_template<scalar_t>(
                count,
                getTensorInfo<scalar_t, int64_t>(input),
                getTensorInfo<scalar_t, int64_t>(grid),
                getTensorInfo<scalar_t, int64_t>(output),
                static_cast<GridSamplerInterpolation>(interpolation_mode),
                static_cast<GridSamplerPadding>(padding_mode),
                align_corners);
          }
        });
  }
  return output;
}

template <typename scalar_t, typename index_t>
struct GridSampler2dBackwardKernelFunctor {
  void operator()(sycl::nd_item<1> item_id) const {
    auto index = item_id.get_global_linear_id();
    if (index >= nthreads)
      return;
    const index_t w = index % out_W;
    const index_t h = (index / out_W) % out_H;
    const index_t n = index / (out_H * out_W);
    const auto grid_offset = n * grid_sN + h * grid_sH + w * grid_sW;

    // get the corresponding input x, y co-ordinates from grid
    scalar_t x = grid.data[grid_offset];
    scalar_t y = grid.data[grid_offset + grid_sCoor];

    // multipliers for gradients on ix and iy
    scalar_t gix_mult, giy_mult;
    scalar_t ix = at::native::xpu::grid_sampler_compute_source_index_set_grad(
        x, inp_W, padding_mode, align_corners, &gix_mult);
    scalar_t iy = at::native::xpu::grid_sampler_compute_source_index_set_grad(
        y, inp_H, padding_mode, align_corners, &giy_mult);

    if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
      // get NE, NW, SE, SW pixel values from (x, y)
      index_t ix_nw = static_cast<index_t>(std::floor(ix));
      index_t iy_nw = static_cast<index_t>(std::floor(iy));
      index_t ix_ne = ix_nw + 1;
      index_t iy_ne = iy_nw;
      index_t ix_sw = ix_nw;
      index_t iy_sw = iy_nw + 1;
      index_t ix_se = ix_nw + 1;
      index_t iy_se = iy_nw + 1;

      // get surfaces to each neighbor:
      scalar_t nw = (ix_se - ix) * (iy_se - iy);
      scalar_t ne = (ix - ix_sw) * (iy_sw - iy);
      scalar_t sw = (ix_ne - ix) * (iy - iy_ne);
      scalar_t se = (ix - ix_nw) * (iy - iy_nw);

      scalar_t gix = static_cast<scalar_t>(0), giy = static_cast<scalar_t>(0);
      scalar_t* gOut_ptr_NCHW =
          grad_output_data + n * gOut_sN + h * gOut_sH + w * gOut_sW;
      scalar_t* gInp_ptr_NC = grad_input_data + n * gInp_sN;
      scalar_t* inp_ptr_NC = input_data + n * inp_sN;
      for (index_t c = 0; c < C; ++c,
                   inp_ptr_NC += inp_sC,
                   gInp_ptr_NC += gInp_sC,
                   gOut_ptr_NCHW += gOut_sC) {
        scalar_t gOut = *gOut_ptr_NCHW;

        if (input_requires_grad) {
          // calculate and set grad_input
          safe_add_2d(
              gInp_ptr_NC,
              iy_nw,
              ix_nw,
              gInp_sH,
              gInp_sW,
              inp_H,
              inp_W,
              nw * gOut);
          safe_add_2d(
              gInp_ptr_NC,
              iy_ne,
              ix_ne,
              gInp_sH,
              gInp_sW,
              inp_H,
              inp_W,
              ne * gOut);
          safe_add_2d(
              gInp_ptr_NC,
              iy_sw,
              ix_sw,
              gInp_sH,
              gInp_sW,
              inp_H,
              inp_W,
              sw * gOut);
          safe_add_2d(
              gInp_ptr_NC,
              iy_se,
              ix_se,
              gInp_sH,
              gInp_sW,
              inp_H,
              inp_W,
              se * gOut);
        }

        // calculate grad_grid
        if (within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W)) {
          scalar_t nw_val = inp_ptr_NC[iy_nw * inp_sH + ix_nw * inp_sW];
          gix -= nw_val * (iy_se - iy) * gOut;
          giy -= nw_val * (ix_se - ix) * gOut;
        }
        if (within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W)) {
          scalar_t ne_val = inp_ptr_NC[iy_ne * inp_sH + ix_ne * inp_sW];
          gix += ne_val * (iy_sw - iy) * gOut;
          giy -= ne_val * (ix - ix_sw) * gOut;
        }
        if (within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W)) {
          scalar_t sw_val = inp_ptr_NC[iy_sw * inp_sH + ix_sw * inp_sW];
          gix -= sw_val * (iy - iy_ne) * gOut;
          giy += sw_val * (ix_ne - ix) * gOut;
        }
        if (within_bounds_2d(iy_se, ix_se, inp_H, inp_W)) {
          scalar_t se_val = inp_ptr_NC[iy_se * inp_sH + ix_se * inp_sW];
          gix += se_val * (iy - iy_nw) * gOut;
          giy += se_val * (ix - ix_nw) * gOut;
        }
      }

      // assuming grad_grid is contiguous
      // thus we can
      //   1. use index with gGrid_sW to directly compute gGrid_ptr_NHW
      //   2. directly assign to gGrid_ptr_NHW[0], gGrid_ptr_NHW[1]
      scalar_t* gGrid_ptr_NHW = grad_grid_data + index * gGrid_sW;
      gGrid_ptr_NHW[0] = gix_mult * gix;
      gGrid_ptr_NHW[1] = giy_mult * giy;
    } else if (interpolation_mode == GridSamplerInterpolation::Nearest) {
      if (input_requires_grad) {
        index_t ix_nearest = static_cast<index_t>(std::nearbyint(ix));
        index_t iy_nearest = static_cast<index_t>(std::nearbyint(iy));

        // assign nearest neighor pixel value to output pixel
        scalar_t* gOut_ptr_NCHW =
            grad_output_data + n * gOut_sN + h * gOut_sH + w * gOut_sW;
        scalar_t* gInp_ptr_NC = grad_input_data + n * gInp_sN;
        for (index_t c = 0; c < C;
             ++c, gInp_ptr_NC += gInp_sC, gOut_ptr_NCHW += gOut_sC) {
          // calculate and set grad_input
          safe_add_2d(
              gInp_ptr_NC,
              iy_nearest,
              ix_nearest,
              gInp_sH,
              gInp_sW,
              inp_H,
              inp_W,
              *gOut_ptr_NCHW);
        }
      }

      // assuming grad_grid is contiguous
      // thus we can
      //   1. use index with gGrid_sW to directly compute gGrid_ptr_NHW
      //   2. directly assign to gGrid_ptr_NHW[0], gGrid_ptr_NHW[1]
      scalar_t* gGrid_ptr_NHW = grad_grid_data + index * gGrid_sW;
      gGrid_ptr_NHW[0] = static_cast<scalar_t>(0);
      gGrid_ptr_NHW[1] = static_cast<scalar_t>(0);
    } else if (interpolation_mode == GridSamplerInterpolation::Bicubic) {
      ix =
          grid_sampler_unnormalize_set_grad(x, inp_W, align_corners, &gix_mult);
      iy =
          grid_sampler_unnormalize_set_grad(y, inp_H, align_corners, &giy_mult);

      scalar_t ix_nw = std::floor(ix);
      scalar_t iy_nw = std::floor(iy);

      const scalar_t tx = ix - ix_nw;
      const scalar_t ty = iy - iy_nw;

      scalar_t x_coeffs[4];
      scalar_t y_coeffs[4];
      scalar_t x_coeffs_grad[4];
      scalar_t y_coeffs_grad[4];

      get_cubic_upsample_coefficients<scalar_t>(x_coeffs, tx);
      get_cubic_upsample_coefficients<scalar_t>(y_coeffs, ty);
      get_cubic_coefficients_grad<scalar_t>(x_coeffs_grad, tx);
      get_cubic_coefficients_grad<scalar_t>(y_coeffs_grad, ty);

      scalar_t gix = static_cast<scalar_t>(0);
      scalar_t giy = static_cast<scalar_t>(0);

      scalar_t* gOut_ptr_NCHW =
          grad_output.data + n * gOut_sN + h * gOut_sH + w * gOut_sW;
      index_t NC_offset = n * gInp_sN;
      scalar_t* inp_ptr_NC = input.data + n * inp_sN;

      for (index_t c = 0; c < C; ++c,
                   gOut_ptr_NCHW += gOut_sC,
                   NC_offset += gInp_sC,
                   inp_ptr_NC += inp_sC) {
        scalar_t gOut = *gOut_ptr_NCHW;

#pragma unroll 4
        for (index_t i = 0; i < 4; ++i) {
#pragma unroll 4
          for (index_t j = 0; j < 4; ++j) {
            if (input_requires_grad) {
              add_value_bounded<scalar_t>(
                  grad_input.data,
                  ix_nw - 1 + i,
                  iy_nw - 1 + j,
                  inp_W,
                  inp_H,
                  gInp_sW,
                  gInp_sH,
                  gOut * x_coeffs[i] * y_coeffs[j],
                  padding_mode,
                  align_corners);
            }

            // set grid gradient
            scalar_t val = get_value_bounded<scalar_t>(
                inp_ptr_NC,
                ix_nw - 1 + i,
                iy_nw - 1 + j,
                inp_W,
                inp_H,
                inp_sW,
                inp_sH,
                padding_mode,
                align_corners);

            gix -= val * x_coeffs_grad[i] * y_coeffs[j] * gOut;
            giy -= val * y_coeffs_grad[j] * x_coeffs[i] * gOut;
          }
        }
      }

      scalar_t* gGrid_ptr_NHW = grad_grid.data + index * gGrid_sW;
      gGrid_ptr_NHW[0] = gix_mult * gix;
      gGrid_ptr_NHW[1] = giy_mult * giy;
    }
  }
  GridSampler2dBackwardKernelFunctor(
      const index_t nthreads_,
      TensorInfo<scalar_t, index_t> grad_output_,
      TensorInfo<scalar_t, index_t> input_,
      TensorInfo<scalar_t, index_t> grid_,
      TensorInfo<scalar_t, index_t> grad_input_,
      TensorInfo<scalar_t, index_t> grad_grid_,
      const GridSamplerInterpolation interpolation_mode_,
      const GridSamplerPadding padding_mode_,
      bool align_corners_,
      const bool input_requires_grad_,
      index_t C_,
      index_t inp_H_,
      index_t inp_W_,
      index_t out_H_,
      index_t out_W_,
      index_t inp_sN_,
      index_t inp_sC_,
      index_t inp_sH_,
      index_t inp_sW_,
      index_t grid_sN_,
      index_t grid_sH_,
      index_t grid_sW_,
      index_t grid_sCoor_,
      index_t gOut_sN_,
      index_t gOut_sC_,
      index_t gOut_sH_,
      index_t gOut_sW_,
      index_t gInp_sN_,
      index_t gInp_sC_,
      index_t gInp_sH_,
      index_t gInp_sW_,
      index_t gGrid_sW_,
      scalar_t* grid_data_,
      scalar_t* input_data_,
      scalar_t* grad_output_data_,
      scalar_t* grad_input_data_,
      scalar_t* grad_grid_data_)
      : nthreads(nthreads_),
        grad_output(grad_output_),
        input(input_),
        grid(grid_),
        grad_input(grad_input_),
        grad_grid(grad_grid_),
        interpolation_mode(interpolation_mode_),
        padding_mode(padding_mode_),
        align_corners(align_corners_),
        input_requires_grad(input_requires_grad_),
        C(C_),
        inp_H(inp_H_),
        inp_W(inp_W_),
        out_H(out_H_),
        out_W(out_W_),
        inp_sN(inp_sN_),
        inp_sC(inp_sC_),
        inp_sH(inp_sH_),
        inp_sW(inp_sW_),
        grid_sN(grid_sN_),
        grid_sH(grid_sH_),
        grid_sW(grid_sW_),
        grid_sCoor(grid_sCoor_),
        gOut_sN(gOut_sN_),
        gOut_sC(gOut_sC_),
        gOut_sH(gOut_sH_),
        gOut_sW(gOut_sW_),
        gInp_sN(gInp_sN_),
        gInp_sC(gInp_sC_),
        gInp_sH(gInp_sH_),
        gInp_sW(gInp_sW_),
        gGrid_sW(gGrid_sW_),
        grid_data(grid_data_),
        input_data(input_data_),
        grad_output_data(grad_output_data_),
        grad_input_data(grad_input_data_),
        grad_grid_data(grad_grid_data_) {}

 private:
  const index_t nthreads;
  TensorInfo<scalar_t, index_t> grad_output;
  TensorInfo<scalar_t, index_t> input;
  TensorInfo<scalar_t, index_t> grid;
  TensorInfo<scalar_t, index_t> grad_input;
  TensorInfo<scalar_t, index_t> grad_grid;
  const GridSamplerInterpolation interpolation_mode;
  const GridSamplerPadding padding_mode;
  bool align_corners;
  const bool input_requires_grad;
  index_t C;
  index_t inp_H;
  index_t inp_W;
  index_t out_H;
  index_t out_W;
  index_t inp_sN;
  index_t inp_sC;
  index_t inp_sH;
  index_t inp_sW;
  index_t grid_sN;
  index_t grid_sH;
  index_t grid_sW;
  index_t grid_sCoor;
  index_t gOut_sN;
  index_t gOut_sC;
  index_t gOut_sH;
  index_t gOut_sW;
  index_t gInp_sN;
  index_t gInp_sC;
  index_t gInp_sH;
  index_t gInp_sW;
  index_t gGrid_sW;
  scalar_t* grid_data;
  scalar_t* input_data;
  scalar_t* grad_output_data;
  scalar_t* grad_input_data;
  scalar_t* grad_grid_data;
};

template <typename scalar_t, typename index_t>
void grid_sampler_2d_backward_template(
    const index_t nthreads,
    TensorInfo<scalar_t, index_t> grad_output,
    TensorInfo<scalar_t, index_t> input,
    TensorInfo<scalar_t, index_t> grid,
    TensorInfo<scalar_t, index_t> grad_input, // initialized to zeros
    // (or unused if input_requires_grad is false)
    TensorInfo<scalar_t, index_t> grad_grid, // initialized to empty
    const GridSamplerInterpolation interpolation_mode,
    const GridSamplerPadding padding_mode,
    bool align_corners,
    const bool input_requires_grad) {
  auto& queue = getCurrentSYCLQueue();
  const auto wgroup_size = syclMaxWorkGroupSize();
  const auto ngroups = (nthreads + wgroup_size - 1) / wgroup_size;

  index_t C = input.sizes[1];
  index_t inp_H = input.sizes[2];
  index_t inp_W = input.sizes[3];
  index_t out_H = grid.sizes[1];
  index_t out_W = grid.sizes[2];
  index_t inp_sN = input.strides[0];
  index_t inp_sC = input.strides[1];
  index_t inp_sH = input.strides[2];
  index_t inp_sW = input.strides[3];
  index_t grid_sN = grid.strides[0];
  index_t grid_sH = grid.strides[1];
  index_t grid_sW = grid.strides[2];
  index_t grid_sCoor = grid.strides[3];
  index_t gOut_sN = grad_output.strides[0];
  index_t gOut_sC = grad_output.strides[1];
  index_t gOut_sH = grad_output.strides[2];
  index_t gOut_sW = grad_output.strides[3];
  // gInp_* are not really needed if input_requires_grad
  // is false.
  index_t gInp_sN;
  index_t gInp_sC;
  index_t gInp_sH;
  index_t gInp_sW;
  if (input_requires_grad) {
    gInp_sN = grad_input.strides[0];
    gInp_sC = grad_input.strides[1];
    gInp_sH = grad_input.strides[2];
    gInp_sW = grad_input.strides[3];
  }
  index_t gGrid_sW = grad_grid.strides[2];

  auto grid_data = grid.data;
  auto input_data = input.data;
  auto grad_output_data = grad_output.data;
  auto grad_input_data = grad_input.data;
  auto grad_grid_data = grad_grid.data;

  GridSampler2dBackwardKernelFunctor<scalar_t, index_t> kfn(
      nthreads,
      grad_output,
      input,
      grid,
      grad_input,
      grad_grid,
      interpolation_mode,
      padding_mode,
      align_corners,
      input_requires_grad,
      C,
      inp_H,
      inp_W,
      out_H,
      out_W,
      inp_sN,
      inp_sC,
      inp_sH,
      inp_sW,
      grid_sN,
      grid_sH,
      grid_sW,
      grid_sCoor,
      gOut_sN,
      gOut_sC,
      gOut_sH,
      gOut_sW,
      gInp_sN,
      gInp_sC,
      gInp_sH,
      gInp_sW,
      gGrid_sW,
      grid_data,
      input_data,
      grad_output_data,
      grad_input_data,
      grad_grid_data);
  sycl_kernel_submit(
      sycl::range<1>(ngroups * wgroup_size),
      sycl::range<1>(wgroup_size),
      queue,
      kfn);
}

void grid_sampler_2d_backward_kernel(
    const Tensor& grad_input,
    const Tensor& grad_grid,
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& grid,
    int64_t interpolation_mode,
    int64_t padding_mode,
    bool align_corners,
    std::array<bool, 2> output_mask) {
  check_grid_sampler_common(input, grid);
  check_grid_sampler_2d(input, grid);

  globalContext().alertNotDeterministic("grid_sampler_2d_backward_xpu");
  auto N = input.size(0);
  auto H = grid.size(1);
  auto W = grid.size(2);
  auto input_requires_grad = output_mask[0];
  int64_t count = N * H * W;
  if (count > 0) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::BFloat16,
        at::ScalarType::Half,
        input.scalar_type(),
        "grid_sampler_2d_backward_xpu",
        [&] {
          if (canUse32BitIndexMath(input) && canUse32BitIndexMath(grid) &&
              canUse32BitIndexMath(grad_output)) {
            grid_sampler_2d_backward_template<scalar_t>(
                static_cast<int>(count),
                getTensorInfo<scalar_t, int>(grad_output),
                getTensorInfo<scalar_t, int>(input),
                getTensorInfo<scalar_t, int>(grid),
                input_requires_grad ? getTensorInfo<scalar_t, int>(grad_input)
                                    : TensorInfo<scalar_t, int>(),
                getTensorInfo<scalar_t, int>(grad_grid),
                static_cast<GridSamplerInterpolation>(interpolation_mode),
                static_cast<GridSamplerPadding>(padding_mode),
                align_corners,
                input_requires_grad);
          } else {
            grid_sampler_2d_backward_template<scalar_t>(
                count,
                getTensorInfo<scalar_t, int64_t>(grad_output),
                getTensorInfo<scalar_t, int64_t>(input),
                getTensorInfo<scalar_t, int64_t>(grid),
                input_requires_grad
                    ? getTensorInfo<scalar_t, int64_t>(grad_input)
                    : TensorInfo<scalar_t, int64_t>(),
                getTensorInfo<scalar_t, int64_t>(grad_grid),
                static_cast<GridSamplerInterpolation>(interpolation_mode),
                static_cast<GridSamplerPadding>(padding_mode),
                align_corners,
                input_requires_grad);
          }
        });
  }
}

} // namespace at::native::xpu
