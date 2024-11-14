#pragma clang diagnostic push
#pragma GCC diagnostic push
// Avoid SYCL compiler return-type error
#pragma clang diagnostic ignored "-Wreturn-type"
#pragma GCC diagnostic ignored "-Wreturn-type"

#include <ATen/OpMathType.h>
#include <ATen/native/CanUse32BitIndexMath.h>
#include <ATen/native/GridSamplerUtils.h>
#include <comm/SYCLContext.h>
#include <comm/TensorInfo.h>
#include <comm/xpu_aten.h>

#include <ATen/native/xpu/UpSample.h>
#include <ATen/native/xpu/sycl/GridSampler.h>

#include <ATen/native/xpu/sycl/GridSamplerKernels.h>

namespace at::native::xpu {

using namespace at::xpu::detail;

template <typename scalar_t, typename index_t>
struct GridSampler2dKernelFunctor {
  using opmath_t = at::opmath_type<scalar_t>;
  void operator()(sycl::nd_item<1> item) const {
    auto index = item.get_global_linear_id();
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
      TensorInfo<const scalar_t, index_t> input,
      TensorInfo<const scalar_t, index_t> grid,
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
  TensorInfo<const scalar_t, index_t> input_;
  TensorInfo<const scalar_t, index_t> grid_;
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
    TensorInfo<const scalar_t, index_t> input,
    TensorInfo<const scalar_t, index_t> grid,
    TensorInfo<scalar_t, index_t> output,
    const GridSamplerInterpolation interpolation_mode,
    const GridSamplerPadding padding_mode,
    bool align_corners) {
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

  const auto wgroup_size = syclMaxWorkGroupSize(kfn);
  const auto ngroups = (nthreads + wgroup_size - 1) / wgroup_size;
  auto& queue = getCurrentSYCLQueue();

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
                getTensorInfo<const scalar_t, int>(input),
                getTensorInfo<const scalar_t, int>(grid),
                getTensorInfo<scalar_t, int>(output),
                static_cast<GridSamplerInterpolation>(interpolation_mode),
                static_cast<GridSamplerPadding>(padding_mode),
                align_corners);
          } else {
            grid_sampler_2d_forward_template<scalar_t>(
                count,
                getTensorInfo<const scalar_t, int64_t>(input),
                getTensorInfo<const scalar_t, int64_t>(grid),
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
  void operator()(sycl::nd_item<1> item) const {
    auto index = item.get_global_linear_id();
    if (index >= nthreads_)
      return;
    const index_t w = index % out_W_;
    const index_t h = (index / out_W_) % out_H_;
    const index_t n = index / (out_H_ * out_W_);
    const auto grid_offset = n * grid_sN_ + h * grid_sH_ + w * grid_sW_;

    // get the corresponding input x, y co-ordinates from grid
    scalar_t x = grid_.data[grid_offset];
    scalar_t y = grid_.data[grid_offset + grid_sCoor_];

    // multipliers for gradients on ix and iy
    scalar_t gix_mult, giy_mult;
    scalar_t ix = at::native::xpu::grid_sampler_compute_source_index_set_grad(
        x, inp_W_, padding_mode_, align_corners_, &gix_mult);
    scalar_t iy = at::native::xpu::grid_sampler_compute_source_index_set_grad(
        y, inp_H_, padding_mode_, align_corners_, &giy_mult);

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
      scalar_t nw = (ix_se - ix) * (iy_se - iy);
      scalar_t ne = (ix - ix_sw) * (iy_sw - iy);
      scalar_t sw = (ix_ne - ix) * (iy - iy_ne);
      scalar_t se = (ix - ix_nw) * (iy - iy_nw);

      scalar_t gix = static_cast<scalar_t>(0), giy = static_cast<scalar_t>(0);
      const scalar_t* gOut_ptr_NCHW =
          grad_output_.data + n * gOut_sN_ + h * gOut_sH_ + w * gOut_sW_;
      index_t NC_offset = n * gInp_sN_;
      const scalar_t* inp_ptr_NC = input_.data + n * inp_sN_;
      for (index_t c = 0; c < C_; ++c,
                   inp_ptr_NC += inp_sC_,
                   NC_offset += gInp_sC_,
                   gOut_ptr_NCHW += gOut_sC_) {
        const scalar_t gOut = *gOut_ptr_NCHW;

        if (input_requires_grad_) {
          // calculate and set grad_input
          at::native::xpu::safe_add_2d(
              grad_input_.data,
              iy_nw,
              ix_nw,
              gInp_sH_,
              gInp_sW_,
              inp_H_,
              inp_W_,
              nw * gOut,
              NC_offset);
          at::native::xpu::safe_add_2d(
              grad_input_.data,
              iy_ne,
              ix_ne,
              gInp_sH_,
              gInp_sW_,
              inp_H_,
              inp_W_,
              ne * gOut,
              NC_offset);
          at::native::xpu::safe_add_2d(
              grad_input_.data,
              iy_sw,
              ix_sw,
              gInp_sH_,
              gInp_sW_,
              inp_H_,
              inp_W_,
              sw * gOut,
              NC_offset);
          at::native::xpu::safe_add_2d(
              grad_input_.data,
              iy_se,
              ix_se,
              gInp_sH_,
              gInp_sW_,
              inp_H_,
              inp_W_,
              se * gOut,
              NC_offset);
        }

        // calculate grad_grid
        if (within_bounds_2d(iy_nw, ix_nw, inp_H_, inp_W_)) {
          scalar_t nw_val = inp_ptr_NC[iy_nw * inp_sH_ + ix_nw * inp_sW_];
          gix -= nw_val * (iy_se - iy) * gOut;
          giy -= nw_val * (ix_se - ix) * gOut;
        }
        if (within_bounds_2d(iy_ne, ix_ne, inp_H_, inp_W_)) {
          scalar_t ne_val = inp_ptr_NC[iy_ne * inp_sH_ + ix_ne * inp_sW_];
          gix += ne_val * (iy_sw - iy) * gOut;
          giy -= ne_val * (ix - ix_sw) * gOut;
        }
        if (within_bounds_2d(iy_sw, ix_sw, inp_H_, inp_W_)) {
          scalar_t sw_val = inp_ptr_NC[iy_sw * inp_sH_ + ix_sw * inp_sW_];
          gix -= sw_val * (iy - iy_ne) * gOut;
          giy += sw_val * (ix_ne - ix) * gOut;
        }
        if (within_bounds_2d(iy_se, ix_se, inp_H_, inp_W_)) {
          scalar_t se_val = inp_ptr_NC[iy_se * inp_sH_ + ix_se * inp_sW_];
          gix += se_val * (iy - iy_nw) * gOut;
          giy += se_val * (ix - ix_nw) * gOut;
        }
      }

      // assuming grad_grid is contiguous
      // thus we can
      //   1. use index with gGrid_sW to directly compute gGrid_ptr_NHW
      //   2. directly assign to gGrid_ptr_NHW[0], gGrid_ptr_NHW[1]
      scalar_t* gGrid_ptr_NHW = grad_grid_.data + index * gGrid_sW_;
      gGrid_ptr_NHW[0] = gix_mult * gix;
      gGrid_ptr_NHW[1] = giy_mult * giy;
    } else if (interpolation_mode_ == GridSamplerInterpolation::Nearest) {
      if (input_requires_grad_) {
        index_t ix_nearest = static_cast<index_t>(std::nearbyint(ix));
        index_t iy_nearest = static_cast<index_t>(std::nearbyint(iy));

        // assign nearest neighor pixel value to output pixel
        const scalar_t* gOut_ptr_NCHW =
            grad_output_.data + n * gOut_sN_ + h * gOut_sH_ + w * gOut_sW_;
        index_t NC_offset = n * gInp_sN_;
        for (index_t c = 0; c < C_;
             ++c, NC_offset += gInp_sC_, gOut_ptr_NCHW += gOut_sC_) {
          // calculate and set grad_input
          at::native::xpu::safe_add_2d(
              grad_input_.data,
              iy_nearest,
              ix_nearest,
              gInp_sH_,
              gInp_sW_,
              inp_H_,
              inp_W_,
              *gOut_ptr_NCHW,
              NC_offset);
        }
      }

      // assuming grad_grid is contiguous
      // thus we can
      //   1. use index with gGrid_sW to directly compute gGrid_ptr_NHW
      //   2. directly assign to gGrid_ptr_NHW[0], gGrid_ptr_NHW[1]
      scalar_t* gGrid_ptr_NHW = grad_grid_.data + index * gGrid_sW_;
      gGrid_ptr_NHW[0] = static_cast<scalar_t>(0);
      gGrid_ptr_NHW[1] = static_cast<scalar_t>(0);
    } else if (interpolation_mode_ == GridSamplerInterpolation::Bicubic) {
      ix = grid_sampler_unnormalize_set_grad(
          x, inp_W_, align_corners_, &gix_mult);
      iy = grid_sampler_unnormalize_set_grad(
          y, inp_H_, align_corners_, &giy_mult);

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

      const scalar_t* gOut_ptr_NCHW =
          grad_output_.data + n * gOut_sN_ + h * gOut_sH_ + w * gOut_sW_;
      index_t NC_offset = n * gInp_sN_;
      const scalar_t* inp_ptr_NC = input_.data + n * inp_sN_;

      for (index_t c = 0; c < C_; ++c,
                   gOut_ptr_NCHW += gOut_sC_,
                   NC_offset += gInp_sC_,
                   inp_ptr_NC += inp_sC_) {
        const scalar_t gOut = *gOut_ptr_NCHW;

#pragma unroll 4
        for (index_t i = 0; i < 4; ++i) {
#pragma unroll 4
          for (index_t j = 0; j < 4; ++j) {
            if (input_requires_grad_) {
              at::native::xpu::add_value_bounded<scalar_t>(
                  grad_input_.data,
                  ix_nw - 1 + i,
                  iy_nw - 1 + j,
                  inp_W_,
                  inp_H_,
                  gInp_sW_,
                  gInp_sH_,
                  gOut * x_coeffs[i] * y_coeffs[j],
                  padding_mode_,
                  align_corners_,
                  NC_offset);
            }

            // set grid gradient
            scalar_t val = get_value_bounded<scalar_t>(
                inp_ptr_NC,
                ix_nw - 1 + i,
                iy_nw - 1 + j,
                inp_W_,
                inp_H_,
                inp_sW_,
                inp_sH_,
                padding_mode_,
                align_corners_);

            gix -= val * x_coeffs_grad[i] * y_coeffs[j] * gOut;
            giy -= val * y_coeffs_grad[j] * x_coeffs[i] * gOut;
          }
        }
      }

      scalar_t* gGrid_ptr_NHW = grad_grid_.data + index * gGrid_sW_;
      gGrid_ptr_NHW[0] = gix_mult * gix;
      gGrid_ptr_NHW[1] = giy_mult * giy;
    }
  }
  GridSampler2dBackwardKernelFunctor(
      const index_t nthreads,
      TensorInfo<const scalar_t, index_t> grad_output,
      TensorInfo<const scalar_t, index_t> input,
      TensorInfo<const scalar_t, index_t> grid,
      TensorInfo<scalar_t, index_t> grad_input,
      TensorInfo<scalar_t, index_t> grad_grid,
      const GridSamplerInterpolation interpolation_mode,
      const GridSamplerPadding padding_mode,
      bool align_corners,
      const bool input_requires_grad,
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
      index_t gOut_sN,
      index_t gOut_sC,
      index_t gOut_sH,
      index_t gOut_sW,
      index_t gInp_sN,
      index_t gInp_sC,
      index_t gInp_sH,
      index_t gInp_sW,
      index_t gGrid_sW)
      : nthreads_(nthreads),
        grad_output_(grad_output),
        input_(input),
        grid_(grid),
        grad_input_(grad_input),
        grad_grid_(grad_grid),
        interpolation_mode_(interpolation_mode),
        padding_mode_(padding_mode),
        align_corners_(align_corners),
        input_requires_grad_(input_requires_grad),
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
        gOut_sN_(gOut_sN),
        gOut_sC_(gOut_sC),
        gOut_sH_(gOut_sH),
        gOut_sW_(gOut_sW),
        gInp_sN_(gInp_sN),
        gInp_sC_(gInp_sC),
        gInp_sH_(gInp_sH),
        gInp_sW_(gInp_sW),
        gGrid_sW_(gGrid_sW) {}

 private:
  const index_t nthreads_;
  TensorInfo<const scalar_t, index_t> grad_output_;
  TensorInfo<const scalar_t, index_t> input_;
  TensorInfo<const scalar_t, index_t> grid_;
  TensorInfo<scalar_t, index_t> grad_input_;
  TensorInfo<scalar_t, index_t> grad_grid_;
  const GridSamplerInterpolation interpolation_mode_;
  const GridSamplerPadding padding_mode_;
  bool align_corners_;
  const bool input_requires_grad_;
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
  index_t gOut_sN_;
  index_t gOut_sC_;
  index_t gOut_sH_;
  index_t gOut_sW_;
  index_t gInp_sN_;
  index_t gInp_sC_;
  index_t gInp_sH_;
  index_t gInp_sW_;
  index_t gGrid_sW_;
};

template <typename scalar_t, typename index_t>
void grid_sampler_2d_backward_template(
    const index_t nthreads,
    TensorInfo<const scalar_t, index_t> grad_output,
    TensorInfo<const scalar_t, index_t> input,
    TensorInfo<const scalar_t, index_t> grid,
    TensorInfo<scalar_t, index_t> grad_input, // initialized to zeros
    // (or unused if input_requires_grad is false)
    TensorInfo<scalar_t, index_t> grad_grid, // initialized to empty
    const GridSamplerInterpolation interpolation_mode,
    const GridSamplerPadding padding_mode,
    bool align_corners,
    const bool input_requires_grad) {
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
      gGrid_sW);

  const auto wgroup_size = syclMaxWorkGroupSize(kfn);
  const auto ngroups = (nthreads + wgroup_size - 1) / wgroup_size;
  auto& queue = getCurrentSYCLQueue();

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
                getTensorInfo<const scalar_t, int>(grad_output),
                getTensorInfo<const scalar_t, int>(input),
                getTensorInfo<const scalar_t, int>(grid),
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
                getTensorInfo<const scalar_t, int64_t>(grad_output),
                getTensorInfo<const scalar_t, int64_t>(input),
                getTensorInfo<const scalar_t, int64_t>(grid),
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

template <typename scalar_t, typename index_t>
struct GridSampler3dKernelFunctor {
  using opmath_t = at::opmath_type<scalar_t>;
  void operator()(sycl::nd_item<1> item) const {
    auto index = item.get_global_linear_id();
    if (index >= nthreads_)
      return;

    const index_t w = index % out_W_;
    const index_t h = (index / out_W_) % out_H_;
    const index_t d = (index / (out_H_ * out_W_)) % out_D_;
    const index_t n = index / (out_D_ * out_H_ * out_W_);
    const index_t grid_offset =
        n * grid_sN_ + d * grid_sD_ + h * grid_sH_ + w * grid_sW_;

    // get the corresponding input_ x, y, z co-ordinates from grid_
    opmath_t ix = grid_.data[grid_offset];
    opmath_t iy = grid_.data[grid_offset + grid_sCoor_];
    opmath_t iz = grid_.data[grid_offset + 2 * grid_sCoor_];

    ix = at::native::xpu::grid_sampler_compute_source_index(
        ix, inp_W_, padding_mode_, align_corners_);
    iy = at::native::xpu::grid_sampler_compute_source_index(
        iy, inp_H_, padding_mode_, align_corners_);
    iz = at::native::xpu::grid_sampler_compute_source_index(
        iz, inp_D_, padding_mode_, align_corners_);

    if (interpolation_mode_ == GridSamplerInterpolation::Bilinear) {
      // get corner pixel values from (x, y, z)
      // for 4d, we used north-east-south-west
      // for 5d, we add top-bottom
      index_t ix_tnw = static_cast<index_t>(std::floor(ix));
      index_t iy_tnw = static_cast<index_t>(std::floor(iy));
      index_t iz_tnw = static_cast<index_t>(std::floor(iz));

      index_t ix_tne = ix_tnw + 1;
      index_t iy_tne = iy_tnw;
      index_t iz_tne = iz_tnw;

      index_t ix_tsw = ix_tnw;
      index_t iy_tsw = iy_tnw + 1;
      index_t iz_tsw = iz_tnw;

      index_t ix_tse = ix_tnw + 1;
      index_t iy_tse = iy_tnw + 1;
      index_t iz_tse = iz_tnw;

      index_t ix_bnw = ix_tnw;
      index_t iy_bnw = iy_tnw;
      index_t iz_bnw = iz_tnw + 1;

      index_t ix_bne = ix_tnw + 1;
      index_t iy_bne = iy_tnw;
      index_t iz_bne = iz_tnw + 1;

      index_t ix_bsw = ix_tnw;
      index_t iy_bsw = iy_tnw + 1;
      index_t iz_bsw = iz_tnw + 1;

      index_t ix_bse = ix_tnw + 1;
      index_t iy_bse = iy_tnw + 1;
      index_t iz_bse = iz_tnw + 1;

      // get surfaces to each neighbor:
      opmath_t tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz);
      opmath_t tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz);
      opmath_t tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz);
      opmath_t tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz);
      opmath_t bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse);
      opmath_t bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw);
      opmath_t bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne);
      opmath_t bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw);

      auto inp_ptr_NC = input_.data + n * inp_sN_;
      auto out_ptr_NCDHW =
          output_.data + n * out_sN_ + d * out_sD_ + h * out_sH_ + w * out_sW_;
      for (index_t c = 0; c < C_;
           ++c, inp_ptr_NC += inp_sC_, out_ptr_NCDHW += out_sC_) {
        //   (c, iz_tnw, iy_tnw, ix_tnw) * tnw + (c, iz_tne, iy_tne, ix_tne) *
        //   tne
        // + (c, iz_tsw, iy_tsw, ix_tsw) * tsw + (c, iz_tse, iy_tse, ix_tse) *
        // tse
        // + (c, iz_bnw, iy_bnw, ix_bnw) * bnw + (c, iz_bne, iy_bne, ix_bne) *
        // bne
        // + (c, iz_bsw, iy_bsw, ix_bsw) * bsw + (c, iz_bse, iy_bse, ix_bse) *
        // bse
        opmath_t out_acc = 0;
        if (within_bounds_3d(iz_tnw, iy_tnw, ix_tnw, inp_D_, inp_H_, inp_W_)) {
          out_acc +=
              inp_ptr_NC
                  [iz_tnw * inp_sD_ + iy_tnw * inp_sH_ + ix_tnw * inp_sW_] *
              tnw;
        }
        if (within_bounds_3d(iz_tne, iy_tne, ix_tne, inp_D_, inp_H_, inp_W_)) {
          out_acc +=
              inp_ptr_NC
                  [iz_tne * inp_sD_ + iy_tne * inp_sH_ + ix_tne * inp_sW_] *
              tne;
        }
        if (within_bounds_3d(iz_tsw, iy_tsw, ix_tsw, inp_D_, inp_H_, inp_W_)) {
          out_acc +=
              inp_ptr_NC
                  [iz_tsw * inp_sD_ + iy_tsw * inp_sH_ + ix_tsw * inp_sW_] *
              tsw;
        }
        if (within_bounds_3d(iz_tse, iy_tse, ix_tse, inp_D_, inp_H_, inp_W_)) {
          out_acc +=
              inp_ptr_NC
                  [iz_tse * inp_sD_ + iy_tse * inp_sH_ + ix_tse * inp_sW_] *
              tse;
        }
        if (within_bounds_3d(iz_bnw, iy_bnw, ix_bnw, inp_D_, inp_H_, inp_W_)) {
          out_acc +=
              inp_ptr_NC
                  [iz_bnw * inp_sD_ + iy_bnw * inp_sH_ + ix_bnw * inp_sW_] *
              bnw;
        }
        if (within_bounds_3d(iz_bne, iy_bne, ix_bne, inp_D_, inp_H_, inp_W_)) {
          out_acc +=
              inp_ptr_NC
                  [iz_bne * inp_sD_ + iy_bne * inp_sH_ + ix_bne * inp_sW_] *
              bne;
        }
        if (within_bounds_3d(iz_bsw, iy_bsw, ix_bsw, inp_D_, inp_H_, inp_W_)) {
          out_acc +=
              inp_ptr_NC
                  [iz_bsw * inp_sD_ + iy_bsw * inp_sH_ + ix_bsw * inp_sW_] *
              bsw;
        }
        if (within_bounds_3d(iz_bse, iy_bse, ix_bse, inp_D_, inp_H_, inp_W_)) {
          out_acc +=
              inp_ptr_NC
                  [iz_bse * inp_sD_ + iy_bse * inp_sH_ + ix_bse * inp_sW_] *
              bse;
        }
        *out_ptr_NCDHW = out_acc;
      }
    } else if (interpolation_mode_ == GridSamplerInterpolation::Nearest) {
      index_t ix_nearest = static_cast<index_t>(std::nearbyint(ix));
      index_t iy_nearest = static_cast<index_t>(std::nearbyint(iy));
      index_t iz_nearest = static_cast<index_t>(std::nearbyint(iz));

      // assign nearest neighor pixel value to output_ pixel
      auto inp_ptr_NC = input_.data + n * inp_sN_;
      auto out_ptr_NCDHW =
          output_.data + n * out_sN_ + d * out_sD_ + h * out_sH_ + w * out_sW_;
      for (index_t c = 0; c < C_;
           ++c, inp_ptr_NC += inp_sC_, out_ptr_NCDHW += out_sC_) {
        if (within_bounds_3d(
                iz_nearest, iy_nearest, ix_nearest, inp_D_, inp_H_, inp_W_)) {
          *out_ptr_NCDHW = inp_ptr_NC
              [iz_nearest * inp_sD_ + iy_nearest * inp_sH_ +
               ix_nearest * inp_sW_];
        } else {
          *out_ptr_NCDHW = static_cast<scalar_t>(0);
        }
      }
    }
  }
  GridSampler3dKernelFunctor(
      const index_t nthreads,
      TensorInfo<const scalar_t, index_t> input,
      TensorInfo<const scalar_t, index_t> grid,
      TensorInfo<scalar_t, index_t> output,
      const GridSamplerInterpolation interpolation_mode,
      const GridSamplerPadding padding_mode,
      const bool align_corners,
      index_t C,
      index_t inp_D,
      index_t inp_H,
      index_t inp_W,
      index_t out_D,
      index_t out_H,
      index_t out_W,
      index_t inp_sN,
      index_t inp_sC,
      index_t inp_sD,
      index_t inp_sH,
      index_t inp_sW,
      index_t grid_sN,
      index_t grid_sD,
      index_t grid_sH,
      index_t grid_sW,
      index_t grid_sCoor,
      index_t out_sN,
      index_t out_sC,
      index_t out_sD,
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
        inp_D_(inp_D),
        inp_H_(inp_H),
        inp_W_(inp_W),
        out_D_(out_D),
        out_H_(out_H),
        out_W_(out_W),
        inp_sN_(inp_sN),
        inp_sC_(inp_sC),
        inp_sD_(inp_sD),
        inp_sH_(inp_sH),
        inp_sW_(inp_sW),
        grid_sN_(grid_sN),
        grid_sD_(grid_sD),
        grid_sH_(grid_sH),
        grid_sW_(grid_sW),
        grid_sCoor_(grid_sCoor),
        out_sN_(out_sN),
        out_sC_(out_sC),
        out_sD_(out_sD),
        out_sH_(out_sH),
        out_sW_(out_sW) {}

 private:
  const index_t nthreads_;
  TensorInfo<const scalar_t, index_t> input_;
  TensorInfo<const scalar_t, index_t> grid_;
  TensorInfo<scalar_t, index_t> output_;
  const GridSamplerInterpolation interpolation_mode_;
  const GridSamplerPadding padding_mode_;
  bool align_corners_;
  index_t C_;
  index_t inp_D_;
  index_t inp_H_;
  index_t inp_W_;
  index_t out_D_;
  index_t out_H_;
  index_t out_W_;
  index_t inp_sN_;
  index_t inp_sC_;
  index_t inp_sD_;
  index_t inp_sH_;
  index_t inp_sW_;
  index_t grid_sN_;
  index_t grid_sD_;
  index_t grid_sH_;
  index_t grid_sW_;
  index_t grid_sCoor_;
  index_t out_sN_;
  index_t out_sC_;
  index_t out_sD_;
  index_t out_sH_;
  index_t out_sW_;
};

template <typename scalar_t, typename index_t>
void grid_sampler_3d_forward_template(
    const index_t nthreads,
    TensorInfo<const scalar_t, index_t> input,
    TensorInfo<const scalar_t, index_t> grid,
    TensorInfo<scalar_t, index_t> output,
    const GridSamplerInterpolation interpolation_mode,
    const GridSamplerPadding padding_mode,
    bool align_corners) {
  index_t C = input.sizes[1];
  index_t inp_D = input.sizes[2];
  index_t inp_H = input.sizes[3];
  index_t inp_W = input.sizes[4];
  index_t out_D = grid.sizes[1];
  index_t out_H = grid.sizes[2];
  index_t out_W = grid.sizes[3];
  index_t inp_sN = input.strides[0];
  index_t inp_sC = input.strides[1];
  index_t inp_sD = input.strides[2];
  index_t inp_sH = input.strides[3];
  index_t inp_sW = input.strides[4];
  index_t grid_sN = grid.strides[0];
  index_t grid_sD = grid.strides[1];
  index_t grid_sH = grid.strides[2];
  index_t grid_sW = grid.strides[3];
  index_t grid_sCoor = grid.strides[4];
  index_t out_sN = output.strides[0];
  index_t out_sC = output.strides[1];
  index_t out_sD = output.strides[2];
  index_t out_sH = output.strides[3];
  index_t out_sW = output.strides[4];

  GridSampler3dKernelFunctor<scalar_t, index_t> kfn(
      nthreads,
      input,
      grid,
      output,
      interpolation_mode,
      padding_mode,
      align_corners,
      C,
      inp_D,
      inp_H,
      inp_W,
      out_D,
      out_H,
      out_W,
      inp_sN,
      inp_sC,
      inp_sD,
      inp_sH,
      inp_sW,
      grid_sN,
      grid_sD,
      grid_sH,
      grid_sW,
      grid_sCoor,
      out_sN,
      out_sC,
      out_sD,
      out_sH,
      out_sW);

  const auto wgroup_size = syclMaxWorkGroupSize(kfn);
  const auto ngroups = (nthreads + wgroup_size - 1) / wgroup_size;
  auto& queue = getCurrentSYCLQueue();

  sycl_kernel_submit(
      sycl::range<1>(ngroups * wgroup_size),
      sycl::range<1>(wgroup_size),
      queue,
      kfn);
}

Tensor grid_sampler_3d_kernel(
    const Tensor& input,
    const Tensor& grid,
    int64_t interpolation_mode,
    int64_t padding_mode,
    bool align_corners) {
  // See NOTE [ grid_sampler Native Functions ].
  // Add checks here in case this is called instead of grid_sampler.
  check_grid_sampler_common(input, grid);
  check_grid_sampler_3d(input, grid, interpolation_mode);

  auto N = input.size(0);
  auto D = grid.size(1);
  auto H = grid.size(2);
  auto W = grid.size(3);
  auto output = at::empty({N, input.size(1), D, H, W}, input.options());
  int64_t count = N * D * H * W;
  if (count > 0) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::BFloat16,
        at::ScalarType::Half,
        input.scalar_type(),
        "grid_sampler_3d_xpu",
        [&] {
          if (canUse32BitIndexMath(input) && canUse32BitIndexMath(grid) &&
              canUse32BitIndexMath(output)) {
            grid_sampler_3d_forward_template<scalar_t>(
                static_cast<int>(count),
                getTensorInfo<const scalar_t, int>(input),
                getTensorInfo<const scalar_t, int>(grid),
                getTensorInfo<scalar_t, int>(output),
                static_cast<GridSamplerInterpolation>(interpolation_mode),
                static_cast<GridSamplerPadding>(padding_mode),
                align_corners);
          } else {
            grid_sampler_3d_forward_template<scalar_t>(
                count,
                getTensorInfo<const scalar_t, int64_t>(input),
                getTensorInfo<const scalar_t, int64_t>(grid),
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
struct GridSampler3dBackwardKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    auto index = item.get_global_linear_id();
    if (index >= nthreads_)
      return;

    const index_t w = index % out_W_;
    const index_t h = (index / out_W_) % out_H_;
    const index_t d = (index / (out_H_ * out_W_)) % out_D_;
    const index_t n = index / (out_D_ * out_H_ * out_W_);
    const auto grid_offset =
        n * grid_sN_ + d * grid_sD_ + h * grid_sH_ + w * grid_sW_;

    // get the corresponding input_ x, y, z co-ordinates from grid_
    scalar_t ix = grid_.data[grid_offset];
    scalar_t iy = grid_.data[grid_offset + grid_sCoor_];
    scalar_t iz = grid_.data[grid_offset + 2 * grid_sCoor_];

    // multipliers for gradients on ix, iy, and iz
    scalar_t gix_mult, giy_mult, giz_mult;
    ix = at::native::xpu::grid_sampler_compute_source_index_set_grad(
        ix, inp_W_, padding_mode_, align_corners_, &gix_mult);
    iy = at::native::xpu::grid_sampler_compute_source_index_set_grad(
        iy, inp_H_, padding_mode_, align_corners_, &giy_mult);
    iz = at::native::xpu::grid_sampler_compute_source_index_set_grad(
        iz, inp_D_, padding_mode_, align_corners_, &giz_mult);

    if (interpolation_mode_ == GridSamplerInterpolation::Bilinear) {
      // get corner pixel values from (x, y, z)
      // for 4d, we used north-east-south-west
      // for 5d, we add top-bottom
      index_t ix_tnw = static_cast<index_t>(std::floor(ix));
      index_t iy_tnw = static_cast<index_t>(std::floor(iy));
      index_t iz_tnw = static_cast<index_t>(std::floor(iz));

      index_t ix_tne = ix_tnw + 1;
      index_t iy_tne = iy_tnw;
      index_t iz_tne = iz_tnw;

      index_t ix_tsw = ix_tnw;
      index_t iy_tsw = iy_tnw + 1;
      index_t iz_tsw = iz_tnw;

      index_t ix_tse = ix_tnw + 1;
      index_t iy_tse = iy_tnw + 1;
      index_t iz_tse = iz_tnw;

      index_t ix_bnw = ix_tnw;
      index_t iy_bnw = iy_tnw;
      index_t iz_bnw = iz_tnw + 1;

      index_t ix_bne = ix_tnw + 1;
      index_t iy_bne = iy_tnw;
      index_t iz_bne = iz_tnw + 1;

      index_t ix_bsw = ix_tnw;
      index_t iy_bsw = iy_tnw + 1;
      index_t iz_bsw = iz_tnw + 1;

      index_t ix_bse = ix_tnw + 1;
      index_t iy_bse = iy_tnw + 1;
      index_t iz_bse = iz_tnw + 1;

      // get surfaces to each neighbor:
      scalar_t tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz);
      scalar_t tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz);
      scalar_t tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz);
      scalar_t tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz);
      scalar_t bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse);
      scalar_t bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw);
      scalar_t bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne);
      scalar_t bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw);

      scalar_t gix = static_cast<scalar_t>(0), giy = static_cast<scalar_t>(0),
               giz = static_cast<scalar_t>(0);
      const scalar_t* gOut_ptr_NCDHW = grad_output_.data + n * gOut_sN_ +
          d * gOut_sD_ + h * gOut_sH_ + w * gOut_sW_;
      index_t NC_offset = n * gInp_sN_;
      const scalar_t* inp_ptr_NC = input_.data + n * inp_sN_;
      // calculate bilinear weighted pixel value and set output pixel
      for (index_t c = 0; c < C_; ++c,
                   gOut_ptr_NCDHW += gOut_sC_,
                   NC_offset += gInp_sC_,
                   inp_ptr_NC += inp_sC_) {
        const scalar_t gOut = *gOut_ptr_NCDHW;

        if (input_requires_grad_) {
          // calculate and set grad_input_
          at::native::xpu::safe_add_3d(
              grad_input_.data,
              iz_tnw,
              iy_tnw,
              ix_tnw,
              gInp_sD_,
              gInp_sH_,
              gInp_sW_,
              inp_D_,
              inp_H_,
              inp_W_,
              tnw * gOut,
              NC_offset);
          at::native::xpu::safe_add_3d(
              grad_input_.data,
              iz_tne,
              iy_tne,
              ix_tne,
              gInp_sD_,
              gInp_sH_,
              gInp_sW_,
              inp_D_,
              inp_H_,
              inp_W_,
              tne * gOut,
              NC_offset);
          at::native::xpu::safe_add_3d(
              grad_input_.data,
              iz_tsw,
              iy_tsw,
              ix_tsw,
              gInp_sD_,
              gInp_sH_,
              gInp_sW_,
              inp_D_,
              inp_H_,
              inp_W_,
              tsw * gOut,
              NC_offset);
          at::native::xpu::safe_add_3d(
              grad_input_.data,
              iz_tse,
              iy_tse,
              ix_tse,
              gInp_sD_,
              gInp_sH_,
              gInp_sW_,
              inp_D_,
              inp_H_,
              inp_W_,
              tse * gOut,
              NC_offset);
          at::native::xpu::safe_add_3d(
              grad_input_.data,
              iz_bnw,
              iy_bnw,
              ix_bnw,
              gInp_sD_,
              gInp_sH_,
              gInp_sW_,
              inp_D_,
              inp_H_,
              inp_W_,
              bnw * gOut,
              NC_offset);
          at::native::xpu::safe_add_3d(
              grad_input_.data,
              iz_bne,
              iy_bne,
              ix_bne,
              gInp_sD_,
              gInp_sH_,
              gInp_sW_,
              inp_D_,
              inp_H_,
              inp_W_,
              bne * gOut,
              NC_offset);
          at::native::xpu::safe_add_3d(
              grad_input_.data,
              iz_bsw,
              iy_bsw,
              ix_bsw,
              gInp_sD_,
              gInp_sH_,
              gInp_sW_,
              inp_D_,
              inp_H_,
              inp_W_,
              bsw * gOut,
              NC_offset);
          at::native::xpu::safe_add_3d(
              grad_input_.data,
              iz_bse,
              iy_bse,
              ix_bse,
              gInp_sD_,
              gInp_sH_,
              gInp_sW_,
              inp_D_,
              inp_H_,
              inp_W_,
              bse * gOut,
              NC_offset);
        }

        // calculate grad_grid_
        if (within_bounds_3d(iz_tnw, iy_tnw, ix_tnw, inp_D_, inp_H_, inp_W_)) {
          scalar_t tnw_val = inp_ptr_NC
              [iz_tnw * inp_sD_ + iy_tnw * inp_sH_ + ix_tnw * inp_sW_];
          gix -= tnw_val * (iy_bse - iy) * (iz_bse - iz) * gOut;
          giy -= tnw_val * (ix_bse - ix) * (iz_bse - iz) * gOut;
          giz -= tnw_val * (ix_bse - ix) * (iy_bse - iy) * gOut;
        }
        if (within_bounds_3d(iz_tne, iy_tne, ix_tne, inp_D_, inp_H_, inp_W_)) {
          scalar_t tne_val = inp_ptr_NC
              [iz_tne * inp_sD_ + iy_tne * inp_sH_ + ix_tne * inp_sW_];
          gix += tne_val * (iy_bsw - iy) * (iz_bsw - iz) * gOut;
          giy -= tne_val * (ix - ix_bsw) * (iz_bsw - iz) * gOut;
          giz -= tne_val * (ix - ix_bsw) * (iy_bsw - iy) * gOut;
        }
        if (within_bounds_3d(iz_tsw, iy_tsw, ix_tsw, inp_D_, inp_H_, inp_W_)) {
          scalar_t tsw_val = inp_ptr_NC
              [iz_tsw * inp_sD_ + iy_tsw * inp_sH_ + ix_tsw * inp_sW_];
          gix -= tsw_val * (iy - iy_bne) * (iz_bne - iz) * gOut;
          giy += tsw_val * (ix_bne - ix) * (iz_bne - iz) * gOut;
          giz -= tsw_val * (ix_bne - ix) * (iy - iy_bne) * gOut;
        }
        if (within_bounds_3d(iz_tse, iy_tse, ix_tse, inp_D_, inp_H_, inp_W_)) {
          scalar_t tse_val = inp_ptr_NC
              [iz_tse * inp_sD_ + iy_tse * inp_sH_ + ix_tse * inp_sW_];
          gix += tse_val * (iy - iy_bnw) * (iz_bnw - iz) * gOut;
          giy += tse_val * (ix - ix_bnw) * (iz_bnw - iz) * gOut;
          giz -= tse_val * (ix - ix_bnw) * (iy - iy_bnw) * gOut;
        }
        if (within_bounds_3d(iz_bnw, iy_bnw, ix_bnw, inp_D_, inp_H_, inp_W_)) {
          scalar_t bnw_val = inp_ptr_NC
              [iz_bnw * inp_sD_ + iy_bnw * inp_sH_ + ix_bnw * inp_sW_];
          gix -= bnw_val * (iy_tse - iy) * (iz - iz_tse) * gOut;
          giy -= bnw_val * (ix_tse - ix) * (iz - iz_tse) * gOut;
          giz += bnw_val * (ix_tse - ix) * (iy_tse - iy) * gOut;
        }
        if (within_bounds_3d(iz_bne, iy_bne, ix_bne, inp_D_, inp_H_, inp_W_)) {
          scalar_t bne_val = inp_ptr_NC
              [iz_bne * inp_sD_ + iy_bne * inp_sH_ + ix_bne * inp_sW_];
          gix += bne_val * (iy_tsw - iy) * (iz - iz_tsw) * gOut;
          giy -= bne_val * (ix - ix_tsw) * (iz - iz_tsw) * gOut;
          giz += bne_val * (ix - ix_tsw) * (iy_tsw - iy) * gOut;
        }
        if (within_bounds_3d(iz_bsw, iy_bsw, ix_bsw, inp_D_, inp_H_, inp_W_)) {
          scalar_t bsw_val = inp_ptr_NC
              [iz_bsw * inp_sD_ + iy_bsw * inp_sH_ + ix_bsw * inp_sW_];
          gix -= bsw_val * (iy - iy_tne) * (iz - iz_tne) * gOut;
          giy += bsw_val * (ix_tne - ix) * (iz - iz_tne) * gOut;
          giz += bsw_val * (ix_tne - ix) * (iy - iy_tne) * gOut;
        }
        if (within_bounds_3d(iz_bse, iy_bse, ix_bse, inp_D_, inp_H_, inp_W_)) {
          scalar_t bse_val = inp_ptr_NC
              [iz_bse * inp_sD_ + iy_bse * inp_sH_ + ix_bse * inp_sW_];
          gix += bse_val * (iy - iy_tnw) * (iz - iz_tnw) * gOut;
          giy += bse_val * (ix - ix_tnw) * (iz - iz_tnw) * gOut;
          giz += bse_val * (ix - ix_tnw) * (iy - iy_tnw) * gOut;
        }
      }

      // assuming grad_grid_ is contiguous
      // thus we can
      //   1. use index with gGrid_sW_ to directly compute gGrid_ptr_NDHW
      //   2. directly assign to gGrid_ptr_NDHW[0], gGrid_ptr_NDHW[1],
      //   gGrid_ptr_NDHW[2]
      scalar_t* gGrid_ptr_NDHW = grad_grid_.data + index * gGrid_sW_;
      gGrid_ptr_NDHW[0] = gix_mult * gix;
      gGrid_ptr_NDHW[1] = giy_mult * giy;
      gGrid_ptr_NDHW[2] = giz_mult * giz;
    } else if (interpolation_mode_ == GridSamplerInterpolation::Nearest) {
      if (input_requires_grad_) {
        auto ix_nearest = static_cast<index_t>(std::round(ix));
        auto iy_nearest = static_cast<index_t>(std::round(iy));
        auto iz_nearest = static_cast<index_t>(std::round(iz));

        // assign nearest neighor pixel value to output pixel
        const scalar_t* gOut_ptr_NCDHW = grad_output_.data + n * gOut_sN_ +
            d * gOut_sD_ + h * gOut_sH_ + w * gOut_sW_;
        index_t NC_offset = n * gInp_sN_;
        for (index_t c = 0; c < C_;
             ++c, gOut_ptr_NCDHW += gOut_sC_, NC_offset += gInp_sC_) {
          // calculate and set grad_input_
          safe_add_3d(
              grad_input_.data,
              iz_nearest,
              iy_nearest,
              ix_nearest,
              gInp_sD_,
              gInp_sH_,
              gInp_sW_,
              inp_D_,
              inp_H_,
              inp_W_,
              *gOut_ptr_NCDHW,
              NC_offset);
        }
      }

      // assuming grad_grid_ is contiguous
      // thus we can
      //   1. use index with gGrid_sW_ to directly compute gGrid_ptr_NDHW
      //   2. directly assign to gGrid_ptr_NDHW[0], gGrid_ptr_NDHW[1],
      //   gGrid_ptr_NDHW[2]
      scalar_t* gGrid_ptr_NDHW = grad_grid_.data + index * gGrid_sW_;
      gGrid_ptr_NDHW[0] = static_cast<scalar_t>(0);
      gGrid_ptr_NDHW[1] = static_cast<scalar_t>(0);
      gGrid_ptr_NDHW[2] = static_cast<scalar_t>(0);
    }
  }
  GridSampler3dBackwardKernelFunctor(
      const index_t nthreads,
      TensorInfo<const scalar_t, index_t> grad_output,
      TensorInfo<const scalar_t, index_t> input,
      TensorInfo<const scalar_t, index_t> grid,
      TensorInfo<scalar_t, index_t> grad_input,
      TensorInfo<scalar_t, index_t> grad_grid,
      const GridSamplerInterpolation interpolation_mode,
      const GridSamplerPadding padding_mode,
      bool align_corners,
      const bool input_requires_grad,
      index_t C,
      index_t inp_D,
      index_t inp_H,
      index_t inp_W,
      index_t out_D,
      index_t out_H,
      index_t out_W,
      index_t inp_sN,
      index_t inp_sC,
      index_t inp_sD,
      index_t inp_sH,
      index_t inp_sW,
      index_t grid_sN,
      index_t grid_sD,
      index_t grid_sH,
      index_t grid_sW,
      index_t grid_sCoor,
      index_t gOut_sN,
      index_t gOut_sC,
      index_t gOut_sD,
      index_t gOut_sH,
      index_t gOut_sW,
      int64_t gInp_sN,
      int64_t gInp_sC,
      int64_t gInp_sD,
      int64_t gInp_sH,
      int64_t gInp_sW,
      index_t gGrid_sW)
      : nthreads_(nthreads),
        grad_output_(grad_output),
        input_(input),
        grid_(grid),
        grad_input_(grad_input),
        grad_grid_(grad_grid),
        interpolation_mode_(interpolation_mode),
        padding_mode_(padding_mode),
        align_corners_(align_corners),
        input_requires_grad_(input_requires_grad),
        C_(C),
        inp_D_(inp_D),
        inp_H_(inp_H),
        inp_W_(inp_W),
        out_D_(out_D),
        out_H_(out_H),
        out_W_(out_W),
        inp_sN_(inp_sN),
        inp_sC_(inp_sC),
        inp_sD_(inp_sD),
        inp_sH_(inp_sH),
        inp_sW_(inp_sW),
        grid_sN_(grid_sN),
        grid_sD_(grid_sD),
        grid_sH_(grid_sH),
        grid_sW_(grid_sW),
        grid_sCoor_(grid_sCoor),
        gOut_sN_(gOut_sN),
        gOut_sC_(gOut_sC),
        gOut_sD_(gOut_sD),
        gOut_sH_(gOut_sH),
        gOut_sW_(gOut_sW),
        gInp_sN_(gInp_sN),
        gInp_sC_(gInp_sC),
        gInp_sD_(gInp_sD),
        gInp_sH_(gInp_sH),
        gInp_sW_(gInp_sW),
        gGrid_sW_(gGrid_sW) {}

 private:
  const index_t nthreads_;
  TensorInfo<const scalar_t, index_t> grad_output_;
  TensorInfo<const scalar_t, index_t> input_;
  TensorInfo<const scalar_t, index_t> grid_;
  TensorInfo<scalar_t, index_t> grad_input_;
  TensorInfo<scalar_t, index_t> grad_grid_;
  const GridSamplerInterpolation interpolation_mode_;
  const GridSamplerPadding padding_mode_;
  bool align_corners_;
  const bool input_requires_grad_;
  index_t C_;
  index_t inp_D_;
  index_t inp_H_;
  index_t inp_W_;
  index_t out_D_;
  index_t out_H_;
  index_t out_W_;
  index_t inp_sN_;
  index_t inp_sC_;
  index_t inp_sD_;
  index_t inp_sH_;
  index_t inp_sW_;
  index_t grid_sN_;
  index_t grid_sD_;
  index_t grid_sH_;
  index_t grid_sW_;
  index_t grid_sCoor_;
  index_t gOut_sN_;
  index_t gOut_sC_;
  index_t gOut_sD_;
  index_t gOut_sH_;
  index_t gOut_sW_;
  int64_t gInp_sN_;
  int64_t gInp_sC_;
  int64_t gInp_sD_;
  int64_t gInp_sH_;
  int64_t gInp_sW_;
  index_t gGrid_sW_;
};

template <typename scalar_t, typename index_t>
void grid_sampler_3d_backward_template(
    const index_t nthreads,
    TensorInfo<const scalar_t, index_t> grad_output,
    TensorInfo<const scalar_t, index_t> input,
    TensorInfo<const scalar_t, index_t> grid,
    TensorInfo<scalar_t, index_t> grad_input, // initialized to zeros
    // (or unused if input_requires_grad is false)
    TensorInfo<scalar_t, index_t> grad_grid, // initialized to empty
    const GridSamplerInterpolation interpolation_mode,
    const GridSamplerPadding padding_mode,
    bool align_corners,
    const bool input_requires_grad) {
  index_t C = input.sizes[1];
  index_t inp_D = input.sizes[2];
  index_t inp_H = input.sizes[3];
  index_t inp_W = input.sizes[4];
  index_t out_D = grid.sizes[1];
  index_t out_H = grid.sizes[2];
  index_t out_W = grid.sizes[3];
  index_t inp_sN = input.strides[0];
  index_t inp_sC = input.strides[1];
  index_t inp_sD = input.strides[2];
  index_t inp_sH = input.strides[3];
  index_t inp_sW = input.strides[4];
  index_t grid_sN = grid.strides[0];
  index_t grid_sD = grid.strides[1];
  index_t grid_sH = grid.strides[2];
  index_t grid_sW = grid.strides[3];
  index_t grid_sCoor = grid.strides[4];
  index_t gOut_sN = grad_output.strides[0];
  index_t gOut_sC = grad_output.strides[1];
  index_t gOut_sD = grad_output.strides[2];
  index_t gOut_sH = grad_output.strides[3];
  index_t gOut_sW = grad_output.strides[4];
  // gInp_* are not really needed if input_requires_grad is false.
  int64_t gInp_sN = 0;
  int64_t gInp_sC = 0;
  int64_t gInp_sD = 0;
  int64_t gInp_sH = 0;
  int64_t gInp_sW = 0;
  if (input_requires_grad) {
    gInp_sN = grad_input.strides[0];
    gInp_sC = grad_input.strides[1];
    gInp_sD = grad_input.strides[2];
    gInp_sH = grad_input.strides[3];
    gInp_sW = grad_input.strides[4];
  }
  index_t gGrid_sW = grad_grid.strides[3];

  GridSampler3dBackwardKernelFunctor<scalar_t, index_t> kfn(
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
      inp_D,
      inp_H,
      inp_W,
      out_D,
      out_H,
      out_W,
      inp_sN,
      inp_sC,
      inp_sD,
      inp_sH,
      inp_sW,
      grid_sN,
      grid_sD,
      grid_sH,
      grid_sW,
      grid_sCoor,
      gOut_sN,
      gOut_sC,
      gOut_sD,
      gOut_sH,
      gOut_sW,
      gInp_sN,
      gInp_sC,
      gInp_sD,
      gInp_sH,
      gInp_sW,
      gGrid_sW);

  const auto wgroup_size = syclMaxWorkGroupSize(kfn);
  const auto ngroups = (nthreads + wgroup_size - 1) / wgroup_size;
  auto& queue = getCurrentSYCLQueue();

  sycl_kernel_submit(
      sycl::range<1>(ngroups * wgroup_size),
      sycl::range<1>(wgroup_size),
      queue,
      kfn);
}

void grid_sampler_3d_backward_kernel(
    const Tensor& grad_input,
    const Tensor& grad_grid,
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& grid,
    int64_t interpolation_mode,
    int64_t padding_mode,
    bool align_corners,
    std::array<bool, 2> output_mask) {
  // See NOTE [ grid_sampler Native Functions ].
  // Add checks here in case this is called instead of grid_sampler.
  check_grid_sampler_common(input, grid);
  check_grid_sampler_3d(input, grid, interpolation_mode);

  globalContext().alertNotDeterministic("grid_sampler_3d_backward_xpu");
  auto input_requires_grad = output_mask[0];
  auto N = input.size(0);
  auto D = grid.size(1);
  auto H = grid.size(2);
  auto W = grid.size(3);
  int64_t count = N * D * H * W;
  if (count > 0) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::BFloat16,
        at::ScalarType::Half,
        input.scalar_type(),
        "grid_sampler_2d_backward_xpu",
        [&] {
          if (canUse32BitIndexMath(input) && canUse32BitIndexMath(grid) &&
              canUse32BitIndexMath(grad_output)) {
            grid_sampler_3d_backward_template<scalar_t>(
                static_cast<int>(count),
                getTensorInfo<const scalar_t, int>(grad_output),
                getTensorInfo<const scalar_t, int>(input),
                getTensorInfo<const scalar_t, int>(grid),
                input_requires_grad ? getTensorInfo<scalar_t, int>(grad_input)
                                    : TensorInfo<scalar_t, int>(),
                getTensorInfo<scalar_t, int>(grad_grid),
                static_cast<GridSamplerInterpolation>(interpolation_mode),
                static_cast<GridSamplerPadding>(padding_mode),
                align_corners,
                input_requires_grad);
          } else {
            grid_sampler_3d_backward_template<scalar_t>(
                count,
                getTensorInfo<const scalar_t, int64_t>(grad_output),
                getTensorInfo<const scalar_t, int64_t>(input),
                getTensorInfo<const scalar_t, int64_t>(grid),
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

#pragma GCC diagnostic pop
#pragma clang diagnostic pop
