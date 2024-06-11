#include "GridSampler.h"
#include <ATen/ATen.h>
#include <ATen/OpMathType.h>
#include <ATen/native/CanUse32BitIndexMath.h>
#include <ATen/native/GridSamplerUtils.h>
#include <comm/SYCLContext.h>
#include <comm/TensorInfo.h>
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

    opmath_t ix = grid_sampler_compute_source_index(
        x, inp_W_, padding_mode_, align_corners_);
    opmath_t iy = grid_sampler_compute_source_index(
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
              get_value_bounded<scalar_t>(
                  inp_ptr_NC,
                  ix_nw - 1,
                  iy_nw - 1 + i,
                  inp_W_,
                  inp_H_,
                  inp_sW_,
                  inp_sH_,
                  padding_mode_,
                  align_corners_),
              get_value_bounded<scalar_t>(
                  inp_ptr_NC,
                  ix_nw + 0,
                  iy_nw - 1 + i,
                  inp_W_,
                  inp_H_,
                  inp_sW_,
                  inp_sH_,
                  padding_mode_,
                  align_corners_),
              get_value_bounded<scalar_t>(
                  inp_ptr_NC,
                  ix_nw + 1,
                  iy_nw - 1 + i,
                  inp_W_,
                  inp_H_,
                  inp_sW_,
                  inp_sH_,
                  padding_mode_,
                  align_corners_),
              get_value_bounded<scalar_t>(
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

} // namespace at::native::xpu