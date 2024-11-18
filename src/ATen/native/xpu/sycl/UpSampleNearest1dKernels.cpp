#include <ATen/AccumulateType.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/xpu/sycl/UpSampleNearest1dKernels.h>
#include <comm/Runtime.h>
#include <comm/SYCLContext.h>
#include <comm/SYCLHelpers.h>

#include <ATen/native/xpu/sycl/UpSampleNearest1dKernels.h>

namespace at::native {
namespace xpu {

template <typename scalar_t, typename accscalar_t, typename index_bw_op_t>
struct UpsampleNearest1dBackwardKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    int dst_idx = item.get_global_linear_id();
    if (dst_idx >= dim_c_ * dst_dim_w_)
      return;
    int c = (dst_idx / (dst_dim_w_)) % dim_c_;
    int dst_x = dst_idx % dst_dim_w_;
    // note that we do not want to clamp src_x to src_dim_w, since we might
    // intentionally want to skip in case of scale_factor < 1.0
    int src_x = index_bw_op_(scale_factor_, dst_x, src_dim_w_);
    int src_x_up = index_bw_op_(scale_factor_, dst_x + 1, src_dim_w_);
    for (int b = 0; b < dim_b_; b++) {
      accscalar_t grad = 0;
      int src_idx = b * dim_c_ * src_dim_w_ + c * src_dim_w_ + src_x;
      for (int x = src_x; x < src_x_up; x++) {
        grad += grad_o_[src_idx++];
      }
      grad_i_[dst_idx] = grad;
      dst_idx += dim_c_ * dst_dim_w_;
    }
  }
  UpsampleNearest1dBackwardKernelFunctor(
      int n,
      const scalar_t* grad_o,
      size_t dim_b,
      size_t dim_c,
      size_t src_dim_w,
      size_t dst_dim_w,
      scalar_t* grad_i,
      float scale_factor,
      index_bw_op_t index_bw_op)
      : n_(n),
        grad_o_(grad_o),
        dim_b_(dim_b),
        dim_c_(dim_c),
        src_dim_w_(src_dim_w),
        dst_dim_w_(dst_dim_w),
        grad_i_(grad_i),
        scale_factor_(scale_factor),
        index_bw_op_(index_bw_op) {}

 private:
  int n_;
  const scalar_t* grad_o_;
  size_t dim_b_;
  size_t dim_c_;
  size_t src_dim_w_;
  size_t dst_dim_w_;
  scalar_t* grad_i_;
  float scale_factor_;
  index_bw_op_t index_bw_op_;
};
template <typename scalar_t, typename accscalar_t, typename index_bw_op_t>
void upsample_nearest1d_backward_frame(
    int n,
    const scalar_t* grad_o,
    size_t dim_b,
    size_t dim_c,
    size_t src_dim_w,
    size_t dst_dim_w,
    scalar_t* grad_i,
    float scale_factor,
    index_bw_op_t index_bw_op) {
  auto& queue = at::xpu::getCurrentSYCLQueue();
  auto work_group_size = syclMaxWorkItemsPerEU();
  int global_range =
      (n + work_group_size - 1) / work_group_size * work_group_size;
  auto caller = UpsampleNearest1dBackwardKernelFunctor<
      scalar_t,
      accscalar_t,
      index_bw_op_t>(
      n,
      grad_o,
      dim_b,
      dim_c,
      src_dim_w,
      dst_dim_w,
      grad_i,
      scale_factor,
      index_bw_op);
  sycl_kernel_submit(global_range, work_group_size, queue, caller);
}

void upsample_nearest1d_backward_kernel(
    const Tensor& grad_input,
    const Tensor& grad_output_,
    IntArrayRef output_size,
    IntArrayRef input_size,
    std::optional<double> scales,
    bool is_exact) {
  TensorArg grad_input_arg{grad_input, "grad_input", 1};
  TensorArg grad_output_arg{grad_output_, "grad_output_", 2};
  checkAllSameGPU(__func__, {grad_input_arg, grad_output_arg});
  int output_width = output_size[0];
  int nbatch = input_size[0];
  int channels = input_size[1];
  int input_width = input_size[2];
  Tensor grad_output = grad_output_.contiguous();
  if (grad_input.numel() == 0) {
    return;
  }
  // upsample_nearest1d meta call makes sure `nbatch != 0`
  unsigned int n = grad_input.numel() / nbatch;
  // safe check for int32 indexing; implicitly restrict launch config for kernel
  TORCH_CHECK(grad_input.numel() <= std::numeric_limits<int32_t>::max());
  Tensor grad_input_c = grad_input.contiguous();
  AT_DISPATCH_FLOATING_TYPES_AND3(
      ScalarType::BFloat16,
      ScalarType::Half,
      ScalarType::Byte,
      grad_output.scalar_type(),
      "upsample_nearest1d_backward_xpu",
      [&] {
        using accscalar_t = acc_type_device<scalar_t, kXPU>;
        auto idata = grad_input_c.mutable_data_ptr<scalar_t>();
        auto odata = grad_output.const_data_ptr<scalar_t>();
        const float scale_factor = compute_scales_value_backwards<float>(
            scales, output_width, input_width);
        if (is_exact) {
          upsample_nearest1d_backward_frame<scalar_t, accscalar_t>(
              n,
              odata,
              nbatch,
              channels,
              output_width,
              input_width,
              idata,
              scale_factor,
              NearestExactBwIndexOp());
        } else {
          upsample_nearest1d_backward_frame<scalar_t, accscalar_t>(
              n,
              odata,
              nbatch,
              channels,
              output_width,
              input_width,
              idata,
              scale_factor,
              NearestBwIndexOp());
        }
      });
  if (!grad_input.is_contiguous()) {
    grad_input.copy_(grad_input_c);
  }
}

template <typename scalar_t, typename index_op_t>
struct UpsampleNearest1dKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    int dst_idx = item.get_global_linear_id();
    if (dst_idx >= dim_c_ * dst_dim_w_)
      return;

    int c = (dst_idx / dst_dim_w_) % dim_c_;

    int dst_x = dst_idx % dst_dim_w_;
    int src_x = index_op_(scale_factor_, dst_x, src_dim_w_);

    int src_idx = c * src_dim_w_ + src_x;
    int src_stride = dim_c_ * src_dim_w_;
    int dst_stride = dim_c_ * dst_dim_w_;

    for (int b = 0; b < dim_b_; b++) {
      output_[dst_idx] = input_[src_idx];
      src_idx += src_stride;
      dst_idx += dst_stride;
    }
  }
  UpsampleNearest1dKernelFunctor(
      int n,
      const scalar_t* input,
      size_t dim_b,
      size_t dim_c,
      size_t src_dim_w,
      size_t dst_dim_w,
      scalar_t* output,
      float scale_factor,
      index_op_t index_op)
      : n_(n),
        input_(input),
        dim_b_(dim_b),
        dim_c_(dim_c),
        src_dim_w_(src_dim_w),
        dst_dim_w_(dst_dim_w),
        output_(output),
        scale_factor_(scale_factor),
        index_op_(index_op) {}

 private:
  int n_;
  const scalar_t* input_;
  size_t dim_b_;
  size_t dim_c_;
  size_t src_dim_w_;
  size_t dst_dim_w_;
  scalar_t* output_;
  float scale_factor_;
  index_op_t index_op_;
};

template <typename scalar_t, typename index_op_t>
void upsample_nearest1d_frame(
    int n,
    const scalar_t* input,
    size_t dim_b,
    size_t dim_c,
    size_t src_dim_w,
    size_t dst_dim_w,
    scalar_t* output,
    float scale_factor,
    index_op_t index_op) {
  auto& queue = at::xpu::getCurrentSYCLQueue();

  auto work_group_size = syclMaxWorkItemsPerEU();
  int global_range =
      (n + work_group_size - 1) / work_group_size * work_group_size;

  auto kfn = UpsampleNearest1dKernelFunctor<scalar_t, index_op_t>(
      n,
      input,
      dim_b,
      dim_c,
      src_dim_w,
      dst_dim_w,
      output,
      scale_factor,
      index_op);

  sycl_kernel_submit(global_range, work_group_size, queue, kfn);
}

void upsample_nearest1d_kernel(
    const Tensor& output,
    const Tensor& input_,
    IntArrayRef output_size,
    c10::optional<double> scales,
    bool is_exact) {
  TensorArg input_arg{input_, "input_", 1};
  TensorArg output_arg{output, "output", 2};
  checkAllSameGPU(__func__, {input_arg, output_arg});
  int output_width = output_size[0];

  int nbatch = input_.size(0);
  int channels = input_.size(1);
  int input_width = input_.size(2);

  Tensor input = input_.contiguous();

  if (input.numel() == 0) {
    return;
  }
  // upsample_nearest1d meta call makes sure `nbatch != 0`
  unsigned int n = output.numel() / nbatch;
  // safe check for int32 indexing; implicitly restrict launch config for
  // kernel
  TORCH_CHECK(output.numel() <= std::numeric_limits<int32_t>::max());
  Tensor output_c = output.contiguous();

  AT_DISPATCH_FLOATING_TYPES_AND3(
      ScalarType::BFloat16,
      ScalarType::Half,
      ScalarType::Byte,
      input.scalar_type(),
      "upsample_nearest1d_xpu",
      [&] {
        auto idata = input.const_data_ptr<scalar_t>();
        auto odata = output_c.mutable_data_ptr<scalar_t>();

        const float scale_factor =
            compute_scales_value<float>(scales, input_width, output_width);
        if (is_exact) {
          upsample_nearest1d_frame<scalar_t>(
              n,
              idata,
              nbatch,
              channels,
              input_width,
              output_width,
              odata,
              scale_factor,
              NearestExactIndexOp());
        } else {
          upsample_nearest1d_frame<scalar_t>(
              n,
              idata,
              nbatch,
              channels,
              input_width,
              output_width,
              odata,
              scale_factor,
              NearestIndexOp());
        }
      });
  if (!output.is_contiguous()) {
    output.copy_(output_c);
  }
}
} // namespace xpu
} // namespace at::native
