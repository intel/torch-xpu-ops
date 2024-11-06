#pragma clang diagnostic push
#pragma GCC diagnostic push
// Avoid SYCL compiler return-type error
#pragma clang diagnostic ignored "-Wreturn-type"
#pragma GCC diagnostic ignored "-Wreturn-type"
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorUtils.h>
#include <ATen/ceil_div.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/empty.h>

#include <ATen/native/xpu/UpSample.h>
#include <ATen/native/xpu/sycl/UpSampleNearest3dKernels.h>
#include <comm/SYCLContext.h>

namespace at::native::xpu {

template <typename scalar_t, typename index_op_t>
struct UpsampleNearest3dKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    int dst_idx = item.get_global_linear_id();

    if (dst_idx >= dim_c_ * dst_dim_d_ * dst_dim_h_ * dst_dim_w_)
      return;

    int dst_c_stride = dst_dim_d_ * dst_dim_h_ * dst_dim_w_;
    int src_c_stride = src_dim_d_ * src_dim_h_ * src_dim_w_;

    int c = (dst_idx / (dst_c_stride)) % dim_c_;

    int dst_z = (dst_idx / dst_dim_h_ / dst_dim_w_) % dst_dim_d_;
    int src_z = index_op_(depth_scale_, dst_z, src_dim_d_);
    int dst_y = (dst_idx / dst_dim_w_) % dst_dim_h_;
    int src_y = index_op_(height_scale_, dst_y, src_dim_h_);

    int dst_x = dst_idx % dst_dim_w_;
    int src_x = index_op_(width_scale_, dst_x, src_dim_w_);

    int src_idx = c * src_c_stride + src_z * src_dim_h_ * src_dim_w_ +
        src_y * src_dim_w_ + src_x;
    for (int b = 0; b < dim_b_; b++) {
      output_[dst_idx] = input_[src_idx];
      src_idx += dim_c_ * src_c_stride;
      dst_idx += dim_c_ * dst_c_stride;
    }
  }
  UpsampleNearest3dKernelFunctor(
      const scalar_t* input,
      size_t dim_b,
      size_t dim_c,
      size_t src_dim_d,
      size_t src_dim_h,
      size_t src_dim_w,
      size_t dst_dim_d,
      size_t dst_dim_h,
      size_t dst_dim_w,
      scalar_t* output,
      float depth_scale,
      float height_scale,
      float width_scale,
      index_op_t index_op)
      : input_(input),
        dim_b_(dim_b),
        dim_c_(dim_c),
        src_dim_d_(src_dim_d),
        src_dim_h_(src_dim_h),
        src_dim_w_(src_dim_w),
        dst_dim_d_(dst_dim_d),
        dst_dim_h_(dst_dim_h),
        dst_dim_w_(dst_dim_w),
        output_(output),
        depth_scale_(depth_scale),
        height_scale_(height_scale),
        width_scale_(width_scale),
        index_op_(index_op) {}

 private:
  const scalar_t* input_;
  size_t dim_b_;
  size_t dim_c_;
  size_t src_dim_d_;
  size_t src_dim_h_;
  size_t src_dim_w_;
  size_t dst_dim_d_;
  size_t dst_dim_h_;
  size_t dst_dim_w_;
  scalar_t* output_;
  float depth_scale_;
  float height_scale_;
  float width_scale_;
  index_op_t index_op_;
};

template <typename scalar_t, typename index_op_t>
void upsample_nearest3d_out_template(
    const scalar_t* input,
    unsigned int n,
    size_t dim_b,
    size_t dim_c,
    size_t src_dim_d,
    size_t src_dim_h,
    size_t src_dim_w,
    size_t dst_dim_d,
    size_t dst_dim_h,
    size_t dst_dim_w,
    scalar_t* output,
    float depth_scale,
    float height_scale,
    float width_scale,
    index_op_t index_op) {
  auto& queue = at::xpu::getCurrentSYCLQueue();
  auto kfn = UpsampleNearest3dKernelFunctor<scalar_t, index_op_t>(
      input,
      dim_b,
      dim_c,
      src_dim_d,
      src_dim_h,
      src_dim_w,
      dst_dim_d,
      dst_dim_h,
      dst_dim_w,
      output,
      depth_scale,
      height_scale,
      width_scale,
      index_op);
  auto work_group_size = syclMaxWorkGroupSize(kfn);
  int64_t work_group_num =
      at::ceil_div((unsigned int)n, (unsigned int)work_group_size);
  sycl_kernel_submit(
      work_group_num * work_group_size, work_group_size, queue, kfn);
}

void upsample_nearest3d_kernel(
    const Tensor& output,
    const Tensor& input_,
    IntArrayRef output_size,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    bool is_exact) {
  TensorArg input_arg{input_, "input_", 1};
  TensorArg output_arg{output, "output", 2};
  checkAllSameGPU(__func__, {input_arg, output_arg});
  if (input_.numel() == 0) {
    return;
  }
  auto output_c = output.is_contiguous()
      ? output
      : at::empty(output.sizes(), output.options());

  int output_depth = output_size[0];
  int output_height = output_size[1];
  int output_width = output_size[2];

  int nbatch = input_.size(0);
  int channels = input_.size(1);
  int input_depth = input_.size(2);
  int input_height = input_.size(3);
  int input_width = input_.size(4);

  Tensor input = input_.contiguous();
  unsigned int n = output.numel() / nbatch;
  TORCH_CHECK(output.numel() <= std::numeric_limits<int32_t>::max());
  AT_DISPATCH_FLOATING_TYPES_AND3(
      ScalarType::Half,
      ScalarType::BFloat16,
      ScalarType::Byte,
      input.scalar_type(),
      "upsample_nearest3d_xpu",
      [&] {
        auto idata = input.const_data_ptr<scalar_t>();
        auto odata = output_c.mutable_data_ptr<scalar_t>();

        const float depth_scale =
            compute_scales_value<float>(scales_d, input_depth, output_depth);
        const float height_scale =
            compute_scales_value<float>(scales_h, input_height, output_height);
        const float width_scale =
            compute_scales_value<float>(scales_w, input_width, output_width);
        if (is_exact) {
          upsample_nearest3d_out_template<scalar_t>(
              idata,
              n,
              nbatch,
              channels,
              input_depth,
              input_height,
              input_width,
              output_depth,
              output_height,
              output_width,
              odata,
              depth_scale,
              height_scale,
              width_scale,
              NearestExactIndexOp());
        } else {
          upsample_nearest3d_out_template<scalar_t>(
              idata,
              n,
              nbatch,
              channels,
              input_depth,
              input_height,
              input_width,
              output_depth,
              output_height,
              output_width,
              odata,
              depth_scale,
              height_scale,
              width_scale,
              NearestIndexOp());
        }
      });
  if (!output.is_contiguous()) {
    output.copy_(output_c);
  }
}

template <typename scalar_t, typename accscalar_t, typename index_bw_op_t>
struct UpsampleNearest3dBackwardFunctor {
  void operator()(sycl::nd_item<1> item) const {
    int dst_idx = item.get_global_linear_id();

    if (dst_idx >= dim_c_ * dst_dim_d_ * dst_dim_h_ * dst_dim_w_)
      return;

    int dst_c_stride = dst_dim_d_ * dst_dim_h_ * dst_dim_w_;
    int src_c_stride = src_dim_d_ * src_dim_h_ * src_dim_w_;

    int c = (dst_idx / (dst_c_stride)) % dim_c_;

    int dst_z = (dst_idx / dst_dim_h_ / dst_dim_w_) % dst_dim_d_;
    int src_z = index_bw_op_(depth_scale_, dst_z, src_dim_d_);
    int src_z_up = index_bw_op_(depth_scale_, dst_z + 1, src_dim_d_);

    int dst_y = (dst_idx / dst_dim_w_) % dst_dim_h_;
    int src_y = index_bw_op_(height_scale_, dst_y, src_dim_h_);
    int src_y_up = index_bw_op_(height_scale_, dst_y + 1, src_dim_h_);

    int dst_x = dst_idx % dst_dim_w_;
    int src_x = index_bw_op_(width_scale_, dst_x, src_dim_w_);
    int src_x_up = index_bw_op_(width_scale_, dst_x + 1, src_dim_w_);

    for (int b = 0; b < dim_b_; b++) {
      accscalar_t grad = 0;
      for (int z = src_z; z < src_z_up; z++) {
        for (int y = src_y; y < src_y_up; y++) {
          for (int x = src_x; x < src_x_up; x++) {
            int src_idx = b * dim_c_ * src_c_stride + c * src_c_stride +
                z * src_dim_h_ * src_dim_w_ + y * src_dim_w_ + x;
            grad += grad_o_[src_idx];
          }
        }
      }
      grad_i_[dst_idx] = grad;
      dst_idx += dim_c_ * dst_c_stride;
    }
  }
  UpsampleNearest3dBackwardFunctor(
      const scalar_t* grad_o,
      size_t dim_b,
      size_t dim_c,
      size_t src_dim_d,
      size_t src_dim_h,
      size_t src_dim_w,
      size_t dst_dim_d,
      size_t dst_dim_h,
      size_t dst_dim_w,
      scalar_t* grad_i,
      float depth_scale,
      float height_scale,
      float width_scale,
      index_bw_op_t index_bw_op)
      : grad_o_(grad_o),
        dim_b_(dim_b),
        dim_c_(dim_c),
        src_dim_d_(src_dim_d),
        src_dim_h_(src_dim_h),
        src_dim_w_(src_dim_w),
        dst_dim_d_(dst_dim_d),
        dst_dim_h_(dst_dim_h),
        dst_dim_w_(dst_dim_w),
        grad_i_(grad_i),
        depth_scale_(depth_scale),
        height_scale_(height_scale),
        width_scale_(width_scale) {}

 private:
  const scalar_t* grad_o_;
  size_t dim_b_;
  size_t dim_c_;
  size_t src_dim_d_;
  size_t src_dim_h_;
  size_t src_dim_w_;
  size_t dst_dim_d_;
  size_t dst_dim_h_;
  size_t dst_dim_w_;
  scalar_t* grad_i_;
  float depth_scale_;
  float height_scale_;
  float width_scale_;
  index_bw_op_t index_bw_op_;
};

template <typename scalar_t, typename accscalar_t, typename index_bw_op_t>
void upsample_nearest3d_backward_template(
    const scalar_t* grad_o,
    unsigned int n,
    size_t dim_b,
    size_t dim_c,
    size_t src_dim_d,
    size_t src_dim_h,
    size_t src_dim_w,
    size_t dst_dim_d,
    size_t dst_dim_h,
    size_t dst_dim_w,
    scalar_t* grad_i,
    float depth_scale,
    float height_scale,
    float width_scale,
    index_bw_op_t index_bw_op) {
  auto& queue = at::xpu::getCurrentSYCLQueue();
  auto kfn =
      UpsampleNearest3dBackwardFunctor<scalar_t, accscalar_t, index_bw_op_t>(
          grad_o,
          dim_b,
          dim_c,
          src_dim_d,
          src_dim_h,
          src_dim_w,
          dst_dim_d,
          dst_dim_h,
          dst_dim_w,
          grad_i,
          depth_scale,
          height_scale,
          width_scale,
          index_bw_op);
  auto work_group_size = syclMaxWorkGroupSize(kfn);
  int64_t work_group_num = at::ceil_div(n, (unsigned int)work_group_size);
  sycl_kernel_submit(
      work_group_num * work_group_size, work_group_size, queue, kfn);
}

void upsample_nearest3d_backward_kernel(
    const Tensor& grad_input,
    const Tensor& grad_output_,
    IntArrayRef output_size,
    IntArrayRef input_size,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    bool is_exact) {
  TensorArg grad_input_arg{grad_input, "grad_input", 1};
  TensorArg grad_output_arg{grad_output_, "grad_output_", 2};
  checkAllSameGPU(__func__, {grad_input_arg, grad_output_arg});

  if (grad_input.numel() == 0) {
    return;
  }

  int output_depth = output_size[0];
  int output_height = output_size[1];
  int output_width = output_size[2];

  int nbatch = input_size[0];
  int channels = input_size[1];
  int input_depth = input_size[2];
  int input_height = input_size[3];
  int input_width = input_size[4];

  Tensor grad_output = grad_output_.contiguous();
  unsigned int n = grad_input.numel() / nbatch;
  TORCH_CHECK(grad_input.numel() <= std::numeric_limits<int32_t>::max());
  TORCH_CHECK(grad_output.numel() <= std::numeric_limits<int32_t>::max());
  AT_DISPATCH_FLOATING_TYPES_AND3(
      ScalarType::Half,
      ScalarType::BFloat16,
      ScalarType::Byte,
      grad_output.scalar_type(),
      "upsample_nearest3d_backward_xpu",
      [&] {
        using accscalar_t = acc_type_device<scalar_t, kXPU>;

        auto idata = grad_input.mutable_data_ptr<scalar_t>();
        auto odata = grad_output.const_data_ptr<scalar_t>();

        float depth_scale = compute_scales_value_backwards<float>(
            scales_d, output_depth, input_depth);
        float height_scale = compute_scales_value_backwards<float>(
            scales_h, output_height, input_height);
        float width_scale = compute_scales_value_backwards<float>(
            scales_w, output_width, input_width);
        if (is_exact) {
          upsample_nearest3d_backward_template<scalar_t, accscalar_t>(
              odata,
              n,
              nbatch,
              channels,
              output_depth,
              output_height,
              output_width,
              input_depth,
              input_height,
              input_width,
              idata,
              depth_scale,
              height_scale,
              width_scale,
              NearestExactBwIndexOp());
        } else {
          upsample_nearest3d_backward_template<scalar_t, accscalar_t>(
              odata,
              n,
              nbatch,
              channels,
              output_depth,
              output_height,
              output_width,
              input_depth,
              input_height,
              input_width,
              idata,
              depth_scale,
              height_scale,
              width_scale,
              NearestBwIndexOp());
        }
      });
}

} // namespace at::native::xpu

#pragma clang diagnostic pop
#pragma GCC diagnostic pop
