#pragma clang diagnostic push
#pragma GCC diagnostic push
// Avoid SYCL compiler return-type error
#pragma clang diagnostic ignored "-Wreturn-type"
#pragma GCC diagnostic ignored "-Wreturn-type"

#include <ATen/Dispatch.h>
#include <ATen/ceil_div.h>
#include <ATen/native/IndexingUtils.h>
#include <ATen/native/Padding.h>
#include <ATen/native/xpu/sycl/Atomics.h>
#include <comm/Runtime.h>
#include <comm/SYCLContext.h>
#include <comm/xpu_aten.h>

namespace at::native::xpu {

inline std::pair<int64_t, int64_t> get_index_mapping2d(
    int64_t input_dim_x,
    int64_t input_dim_y,
    int64_t output_dim_x,
    int64_t output_dim_y,
    int64_t pad_l,
    int64_t pad_t,
    int64_t output_xy,
    const sycl::nd_item<3> item) {
  // 3D grid of 1D blocks
  auto input_offset =
      (item.get_group(1) + item.get_group(2) * item.get_group_range(1)) *
      input_dim_x * input_dim_y;
  auto output_offset =
      (item.get_group(1) + item.get_group(2) * item.get_group_range(1)) *
      output_dim_x * output_dim_y;

  auto output_x = output_xy % output_dim_x;
  auto output_y = output_xy / output_dim_x;

  auto i_start_x = std::max(int64_t(0), -pad_l);
  auto i_start_y = std::max(int64_t(0), -pad_t);
  auto o_start_x = std::max(int64_t(0), pad_l);
  auto o_start_y = std::max(int64_t(0), pad_t);

  int64_t input_x = std::abs(output_x - pad_l) -
      std::abs(output_x - (input_dim_x + pad_l - 1)) - output_x + 2 * pad_l +
      input_dim_x - 1 - o_start_x + i_start_x;

  int64_t input_y = std::abs(output_y - pad_t) -
      std::abs(output_y - (input_dim_y + pad_t - 1)) - output_y + 2 * pad_t +
      input_dim_y - 1 - o_start_y + i_start_y;

  return std::make_pair<int64_t, int64_t>(
      input_offset + input_y * input_dim_x + input_x,
      output_offset + output_y * output_dim_x + output_x);
}

template <typename scalar_t>
struct ReflectionPad2dKernellFunctor {
  void operator()(sycl::nd_item<3> item) const {
    auto output_xy = item.get_global_id(0);

    if (output_xy < output_dim_x_ * output_dim_y_) {
      // input index and output index mapping
      auto index_pair = get_index_mapping2d(
          input_dim_x_,
          input_dim_y_,
          output_dim_x_,
          output_dim_y_,
          pad_l_,
          pad_t_,
          output_xy,
          item);
      output_[index_pair.second] = input_[index_pair.first];
    }
  }
  ReflectionPad2dKernellFunctor(
      scalar_t* input,
      scalar_t* output,
      int64_t input_dim_x,
      int64_t input_dim_y,
      int64_t pad_t,
      int64_t pad_l,
      int64_t output_dim_x,
      int64_t output_dim_y)
      : input_(input),
        output_(output),
        input_dim_x_(input_dim_x),
        input_dim_y_(input_dim_y),
        pad_t_(pad_t),
        pad_l_(pad_l),
        output_dim_x_(output_dim_x),
        output_dim_y_(output_dim_y) {}

 private:
  scalar_t* input_;
  scalar_t* output_;
  int64_t input_dim_x_;
  int64_t input_dim_y_;
  int64_t pad_t_;
  int64_t pad_l_;
  int64_t output_dim_x_;
  int64_t output_dim_y_;
};

template <typename scalar_t>
void reflection_pad2d_template(
    scalar_t* input,
    scalar_t* output,
    int64_t input_dim_x,
    int64_t input_dim_y,
    int64_t pad_t,
    int64_t pad_b,
    int64_t pad_l,
    int64_t pad_r,
    int64_t nbatch,
    int64_t nplane) {
  int64_t output_dim_x = input_dim_x + pad_l + pad_r;
  int64_t output_dim_y = input_dim_y + pad_t + pad_b;

  auto queue = getCurrentSYCLQueue();
  int64_t work_group_size = syclMaxWorkItemsPerEU();
  int64_t work_group_num =
      at::ceil_div(output_dim_x * output_dim_y, work_group_size);

  ReflectionPad2dKernellFunctor<scalar_t> kfn(
      input,
      output,
      input_dim_x,
      input_dim_y,
      pad_t,
      pad_l,
      output_dim_x,
      output_dim_y);
  sycl_kernel_submit(
      sycl::range<3>(work_group_size * work_group_num, nplane, nbatch),
      sycl::range<3>(work_group_size, 1, 1),
      queue,
      kfn);
}

template <typename scalar_t>
struct ReflectionPad2dBackwardKernelFunctor {
  void operator()(sycl::nd_item<3> item) const {
    auto output_xy = item.get_global_id(0);

    if (output_xy < output_dim_x_ * output_dim_y_) {
      // grad input index and grad output index mapping
      auto index_pair = get_index_mapping2d(
          input_dim_x_,
          input_dim_y_,
          output_dim_x_,
          output_dim_y_,
          pad_l_,
          pad_t_,
          output_xy,
          item);
      atomicAdd(
          (sycl_global_ptr<scalar_t>)&grad_input_[index_pair.first],
          grad_output_[index_pair.second]);
    }
  }
  ReflectionPad2dBackwardKernelFunctor(
      scalar_t* grad_input,
      scalar_t* grad_output,
      int64_t input_dim_x,
      int64_t input_dim_y,
      int64_t pad_t,
      int64_t pad_l,
      int64_t output_dim_x,
      int64_t output_dim_y)
      : grad_input_(grad_input),
        grad_output_(grad_output),
        input_dim_x_(input_dim_x),
        input_dim_y_(input_dim_y),
        pad_t_(pad_t),
        pad_l_(pad_l),
        output_dim_x_(output_dim_x),
        output_dim_y_(output_dim_y) {}

 private:
  scalar_t* grad_input_;
  scalar_t* grad_output_;
  int64_t input_dim_x_;
  int64_t input_dim_y_;
  int64_t pad_t_;
  int64_t pad_l_;
  int64_t output_dim_x_;
  int64_t output_dim_y_;
};

template <typename scalar_t>
void reflection_pad2d_backward_template(
    scalar_t* grad_input,
    scalar_t* grad_output,
    int64_t input_dim_x,
    int64_t input_dim_y,
    int64_t pad_t,
    int64_t pad_b,
    int64_t pad_l,
    int64_t pad_r,
    int64_t nbatch,
    int64_t nplane) {
  auto queue = getCurrentSYCLQueue();
  int64_t output_dim_x = input_dim_x + pad_l + pad_r;
  int64_t output_dim_y = input_dim_y + pad_t + pad_b;
  int64_t work_group_size = syclMaxWorkItemsPerEU();
  int64_t work_group_num =
      at::ceil_div(output_dim_x * output_dim_y, work_group_size);

  ReflectionPad2dBackwardKernelFunctor<scalar_t> kfn(
      grad_input,
      grad_output,
      input_dim_x,
      input_dim_y,
      pad_t,
      pad_l,
      output_dim_x,
      output_dim_y);
  sycl_kernel_submit(
      sycl::range<3>(work_group_size * work_group_num, nplane, nbatch),
      sycl::range<3>(work_group_size, 1, 1),
      queue,
      kfn);
}

void reflection_pad2d_kernel(
    Tensor& output,
    const Tensor& input_,
    IntArrayRef padding) {
  TORCH_CHECK(
      canUse32BitIndexMath(input_),
      "input tensor must fit into 32-bit index math");

  int plane_dim = 0;
  int dim_h = 1;
  int dim_w = 2;
  int nbatch = 1;

  at::native::padding::check_valid_input<2>(input_, padding);

  if (input_.ndimension() == 4) {
    nbatch = input_.size(0);
    plane_dim++;
    dim_h++;
    dim_w++;
  }

  int64_t pad_l = padding[0];
  int64_t pad_r = padding[1];
  int64_t pad_t = padding[2];
  int64_t pad_b = padding[3];

  int64_t nplane = input_.size(plane_dim);
  int64_t input_h = input_.size(dim_h);
  int64_t input_w = input_.size(dim_w);

  TORCH_CHECK(
      pad_l < input_w && pad_r < input_w,
      "Padding size should be less than the corresponding input dimension, but "
      "got: padding (",
      pad_l,
      ", ",
      pad_r,
      ") at dimension ",
      dim_w,
      " of input ",
      input_.sizes());

  TORCH_CHECK(
      pad_t < input_h && pad_b < input_h,
      "Padding size should be less than the corresponding input dimension, but "
      "got: padding (",
      pad_t,
      ", ",
      pad_b,
      ") at dimension ",
      dim_h,
      " of input ",
      input_.sizes());
  int64_t output_h = input_h + pad_t + pad_b;
  int64_t output_w = input_w + pad_l + pad_r;

  TORCH_CHECK(
      output_w >= 1 || output_h >= 1,
      "input (H: ",
      input_h,
      ", W: ",
      input_w,
      ")is too small.  Calculated "
      "output H: ",
      output_h,
      " W: ",
      output_w);

  if (input_.ndimension() == 3) {
    output.resize_({nplane, output_h, output_w});
  } else {
    output.resize_({nbatch, nplane, output_h, output_w});
  }

  if (output.numel() == 0) {
    return;
  }

  Tensor input = input_.contiguous();

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      kHalf, kBFloat16, input.scalar_type(), "reflection_pad2d_xpu", [&] {
        reflection_pad2d_template<scalar_t>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            input_w,
            input_h,
            pad_t,
            pad_b,
            pad_l,
            pad_r,
            nbatch,
            nplane);
      });
}

void reflection_pad2d_backward_kernel(
    Tensor& grad_input,
    const Tensor& grad_output_,
    const Tensor& input,
    IntArrayRef padding) {
  if (grad_input.numel() == 0) {
    return;
  }

  TORCH_CHECK(
      canUse32BitIndexMath(input),
      "input tensor must fit into 32-bit index math");
  TORCH_CHECK(
      canUse32BitIndexMath(grad_output_),
      "output gradient tensor must fit into 32-bit index math");

  int64_t plane_dim = 0;
  int64_t dim_h = 1;
  int64_t dim_w = 2;
  int64_t nbatch = 1;

  if (input.ndimension() == 4) {
    nbatch = input.size(0);
    plane_dim++;
    dim_h++;
    dim_w++;
  }

  int64_t pad_l = padding[0];
  int64_t pad_r = padding[1];
  int64_t pad_t = padding[2];
  int64_t pad_b = padding[3];

  int64_t nplane = input.size(plane_dim);
  int64_t input_h = input.size(dim_h);
  int64_t input_w = input.size(dim_w);

  int64_t output_h = input_h + pad_t + pad_b;
  int64_t output_w = input_w + pad_l + pad_r;

  TORCH_CHECK(
      output_w == grad_output_.size(dim_w),
      "grad_output width "
      "unexpected. Expected: ",
      output_w,
      ", Got: ",
      grad_output_.size(dim_w));
  TORCH_CHECK(
      output_h == grad_output_.size(dim_h),
      "grad_output height "
      "unexpected. Expected: ",
      output_h,
      ", Got: ",
      grad_output_.size(dim_h));

  Tensor grad_output = grad_output_.contiguous();

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      kHalf,
      kBFloat16,
      input.scalar_type(),
      "reflection_pad2d_backward_xpu",
      [&] {
        reflection_pad2d_backward_template<scalar_t>(
            grad_input.data_ptr<scalar_t>(),
            grad_output.data_ptr<scalar_t>(),
            input_w,
            input_h,
            pad_t,
            pad_b,
            pad_l,
            pad_r,
            nbatch,
            nplane);
      });
}

} // namespace at::native::xpu

#pragma GCC diagnostic pop
#pragma clang diagnostic pop