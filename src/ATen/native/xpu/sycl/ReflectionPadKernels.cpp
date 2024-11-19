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

#include <ATen/native/xpu/sycl/ReflectionPadKernels.h>

namespace at::native::xpu {

inline std::pair<int64_t, int64_t> get_index_mapping1d(
    int64_t input_w,
    int64_t output_w,
    int64_t output_x,
    int64_t pad_l,
    const sycl::nd_item<3> item) {
  auto input_offset =
      (item.get_group(1) + item.get_group(0) * item.get_group_range(1)) *
      input_w;
  auto output_offset =
      (item.get_group(1) + item.get_group(0) * item.get_group_range(1)) *
      output_w;

  auto i_start_x = std::max(int64_t(0), -pad_l);
  auto o_start_x = std::max(int64_t(0), pad_l);

  int64_t input_x = std::abs(output_x - pad_l) -
      std::abs(output_x - (input_w + pad_l - 1)) - output_x + 2 * pad_l +
      input_w - 1 - o_start_x + i_start_x;

  return std::make_pair<int64_t, int64_t>(
      input_offset + input_x, output_offset + output_x);
}

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
      (item.get_group(1) + item.get_group(0) * item.get_group_range(1)) *
      input_dim_x * input_dim_y;
  auto output_offset =
      (item.get_group(1) + item.get_group(0) * item.get_group_range(1)) *
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
struct ReflectionPad1dKernelFunctor {
  void operator()(sycl::nd_item<3> item) const {
    auto output_x = item.get_global_id(2);

    if (output_x < output_w_) {
      // input index and output index mapping
      auto index_pair =
          get_index_mapping1d(input_w_, output_w_, output_x, pad_l_, item);
      output_data_[index_pair.second] = input_data_[index_pair.first];
    }
  }
  ReflectionPad1dKernelFunctor(
      const scalar_t* input_data,
      scalar_t* output_data,
      int64_t input_w,
      int64_t pad_l,
      int64_t output_w)
      : input_data_(input_data),
        output_data_(output_data),
        input_w_(input_w),
        pad_l_(pad_l),
        output_w_(output_w) {}

 private:
  const scalar_t* input_data_;
  scalar_t* output_data_;
  int64_t input_w_;
  int64_t pad_l_;
  int64_t output_w_;
};

template <typename scalar_t>
void reflection_pad1d_template(
    const scalar_t* input,
    scalar_t* output,
    int64_t input_w,
    int64_t pad_l,
    int64_t pad_r,
    int64_t nbatch,
    int64_t nplane,
    int64_t output_w) {
  auto queue = getCurrentSYCLQueue();
  int64_t work_group_size = syclMaxWorkItemsPerEU();
  int64_t work_group_num = at::ceil_div(output_w, work_group_size);

  ReflectionPad1dKernelFunctor<scalar_t> kfn(
      input, output, input_w, pad_l, output_w);
  sycl_kernel_submit(
      sycl::range<3>(nbatch, nplane, work_group_size * work_group_num),
      sycl::range<3>(1, 1, work_group_size),
      queue,
      kfn);
}

template <typename scalar_t>
struct ReflectionPad1dBackwardKernelFunctor {
  void operator()(sycl::nd_item<3> item) const {
    auto output_x = item.get_global_id(2);

    if (output_x < output_w_) {
      // grad input index and grad output index mapping
      auto index_pair =
          get_index_mapping1d(input_w_, output_w_, output_x, pad_l_, item);
      atomicAdd(
          (sycl_global_ptr<scalar_t>)&grad_input_data_[index_pair.first],
          grad_output_data_[index_pair.second]);
    }
  }
  ReflectionPad1dBackwardKernelFunctor(
      scalar_t* grad_input_data,
      const scalar_t* grad_output_data,
      int64_t input_w,
      int64_t pad_l,
      int64_t output_w)
      : grad_input_data_(grad_input_data),
        grad_output_data_(grad_output_data),
        input_w_(input_w),
        pad_l_(pad_l),
        output_w_(output_w) {}

 private:
  scalar_t* grad_input_data_;
  const scalar_t* grad_output_data_;
  int64_t input_w_;
  int64_t pad_l_;
  int64_t output_w_;
};

template <typename scalar_t>
void reflection_pad1d_backward_template(
    scalar_t* grad_input,
    const scalar_t* grad_output,
    int64_t input_w,
    int64_t pad_l,
    int64_t pad_r,
    int64_t nbatch,
    int64_t nplane,
    int64_t output_w) {
  auto queue = getCurrentSYCLQueue();
  int64_t work_group_size = syclMaxWorkItemsPerEU();
  int64_t work_group_num = at::ceil_div(output_w, work_group_size);

  ReflectionPad1dBackwardKernelFunctor<scalar_t> kfn(
      grad_input, grad_output, input_w, pad_l, output_w);
  sycl_kernel_submit(
      sycl::range<3>(nbatch, nplane, work_group_size * work_group_num),
      sycl::range<3>(1, 1, work_group_size),
      queue,
      kfn);
}

template <typename scalar_t>
struct ReflectionPad2dKernellFunctor {
  void operator()(sycl::nd_item<3> item) const {
    auto output_xy = item.get_global_id(2);

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
      const scalar_t* input,
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
  const scalar_t* input_;
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
    const scalar_t* input,
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
      sycl::range<3>(nbatch, nplane, work_group_size * work_group_num),
      sycl::range<3>(1, 1, work_group_size),
      queue,
      kfn);
}

template <typename scalar_t>
struct ReflectionPad2dBackwardKernelFunctor {
  void operator()(sycl::nd_item<3> item) const {
    auto output_xy = item.get_global_id(2);

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
      const scalar_t* grad_output,
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
  const scalar_t* grad_output_;
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
    const scalar_t* grad_output,
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
      sycl::range<3>(nbatch, nplane, work_group_size * work_group_num),
      sycl::range<3>(1, 1, work_group_size),
      queue,
      kfn);
}

template <typename input_scalar_t, typename output_scalar_t, typename F>
struct ParallelReflectionPad3dKernelFunctor {
  void operator()(sycl::nd_item<3> item) const {
    auto output_id = item.get_global_id(2);
    if (output_id >= output_plane_size_) {
      return;
    }

    int64_t output_x = output_id % output_.size(4);
    int64_t output_y = (output_id / output_.size(4)) % output_.size(3);
    int64_t output_z = output_id / (output_.size(3) * output_.size(4));

    int64_t i_start_x = std::max(int64_t(0), -pad_left_);
    int64_t o_start_x = std::max(int64_t(0), pad_left_);
    int64_t i_start_y = std::max(int64_t(0), -pad_top_);
    int64_t o_start_y = std::max(int64_t(0), pad_top_);
    int64_t i_start_z = std::max(int64_t(0), -pad_front_);
    int64_t o_start_z = std::max(int64_t(0), pad_front_);

    int64_t input_x = std::abs(output_x - pad_left_) -
        std::abs(output_x - (input_.size(4) + pad_left_ - 1)) - output_x +
        2 * pad_left_ + input_.size(4) - 1 - o_start_x + i_start_x;
    int64_t input_y = std::abs(output_y - pad_top_) -
        std::abs(output_y - (input_.size(3) + pad_top_ - 1)) - output_y +
        2 * pad_top_ + input_.size(3) - 1 - o_start_y + i_start_y;

    int64_t input_z = std::abs(output_z - pad_front_) -
        std::abs(output_z - (input_.size(2) + pad_front_ - 1)) - output_z +
        2 * pad_front_ + input_.size(2) - 1 - o_start_z + i_start_z;

    f_(input_,
       output_,
       item.get_group(1),
       item.get_group(0),
       output_z,
       output_y,
       output_x,
       input_z,
       input_y,
       input_x);
  }
  ParallelReflectionPad3dKernelFunctor(
      PackedTensorAccessor64<input_scalar_t, 5> input,
      PackedTensorAccessor64<output_scalar_t, 5> output,
      int64_t pad_left,
      int64_t pad_top,
      int64_t pad_front,
      const F f,
      int64_t output_plane_size)
      : input_(input),
        output_(output),
        pad_left_(pad_left),
        pad_top_(pad_top),
        pad_front_(pad_front),
        f_(f),
        output_plane_size_(output_plane_size) {}

 private:
  PackedTensorAccessor64<input_scalar_t, 5> input_;
  PackedTensorAccessor64<output_scalar_t, 5> output_;
  int64_t pad_left_;
  int64_t pad_top_;
  int64_t pad_front_;
  const F f_;
  int64_t output_plane_size_;
};

template <typename input_scalar_t, typename output_scalar_t, typename F>
inline void parallel_reflection_pad3d(
    PackedTensorAccessor64<input_scalar_t, 5> input,
    PackedTensorAccessor64<output_scalar_t, 5> output,
    int64_t pad_left,
    int64_t pad_top,
    int64_t pad_front,
    const F& f) {
  auto queue = getCurrentSYCLQueue();
  int64_t output_plane_size = output.size(2) * output.size(3) * output.size(4);
  int64_t work_group_size = syclMaxWorkItemsPerEU();
  int64_t work_group_num = at::ceil_div(output_plane_size, work_group_size);
  int64_t nplane = input.size(1);
  int64_t nbatch = input.size(0);

  ParallelReflectionPad3dKernelFunctor<input_scalar_t, output_scalar_t, F> kfn(
      input, output, pad_left, pad_top, pad_front, f, output_plane_size);
  sycl_kernel_submit(
      sycl::range<3>(nbatch, nplane, work_group_size * work_group_num),
      sycl::range<3>(1, 1, work_group_size),
      queue,
      kfn);
}

template <typename scalar_t>
struct reflection_pad3d_kernel_functor {
  void operator()(
      PackedTensorAccessor64<const scalar_t, 5> input,
      PackedTensorAccessor64<scalar_t, 5> output,
      int64_t plane,
      int64_t batch,
      int64_t output_z,
      int64_t output_y,
      int64_t output_x,
      int64_t input_z,
      int64_t input_y,
      int64_t input_x) const {
    auto value_to_copy = input[batch][plane][input_z][input_y][input_x];
    output[batch][plane][output_z][output_y][output_x] = value_to_copy;
  }
};

template <typename scalar_t>
void reflection_pad3d_template(
    PackedTensorAccessor64<const scalar_t, 5> input,
    PackedTensorAccessor64<scalar_t, 5> output,
    int64_t pad_left,
    int64_t pad_top,
    int64_t pad_front) {
  reflection_pad3d_kernel_functor<scalar_t> f;
  parallel_reflection_pad3d(input, output, pad_left, pad_top, pad_front, f);
}

template <typename scalar_t>
struct reflection_pad3d_backward_kernel_functor {
  void operator()(
      PackedTensorAccessor64<scalar_t, 5> grad_input,
      PackedTensorAccessor64<const scalar_t, 5> grad_output,
      int64_t plane,
      int64_t batch,
      int64_t output_z,
      int64_t output_y,
      int64_t output_x,
      int64_t input_z,
      int64_t input_y,
      int64_t input_x) const {
    auto value_to_add = grad_output[batch][plane][output_z][output_y][output_x];
    auto target = (sycl_global_ptr<scalar_t>)&grad_input[batch][plane][input_z]
                                                        [input_y][input_x];
    atomicAdd(target, value_to_add);
  }
};

template <typename scalar_t>
void reflection_pad3d_backward_template(
    PackedTensorAccessor64<scalar_t, 5> grad_input,
    PackedTensorAccessor64<const scalar_t, 5> grad_output,
    int64_t pad_left,
    int64_t pad_top,
    int64_t pad_front) {
  reflection_pad3d_backward_kernel_functor<scalar_t> f;
  parallel_reflection_pad3d(
      grad_input, grad_output, pad_left, pad_top, pad_front, f);
}

void reflection_pad1d_kernel(
    const Tensor& output,
    const Tensor& input_,
    IntArrayRef padding) {
  TORCH_CHECK(
      canUse32BitIndexMath(input_),
      "input tensor must fit into 32-bit index math");

  if (output.numel() == 0) {
    return;
  }

  int64_t dim_plane = 0;
  int64_t dim_w = 1;
  int64_t nbatch = 1;

  if (input_.ndimension() == 3) {
    nbatch = input_.size(0);
    dim_plane++;
    dim_w++;
  }

  int64_t pad_l = padding[0];
  int64_t pad_r = padding[1];

  int64_t nplane = input_.size(dim_plane);
  int64_t input_w = input_.size(dim_w);
  int64_t output_w = input_w + pad_l + pad_r;

  Tensor input = input_.contiguous();

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      kHalf, kBFloat16, input.scalar_type(), "reflection_pad1d_xpu", [&] {
        reflection_pad1d_template<scalar_t>(
            input.const_data_ptr<scalar_t>(),
            output.mutable_data_ptr<scalar_t>(),
            input_w,
            pad_l,
            pad_r,
            nbatch,
            nplane,
            output_w);
      });
}

void reflection_pad1d_backward_kernel(
    const Tensor& grad_input,
    const Tensor& grad_output_,
    const Tensor& input,
    IntArrayRef padding) {
  globalContext().alertNotDeterministic("reflection_pad1d_backward_out_xpu");
  grad_input.zero_();

  if (grad_input.numel() == 0) {
    return;
  }

  TORCH_CHECK(
      canUse32BitIndexMath(input),
      "input tensor must fit into 32-bit index math");

  TORCH_CHECK(
      canUse32BitIndexMath(grad_output_),
      "input tensor must fit into 32-bit index math");

  int64_t dim_plane = 0;
  int64_t dim_w = 1;
  int64_t nbatch = 1;

  if (input.ndimension() == 3) {
    nbatch = input.size(0);
    dim_plane++;
    dim_w++;
  }

  int64_t pad_l = padding[0];
  int64_t pad_r = padding[1];

  int64_t nplane = input.size(dim_plane);
  int64_t input_w = input.size(dim_w);
  int64_t output_w = input_w + pad_l + pad_r;

  Tensor grad_output = grad_output_.contiguous();

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      kHalf,
      kBFloat16,
      grad_input.scalar_type(),
      "reflection_pad1d_backward_xpu",
      [&] {
        reflection_pad1d_backward_template<scalar_t>(
            grad_input.mutable_data_ptr<scalar_t>(),
            grad_output.const_data_ptr<scalar_t>(),
            input_w,
            pad_l,
            pad_r,
            nbatch,
            nplane,
            output_w);
      });
}

void reflection_pad2d_kernel(
    const Tensor& output,
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
            input.const_data_ptr<scalar_t>(),
            output.mutable_data_ptr<scalar_t>(),
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
    const Tensor& grad_input,
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
            grad_input.mutable_data_ptr<scalar_t>(),
            grad_output.const_data_ptr<scalar_t>(),
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

void reflection_pad3d_kernel(
    const Tensor& output,
    const Tensor& input_,
    IntArrayRef padding) {
  TORCH_CHECK(
      canUse32BitIndexMath(input_),
      "input tensor must fit into 32-bit index math");

  if (output.numel() == 0) {
    return;
  }

  int64_t pad_left = padding[0];
  int64_t pad_top = padding[2];
  int64_t pad_front = padding[4];

  auto input = input_.contiguous();
  bool batch_mode = (input.dim() == 5);

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      kHalf, kBFloat16, input.scalar_type(), "reflection_pad3d_xpu", [&] {
        auto input_inner = input;
        auto output_inner = output;
        if (!batch_mode) {
          input_inner = input.unsqueeze(0);
          output_inner = output.unsqueeze(0);
        }

        auto input_packed = input_inner.packed_accessor64<const scalar_t, 5>();
        auto output_packed = output_inner.packed_accessor64<scalar_t, 5>();

        reflection_pad3d_template<scalar_t>(
            input_packed, output_packed, pad_left, pad_top, pad_front);
      });
}

void reflection_pad3d_backward_kernel(
    const Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef padding) {
  globalContext().alertNotDeterministic("reflection_pad3d_backward_out_xpu");
  TORCH_CHECK(
      canUse32BitIndexMath(input),
      "input tensor must fit into 32-bit index math");
  TORCH_CHECK(
      canUse32BitIndexMath(grad_output),
      "input tensor must fit into 32-bit index math");

  if (grad_input.numel() == 0) {
    return;
  }
  grad_input.zero_();

  int64_t pad_left = padding[0];
  int64_t pad_top = padding[2];
  int64_t pad_front = padding[4];

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      kHalf,
      kBFloat16,
      input.scalar_type(),
      "reflection_pad3d_backward_xpu",
      [&] {
        auto grad_input_ = grad_input;
        auto grad_output_ = grad_output;
        if (input.dim() == 4) {
          // non-batch mode
          grad_input_ = grad_input.unsqueeze(0);
          grad_output_ = grad_output.unsqueeze(0);
        }

        auto grad_input_packed = grad_input_.packed_accessor64<scalar_t, 5>();
        auto grad_output_packed =
            grad_output_.packed_accessor64<const scalar_t, 5>();

        reflection_pad3d_backward_template<scalar_t>(
            grad_input_packed,
            grad_output_packed,
            pad_left,
            pad_top,
            pad_front);
      });
}

} // namespace at::native::xpu

#pragma GCC diagnostic pop
#pragma clang diagnostic pop
