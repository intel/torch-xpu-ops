#pragma clang diagnostic push
#pragma GCC diagnostic push
// Avoid SYCL compiler return-type error
#pragma clang diagnostic ignored "-Wreturn-type"
#pragma GCC diagnostic ignored "-Wreturn-type"

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/ceil_div.h>
#include <ATen/native/IndexingUtils.h>
#include <ATen/native/xpu/sycl/Atomics.h>
#include <comm/Runtime.h>
#include <comm/SYCLContext.h>

#include <ATen/native/xpu/sycl/ReplicationPaddingKernels.h>

namespace at::native::xpu {

inline int imin(int a, int b) {
  return a > b ? b : a;
}
inline int imax(int a, int b) {
  return a > b ? a : b;
}

template <typename input_scalar_t, typename output_scalar_t, typename F>
struct ParallelReplicationPad1dKernelFunctor {
  void operator()(sycl::nd_item<3> item) const {
    auto output_id = item.get_global_id(2);
    if (output_id < output_plane_size_) {
      int64_t output_x = output_id % output_.size(2);
      int64_t i_start_x = imax(0, -pad_left_);
      int64_t o_start_x = imax(0, pad_left_);
      int64_t input_x =
          imin(imax(pad_left_, output_x), input_.size(2) + pad_left_ - 1) -
          o_start_x + i_start_x;

      f_(input_,
         output_,
         item.get_group(1),
         item.get_group(0),
         output_x,
         input_x);
    }
  }
  ParallelReplicationPad1dKernelFunctor(
      PackedTensorAccessor64<input_scalar_t, 3> input,
      PackedTensorAccessor64<output_scalar_t, 3> output,
      int64_t pad_left,
      int64_t pad_right,
      const F f,
      int64_t output_plane_size)
      : input_(input),
        output_(output),
        pad_left_(pad_left),
        pad_right_(pad_right),
        f_(f),
        output_plane_size_(output_plane_size) {}

 private:
  PackedTensorAccessor64<input_scalar_t, 3> input_;
  PackedTensorAccessor64<output_scalar_t, 3> output_;
  int64_t pad_left_;
  int64_t pad_right_;
  const F f_;
  int64_t output_plane_size_;
};

template <typename input_scalar_t, typename output_scalar_t, typename F>
void parallel_replication_pad1d(
    PackedTensorAccessor64<input_scalar_t, 3> input,
    PackedTensorAccessor64<output_scalar_t, 3> output,
    int64_t pad_left,
    int64_t pad_right,
    const F& f) {
  auto queue = getCurrentSYCLQueue();
  int64_t output_plane_size = output.size(2);

  ParallelReplicationPad1dKernelFunctor<input_scalar_t, output_scalar_t, F> kfn(
      input, output, pad_left, pad_right, f, output_plane_size);

  int64_t work_group_size = syclMaxWorkGroupSize(kfn);
  int64_t work_group_num = at::ceil_div(output_plane_size, work_group_size);
  int64_t nplane = output.size(1);
  int64_t nbatch = output.size(0);

  sycl_kernel_submit(
      sycl::range<3>(nbatch, nplane, work_group_size * work_group_num),
      sycl::range<3>(1, 1, work_group_size),
      queue,
      kfn);
}

template <typename scalar_t>
struct ReplicationPad1dForwardFunctor {
  void operator()(
      PackedTensorAccessor64<const scalar_t, 3> input,
      PackedTensorAccessor64<scalar_t, 3> output,
      int64_t plane,
      int64_t batch,
      int64_t output_x,
      int64_t intput_x) const {
    auto value_to_copy = input[batch][plane][intput_x];
    output[batch][plane][output_x] = value_to_copy;
  }
};

template <typename scalar_t>
void replication_pad1d_forward_template(
    PackedTensorAccessor64<const scalar_t, 3> input,
    PackedTensorAccessor64<scalar_t, 3> output,
    int64_t pad_left,
    int64_t pad_right) {
  ReplicationPad1dForwardFunctor<scalar_t> f;
  parallel_replication_pad1d(input, output, pad_left, pad_right, f);
}

template <typename scalar_t>
struct ReplicationPad1dBackwardFunctor {
  void operator()(
      PackedTensorAccessor64<scalar_t, 3> grad_input,
      PackedTensorAccessor64<const scalar_t, 3> grad_output,
      int64_t plane,
      int64_t batch,
      int64_t output_x,
      int64_t intput_x) const {
    auto value_to_add = grad_output[batch][plane][output_x];
    auto target =
        (sycl_global_ptr<scalar_t>)&grad_input[batch][plane][intput_x];
    atomicAdd(target, value_to_add);
  }
};

template <typename scalar_t>
void replication_pad1d_backward_template(
    PackedTensorAccessor64<scalar_t, 3> grad_input,
    PackedTensorAccessor64<const scalar_t, 3> grad_output,
    int64_t pad_left,
    int64_t pad_right) {
  ReplicationPad1dBackwardFunctor<scalar_t> f;
  parallel_replication_pad1d(grad_input, grad_output, pad_left, pad_right, f);
}

template <typename input_scalar_t, typename output_scalar_t, typename F>
struct ParallelReplicationPad2dKernelFunctor {
  void operator()(sycl::nd_item<3> item) const {
    const int output_id = item.get_global_id(2);
    const int batch = item.get_global_id(0);
    const int plane = item.get_global_id(1);

    if (output_id < output_.size(2) * output_.size(3)) {
      const int output_x = output_id / output_.size(3); // height
      const int output_y = output_id % output_.size(3); // width

      const int iStartX = imax(0, -padT_);
      const int iStartY = imax(0, -padL_);
      const int oStartX = imax(0, padT_);
      const int oStartY = imax(0, padL_);

      const int input_x =
          imin(imax(padT_, output_x), input_.size(2) + padT_ - 1) - oStartX +
          iStartX;
      const int input_y =
          imin(imax(padL_, output_y), input_.size(3) + padL_ - 1) - oStartY +
          iStartY;

      f_(input_, output_, batch, plane, input_x, input_y, output_x, output_y);
    }
  }
  ParallelReplicationPad2dKernelFunctor(
      PackedTensorAccessor64<input_scalar_t, 4> input,
      PackedTensorAccessor64<output_scalar_t, 4> output,
      int64_t padT,
      int64_t padL,
      const F f)
      : input_(input), output_(output), padT_(padT), padL_(padL), f_(f) {}

 private:
  PackedTensorAccessor64<input_scalar_t, 4> input_;
  PackedTensorAccessor64<output_scalar_t, 4> output_;
  int64_t padT_;
  int64_t padL_;
  const F f_;
};

template <typename input_scalar_t, typename output_scalar_t, typename F>
void parallel_replication_pad2d(
    PackedTensorAccessor64<input_scalar_t, 4> input,
    PackedTensorAccessor64<output_scalar_t, 4> output,
    const int padT,
    const int padL,
    const F& f) {
  auto queue = getCurrentSYCLQueue();
  int64_t output_plane_size = output.size(2) * output.size(3);

  ParallelReplicationPad2dKernelFunctor<input_scalar_t, output_scalar_t, F> kfn(
      input, output, padT, padL, f);

  int64_t work_group_size = syclMaxWorkGroupSize(kfn);
  int64_t work_group_num = at::ceil_div(output_plane_size, work_group_size);
  int64_t nplane = output.size(1);
  int64_t nbatch = output.size(0);

  sycl_kernel_submit(
      sycl::range<3>(nbatch, nplane, work_group_size * work_group_num),
      sycl::range<3>(1, 1, work_group_size),
      queue,
      kfn);
}

template <typename scalar_t>
struct ReplicationPad2dForwardFunctor {
  void operator()(
      PackedTensorAccessor64<const scalar_t, 4> input,
      PackedTensorAccessor64<scalar_t, 4> output,
      int64_t batch,
      int64_t plane,
      int64_t input_x,
      int64_t input_y,
      int64_t output_x,
      int64_t output_y) const {
    scalar_t valueToCopy = input[batch][plane][input_x][input_y];
    output[batch][plane][output_x][output_y] = valueToCopy;
  }
};

template <typename scalar_t>
void replication_pad2d_forward_template(
    PackedTensorAccessor64<const scalar_t, 4> input,
    PackedTensorAccessor64<scalar_t, 4> output,
    int64_t padT,
    int64_t padL) {
  ReplicationPad2dForwardFunctor<scalar_t> f;
  parallel_replication_pad2d(input, output, padT, padL, f);
}

template <typename scalar_t>
struct ReplicationPad2dBackwardFunctor {
  void operator()(
      PackedTensorAccessor64<scalar_t, 4> grad_input,
      PackedTensorAccessor64<const scalar_t, 4> grad_output,
      int64_t batch,
      int64_t plane,
      int64_t input_x,
      int64_t input_y,
      int64_t output_x,
      int64_t output_y) const {
    scalar_t valueToAdd = grad_output[batch][plane][output_x][output_y];
    auto target =
        (sycl_global_ptr<scalar_t>)&grad_input[batch][plane][input_x][input_y];
    atomicAdd(target, valueToAdd);
  }
};

template <typename scalar_t>
void replication_pad2d_backward_template(
    PackedTensorAccessor64<scalar_t, 4> grad_input,
    PackedTensorAccessor64<const scalar_t, 4> grad_output,
    const int padT,
    const int padL) {
  ReplicationPad2dBackwardFunctor<scalar_t> f;
  parallel_replication_pad2d(grad_input, grad_output, padT, padL, f);
}

template <typename input_scalar_t, typename output_scalar_t, typename F>
struct ParallelReplicationPad3dKernelFunctor {
  void operator()(sycl::nd_item<3> item) const {
    auto output_id = item.get_global_id(2);
    if (output_id < output_plane_size_) {
      int64_t output_x = output_id % output_.size(4);
      int64_t output_y = (output_id / output_.size(4)) % output_.size(3);
      int64_t output_z = output_id / (output_.size(3) * output_.size(4));

      int64_t i_start_x = imax(0, -pad_left_);
      int64_t i_start_y = imax(0, -pad_top_);
      int64_t i_start_z = imax(0, -pad_front_);
      int64_t o_start_x = imax(0, pad_left_);
      int64_t o_start_y = imax(0, pad_top_);
      int64_t o_start_z = imax(0, pad_front_);

      int64_t input_x =
          imin(imax(pad_left_, output_x), input_.size(4) + pad_left_ - 1) -
          o_start_x + i_start_x;
      int64_t input_y =
          imin(imax(pad_top_, output_y), input_.size(3) + pad_top_ - 1) -
          o_start_y + i_start_y;
      int64_t input_z =
          imin(imax(pad_front_, output_z), input_.size(2) + pad_front_ - 1) -
          o_start_z + i_start_z;

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
  }
  ParallelReplicationPad3dKernelFunctor(
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
void parallel_replication_pad3d(
    PackedTensorAccessor64<input_scalar_t, 5> input,
    PackedTensorAccessor64<output_scalar_t, 5> output,
    int64_t pad_left,
    int64_t pad_top,
    int64_t pad_front,
    const F& f) {
  auto queue = getCurrentSYCLQueue();
  int64_t output_plane_size = output.size(2) * output.size(3) * output.size(4);

  ParallelReplicationPad3dKernelFunctor<input_scalar_t, output_scalar_t, F> kfn(
      input, output, pad_left, pad_top, pad_front, f, output_plane_size);
  int64_t work_group_size = syclMaxWorkGroupSize(kfn);
  int64_t work_group_num = at::ceil_div(output_plane_size, work_group_size);
  int64_t nplane = output.size(1);
  int64_t nbatch = output.size(0);

  sycl_kernel_submit(
      sycl::range<3>(nbatch, nplane, work_group_size * work_group_num),
      sycl::range<3>(1, 1, work_group_size),
      queue,
      kfn);
}

template <typename scalar_t>
struct ReplicationPad3dForwardFunctor {
  void operator()(
      PackedTensorAccessor64<const scalar_t, 5> input,
      PackedTensorAccessor64<scalar_t, 5> output,
      int64_t plane,
      int64_t batch,
      int64_t output_z,
      int64_t output_y,
      int64_t output_x,
      int64_t intput_z,
      int64_t intput_y,
      int64_t intput_x) const {
    auto value_to_copy = input[batch][plane][intput_z][intput_y][intput_x];
    output[batch][plane][output_z][output_y][output_x] = value_to_copy;
  }
};

template <typename scalar_t>
void replication_pad3d_forward_template(
    PackedTensorAccessor64<const scalar_t, 5> input,
    PackedTensorAccessor64<scalar_t, 5> output,
    int64_t pad_left,
    int64_t pad_top,
    int64_t pad_front) {
  ReplicationPad3dForwardFunctor<scalar_t> f;
  parallel_replication_pad3d(input, output, pad_left, pad_top, pad_front, f);
}

template <typename scalar_t>
struct ReplicationPad3dBackwardFunctor {
  void operator()(
      PackedTensorAccessor64<scalar_t, 5> grad_input,
      PackedTensorAccessor64<const scalar_t, 5> grad_output,
      int64_t plane,
      int64_t batch,
      int64_t output_z,
      int64_t output_y,
      int64_t output_x,
      int64_t intput_z,
      int64_t intput_y,
      int64_t intput_x) const {
    auto value_to_add = grad_output[batch][plane][output_z][output_y][output_x];
    auto target = (sycl_global_ptr<scalar_t>)&grad_input[batch][plane][intput_z]
                                                        [intput_y][intput_x];
    atomicAdd(target, value_to_add);
  }
};

template <typename scalar_t>
void replication_pad3d_backward_template(
    PackedTensorAccessor64<scalar_t, 5> grad_input,
    PackedTensorAccessor64<const scalar_t, 5> grad_output,
    int64_t pad_left,
    int64_t pad_top,
    int64_t pad_front) {
  ReplicationPad3dBackwardFunctor<scalar_t> f;
  parallel_replication_pad3d(
      grad_input, grad_output, pad_left, pad_top, pad_front, f);
}

void replication_pad1d_kernel(
    const Tensor& output,
    const Tensor& input,
    IntArrayRef padding) {
  TORCH_CHECK(
      input.numel() < std::numeric_limits<int64_t>::max(),
      "replication_pad1d only supports input tensors with less than 2^63 - 1 elements");

  if (input.numel() == 0) {
    return;
  }

  int64_t pad_left = padding[0];
  int64_t pad_right = padding[1];
  int64_t num_input_dims = input.dim();

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      kHalf, kBFloat16, input.scalar_type(), "replication_pad1d_xpu", [&] {
        auto input_ = input;
        auto output_ = output;
        if (num_input_dims == 2) {
          input_ = input.unsqueeze(0);
          output_ = output.unsqueeze(0);
        }

        auto input_packed = input_.packed_accessor64<const scalar_t, 3>();
        auto output_packed = output_.packed_accessor64<scalar_t, 3>();

        replication_pad1d_forward_template<scalar_t>(
            input_packed, output_packed, pad_left, pad_right);
      });
}

void replication_pad1d_backward_kernel(
    const Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef padding) {
  // See Note [Writing Nondeterministic Operations]
  // Nondeterministic because of atomicAdd usage
  globalContext().alertNotDeterministic("replication_pad1d_backward_xpu");

  TORCH_CHECK(
      input.numel() < std::numeric_limits<int64_t>::max(),
      "replication_pad1d only supports input tensors with less than 2^63 - 1 elements");
  TORCH_CHECK(
      grad_output.numel() < std::numeric_limits<int64_t>::max(),
      "replication_pad1d only supports output tensors with less than 2^63 - 1 elements");

  if (grad_input.numel() == 0) {
    return;
  }
  grad_input.zero_();

  int pad_left = padding[0];
  int pad_right = padding[1];
  int num_input_dims = input.ndimension();

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      kHalf,
      kBFloat16,
      input.scalar_type(),
      "replication_pad1d_backward_xpu",
      [&] {
        auto grad_input_ = grad_input;
        auto grad_output_ = grad_output;
        if (num_input_dims == 2) {
          grad_input_ = grad_input.unsqueeze(0);
          grad_output_ = grad_output.unsqueeze(0);
        }
        auto grad_input_packed = grad_input_.packed_accessor64<scalar_t, 3>();
        auto grad_output_packed =
            grad_output_.packed_accessor64<const scalar_t, 3>();

        replication_pad1d_backward_template<scalar_t>(
            grad_input_packed, grad_output_packed, pad_left, pad_right);
      });
}

void replication_pad2d_kernel(
    const Tensor& output,
    const Tensor& input,
    IntArrayRef padding) {
  TORCH_CHECK(
      canUse32BitIndexMath(input),
      "input tensor must fit into 32-bit index math");
  if (input.numel() == 0) {
    return;
  }
  const auto padL = padding[0];
  const auto padT = padding[2];
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      kHalf, kBFloat16, input.scalar_type(), "replication_pad2d_xpu", [&] {
        Tensor input_ = input;
        Tensor output_ = output;
        if (input.dim() == 3) {
          input_ = input.unsqueeze(0);
          output_ = output.unsqueeze(0);
        }
        auto devInput = input_.packed_accessor64<const scalar_t, 4>();
        auto devOutput = output_.packed_accessor64<scalar_t, 4>();
        replication_pad2d_forward_template<scalar_t>(
            devInput, devOutput, padT, padL);
      });
}

void replication_pad2d_backward_kernel(
    const Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef padding) {
  // See Note [Writing Nondeterministic Operations]
  // Nondeterministic because of atomicAdd usage
  globalContext().alertNotDeterministic("replication_pad2d_backward_xpu");
  TORCH_CHECK(
      canUse32BitIndexMath(input),
      "input tensor must fit into 32-bit index math");
  TORCH_CHECK(
      canUse32BitIndexMath(grad_output),
      "output gradient tensor must fit into 32-bit index math");
  TORCH_CHECK(padding.size() == 4, "padding Size is expected to be 4");

  const auto padL = padding[0];
  const auto padR = padding[1];
  const auto padT = padding[2];
  const auto padB = padding[3];
  int dimh = 1;
  int dimw = 2;

  int numInputDims = input.dim();
  if (numInputDims == 4) {
    dimh++;
    dimw++;
  }
  const auto iheight = input.size(dimh);
  const auto iwidth = input.size(dimw);
  const auto oheight = iheight + padT + padB;
  const auto owidth = iwidth + padL + padR;

  TORCH_CHECK(
      owidth == grad_output.size(dimw),
      "grad_output width unexpected. Expected: ",
      owidth,
      ", Got: ",
      grad_output.size(dimw));
  TORCH_CHECK(
      oheight == grad_output.size(dimh),
      "grad_output height unexpected. Expected: ",
      oheight,
      ", Got: ",
      grad_output.size(dimh));

  grad_input.resize_as_(input);
  if (grad_input.numel() == 0) {
    return;
  }
  grad_input.zero_();

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      kHalf,
      kBFloat16,
      input.scalar_type(),
      "replication_pad2d_backward_xpu",
      [&] {
        auto grad_input_ = grad_input;
        auto grad_output_ = grad_output;
        if (numInputDims == 3) {
          grad_input_ = grad_input.unsqueeze(0);
          grad_output_ = grad_output.unsqueeze(0);
        }
        auto grad_input_packed = grad_input_.packed_accessor64<scalar_t, 4>();
        auto grad_output_packed =
            grad_output_.packed_accessor64<const scalar_t, 4>();

        replication_pad2d_backward_template<scalar_t>(
            grad_input_packed, grad_output_packed, padT, padL);
      });
}

void replication_pad3d_kernel(
    const Tensor& output,
    const Tensor& input,
    IntArrayRef padding) {
  if (input.numel() == 0) {
    return;
  }
  int64_t pad_left = padding[0];
  int64_t pad_top = padding[2];
  int64_t pad_front = padding[4];

  int64_t num_input_dims = input.dim();

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      kHalf, kBFloat16, input.scalar_type(), "replication_pad3d_xpu", [&] {
        auto input_ = input;
        auto output_ = output;
        if (num_input_dims == 4) {
          input_ = input.unsqueeze(0);
          output_ = output.unsqueeze(0);
        }

        auto input_packed = input_.packed_accessor64<const scalar_t, 5>();
        auto output_packed = output_.packed_accessor64<scalar_t, 5>();

        replication_pad3d_forward_template<scalar_t>(
            input_packed, output_packed, pad_left, pad_top, pad_front);
      });
}

static inline void shapeAndGradOutputCheck3d(
    const Tensor& input,
    const Tensor& grad_output,
    int64_t pad_left,
    int64_t pad_right,
    int64_t pad_top,
    int64_t pad_bottom,
    int64_t pad_front,
    int64_t pad_back) {
  TORCH_CHECK(
      canUse32BitIndexMath(input),
      "input tensor must fit into 32-bit index math");
  int64_t num_input_dims = input.dim();

  bool valid_dims =
      input.size(1) != 0 && input.size(2) != 0 && input.size(3) != 0;
  TORCH_CHECK(
      (num_input_dims == 4 && valid_dims) ||
          (num_input_dims == 5 && valid_dims && input.size(4) != 0),
      "Expected 4D or 5D (batch mode) tensor with possibly 0 batch size and other non-zero dimensions for input, but got: ",
      input.sizes());

  int plane_dim = 0;
  int dimd = 1;
  int dimh = 2;
  int dimw = 3;
  if (num_input_dims == 5) {
    plane_dim++;
    dimd++;
    dimh++;
    dimw++;
  }

  int64_t num_planes = input.size(plane_dim);
  int64_t idepth = input.size(dimd);
  int64_t iheight = input.size(dimh);
  int64_t iwidth = input.size(dimw);
  int64_t odepth = idepth + pad_front + pad_back;
  int64_t oheight = iheight + pad_top + pad_bottom;
  int64_t owidth = iwidth + pad_left + pad_right;
  TORCH_CHECK(
      owidth >= 1 || oheight >= 1 || odepth >= 1,
      "input (D: ",
      idepth,
      " H: ",
      iheight,
      ", W: ",
      iwidth,
      ") is too small."
      " Calculated output D: ",
      odepth,
      " H: ",
      oheight,
      " W: ",
      owidth);

  TORCH_CHECK(
      canUse32BitIndexMath(grad_output),
      "output gradient tensor must fit into 32-bit index math");

  TORCH_CHECK(
      num_planes == grad_output.size(plane_dim),
      "grad_output width unexpected. Expected: ",
      num_planes,
      ", Got: ",
      grad_output.size(plane_dim));
  TORCH_CHECK(
      owidth == grad_output.size(dimw),
      "grad_output width unexpected. Expected: ",
      owidth,
      ", Got: ",
      grad_output.size(dimw));
  TORCH_CHECK(
      oheight == grad_output.size(dimh),
      "grad_output height unexpected. Expected: ",
      oheight,
      ", Got: ",
      grad_output.size(dimh));
  TORCH_CHECK(
      odepth == grad_output.size(dimd),
      "grad_output depth unexpected. Expected: ",
      odepth,
      ", Got: ",
      grad_output.size(dimd));
}

void replication_pad3d_backward_kernel(
    const Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef padding) {
  // See Note [Writing Nondeterministic Operations]
  // Nondeterministic because of atomicAdd usage
  globalContext().alertNotDeterministic("replication_pad3d_backward_xpu");
  TORCH_CHECK(padding.size() == 6, "padding Size is expected to be 6");

  int pad_left = padding[0];
  int pad_right = padding[1];
  int pad_top = padding[2];
  int pad_bottom = padding[3];
  int pad_front = padding[4];
  int pad_back = padding[5];
  shapeAndGradOutputCheck3d(
      input,
      grad_output,
      pad_left,
      pad_right,
      pad_top,
      pad_bottom,
      pad_front,
      pad_back);

  grad_input.resize_as_(input);
  if (grad_input.numel() == 0) {
    return;
  }
  grad_input.zero_();
  int num_input_dims = input.dim();

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      kHalf,
      kBFloat16,
      input.scalar_type(),
      "replication_pad3d_backward_xpu",
      [&] {
        auto grad_input_ = grad_input;
        auto grad_output_ = grad_output;
        if (num_input_dims == 4) {
          grad_input_ = grad_input.unsqueeze(0);
          grad_output_ = grad_output.unsqueeze(0);
        }
        auto grad_input_packed = grad_input_.packed_accessor64<scalar_t, 5>();
        auto grad_output_packed =
            grad_output_.packed_accessor64<const scalar_t, 5>();
        replication_pad3d_backward_template<scalar_t>(
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
