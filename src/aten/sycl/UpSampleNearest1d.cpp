#include <ATen/AccumulateType.h>
#include <ATen/core/Tensor.h>
#include <aten/sycl/UpSampleNearest1d.h>
#include <comm/Runtime.h>
#include <comm/SYCLContext.h>
#include <comm/SYCLHelpers.h>

namespace at ::native {
namespace xpu {
template <typename scalar_t, typename index_op_t>
struct UpsampleNearest1dOutKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    int dst_idx = item.get_global_linear_id();
    if (dst_idx >= dim_c * dst_dim_w)
      return;

    int c = (dst_idx / dst_dim_w) % dim_c;

    int dst_x = dst_idx % dst_dim_w;
    int src_x = index_op(scale_factor, dst_x, src_dim_w);

    int src_idx = c * src_dim_w + src_x;
    int src_stride = dim_c * src_dim_w;
    int dst_stride = dim_c * dst_dim_w;

    for (int b = 0; b < dim_b; b++) {
      output[dst_idx] = input[src_idx];
      src_idx += src_stride;
      dst_idx += dst_stride;
    }
  }
  UpsampleNearest1dOutKernelFunctor(
      int n_,
      const scalar_t* input_,
      size_t dim_b_,
      size_t dim_c_,
      size_t src_dim_w_,
      size_t dst_dim_w_,
      scalar_t* output_,
      float scale_factor_,
      index_op_t index_op_)
      : n(n_),
        input(input_),
        dim_b(dim_b_),
        dim_c(dim_c_),
        src_dim_w(src_dim_w_),
        dst_dim_w(dst_dim_w_),
        output(output_),
        scale_factor(scale_factor_),
        index_op(index_op_) {}

 private:
  int n;
  const scalar_t* input;
  size_t dim_b;
  size_t dim_c;
  size_t src_dim_w;
  size_t dst_dim_w;
  scalar_t* output;
  float scale_factor;
  index_op_t index_op;
};

template <typename scalar_t, typename index_op_t>
void upsample_nearest1d_out_frame(
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
  // auto dev_id = dpcppGetDeviceIdOfCurrentQueue();

  auto work_group_size = syclMaxWorkItemsPerEU();
  int global_range =
      (n + work_group_size - 1) / work_group_size * work_group_size;

  auto caller = UpsampleNearest1dOutKernelFunctor<scalar_t, index_op_t>(
      n,
      input,
      dim_b,
      dim_c,
      src_dim_w,
      dst_dim_w,
      output,
      scale_factor,
      index_op);

  sycl_kernel_submit(global_range, work_group_size, queue, caller);
}

void upsample_nearest1d_out_kernel(
    Tensor& output,
    const Tensor& input_,
    IntArrayRef output_size,
    c10::optional<double> scales,
    bool is_exact) {
  printf("in kernel\n");
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
      "upsample_nearest1d_out_frame",
      [&] {
        using accscalar_t = acc_type<scalar_t, true>;

        auto idata = input.data_ptr<scalar_t>();
        auto odata = output_c.data_ptr<scalar_t>();

        const float scale_factor =
            compute_scales_value<float>(scales, input_width, output_width);
        if (is_exact) {
          upsample_nearest1d_out_frame<scalar_t>(
              n,
              idata,
              nbatch,
              channels,
              input_width,
              output_width,
              odata,
              scale_factor,
              Nearest_exact_index_op());
        } else {
          upsample_nearest1d_out_frame<scalar_t>(
              n,
              idata,
              nbatch,
              channels,
              input_width,
              output_width,
              odata,
              scale_factor,
              Nearest_index_op());
        }
      });
  if (!output.is_contiguous()) {
    output.copy_(output_c);
  }
}
} // namespace xpu
} // namespace at::native