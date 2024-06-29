#include <ATen/ScalarOps.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/Resize.h>
#include <ATen/xpu/XPUNativeFunctions.h>

#include <ATen/native/xpu/sycl/ActivationGluKernels.h>

namespace at {

TensorIterator glu_meta(const Tensor& self, int64_t dim, Tensor& out) {
  // this can't pass anyway because a 0-dimensional tensor has "size" 1, which
  // can't be evenly halved, but give a nicer error message here.
  TORCH_CHECK(self.dim() > 0, "glu does not support 0-dimensional tensors");
  auto wrap_dim = maybe_wrap_dim(dim, self.dim());
  const int64_t nIn = self.size(wrap_dim);
  TORCH_CHECK(
      nIn % 2 == 0,
      "Halving dimension must be even, but dimension ",
      wrap_dim,
      " is size ",
      nIn);

  // size output to half of input
  const int64_t selfSize = nIn / 2;
  Tensor firstHalf = self.narrow(wrap_dim, 0, selfSize);
  Tensor secondHalf = self.narrow(wrap_dim, selfSize, selfSize);
  return TensorIterator::borrowing_binary_op(out, firstHalf, secondHalf);
}

Tensor& XPUNativeFunctions::glu_out(
    const Tensor& self,
    int64_t dim,
    Tensor& out) {
  auto iter = glu_meta(self, dim, out);
  native::xpu::glu_kernel(iter);
  return out;
}

Tensor XPUNativeFunctions::glu(const Tensor& self, int64_t dim) {
  Tensor out;
  auto iter = glu_meta(self, dim, out);
  native::xpu::glu_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::glu_backward_out(
    const Tensor& grad_output,
    const Tensor& input,
    int64_t dim,
    Tensor& grad_input) {
  TORCH_CHECK(input.dim() > 0, "glu does not support 0-dimensional tensors");
  auto wrap_dim = maybe_wrap_dim(dim, input.dim());
  auto input_sizes = input.sizes();
  const int64_t nIn = input_sizes[wrap_dim];
  TORCH_CHECK(
      nIn % 2 == 0,
      "Halving dimension must be even, but dimension ",
      wrap_dim,
      " is size ",
      nIn);

  native::resize_output(grad_input, input_sizes);

  DimVector iter_shape(input_sizes);
  const auto dim_size = nIn / 2;
  iter_shape[wrap_dim] = dim_size;
  TORCH_CHECK(grad_output.sizes() == IntArrayRef{iter_shape});

  const auto iter = at::TensorIteratorConfig()
                        .add_output(grad_input)
                        .add_const_input(input)
                        .add_const_input(grad_output)
                        .resize_outputs(false)
                        .declare_static_shape(iter_shape)
                        .build();

  if (iter.numel() == 0) {
    return grad_input;
  }

  const auto I_stride = input.strides()[wrap_dim] * dim_size;
  const auto gI_stride = grad_input.strides()[wrap_dim] * dim_size;

  if (iter.can_use_32bit_indexing()) {
    native::xpu::glu_backward_kernel(iter, gI_stride, I_stride);
  } else {
    for (const auto& sub_iter : iter.with_32bit_indexing()) {
      native::xpu::glu_backward_kernel(sub_iter, gI_stride, I_stride);
    }
  }
  return grad_input;
}

Tensor XPUNativeFunctions::glu_backward(
    const Tensor& grad_output,
    const Tensor& input,
    int64_t dim) {
  auto grad_input = at::empty({0}, input.options());
  return glu_backward_out(grad_output, input, dim, grad_input);
}

} // namespace at
