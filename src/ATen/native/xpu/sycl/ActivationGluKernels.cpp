#include <ATen/ATen.h>

namespace at::native::xpu {

template <typename scalar_t>
static void GatedLinearUnit_updateOutput(
    Tensor& output,
    const Tensor& input,
    int64_t dim) {
  auto wrap_dim = maybe_wrap_dim(dim, input.dim());
  const int64_t nln = input.size(wrap_dim);
  const int64_t inputSize = nln / 2;
  Tensor firstHalf = input.narrow(wrap_dim, 0, inputSize);
  Tensor secondHalf = input.narrow(wrap_dim, inputSize, inputSize);
  // output = output + firstHalf * sigmoid(secondHalf)
  Tensor sigNum = at::empty_like(secondHalf);
  at::sigmoid_out(sigNum, secondHalf);
  output = at::mul(firstHalf, sigNum);
}

void glu_kernel(const Tensor& self, int64_t dim, Tensor& out) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      self.scalar_type(),
      "glu_xpu",
      [&] { GatedLinearUnit_updateOutput<scalar_t>(out, self, dim); });
}

template <typename scalar_t>
static void GatedLinearUnit_updateGradInput(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    int64_t dim) {
  auto wrap_dim = maybe_wrap_dim(dim, input.dim());
  const int64_t nln = input.size(wrap_dim);

  grad_input.resize_as_(input);
  const int64_t inputSize = nln / 2;
  Tensor firstHalf = input.narrow(wrap_dim, 0, inputSize);
  Tensor secondHalf = input.narrow(wrap_dim, inputSize, inputSize);
  Tensor gradInputfirstHalf = grad_input.narrow(wrap_dim, 0, inputSize);
  Tensor gradInputsecondHalf =
      grad_input.narrow(wrap_dim, inputSize, inputSize);

  // gradInputfirstHalf = grad_output * sigmoid(secondHalf)
  // gradInputsecondHalf = (1 - sigmoid(secondHalf)) * sigmoid(secondHalf) *
  // input * grad_output
  at::sigmoid_out(gradInputfirstHalf, secondHalf);
  gradInputsecondHalf.fill_((scalar_t)(1));
  gradInputsecondHalf.sub_(gradInputfirstHalf)
      .mul_(gradInputfirstHalf)
      .mul_(firstHalf);
  gradInputfirstHalf.mul_(grad_output);
  gradInputsecondHalf.mul_(grad_output);
}

void glu_backward_kernel(
    const Tensor& grad_output,
    const Tensor& self,
    int64_t dim,
    Tensor& grad_input) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      self.scalar_type(),
      "glu_backward_xpu",
      [&] {
        GatedLinearUnit_updateGradInput<scalar_t>(
            grad_input, grad_output, self, dim);
      });
}

} // namespace at::native::xpu