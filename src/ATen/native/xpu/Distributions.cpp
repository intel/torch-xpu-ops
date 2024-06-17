#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/ScalarOps.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>
#include <comm/xpu_aten.h>

#include <ATen/native/DistributionTemplates.h>
#include <ATen/native/Distributions.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/UnaryOps.h>
// #include <ATen/xpu/XPUNativeFunctions.h>
#include <ATen/native/xpu/sycl/DistributionKernels.h>

namespace at {

namespace native {
REGISTER_XPU_DISPATCH(normal_stub, xpu::normal_kernel);
REGISTER_XPU_DISPATCH(uniform_stub, xpu::uniform_kernel);
REGISTER_XPU_DISPATCH(bernoulli_scalar_stub, xpu::bernoulli_scalar_kernel);
REGISTER_XPU_DISPATCH(bernoulli_tensor_stub, xpu::bernoulli_tensor_kernel);
REGISTER_XPU_DISPATCH(random_from_to_stub, native::xpu::random_from_to_kernel);
REGISTER_XPU_DISPATCH(
    random_full_64_bits_range_stub,
    native::xpu::random_full_64_bits_range_kernel);
} // namespace native

// template <typename RNG>
// struct NormalStub {
//   void operator()(
//       Tensor& self,
//       double mean,
//       double std,
//       c10::optional<Generator> gen) {
//     native::xpu::normal_kernel(self, mean, std, gen);
//   }
// };

// Tensor& XPUNativeFunctions::normal_(
//     Tensor& self,
//     double mean,
//     double std,
//     ::std::optional<Generator> generator) {
//   return native::templates::normal_impl_<NormalStub, Generator>(
//       self, mean, std, std::move(generator));
// }

// template <typename RNG>
// struct UniformStub {
//   void operator()(
//       TensorIteratorBase& iter,
//       double from,
//       double to,
//       c10::optional<Generator> gen) {
//     native::xpu::uniform_kernel(iter, from, to, gen);
//   }
// };

// Tensor& XPUNativeFunctions::uniform_(
//     Tensor& self,
//     double from,
//     double to,
//     ::std::optional<Generator> generator) {
//   return native::templates::uniform_impl_<UniformStub, Generator>(
//       self, from, to, std::move(generator));
// }

// template <typename RNG>
// struct BernoulliStub {
//   void operator()(
//       Tensor& self,
//       const Tensor& p_,
//       c10::optional<Generator> gen) {
//     native::xpu::bernoulli_tensor_kernel(self, p_, gen);
//   }
//   void operator()(Tensor& self, double p, c10::optional<Generator> gen) {
//     native::xpu::bernoulli_scalar_kernel(self, p, gen);
//   }
// };

// Tensor& XPUNativeFunctions::bernoulli_(
//     Tensor& self,
//     const Tensor& p_,
//     ::std::optional<Generator> generator) {
//   return native::templates::bernoulli_impl_<BernoulliStub, Generator>(
//       self, p_, std::move(generator));
// }

// Tensor& XPUNativeFunctions::bernoulli_(
//     Tensor& self,
//     double p,
//     ::std::optional<Generator> generator) {
//   return native::templates::bernoulli_impl_<BernoulliStub, Generator>(
//       self, p, std::move(generator));
// }

// template <typename RNG>
// struct RandomStub {
//   void operator()(TensorIteratorBase& iter, c10::optional<Generator> gen) {
//     native::xpu::random_kernel(iter, gen);
//   }
// };

// Tensor& XPUNativeFunctions::random_(
//     Tensor& self,
//     ::std::optional<Generator> generator) {
//   return native::templates::random_impl<RandomStub, Generator>(
//       self, std::move(generator));
// }

// template <typename RNG>
// struct RandomFromToStub {
//   void operator()(
//       TensorIteratorBase& iter,
//       uint64_t range,
//       int64_t from,
//       c10::optional<Generator> gen) {
//     native::xpu::random_from_to_kernel(iter, range, from, gen);
//   }
//   void operator()(TensorIteratorBase& iter, c10::optional<Generator> gen) {
//     native::xpu::random_full_64_bits_range_kernel(iter, gen);
//   }
// };

// Tensor& random_(
//     Tensor& self,
//     int64_t from,
//     c10::optional<int64_t> to_opt,
//     ::std::optional<Generator> generator) {
//   return native::templates::random_from_to_impl<RandomFromToStub, Generator>(
//       self, from, to_opt, std::move(generator));
// }

} // namespace at
