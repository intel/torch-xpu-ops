#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/ScalarOps.h>
#include <ATen/xpu/XPUNativeFunctions.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/DistributionTemplates.h>
#include <ATen/native/TensorIterator.h>

#include <ATen/native/xpu/sycl/DistributionKernels.h>

namespace at {

template <typename RNG>
struct NormalStub {
  void operator()(
      Tensor& self,
      double mean,
      double std,
      c10::optional<Generator> gen) {
    native::xpu::normal_kernel(self, mean, std, gen);
  }
};

Tensor& XPUNativeFunctions::normal_(
    Tensor& self,
    double mean,
    double std,
    ::std::optional<Generator> generator) {
  return native::templates::normal_impl_<NormalStub, Generator>(
      self, mean, std, std::move(generator));
}

// out tensor float
Tensor& XPUNativeFunctions::normal_out(
    const Tensor& mean,
    double std,
    c10::optional<Generator> gen,
    Tensor& output) {
  return at::native::templates::normal_out_impl<NormalStub, Generator>(
      output, mean, std, std::move(gen));
}

// functional tensor float
Tensor XPUNativeFunctions::normal(
    const Tensor& mean,
    double std,
    c10::optional<Generator> gen) {
  return at::native::templates::normal_impl<NormalStub, Generator>(
      mean, std, std::move(gen));
}

// out float tensor
Tensor& XPUNativeFunctions::normal_out(
    double mean,
    const Tensor& std,
    c10::optional<Generator> gen,
    Tensor& output) {
  return at::native::templates::normal_out_impl<NormalStub, Generator>(
      output, mean, std, std::move(gen));
}

// functional float tensor
Tensor XPUNativeFunctions::normal(
    double mean,
    const Tensor& std,
    c10::optional<Generator> gen) {
  return at::native::templates::normal_impl<NormalStub, Generator>(
      mean, std, std::move(gen));
}

// out tensor tensor
Tensor& XPUNativeFunctions::normal_out(
    const Tensor& mean,
    const Tensor& std,
    c10::optional<Generator> gen,
    Tensor& output) {
  return at::native::templates::normal_out_impl<NormalStub, Generator>(
      output, mean, std, std::move(gen));
}

// functional tensor tensor
Tensor XPUNativeFunctions::normal(
    const Tensor& mean,
    const Tensor& std,
    c10::optional<Generator> gen) {
  return at::native::templates::normal_impl<NormalStub, Generator>(
      mean, std, std::move(gen));
}

template <typename RNG>
struct UniformStub {
  void operator()(
      TensorIteratorBase& iter,
      double from,
      double to,
      c10::optional<Generator> gen) {
    native::xpu::uniform_kernel(iter, from, to, gen);
  }
};

Tensor& XPUNativeFunctions::uniform_(
    Tensor& self,
    double from,
    double to,
    ::std::optional<Generator> generator) {
  return native::templates::uniform_impl_<UniformStub, Generator>(
      self, from, to, std::move(generator));
}

template <typename RNG>
struct BernoulliStub {
  void operator()(
      Tensor& self,
      const Tensor& p_,
      c10::optional<Generator> gen) {
    native::xpu::bernoulli_tensor_kernel(self, p_, gen);
  }
  void operator()(Tensor& self, double p, c10::optional<Generator> gen) {
    native::xpu::bernoulli_scalar_kernel(self, p, gen);
  }
};

Tensor& XPUNativeFunctions::bernoulli_(
    Tensor& self,
    const Tensor& p_,
    ::std::optional<Generator> generator) {
  return native::templates::bernoulli_impl_<BernoulliStub, Generator>(
      self, p_, std::move(generator));
}

Tensor& XPUNativeFunctions::bernoulli_(
    Tensor& self,
    double p,
    ::std::optional<Generator> generator) {
  return native::templates::bernoulli_impl_<BernoulliStub, Generator>(
      self, p, std::move(generator));
}

Tensor& XPUNativeFunctions::bernoulli_out(
    const Tensor& self,
    c10::optional<Generator> gen,
    Tensor& result) {
  return native::templates::bernoulli_out_impl<BernoulliStub, Generator>(
      result, self, std::move(gen));
}

template <typename RNG>
struct RandomStub {
  void operator()(TensorIteratorBase& iter, c10::optional<Generator> gen) {
    native::xpu::random_kernel(iter, gen);
  }
};

Tensor& XPUNativeFunctions::random_(
    Tensor& self,
    ::std::optional<Generator> generator) {
  return native::templates::random_impl<RandomStub, Generator>(
      self, std::move(generator));
}

template <typename RNG>
struct RandomFromToStub {
  void operator()(
      TensorIteratorBase& iter,
      uint64_t range,
      int64_t from,
      c10::optional<Generator> gen) {
    native::xpu::random_from_to_kernel(iter, range, from, gen);
  }
  void operator()(TensorIteratorBase& iter, c10::optional<Generator> gen) {
    native::xpu::random_full_64_bits_range_kernel(iter, gen);
  }
};

Tensor& XPUNativeFunctions::random_(
    Tensor& self,
    int64_t from,
    c10::optional<int64_t> to_opt,
    ::std::optional<Generator> generator) {
  return native::templates::random_from_to_impl<RandomFromToStub, Generator>(
      self, from, to_opt, std::move(generator));
}

Tensor& XPUNativeFunctions::random_(
    Tensor& self,
    int64_t to,
    ::std::optional<Generator> generator) {
  return random_(self, 0, to, std::move(generator));
}

} // namespace at
