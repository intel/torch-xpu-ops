#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/ScalarOps.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/DistributionTemplates.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/xpu/XPUNativeFunctions.h>

#include <ATen/native/xpu/sycl/DistributionKernels.h>
#include <ATen/native/xpu/sycl/Distributions.h>
#include <ATen/native/xpu/sycl/MultinomialKernel.h>
#include <ATen/ops/div.h>

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

template <typename RNG>
struct ExponentialStub {
  void operator()(
      TensorIteratorBase& iter,
      double lambda,
      c10::optional<Generator> gen) {
    native::xpu::exponential_kernel(iter, lambda, gen);
  }
};

Tensor& XPUNativeFunctions::exponential_(
    Tensor& self,
    double lambda,
    std::optional<Generator> generator) {
  return native::templates::exponential_impl_<ExponentialStub, Generator>(
      self, lambda, std::move(generator));
}

/* The largest consecutive integer representable in float32 (2^24) */
constexpr int64_t FLOAT32_MAX_CONSECUTIVE_INT = 1 << (24);

Tensor& XPUNativeFunctions::multinomial_out(
    const Tensor& self,
    int64_t n_sample,
    bool with_replacement,
    ::std::optional<at::Generator> gen,
    at::Tensor& result) {
  TORCH_CHECK(
      result.device() == self.device(),
      "multinomial arguments must have the same device");
  TORCH_CHECK(
      self.dim() > 0 && self.dim() <= 2, "prob_dist must be 1 or 2 dim");
  TORCH_CHECK(
      at::isFloatingType(self.scalar_type()),
      "multinomial only supports floating-point dtypes for input, got: ",
      self.scalar_type());
  TORCH_CHECK(
      result.scalar_type() == ScalarType::Long,
      "multinomial expects Long tensor out, got: ",
      result.scalar_type());
  TORCH_CHECK(n_sample > 0, "cannot sample n_sample <= 0 samples");
  int64_t n_categories = self.size(-1);
  TORCH_CHECK(
      with_replacement || (n_sample <= n_categories),
      "cannot sample n_sample > prob_dist.size(-1) samples without replacement");
  // Since the index tensor is float, numCategories cannot exceed max
  // float integer precision
  TORCH_CHECK(
      n_categories <= FLOAT32_MAX_CONSECUTIVE_INT,
      "number of categories cannot exceed 2^24");

  if (self.dim() == 1) {
    result.resize_({n_sample});
  } else {
    const int64_t n_dist = self.size(0);
    result.resize_({n_dist, n_sample});
  }
  if (result.numel() == 0) {
    return result;
  }

  // Fast-path for no replacement or if only one sample is drawn.
  // Reference:
  // https://github.com/pytorch/pytorch/issues/11931#issuecomment-625882503
  if (!with_replacement || n_sample == 1) {
    // Sanity checks on `self`.
    auto is_valid = ((self.max() < INFINITY) & (self.min() >= 0)).item();
    TORCH_CHECK(
        is_valid.to<bool>(),
        "probability tensor contains either `inf`, `nan` or element < 0");
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    bool zero_prob_condition;
    if (self.dim() == 1) {
      zero_prob_condition = (self.sum() == 0).item().to<bool>();
    } else {
      zero_prob_condition = (self.sum(1) == 0).sum().item().to<bool>();
    }
    TORCH_CHECK(
        !zero_prob_condition,
        "invalid multinomial distribution (sum of probabilities <= 0)");

    // The algorithm is from gumbel softmax.
    // s = argmax( logp - log(-log(eps)) ) where eps ~ U(0, 1)
    // Here we can apply exp to the formula which will not affect result of
    // argmax or topk. Then we have
    // s = argmax( p / (-log(eps)) ) where eps ~ U(0, 1).
    // We can also simplify the formula above by
    // s = argmax( p / q ) where q ~ Exp(1)
    Tensor q = at::empty_like(self).exponential_(1, std::move(gen));
    // In theory the probability to generate 0 from exponential distribution is
    // 0. However, on CUDA side there is a protection to avoid 0s, but on CPU
    // side, there is a very low probability to generate 0 from
    // exponential<double>. The probability is about 2^(-DBL_MANT_DIG). We just
    // ignore it here, but there may be some risk to get invalid output on CPU.
    at::div_out(q, self, q);
    if (n_sample == 1) {
      at::argmax_out(result, q, /*dim=*/-1, /*keepdim=*/true);
    } else {
      Tensor vals = at::empty(result.sizes(), self.options());
      at::topk_out(vals, result, q, n_sample);
    }
    return result;
  }

  at::native::xpu::multinomial_kernel(result, self, n_sample, gen);
  return result;
}

Tensor XPUNativeFunctions::multinomial(
    const Tensor& self,
    int64_t n_sample,
    bool with_replacement,
    ::std::optional<at::Generator> gen) {
  Tensor result = at::empty({0}, self.options().dtype(kLong));

  XPUNativeFunctions::multinomial_out(
      self, n_sample, with_replacement, std::move(gen), result);
  return result;
}

template <typename RNG>
struct CauchyStub {
  void operator()(
      TensorIteratorBase& iter,
      double median,
      double sigma,
      c10::optional<Generator> gen) {
    native::xpu::cauchy_kernel(iter, median, sigma, gen);
  }
};

Tensor& XPUNativeFunctions::cauchy_(
    Tensor& self,
    double median,
    double sigma,
    ::std::optional<Generator> generator) {
  return native::templates::cauchy_impl_<CauchyStub, Generator>(
      self, median, sigma, std::move(generator));
}

Tensor XPUNativeFunctions::binomial(
    const Tensor& count,
    const Tensor& prob,
    ::std::optional<Generator> generator) {
  auto gen = get_generator_or_default<at::XPUGeneratorImpl>(
      generator, at::xpu::detail::getDefaultXPUGenerator());
  Tensor ret = at::empty(count.sizes(), count.options());
  at::TensorIterator iter = at::TensorIteratorConfig()
                                .add_output(ret)
                                .add_input(count)
                                .add_input(prob)
                                .build();
  at::native::xpu::launch_binomial_xpu_kernel(iter, gen);
  return ret;
}

Tensor& XPUNativeFunctions::binomial_out(
    const Tensor& count,
    const Tensor& prob,
    ::std::optional<Generator> generator,
    Tensor& out) {
  out = XPUNativeFunctions::binomial(count, prob, generator);
  return out;
}

Tensor XPUNativeFunctions::_sample_dirichlet(
    const Tensor& alpha,
    ::std::optional<Generator> generator) {
  auto gen = get_generator_or_default<at::XPUGeneratorImpl>(
      generator, at::xpu::detail::getDefaultXPUGenerator());
  Tensor ret = at::empty(alpha.sizes(), alpha.options());
  at::native::xpu::launch_gamma_kernel(ret, alpha, gen);
  auto gamma_sum = ret.sum(/*dim=*/-1, /*keepdim=*/true);
  auto iter = at::TensorIteratorConfig()
                  .add_output(ret)
                  .add_input(ret)
                  .add_input(gamma_sum)
                  .build();
  at::native::xpu::launch_dirichlet_kernel(iter);
  return ret;
}

Tensor& XPUNativeFunctions::_sample_dirichlet_out(
    const Tensor& alpha,
    ::std::optional<Generator> generator,
    Tensor& out) {
  out = XPUNativeFunctions::_sample_dirichlet(alpha, generator);
  return out;
}

Tensor XPUNativeFunctions::_dirichlet_grad(
    const Tensor& x,
    const Tensor& alpha,
    const Tensor& total) {
  Tensor ret = at::empty(x.sizes(), x.options());
  auto iter = at::TensorIteratorConfig()
                  .add_output(ret)
                  .add_input(x)
                  .add_input(alpha)
                  .add_input(total)
                  .build();
  at::native::xpu::launch_dirichlet_grad_kernel(iter);
  return ret;
}

Tensor& XPUNativeFunctions::_dirichlet_grad_out(
    const Tensor& x,
    const Tensor& alpha,
    const Tensor& total,
    Tensor& out) {
  out = XPUNativeFunctions::_dirichlet_grad(x, alpha, total);
  return out;
}
} // namespace at
