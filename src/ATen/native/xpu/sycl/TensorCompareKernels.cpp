#include <ATen/Dispatch.h>
#include <ATen/NumericUtils.h>
#include <ATen/native/TensorCompare.h>
#include <ATen/native/TensorIterator.h>
#include <comm/xpu_aten.h>

#include <ATen/native/xpu/sycl/Loops.h>

#include <ATen/native/xpu/sycl/TensorCompareKernels.h>

namespace at {
namespace native {
namespace xpu {

template <typename scalar_t>
struct WhereFunctor {
  scalar_t operator()(bool cond_val, scalar_t self_val, scalar_t other_val)
      const {
    return cond_val ? self_val : other_val;
  }
};

template <typename scalar_t>
struct IsposinfFunctor {
  bool operator()(scalar_t a) const {
    return a == std::numeric_limits<scalar_t>::infinity();
  }
};

template <typename scalar_t>
struct IsneginfFunctor {
  bool operator()(scalar_t a) const {
    return a == -std::numeric_limits<scalar_t>::infinity();
  }
};

template <typename scalar_t>
struct ClampFunctor {
  scalar_t operator()(scalar_t v, scalar_t lower, scalar_t upper) const {
    if (at::_isnan(v)) {
      return v;
    }
    if (at::_isnan(lower)) {
      return lower;
    }
    if (at::_isnan(upper)) {
      return upper;
    } else {
      return std::min(std::max(v, lower), upper);
    }
  }
};

template <typename scalar_t>
struct ClampScalarFunctor {
  using opmath_t = at::opmath_type<scalar_t>;
  scalar_t operator()(scalar_t v) const {
    if (_isnan(static_cast<opmath_t>(v))) {
      return v;
    } else if (minmax_ == at::native::detail::ClampLimits::Min) {
      return std::max(static_cast<opmath_t>(v), lim0_val_);
    } else if (minmax_ == at::native::detail::ClampLimits::Max) {
      return std::min(static_cast<opmath_t>(v), lim0_val_);
    } else {
      return std::min(std::max(static_cast<opmath_t>(v), lim0_val_), lim1_val_);
    }
  }
  ClampScalarFunctor(
      opmath_t lim0_val,
      opmath_t lim1_val,
      at::native::detail::ClampLimits minmax)
      : lim0_val_(lim0_val), lim1_val_(lim1_val), minmax_(minmax) {}

 private:
  opmath_t lim0_val_;
  opmath_t lim1_val_;
  at::native::detail::ClampLimits minmax_;
};

void where_kernel(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
      kComplexHalf, kHalf, kBFloat16, kBool, iter.dtype(), "where_xpu", [&] {
        gpu_kernel(iter, WhereFunctor<scalar_t>());
      });
}

void isposinf_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.input_dtype(),
      "isposinf_xpu",
      [&] { gpu_kernel(iter, IsposinfFunctor<scalar_t>()); });
}

void isneginf_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.input_dtype(),
      "isneginf_xpu",
      [&] { gpu_kernel(iter, IsneginfFunctor<scalar_t>()); });
}

void clamp_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_ALL_TYPES_AND2(
      kHalf, kBFloat16, iter.common_dtype(), "clamp_xpu", [&] {
        gpu_kernel(iter, ClampFunctor<scalar_t>());
      });
}

void inline launch_clamp_scalar(
    TensorIteratorBase& iter,
    Scalar lim0,
    Scalar lim1,
    at::native::detail::ClampLimits minmax) {
  AT_DISPATCH_ALL_TYPES_AND2(
      kHalf, kBFloat16, iter.common_dtype(), "clamp_scalar_xpu", [&] {
        using opmath_t = at::opmath_type<scalar_t>;
        auto lim0_val = lim0.to<opmath_t>();
        auto lim1_val = lim1.to<opmath_t>();
        gpu_kernel(
            iter, ClampScalarFunctor<scalar_t>(lim0_val, lim1_val, minmax));
      });
}

void clamp_scalar_kernel(
    TensorIteratorBase& iter,
    const Scalar& min,
    const Scalar& max) {
  launch_clamp_scalar(iter, min, max, at::native::detail::ClampLimits::MinMax);
}

void clamp_min_scalar_kernel(TensorIteratorBase& iter, Scalar min) {
  launch_clamp_scalar(iter, min, min, at::native::detail::ClampLimits::Min);
}

void clamp_max_scalar_kernel(TensorIteratorBase& iter, Scalar max) {
  launch_clamp_scalar(iter, max, max, at::native::detail::ClampLimits::Max);
}

void isin_kernel(
    const Tensor& elements,
    const Tensor& test_elements,
    bool invert,
    const Tensor& out) {
  std::vector<int64_t> bc_shape(elements.dim(), 1);
  bc_shape.push_back(-1);
  out.copy_(
      invert ? elements.unsqueeze(-1).ne(test_elements.view(bc_shape)).all(-1)
             : elements.unsqueeze(-1).eq(test_elements.view(bc_shape)).any(-1));
}

struct Msg {
  static constexpr size_t MAX_MSG_LENGTH = 256;
  char msg[MAX_MSG_LENGTH];
};

// SYCL_KERNEL_ASSERT_MSG is not ready
template <typename scalar_t>
struct AssertAsyncKernelFunctor1 {
  void operator()(sycl::nd_item<1> item) const {
    SYCL_KERNEL_ASSERT(input_[0] != 0);
  }
  AssertAsyncKernelFunctor1(const scalar_t* input, Msg msg)
      : input_(input), msg_(msg) {}

 private:
  const scalar_t* input_;
  Msg msg_;
};

struct AssertAsyncKernelFunctor2 {
  void operator()(sycl::nd_item<1> item) const {
    SYCL_KERNEL_ASSERT(input_[0] != c10::complex<float>(0, 0));
  }
  AssertAsyncKernelFunctor2(const c10::complex<float>* input, Msg msg)
      : input_(input), msg_(msg) {}

 private:
  const c10::complex<float>* input_;
  Msg msg_;
};

struct AssertAsyncKernelFunctor3 {
  void operator()(sycl::nd_item<1> item) const {
    SYCL_KERNEL_ASSERT(input_[0] != c10::complex<double>(0, 0));
  }
  AssertAsyncKernelFunctor3(const c10::complex<double>* input, Msg msg)
      : input_(input), msg_(msg) {}

 private:
  const c10::complex<double>* input_;
  Msg msg_;
};

template <typename scalar_t>
void launch_assert_async_kernel(const scalar_t* input, Msg msg) {
  AssertAsyncKernelFunctor1<scalar_t> kfn(input, msg);
  sycl_kernel_submit(1, 1, getCurrentSYCLQueue(), kfn);
}

template <>
void launch_assert_async_kernel(const c10::complex<float>* input, Msg msg) {
  AssertAsyncKernelFunctor2 kfn(input, msg);
  sycl_kernel_submit(1, 1, getCurrentSYCLQueue(), kfn);
}

template <>
void launch_assert_async_kernel(const c10::complex<double>* input, Msg msg) {
  AssertAsyncKernelFunctor3 kfn(input, msg);
  sycl_kernel_submit(1, 1, getCurrentSYCLQueue(), kfn);
}

void _assert_async_msg_kernel(
    const Tensor& self_tensor,
    std::string_view assert_msg) {
  const TensorBase& self = get_tensor_base(self_tensor);
  auto n = self.numel();
  TORCH_CHECK(n != 0, "Boolean value of Tensor with no values is ambiguous");
  TORCH_CHECK(
      n < 2, "Boolean value of Tensor with more than one value is ambiguous");
  Msg msg;
  size_t copy_length = assert_msg.length();
  TORCH_CHECK(
      copy_length < Msg::MAX_MSG_LENGTH - 1,
      "Message length must be smaller than " +
          std::to_string(Msg::MAX_MSG_LENGTH - 1));
  std::copy_n(assert_msg.data(), copy_length, msg.msg);
  msg.msg[copy_length] = '\0'; // Ensure null-termination
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      at::ScalarType::Half,
      at::ScalarType::Bool,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "_assert_async_xpu",
      [&] {
        launch_assert_async_kernel<scalar_t>(
            self.const_data_ptr<scalar_t>(), msg);
      });
}

} // namespace xpu
} // namespace native
} // namespace at
