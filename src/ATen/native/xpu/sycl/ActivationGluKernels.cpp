#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/TensorIterator.h>

#include <ATen/native/xpu/sycl/Loops.h>
#include <comm/SYCLContext.h>

#include <ATen/native/xpu/sycl/ActivationGluKernels.h>

namespace at::native::xpu {

template <typename scalar_t>
struct GluFunctor {
  using opmath_t = at::opmath_type<scalar_t>;
  scalar_t operator()(scalar_t a_, scalar_t b_) const {
    const opmath_t a = a_;
    const opmath_t b = b_;
    const opmath_t one = opmath_t(1);
    const opmath_t sigmoid = one / (one + std::exp(-b));
    return a * sigmoid;
  }
};

void glu_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      iter.dtype(),
      "glu_xpu",
      [&] { gpu_kernel(iter, GluFunctor<scalar_t>()); });
}

template <typename scalar_t>
struct GluJvpFunctor {
  using opmath_t = at::opmath_type<scalar_t>;
  scalar_t operator()(scalar_t res_, scalar_t b_, scalar_t da_, scalar_t db_)
      const {
    const opmath_t res = res_;
    const opmath_t b = b_;
    const opmath_t da = da_;
    const opmath_t db = db_;
    const opmath_t one = opmath_t(1);

    const opmath_t sig_b = one / (one + std::exp(-b));
    return (da * sig_b + res * (db - sig_b * db));
  }
};

void glu_jvp_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      iter.dtype(),
      "glu_xpu",
      [&] { gpu_kernel(iter, GluJvpFunctor<scalar_t>()); });
}

// Byte offsets don't require multiplication by sizeof(T), so are slightly
// cheaper. For fixed offsets, this removes all penalty from 64-bit indexing.
template <typename T>
T* byte_offset(T* ptr, int64_t offset) {
  using byte_ptr_t = typename std::
      conditional<std::is_const<T>::value, const char*, char*>::type;
  return reinterpret_cast<T*>(reinterpret_cast<byte_ptr_t>(ptr) + offset);
}

template <typename scalar_t, typename OffsetCalc>
struct GluBackwardKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    using opmath_t = at::opmath_type<scalar_t>;

    const uint32_t linear_index = item.get_global_linear_id();
    if (linear_index >= numel_) {
      return;
    }
    const auto offsets = offset_calculator_.get(linear_index);

    const opmath_t a = I_[offsets[1]];
    const opmath_t b = *byte_offset(I_ + offsets[1], I_byte_offset_);
    const opmath_t gO_val = gO_[offsets[2]];

    const auto one = opmath_t(1);
    const opmath_t sigmoid = one / (one + std::exp(-b));

    auto* gA = gI_ + offsets[0];
    *gA = sigmoid * gO_val;

    auto* gB = byte_offset(gA, gI_byte_offset_);
    *gB = (one - sigmoid) * sigmoid * gO_val * a;
  }

  GluBackwardKernelFunctor(
      int numel,
      scalar_t* gI,
      const scalar_t* I,
      const scalar_t* gO,
      OffsetCalc offset_calculator,
      int64_t gI_byte_offset,
      int64_t I_byte_offset)
      : numel_(numel),
        gI_(gI),
        I_(I),
        gO_(gO),
        offset_calculator_(offset_calculator),
        gI_byte_offset_(gI_byte_offset),
        I_byte_offset_(I_byte_offset) {}

 private:
  int numel_;
  scalar_t* gI_;
  const scalar_t* I_;
  const scalar_t* gO_;
  OffsetCalc offset_calculator_;
  int64_t gI_byte_offset_;
  int64_t I_byte_offset_;
};

template <typename scalar_t, typename OffsetCalc>
void launch_glu_backward_kernel(
    int numel,
    scalar_t* gI,
    const scalar_t* I,
    const scalar_t* gO,
    OffsetCalc offset_calculator,
    int64_t gI_byte_offset,
    int64_t I_byte_offset) {
  GluBackwardKernelFunctor<scalar_t, OffsetCalc> kfn(
      numel, gI, I, gO, offset_calculator, gI_byte_offset, I_byte_offset);

  const int64_t local_size = syclMaxWorkGroupSize(kfn);
  const int64_t num_wg = (numel + local_size - 1) / local_size;
  const int64_t global_size = num_wg * local_size;

  sycl_kernel_submit(global_size, local_size, getCurrentSYCLQueue(), kfn);
}

void glu_backward_kernel(
    const TensorIteratorBase& iter,
    int64_t gI_stride,
    int64_t I_stride) {
  const auto N = iter.numel();
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      N > 0 && N <= std::numeric_limits<int32_t>::max());
  const auto offset_calculator = make_element_offset_calculator<3>(iter);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf, kBFloat16, iter.common_dtype(), "glu_backward_xpu", [&] {
        auto gI = static_cast<scalar_t*>(iter.data_ptr(0));
        auto I = static_cast<const scalar_t*>(iter.data_ptr(1));
        auto gO = static_cast<const scalar_t*>(iter.data_ptr(2));
        launch_glu_backward_kernel(
            N,
            gI,
            I,
            gO,
            offset_calculator,
            gI_stride * sizeof(scalar_t),
            I_stride * sizeof(scalar_t));
      });
}

} // namespace at::native::xpu
