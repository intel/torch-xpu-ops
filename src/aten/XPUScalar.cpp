#include <ATen/Dispatch_v2.h>
#include <ATen/ScalarOps.h>
#include <ATen/XPUNativeFunctions.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/TensorIterator.h>
#include <comm/SYCLContext.h>
#include <torch/library.h>

namespace at {

Scalar XPUNativeFunctions::_local_scalar_dense(const Tensor& self) {
  Scalar r;
  AT_DISPATCH_V2(
      self.scalar_type(),
      "_local_scalar_dense_xpu",
      AT_WRAP([&] {
        auto value = at::detail::empty_cpu(
            {1}, /* size */
            c10::CppTypeToScalarType<scalar_t>(), /* dtype */
            c10::nullopt, /* layout */
            c10::nullopt, /* device */
            true, /* pin_memory */
            c10::nullopt /* memory format */
        );
        auto q = at::xpu::getCurrentSYCLQueue();

        q.memcpy(
            (void*)value.const_data_ptr<scalar_t>(),
            self.const_data_ptr<scalar_t>(),
            sizeof(scalar_t));
        r = Scalar(*value.const_data_ptr<scalar_t>());
      }),
      AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX),
      kComplexHalf,
      kHalf,
      kBool,
      kBFloat16,
      AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
  return r;
}

} // namespace at
