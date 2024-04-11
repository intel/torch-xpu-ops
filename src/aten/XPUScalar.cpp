#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch_v2.h>
#include <ATen/EmptyTensor.h>
#include <ATen/core/Tensor.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_local_scalar_dense_native.h>
#endif

#include <ATen/XPUNativeFunctions.h>
#include <comm/SYCLContext.h>

namespace at {

Scalar XPUNativeFunctions::_local_scalar_dense(const Tensor& self) {
  Scalar r;
  AT_DISPATCH_V2(
      self.scalar_type(),
      "_local_scalar_dense",
      AT_WRAP([&] {
        // Create pinned memory for the scalar value to avoid implicit
        // locking/sync in xpu library due to pageable memory
        auto value = at::detail::empty_cpu(
            {1}, /* size */
            c10::CppTypeToScalarType<scalar_t>(), /* dtype */
            c10::nullopt, /* layout */
            c10::nullopt, /* device */
            true, /* pin_memory */
            c10::nullopt /* memory format */
        );

        auto queue = at::xpu::getCurrentSYCLQueue();
        queue.memcpy(
            (void*)value.const_data_ptr<scalar_t>(),
            self.const_data_ptr<scalar_t>(),
            sizeof(scalar_t));
        queue.wait();

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
