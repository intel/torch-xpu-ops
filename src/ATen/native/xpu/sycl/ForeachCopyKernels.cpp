#include <ATen/Dispatch.h>
#include <comm/xpu_aten.h>

#include <ATen/native/xpu/sycl/ForeachFunctors.h>
#include <ATen/native/xpu/sycl/MultiTensorApply.h>

#include <ATen/native/xpu/sycl/ForeachCopyKernels.h>

#define AT_DISPATCH_SOURCE_TYPES(TYPE, NAME, ...)                                                \
  AT_DISPATCH_SWITCH(                                                                            \
      TYPE,                                                                                      \
      NAME,                                                                                      \
      AT_PRIVATE_CASE_TYPE_USING_HINT(                                                           \
          at::ScalarType::Byte,                                                                  \
          src_t,                                                                                 \
          __VA_ARGS__) AT_PRIVATE_CASE_TYPE_USING_HINT(at::ScalarType::Char, src_t, __VA_ARGS__) \
          AT_PRIVATE_CASE_TYPE_USING_HINT(                                                       \
              at::ScalarType::Long, src_t, __VA_ARGS__)                                          \
              AT_PRIVATE_CASE_TYPE_USING_HINT(                                                   \
                  at::ScalarType::Short, src_t, __VA_ARGS__)                                     \
                  AT_PRIVATE_CASE_TYPE_USING_HINT(                                               \
                      at::ScalarType::Int, src_t, __VA_ARGS__)                                   \
                      AT_PRIVATE_CASE_TYPE_USING_HINT(                                           \
                          at::ScalarType::Double, src_t, __VA_ARGS__)                            \
                          AT_PRIVATE_CASE_TYPE_USING_HINT(                                       \
                              at::ScalarType::Float, src_t, __VA_ARGS__)                         \
                              AT_PRIVATE_CASE_TYPE_USING_HINT(                                   \
                                  at::ScalarType::ComplexDouble,                                 \
                                  src_t,                                                         \
                                  __VA_ARGS__)                                                   \
                                  AT_PRIVATE_CASE_TYPE_USING_HINT(                               \
                                      at::ScalarType::ComplexFloat,                              \
                                      src_t,                                                     \
                                      __VA_ARGS__)                                               \
                                      AT_PRIVATE_CASE_TYPE_USING_HINT(                           \
                                          at::ScalarType::Half,                                  \
                                          src_t,                                                 \
                                          __VA_ARGS__)                                           \
                                          AT_PRIVATE_CASE_TYPE_USING_HINT(                       \
                                              at::ScalarType::BFloat16,                          \
                                              src_t,                                             \
                                              __VA_ARGS__)                                       \
                                              AT_PRIVATE_CASE_TYPE_USING_HINT(                   \
                                                  at::ScalarType::Bool,                          \
                                                  src_t,                                         \
                                                  __VA_ARGS__)                                   \
                                                  AT_PRIVATE_CASE_TYPE_USING_HINT(               \
                                                      at::ScalarType::                           \
                                                          Float8_e4m3fn,                         \
                                                      src_t,                                     \
                                                      __VA_ARGS__)                               \
                                                      AT_PRIVATE_CASE_TYPE_USING_HINT(           \
                                                          at::ScalarType::                       \
                                                              Float8_e4m3fnuz,                   \
                                                          src_t,                                 \
                                                          __VA_ARGS__)                           \
                                                          AT_PRIVATE_CASE_TYPE_USING_HINT(       \
                                                              at::ScalarType::                   \
                                                                  Float8_e5m2,                   \
                                                              src_t,                             \
                                                              __VA_ARGS__)                       \
                                                              AT_PRIVATE_CASE_TYPE_USING_HINT(   \
                                                                  at::ScalarType::               \
                                                                      Float8_e5m2fnuz,           \
                                                                  src_t,                         \
                                                                  __VA_ARGS__))

namespace at::native::xpu {
template <typename T>
struct Identity {
  T operator()(const T& x) {
    return x;
  }
};

void foreach_copy_list_kernel_(TensorList self, TensorList src) {
  std::vector<std::vector<at::Tensor>> tensor_lists{src.vec(), self.vec()};

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND7(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Bool,
      ScalarType::Float8_e4m3fn,
      ScalarType::Float8_e4m3fnuz,
      ScalarType::Float8_e5m2,
      ScalarType::Float8_e5m2fnuz,
      self[0].scalar_type(),
      "foreach_tensor_copy",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        AT_DISPATCH_SOURCE_TYPES(
            src[0].scalar_type(), "foreach_tensor_copy", [&] {
              multi_tensor_apply<2>(
                  tensor_lists,
                  UnaryOpFunctor<
                      scalar_t,
                      /* depth */ 2,
                      /* r_args_depth */ 1,
                      /* res_arg_index */ 1>(),
                  Identity<opmath_t>());
            });
      });

  // AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
  //     at::ScalarType::Half,
  //     at::ScalarType::BFloat16,
  //     at::ScalarType::Bool,
  //     self[0].scalar_type(),
  //     "foreach_tensor_copy",
  //     [&]() {
  //       using opmath_t = at::opmath_type<scalar_t>;
  //       multi_tensor_apply<2>(
  //           tensor_lists,
  //           UnaryOpFunctor<
  //               scalar_t,
  //               /* depth */ 2,
  //               /* r_args_depth */ 1,
  //               /* res_arg_index */ 1>(),
  //           Identity<opmath_t>());
  //     });
}

#undef AT_DISPATCH_SOURCE_TYPES
} // namespace at::native::xpu
