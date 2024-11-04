#include <ATen/Dispatch.h>
#include <comm/xpu_aten.h>

#include <ATen/native/xpu/sycl/ForeachFunctors.h>
#include <ATen/native/xpu/sycl/MultiTensorApply.h>

#include <ATen/native/xpu/sycl/ForeachCopyKernels.h>

namespace at::native::xpu {
template <typename T>
struct Identity {
  T operator()(const T& x) {
    return x;
  }
};

void foreach_copy_list_kernel_(TensorList self, TensorList src) {
  std::vector<std::vector<at::Tensor>> tensor_lists{src.vec(), self.vec()};

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Bool,
      self[0].scalar_type(),
      "foreach_tensor_copy",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        multi_tensor_apply<2>(
            tensor_lists,
            UnaryOpFunctor<
                scalar_t,
                /* depth */ 2,
                /* r_args_depth */ 1,
                /* res_arg_index */ 1>(),
            Identity<opmath_t>());
      });
}

} // namespace at::native::xpu
