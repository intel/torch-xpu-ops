#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/op_registration/adaption.h>
<<<<<<< HEAD
#include <ATen/native/DispatchStub.h>
#include <ATen/native/LinearAlgebra.h>
=======
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/ReduceOpsUtils.h>
>>>>>>> main
#include <ATen/native/utils/ParamUtils.h>

#include <ATen/native/xpu/sycl/LinearAlgebraKernels.h>
#include <ATen/native/xpu/sycl/ReduceNormKernel.h>
#include <comm/RegisterUtils.h>

namespace at {
<<<<<<< HEAD
namespace native {
REGISTER_XPU_DISPATCH(addr_stub, xpu::addr_kernel);
=======
namespace detail {

static void check_linalg_norm_dtype(
    optional<ScalarType> opt_dtype,
    ScalarType self_dtype,
    const char* const name) {
  if (opt_dtype.has_value()) {
    auto dtype = opt_dtype.value();
    TORCH_CHECK(
        isFloatingType(dtype) || isComplexType(dtype),
        name,
        ": dtype should"
        " be floating point or complex, but got ",
        dtype);
    TORCH_CHECK(
        isComplexType(self_dtype) == isComplexType(dtype),
        name,
        ": dtype should be ",
        isComplexType(self_dtype) ? "complex" : "real",
        " for ",
        isComplexType(self_dtype) ? "complex" : "real",
        " inputs, but got ",
        dtype);
    TORCH_CHECK(
        promoteTypes(self_dtype, dtype) == dtype,
        name,
        ": the dtype of the input ",
        "(",
        self_dtype,
        ") should be convertible ",
        "without narrowing to the specified dtype (",
        dtype,
        ")");
  }
}

} // namespace detail

Tensor& linalg_vector_norm_meta(
    const Tensor& self,
    const Scalar& scalar_ord,
    OptionalIntArrayRef opt_dim,
    bool keepdim,
    optional<ScalarType> opt_dtype,
    Tensor& output) {
  at::native::checkFloatingOrComplex(self, "linalg.vector_norm");

  auto dim = opt_dim.value_or(IntArrayRef{});
  // Casting a large integer to a double will just introduce an error for
  // values larger than 10^53 (same for negative numbers), so that's fine.
  auto ord = scalar_ord.toDouble();

  // For more context, see issue 52783
  // If the tensor is empty and norm < 0 || norm == infty
  //   - We cannot reduce the whole tensor
  //   - We cannot reduce over an empty dimension
  if (self.numel() == 0 && (ord < 0. || ord == INFINITY)) {
    // dim=None or dim=() reduces the whole tensor
    TORCH_CHECK(
        opt_dim.has_value() && !opt_dim->empty(),
        "linalg.vector_norm cannot compute the ",
        scalar_ord,
        " norm on an empty ",
        "tensor because the operation does not have an identity");
    for (auto dim_num : dim) {
      TORCH_CHECK(
          self.size(dim_num) != 0,
          "linalg.vector_norm cannot compute the ",
          scalar_ord,
          " norm on the dimension ",
          dim_num,
          "because this dimension is empty and the operation does not have an identity");
    }
  }

  at::detail::check_linalg_norm_dtype(
      opt_dtype, self.scalar_type(), "linalg.vector_norm");

  auto mask = at::native::make_dim_mask(dim, self.dim());
  auto shape = at::native::shape_from_dim_mask(self, std::move(mask), keepdim);
  auto options = self.options().dtype(
      toRealValueType(opt_dtype.value_or(self.scalar_type())));
  if (output.defined()) {
    at::xpu::resize_out(output, shape, {}, options);
  } else {
    output = at::xpu::create_out(shape, {}, options);
  }
  return output;
}
>>>>>>> main

}

Tensor XPUNativeFunctions::linalg_vector_norm(
    const Tensor& self,
    const Scalar& scalar_ord,
    OptionalIntArrayRef opt_dim,
    bool keepdim,
    optional<ScalarType> opt_dtype) {
  Tensor result;
  linalg_vector_norm_out(self, scalar_ord, opt_dim, keepdim, opt_dtype, result);
  return result;
}

Tensor& XPUNativeFunctions::linalg_vector_norm_out(
    const Tensor& self,
    const Scalar& scalar_ord,
    OptionalIntArrayRef opt_dim,
    bool keepdim,
    optional<ScalarType> opt_dtype,
    Tensor& result) {
  result = linalg_vector_norm_meta(
      self, scalar_ord, opt_dim, keepdim, opt_dtype, result);
  auto ord = scalar_ord.toDouble();
  auto dim = opt_dim.value_or(IntArrayRef{});
  auto size = self.sizes();
  auto ndim = self.dim();

  auto opt_dim_ = dim.vec();
  maybe_wrap_dims(opt_dim_, ndim);

  using Int = IntArrayRef::value_type;
  std::vector<Int> all_dim(ndim);
  std::iota(all_dim.begin(), all_dim.end(), 0);

  bool is_all_reduce = !opt_dim.has_value() || opt_dim.value().empty();
  auto reduce_dim = is_all_reduce ? all_dim : opt_dim_;

  bool is_reduce_over_1D_vector = true;
  for (auto i : reduce_dim) {
    if (size[i] != 1) {
      is_reduce_over_1D_vector = false;
      break;
    }
  }

  if (is_reduce_over_1D_vector) {
    Tensor self_;
    if (opt_dtype.has_value()) {
      self_ = self.to(*opt_dtype);
    } else {
      self_ = self;
    }
    if (ord != 0.0) {
      keepdim ? at::abs_outf(self_, const_cast<Tensor&>(result))
              : at::abs_outf(
                    self_.squeeze(reduce_dim), const_cast<Tensor&>(result));
    } else {
      keepdim ? at::ne_outf(self_, 0, const_cast<Tensor&>(result))
              : at::ne_outf(
                    self_.squeeze(reduce_dim), 0, const_cast<Tensor&>(result));
    }
    return result;
  }

  auto iter = at::native::make_reduction(
      "vector_norm",
      const_cast<Tensor&>(result),
      self,
      dim,
      keepdim,
      result.scalar_type());
  native::xpu::norm_kernel(iter, ord);
  return result;
}

} // namespace at
