#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/xpu/EmptyTensor.h>
#include <comm/RegisterUtils.h>

namespace at {
namespace xpu {

inline Tensor& resize_reduction(
    Tensor& out,
    const Tensor& self,
    OptionalIntArrayRef opt_dims,
    bool keepdim,
    ScalarType out_dtype,
    bool allow_empty_dims = false) {
  DimVector dims_ = at::native::make_dim_vector(opt_dims, self.dim());
  maybe_wrap_dims(dims_, self.dim());
  auto shape =
      at::meta::get_reduction_shape(self, dims_, keepdim, allow_empty_dims);
  if (self.layout() == kStrided) {
    if (out.defined()) {
      resize_out(out, shape, {}, self.options().dtype(out_dtype));
    } else {
      out = create_out(shape, {}, self.options().dtype(out_dtype));
    }
  } else if (shape.size() == 0) {
    if (out.defined()) {
      resize_out(
          out, shape, {}, self.options().dtype(out_dtype).layout(kStrided));
    } else {
      out = create_out(
          shape, {}, self.options().dtype(out_dtype).layout(kStrided));
    }
  } else {
    TORCH_CHECK(
        false,
        "resize_reduction: support for output with ",
        self.layout(),
        " layout is not implemented yet");
  }
  namedinference::propagate_names_for_reduction(out, self, dims_, keepdim);
  return out;
}

inline std::tuple<Tensor&, Tensor&> resize_reduction_with_indices(
    Tensor& out,
    Tensor& out_indice,
    const Tensor& self,
    IntArrayRef dims,
    bool keepdim,
    ScalarType out_dtype) {
  DimVector dims_(dims);
  maybe_wrap_dims(dims_, self.dim());
  auto shape = at::meta::get_reduction_shape(self, dims_, keepdim);
  if (out.defined()) {
    resize_out(out, shape, {}, self.options().dtype(out_dtype));
  } else {
    out = create_out(shape, {}, self.options().dtype(out_dtype));
  }
  if (out_indice.defined()) {
    resize_out(out_indice, shape, {}, self.options().dtype(kLong));
  } else {
    out_indice = create_out(shape, {}, self.options().dtype(kLong));
  }
  namedinference::propagate_names_for_reduction(out, self, dims_, keepdim);
  namedinference::propagate_names_for_reduction(
      out_indice, self, dims_, keepdim);
  return std::forward_as_tuple(out, out_indice);
}

inline void allocate_reduction_result(
    Tensor& result,
    const Tensor& self,
    native::DimMask mask,
    bool keepdim,
    ScalarType dtype) {
  auto shape = DimVector(self.sizes());
  for (int dim = shape.size() - 1; dim >= 0; dim--) {
    if (mask[dim]) {
      if (keepdim) {
        shape[dim] = 1;
      } else {
        shape.erase(shape.begin() + dim);
      }
    }
  }
  if (result.defined()) {
    result.resize_(shape);
  } else {
    result = at::empty(shape, self.options().dtype(dtype));
  }
}

} // namespace xpu
} // namespace at
