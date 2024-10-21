#pragma once

#include <ATen/TensorIterator.h>
#include <ATen/native/SparseTensorUtils.h>

using namespace at::sparse;
namespace at::native::xpu {

template <typename T>
struct TensorCAddOp {
  TensorCAddOp(T v) : val(v) {}

  void operator()(T* out, T* in) const {
    *out += val * *in;
  }

  void operator()(T* out, T* in1, T* in2) const {
    *out = *in1 + val * *in2;
  }

  T val;
};

TORCH_XPU_API void add_out_dense_sparse_kernel(
    Tensor& r_,
    const Tensor& dense,
    const SparseTensor& sparse,
    const Scalar& value);

TORCH_XPU_API void add_out_sparse_kernel(
    const SparseTensor& t,
    const SparseTensor& src,
    const Scalar& value,
    SparseTensor& r_);
} // namespace at::native::xpu
