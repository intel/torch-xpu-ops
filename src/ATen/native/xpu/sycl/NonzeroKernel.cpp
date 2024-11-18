#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>

#include <ATen/native/xpu/sycl/pstl/PSTLFunctions.h>
#include <comm/SYCLHelpers.h>
#include <comm/TensorInfo.h>

#include <ATen/native/xpu/sycl/NonzeroKernel.h>

namespace at::native::xpu {

struct FlattenIdxtoRealIdxKernelFunctor {
  void operator()(sycl::nd_item<1> item_id) const {
    auto global_id = item_id.get_global_linear_id();

    if (global_id < N_) {
      auto index = global_id / num_dim_;
      auto dim = global_id % num_dim_;
      tensor_begin_[global_id] =
          idx_flat_begin_[index] / divisor_[dim] % sizes_[dim];
    }
  }
  FlattenIdxtoRealIdxKernelFunctor(
      int64_t N,
      const int64_t num_dim,
      int64_t* tensor_begin,
      int64_t* idx_flat_begin,
      int64_t* divisor,
      int64_t* sizes)
      : N_(N),
        num_dim_(num_dim),
        tensor_begin_(tensor_begin),
        idx_flat_begin_(idx_flat_begin) {
    for (auto dim = num_dim - 1; dim >= 0; dim--) {
      sizes_[dim] = sizes[dim];
      divisor_[dim] = divisor[dim];
    }
  }

 private:
  int64_t N_;
  const int64_t num_dim_;
  int64_t* tensor_begin_;
  int64_t* idx_flat_begin_;
  int64_t divisor_[XPU_MAX_TENSORINFO_DIMS];
  int64_t sizes_[XPU_MAX_TENSORINFO_DIMS];
};

template <typename scalar_t>
struct CopyIfFunc {
  bool operator()(int64_t x) const {
    return self_begin_[x] != scalar_t(0);
  }
  CopyIfFunc(const scalar_t* self_begin) : self_begin_(self_begin) {}

 private:
  const scalar_t* self_begin_;
};

template <>
struct CopyIfFunc<bool> {
  bool operator()(int64_t x) const {
    // Using data type conversion to break deduce of execution chain in bool.
    // Bool operations will be removed in the compiler optimization.
    // The function returns a bool variable with one byte value stored in
    // self_begin_ not 1 specified here.
    volatile int in = (int)self_begin_[x];
    bool res = in != int(0) ? 1 : 0;
    return res;
  }
  CopyIfFunc(const bool* self_begin) : self_begin_(self_begin) {}

 private:
  const bool* self_begin_;
};

template <typename scalar_t>
void nonzero_template(const Tensor& self_, Tensor& tensor) {
  Tensor self = self_.contiguous();

  const int64_t num_dim = self.dim();
  TORCH_CHECK(num_dim <= XPU_MAX_TENSORINFO_DIMS, "dim exceed max allowed dim");

  int64_t N = self.numel();

  if (N > 0) {
    Tensor idx_flat = at::empty(
        {N}, tensor.options().memory_format(LEGACY_CONTIGUOUS_MEMORY_FORMAT));
    Tensor range = at::empty(
        {N}, tensor.options().memory_format(LEGACY_CONTIGUOUS_MEMORY_FORMAT));

    const scalar_t* self_begin = self.const_data_ptr<scalar_t>();
    int64_t* idx_flat_begin = idx_flat.data_ptr<int64_t>();
    int64_t* range_begin = nullptr;

    CopyIfFunc<scalar_t> f(self_begin);
    auto idx_flat_end =
        pstl::copy_if<int64_t>(range_begin, range_begin + N, idx_flat_begin, f);

    auto num_nonzeros = std::distance(idx_flat_begin, idx_flat_end);

    Tensor tensor_ = tensor.resize_({num_nonzeros, num_dim}).contiguous();
    if (num_nonzeros > 0 && num_dim > 0) {
      int64_t* tensor_begin = tensor_.data_ptr<int64_t>();

      // preload sizes tensor for index calculation
      int64_t sizes[XPU_MAX_TENSORINFO_DIMS];
      int64_t divisor[XPU_MAX_TENSORINFO_DIMS];
      sizes[num_dim - 1] = self.size(num_dim - 1);
      divisor[num_dim - 1] = 1;
      for (auto dim = num_dim - 2; dim >= 0; dim--) {
        sizes[dim] = self.size(dim);
        divisor[dim] = sizes[dim + 1] * divisor[dim + 1];
      }

      const int64_t N = num_nonzeros * num_dim;
      // restore flatten idx to indices
      FlattenIdxtoRealIdxKernelFunctor kfn(
          N, num_dim, tensor_begin, idx_flat_begin, divisor, sizes);

      const auto wg_sz = std::min(syclMaxWorkGroupSize(kfn), N);
      const auto num_wg = (N + wg_sz - 1) / wg_sz;

      sycl_kernel_submit(wg_sz * num_wg, wg_sz, getCurrentSYCLQueue(), kfn);

      // Support non-contiguous/outplace cases
      // TODO: Next step, we will give state of art algo/implementation.
      // Non-contiguous/outplace cases performance will be covered there.
      if (tensor.data_ptr() != tensor_.data_ptr()) {
        tensor.copy_(tensor_);
      }
    }
  } else {
    tensor = tensor.resize_({N, num_dim}).contiguous();
  }
}

void nonzero_kernel(const Tensor& self, Tensor& out) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
      at::ScalarType::ComplexHalf,
      at::ScalarType::Bool,
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      self.scalar_type(),
      "nonzero_xpu",
      [&] { nonzero_template<scalar_t>(self, out); });
}
} // namespace at::native::xpu
