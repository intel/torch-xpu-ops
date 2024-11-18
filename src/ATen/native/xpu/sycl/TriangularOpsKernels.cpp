#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/ceil_div.h>
#include <ATen/native/CanUse32BitIndexMath.h>
#include <ATen/native/Resize.h>
#include <comm/SYCLContext.h>

#include <ATen/native/xpu/sycl/TriangularOpsKernels.h>

namespace at::native::xpu {

using namespace at::xpu;

template <typename scalar_t, typename IndexType, bool upper>
struct ApplyTriuTrilKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    for (size_t linearIndex = item.get_global_id(0); linearIndex < (size_t)N;
         linearIndex += item.get_global_range()[0]) {
      IndexType batch_id = linearIndex / (self_size_0 * self_size_1);
      IndexType row = (linearIndex % (self_size_0 * self_size_1)) / self_size_1;
      IndexType col = (linearIndex % (self_size_0 * self_size_1)) % self_size_1;

      IndexType src_index =
          batch_id * self_stride + row * self_stride_0 + col * self_stride_1;
      IndexType tgt_index = batch_id * result_stride + row * result_stride_0 +
          col * result_stride_1;

      bool mask = upper ? (col - row >= k) : (col - row <= k);
      result_ptr[tgt_index] = mask ? self_ptr[src_index] : scalar_t(0);
    }
  }
  ApplyTriuTrilKernelFunctor(
      const int64_t k_,
      int64_t N_,
      IndexType self_size_0_,
      IndexType self_size_1_,
      IndexType self_stride_,
      IndexType self_stride_0_,
      IndexType self_stride_1_,
      IndexType result_stride_,
      IndexType result_stride_0_,
      IndexType result_stride_1_,
      scalar_t* result_ptr_,
      const scalar_t* self_ptr_)
      : k(k_),
        N(N_),
        self_size_0(self_size_0_),
        self_size_1(self_size_1_),
        self_stride(self_stride_),
        self_stride_0(self_stride_0_),
        self_stride_1(self_stride_1_),
        result_stride(result_stride_),
        result_stride_0(result_stride_0_),
        result_stride_1(result_stride_1_),
        result_ptr(result_ptr_),
        self_ptr(self_ptr_) {}

 private:
  const int64_t k;
  int64_t N;
  IndexType self_size_0;
  IndexType self_size_1;
  IndexType self_stride;
  IndexType self_stride_0;
  IndexType self_stride_1;
  IndexType result_stride;
  IndexType result_stride_0;
  IndexType result_stride_1;
  scalar_t* result_ptr;
  const scalar_t* self_ptr;
};

template <typename scalar_t, typename IndexType, bool upper>
void apply_triu_tril(
    const Tensor& result,
    const Tensor& self,
    const int64_t k) {
  auto N = self.numel();
  IndexType self_size_0 = (IndexType)self.size(-2);
  IndexType self_size_1 = (IndexType)self.size(-1);
  IndexType self_stride = (IndexType)(self.dim() > 2 ? self.stride(-3) : 1);
  IndexType self_stride_0 = (IndexType)self.stride(-2);
  IndexType self_stride_1 = (IndexType)self.stride(-1);
  IndexType result_stride =
      (IndexType)(result.dim() > 2 ? result.stride(-3) : 1);
  IndexType result_stride_0 = (IndexType)result.stride(-2);
  IndexType result_stride_1 = (IndexType)result.stride(-1);

  scalar_t* result_ptr = result.data_ptr<scalar_t>();
  const scalar_t* self_ptr = self.const_data_ptr<scalar_t>();

  ApplyTriuTrilKernelFunctor<scalar_t, IndexType, upper> kfn(
      k,
      N,
      self_size_0,
      self_size_1,
      self_stride,
      self_stride_0,
      self_stride_1,
      result_stride,
      result_stride_0,
      result_stride_1,
      result_ptr,
      self_ptr);

  int64_t group_size = syclMaxWorkGroupSize(kfn);
  auto num_groups = ceil_div(N, group_size);
  auto total_items = num_groups * group_size;
  auto& queue = getCurrentSYCLQueue();

  sycl_kernel_submit(
      sycl::range<1>(total_items), sycl::range<1>(group_size), queue, kfn);
}

#define TRIU_TRIL_LAMBDA(upper)                                   \
  [&] {                                                           \
    if (canUse32BitIndexMath(self)) {                             \
      apply_triu_tril<scalar_t, int32_t, upper>(result, self, k); \
    } else {                                                      \
      apply_triu_tril<scalar_t, int64_t, upper>(result, self, k); \
    }                                                             \
  }

void tril_kernel(const Tensor& result, const Tensor& self, int64_t k) {
  if (result.sizes() != self.sizes()) {
    result.resize_as_(self);
  }
  if (self.numel() == 0) {
    // return result;
    return;
  }

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
      at::ScalarType::Half,
      at::ScalarType::Bool,
      at::ScalarType::BFloat16,
      at::ScalarType::ComplexHalf,
      self.scalar_type(),
      "tril_xpu",
      TRIU_TRIL_LAMBDA(false));

  // return result;
}

void triu_kernel(const Tensor& result, const Tensor& self, int64_t k) {
  if (result.sizes() != self.sizes()) {
    result.resize_as_(self);
  }
  if (self.numel() == 0) {
    // return result;
    return;
  }
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
      at::ScalarType::Half,
      at::ScalarType::Bool,
      at::ScalarType::BFloat16,
      at::ScalarType::ComplexHalf,
      self.scalar_type(),
      "triu_xpu",
      TRIU_TRIL_LAMBDA(true));

  // return result;
}

} // namespace at::native::xpu
