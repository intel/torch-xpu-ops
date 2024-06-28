#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/EmbeddingBackwardKernel.h>
#include <ATen/native/xpu/sycl/pstl/PSTLFunctions.h>

namespace at {
namespace native {
namespace xpu {

template <typename index_t>
struct EmbeddingDenseBackwardEqFunctor {
  auto operator()(index_t a, index_t b) const {
    return a == b;
  }
};

Tensor embedding_dense_backward_kernel(
    const Tensor& grad_,
    const Tensor& indices_,
    int64_t num_weights,
    int64_t padding_idx,
    bool scale_grad_by_freq) {
  auto grad_arg = TensorArg(grad_, "grad", 1);
  auto indices_arg = TensorArg(indices_, "indices", 1);
  checkScalarTypes("embedding_backward", indices_arg, {kLong, kInt});
  checkSameGPU("embedding_backward", grad_arg, indices_arg);

  auto indices = indices_.contiguous();

  auto num_indices = indices.numel();
  auto grad = grad_.contiguous().view({num_indices, grad_.size(-1)});

  auto sorted_indices =
      at::empty_like(indices, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto orig_indices = at::empty_like(indices, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  Tensor count;

  Tensor grad_weight;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      grad.scalar_type(),
      "embedding_backward",
      [&]() {
        AT_DISPATCH_INDEX_TYPES(
            indices.scalar_type(), "embedding_backward", [&] {
              // TODO: port pstl functions
              index_t* sorted_begin = sorted_indices.data_ptr<index_t>();
              index_t* orig_begin = orig_indices.data_ptr<index_t>();
              {
                sorted_indices.copy_(indices);
                pstl::itoa(orig_begin, orig_begin + num_indices, (index_t)0);
                pstl::sort<index_t, index_t>(
                    indices.data_ptr<index_t>(),
                    sorted_begin,
                    orig_begin,
                    num_indices,
                    false);
              }

              if (scale_grad_by_freq) {
                count = at::empty_like(sorted_indices);
                index_t* count_begin = count.data_ptr<index_t>();
                // Take the maximum of each count per unique key:
                // sorted: 2 5 5 5 7 7 8 9 9
                //  count: 1 3 3 3 2 2 1 2 2
                //
                EmbeddingDenseBackwardEqFunctor<index_t> f;
                pstl::count_by_segment<index_t, index_t, index_t>(
                    sorted_begin, sorted_begin + num_indices, count_begin, f);
              }
              grad_weight =
                  embedding_backward_deterministic_kernel<scalar_t, index_t>(
                      grad,
                      orig_indices,
                      sorted_indices,
                      count,
                      num_weights,
                      padding_idx);
            });
      });
  return grad_weight;
}

} // namespace xpu
} // namespace native
} // namespace at
