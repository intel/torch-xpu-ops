#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/EmbeddingBackwardKernel.h>
#include <ATen/native/xpu/sycl/SYCLGroupAlgorithm.h>
#include <ATen/native/xpu/sycl/pstl/PSTLFunctions.h>
#include <comm/DeviceProperties.h>

#include <ATen/native/xpu/sycl/EmbeddingKernels.h>
namespace at ::native::xpu {

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
                    indices.const_data_ptr<index_t>(),
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

template <typename scalar_t, typename accscalar_t, typename index_t>
struct RenormKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    int tid = item.get_local_linear_id();
    int sgSize = item.get_local_range(0);
    auto group_idx = item.get_group(0);
    if (group_idx >= num_unique_indices_) {
      return;
    }

    int base_index = indices_[group_idx] * weights_stride0_;

    accscalar_t v = static_cast<accscalar_t>(0);
    for (int i = tid; i < dim_; i += sgSize) {
      auto x =
          static_cast<accscalar_t>(weights_[base_index + i * weights_stride1_]);
      if (norm_type_ == 1) {
        v += std::abs(x);
      } else if (norm_type_ == 2) {
        v += x * x;
      } else {
        v += std::pow(x, norm_type_);
      }
    }

    v = GroupReduceSumSGSizeEqualstoNumSG(
        item,
        v,
        static_cast<accscalar_t*>(
            smem_.template get_multi_ptr<sycl::access::decorated::no>().get()));

    if (tid == 0) {
      smem_[0] = std::pow(v, static_cast<accscalar_t>(1.0 / norm_type_));
    }
    item.barrier(sycl_local_fence);

    if (smem_[0] > max_norm_) {
      auto factor = static_cast<scalar_t>(
          max_norm_ / (smem_[0] + std::numeric_limits<accscalar_t>::epsilon()));
      for (int i = tid; i < dim_; i += sgSize) {
        weights_[base_index + i * weights_stride1_] *= factor;
      }
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    smem_ = sycl_local_acc_t<scalar_t>(smem_size_, cgh);
  }
  RenormKernelFunctor(
      scalar_t* weights,
      index_t* indices,
      accscalar_t max_norm,
      accscalar_t norm_type,
      int64_t dim,
      int64_t weights_stride0,
      int64_t weights_stride1,
      int64_t num_unique_indices,
      int64_t smem_size)
      : weights_(weights),
        indices_(indices),
        max_norm_(max_norm),
        norm_type_(norm_type),
        dim_(dim),
        weights_stride0_(weights_stride0),
        weights_stride1_(weights_stride1),
        num_unique_indices_(num_unique_indices),
        smem_size_(smem_size) {}

 private:
  scalar_t* weights_;
  index_t* indices_;
  accscalar_t max_norm_;
  accscalar_t norm_type_;
  int64_t dim_;
  int64_t weights_stride0_;
  int64_t weights_stride1_;
  int64_t num_unique_indices_;
  sycl_local_acc_t<accscalar_t> smem_;
  int64_t smem_size_;
};

template <typename scalar_t, typename accscalar_t, typename index_t>
void embedding_renorm_template(
    scalar_t* weights,
    index_t* indices,
    accscalar_t max_norm,
    accscalar_t norm_type,
    int64_t dim,
    int64_t weights_stride0,
    int64_t weights_stride1,
    int64_t num_unique_indices) {
  const int64_t work_group_size = syclMaxWorkItemsPerEU();
  auto kfn = RenormKernelFunctor<scalar_t, accscalar_t, index_t>(
      weights,
      indices,
      max_norm,
      norm_type,
      dim,
      weights_stride0,
      weights_stride1,
      num_unique_indices,
      work_group_size / 8);
  auto& queue = at::xpu::getCurrentSYCLQueue();
  sycl_kernel_submit(
      work_group_size * num_unique_indices, work_group_size, queue, kfn);
}
struct EmbeddingRenormCmpFunctor {
  template <typename T>
  bool operator()(T lhs, T rhs) const {
    if (lhs != rhs) {
      return false;
    }
    return true;
  }
};

Tensor& embedding_renorm_kernel(
    Tensor& self,
    const Tensor& indices,
    double max_norm,
    double norm_type) {
  auto self_arg = TensorArg(self, "self", 1);
  auto indices_arg = TensorArg(indices, "indices", 1);
  checkDim("embedding_renorm_", self_arg, 2);
  checkSameGPU("embedding_renorm", self_arg, indices_arg);
  AT_DISPATCH_INDEX_TYPES(
      indices.scalar_type(), "embedding_renorm_xpu_", [&]() {
        auto num_indices = indices.numel();
        auto indices_contig = std::get<0>(indices.sort()).contiguous();
        auto unique_indices = indices_contig;
        int64_t num_unique_indices;
        EmbeddingRenormCmpFunctor f;
        num_unique_indices =
            pstl::unique<index_t, index_t>(
                unique_indices.data_ptr<index_t>(),
                unique_indices.data_ptr<index_t>() + num_indices,
                f) -
            unique_indices.data_ptr<index_t>();

        int dim = self.stride(0);

        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            self.scalar_type(),
            "embedding_renorm_xpu_",
            [&] {
              using accscalar_t = acc_type_device<scalar_t, kXPU>;
              embedding_renorm_template(
                  self.data_ptr<scalar_t>(),
                  unique_indices.data_ptr<index_t>(),
                  static_cast<accscalar_t>(max_norm),
                  static_cast<accscalar_t>(norm_type),
                  dim,
                  self.stride(0),
                  self.stride(1),
                  num_unique_indices);
            });
      });
  return self;
}

} // namespace at::native::xpu
