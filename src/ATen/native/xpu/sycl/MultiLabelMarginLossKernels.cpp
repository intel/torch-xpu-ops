#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/Resize.h>
#include <ATen/native/xpu/sycl/GroupReduceUtils.h>
#include <comm/SYCLContext.h>

#include <ATen/native/xpu/sycl/MultiLabelMarginLossKernels.h>

namespace at::native::xpu {

const int MULTILABELMARGIN_SUB_GROUP_SIZE = 32;
const int MULTILABELMARGIN_THREADS =
    MULTILABELMARGIN_SUB_GROUP_SIZE * MULTILABELMARGIN_SUB_GROUP_SIZE;

using namespace at::xpu;

void multilabel_margin_loss_shape_check(
    int64_t& nframe,
    int64_t& dim,
    const int64_t& ndims,
    const Tensor& input,
    const Tensor& target) {
  TORCH_CHECK(
      (ndims == 2 && input.size(1) != 0) ||
          (ndims == 1 && input.size(0) != 0) || ndims == 0,
      "Expected non-empty vector or matrix with optional 0-dim batch size, but got: ",
      input.sizes());

  if (ndims <= 1) {
    nframe = 1;
    dim = ndims == 0 ? 1 : input.size(0);
    TORCH_CHECK(
        target.dim() <= 1 && target.numel() == dim,
        "inconsistent target size: ",
        target.sizes(),
        " for input of size: ",
        input.sizes());
  } else {
    nframe = input.size(0);
    dim = input.size(1);
    TORCH_CHECK(
        target.dim() == 2 && target.size(0) == nframe && target.size(1) == dim,
        "inconsistent target size: ",
        target.sizes(),
        " for input of size: ",
        input.sizes());
  }
}

template <typename scalar_t, typename accscalar_t>
struct MultilabelMarginLossForwardKernelFunctor
    : public __SYCL_KER_CONFIG_CONVENTION__ {
  [[intel::reqd_sub_group_size(MULTILABELMARGIN_SUB_GROUP_SIZE)]] void
  operator()(sycl::nd_item<1> item) const {
    int k = item.get_group(0);
    const scalar_t* input_k = input_ + k * dim_;
    const int64_t* target_k = target_ + k * dim_;
    scalar_t* output_k = output_ + k;
    scalar_t* is_target_k = is_target_ + k * dim_;
    for (int d = item.get_local_linear_id(); d < dim_;
         d += item.get_local_range(0)) {
      is_target_k[d] = static_cast<scalar_t>(0);
    }
    item.barrier(sycl_local_fence);

    if (item.get_local_linear_id() == 0) {
      for (int dt = 0; dt < dim_; dt++) {
        int target_idx = target_k[dt];
        if (target_idx < 0) {
          break;
        }
        is_target_k[target_idx] = static_cast<scalar_t>(1);
      }
    }
    item.barrier(sycl_local_fence);

    accscalar_t sum = 0;
    for (int dt = 0; dt < dim_; dt++) {
      // next target:
      int target_idx = target_k[dt];
      if (target_idx < 0) {
        break;
      }

      // current value for target
      scalar_t input_target_k = input_k[target_idx];

      // compare to all inputs (multithreaded):
      for (int d = item.get_local_linear_id(); d < dim_;
           d += item.get_local_range(0)) {
        // contribute to loss only if not a target
        if (!static_cast<int>(is_target_k[d])) {
          scalar_t z = 1 - input_target_k + input_k[d];
          if (z > 0) {
            sum += z;
          }
        }
      }
    }

    accscalar_t total_sum = GroupReduceSumWithoutBroadcast<
        accscalar_t,
        MULTILABELMARGIN_SUB_GROUP_SIZE>(item, sum, smem_);

    if (item.get_local_linear_id() == 0) {
      if (size_average_) {
        *output_k = static_cast<scalar_t>((total_sum / dim_) / nframe_);
      } else {
        *output_k = static_cast<scalar_t>(total_sum / dim_);
      }
    }
  }
  void sycl_ker_config_convention(sycl::handler& cgh) {
    smem_ = sycl_local_acc_t<accscalar_t>(smem_size_, cgh);
  }
  MultilabelMarginLossForwardKernelFunctor(
      scalar_t* output,
      const scalar_t* input,
      const int64_t* target,
      scalar_t* is_target,
      int nframe,
      int dim,
      bool size_average,
      int64_t smem_size)
      : output_(output),
        input_(input),
        target_(target),
        is_target_(is_target),
        nframe_(nframe),
        dim_(dim),
        size_average_(size_average),
        smem_size_(smem_size) {}

 private:
  scalar_t* output_;
  const scalar_t* input_;
  const int64_t* target_;
  const scalar_t* weights_;
  scalar_t* is_target_;
  int nframe_;
  int dim_;
  bool size_average_;
  int64_t smem_size_;
  sycl_local_acc_t<accscalar_t> smem_;
};

template <typename scalar_t, typename accscalar_t>
struct MultilabelMarginLossBackwardKernelFunctor
    : public __SYCL_KER_CONFIG_CONVENTION__ {
  [[intel::reqd_sub_group_size(MULTILABELMARGIN_SUB_GROUP_SIZE)]] void
  operator()(sycl::nd_item<1> item) const {
    int k = item.get_group(0);
    const scalar_t* input_k = input_ + k * dim_;
    scalar_t* grad_input_k = grad_input_ + k * dim_;
    const int64_t* target_k = target_ + k * dim_;
    const scalar_t* is_target_k = is_target_ + k * dim_;

    const scalar_t* grad_output_k = grad_output_;
    if (!reduce_) {
      grad_output_k += k;
    }

    // gain:
    scalar_t g = static_cast<scalar_t>(
        size_average_ && reduce_
            ? accscalar_t(1) / static_cast<accscalar_t>(nframe_ * dim_)
            : accscalar_t(1) / static_cast<accscalar_t>(dim_));

    // zero gradients:
    for (int d = item.get_local_id(0); d < dim_; d += item.get_local_range(0)) {
      grad_input_k[d] = static_cast<scalar_t>(0);
    }
    item.barrier(sycl_local_fence);

    // iterate over targets
    for (int dt = 0; dt < dim_; dt++) {
      // next target:
      int target_idx = static_cast<int>(target_k[dt]);
      if (target_idx < 0) {
        break;
      }

      // current value for target
      scalar_t input_target_k = input_k[target_idx];

      // compare to all inputs (multithreaded):
      accscalar_t sum = 0;
      for (int d = item.get_local_id(0); d < dim_;
           d += item.get_local_range(0)) {
        // contribute to loss only if not a target
        if (!static_cast<int>(is_target_k[d])) {
          scalar_t z = 1 - input_target_k + input_k[d];
          if (z > 0) {
            sum -= g;
            grad_input_k[d] += g;
          }
        }
      }
      item.barrier(sycl_local_fence);

      sum = GroupReduceSumWithoutBroadcast<
          accscalar_t,
          MULTILABELMARGIN_SUB_GROUP_SIZE>(item, sum, smem_);

      if (item.get_local_id(0) == 0) {
        grad_input_k[target_idx] += static_cast<scalar_t>(sum);
      }
    }

    for (int d = item.get_local_id(0); d < dim_; d += item.get_local_range(0)) {
      grad_input_k[d] *= *grad_output_k;
    }
  }
  void sycl_ker_config_convention(sycl::handler& cgh) {
    smem_ = sycl_local_acc_t<accscalar_t>(smem_size_, cgh);
  }
  MultilabelMarginLossBackwardKernelFunctor(
      scalar_t* grad_input,
      const scalar_t* grad_output,
      const scalar_t* input,
      const int64_t* target,
      const scalar_t* is_target,
      int nframe,
      int dim,
      bool size_average,
      bool reduce,
      int64_t smem_size)
      : grad_input_(grad_input),
        grad_output_(grad_output),
        input_(input),
        target_(target),
        is_target_(is_target),
        nframe_(nframe),
        dim_(dim),
        size_average_(size_average),
        reduce_(reduce),
        smem_size_(smem_size) {}

 private:
  scalar_t* grad_input_;
  const scalar_t* grad_output_;
  const scalar_t* input_;
  const int64_t* target_;
  const scalar_t* is_target_;
  int nframe_;
  int dim_;
  bool size_average_;
  bool reduce_;
  int64_t smem_size_;
  sycl_local_acc_t<accscalar_t> smem_;
};

void multilabel_margin_loss_kernel(
    const Tensor& input,
    const Tensor& target,
    int64_t reduction,
    Tensor& output,
    Tensor& is_target) {
  int64_t nframe, dim;
  const int64_t ndims = input.dim();
  multilabel_margin_loss_shape_check(nframe, dim, ndims, input, target);

  if (input.numel() == 0) {
    return;
  }

  auto input_ = input.contiguous();
  auto target_ = target.contiguous();
  auto is_target_ = is_target.contiguous();
  is_target_.resize_as_(target);

  if (input.dim() <= 1) {
    output.resize_({});

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input.scalar_type(),
        "multilabel_margin_loss_xpu",
        [&] {
          using accscalar_t = acc_type_device<scalar_t, kXPU>;
          using KernelClass =
              MultilabelMarginLossForwardKernelFunctor<scalar_t, accscalar_t>;
          int64_t local_size = MULTILABELMARGIN_THREADS;
          auto kfn = KernelClass(
              output.mutable_data_ptr<scalar_t>(),
              input_.const_data_ptr<scalar_t>(),
              target_.const_data_ptr<int64_t>(),
              is_target_.mutable_data_ptr<scalar_t>(),
              1,
              dim,
              reduction == at::Reduction::Mean,
              local_size);
          sycl_kernel_submit(
              local_size, local_size, getCurrentSYCLQueue(), kfn);
        });
  } else if (input.dim() == 2) {
    if (reduction != at::Reduction::None) {
      auto output_tmp = at::empty({input_.size(0)}, input_.options());
      output.resize_({});
      AT_DISPATCH_FLOATING_TYPES_AND2(
          at::ScalarType::Half,
          at::ScalarType::BFloat16,
          input.scalar_type(),
          "multilabel_margin_loss_xpu",
          [&] {
            using accscalar_t = acc_type_device<scalar_t, kXPU>;
            using KernelClass =
                MultilabelMarginLossForwardKernelFunctor<scalar_t, accscalar_t>;
            int64_t local_size = MULTILABELMARGIN_THREADS;
            auto kfn = KernelClass(
                output_tmp.mutable_data_ptr<scalar_t>(),
                input_.const_data_ptr<scalar_t>(),
                target_.const_data_ptr<int64_t>(),
                is_target_.mutable_data_ptr<scalar_t>(),
                nframe,
                dim,
                reduction == at::Reduction::Mean,
                local_size);
            sycl_kernel_submit(
                input.size(0) * local_size,
                local_size,
                getCurrentSYCLQueue(),
                kfn);
          });
      at::sum_out(
          output,
          output_tmp,
          at::IntArrayRef(std::vector<int64_t>{}),
          false,
          output.scalar_type());
    } else {
      output.resize_({input.size(0)});
      AT_DISPATCH_FLOATING_TYPES_AND2(
          at::ScalarType::Half,
          at::ScalarType::BFloat16,
          input.scalar_type(),
          "multilabel_margin_loss_xpu",
          [&] {
            using accscalar_t = acc_type_device<scalar_t, kXPU>;
            using KernelClass =
                MultilabelMarginLossForwardKernelFunctor<scalar_t, accscalar_t>;
            int64_t local_size = MULTILABELMARGIN_THREADS;
            auto kfn = KernelClass(
                output.mutable_data_ptr<scalar_t>(),
                input_.const_data_ptr<scalar_t>(),
                target_.const_data_ptr<int64_t>(),
                is_target_.mutable_data_ptr<scalar_t>(),
                nframe,
                dim,
                false,
                local_size);
            sycl_kernel_submit(
                input.size(0) * local_size,
                local_size,
                getCurrentSYCLQueue(),
                kfn);
          });
    }

  } else {
    TORCH_CHECK(
        false,
        "Expected 2D input with optional zero batch dim, or 1D input with non-zero dims, but got sizes: ",
        input.sizes());
  }
}

void multilabel_margin_loss_backward_kernel(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& target,
    int64_t reduction,
    const Tensor& is_target,
    Tensor& grad_input) {
  int64_t nframe, dim;
  const int64_t ndims = input.dim();
  multilabel_margin_loss_shape_check(nframe, dim, ndims, input, target);

  if (input.numel() == 0) {
    return;
  }

  auto input_ = input.contiguous();
  auto target_ = target.contiguous();
  auto is_target_ = is_target.contiguous();
  auto grad_output_ = grad_output.contiguous();
  grad_input.resize_as_(input_);

  if (grad_input.dim() <= 1) {
    int target_size = target_.dim() == 0 ? 1 : target_.size(0);
    TORCH_CHECK(
        (target_.numel() != 0) && (target_.dim() <= 1) && (target_size == dim),
        "inconsistent target size");
    TORCH_CHECK(
        target_.sizes() == is_target_.sizes(), "inconsistent is_target size");

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input.scalar_type(),
        "multilabel_margin_loss_backward_kernel",
        [&] {
          using accscalar_t = acc_type_device<scalar_t, kXPU>;
          int64_t local_size = MULTILABELMARGIN_THREADS;
          auto kfn =
              MultilabelMarginLossBackwardKernelFunctor<scalar_t, accscalar_t>(
                  grad_input.mutable_data_ptr<scalar_t>(),
                  grad_output_.const_data_ptr<scalar_t>(),
                  input_.const_data_ptr<scalar_t>(),
                  target_.const_data_ptr<int64_t>(),
                  is_target_.const_data_ptr<scalar_t>(),
                  1,
                  dim,
                  reduction == at::Reduction::Mean,
                  reduction != at::Reduction::None,
                  local_size);
          sycl_kernel_submit(
              local_size, local_size, getCurrentSYCLQueue(), kfn);
        });
  } else if (grad_input.dim() == 2) {
    TORCH_CHECK(
        (input_.size(1) != 0) && (target_.dim() == 2) &&
            (target_.size(0) == nframe) && (target_.size(1) == dim),
        "inconsistent target size");
    TORCH_CHECK(
        target_.sizes() == is_target_.sizes(), "inconsistent is_target size");

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input.scalar_type(),
        "multilabel_margin_loss_backward_kernel",
        [&] {
          using accscalar_t = acc_type_device<scalar_t, kXPU>;
          int64_t local_size = MULTILABELMARGIN_THREADS;
          auto kfn =
              MultilabelMarginLossBackwardKernelFunctor<scalar_t, accscalar_t>(
                  grad_input.mutable_data_ptr<scalar_t>(),
                  grad_output_.const_data_ptr<scalar_t>(),
                  input_.const_data_ptr<scalar_t>(),
                  target_.const_data_ptr<int64_t>(),
                  is_target_.const_data_ptr<scalar_t>(),
                  grad_input.size(0),
                  grad_input.size(1),
                  reduction == at::Reduction::Mean,
                  reduction != at::Reduction::None,
                  local_size);
          sycl_kernel_submit(
              grad_input.size(0) * local_size,
              local_size,
              getCurrentSYCLQueue(),
              kfn);
        });
  } else {
    TORCH_CHECK(
        false,
        "Expected 2D input with optional zero batch dim, or 1D input with non-zero dims, but got sizes: ",
        grad_input.sizes());
  }
}

} // namespace at::native::xpu
