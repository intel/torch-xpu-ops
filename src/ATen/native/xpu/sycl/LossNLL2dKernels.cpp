#pragma clang diagnostic push
#pragma GCC diagnostic push
// Avoid SYCL compiler return-type error
#pragma clang diagnostic ignored "-Wreturn-type"
#pragma GCC diagnostic ignored "-Wreturn-type"
#include <ATen/ATen.h>
#include <ATen/core/TensorAccessor.h>
#include <ATen/native/CanUse32BitIndexMath.h>
#include <ATen/native/Resize.h>
#include <ATen/native/xpu/sycl/Atomics.h>
#include <ATen/native/xpu/sycl/GroupReduceUtils.h>
#include <comm/SYCLContext.h>

#include <ATen/native/xpu/sycl/LossNLL2dKernels.h>

namespace at::native::xpu {
inline Tensor optional_contiguous(const Tensor& source) {
  return source.defined() ? source.contiguous() : source;
}

template <typename scalar_t>
inline const scalar_t* optional_data(const Tensor& source) {
  return source.defined() ? source.const_data_ptr<scalar_t>() : nullptr;
}

template <typename scalar_t>
struct NllLoss2dForwardNoReduceKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    int64_t batch_size = input_.size(0);
    int64_t n_classes = input_.size(1);
    int64_t H = input_.size(2);
    int64_t W = input_.size(3);

    int64_t linear_id =
        item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
    PackedTensorAccessor64<scalar_t, 3> output_result = output_;
    for (int32_t index = linear_id; linear_id < (n_threads_);
         linear_id += item.get_group_range(0) * item.get_local_range(0),
                 index = linear_id) {
      const int64_t b = index % batch_size;
      const int64_t h = (index / batch_size) % H;
      const int64_t w = (index / (batch_size * H)) % W;

      int64_t cur_target = target_[b][h][w];
      if (cur_target == ignore_index_) {
        output_result[b][h][w] = static_cast<scalar_t>(0);
        continue;
      }
      SYCL_KERNEL_ASSERT(cur_target >= 0 && cur_target < n_classes);
      scalar_t value = input_[b][cur_target][h][w];
      scalar_t cur_weight =
          weight_ != nullptr ? weight_[cur_target] : static_cast<scalar_t>(1);
      output_result[b][h][w] = -value * cur_weight;
    }
  }

  NllLoss2dForwardNoReduceKernelFunctor(
      int64_t n_threads,
      PackedTensorAccessor64<scalar_t, 4> input,
      PackedTensorAccessor64<int64_t, 3> target,
      PackedTensorAccessor64<scalar_t, 3> output,
      const scalar_t* weight,
      int64_t ignore_index)
      : n_threads_(n_threads),
        input_(input),
        target_(target),
        output_(output),
        weight_(weight),
        ignore_index_(ignore_index) {}

 private:
  int64_t n_threads_;
  PackedTensorAccessor64<scalar_t, 4> input_;
  PackedTensorAccessor64<int64_t, 3> target_;
  PackedTensorAccessor64<scalar_t, 3> output_;
  const scalar_t* weight_;
  int64_t ignore_index_;
};

template <typename scalar_t, typename accscalar_t, typename index_t, int SIMD>
struct NllLoss2dForwardKernelFunctor : public __SYCL_KER_CONFIG_CONVENTION__ {
  [[intel::reqd_sub_group_size(SIMD)]] void operator()(
      sycl::nd_item<1> item) const {
    scalar_t cur_weight;
    accscalar_t input_sum = 0;
    accscalar_t acc_weight = 0;

    index_t sample = item.get_group(0) / blocks_per_sample_;
    index_t toffset = sample * map_nelem_;
    index_t ioffset = sample * map_nelem_ * n_classes_;
    int step = item.get_local_range(0) * blocks_per_sample_;
    auto start =
        (item.get_group(0) % blocks_per_sample_) * item.get_local_range(0) +
        item.get_local_id(0);
    for (int i = start; i < map_nelem_; i += step) {
      index_t t = target_[toffset + i];
      if (t != ignore_index_) {
        SYCL_KERNEL_ASSERT(t >= 0 && t < n_classes_);
        cur_weight = weight_ != nullptr ? weight_[t] : static_cast<scalar_t>(1);
        const auto input_index = ioffset + i + map_nelem_ * t;
        SYCL_KERNEL_ASSERT(input_index >= 0);
        input_sum -= input_[input_index] * cur_weight;
        acc_weight += cur_weight;
      }
    }

    auto acc_weight_reduce = GroupReduceSumWithoutBroadcast<accscalar_t, SIMD>(
        item, acc_weight, acc_weight_smem_);
    auto input_sum_reduce = GroupReduceSumWithoutBroadcast<accscalar_t, SIMD>(
        item, input_sum, input_sum_smem_);

    if (item.get_local_id(0) == 0) {
      atomicAdd(
          sycl_global_ptr<scalar_t>(total_weight_),
          static_cast<scalar_t>(acc_weight_reduce));
      atomicAdd(
          sycl_global_ptr<scalar_t>(output_),
          static_cast<scalar_t>(input_sum_reduce));
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    acc_weight_smem_ = sycl_local_acc_t<accscalar_t>(work_group_size_, cgh);
    input_sum_smem_ = sycl_local_acc_t<accscalar_t>(work_group_size_, cgh);
  }

  NllLoss2dForwardKernelFunctor(
      scalar_t* output,
      scalar_t* total_weight,
      const scalar_t* input,
      const int64_t* target,
      const scalar_t* weight,
      int n_classes,
      int map_nelem,
      int blocks_per_sample,
      int64_t ignore_index,
      int64_t work_group_size)
      : output_(output),
        total_weight_(total_weight),
        input_(input),
        target_(target),
        weight_(weight),
        n_classes_(n_classes),
        map_nelem_(map_nelem),
        blocks_per_sample_(blocks_per_sample),
        ignore_index_(ignore_index),
        work_group_size_(work_group_size) {}

 private:
  scalar_t* output_;
  scalar_t* total_weight_;
  const scalar_t* input_;
  const int64_t* target_;
  const scalar_t* weight_;
  int n_classes_;
  int map_nelem_;
  int blocks_per_sample_;
  int64_t ignore_index_;
  int64_t work_group_size_;
  sycl_local_acc_t<accscalar_t> acc_weight_smem_;
  sycl_local_acc_t<accscalar_t> input_sum_smem_;
};

template <typename scalar_t>
struct NllLoss2dForwardAverageKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    *output_ /= *total_weight_;
  }

  NllLoss2dForwardAverageKernelFunctor(
      scalar_t* output,
      const scalar_t* total_weight)
      : output_(output), total_weight_(total_weight) {}

 private:
  scalar_t* output_;
  const scalar_t* total_weight_;
};

void nll_loss2d_forward_kernel(
    Tensor& output,
    Tensor& total_weight,
    const Tensor& input,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index) {
  if (reduction != at::Reduction::None) {
    at::globalContext().alertNotDeterministic("nll_loss2d_forward_xpu");
  }

  total_weight.resize_({});

  if (reduction == at::Reduction::None) {
    int64_t batch_size = input.size(0);
    int64_t H = input.size(2);
    int64_t W = input.size(3);
    int64_t count = batch_size * H * W;

    at::native::resize_output(output, {batch_size, H, W});
    if (count == 0) {
      // This guards from unnecessary operations and launching CUDA kernel with
      // 0 blocks.
      return;
    }
    auto weight_ = optional_contiguous(weight);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input.scalar_type(),
        "nll_loss2d_forward_no_reduce_kernel",
        [&] {
          NllLoss2dForwardNoReduceKernelFunctor kfn(
              count,
              input.packed_accessor64<scalar_t, 4>(),
              target.packed_accessor64<int64_t, 3>(),
              output.packed_accessor64<scalar_t, 3>(),
              optional_data<scalar_t>(weight_),
              ignore_index);
          int64_t local_range = syclMaxWorkGroupSize(kfn);
          auto global_range = (count + local_range - 1) / local_range;
          sycl_kernel_submit(
              global_range * local_range,
              local_range,
              getCurrentSYCLQueue(),
              kfn);
        });
    return;
  }

  // produce scalar outputs for the reduction case
  at::native::resize_output(output, {});

  if (target.numel() == 0) {
    if (reduction == Reduction::Mean) {
      output.fill_(std::numeric_limits<double>::quiet_NaN());
    } else {
      output.zero_();
    }
    total_weight.zero_();
    return;
  }

  auto input_ = input.contiguous();
  auto weight_ = optional_contiguous(weight);
  auto target_ = target.contiguous();

  output.zero_();
  total_weight.zero_();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "nll_loss2d_forward_kernel",
      [&input_,
       &weight_,
       &target_,
       &output,
       &total_weight,
       &input,
       &target,
       &reduction,
       &ignore_index] {
        using accscalar_t = acc_type_device<scalar_t, kXPU>;
        AT_DISPATCH_INDEX_TYPES(
            at::native::canUse32BitIndexMath(input_, INT_MAX)
                ? ScalarType::Int
                : ScalarType::Long,
            "nll_loss2d_forward_launcher",
            [&] {
              auto batch_size = target.size(0);
              int64_t map_nelem = target.numel() / batch_size;
              const int simd = 32;
              int64_t work_group_size = get_group_reduce_group_size(simd);
              int blocks_per_sample =
                  (map_nelem + work_group_size - 1) / work_group_size / 128;
              blocks_per_sample =
                  (blocks_per_sample == 0) ? 1 : blocks_per_sample;
              int total_blocks = blocks_per_sample * batch_size;
              NllLoss2dForwardKernelFunctor<
                  scalar_t,
                  accscalar_t,
                  index_t,
                  simd>
                  kfn(output.mutable_data_ptr<scalar_t>(),
                      total_weight.mutable_data_ptr<scalar_t>(),
                      input_.const_data_ptr<scalar_t>(),
                      target_.const_data_ptr<int64_t>(),
                      optional_data<scalar_t>(weight_),
                      input_.size(1),
                      input_.size(2) * input_.size(3),
                      blocks_per_sample,
                      ignore_index,
                      work_group_size);
              sycl_kernel_submit(
                  total_blocks * work_group_size,
                  work_group_size,
                  getCurrentSYCLQueue(),
                  kfn);
              // Divide by total_weight
              if (reduction == at::Reduction::Mean) {
                NllLoss2dForwardAverageKernelFunctor kfn_average(
                    output.mutable_data_ptr<scalar_t>(),
                    total_weight.const_data_ptr<scalar_t>());
                sycl_kernel_submit(1, 1, getCurrentSYCLQueue(), kfn_average);
              }
            });
      });
}

template <typename scalar_t>
struct NllLoss2dBackwardNoReduceKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    int64_t batch_size = target_.size(0);
    int64_t H = target_.size(1);
    int64_t W = target_.size(2);

    auto grad_input_result = grad_input_;
    int64_t linear_id =
        item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
    for (int32_t index = linear_id; linear_id < (n_threads_);
         linear_id += item.get_group_range(0) * item.get_local_range(0),
                 index = linear_id) {
      const int64_t b = index % batch_size;
      const int64_t h = (index / batch_size) % H;
      const int64_t w = (index / (batch_size * H)) % W;

      int64_t cur_target = target_[b][h][w];
      if (cur_target == ignore_index_) {
        continue;
      }
      scalar_t value = -(
          weight_ != nullptr ? weight_[cur_target] : static_cast<scalar_t>(1));
      grad_input_result[b][cur_target][h][w] = value * grad_output_[b][h][w];
    }
  }

  NllLoss2dBackwardNoReduceKernelFunctor(
      int64_t n_threads,
      PackedTensorAccessor64<int64_t, 3> target,
      PackedTensorAccessor64<scalar_t, 3> grad_output,
      PackedTensorAccessor64<scalar_t, 4> grad_input,
      const scalar_t* weight,
      int64_t ignore_index)
      : n_threads_(n_threads),
        target_(target),
        grad_output_(grad_output),
        grad_input_(grad_input),
        weight_(weight),
        ignore_index_(ignore_index) {}

 private:
  int64_t n_threads_;
  PackedTensorAccessor64<int64_t, 3> target_;
  PackedTensorAccessor64<scalar_t, 3> grad_output_;
  PackedTensorAccessor64<scalar_t, 4> grad_input_;
  const scalar_t* weight_;
  int64_t ignore_index_;
};

template <typename scalar_t>
struct NllLoss2dBackwardKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    const auto grad =
        -(size_average_ ? *grad_output_ / *total_weight_ : *grad_output_);

    const int sample = item.get_group(0) / blocks_per_sample_;
    const int step = item.get_local_range(0) * blocks_per_sample_;

    const int toffset = sample * map_nelem_;
    const auto* const target_thread = target_ + toffset;

    const int ioffset = sample * map_nelem_ * n_classes_;
    auto* const grad_input_thread = grad_input_ + ioffset;

    for (int i = (item.get_group(0) % blocks_per_sample_) *
                 item.get_local_range(0) +
             item.get_local_id(0);
         i < map_nelem_;
         i += step) {
      const int64_t t = target_thread[i];
      if (t != ignore_index_) {
        SYCL_KERNEL_ASSERT(t >= 0 && t < n_classes_);
        const auto grad_input_index = i + map_nelem_ * t;
        SYCL_KERNEL_ASSERT(grad_input_index >= 0);
        grad_input_thread[i + map_nelem_ * t] =
            weights_ != nullptr ? weights_[t] * grad : grad;
      }
    }
  }
  NllLoss2dBackwardKernelFunctor(
      scalar_t* grad_input,
      const scalar_t* grad_output,
      const int64_t* target,
      const scalar_t* weights,
      const scalar_t* total_weight,
      bool size_average,
      int n_classes,
      int map_nelem,
      int blocks_per_sample,
      int64_t ignore_index)
      : grad_input_(grad_input),
        grad_output_(grad_output),
        target_(target),
        weights_(weights),
        total_weight_(total_weight),
        size_average_(size_average),
        n_classes_(n_classes),
        map_nelem_(map_nelem),
        blocks_per_sample_(blocks_per_sample),
        ignore_index_(ignore_index) {}

 private:
  scalar_t* grad_input_;
  const scalar_t* grad_output_;
  const int64_t* target_;
  const scalar_t* weights_;
  const scalar_t* total_weight_;
  bool size_average_;
  int n_classes_;
  int map_nelem_;
  int blocks_per_sample_;
  int64_t ignore_index_;
};

void nll_loss2d_backward_kernel(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index,
    const Tensor& total_weight) {
  grad_input.resize_as_(input);
  grad_input.zero_();
  TORCH_CHECK(grad_input.is_contiguous(), "grad_input must be contiguous");
  TORCH_CHECK(
      total_weight.numel() == 1,
      "expected total_weight to be a single element tensor, got: ",
      total_weight.sizes(),
      " (",
      total_weight.numel(),
      " elements)");

  if (reduction == at::Reduction::None) {
    TORCH_CHECK(
        grad_output.dim() == 3,
        "grad_output must have same dimension as target (3) but got dimension: ",
        grad_output.sizes());
    TORCH_CHECK(
        grad_output.size(0) == target.size(0) &&
            grad_output.size(1) == target.size(1) &&
            grad_output.size(2) == target.size(2),
        "grad_output sizes don't match target sizes: target ",
        target.sizes(),
        ", grad_output ",
        grad_output.sizes())

    int64_t batch_size = input.size(0);
    int64_t H = input.size(2);
    int64_t W = input.size(3);
    int64_t count = batch_size * H * W;

    if (count == 0) {
      // This guards from unnecessary operations and launching kernel with
      // 0 blocks.
      return;
    }

    auto weight_ = optional_contiguous(weight);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input.scalar_type(),
        "nll_loss2d_backward_no_reduce_kernel",
        [&] {
          NllLoss2dBackwardNoReduceKernelFunctor kfn(
              count,
              target.packed_accessor64<int64_t, 3>(),
              grad_output.packed_accessor64<scalar_t, 3>(),
              grad_input.packed_accessor64<scalar_t, 4>(),
              optional_data<scalar_t>(weight_),
              ignore_index);
          int64_t local_range = syclMaxWorkGroupSize(kfn);
          auto global_range = (count + local_range - 1) / local_range;
          sycl_kernel_submit(
              global_range * local_range,
              local_range,
              getCurrentSYCLQueue(),
              kfn);
        });
    return;
  }

  int64_t batch_size = target.size(0);
  auto target_numel = target.numel();
  if (batch_size != 0 && target_numel != 0) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input.scalar_type(),
        "nll_loss2d_backward_kernel",
        [&] {
          // This guards from unnecessary operations and launching kernel with 1
          // blocks.
          auto target_ = target.contiguous();
          auto weight_ = optional_contiguous(weight);
          int64_t map_nelem = target_numel / batch_size;
          using KernelClass = NllLoss2dBackwardKernelFunctor<scalar_t>;
          int64_t max_work_group_size = syclMaxWorkGroupSize<KernelClass>();
          int blocks_per_sample =
              (map_nelem + max_work_group_size - 1) / max_work_group_size / 128;
          blocks_per_sample = (blocks_per_sample == 0) ? 1 : blocks_per_sample;
          int total_blocks = blocks_per_sample * batch_size;
          KernelClass kfn(
              grad_input.mutable_data_ptr<scalar_t>(),
              grad_output.const_data_ptr<scalar_t>(),
              target_.const_data_ptr<int64_t>(),
              optional_data<scalar_t>(weight_),
              total_weight.const_data_ptr<scalar_t>(),
              reduction == at::Reduction::Mean,
              input.size(1),
              map_nelem,
              blocks_per_sample,
              ignore_index);
          sycl_kernel_submit(
              total_blocks * max_work_group_size,
              max_work_group_size,
              getCurrentSYCLQueue(),
              kfn);
        });
  }
}

} // namespace at::native::xpu

#pragma GCC diagnostic pop
#pragma clang diagnostic pop
