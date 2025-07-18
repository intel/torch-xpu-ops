#include <ATen/AccumulateType.h>
#include <ATen/Functions.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/Reduction.h>
#include <comm/SYCLContext.h>
#include <comm/xpu_aten.h>

#include <ATen/native/xpu/sycl/LossNLLKernel.h>

namespace at::native::xpu {

using namespace at::xpu;
template <typename scalar_t, typename index_t>
struct NllLossForwardNoReduceKernelFunctor {
  void operator()(sycl::item<1> item_id) const {
    auto input_ptr = input_data;
    auto target_ptr = target_data;
    auto weight_ptr = has_weight ? weight_data : NULL;
    auto output_ptr = output_data;
    auto local_item_id = item_id.get_id(0);
    for (int i = local_item_id; i < batch_size; i += local_size) {
      int cur_target = target_ptr[i * target_stride];
      if (cur_target == ignore_index) {
        output_ptr[i * output_stride_0] = 0.0f;
        continue;
      }
      scalar_t cur_weight =
          has_weight ? weight_ptr[cur_target] : static_cast<scalar_t>(1.0f);
      output_ptr[i * output_stride_0] =
          -static_cast<scalar_t>(
              input_ptr[i * input_stride_0 + cur_target * input_stride_1]) *
          cur_weight;
    }
  }
  NllLossForwardNoReduceKernelFunctor(
      scalar_t* input_data_,
      index_t* target_data_,
      scalar_t* weight_data_,
      scalar_t* output_data_,
      bool has_weight_,
      int64_t batch_size_,
      int64_t local_size_,
      int64_t target_stride_,
      int n_classes_,
      int64_t ignore_index_,
      int64_t output_stride_0_,
      int64_t input_stride_0_,
      int64_t input_stride_1_)
      : input_data(input_data_),
        target_data(target_data_),
        weight_data(weight_data_),
        output_data(output_data_),
        has_weight(has_weight_),
        batch_size(batch_size_),
        local_size(local_size_),
        target_stride(target_stride_),
        n_classes(n_classes_),
        ignore_index(ignore_index_),
        output_stride_0(output_stride_0_),
        input_stride_0(input_stride_0_),
        input_stride_1(input_stride_1_) {}

 private:
  scalar_t* input_data;
  index_t* target_data;
  scalar_t* weight_data;
  scalar_t* output_data;
  bool has_weight;
  int64_t batch_size;
  int64_t local_size;
  int64_t target_stride;
  int n_classes;
  int64_t ignore_index;
  int64_t output_stride_0;
  int64_t input_stride_0;
  int64_t input_stride_1;
};

template <typename scalar_t, typename index_t>
struct NllLossForwardReduce1DKernelFunctor {
  void operator()(sycl::item<1> item_id) const {
    auto input_ptr = input_data;
    auto target_ptr = target_data;
    auto weight_ptr = has_weight ? weight_data : NULL;
    auto total_weight_ptr = total_weight_data;
    auto output_ptr = output_data;
    int cur_target = target_ptr[0];
    total_weight_ptr[0] =
        has_weight ? weight_ptr[cur_target] : static_cast<scalar_t>(1.0f);
    if (cur_target != ignore_index) {
      output_ptr[0] = -static_cast<scalar_t>(input_ptr[cur_target]) *
          static_cast<scalar_t>(total_weight_ptr[0]);
    } else {
      output_ptr[0] = static_cast<scalar_t>(0.f);
      total_weight_ptr[0] = static_cast<scalar_t>(0.f);
    }
    if (reduction == at::Reduction::Mean) {
      output_ptr[0] /= total_weight_ptr[0];
    }
  }
  NllLossForwardReduce1DKernelFunctor(
      scalar_t* input_data_,
      index_t* target_data_,
      scalar_t* weight_data_,
      scalar_t* output_data_,
      scalar_t* total_weight_data_,
      bool has_weight_,
      int64_t ignore_index_,
      int64_t reduction_)
      : input_data(input_data_),
        target_data(target_data_),
        weight_data(weight_data_),
        output_data(output_data_),
        total_weight_data(total_weight_data_),
        has_weight(has_weight_),
        ignore_index(ignore_index_),
        reduction(reduction_) {}

 private:
  scalar_t* input_data;
  index_t* target_data;
  scalar_t* weight_data;
  scalar_t* output_data;
  scalar_t* total_weight_data;
  bool has_weight;
  int64_t ignore_index;
  int64_t reduction;
};

template <typename scalar_t, typename index_t, typename accscalar_t>
struct NllLossForwardReduce2DKernelFunctor
    : public __SYCL_KER_CONFIG_CONVENTION__ {
  void operator()(sycl::nd_item<1> item_id) const {
    auto input_ptr = input_data;
    auto target_ptr = target_data;
    auto weight_ptr = has_weight ? weight_data : NULL;
    auto total_weight_ptr = total_weight_data;
    auto output_ptr = output_data;
    int64_t local_id = item_id.get_local_id(0);
    local_output_acc[local_id] = accscalar_t(0.0);
    local_total_weight_acc[local_id] = accscalar_t(0.0);
    for (int i = local_id; i < batch_size; i += local_size) {
      int cur_target = target_ptr[i];
      if (cur_target != ignore_index) {
        scalar_t cur_weight =
            has_weight ? weight_ptr[cur_target] : static_cast<scalar_t>(1.0f);
        local_total_weight_acc[local_id] +=
            static_cast<accscalar_t>(cur_weight);
        local_output_acc[local_id] -=
            static_cast<accscalar_t>(input_ptr[i * n_target + cur_target]) *
            static_cast<accscalar_t>(cur_weight);
      }
    }

    // reduce
    for (int64_t i = (local_size >> 1); i > 0; i >>= 1) {
      item_id.barrier(sycl_global_and_local_fence);
      if (local_id < i) {
        local_total_weight_acc[local_id] +=
            local_total_weight_acc[local_id + i];
        local_output_acc[local_id] += local_output_acc[local_id + i];
      }
    }
    item_id.barrier(sycl_global_and_local_fence);

    if (reduction == at::Reduction::Mean) {
      output_ptr[0] = static_cast<scalar_t>(
          local_output_acc[0] / local_total_weight_acc[0]);
    } else {
      output_ptr[0] = static_cast<scalar_t>(local_output_acc[0]);
    }
    total_weight_ptr[0] = static_cast<scalar_t>(local_total_weight_acc[0]);
  }
  NllLossForwardReduce2DKernelFunctor(
      scalar_t* input_data_,
      index_t* target_data_,
      scalar_t* weight_data_,
      scalar_t* output_data_,
      scalar_t* total_weight_data_,
      bool has_weight_,
      int64_t batch_size_,
      int64_t local_size_,
      int64_t ignore_index_,
      int n_target_,
      int64_t reduction_)
      : input_data(input_data_),
        target_data(target_data_),
        weight_data(weight_data_),
        output_data(output_data_),
        total_weight_data(total_weight_data_),
        has_weight(has_weight_),
        batch_size(batch_size_),
        local_size(local_size_),
        ignore_index(ignore_index_),
        n_target(n_target_),
        reduction(reduction_) {}

  void sycl_ker_config_convention(sycl::handler& cgh) {
    local_output_acc = sycl_local_acc_t<accscalar_t>(local_size, cgh);
    local_total_weight_acc = sycl_local_acc_t<accscalar_t>(local_size, cgh);
  }

 private:
  scalar_t* input_data;
  index_t* target_data;
  scalar_t* weight_data;
  scalar_t* output_data;
  scalar_t* total_weight_data;
  bool has_weight;
  int64_t batch_size;
  int64_t local_size;
  int64_t ignore_index;
  int n_target;
  sycl_local_acc_t<accscalar_t> local_output_acc;
  sycl_local_acc_t<accscalar_t> local_total_weight_acc;
  int64_t reduction;
};

template <typename scalar_t, typename index_t>
void nll_loss_forward_template(
    const Tensor& input,
    const Tensor& target,
    const Tensor& output,
    const Tensor& weight,
    const Tensor& total_weight,
    int64_t reduction,
    int64_t ignore_index) {
  int n_dims = input.dim();
  int n_classes = input.size(-1);
  ignore_index -= 0;

  int64_t batch_size = input.size(0);

  if (reduction == at::Reduction::None && n_dims == 2) {
    using NllLossForwardNoReduceKernel =
        NllLossForwardNoReduceKernelFunctor<scalar_t, index_t>;

    output.resize_({batch_size});
    total_weight.zero_();
    int64_t target_stride = target.stride(0);

    auto weight_cont = weight.defined() ? weight.contiguous() : weight;

    auto& queue = getCurrentSYCLQueue();
    int64_t local_size = syclMaxWorkGroupSize<NllLossForwardNoReduceKernel>();
    bool has_weight = weight.defined()
        ? true
        : false; // sycl kernel can not accept host pointer

    auto output_stride_0 = output.stride(0);
    auto input_stride_0 = input.stride(0);
    auto input_stride_1 = input.stride(1);

    auto input_data = input.data_ptr<scalar_t>();
    auto target_data = target.data_ptr<index_t>();
    auto weight_data = has_weight
        ? weight_cont.data_ptr<scalar_t>()
        : input_data; // use the input as the dummy data.
    auto output_data = output.data_ptr<scalar_t>();
    NllLossForwardNoReduceKernel kfn(
        input_data,
        target_data,
        weight_data,
        output_data,
        has_weight,
        batch_size,
        local_size,
        target_stride,
        n_classes,
        ignore_index,
        output_stride_0,
        input_stride_0,
        input_stride_1);

    sycl_kernel_submit(sycl::range<1>(local_size), queue, kfn);
    return;
  }

  output.resize_({});
  total_weight.resize_({});

  auto input_cont = input.contiguous();
  auto weight_cont = weight.defined() ? weight.contiguous() : weight;
  auto target_cont = target.contiguous();

  scalar_t* _input_data = input_cont.data_ptr<scalar_t>();
  scalar_t* _weight_data =
      weight.defined() ? weight_cont.data_ptr<scalar_t>() : NULL;
  index_t* _target_data = target_cont.data_ptr<index_t>();
  scalar_t* _output_data = output.data_ptr<scalar_t>();
  scalar_t* _total_weight_data = total_weight.data_ptr<scalar_t>();
  bool has_weight = _weight_data != NULL ? true : false;
  auto& queue = getCurrentSYCLQueue();

  if (input_cont.dim() == 1 || input_cont.dim() == 0) {
    int64_t local_size = 1;
    auto input_data = _input_data;
    auto weight_data = has_weight
        ? _weight_data
        : input_data; // use the input as the dummy data.
    auto target_data = _target_data;
    auto total_weight_data = _total_weight_data;
    auto output_data = _output_data;
    NllLossForwardReduce1DKernelFunctor<scalar_t, index_t> kfn(
        input_data,
        target_data,
        weight_data,
        output_data,
        total_weight_data,
        has_weight,
        ignore_index,
        reduction);

    sycl_kernel_submit(sycl::range<1>(local_size), queue, kfn);
  } else if (input_cont.dim() == 2) {
    using accscalar_t = at::acc_type<scalar_t, true>;
    using NllLossForwardReduce2DKernel =
        NllLossForwardReduce2DKernelFunctor<scalar_t, index_t, accscalar_t>;

    int64_t batch_size = input.size(0);
    int n_target = input.size(1);
    int64_t local_size = syclMaxWorkGroupSize<NllLossForwardReduce2DKernel>();
    auto input_data = _input_data;
    auto weight_data = has_weight
        ? _weight_data
        : input_data; // use the input as the dummy data.
    auto target_data = _target_data;
    auto total_weight_data = _total_weight_data;
    auto output_data = _output_data;
    NllLossForwardReduce2DKernelFunctor<scalar_t, index_t, accscalar_t> kfn(
        input_data,
        target_data,
        weight_data,
        output_data,
        total_weight_data,
        has_weight,
        batch_size,
        local_size,
        ignore_index,
        n_target,
        reduction);

    sycl_kernel_submit(
        sycl::range<1>(local_size), sycl::range<1>(local_size), queue, kfn);
  }
}

template <typename scalar_t, typename index_t>
struct NllLossBackwardNoReduceKernelFunctor {
  void operator()(sycl::nd_item<1> item_id) const {
    auto target_ptr = target_data;
    auto gradOutput_ptr = gradOutput_data;
    auto weights_ptr = has_weights ? weights_data : NULL;
    auto gradInput_ptr = gradInput_data;

    auto local_id = item_id.get_local_id(0);
    auto group_id = item_id.get_group(0);

    for (int i = group_id * local_size + local_id; i < batch_size;
         i += item_id.get_global_range(0)) {
      int cur_target = target_ptr[i * target_stride];
      if (cur_target == ignore_index) {
        continue;
      }
      scalar_t cur_weight =
          has_weights ? weights_ptr[cur_target] : static_cast<scalar_t>(1.0f);
      gradInput_ptr[i * gradInput_stride_0 + cur_target * gradInput_stride_1] =
          -cur_weight *
          static_cast<scalar_t>(gradOutput_ptr[i * gradOutput_stride_0]);
    }
  }
  NllLossBackwardNoReduceKernelFunctor(
      index_t* target_data_,
      scalar_t* gradOutput_data_,
      scalar_t* weights_data_,
      scalar_t* gradInput_data_,
      bool has_weights_,
      int64_t local_size_,
      int64_t batch_size_,
      int64_t target_stride_,
      int64_t ignore_index_,
      int64_t gradInput_stride_0_,
      int64_t gradInput_stride_1_,
      int64_t gradOutput_stride_0_)
      : target_data(target_data_),
        gradOutput_data(gradOutput_data_),
        weights_data(weights_data_),
        gradInput_data(gradInput_data_),
        has_weights(has_weights_),
        local_size(local_size_),
        batch_size(batch_size_),
        target_stride(target_stride_),
        ignore_index(ignore_index_),
        gradInput_stride_0(gradInput_stride_0_),
        gradInput_stride_1(gradInput_stride_1_),
        gradOutput_stride_0(gradOutput_stride_0_) {}

 private:
  index_t* target_data;
  scalar_t* gradOutput_data;
  scalar_t* weights_data;
  scalar_t* gradInput_data;
  bool has_weights;
  int64_t local_size;
  int64_t batch_size;
  int64_t target_stride;
  int64_t ignore_index;
  int64_t gradInput_stride_0;
  int64_t gradInput_stride_1;
  int64_t gradOutput_stride_0;
};

template <typename scalar_t, typename index_t>
struct NllLossBackwardReduce1DKernelFunctor {
  void operator()(sycl::item<1> item_id) const {
    auto gradOutput_ptr = gradOutput_data;
    auto weights_ptr = has_weights ? weights_data : NULL;
    auto gradInput_ptr = gradInput_data;
    auto target_ptr = target_data;
    auto total_weight_ptr = total_weight_data;

    int t = (int)*target_ptr;
    if (t != (int)ignore_index) {
      scalar_t grad =
          -((reduction == at::Reduction::Mean)
                ? static_cast<scalar_t>(gradOutput_ptr[0]) /
                    static_cast<scalar_t>(*total_weight_ptr)
                : static_cast<scalar_t>(gradOutput_ptr[0]));
      gradInput_ptr[t] = has_weights ? weights_ptr[t] * grad : grad;
    }
  }
  NllLossBackwardReduce1DKernelFunctor(
      index_t* target_data_,
      scalar_t* gradOutput_data_,
      scalar_t* weights_data_,
      scalar_t* gradInput_data_,
      scalar_t* total_weight_data_,
      bool has_weights_,
      int64_t ignore_index_,
      int64_t reduction_)
      : target_data(target_data_),
        gradOutput_data(gradOutput_data_),
        weights_data(weights_data_),
        gradInput_data(gradInput_data_),
        total_weight_data(total_weight_data_),
        has_weights(has_weights_),
        ignore_index(ignore_index_),
        reduction(reduction_) {}

 private:
  index_t* target_data;
  scalar_t* gradOutput_data;
  scalar_t* weights_data;
  scalar_t* gradInput_data;
  scalar_t* total_weight_data;
  bool has_weights;
  int64_t ignore_index;
  int64_t reduction;
};

template <typename scalar_t, typename index_t>
struct NllLossBackwardReduce2DKernelFunctor {
  void operator()(sycl::item<1> item_id) const {
    auto gradOutput_ptr = gradOutput_data;
    auto weights_ptr = has_weights ? weights_data : NULL;
    auto gradInput_ptr = gradInput_data;
    auto target_ptr = target_data;
    auto total_weight_ptr = total_weight_data;

    auto local_item_id = item_id.get_id(0);

    int i, t;

    scalar_t grad =
        -((reduction == at::Reduction::Mean)
              ? static_cast<scalar_t>(gradOutput_ptr[0]) /
                  static_cast<scalar_t>(*total_weight_ptr)
              : static_cast<scalar_t>(gradOutput_ptr[0]));
    for (i = local_item_id; i < nframe; i += local_size) {
      t = (int)target_ptr[i];
      if (t != (int)ignore_index) {
        gradInput_ptr[i * ndim + t] =
            has_weights ? weights_ptr[t] * grad : grad;
      }
    }
  }
  NllLossBackwardReduce2DKernelFunctor(
      index_t* target_data_,
      scalar_t* gradOutput_data_,
      scalar_t* weights_data_,
      scalar_t* gradInput_data_,
      scalar_t* total_weight_data_,
      bool has_weights_,
      int64_t ignore_index_,
      int64_t reduction_,
      int ndim_,
      int64_t local_size_,
      int nframe_)
      : target_data(target_data_),
        gradOutput_data(gradOutput_data_),
        weights_data(weights_data_),
        gradInput_data(gradInput_data_),
        total_weight_data(total_weight_data_),
        has_weights(has_weights_),
        ignore_index(ignore_index_),
        reduction(reduction_),
        ndim(ndim_),
        local_size(local_size_),
        nframe(nframe_) {}

 private:
  index_t* target_data;
  scalar_t* gradOutput_data;
  scalar_t* weights_data;
  scalar_t* gradInput_data;
  scalar_t* total_weight_data;
  bool has_weights;
  int64_t ignore_index;
  int64_t reduction;
  int ndim;
  int64_t local_size;
  int nframe;
};

template <typename scalar_t, typename index_t>
static inline void nll_loss_backward_template(
    const Tensor& input,
    const Tensor& target,
    const Tensor& gradOutput,
    const Tensor& gradInput,
    int64_t reduction,
    const Tensor& weight,
    const Tensor& total_weight,
    int64_t ignore_index) {
  int n_dims = input.dim();

  gradInput.resize_as_(input);
  gradInput.zero_();

  int64_t batch_size = input.size(0);

  if (reduction == at::Reduction::None && n_dims == 2) {
    using NllLossBackwardNoReduceKernel =
        NllLossBackwardNoReduceKernelFunctor<scalar_t, index_t>;

    int64_t target_stride = target.stride(0);
    check_dim_size(gradOutput, 1, 0, batch_size);
    auto weight_cont = weight.defined() ? weight.contiguous() : weight;

    auto& queue = getCurrentSYCLQueue();
    int64_t local_size = syclMaxWorkGroupSize<NllLossBackwardNoReduceKernel>();
    int64_t global_size =
        ((batch_size + local_size - 1) / local_size) * local_size;
    bool has_weight = weight.defined() ? true : false;

    auto gradInput_stride_0 = gradInput.stride(0);
    auto gradInput_stride_1 = gradInput.stride(1);
    auto gradOutput_stride_0 = gradOutput.stride(0);

    auto target_data = target.data_ptr<index_t>();
    auto gradOutput_data = gradOutput.data_ptr<scalar_t>();
    auto weight_data = has_weight
        ? weight_cont.data_ptr<scalar_t>()
        : gradOutput_data; // Use gradOutput handler as dummy weight
    auto gradInput_data = gradInput.data_ptr<scalar_t>();
    NllLossBackwardNoReduceKernel kfn(
        target_data,
        gradOutput_data,
        weight_data,
        gradInput_data,
        has_weight,
        local_size,
        batch_size,
        target_stride,
        ignore_index,
        gradInput_stride_0,
        gradInput_stride_1,
        gradOutput_stride_0);

    sycl_kernel_submit(
        sycl::range<1>(global_size), sycl::range<1>(local_size), queue, kfn);
    return;
  }

  auto weight_cont = weight.defined() ? weight.contiguous() : weight;
  auto target_cont = target.contiguous();
  bool has_weight = weight.defined() ? true : false;

  TORCH_CHECK(
      gradOutput.dim() <= 1 && gradOutput.numel() == 1,
      "Expected a single element grad_output tensor, but got: ",
      gradOutput.sizes());

  auto& queue = getCurrentSYCLQueue();
  if (n_dims == 1) {
    auto gradOutput_data = gradOutput.data_ptr<scalar_t>();
    auto weight_data = has_weight
        ? weight_cont.data_ptr<scalar_t>()
        : gradOutput_data; // Use gradOutput handler as dummy weight
    auto gradInput_data = gradInput.data_ptr<scalar_t>();
    auto target_data = target_cont.data_ptr<index_t>();
    auto total_weight_data = total_weight.data_ptr<scalar_t>();
    NllLossBackwardReduce1DKernelFunctor<scalar_t, index_t> kfn(
        target_data,
        gradOutput_data,
        weight_data,
        gradInput_data,
        total_weight_data,
        has_weight,
        ignore_index,
        reduction);

    sycl_kernel_submit(sycl::range<1>(1), queue, kfn);
  } else {
    int nframe = input.size(0);
    int ndim = input.size(1);
    int64_t local_size = 32;

    auto gradOutput_data = gradOutput.data_ptr<scalar_t>();
    auto weight_data = has_weight
        ? weight_cont.data_ptr<scalar_t>()
        : gradOutput_data; // use the gradOutput handler as dummy weight
    auto gradInput_data = gradInput.data_ptr<scalar_t>();
    auto target_data = target_cont.data_ptr<index_t>();
    auto total_weight_data = total_weight.data_ptr<scalar_t>();
    NllLossBackwardReduce2DKernelFunctor<scalar_t, index_t> kfn(
        target_data,
        gradOutput_data,
        weight_data,
        gradInput_data,
        total_weight_data,
        has_weight,
        ignore_index,
        reduction,
        ndim,
        local_size,
        nframe);

    sycl_kernel_submit(sycl::range<1>(local_size), queue, kfn);
  }
}

#define AT_DISPATCH_NLL_LOSS_INDEX_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                     \
      TYPE,                                               \
      NAME,                                               \
      AT_PRIVATE_CASE_TYPE_USING_HINT(                    \
          at::ScalarType::Byte, index_t, __VA_ARGS__)     \
          AT_PRIVATE_CASE_TYPE_USING_HINT(                \
              at::ScalarType::Long, index_t, __VA_ARGS__))

void nll_loss_forward_kernel(
    const Tensor& self,
    const Tensor& target,
    const OptionalTensorRef weight_opt,
    int64_t reduction,
    int64_t ignore_index,
    const Tensor& output,
    const Tensor& total_weight) {
  const Tensor& weight = weight_opt.getTensorRef();
  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "nll_loss_forward_out_kernel",
      [&]() {
        AT_DISPATCH_NLL_LOSS_INDEX_TYPES(
            target.scalar_type(), "nll_loss_forward_out_kernel_index", [&]() {
              nll_loss_forward_template<scalar_t, index_t>(
                  self,
                  target,
                  output,
                  weight,
                  total_weight,
                  reduction,
                  ignore_index);
            });
      });

  // return std::tuple<Tensor&, Tensor&>(output, total_weight);
}

void nll_loss_backward_kernel(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    const OptionalTensorRef weight_opt,
    int64_t reduction,
    int64_t ignore_index,
    const Tensor& total_weight,
    const Tensor& grad_input) {
  const Tensor& weight = weight_opt.getTensorRef();
  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "nll_loss_backward_out_kernel",
      [&]() {
        AT_DISPATCH_NLL_LOSS_INDEX_TYPES(
            target.scalar_type(), "nll_loss_backward_out_kernel_index", [&]() {
              nll_loss_backward_template<scalar_t, index_t>(
                  self,
                  target,
                  grad_output,
                  grad_input,
                  reduction,
                  weight,
                  total_weight,
                  ignore_index);
            });
      });
  // return grad_input;
}

#undef AT_DISPATCH_NLL_LOSS_INDEX_TYPES
} // namespace at::native::xpu
