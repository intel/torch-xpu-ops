#include <ATen/ATen.h>
#include <ATen/Functions.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/Reduction.h>

#include <core/Device.h>
#include <core/Memory.h>
#include <comm/SYCLContext.h>

namespace at::native::xpu {

template <typename scalar_t>
struct ClassNLLCriterionUpdateOutputKernelFunctor {
  void operator()(sycl::item<1> item_id) const {
    auto input_ptr = input_data;
    auto target_ptr = target_data;
    auto weights_ptr = has_weights ? weights_data : NULL;
    auto output_ptr = output_data;
    auto local_item_id = item_id.get_id(0);
    for (int i = local_item_id; i < batch_size; i += local_size) {
      int cur_target = target_ptr[i * target_stride];
      if (cur_target >= 0 && cur_target < n_classes)
        if (cur_target == ignore_index) {
          output_ptr[i * output_stride_0] = 0.0f;
          continue;
        }
      scalar_t cur_weight =
          has_weights ? weights_ptr[cur_target] : static_cast<scalar_t>(1.0f);
      output_ptr[i * output_stride_0] =
          -static_cast<scalar_t>(
              input_ptr[i * input_stride_0 + cur_target * input_stride_1]) *
          cur_weight;
    }
  }
  ClassNLLCriterionUpdateOutputKernelFunctor(
      scalar_t* input_data_,
      int64_t* target_data_,
      scalar_t* weights_data_,
      scalar_t* output_data_,
      bool has_weights_,
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
        weights_data(weights_data_),
        output_data(output_data_),
        has_weights(has_weights_),
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
  int64_t* target_data;
  scalar_t* weights_data;
  scalar_t* output_data;
  bool has_weights;
  int64_t batch_size;
  int64_t local_size;
  int64_t target_stride;
  int n_classes;
  int64_t ignore_index;
  int64_t output_stride_0;
  int64_t input_stride_0;
  int64_t input_stride_1;
};

template <typename scalar_t>
struct ClassNLLCriterionUpdateOutputKernelFunctor2 {
  void operator()(sycl::item<1> item_id) const {
    auto input_ptr = input_data;
    auto target_ptr = target_data;
    auto weights_ptr = has_weights ? weights_data : NULL;
    auto total_weight_ptr = total_weight_data;
    auto output_ptr = output_data;
    // auto local_item_id = item_id.get_id(0);
    int cur_target = target_ptr[0];
    if (cur_target != ignore_index) {
      total_weight_ptr[0] =
          has_weights ? weights_ptr[cur_target] : static_cast<scalar_t>(1.0f);
      output_ptr[0] = -static_cast<scalar_t>(input_ptr[cur_target]) *
          static_cast<scalar_t>(total_weight_ptr[0]);
    }
    if (reduction == at::Reduction::Mean && total_weight_ptr[0]) {
      output_ptr[0] /= total_weight_ptr[0];
    }
  }
  ClassNLLCriterionUpdateOutputKernelFunctor2(
      scalar_t* input_data_,
      int64_t* target_data_,
      scalar_t* weights_data_,
      scalar_t* output_data_,
      scalar_t* total_weight_data_,
      bool has_weights_,
      int64_t batch_size_,
      int64_t local_size_,
      int64_t target_stride_,
      int n_classes_,
      int64_t ignore_index_,
      int64_t reduction_)
      : input_data(input_data_),
        target_data(target_data_),
        weights_data(weights_data_),
        output_data(output_data_),
        total_weight_data(total_weight_data_),
        has_weights(has_weights_),
        batch_size(batch_size_),
        local_size(local_size_),
        target_stride(target_stride_),
        n_classes(n_classes_),
        ignore_index(ignore_index_),
        reduction(reduction_) {}

 private:
  scalar_t* input_data;
  int64_t* target_data;
  scalar_t* weights_data;
  scalar_t* output_data;
  scalar_t* total_weight_data;
  bool has_weights;
  int64_t batch_size;
  int64_t local_size;
  int64_t target_stride;
  int n_classes;
  int64_t ignore_index;
  int64_t reduction;
};

template <typename scalar_t>
struct ClassNLLCriterionUpdateOutputKernelFunctor3 {
  void operator()(sycl::nd_item<1> item_id) const {
    auto input_ptr = input_data;
    auto target_ptr = target_data;
    auto weights_ptr = has_weights ? weights_data : NULL;
    auto total_weight_ptr = total_weight_data;
    auto output_ptr = output_data;
    int64_t local_id = item_id.get_local_id(0);
    local_output_acc[local_id] = 0.0;
    local_total_weight_acc[local_id] = 0.0;
    for (int i = local_id; i < batch_size; i += local_size) {
      int cur_target = target_ptr[i];
      if (cur_target != ignore_index) {
        scalar_t cur_weight =
            has_weights ? weights_ptr[cur_target] : static_cast<scalar_t>(1.0f);
        local_total_weight_acc[local_id] += cur_weight;
        local_output_acc[local_id] -=
            static_cast<scalar_t>(input_ptr[i * n_target + cur_target]) *
            static_cast<scalar_t>(cur_weight);
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

    output_ptr[0] = local_output_acc[0];
    total_weight_ptr[0] = local_total_weight_acc[0];
    if (reduction == at::Reduction::Mean && total_weight_ptr[0]) {
      output_ptr[0] /= total_weight_ptr[0];
    }
  }
  ClassNLLCriterionUpdateOutputKernelFunctor3(
      scalar_t* input_data_,
      int64_t* target_data_,
      scalar_t* weights_data_,
      scalar_t* output_data_,
      scalar_t* total_weight_data_,
      bool has_weights_,
      int64_t batch_size_,
      int64_t local_size_,
      int64_t target_stride_,
      int n_classes_,
      int64_t ignore_index_,
      int n_target_,
      sycl_local_acc_t<scalar_t> local_output_acc_,
      sycl_local_acc_t<scalar_t> local_total_weight_acc_,
      int64_t reduction_)
      : input_data(input_data_),
        target_data(target_data_),
        weights_data(weights_data_),
        output_data(output_data_),
        total_weight_data(total_weight_data_),
        has_weights(has_weights_),
        batch_size(batch_size_),
        local_size(local_size_),
        target_stride(target_stride_),
        n_classes(n_classes_),
        ignore_index(ignore_index_),
        n_target(n_target_),
        local_output_acc(local_output_acc_),
        local_total_weight_acc(local_total_weight_acc_),
        reduction(reduction_) {}

 private:
  scalar_t* input_data;
  int64_t* target_data;
  scalar_t* weights_data;
  scalar_t* output_data;
  scalar_t* total_weight_data;
  bool has_weights;
  int64_t batch_size;
  int64_t local_size;
  int64_t target_stride;
  int n_classes;
  int64_t ignore_index;
  int n_target;
  sycl_local_acc_t<scalar_t> local_output_acc;
  sycl_local_acc_t<scalar_t> local_total_weight_acc;
  int64_t reduction;
};


template <typename scalar_t>
void ClassNLLCriterion_updateOutput(
    const Tensor& input,
    const Tensor& target,
    Tensor& output,
    const optional<Tensor>& weights,
    Tensor& total_weight,
    int64_t reduction,
    int64_t ignore_index) {
  TORCH_CHECK(
      input.dim() > 0 && input.dim() <= 2, "input tensor should be 1D or 2D");
  TORCH_CHECK(
      target.dim() == 1,
      "1D target tensor expected, multi-target not supported");
  TORCH_CHECK(
      input.size(0) == target.size(0),
      "size mismatch (got input: ",
      input.sizes(),
      ", target: ",
      target.sizes(),
      ")")

  int n_dims = input.dim();
  int n_classes = input.size(-1);
  ignore_index -= 0;

  int64_t batch_size = input.size(0);
  int64_t num_targets = target.size(0);
  int64_t target_stride = target.stride(0);

  const Tensor weights_val = weights.value();
  TORCH_CHECK(
      !weights_val.defined() ||
          (weights_val.dim() == 1 && weights_val.numel() == n_classes),
      "weight tensor should be defined either for all ",
      n_classes,
      " classes or no classes"
      " but got weight tensor of shape: ",
      weights_val.sizes());

  if (reduction == at::Reduction::None && n_dims == 2) {
    output.resize_({batch_size});

    auto weights_cont =
        weights_val.defined() ? weights_val.contiguous() : weights_val;

    auto& queue = getCurrentSYCLQueue();
    auto dev_id = getDeviceIndexOfCurrentQueue();
    int64_t local_size = syclMaxWorkGroupSize(dev_id);
    bool has_weights = weights_val.defined()
        ? true
        : false; // sycl kernel can not accept host pointer

    auto output_stride_0 = output.stride(0);
    auto input_stride_0 = input.stride(0);
    auto input_stride_1 = input.stride(1);

    auto input_data = input.data_ptr<scalar_t>();
    auto target_data = target.data_ptr<int64_t>();
    auto weights_data = has_weights
        ? weights_cont.data_ptr<scalar_t>()
        : input_data; // use the input as the dummy data.
    auto output_data = output.data_ptr<scalar_t>();
    ClassNLLCriterionUpdateOutputKernelFunctor<scalar_t> kfn(
        input_data,
        target_data,
        weights_data,
        output_data,
        has_weights,
        batch_size,
        local_size,
        target_stride,
        n_classes,
        ignore_index,
        output_stride_0,
        input_stride_0,
        input_stride_1);

    // DPCPP_Q_SUBMIT(queue, cgf);
    sycl_kernel_submit(sycl::range<1>(local_size), queue, kfn);
    return;
  }

  output.resize_({});
  total_weight.resize_({});

  auto input_cont = input.contiguous();
  auto weights_cont =
      weights_val.defined() ? weights_val.contiguous() : weights_val;
  auto target_cont = target.contiguous();

  scalar_t* _input_data = input_cont.data_ptr<scalar_t>();
  scalar_t* _weights_data =
      weights_val.defined() ? weights_cont.data_ptr<scalar_t>() : NULL;
  int64_t* _target_data = target_cont.data_ptr<int64_t>();
  scalar_t* _output_data = output.data_ptr<scalar_t>();
  scalar_t* _total_weight_data = total_weight.data_ptr<scalar_t>();
  bool has_weights = _weights_data != NULL ? true : false;
  auto& queue = getCurrentSYCLQueue();

  if (input_cont.dim() == 1 || input_cont.dim() == 0) {
    int64_t local_size = 1;

    auto cgf = DPCPP_Q_CGF(cgh) {
      auto input_data = _input_data;
      auto weights_data = has_weights
          ? _weights_data
          : input_data; // use the input as the dummy data.
      auto target_data = _target_data;
      auto total_weight_data = _total_weight_data;
      auto output_data = _output_data;
      ClassNLLCriterionUpdateOutputKernelFunctor2<scalar_t> kfn(
          input_data,
          target_data,
          weights_data,
          output_data,
          total_weight_data,
          has_weights,
          batch_size,
          local_size,
          target_stride,
          n_classes,
          ignore_index,
          reduction);
      cgh.parallel_for<decltype(kfn)>(sycl::range<1>(local_size), kfn);
    };

    DPCPP_Q_SUBMIT(queue, cgf);
  } else if (input.dim() == 2) {
    int64_t batch_size = input.size(0);
    int n_target = input.size(1);
    auto dev_id = getDeviceIndexOfCurrentQueue();
    int64_t local_size = syclMaxWorkGroupSize(dev_id);

    auto cgf = DPCPP_Q_CGF(cgh) {
      auto input_data = _input_data;
      auto weights_data = has_weights
          ? _weights_data
          : input_data; // use the input as the dummy data.
      auto target_data = _target_data;
      auto total_weight_data = _total_weight_data;
      auto output_data = _output_data;
      auto local_output_acc = sycl_local_acc_t<scalar_t>(local_size, cgh);
      auto local_total_weight_acc =
          sycl_local_acc_t<scalar_t>(local_size, cgh);

      ClassNLLCriterionUpdateOutputKernelFunctor3<scalar_t> kfn(
          input_data,
          target_data,
          weights_data,
          output_data,
          total_weight_data,
          has_weights,
          batch_size,
          local_size,
          target_stride,
          n_classes,
          ignore_index,
          n_target,
          local_output_acc,
          local_total_weight_acc,
          reduction);
      cgh.parallel_for<decltype(kfn)>(
          sycl::nd_range<1>(
              sycl::range<1>(local_size), sycl::range<1>(local_size)),
          kfn);
    };

    DPCPP_Q_SUBMIT(queue, cgf);
  }
}

template <typename scalar_t>
struct ClassNLLCriterionUpdateGradInputKernelFunctor {
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
  ClassNLLCriterionUpdateGradInputKernelFunctor(
      int64_t* target_data_,
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
  int64_t* target_data;
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

template <typename scalar_t>
struct ClassNLLCriterionUpdateGradInputKernelFunctor2 {
  void operator()(sycl::item<1> item_id) const {
    auto gradOutput_ptr = gradOutput_data;
    auto weights_ptr = has_weights ? weights_data : NULL;
    auto gradInput_ptr = gradInput_data;
    auto target_ptr = target_data;
    auto total_weight_ptr = total_weight_data;

    if (*total_weight_ptr <= 0)
      return;
    scalar_t norm = (reduction == at::Reduction::Mean)
        ? (static_cast<scalar_t>(1) / static_cast<scalar_t>(*total_weight_ptr))
        : static_cast<scalar_t>(1);
    int t = (int)*target_ptr;
    if (t != (int)ignore_index) {
      gradInput_ptr[t] =
          -(has_weights ? weights_ptr[t] : static_cast<scalar_t>(1)) * norm *
          gradOutput_ptr[0];
    }
  }
  ClassNLLCriterionUpdateGradInputKernelFunctor2(
      int64_t* target_data_,
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
  int64_t* target_data;
  scalar_t* gradOutput_data;
  scalar_t* weights_data;
  scalar_t* gradInput_data;
  scalar_t* total_weight_data;
  bool has_weights;
  int64_t ignore_index;
  int64_t reduction;
};

template <typename scalar_t>
struct ClassNLLCriterionUpdateGradInputKernelFunctor3 {
  void operator()(sycl::item<1> item_id) const {
    auto gradOutput_ptr = gradOutput_data;
    auto weights_ptr = has_weights ? weights_data : NULL;
    auto gradInput_ptr = gradInput_data;
    auto target_ptr = target_data;
    auto total_weight_ptr = total_weight_data;

    auto local_item_id = item_id.get_id(0);

    if (*total_weight_ptr <= 0)
      return;
    int i, t;
    scalar_t norm = (reduction == at::Reduction::Mean)
        ? (static_cast<scalar_t>(1.0f) /
           static_cast<scalar_t>(*total_weight_ptr))
        : static_cast<scalar_t>(1);
    for (i = local_item_id; i < nframe; i += local_size) {
      t = (int)target_ptr[i];
      if (t != (int)ignore_index) {
        // assert(t >= 0 && t < n_classes)
        gradInput_ptr[i * ndim + t] =
            -(has_weights ? weights_ptr[t] : static_cast<scalar_t>(1)) * norm *
            gradOutput_ptr[0];
      }
    }
  }
  ClassNLLCriterionUpdateGradInputKernelFunctor3(
      int64_t* target_data_,
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
  int64_t* target_data;
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

template <typename scalar_t>
void ClassNLLCriterion_updateGradInput(
    const Tensor& input,
    const Tensor& target,
    const Tensor& gradOutput,
    Tensor& gradInput,
    int64_t reduction,
    const optional<Tensor>& weights,
    const Tensor& total_weight,
    int64_t ignore_index) {
  TORCH_CHECK(
      input.dim() > 0 && input.dim() <= 2, "input tensor should be 1D or 2D");
  TORCH_CHECK(
      target.dim() <= 1,
      "0D or 1D target tensor expected, multi-target not supported");

  auto no_batch_dim = input.dim() == 1 && target.dim() == 0;
  TORCH_CHECK(
      no_batch_dim || (input.size(0) == target.size(0)),
      "size mismatch (got input: ",
      input.sizes(),
      ", target: ",
      target.sizes(),
      ")");
  TORCH_CHECK(
      total_weight.numel() == 1,
      "expected total_weight to be a  single element tensor, got: ",
      total_weight.sizes(),
      " (",
      total_weight.numel(),
      " elements)");

  int n_dims = input.dim();
  int n_classes = input.size(-1);

  gradInput.resize_as_(input);
  gradInput.zero_();
  TORCH_CHECK(gradInput.is_contiguous(), "gradInput must be contiguous");

  TORCH_CHECK(
      input.defined() && (n_dims <= 2 && n_dims > 0),
      "input tensor should be 1D or 2D");

  int64_t batch_size = input.size(0);
  int64_t num_targets = target.size(0);
  int64_t target_stride = target.stride(0);

  TORCH_CHECK(
      batch_size == num_targets,
      "mismatch between the batch size of input and that of target")

  const Tensor weights_val = weights.value();
  TORCH_CHECK(
      !weights_val.defined() || weights_val.numel() == input.size(-1),
      "weight tensor should be defined either for all or no classes");

  if (reduction == at::Reduction::None && n_dims == 2) {
    check_dim_size(gradOutput, 1, 0, batch_size);
    auto weights_cont =
        weights_val.defined() ? weights_val.contiguous() : weights_val;

    auto& queue = getCurrentSYCLQueue();
    auto dev_id = getDeviceIndexOfCurrentQueue();
    int64_t local_size = syclMaxWorkGroupSize(dev_id);
    int64_t global_size =
        ((batch_size + local_size - 1) / local_size) * local_size;
    bool has_weights = weights_val.defined() ? true : false;

    auto gradInput_stride_0 = gradInput.stride(0);
    auto gradInput_stride_1 = gradInput.stride(1);
    auto gradOutput_stride_0 = gradOutput.stride(0);
    auto cgf = DPCPP_Q_CGF(cgh) {
      auto target_data = target.data_ptr<int64_t>();
      auto gradOutput_data = gradOutput.data_ptr<scalar_t>();
      auto weights_data = has_weights
          ? weights_cont.data_ptr<scalar_t>()
          : gradOutput_data; // Use gradOutput handler as dummy weights
      auto gradInput_data = gradInput.data_ptr<scalar_t>();
      ClassNLLCriterionUpdateGradInputKernelFunctor<scalar_t> kfn(
          target_data,
          gradOutput_data,
          weights_data,
          gradInput_data,
          has_weights,
          local_size,
          batch_size,
          target_stride,
          ignore_index,
          gradInput_stride_0,
          gradInput_stride_1,
          gradOutput_stride_0);
      cgh.parallel_for<decltype(kfn)>(
          sycl::nd_range<1>(
              sycl::range<1>(global_size), sycl::range<1>(local_size)),
          kfn);
    };

    DPCPP_Q_SUBMIT(queue, cgf);
    return;
  }

  auto weights_cont =
      weights_val.defined() ? weights_val.contiguous() : weights_val;
  auto target_cont = target.contiguous();
  bool has_weights = weights_val.defined() ? true : false;

  TORCH_CHECK(
      gradOutput.dim() <= 1 && gradOutput.numel() == 1,
      "Expected a single element grad_output tensor, but got: ",
      gradOutput.sizes());

  auto& queue = getCurrentSYCLQueue();
  if (input.dim() == 1) {
    auto cgf = DPCPP_Q_CGF(cgh) {
      auto gradOutput_data = gradOutput.data_ptr<scalar_t>();
      auto weights_data = has_weights
          ? weights_cont.data_ptr<scalar_t>()
          : gradOutput_data; // Use gradOutput handler as dummy weights
      auto gradInput_data = gradInput.data_ptr<scalar_t>();
      auto target_data = target_cont.data_ptr<int64_t>();
      auto total_weight_data = total_weight.data_ptr<scalar_t>();
      ClassNLLCriterionUpdateGradInputKernelFunctor2<scalar_t> kfn(
          target_data,
          gradOutput_data,
          weights_data,
          gradInput_data,
          total_weight_data,
          has_weights,
          ignore_index,
          reduction);
      cgh.parallel_for<decltype(kfn)>(sycl::range<1>(1), kfn);
    };
    DPCPP_Q_SUBMIT(queue, cgf);
  } else {
    int nframe = input.size(0);
    int ndim = input.size(1);
    int64_t local_size = 32;
    auto cgf = DPCPP_Q_CGF(cgh) {
      auto gradOutput_data = gradOutput.data_ptr<scalar_t>();
      auto weights_data = has_weights
          ? weights_cont.data_ptr<scalar_t>()
          : gradOutput_data; // use the gradOutput handler as dummy weights
      auto gradInput_data = gradInput.data_ptr<scalar_t>();
      auto target_data = target_cont.data_ptr<int64_t>();
      auto total_weight_data = total_weight.data_ptr<scalar_t>();
      ClassNLLCriterionUpdateGradInputKernelFunctor3<scalar_t> kfn(
          target_data,
          gradOutput_data,
          weights_data,
          gradInput_data,
          total_weight_data,
          has_weights,
          ignore_index,
          reduction,
          ndim,
          local_size,
          nframe);
      cgh.parallel_for<decltype(kfn)>(sycl::range<1>(local_size), kfn);
    };

    DPCPP_Q_SUBMIT(queue, cgf);
  }
}

std::tuple<Tensor&, Tensor&> launch_nll_loss_forward_kernel(
    const Tensor& self,
    const Tensor& target,
    const optional<Tensor>& weight,
    int64_t reduction,
    int64_t ignore_index,
    Tensor& output,
    Tensor& total_weight) {
  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "ClassNLLCriterion_updateOutput",
      [&]() {
        impl::ClassNLLCriterion_updateOutput<scalar_t>(
            self,
            target,
            output,
            weight,
            total_weight,
            reduction,
            ignore_index);
      });

  return std::tuple<Tensor&, Tensor&>(output, total_weight);
}

Tensor& launch_nll_loss_backward_kernel(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    const optional<Tensor>& weight,
    int64_t reduction,
    int64_t ignore_index,
    const Tensor& total_weight,
    Tensor& grad_input) {
  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "ClassNLLCriterion_updateGradInput",
      [&]() {
        impl::ClassNLLCriterion_updateGradInput<scalar_t>(
            self,
            target,
            grad_output,
            grad_input,
            reduction,
            weight,
            total_weight,
            ignore_index);
      });
  return grad_input;
}
} // namespace at::native::xpu