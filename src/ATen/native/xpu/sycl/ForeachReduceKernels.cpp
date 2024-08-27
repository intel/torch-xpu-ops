#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/native/ForeachUtils.h>

#include <ATen/native/xpu/sycl/ForeachFunctors.h>
#include <ATen/native/xpu/sycl/MultiTensorApply.h>
#include <comm/SYCLContext.h>

enum class NormType { L1, L2 };

namespace at::native::xpu {
template <
    typename T,
    NormType norm_type,
    typename opmath_t,
    int depth = 1,
    int r_args_depth = 1,
    int res_arg_index = 0>
struct LpNormFunctor {
  static_assert(
      norm_type == NormType::L1 || norm_type == NormType::L2,
      "foreach_norm supports only L1 and L2 norm");
  template <typename TLA, typename TLW>
  void operator()(
      const int64_t chunk_size,
      TLA tlAddress,
      TLW tlWGMeta,
      sycl::nd_item<1> item_id,
      opmath_t* output_per_tensor,
      const int max_chunks_per_tensor) const {
    auto item_idx = item_id.get_local_id(0);
    auto item_range = item_id.get_local_range(0);
    auto group_idx = item_id.get_group(0);
    int tensor_loc = tlWGMeta[group_idx].wg_to_tensor;
    int chunk_idx = tlWGMeta[group_idx].wg_to_chunk;
    int64_t n = tlAddress[tensor_loc].numel_to_tensor;

    T* x = (T*)tlAddress[tensor_loc].addresses[0] + chunk_idx * chunk_size;
    n -= chunk_idx * chunk_size;

    opmath_t vals[kILP];
    T r_x[kILP];
    for (int i = 0; i < kILP; i++) {
      vals[i] = opmath_t(0.0f);
      r_x[i] = T(0.0f);
    }

    if (n % kILP == 0 && (chunk_size & kILP) == 0 && is_aligned(x)) {
      for (int64_t i_start = item_idx;
           i_start * kILP < n && i_start * kILP < chunk_size;
           i_start += item_range) {
        // load
        load_store(r_x, x, 0, i_start);
#pragma unroll
        for (int ii = 0; ii < kILP; ii++) {
          opmath_t next = static_cast<opmath_t>(r_x[ii]);
          vals[ii] += norm_type == NormType::L1
              ? static_cast<opmath_t>(std::fabs((opmath_t)next))
              : static_cast<opmath_t>(next * next);
        }
      }
    } else {
      for (int64_t i_start = 0; i_start < n && i_start < chunk_size;
           i_start += item_range * kILP) {
#pragma unroll
        for (int ii = 0; ii < kILP; ii++) {
          int i = i_start + item_idx + ii * item_range;
          if (i < n && i < chunk_size) {
            opmath_t next = static_cast<opmath_t>(x[i]);
            vals[ii] += norm_type == NormType::L1
                ? static_cast<opmath_t>(std::fabs((opmath_t)next))
                : static_cast<opmath_t>(next * next);
          }
        }
      }
    }

    auto val = opmath_t(0);
    for (int i = 0; i < kILP; i++) {
      val += vals[i];
    }

    auto sum_val = sycl::reduce_over_group(
        item_id.get_group(), val, sycl::plus<opmath_t>());

    if (item_idx == 0) {
      output_per_tensor[tensor_loc * max_chunks_per_tensor + chunk_idx] =
          sum_val;
    }
  }
};

template <typename out_t, NormType norm_type, typename opmath_t>
struct lpnormChunkReduceKernelFunctor {
  void operator()(sycl::nd_item<1> item_id) const {
    auto lid = item_id.get_local_linear_id();
    auto group_id = item_id.get_group(0);

    const opmath_t* output_this_tensor =
        output_per_tensor_ + group_id * max_chunks_per_tensor_;
    opmath_t val = 0;
    for (int i = lid; i < max_chunks_per_tensor_; i += wg_size_) {
      val += output_this_tensor[i];
    }
    auto sum_val = sycl::reduce_over_group(
        item_id.get_group(), val, sycl::plus<opmath_t>());
    if (lid == 0) {
      *(ret_per_tensor_[group_id]) =
          norm_type == NormType::L1 ? sum_val : std::sqrt((opmath_t)sum_val);
    }
  }
  lpnormChunkReduceKernelFunctor(
      const opmath_t* output_per_tensor,
      out_t** ret_per_tensor,
      int max_chunks_per_tensor,
      int wg_size)
      : output_per_tensor_(output_per_tensor),
        ret_per_tensor_(ret_per_tensor),
        max_chunks_per_tensor_(max_chunks_per_tensor),
        wg_size_(wg_size) {}

 private:
  const opmath_t* output_per_tensor_;
  out_t** ret_per_tensor_;
  int max_chunks_per_tensor_;
  int wg_size_;
};

template <typename out_t, NormType norm_type, typename out_opmath_t>
void launch_lpnorm_chunk_reduce_kernel(
    const out_opmath_t* output_per_tensor,
    out_t** ret_per_tensor,
    int wg_size,
    int max_chunks_per_tensor,
    int n_tensor) {
  lpnormChunkReduceKernelFunctor<out_t, norm_type, out_opmath_t> kfn(
      output_per_tensor, ret_per_tensor, max_chunks_per_tensor, wg_size);

  sycl_kernel_submit(
      sycl::range<1>(n_tensor * wg_size),
      sycl::range<1>(wg_size),
      getCurrentSYCLQueue(),
      kfn);
}

#define AT_DISPATCH_OUT_DTYPES(TYPE, NAME, ...)             \
  AT_DISPATCH_SWITCH(                                       \
      TYPE,                                                 \
      NAME,                                                 \
      AT_PRIVATE_CASE_TYPE_USING_HINT(                      \
          at::ScalarType::Double, out_t, __VA_ARGS__)       \
          AT_PRIVATE_CASE_TYPE_USING_HINT(                  \
              at::ScalarType::Float, out_t, __VA_ARGS__)    \
              AT_PRIVATE_CASE_TYPE_USING_HINT(              \
                  at::ScalarType::Half, out_t, __VA_ARGS__) \
                  AT_PRIVATE_CASE_TYPE_USING_HINT(          \
                      at::ScalarType::BFloat16, out_t, __VA_ARGS__))

template <class KernelClass>
void foreach_norn_kernel_config(
    TensorList tensors,
    TensorOptions output_per_tensor_option,
    int64_t& wg_size,
    int& max_chunks_per_tensor,
    Tensor& output_per_tensor) {
  const int ntensors = tensors.size();

  max_chunks_per_tensor = -1;
  wg_size = multi_tensor_apply_kernel_get_wg_size<KernelClass>();
  int64_t kChunkSize = multi_tensor_apply_kernel_get_chunk_size<KernelClass>();

  for (int t = 0; t < ntensors; t++) {
    int max_chunks_this_tensor =
        (tensors[t].numel() + kChunkSize - 1) / kChunkSize;
    if (max_chunks_this_tensor > max_chunks_per_tensor) {
      max_chunks_per_tensor = max_chunks_this_tensor;
    }
  }

  output_per_tensor = at::zeros(
      {static_cast<int64_t>(ntensors) * max_chunks_per_tensor},
      output_per_tensor_option);
}

std::vector<Tensor> foreach_norm_kernel(
    TensorList tensors,
    const Scalar& ord,
    double p,
    c10::optional<ScalarType> dtype) {
  const int ntensors = tensors.size();

  const ScalarType output_dtype = // tensors[0].scalar_type();
      dtype.has_value() ? dtype.value() : tensors[0].scalar_type();
  const auto options = tensors[0].options();
  auto output_per_tensor_option = options.dtype(toOpMathType(output_dtype));
  std::vector<at::Tensor> ret_per_tensor;
  ret_per_tensor.reserve(ntensors);
  const auto res_option = options.dtype(output_dtype);
  for (int i = 0; i < ntensors; i++) {
    ret_per_tensor.push_back(at::empty({}, res_option));
  }
  auto& q = getCurrentSYCLQueue();
  auto addressStorage =
      at::empty({(int)(sizeof(void*) * ntensors)}, options.dtype(at::kByte));
  auto metaAddress = static_cast<void**>(addressStorage.mutable_data_ptr());
  void** tensor_list_addresses = nullptr;

  auto tensor_list_addresses_dptr =
      at::xpu::HostAlloc(sizeof(void*) * ntensors);
  tensor_list_addresses = (void**)tensor_list_addresses_dptr.get();

  auto tensor_lists = std::vector<std::vector<Tensor>>{tensors.vec()};

  int64_t wg_size;
  int max_chunks_per_tensor;
  Tensor output_per_tensor;
  if (p == static_cast<double>(1)) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf,
        kBFloat16,
        tensor_lists[0][0].scalar_type(),
        "foreach_norm",
        [&]() {
          AT_DISPATCH_OUT_DTYPES(
              output_dtype, "foreach_norm_out_dtype_xpu", [&]() {
                using out_opmath_t = typename at::opmath_type<out_t>;
                using KernelClass = lpnormChunkReduceKernelFunctor<
                    out_t,
                    NormType::L1,
                    out_opmath_t>;
                foreach_norn_kernel_config<KernelClass>(
                    tensors,
                    output_per_tensor_option,
                    wg_size,
                    max_chunks_per_tensor,
                    output_per_tensor);

                // sum temp val for each chunk
                multi_tensor_apply<1>(
                    tensor_lists,
                    LpNormFunctor<scalar_t, NormType::L1, out_opmath_t>(),
                    output_per_tensor.mutable_data_ptr<out_opmath_t>(),
                    max_chunks_per_tensor);
                for (int i = 0; i < ntensors; i++) {
                  tensor_list_addresses[i] =
                      ret_per_tensor[i].mutable_data_ptr<out_t>();
                }
                q.memcpy(
                    (void*)metaAddress,
                    (void*)tensor_list_addresses,
                    sizeof(void*) * ntensors);

                at::xpu::CachingHostAllocator_recordEvent(
                    (void*)tensor_list_addresses,
                    tensor_list_addresses_dptr.get_context(),
                    at::xpu::getCurrentXPUStream());
                launch_lpnorm_chunk_reduce_kernel<
                    out_t,
                    NormType::L1,
                    out_opmath_t>(
                    output_per_tensor.mutable_data_ptr<out_opmath_t>(),
                    (out_t**)(metaAddress),
                    wg_size,
                    max_chunks_per_tensor,
                    ntensors);
              });
        });
  } else if (p == static_cast<double>(2)) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf,
        kBFloat16,
        tensor_lists[0][0].scalar_type(),
        "foreach_norm",
        [&]() {
          AT_DISPATCH_OUT_DTYPES(
              output_dtype, "foreach_norm_out_dtype_xpu", [&]() {
                using out_opmath_t = typename at::opmath_type<out_t>;
                using KernelClass = lpnormChunkReduceKernelFunctor<
                    out_t,
                    NormType::L2,
                    out_opmath_t>;
                foreach_norn_kernel_config<KernelClass>(
                    tensors,
                    output_per_tensor_option,
                    wg_size,
                    max_chunks_per_tensor,
                    output_per_tensor);

                multi_tensor_apply<1>(
                    tensor_lists,
                    LpNormFunctor<scalar_t, NormType::L2, out_opmath_t>(),
                    output_per_tensor.mutable_data_ptr<out_opmath_t>(),
                    max_chunks_per_tensor);
                for (int i = 0; i < ntensors; i++) {
                  tensor_list_addresses[i] =
                      ret_per_tensor[i].mutable_data_ptr<out_t>();
                }
                q.memcpy(
                    (void*)metaAddress,
                    (void*)tensor_list_addresses,
                    sizeof(void*) * ntensors);

                at::xpu::CachingHostAllocator_recordEvent(
                    (void*)tensor_list_addresses,
                    tensor_list_addresses_dptr.get_context(),
                    at::xpu::getCurrentXPUStream());
                launch_lpnorm_chunk_reduce_kernel<
                    out_t,
                    NormType::L2,
                    out_opmath_t>(
                    output_per_tensor.mutable_data_ptr<out_opmath_t>(),
                    (out_t**)(metaAddress),
                    wg_size,
                    max_chunks_per_tensor,
                    ntensors);
              });
        });
  } else {
    TORCH_CHECK(false, "foreach_norm fast path got unexpected ord value: ", p);
  }

  std::vector<Tensor> result;
  result.reserve(ntensors);
  for (const auto& i : c10::irange(ntensors)) {
    result.emplace_back(ret_per_tensor[i]);
  }
  return result;
}

} // namespace at::native::xpu
