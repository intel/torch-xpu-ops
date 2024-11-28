#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/native/ForeachUtils.h>

#include <ATen/native/xpu/sycl/ForeachFunctors.h>
#include <ATen/native/xpu/sycl/MultiTensorApply.h>
#include <ATen/xpu/EmptyTensor.h>
#include <comm/SYCLContext.h>

#include <ATen/native/xpu/sycl/ForeachReduceKernels.h>

namespace at::native::xpu {

enum class NormType { L1, L2, LInf };
#define SIMD16 16
#define SIMD32 32

template <
    typename T,
    NormType norm_type,
    typename opmath_t,
    int SIMD,
    int depth = 1,
    int r_args_depth = 1,
    int res_arg_index = 0>
struct LpNormFunctor : public __SYCL_KER_CONFIG_CONVENTION__ {
  template <typename TLA, typename TLW>
  [[intel::reqd_sub_group_size(SIMD)]] void operator()(
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
          if constexpr (norm_type == NormType::LInf) {
            vals[ii] = max_impl(vals[ii], std::fabs((opmath_t)next));
          } else {
            vals[ii] += norm_type == NormType::L1
                ? static_cast<opmath_t>(std::fabs((opmath_t)next))
                : static_cast<opmath_t>(next * next);
          }
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
            if constexpr (norm_type == NormType::LInf) {
              vals[ii] = max_impl(vals[ii], ::abs(std::fabs((opmath_t)next)));
            } else {
              vals[ii] += norm_type == NormType::L1
                  ? static_cast<opmath_t>(std::fabs((opmath_t)next))
                  : static_cast<opmath_t>(next * next);
            }
          }
        }
      }
    }

    auto val = opmath_t(0);
    for (int i = 0; i < kILP; i++) {
      if constexpr (norm_type == NormType::LInf) {
        val = max_impl(val, vals[i]);
      } else {
        val += vals[i];
      }
    }

    auto sum_val = norm_type == NormType::L1 || norm_type == NormType::L2
        ? GroupReduceSumWithoutBroadcast<opmath_t, SIMD>(item_id, val, shared_)
        : GroupReduceMaxWithoutBroadcast<opmath_t, SIMD>(item_id, val, shared_);

    if (item_idx == 0) {
      output_per_tensor[tensor_loc * max_chunks_per_tensor + chunk_idx] =
          sum_val;
    }
  }
  void sycl_ker_config_convention(sycl::handler& cgh) {
    shared_ =
        sycl_local_acc_t<opmath_t>(get_group_reduce_group_size(SIMD), cgh);
  }

 private:
  sycl_local_acc_t<opmath_t> shared_;
};

template <typename out_t, NormType norm_type, typename opmath_t, int SIMD>
struct lpnormChunkReduceKernelFunctor : public __SYCL_KER_CONFIG_CONVENTION__ {
  [[intel::reqd_sub_group_size(SIMD)]] void operator()(
      sycl::nd_item<1> item_id) const {
    auto lid = item_id.get_local_linear_id();
    auto group_id = item_id.get_group(0);

    const opmath_t* output_this_tensor =
        output_per_tensor_ + group_id * max_chunks_per_tensor_;
    opmath_t val = 0;
    for (int i = lid; i < max_chunks_per_tensor_; i += wg_size_) {
      if constexpr (norm_type == NormType::LInf) {
        val = max_impl(val, output_this_tensor[i]);
      } else {
        val += output_this_tensor[i];
      }
    }
    auto sum_val = norm_type == NormType::L1 || norm_type == NormType::L2
        ? GroupReduceSumWithoutBroadcast<opmath_t, SIMD>(item_id, val, shared_)
        : GroupReduceMaxWithoutBroadcast<opmath_t, SIMD>(item_id, val, shared_);
    if (lid == 0) {
      *(ret_per_tensor_[group_id]) =
          norm_type == NormType::L1 || norm_type == NormType::LInf
          ? sum_val
          : std::sqrt((opmath_t)sum_val);
    }
  }
  void sycl_ker_config_convention(sycl::handler& cgh) {
    shared_ =
        sycl_local_acc_t<opmath_t>(get_group_reduce_group_size(SIMD), cgh);
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
  sycl_local_acc_t<opmath_t> shared_;
};

template <typename out_t, NormType norm_type, typename out_opmath_t, int SIMD>
void launch_lpnorm_chunk_reduce_kernel(
    const out_opmath_t* output_per_tensor,
    out_t** ret_per_tensor,
    int wg_size,
    int max_chunks_per_tensor,
    int n_tensor) {
  lpnormChunkReduceKernelFunctor<out_t, norm_type, out_opmath_t, SIMD> kfn(
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

void foreach_norn_kernel_config(
    TensorList tensors,
    TensorOptions output_per_tensor_option,
    int64_t simd,
    int64_t& wg_size,
    int& max_chunks_per_tensor,
    Tensor& output_per_tensor) {
  const int ntensors = tensors.size();

  max_chunks_per_tensor = -1;
  wg_size = multi_tensor_apply_kernel_get_wg_size(simd);
  int64_t kChunkSize = multi_tensor_apply_kernel_get_chunk_size(simd);

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
  int64_t simd = syclMaxSubGroupSize();
  foreach_norn_kernel_config(
      tensors,
      output_per_tensor_option,
      simd,
      wg_size,
      max_chunks_per_tensor,
      output_per_tensor);
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
                // sum temp val for each chunk
                if (simd == SIMD32) {
                  multi_tensor_apply<1>(
                      tensor_lists,
                      LpNormFunctor<
                          scalar_t,
                          NormType::L1,
                          out_opmath_t,
                          SIMD32>(),
                      output_per_tensor.mutable_data_ptr<out_opmath_t>(),
                      max_chunks_per_tensor);
                } else {
                  multi_tensor_apply<1>(
                      tensor_lists,
                      LpNormFunctor<
                          scalar_t,
                          NormType::L1,
                          out_opmath_t,
                          SIMD16>(),
                      output_per_tensor.mutable_data_ptr<out_opmath_t>(),
                      max_chunks_per_tensor);
                }
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
                if (simd == SIMD32) {
                  launch_lpnorm_chunk_reduce_kernel<
                      out_t,
                      NormType::L1,
                      out_opmath_t,
                      SIMD32>(
                      output_per_tensor.mutable_data_ptr<out_opmath_t>(),
                      (out_t**)(metaAddress),
                      wg_size,
                      max_chunks_per_tensor,
                      ntensors);
                } else {
                  launch_lpnorm_chunk_reduce_kernel<
                      out_t,
                      NormType::L1,
                      out_opmath_t,
                      SIMD16>(
                      output_per_tensor.mutable_data_ptr<out_opmath_t>(),
                      (out_t**)(metaAddress),
                      wg_size,
                      max_chunks_per_tensor,
                      ntensors);
                }
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
                if (simd == SIMD32) {
                  multi_tensor_apply<1>(
                      tensor_lists,
                      LpNormFunctor<
                          scalar_t,
                          NormType::L2,
                          out_opmath_t,
                          SIMD32>(),
                      output_per_tensor.mutable_data_ptr<out_opmath_t>(),
                      max_chunks_per_tensor);
                } else {
                  multi_tensor_apply<1>(
                      tensor_lists,
                      LpNormFunctor<
                          scalar_t,
                          NormType::L2,
                          out_opmath_t,
                          SIMD16>(),
                      output_per_tensor.mutable_data_ptr<out_opmath_t>(),
                      max_chunks_per_tensor);
                }
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
                if (simd == SIMD32) {
                  launch_lpnorm_chunk_reduce_kernel<
                      out_t,
                      NormType::L2,
                      out_opmath_t,
                      SIMD32>(
                      output_per_tensor.mutable_data_ptr<out_opmath_t>(),
                      (out_t**)(metaAddress),
                      wg_size,
                      max_chunks_per_tensor,
                      ntensors);
                } else {
                  launch_lpnorm_chunk_reduce_kernel<
                      out_t,
                      NormType::L2,
                      out_opmath_t,
                      SIMD16>(
                      output_per_tensor.mutable_data_ptr<out_opmath_t>(),
                      (out_t**)(metaAddress),
                      wg_size,
                      max_chunks_per_tensor,
                      ntensors);
                }
              });
        });
  } else if (p == std::numeric_limits<double>::infinity()) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf,
        kBFloat16,
        tensor_lists[0][0].scalar_type(),
        "foreach_norm",
        [&]() {
          AT_DISPATCH_OUT_DTYPES(
              output_dtype, "foreach_norm_out_dtype_xpu", [&]() {
                using out_opmath_t = typename at::opmath_type<out_t>;
                if (simd == SIMD32) {
                  multi_tensor_apply<1>(
                      tensor_lists,
                      LpNormFunctor<
                          scalar_t,
                          NormType::LInf,
                          out_opmath_t,
                          SIMD32>(),
                      output_per_tensor.mutable_data_ptr<out_opmath_t>(),
                      max_chunks_per_tensor);
                } else {
                  multi_tensor_apply<1>(
                      tensor_lists,
                      LpNormFunctor<
                          scalar_t,
                          NormType::LInf,
                          out_opmath_t,
                          SIMD16>(),
                      output_per_tensor.mutable_data_ptr<out_opmath_t>(),
                      max_chunks_per_tensor);
                }
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
                if (simd == SIMD32) {
                  launch_lpnorm_chunk_reduce_kernel<
                      out_t,
                      NormType::LInf,
                      out_opmath_t,
                      SIMD32>(
                      output_per_tensor.mutable_data_ptr<out_opmath_t>(),
                      (out_t**)(metaAddress),
                      wg_size,
                      max_chunks_per_tensor,
                      ntensors);
                } else {
                  launch_lpnorm_chunk_reduce_kernel<
                      out_t,
                      NormType::LInf,
                      out_opmath_t,
                      SIMD16>(
                      output_per_tensor.mutable_data_ptr<out_opmath_t>(),
                      (out_t**)(metaAddress),
                      wg_size,
                      max_chunks_per_tensor,
                      ntensors);
                }
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

template <typename T, int SIMD>
struct LpMaxFunctor : public __SYCL_KER_CONFIG_CONVENTION__ {
  template <typename TLA, typename TLW>
  [[intel::reqd_sub_group_size(SIMD)]] void operator()(
      int64_t chunk_size,
      TLA tlAddressMeta,
      TLW tlWGMeta,
      sycl::nd_item<1> item,
      T* output_per_tensor_ptr,
      const int max_chunks_per_tensor) const {
    auto workgroup_id = item.get_group(0);
    auto item_id = item.get_local_id(0);
    auto local_range = item.get_local_range(0);

    const auto tensor_loc = tlWGMeta[workgroup_id].wg_to_tensor;
    const auto chunk_idx = tlWGMeta[workgroup_id].wg_to_chunk;
    auto n = tlAddressMeta[tensor_loc].numel_to_tensor;

    T* x = (T*)tlAddressMeta[tensor_loc].addresses[0];
    x += chunk_idx * chunk_size;
    n -= chunk_idx * chunk_size;

    T vals[kILP];
    T r_x[kILP];
    for (int64_t i = 0; i < kILP; i++) {
      vals[i] = T(std::numeric_limits<T>::lowest());
      r_x[i] = T(std::numeric_limits<T>::lowest());
    }

    if (n % kILP == 0 && (chunk_size & kILP) == 0 && is_aligned(x)) {
      for (int64_t i_start = item_id;
           i_start * kILP < n && i_start * kILP < chunk_size;
           i_start += local_range) {
        // load
        load_store(r_x, x, 0, i_start);
#pragma unroll
        for (int ii = 0; ii < kILP; ii++) {
          vals[ii] = max_impl(vals[ii], r_x[ii]);
        }
      }
    } else {
      for (int64_t i_start = 0; i_start < n && i_start < chunk_size;
           i_start += local_range * kILP) {
#pragma unroll
        for (int ii = 0; ii < kILP; ii++) {
          int i = i_start + item_id + ii * local_range;
          if (i < n && i < chunk_size) {
            vals[ii] = max_impl(vals[ii], x[i]);
          }
        }
      }
    }

    auto val = T(std::numeric_limits<T>::lowest());
    for (int i = 0; i < kILP; i++) {
      val = max_impl(val, vals[i]);
    }
    auto final_val =
        GroupReduceMaxWithoutBroadcast<T, SIMD>(item, val, shared_);

    if (item_id == 0) {
      output_per_tensor_ptr[tensor_loc * max_chunks_per_tensor + chunk_idx] =
          final_val;
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    shared_ = sycl_local_acc_t<T>(SIMD, cgh);
  }

 private:
  sycl_local_acc_t<T> shared_;
};

template <typename T, int SIMD>
struct LpmaxChunkReduceKernelFunctor : public __SYCL_KER_CONFIG_CONVENTION__ {
  [[intel::reqd_sub_group_size(SIMD)]] void operator()(
      sycl::nd_item<1> item_id) const {
    auto local_range = item_id.get_local_range(0);
    auto lid = item_id.get_local_linear_id();
    auto group_id = item_id.get_group(0);

    const T* output_this_tensor =
        output_per_tensor_ + group_id * max_chunks_per_tensor_;
    int chunks_this_tensor = chunks_per_tensor_[group_id];
    T val = std::numeric_limits<T>::lowest();
    for (int i = lid; i < chunks_this_tensor; i += local_range) {
      val = max_impl(val, output_this_tensor[i]);
    }
    T final_value =
        GroupReduceMaxWithoutBroadcast<T, SIMD>(item_id, val, shared_);
    if (lid == 0) {
      *(ret_per_tensor_[group_id]) = final_value;
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    shared_ = sycl_local_acc_t<T>(SIMD, cgh);
  }

  LpmaxChunkReduceKernelFunctor(
      const T* output_per_tensor,
      T** ret_per_tensor,
      int* chunks_per_tensor,
      int max_chunks_per_tensor)
      : output_per_tensor_(output_per_tensor),
        ret_per_tensor_(ret_per_tensor),
        chunks_per_tensor_(chunks_per_tensor),
        max_chunks_per_tensor_(max_chunks_per_tensor) {}

 private:
  const T* output_per_tensor_;
  T** ret_per_tensor_;
  int* chunks_per_tensor_;
  int max_chunks_per_tensor_;
  sycl_local_acc_t<T> shared_;
};

template <typename T, int SIMD>
void launch_lpmax_chunk_reduce_kernel(
    const T* output_per_tensor,
    T** ret_per_tensor,
    int* chunks_per_tensor,
    int max_chunks_per_tensor,
    int n_tensor) {
  int wg_size = multi_tensor_apply_kernel_get_wg_size(SIMD);
  LpmaxChunkReduceKernelFunctor<T, SIMD> kfn(
      output_per_tensor,
      ret_per_tensor,
      chunks_per_tensor,
      max_chunks_per_tensor);

  sycl_kernel_submit(
      sycl::range<1>(n_tensor * wg_size),
      sycl::range<1>(wg_size),
      getCurrentSYCLQueue(),
      kfn);
}

std::vector<Tensor> foreach_max_kernel(TensorList tensors) {
  const size_t ntensors = tensors.size();
  const auto options = tensors[0].options();

  auto& q = getCurrentSYCLQueue();
  // Store output address for each tensor
  auto addressStorage =
      at::empty({(int)(sizeof(void*) * ntensors)}, options.dtype(at::kByte));
  auto metaAddress = static_cast<void**>(addressStorage.mutable_data_ptr());
  void** tensor_list_addresses = nullptr;
  auto tensor_list_addresses_dptr =
      at::xpu::HostAlloc(sizeof(void*) * ntensors);
  tensor_list_addresses = (void**)tensor_list_addresses_dptr.get();

  // Store thunks count for each tensor
  auto countsStorage =
      at::empty({(int)(sizeof(int) * ntensors)}, options.dtype(at::kByte));
  auto metaCounts = static_cast<int*>(countsStorage.mutable_data_ptr());
  int* thunk_counts = nullptr;
  auto thunk_counts_dptr = at::xpu::HostAlloc(sizeof(int) * ntensors);
  thunk_counts = (int*)thunk_counts_dptr.get();

  int max_chunks_per_tensor = -1;
  int64_t simd = syclMaxSubGroupSize();
  int64_t kChunkSize = multi_tensor_apply_kernel_get_chunk_size(simd);
  for (const auto t : c10::irange(ntensors)) {
    int max_chunks_this_tensor =
        (tensors[t].numel() + kChunkSize - 1) / kChunkSize;
    thunk_counts[t] = max_chunks_this_tensor;
    if (max_chunks_this_tensor > max_chunks_per_tensor) {
      max_chunks_per_tensor = max_chunks_this_tensor;
    }
  }
  auto output_per_tensor = at::zeros(
      {static_cast<int64_t>(ntensors) * max_chunks_per_tensor}, options);

  std::vector<at::Tensor> vec_res;
  vec_res.reserve(ntensors);
  for (const auto i : c10::irange(ntensors)) {
    vec_res.push_back(at::detail::empty_xpu(
        {},
        optTypeMetaToScalarType(options.dtype_opt()),
        options.layout_opt(),
        options.device_opt(),
        options.pinned_memory_opt(),
        options.memory_format_opt()));
  }

  auto tensor_lists = std::vector<std::vector<Tensor>>{tensors.vec()};
  AT_DISPATCH_ALL_TYPES_AND3(
      kHalf,
      kBFloat16,
      kBool,
      tensor_lists[0][0].scalar_type(),
      "foreach_tensor_max_xpu_scalar_type",
      [&]() {
        if (simd == SIMD32) {
          multi_tensor_apply<1>(
              tensor_lists,
              LpMaxFunctor<scalar_t, SIMD32>(),
              output_per_tensor.mutable_data_ptr<scalar_t>(),
              max_chunks_per_tensor);
        } else if (simd == SIMD16) {
          multi_tensor_apply<1>(
              tensor_lists,
              LpMaxFunctor<scalar_t, SIMD16>(),
              output_per_tensor.mutable_data_ptr<scalar_t>(),
              max_chunks_per_tensor);
        } else {
          TORCH_CHECK(
              false,
              "foreach_max_xpu_kernel does not support SIMD width: ",
              simd);
        }

        for (int i = 0; i < ntensors; i++) {
          tensor_list_addresses[i] = vec_res[i].mutable_data_ptr<scalar_t>();
        }
        q.memcpy(
            (void*)metaAddress,
            (void*)tensor_list_addresses,
            sizeof(void*) * ntensors);
        at::xpu::CachingHostAllocator_recordEvent(
            (void*)tensor_list_addresses,
            tensor_list_addresses_dptr.get_context(),
            at::xpu::getCurrentXPUStream());
        q.memcpy(
            (void*)metaCounts, (void*)thunk_counts, sizeof(int) * ntensors);
        at::xpu::CachingHostAllocator_recordEvent(
            (void*)thunk_counts,
            thunk_counts_dptr.get_context(),
            at::xpu::getCurrentXPUStream());
        if (simd == SIMD32) {
          launch_lpmax_chunk_reduce_kernel<scalar_t, SIMD32>(
              output_per_tensor.mutable_data_ptr<scalar_t>(),
              (scalar_t**)(metaAddress),
              (int*)(metaCounts),
              max_chunks_per_tensor,
              ntensors);
        } else {
          launch_lpmax_chunk_reduce_kernel<scalar_t, SIMD16>(
              output_per_tensor.mutable_data_ptr<scalar_t>(),
              (scalar_t**)(metaAddress),
              (int*)(metaCounts),
              max_chunks_per_tensor,
              ntensors);
        }
      });

  // correctly assign values to only non-empty slots, as the empty slots should
  // get skipped
  std::vector<Tensor> result;
  result.reserve(ntensors);
  int i = 0;
  for (const auto& t : tensors) {
    if (t.numel() != 0) {
      result.emplace_back(vec_res[i]);
      i++;
    } else {
      result.emplace_back(at::detail::empty_xpu(
          {},
          optTypeMetaToScalarType(options.dtype_opt()),
          options.layout_opt(),
          options.device_opt(),
          options.pinned_memory_opt(),
          options.memory_format_opt()));
    }
  }
  return result;
}

} // namespace at::native::xpu
