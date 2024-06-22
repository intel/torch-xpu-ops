#pragma once
#include <ATen/OpMathType.h>
#include <ATen/native/ForeachUtils.h>

#include <ATen/native/xpu/sycl/MultiTensorApply.h>
#include <ATen/native/xpu/sycl/Pow.h>

namespace at::native::xpu {
namespace {

inline void increment_version(TensorList tensors) {
  for (const auto& t : tensors) {
    t.unsafeGetTensorImpl()->bump_version();
  }
}

template <int depth, typename T, typename TL>
bool init_args(
    T** args,
    TL tl,
    int64_t chunk_idx,
    int64_t chunk_size,
    int tensor_loc) {
  bool aligned = true;
  for (int i = 0; i < depth; ++i) {
    args[i] = (T*)tl[tensor_loc].addresses[i];
    args[i] += chunk_idx * chunk_size;
    if (!is_aligned(args[i])) {
      aligned = false;
    }
  }
  return aligned;
}

template <int depth, typename T>
void load_args(
    T r_args[][kILP],
    T** args,
    int64_t i,
    int64_t chunk_size,
    int64_t n,
    int64_t item_idx,
    int64_t item_range) {
#pragma unroll
  for (int ii = 0; ii < kILP; ++ii) {
    int64_t i_start = i + item_idx + ii * item_range;
#pragma unroll
    for (int index = 0; index < depth; ++index) {
      r_args[index][ii] = 0;
      if (i_start < n && i_start < chunk_size) {
        r_args[index][ii] = args[index][i_start];
      }
    }
  }
}

template <typename T>
void store_args(
    T* dst,
    T* src,
    int64_t i_start,
    int64_t chunk_size,
    int64_t n,
    int64_t item_idx,
    int64_t group_range) {
#pragma unroll
  for (int ii = 0; ii < kILP; ++ii) {
    int64_t i = i_start + item_idx + group_range * ii;
    if (i < n && i < chunk_size) {
      dst[i] = src[ii];
    }
  }
}
} // namespace

namespace foreach_internal {
// Implement std::isnan<IntegralType> for MSVC.
// taken from
// https://github.com/pytorch/pytorch/blob/a67691e50803b31e98043d22ae29da4c0135ab3c/aten/src/ATen/native/ReduceOps.cpp#L239-L256
namespace {
#ifdef _MSC_VER
template <typename T>
inline typename std::enable_if<std::is_integral<T>::value, bool>::type isnan_(
    T x) {
  return false;
}
template <typename T>
inline typename std::enable_if<!std::is_integral<T>::value, bool>::type isnan_(
    T x) {
  return std::isnan(x);
}
#else
template <typename T>
inline bool isnan_(T x) {
  return std::isnan(x);
}
#endif
} // namespace

template <typename T>
struct minimum {
  T operator()(const T& a, const T& b) {
    return (isnan_(a) || a < b) ? a : b;
  }
};
template <typename T>
struct maximum {
  T operator()(const T& a, const T& b) {
    return (isnan_(a) || a > b) ? a : b;
  }
};
} // namespace foreach_internal

template <typename T, int depth, int r_args_depth, int res_arg_index>
struct UnaryOpFunctor {
  using opmath_t = at::opmath_type<T>;
  template <typename TLA, typename TLW, typename Op>
  void operator()(
      const int64_t chunk_size,
      TLA tlAddress,
      TLW tlWGMeta,
      sycl::nd_item<1> item_id,
      Op op) const {
    auto item_idx = item_id.get_local_id(0);
    auto item_range = item_id.get_local_range(0);
    auto group_idx = item_id.get_group(0);
    int tensor_loc = tlWGMeta[group_idx].wg_to_tensor;
    int chunk_idx = tlWGMeta[group_idx].wg_to_chunk;
    int64_t n = tlAddress[tensor_loc].numel_to_tensor;
    T* args[depth];
    bool all_aligned =
        init_args<depth>(args, tlAddress, chunk_idx, chunk_size, tensor_loc);
    n -= chunk_idx * chunk_size;
    T r_args[r_args_depth][kILP];
    // vec path
    if (n % kILP == 0 && chunk_size % kILP == 0 && all_aligned) {
      for (int64_t i = item_idx; i * kILP < n && i * kILP < chunk_size;
           i += item_range) {
        load_store(r_args[0], args[0], 0, i);
#pragma unroll
        for (int ii = 0; ii < kILP; ++ii) {
          r_args[0][ii] =
              static_cast<T>(op(static_cast<opmath_t>(r_args[0][ii])));
        }
        load_store(args[res_arg_index], r_args[0], i, 0);
      }
      // non-vec path
    } else {
      for (int64_t i = 0; i < n && i < chunk_size; i += item_range * kILP) {
        load_args<r_args_depth>(
            r_args, args, i, chunk_size, n, item_idx, item_range);
#pragma unroll
        for (int ii = 0; ii < kILP; ++ii) {
          r_args[0][ii] =
              static_cast<T>(op(static_cast<opmath_t>(r_args[0][ii])));
        }
        store_args(
            args[res_arg_index],
            r_args[0],
            i,
            chunk_size,
            n,
            item_idx,
            item_range);
      }
    }
  }
};

template <typename T, int depth, int r_args_depth, int res_arg_index>
struct ZeroFunctor {
  using opmath_t = at::opmath_type<T>;
  template <typename TLA, typename TLW>
  void operator()(
      const int64_t chunk_size,
      TLA tlAddress,
      TLW tlWGMeta,
      sycl::nd_item<1> item_id) const {
    auto item_idx = item_id.get_local_id(0);
    auto item_range = item_id.get_local_range(0);
    auto group_idx = item_id.get_group(0);
    int tensor_loc = tlWGMeta[group_idx].wg_to_tensor;
    int chunk_idx = tlWGMeta[group_idx].wg_to_chunk;
    int64_t n = tlAddress[tensor_loc].numel_to_tensor;
    T* args[depth];
    bool all_aligned =
        init_args<depth>(args, tlAddress, chunk_idx, chunk_size, tensor_loc);
    n -= chunk_idx * chunk_size;
    T r_args[r_args_depth][kILP];
    // vec path
    if (n % kILP == 0 && chunk_size % kILP == 0 && all_aligned) {
      for (int64_t i = item_idx; i * kILP < n && i * kILP < chunk_size;
           i += item_range) {
#pragma unroll
        for (int ii = 0; ii < kILP; ++ii) {
          r_args[0][ii] = 0;
        }
        load_store(args[res_arg_index], r_args[0], i, 0);
      }
      // non-vec path
    } else {
      for (int64_t i = 0; i < n && i < chunk_size; i += item_range * kILP) {
#pragma unroll
        for (int ii = 0; ii < kILP; ++ii) {
          r_args[0][ii] = 0;
        }
        store_args(
            args[res_arg_index],
            r_args[0],
            i,
            chunk_size,
            n,
            item_idx,
            item_range);
      }
    }
  }
};

template <typename T, int depth, int r_args_depth, int res_arg_index>
struct PointwiseOpScalarFunctor {
  using opmath_t = at::opmath_type<T>;
  template <typename Op, typename TLA, typename TLW>
  void operator()(
      const int chunk_size,
      TLA tlAddress,
      TLW tlWGMeta,
      sycl::nd_item<1> item_id,
      Op op,
      opmath_t scalar) const {
    auto item_idx = item_id.get_local_id(0);
    auto item_range = item_id.get_local_range(0);
    auto group_idx = item_id.get_group(0);
    int tensor_loc = tlWGMeta[group_idx].wg_to_tensor;
    int chunk_idx = tlWGMeta[group_idx].wg_to_chunk;
    int64_t n = tlAddress[tensor_loc].numel_to_tensor;
    T* args[depth];
    bool all_aligned =
        init_args<depth>(args, tlAddress, chunk_idx, chunk_size, tensor_loc);
    n -= chunk_idx * chunk_size;
    T r_args[r_args_depth][kILP];

    if (n % kILP == 0 && chunk_size % kILP == 0 && all_aligned) {
      for (int i_start = item_idx;
           i_start * kILP < n && i_start * kILP < chunk_size;
           i_start += item_range) {
        load_store(r_args[0], args[0], 0, i_start);
        load_store(r_args[1], args[1], 0, i_start);
        load_store(r_args[2], args[2], 0, i_start);
#pragma unroll
        for (int ii = 0; ii < kILP; ii++) {
          r_args[0][ii] = static_cast<T>(
              static_cast<opmath_t>(r_args[0][ii]) +
              scalar *
                  op(static_cast<opmath_t>(r_args[1][ii]),
                     static_cast<opmath_t>(r_args[2][ii])));
        }
        load_store(args[res_arg_index], r_args[0], i_start, 0);
      }
    } else {
      for (int i_start = 0; i_start < n && i_start < chunk_size;
           i_start += item_range * kILP) {
        load_args<3>(
            r_args, args, i_start, chunk_size, n, item_idx, item_range);
#pragma unroll
        for (int ii = 0; ii < kILP; ii++) {
          r_args[0][ii] = static_cast<T>(
              static_cast<opmath_t>(r_args[0][ii]) +
              scalar *
                  op(static_cast<opmath_t>(r_args[1][ii]),
                     static_cast<opmath_t>(r_args[2][ii])));
        }
        store_args(
            args[res_arg_index],
            r_args[0],
            i_start,
            chunk_size,
            n,
            item_idx,
            item_range);
      }
    }
  }
};

template <typename T, int depth, int r_args_depth, int res_arg_index>
struct PointwiseOpScalarListFunctor {
  using opmath_t = at::opmath_type<T>;
  template <typename Op, typename TLA, typename TLW>
  void operator()(
      const int chunk_size,
      TLA tlAddress,
      TLW tlWGMeta,
      sycl::nd_item<1> item_id,
      Op op) const {
    auto item_idx = item_id.get_local_id(0);
    auto item_range = item_id.get_local_range(0);
    auto group_idx = item_id.get_group(0);
    int tensor_loc = tlWGMeta[group_idx].wg_to_tensor;
    int chunk_idx = tlWGMeta[group_idx].wg_to_chunk;
    int64_t n = tlAddress[tensor_loc].numel_to_tensor;
    opmath_t scalar = tlAddress[tensor_loc].scalar_vals;
    T* args[depth];
    bool all_aligned =
        init_args<depth>(args, tlAddress, chunk_idx, chunk_size, tensor_loc);
    n -= chunk_idx * chunk_size;
    T r_args[r_args_depth][kILP];

    if (n % kILP == 0 && chunk_size % kILP == 0 && all_aligned) {
      for (int i_start = item_idx;
           i_start * kILP < n && i_start * kILP < chunk_size;
           i_start += item_range) {
        load_store(r_args[0], args[0], 0, i_start);
        load_store(r_args[1], args[1], 0, i_start);
        load_store(r_args[2], args[2], 0, i_start);
#pragma unroll
        for (int ii = 0; ii < kILP; ii++) {
          r_args[0][ii] = static_cast<T>(
              static_cast<opmath_t>(r_args[0][ii]) +
              scalar *
                  op(static_cast<opmath_t>(r_args[1][ii]),
                     static_cast<opmath_t>(r_args[2][ii])));
        }
        load_store(args[res_arg_index], r_args[0], i_start, 0);
      }
    } else {
      for (int i_start = 0; i_start < n && i_start < chunk_size;
           i_start += item_range * kILP) {
        load_args<3>(
            r_args, args, i_start, chunk_size, n, item_idx, item_range);
#pragma unroll
        for (int ii = 0; ii < kILP; ii++) {
          r_args[0][ii] = static_cast<T>(
              static_cast<opmath_t>(r_args[0][ii]) +
              scalar *
                  op(static_cast<opmath_t>(r_args[1][ii]),
                     static_cast<opmath_t>(r_args[2][ii])));
        }
        store_args(
            args[res_arg_index],
            r_args[0],
            i_start,
            chunk_size,
            n,
            item_idx,
            item_range);
      }
    }
  }
};

template <typename T, int depth>
struct PointwiseOpListFunctor {
  using opmath_t = at::opmath_type<T>;
  template <typename TLA, typename TLW, typename Op>
  void operator()(
      const int64_t chunk_size,
      TLA tlAddress,
      TLW tlWGMeta,
      sycl::nd_item<1> item_id,
      Op op) const {
    auto item_idx = item_id.get_local_id(0);
    auto item_range = item_id.get_local_range(0);
    auto group_idx = item_id.get_group(0);
    int tensor_loc = tlWGMeta[group_idx].wg_to_tensor;
    int chunk_idx = tlWGMeta[group_idx].wg_to_chunk;
    int64_t n = tlAddress[tensor_loc].numel_to_tensor;
    T* args[depth];
    bool all_aligned =
        init_args<depth>(args, tlAddress, chunk_idx, chunk_size, tensor_loc);
    n -= chunk_idx * chunk_size;
    T r_args[depth - 1][kILP];

    if (n % kILP == 0 && chunk_size % kILP == 0 && all_aligned) {
      for (int64_t i = item_idx; i * kILP < n && i * kILP < chunk_size;
           i += item_range) {
        load_store(r_args[0], args[0], 0, i);
        load_store(r_args[1], args[1], 0, i);
#pragma unroll
        for (int ii = 0; ii < kILP; ++ii) {
          r_args[0][ii] = static_cast<T>(
              op(static_cast<opmath_t>(r_args[0][ii]),
                 static_cast<opmath_t>(r_args[1][ii])));
        }
        load_store(args[2], r_args[0], i, 0);
      } // without simd
    } else {
      for (int64_t i = 0; i < n && i < chunk_size; i += item_range * kILP) {
        load_args<depth - 1>(
            r_args, args, i, chunk_size, n, item_idx, item_range);
#pragma unroll
        for (int ii = 0; ii < kILP; ++ii) {
          r_args[0][ii] = static_cast<T>(
              op(static_cast<opmath_t>(r_args[0][ii]),
                 static_cast<opmath_t>(r_args[1][ii])));
        }
        store_args(args[2], r_args[0], i, chunk_size, n, item_idx, item_range);
      }
    }
  }
};

template <
    int r_args_depth,
    int res_arg_index,
    typename Op,
    typename T,
    typename opmath_t>
void binary_op_scalar(
    T r_args[][kILP],
    T** args,
    opmath_t scalar,
    int n,
    int chunk_size,
    bool all_aligned,
    Op op,
    size_t item_range,
    size_t item_idx) {
  // to make things simple, we put aligned case in a different code path
  if (n % kILP == 0 && chunk_size % kILP == 0 && all_aligned) {
    for (int64_t i = item_idx; i * kILP < n && i * kILP < chunk_size;
         i += item_range) {
      // load
      load_store(r_args[0], args[0], 0, i);
#pragma unroll
      for (int ii = 0; ii < kILP; ++ii) {
        r_args[0][ii] = static_cast<T>(
            op(static_cast<opmath_t>(r_args[0][ii]),
               static_cast<opmath_t>(scalar)));
      }
      // store
      load_store(args[res_arg_index], r_args[0], i, 0);
    }
  } else {
    for (int64_t i = 0; i < n && i < chunk_size; i += item_range * kILP) {
      // Regardless if depth is 1 (for inplace) or 2 (for out of place), r_args
      // has depth 1
      load_args<r_args_depth>(
          r_args, args, i, chunk_size, n, item_idx, item_range);
#pragma unroll
      for (int ii = 0; ii < kILP; ++ii) {
        r_args[0][ii] = static_cast<T>(
            op(static_cast<opmath_t>(r_args[0][ii]),
               static_cast<opmath_t>(scalar)));
      }
      store_args(
          args[res_arg_index],
          r_args[0],
          i,
          chunk_size,
          n,
          item_idx,
          item_range);
    }
  }
}

template <
    int r_args_depth,
    int res_arg_index,
    typename Op,
    typename T,
    typename scalar_t = T>
void binary_op_scalar_tensor(
    T r_args[][kILP],
    T** args,
    scalar_t* scalar,
    int n,
    int chunk_size,
    bool all_aligned,
    Op op,
    size_t item_range,
    size_t item_idx) {
  using opmath_t = at::opmath_type<T>;
  // to make things simple, we put aligned case in a different code path
  if (n % kILP == 0 && chunk_size % kILP == 0 && all_aligned) {
    for (int64_t i = item_idx; i * kILP < n && i * kILP < chunk_size;
         i += item_range) {
      // load
      load_store(r_args[0], args[0], 0, i);
#pragma unroll
      for (int ii = 0; ii < kILP; ii++) {
        r_args[0][ii] = static_cast<T>(
            op(static_cast<opmath_t>(r_args[0][ii]),
               static_cast<opmath_t>(*scalar)));
      }
      // store
      load_store(args[res_arg_index], r_args[0], i, 0);
    }
  } else {
    for (int64_t i = 0; i < n && i < chunk_size; i += item_range * kILP) {
      // Regardless if depth is 1 (for inplace) or 2 (for out of place), r_args
      // has depth 1
      load_args<r_args_depth>(
          r_args, args, i, chunk_size, n, item_idx, item_range);
#pragma unroll
      for (int ii = 0; ii < kILP; ii++) {
        r_args[0][ii] = static_cast<T>(
            op(static_cast<opmath_t>(r_args[0][ii]),
               static_cast<opmath_t>(*scalar)));
      }
      store_args(
          args[res_arg_index],
          r_args[0],
          i,
          chunk_size,
          n,
          item_idx,
          item_range);
    }
  }
}

template <typename T, int depth, int r_args_depth, int res_arg_index>
struct BinaryOpScalarFunctor {
  using opmath_t = at::opmath_type<T>;
  template <typename TLA, typename TLW, typename Op>
  void operator()(
      const int64_t chunk_size,
      TLA tlAddress,
      TLW tlWGMeta,
      sycl::nd_item<1> item_id,
      Op op,
      opmath_t scalar) const {
    auto item_idx = item_id.get_local_id(0);
    auto item_range = item_id.get_local_range(0);
    auto group_idx = item_id.get_group(0);
    int tensor_loc = tlWGMeta[group_idx].wg_to_tensor;
    int chunk_idx = tlWGMeta[group_idx].wg_to_chunk;
    int64_t n = tlAddress[tensor_loc].numel_to_tensor;
    T* args[depth];
    bool all_aligned =
        init_args<depth>(args, tlAddress, chunk_idx, chunk_size, tensor_loc);
    n -= chunk_idx * chunk_size;
    T r_args[r_args_depth][kILP];
    binary_op_scalar<r_args_depth, res_arg_index>(
        r_args,
        args,
        scalar,
        n,
        chunk_size,
        all_aligned,
        op,
        item_range,
        item_idx);
  }
};

template <typename T, int depth, int r_args_depth, int res_arg_index>
struct BinaryOpListAlphaFunctor {
  using opmath_t = at::opmath_type<T>;
  template <typename TLA, typename TLW, typename Op>
  void operator()(
      const int64_t chunk_size,
      TLA tlAddress,
      TLW tlWGMeta,
      sycl::nd_item<1> item_id,
      Op op,
      opmath_t alpha) const {
    auto item_idx = item_id.get_local_id(0);
    auto item_range = item_id.get_local_range(0);
    auto group_idx = item_id.get_group(0);
    int tensor_loc = tlWGMeta[group_idx].wg_to_tensor;
    int chunk_idx = tlWGMeta[group_idx].wg_to_chunk;
    int64_t n = tlAddress[tensor_loc].numel_to_tensor;
    T* args[depth];
    bool all_aligned =
        init_args<depth>(args, tlAddress, chunk_idx, chunk_size, tensor_loc);
    n -= chunk_idx * chunk_size;
    T r_args[r_args_depth][kILP];
    // vec path
    if (n % kILP == 0 && chunk_size % kILP == 0 && all_aligned) {
      for (int64_t i = item_idx; i * kILP < n && i * kILP < chunk_size;
           i += item_range) {
        load_store(r_args[0], args[0], 0, i);
        load_store(r_args[1], args[1], 0, i);
#pragma unroll
        for (int ii = 0; ii < kILP; ++ii) {
          if constexpr (!std::is_same_v<opmath_t, bool>) {
            r_args[0][ii] = static_cast<T>(
                op(static_cast<opmath_t>(r_args[0][ii]),
                   alpha * static_cast<opmath_t>(r_args[1][ii])));
          } else {
            r_args[0][ii] = static_cast<T>(
                op(static_cast<opmath_t>(r_args[0][ii]),
                   alpha && static_cast<opmath_t>(r_args[1][ii])));
          }
        }
        load_store(args[res_arg_index], r_args[0], i, 0);
      }
      // non-vec path
    } else {
      for (int64_t i = 0; i < n && i < chunk_size; i += item_range * kILP) {
        load_args<r_args_depth>(
            r_args, args, i, chunk_size, n, item_idx, item_range);
#pragma unroll
        for (int ii = 0; ii < kILP; ++ii) {
          if constexpr (!std::is_same_v<opmath_t, bool>) {
            r_args[0][ii] = static_cast<T>(
                op(static_cast<opmath_t>(r_args[0][ii]),
                   alpha * static_cast<opmath_t>(r_args[1][ii])));
          } else {
            r_args[0][ii] = static_cast<T>(
                op(static_cast<opmath_t>(r_args[0][ii]),
                   alpha && static_cast<opmath_t>(r_args[1][ii])));
          }
        }
        store_args(
            args[res_arg_index],
            r_args[0],
            i,
            chunk_size,
            n,
            item_idx,
            item_range);
      }
    }
  }
};

template <typename T, int depth, int r_args_depth, int res_arg_index>
struct BinaryOpScalarListFunctor {
  using opmath_t = at::opmath_type<T>;
  template <typename TLA, typename TLW, typename Op>
  void operator()(
      int chunk_size,
      TLA tlAddress,
      TLW tlWGMeta,
      sycl::nd_item<1> item_id,
      Op op) const {
    auto item_idx = item_id.get_local_id(0);
    auto item_range = item_id.get_local_range(0);
    auto group_idx = item_id.get_group(0);
    int tensor_loc = tlWGMeta[group_idx].wg_to_tensor;
    int chunk_idx = tlWGMeta[group_idx].wg_to_chunk;
    int64_t n = tlAddress[tensor_loc].numel_to_tensor;
    opmath_t scalar = tlAddress[tensor_loc].scalar_vals;

    T* args[depth];
    bool all_aligned =
        init_args<depth>(args, tlAddress, chunk_idx, chunk_size, tensor_loc);
    n -= chunk_idx * chunk_size;
    T r_args[r_args_depth][kILP];
    binary_op_scalar<r_args_depth, res_arg_index>(
        r_args,
        args,
        scalar,
        n,
        chunk_size,
        all_aligned,
        op,
        item_range,
        item_idx);
  }
};

template <typename T, int depth, int r_args_depth, int res_arg_index>
struct BinaryOpScalarTensorFunctor {
  using opmath_t = at::opmath_type<T>;
  template <typename TLA, typename TLW, typename Op>
  void operator()(
      const int64_t chunk_size,
      TLA tlAddress,
      TLW tlWGMeta,
      sycl::nd_item<1> item_id,
      Op op,
      T* scalar) const {
    auto item_idx = item_id.get_local_id(0);
    auto item_range = item_id.get_local_range(0);
    auto group_idx = item_id.get_group(0);
    int tensor_loc = tlWGMeta[group_idx].wg_to_tensor;
    int chunk_idx = tlWGMeta[group_idx].wg_to_chunk;
    int64_t n = tlAddress[tensor_loc].numel_to_tensor;

    T* args[depth];
    bool all_aligned =
        init_args<depth>(args, tlAddress, chunk_idx, chunk_size, tensor_loc);
    n -= chunk_idx * chunk_size;
    T r_args[r_args_depth][kILP];

    binary_op_scalar_tensor<r_args_depth, res_arg_index>(
        r_args,
        args,
        scalar,
        n,
        chunk_size,
        all_aligned,
        op,
        item_range,
        item_idx);
  }
};

template <typename T, int depth, int r_args_depth, int res_arg_index>
struct TernaryOpListFunctor {
  using opmath_t = at::opmath_type<T>;
  template <typename TLA, typename TLW, typename Op>
  void operator()(
      int chunk_size,
      TLA tlAddress,
      TLW tlWGMeta,
      sycl::nd_item<1> item_id,
      Op op) const {
    static_assert(depth == 3 || depth == 4, "");
    static_assert(depth >= r_args_depth, "");
    static_assert(res_arg_index == depth - 1 || res_arg_index == 0, "");
    auto item_idx = item_id.get_local_id(0);
    auto item_range = item_id.get_local_range(0);
    auto group_idx = item_id.get_group(0);
    int tensor_loc = tlWGMeta[group_idx].wg_to_tensor;
    int chunk_idx = tlWGMeta[group_idx].wg_to_chunk;
    int64_t n = tlAddress[tensor_loc].numel_to_tensor;

    T* args[depth];
    const bool all_aligned =
        init_args<depth>(args, tlAddress, chunk_idx, chunk_size, tensor_loc);
    n -= chunk_idx * chunk_size;
    T r_args[r_args_depth][kILP];

    if (n % kILP == 0 && chunk_size % kILP == 0 && all_aligned) {
      for (int64_t i_start = item_idx;
           i_start * kILP < n && i_start * kILP < chunk_size;
           i_start += item_range) {
        load_store(r_args[0], args[0], 0, i_start);
        load_store(r_args[1], args[1], 0, i_start);
        load_store(r_args[2], args[2], 0, i_start);
#pragma unroll
        for (int ii = 0; ii < kILP; ii++) {
          r_args[0][ii] =
              op(static_cast<opmath_t>(r_args[0][ii]),
                 static_cast<opmath_t>(r_args[1][ii]),
                 static_cast<opmath_t>(r_args[2][ii]));
        }
        load_store(args[res_arg_index], r_args[0], i_start, 0);
      }
    } else {
      for (int64_t i_start = 0; i_start < n && i_start < chunk_size;
           i_start += item_range * kILP) {
        load_args<r_args_depth>(
            r_args, args, i_start, chunk_size, n, item_idx, item_range);
#pragma unroll
        for (int ii = 0; ii < kILP; ii++) {
          r_args[0][ii] =
              op(static_cast<opmath_t>(r_args[0][ii]),
                 static_cast<opmath_t>(r_args[1][ii]),
                 static_cast<opmath_t>(r_args[2][ii]));
        }
        store_args(
            args[res_arg_index],
            r_args[0],
            i_start,
            chunk_size,
            n,
            item_idx,
            item_range);
      }
    }
  }
};

template <typename T, int depth, int r_args_depth, int res_arg_index>
struct TernaryOpScalarFunctor {
  using opmath_t = at::opmath_type<T>;
  template <typename TLA, typename TLW, typename Op>
  void operator()(
      int chunk_size,
      TLA tlAddress,
      TLW tlWGMeta,
      sycl::nd_item<1> item_id,
      Op op,
      opmath_t alpha) const {
    static_assert(depth == 2 || depth == 3, "");
    static_assert(depth >= r_args_depth, "");
    static_assert(res_arg_index == depth - 1 || res_arg_index == 0, "");
    auto item_idx = item_id.get_local_id(0);
    auto item_range = item_id.get_local_range(0);
    auto group_idx = item_id.get_group(0);
    int tensor_loc = tlWGMeta[group_idx].wg_to_tensor;
    int chunk_idx = tlWGMeta[group_idx].wg_to_chunk;
    int64_t n = tlAddress[tensor_loc].numel_to_tensor;

    T* args[depth];
    const bool all_aligned =
        init_args<depth>(args, tlAddress, chunk_idx, chunk_size, tensor_loc);
    n -= chunk_idx * chunk_size;
    T r_args[r_args_depth][kILP];

    // to make things simple, we put aligned case in a different code path
    if (n % kILP == 0 && chunk_size % kILP == 0 && all_aligned) {
      for (int64_t i_start = item_idx;
           i_start * kILP < n && i_start * kILP < chunk_size;
           i_start += item_range) {
        // load
        load_store(r_args[0], args[0], 0, i_start);
        load_store(r_args[1], args[1], 0, i_start);
#pragma unroll
        for (int ii = 0; ii < kILP; ii++) {
          r_args[0][ii] =
              op(static_cast<opmath_t>(r_args[0][ii]),
                 static_cast<opmath_t>(r_args[1][ii]),
                 alpha);
        }
        // store
        load_store(args[res_arg_index], r_args[0], i_start, 0);
      }
    } else {
      for (int64_t i_start = 0; i_start < n && i_start < chunk_size;
           i_start += item_range * kILP) {
        load_args<r_args_depth>(
            r_args, args, i_start, chunk_size, n, item_idx, item_range);
#pragma unroll
        for (int ii = 0; ii < kILP; ii++) {
          r_args[0][ii] =
              op(static_cast<opmath_t>(r_args[0][ii]),
                 static_cast<opmath_t>(r_args[1][ii]),
                 alpha);
        }
        store_args(
            args[res_arg_index],
            r_args[0],
            i_start,
            chunk_size,
            n,
            item_idx,
            item_range);
      }
    }
  }
};

template <typename T>
struct power_functor {
  T operator()(const T& a, const T& b) const {
    return at::native::xpu::pow_(a, b);
  }
};

template <typename T>
struct reverse_power_functor {
  T operator()(const T& a, const T& b) const {
    return at::native::xpu::pow_(b, a);
  }
};

} // namespace at::native::xpu
