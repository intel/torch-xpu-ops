#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/AccumulateType.h>
#include <ATen/native/sparse/SparseBinaryOpIntersectionCommon.h>
#include <ATen/native/sparse/SparseStubs.h>

#include <ATen/native/sparse/xpu/sycl/SparseBinaryOpIntersectionKernels.h>
#include <ATen/native/xpu/sycl/Loops.h>
#include <comm/SYCLContext.h>

namespace at::native::xpu {

template <int nt, int vt, typename loop_t>
struct ApplyKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    constexpr int nv = nt * vt;
    int idx = nv * item.get_group(0) + item.get_local_id(0);
#pragma unroll
    for (int i = 0; i < vt; ++i) {
      if (idx < n_) {
        loop_(idx);
        idx += nt;
      }
    }
  }
  ApplyKernelFunctor(int64_t n, loop_t loop) : n_(n), loop_(loop) {}

 private:
  int64_t n_;
  loop_t loop_;
};

template <int nt, int vt, typename loop_t>
void launch_kernel(int64_t n, loop_t loop) {
  TORCH_INTERNAL_ASSERT(0 <= n && n <= std::numeric_limits<int32_t>::max());
  if (!n) {
    return;
  }
  size_t local_range = nt;
  size_t global_range = (n + nt * vt - 1) / (nt * vt) * local_range;
  auto caller = ApplyKernelFunctor<nt, vt, loop_t>(n, loop);
  sycl_kernel_submit(global_range, local_range, getCurrentSYCLQueue(), caller);
}

template <
    typename scalar_t,
    typename index_t,
    typename offset_calc_t,
    typename binary_op_t>
struct BinaryOpIntersectionKernelFunctor {
  void operator()(int i) const {
    auto offsets = offset_calc_.get(i);

    auto* ptr_res_values =
        reinterpret_cast<scalar_t*>(ptr_res_values_bytes_ + offsets[0]);
    const auto* ptr_lhs_values =
        reinterpret_cast<const scalar_t*>(ptr_lhs_values_bytes_ + offsets[1]);
    const auto lhs_nnz_idx = *reinterpret_cast<const index_t*>(
        ptr_lhs_select_idx_bytes_ + offsets[2]);
    const auto* ptr_rhs_values =
        reinterpret_cast<const scalar_t*>(ptr_rhs_values_bytes_ + offsets[3]);
    const auto rhs_nnz_idx = *reinterpret_cast<const index_t*>(
        ptr_rhs_select_idx_bytes_ + offsets[4]);
    const auto count = *reinterpret_cast<const int64_t*>(
        ptr_intersction_counts_bytes_ + offsets[5]);

    const auto* ptr_lhs_begin = ptr_lhs_values + lhs_nnz_idx * lhs_nnz_stride_;
    const auto* ptr_rhs_sorted_nnz_idx = ptr_argsort_ + rhs_nnz_idx;

    using accscalar_t = acc_type_device<scalar_t, kXPU>;
    accscalar_t res_values = 0;
    accscalar_t lhs_values = static_cast<accscalar_t>(*ptr_lhs_begin);
    accscalar_t rhs_values;
    index_t rhs_sorted_nnz_idx;
    const auto match_count =
        accumulate_matches_ ? count : std::min<int64_t>(count, 1);
    for (int64_t c = 0; c < match_count; ++c) {
      rhs_sorted_nnz_idx = *ptr_rhs_sorted_nnz_idx++;
      rhs_values = static_cast<accscalar_t>(
          *(ptr_rhs_values + rhs_sorted_nnz_idx * rhs_nnz_stride_));
      res_values += binary_op_t::apply(lhs_values, rhs_values);
    }
    *ptr_res_values = static_cast<scalar_t>(res_values);
  }
  BinaryOpIntersectionKernelFunctor(
      offset_calc_t offset_calc,
      char* ptr_res_values_bytes,
      const char* ptr_lhs_values_bytes,
      const char* ptr_lhs_select_idx_bytes,
      const char* ptr_rhs_values_bytes,
      const char* ptr_rhs_select_idx_bytes,
      const char* ptr_intersction_counts_bytes,
      const index_t* ptr_argsort,
      int64_t lhs_nnz_stride,
      int64_t rhs_nnz_stride,
      const bool accumulate_matches)
      : offset_calc_(offset_calc),
        ptr_res_values_bytes_(ptr_res_values_bytes),
        ptr_lhs_values_bytes_(ptr_lhs_values_bytes),
        ptr_lhs_select_idx_bytes_(ptr_lhs_select_idx_bytes),
        ptr_rhs_values_bytes_(ptr_rhs_values_bytes),
        ptr_rhs_select_idx_bytes_(ptr_rhs_select_idx_bytes),
        ptr_intersction_counts_bytes_(ptr_intersction_counts_bytes),
        ptr_argsort_(ptr_argsort),
        lhs_nnz_stride_(lhs_nnz_stride),
        rhs_nnz_stride_(rhs_nnz_stride),
        accumulate_matches_(accumulate_matches) {}

 private:
  offset_calc_t offset_calc_;
  char* ptr_res_values_bytes_;
  const char* ptr_lhs_values_bytes_;
  const char* ptr_lhs_select_idx_bytes_;
  const char* ptr_rhs_values_bytes_;
  const char* ptr_rhs_select_idx_bytes_;
  const char* ptr_intersction_counts_bytes_;
  const index_t* ptr_argsort_;
  int64_t lhs_nnz_stride_;
  int64_t rhs_nnz_stride_;
  const bool accumulate_matches_;
};

template <typename binary_op_t, typename scalar_t, typename index_t>
void binary_op_intersection_kernel(
    TensorIterator& iter,
    int64_t lhs_nnz_stride,
    int64_t rhs_nnz_stride,
    const Tensor& argsort,
    const bool accumulate_matches) {
  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      binary_op_intersection_kernel<binary_op_t, scalar_t, index_t>(
          sub_iter,
          lhs_nnz_stride,
          rhs_nnz_stride,
          argsort,
          accumulate_matches);
    }
    return;
  }

  auto* ptr_res_values_bytes = reinterpret_cast<char*>(iter.data_ptr(0));
  const auto* ptr_lhs_values_bytes = reinterpret_cast<char*>(iter.data_ptr(1));
  const auto* ptr_lhs_select_idx_bytes =
      reinterpret_cast<char*>(iter.data_ptr(2));
  const auto* ptr_rhs_values_bytes = reinterpret_cast<char*>(iter.data_ptr(3));
  const auto* ptr_rhs_select_idx_bytes =
      reinterpret_cast<char*>(iter.data_ptr(4));
  const auto* ptr_intersction_counts_bytes =
      reinterpret_cast<char*>(iter.data_ptr(5));
  const auto* ptr_argsort = argsort.const_data_ptr<index_t>();

  auto offset_calc = make_offset_calculator<6>(iter);
  auto fn = BinaryOpIntersectionKernelFunctor<
      scalar_t,
      index_t,
      decltype(offset_calc),
      binary_op_t>(
      offset_calc,
      ptr_res_values_bytes,
      ptr_lhs_values_bytes,
      ptr_lhs_select_idx_bytes,
      ptr_rhs_values_bytes,
      ptr_rhs_select_idx_bytes,
      ptr_intersction_counts_bytes,
      ptr_argsort,
      lhs_nnz_stride,
      rhs_nnz_stride,
      accumulate_matches);
  launch_kernel<128, 4, decltype(fn)>(iter.numel(), fn);
}

template <typename binary_op_t>
struct ValueSelectionIntersectionKernel {
  static Tensor apply(
      const Tensor& lhs_values,
      const Tensor& lhs_select_idx,
      const Tensor& rhs_values,
      const Tensor& rhs_select_idx,
      const Tensor& intersection_counts,
      const Tensor& argsort,
      const bool accumulate_matches) {
    auto iter = make_value_selection_intersection_iter(
        lhs_values,
        lhs_select_idx,
        rhs_values,
        rhs_select_idx,
        intersection_counts);
    auto res_values = iter.tensor(0);

    // If res_values is empty, we can return it right away.
    // Otherwise floating point issues with OffsetCalculator.
    if (!res_values.numel()) {
      return res_values;
    }

    const auto lhs_nnz_stride = lhs_values.stride(0);
    const auto rhs_nnz_stride = rhs_values.stride(0);

    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
        ScalarType::Bool,
        ScalarType::Half,
        ScalarType::BFloat16,
        at::ScalarType::ComplexHalf,
        res_values.scalar_type(),
        "binary_op_intersection_xpu",
        [&] {
          // COO indices are only 64-bit for now.
          using index_t = int64_t;
          binary_op_intersection_kernel<binary_op_t, scalar_t, index_t>(
              iter,
              lhs_nnz_stride,
              rhs_nnz_stride,
              argsort,
              accumulate_matches);
        });

    return res_values;
  }
};

template <typename index_t, typename hash_coeffs_t>
struct SparseBinaryOpIntersectionAFunctor {
  int64_t operator()(index_t nnz_idx) const {
    int64_t hash = 0;
    if (!ptr_indices_) {
      return hash;
    }
    const auto* RESTRICT ptr_indices_dim =
        ptr_indices_ + nnz_idx * indices_nnz_stride_;
    for (int64_t dim = 0; dim < sparse_dim_; ++dim) {
      const auto dim_hash_coeff = hash_coeffs_[dim];
      const auto dim_index = ptr_indices_dim[dim * indices_dim_stride_];
      hash += dim_index * dim_hash_coeff;
    }
    return hash;
  }
  SparseBinaryOpIntersectionAFunctor(
      const index_t* RESTRICT ptr_indices,
      int64_t indices_nnz_stride,
      int64_t sparse_dim,
      hash_coeffs_t hash_coeffs,
      int64_t indices_dim_stride)
      : ptr_indices_(ptr_indices),
        indices_nnz_stride_(indices_nnz_stride),
        sparse_dim_(sparse_dim),
        hash_coeffs_(hash_coeffs),
        indices_dim_stride_(indices_dim_stride) {}

 private:
  const index_t* RESTRICT ptr_indices_;
  int64_t indices_nnz_stride_;
  int64_t sparse_dim_;
  hash_coeffs_t hash_coeffs_;
  int64_t indices_dim_stride_;
};

template <typename index_t, typename hash_coeffs_t, typename hash_t = int64_t>
struct SparseBinaryOpIntersectionBFunctor {
  index_t operator()(index_t nnz_idx) const {
    int64_t hash = 0;
    if (hash_ptr_) {
      hash = hash_ptr_[nnz_idx];
    } else if (sparse_dim_) {
      // Compute hash value
      const auto* RESTRICT ptr_indices_dim =
          ptr_indices_ + nnz_idx * indices_nnz_stride_;
      for (int64_t dim = 0; dim < sparse_dim_; ++dim) {
        const auto dim_hash_coeff = hash_coeffs_[dim];
        const auto dim_index = ptr_indices_dim[dim * indices_dim_stride_];
        hash += dim_index * dim_hash_coeff;
      }
    }

    // Perform hash values intersection
    const auto* RESTRICT lb =
        find_bound<const int64_t*, int64_t, /*is_lower=*/true>(
            ptr_sorted_hash_, ptr_sorted_hash_ + sorted_hash_len_, hash);

    const auto* RESTRICT ub =
        find_bound<const int64_t*, int64_t, /*is_lower=*/false>(
            ptr_sorted_hash_, ptr_sorted_hash_ + sorted_hash_len_, hash);

    ptr_intersection_count_[nnz_idx] = ub - lb;
    ptr_intersection_first_idx_[nnz_idx] = lb - ptr_sorted_hash_;

    return 0;
  }
  SparseBinaryOpIntersectionBFunctor(
      const hash_t* RESTRICT hash_ptr,
      const index_t* RESTRICT ptr_indices,
      int64_t indices_nnz_stride,
      int64_t sparse_dim,
      hash_coeffs_t hash_coeffs,
      int64_t indices_dim_stride,
      const int64_t* RESTRICT ptr_sorted_hash,
      int64_t sorted_hash_len,
      int64_t* RESTRICT ptr_intersection_count,
      int64_t* RESTRICT ptr_intersection_first_idx)
      : hash_ptr_(hash_ptr),
        ptr_indices_(ptr_indices),
        indices_nnz_stride_(indices_nnz_stride),
        sparse_dim_(sparse_dim),
        hash_coeffs_(hash_coeffs),
        indices_dim_stride_(indices_dim_stride),
        ptr_sorted_hash_(ptr_sorted_hash),
        sorted_hash_len_(sorted_hash_len),
        ptr_intersection_count_(ptr_intersection_count),
        ptr_intersection_first_idx_(ptr_intersection_first_idx) {}

 private:
  const hash_t* RESTRICT hash_ptr_;
  const index_t* RESTRICT ptr_indices_;
  int64_t indices_nnz_stride_;
  int64_t sparse_dim_;
  hash_coeffs_t hash_coeffs_;
  int64_t indices_dim_stride_;
  const int64_t* RESTRICT ptr_sorted_hash_;
  int64_t sorted_hash_len_;
  int64_t* RESTRICT ptr_intersection_count_;
  int64_t* RESTRICT ptr_intersection_first_idx_;
};

template <
    typename value_selection_intersection_kernel_t,
    typename index_t = int64_t,
    int64_t max_static_len = 0>
void _sparse_binary_op_intersection_kernel_impl(
    Tensor& res,
    const Tensor& x_,
    const Tensor& y_,
    const std::vector<int64_t>& broadcasted_shape,
    const std::optional<Tensor>& x_hash_opt_ = std::nullopt,
    const std::optional<Tensor>& y_hash_opt_ = std::nullopt,
    const bool accumulate_matches = true,
    const bool distributive_with_sum = true) {
  // The common dtype check is relevant when op is done in-place.
  // This is because binary_of_t produces new values and it could be that
  // new_values.dtype != res.dtype. In such a case we should error out
  // as soon as possible to avoid redundant kernel runs.
  const auto common_dtype = at::result_type(x_, y_);
  TORCH_CHECK(
      canCast(common_dtype, res.scalar_type()),
      "Can't convert result type ",
      common_dtype,
      " to output ",
      res.scalar_type());

  using OptTensor = std::optional<Tensor>;

  // If the op and sum are not distributive, coalesce is required.
  const auto coalesce_if_not_distributive = [distributive_with_sum](
      const Tensor& t, const OptTensor& t_hash_opt) -> auto{
    // No need to coalesce in such a case.
    if (distributive_with_sum) {
      return std::make_tuple(t, t_hash_opt);
    } else {
      // Otherwise coalesce and force hash recompute.
      return std::make_tuple(
          t.coalesce(), static_cast<OptTensor>(std::nullopt));
    }
  };

  Tensor x, y;
  OptTensor x_hash_opt, y_hash_opt;
  std::tie(x, x_hash_opt) = coalesce_if_not_distributive(x_, x_hash_opt_);
  std::tie(y, y_hash_opt) = coalesce_if_not_distributive(y_, y_hash_opt_);

  // Given sparse tensors x and y we decide which one is source, and which one
  // is probably_coalesced. The indices of both source and probably_coalesced
  // are hashed and then the hash values of the source's indices are
  // binary-searched into the hash values of the probably_coalesced's indices.
  // If probably_coalesce is coalesced, by the property of the hashing method
  // (see below), the hash values are already sorted and we can avoid any
  // explicit sorting routines.
  Tensor probably_coalesced, source;
  OptTensor probably_coalesced_indices_hash_opt, source_indices_hash_opt;
  std::tie(
      probably_coalesced,
      probably_coalesced_indices_hash_opt,
      source,
      source_indices_hash_opt) = [&]() -> auto{
    // Case 1: either x or y is coalesced.
    if ((x.is_coalesced() ^ y.is_coalesced())) {
      return x.is_coalesced() ? std::make_tuple(x, x_hash_opt, y, y_hash_opt)
                              : std::make_tuple(y, y_hash_opt, x, x_hash_opt);
    }
    // Case 2: Both x and y are either coalesced or non-coalesced.
    // If both are coalesced, search into the larger tensor is faster.
    // Same holds when both are non-coalesced.
    else {
      Tensor larger, smaller;
      OptTensor larger_hash_opt, smaller_hash_opt;
      std::tie(
          larger, larger_hash_opt, smaller, smaller_hash_opt) = [&]() -> auto{
        return x._nnz() >= y._nnz()
            ? std::make_tuple(x, x_hash_opt, y, y_hash_opt)
            : std::make_tuple(y, y_hash_opt, x, x_hash_opt);
      }
      ();

      // If under a uniform distribution it is likely to hit many elements in
      // larger, it is best to coalesce it for better performance.
      const auto larger_sizes = larger.sizes();
      const auto sparse_dim_numel = std::accumulate(
          larger_sizes.begin(),
          larger_sizes.begin() + larger.sparse_dim(),
          1,
          std::multiplies<int64_t>());
      // If nnz > prod(larger.shape[:sparse_dim]), by the pidgeonhole principle,
      // there is at least one bucket with nnz / prod(larger.shape[:sparse_dim])
      // elements. It provides a lower bound for the max count in the
      // intersection. This condition is very conservative as we do not check
      // whether such an event actually occurred, although it is very likely
      // under a uniform distribution, the distribution with the highest
      // uncertainty (maximizes entropy).
      const auto max_count_lower_bound = larger._nnz() / sparse_dim_numel;
      constexpr int64_t MAX_COPIES_PER_THREAD = 50;
      return max_count_lower_bound > MAX_COPIES_PER_THREAD
          // coalesce invalidates hash values, so force-recompute
          ? std::make_tuple(
                larger.coalesce(),
                static_cast<OptTensor>(std::nullopt),
                smaller,
                smaller_hash_opt)
          : std::make_tuple(larger, larger_hash_opt, smaller, smaller_hash_opt);
    }
  }
  ();

  // The employed hash function maps a d-dim index to a linear offset
  // into a contiguous memory that is sufficient to fit a dense tensor
  // of shape broadcasted_shape(x.shape, y.shape), i.e.
  // idx -> \sum_{i = 0}^d idx[i] * hash_coeffs[i], where
  // hash_coeffs are the strides of a contiguous tensor of shape
  // broadcasted_shape(x.shape, y.shape).
  // Assuming the following order on the dimensions, i.e. the right-most dim is
  // the fastest-changing dim, and the left-most is the slowest-changing dim,
  // which is implicit in the definition of hash_coeffs,
  // it could be shown that the hash function is actually bijective and, hence,
  // is a perfect hash function (no collisions ever).

  // Need owning storage in case of the Tensor class.
  const auto hash_coeffs_storage = [&]() -> auto{
    const auto broadcasted_sparse_dim_shape = std::vector<int64_t>(
        broadcasted_shape.begin(),
        broadcasted_shape.begin() + probably_coalesced.sparse_dim());
    auto strides = c10::contiguous_strides(broadcasted_sparse_dim_shape);
    return at::sparse::TensorGeometryHolder<max_static_len>(
        strides, strides, probably_coalesced.options());
  }
  ();

  const auto hash_coeffs = std::get<0>(*hash_coeffs_storage);

  const auto nnz_arange = at::arange(
      std::max(probably_coalesced._nnz(), source._nnz()),
      source._indices().options());
  const auto probably_coalesced_nnz_arange =
      nnz_arange.narrow(-1, 0, probably_coalesced._nnz());

  // non-const because of gcc-5/clang-5 issues
  auto sparse_dim = probably_coalesced.sparse_dim();

  // Apply the hash function to probably_coalesced.indices
  const auto probably_coalesced_indices_hash = [&]() -> Tensor {
    // probably_coalesced is coalesced and hash provided? Reuse it!
    if (probably_coalesced_indices_hash_opt.has_value()) {
      return (*probably_coalesced_indices_hash_opt).contiguous();
    }

    const auto indices = probably_coalesced._indices();
    // non-const because of gcc-5/clang-5 issues
    auto indices_dim_stride = indices.stride(0);
    auto indices_nnz_stride = indices.stride(1);

    auto hash =
        at::empty({probably_coalesced._nnz()}, indices.options().dtype(kLong));

    auto iter = TensorIteratorConfig()
                    .check_all_same_dtype(false)
                    .add_output(hash)
                    .add_input(probably_coalesced_nnz_arange)
                    .build();

    {
      const auto* RESTRICT ptr_indices = indices.const_data_ptr<index_t>();
      auto fn =
          SparseBinaryOpIntersectionAFunctor<index_t, decltype(hash_coeffs)>(
              ptr_indices,
              indices_nnz_stride,
              sparse_dim,
              hash_coeffs,
              indices_dim_stride);
      gpu_kernel(iter, fn);
    }

    return hash;
  }();

  // Now that we have hash values of probably_coalesced.indices,
  // we need to decide whether they need to get sorted.
  // The sort is not requires if probably_coalesced is coalesced.
  Tensor sorted_hash, argsort_hash;
  std::tie(sorted_hash, argsort_hash) = [&]() -> std::tuple<Tensor, Tensor> {
    if (probably_coalesced.is_coalesced()) {
      // NOTE: argsort.dtype == nnz_arange.dtype
      const auto argsort = nnz_arange.narrow(-1, 0, probably_coalesced._nnz());
      return std::make_tuple(probably_coalesced_indices_hash, argsort);
    } else {
      // NOTE: we want argsort.dtype == nnz_arange.dtype,
      // but sort() produces indices of type int64_t,
      // so we convert to nnz_arange.dtype to avoid issues
      // with pointer types in the kernels below.
      Tensor sorted, argsort;
      std::tie(sorted, argsort) = probably_coalesced_indices_hash.sort();
      return std::make_tuple(sorted, argsort.to(nnz_arange.scalar_type()));
    }
  }();

  // Perform hash intersection.
  // Let  s_hash = hash(source.indices),
  //     pc_hash = hash(probably_coalesced.indices), then
  // for i = 0, ..., len(s_hash) - 1:
  //     lb = <index of a value in pc_hash[argsort_hash] which is a lower bound
  //     for s_hash[i]>, up = <index of a value in pc_hash[argsort_hash] which
  //     is an upper bound for s_hash[i]>, intersection_count[i] = up - lb
  //     intersection_first_idx[i] = lb.
  //
  // intersection_count and intersection_first_idx are used to form indices at
  // which intersection values are selected.
  auto [intersection_count, intersection_first_idx] =
      [&]() -> std::tuple<Tensor, Tensor> {
    const auto source_nnz = source._nnz();
    auto intersection_buffer =
        at::empty({2, source_nnz}, sorted_hash.options());
    auto intersection_count = intersection_buffer.select(0, 0);
    auto intersection_first_idx = intersection_buffer.select(0, 1);

    const auto source_indices = source._indices();
    const auto source_arange = nnz_arange.narrow(-1, 0, source_nnz);
    // non-const because of gcc-5/clang-5 issues
    auto indices_dim_stride = source_indices.stride(0);
    auto indices_nnz_stride = source_indices.stride(1);
    auto dummy = at::empty({1}, source_arange.options());

    auto hash = source_indices_hash_opt.has_value()
        ? (*source_indices_hash_opt).contiguous()
        : at::empty({0}, probably_coalesced._indices().options().dtype(kLong));
    const auto* RESTRICT hash_ptr = source_indices_hash_opt.has_value()
        ? hash.data_ptr<int64_t>()
        : nullptr;

    auto iter = TensorIteratorConfig()
                    .set_check_mem_overlap(false)
                    .add_owned_output(dummy.expand_as(source_arange))
                    .add_input(source_arange)
                    .build();

    {
      const auto* RESTRICT ptr_indices =
          source_indices.const_data_ptr<index_t>();
      const auto* RESTRICT ptr_sorted_hash =
          sorted_hash.const_data_ptr<int64_t>();
      const auto sorted_hash_len = sorted_hash.numel();
      auto* RESTRICT ptr_intersection_count =
          intersection_count.data_ptr<int64_t>();
      auto* RESTRICT ptr_intersection_first_idx =
          intersection_first_idx.data_ptr<int64_t>();

      // Fusing hash computation with hash intersection.
      auto fn =
          SparseBinaryOpIntersectionBFunctor<index_t, decltype(hash_coeffs)>(
              hash_ptr,
              ptr_indices,
              indices_nnz_stride,
              sparse_dim,
              hash_coeffs,
              indices_dim_stride,
              ptr_sorted_hash,
              sorted_hash_len,
              ptr_intersection_count,
              ptr_intersection_first_idx);
      gpu_kernel(iter, fn);
    }

    return std::make_tuple(intersection_count, intersection_first_idx);
  }();

  const auto res_indices = source._indices().clone();
  const auto binary_op_res_dtype =
      at::result_type(source._values(), probably_coalesced._values());
  const auto res_values =
      value_selection_intersection_kernel_t::apply(
          source._values().to(binary_op_res_dtype),
          nnz_arange.narrow(-1, 0, source._nnz()),
          probably_coalesced._values().to(binary_op_res_dtype),
          intersection_first_idx.to(nnz_arange.scalar_type()),
          intersection_count,
          argsort_hash,
          accumulate_matches)
          .to(res.scalar_type());
  const auto res_sparse_dim = source.sparse_dim();
  const auto res_dense_dim = source.dense_dim();
  const auto& res_shape = broadcasted_shape;
  const auto res_nnz = source._nnz();

  auto* res_sparse_impl = get_sparse_impl(res);
  res_sparse_impl->raw_resize_(res_sparse_dim, res_dense_dim, res_shape);
  res_sparse_impl->set_indices_and_values_unsafe(res_indices, res_values);
  res_sparse_impl->set_nnz_and_narrow(res_nnz);
  res._coalesced_(source.is_coalesced());
}

template <typename value_selection_intersection_kernel_t>
void _sparse_binary_op_intersection_kernel_out(
    Tensor& res,
    const Tensor& x,
    const Tensor& y,
    const std::optional<Tensor>& x_hash_opt = std::nullopt,
    const std::optional<Tensor>& y_hash_opt = std::nullopt,
    // If op distributes with the sum, the arguments are processed as is,
    // without the calls to coalesce().
    const bool distributive_with_sum = true) {
  TORCH_CHECK(
      (x.is_sparse() && y.is_sparse()) && (x.dim() == y.dim()) &&
          (x.sparse_dim() == y.sparse_dim()) &&
          (x.sizes().slice(0, x.sparse_dim()) ==
           y.sizes().slice(0, y.sparse_dim())),
      NAME,
      "(): expects sparse inputs with equal dimensionality, ",
      "number of sparse dimensions, and shape of sparse dimensions");
  TORCH_CHECK(
      x._indices().scalar_type() == y._indices().scalar_type(),
      NAME,
      "(): expects inputs' indices to be of the same dtype (i.e. long or int)");

  const auto check_hash_validity = [](const Tensor& t,
                                      const std::optional<Tensor>& t_hash_opt) {
    if (!t_hash_opt.has_value()) {
      return;
    }

    const auto& t_hash = *t_hash_opt;
    TORCH_INTERNAL_ASSERT(
        t_hash.dim() == 1 && t_hash.scalar_type() == kLong &&
            t_hash.size(-1) == t._indices().size(-1),
        NAME,
        "(): explicit hash values need to be a 1-dim Long tensor with the ",
        "NSE matching that of the corresponding sparse tensor.");
  };

  check_hash_validity(x, x_hash_opt);
  check_hash_validity(y, y_hash_opt);

  const auto broadcasted_shape = infer_size(x.sizes(), y.sizes());

  // 8 sparse dims should be more than enough?
  constexpr int64_t max_sparse_dims = 8;

  // COO indices are only 64-bit integers for now.
  using index_t = int64_t;

  if (max_sparse_dims > x.sparse_dim()) {
    _sparse_binary_op_intersection_kernel_impl<
        // For some reason MSVC complaints about passing constexpr
        // max_sparse_dims as a template parameter claiming as if it is not know
        // at compile time.
        value_selection_intersection_kernel_t,
        index_t,
        8>(
        res,
        x,
        y,
        broadcasted_shape,
        x_hash_opt,
        y_hash_opt,
        distributive_with_sum);
  } else {
    _sparse_binary_op_intersection_kernel_impl<
        value_selection_intersection_kernel_t,
        index_t>(
        res,
        x,
        y,
        broadcasted_shape,
        x_hash_opt,
        y_hash_opt,
        distributive_with_sum);
  }
}

struct MulOp {
  template <typename scalar_t>
  static inline scalar_t apply(scalar_t a, scalar_t b) {
    return a * b;
  }
};

template <>
inline bool MulOp::apply(bool a, bool b) {
  return a && b;
}

void mul_sparse_sparse_kernel(
    Tensor& result,
    const Tensor& x,
    const Tensor& y) {
  using ValueSelectionMulKernel = ValueSelectionIntersectionKernel<MulOp>;
  _sparse_binary_op_intersection_kernel_out<ValueSelectionMulKernel>(
      result, x, y);
}

struct RhsProjOp {
  template <typename scalar_t>
  static inline scalar_t apply(scalar_t a, scalar_t b) {
    return b;
  }
};

void sparse_mask_intersection_kernel(
    Tensor& result,
    const Tensor& x,
    const Tensor& y,
    const OptTensor& x_hash_opt) {
  using ValueRhsProjKernel = ValueSelectionIntersectionKernel<RhsProjOp>;
  _sparse_binary_op_intersection_kernel_out<ValueRhsProjKernel>(
      result, x, y, x_hash_opt);
}

struct LhsProjOp {
  template <typename scalar_t>
  static inline scalar_t apply(scalar_t a, scalar_t b) {
    return a;
  }
};

void sparse_mask_projection_kernel(
    Tensor& result,
    const Tensor& x,
    const Tensor& y,
    const OptTensor& x_hash_opt,
    bool accumulate_matches) {
  using ValueLhsProjKernel = ValueSelectionIntersectionKernel<LhsProjOp>;
  _sparse_binary_op_intersection_kernel_out<ValueLhsProjKernel>(
      result, x, y, x_hash_opt, std::nullopt, accumulate_matches);
}

} // namespace at::native::xpu
