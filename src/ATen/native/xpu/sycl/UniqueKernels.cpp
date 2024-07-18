#include <ATen/ATen.h>
#include <ATen/native/xpu/sycl/ScanUtils.h>
#include <ATen/native/xpu/sycl/pstl/PSTLFunctions.h>
#include <comm/SYCLContext.h>

namespace at::native::xpu {

struct UniqueNotEqualFunctor {
  template <typename T>
  bool operator()(T lhs, T rhs) const {
    if (lhs != rhs) {
      return true;
    }
    return false;
  }
};

struct UniqueEqualFunctor {
  template <typename T>
  bool operator()(T lhs, T rhs) const {
    if (lhs != rhs) {
      return false;
    }
    return true;
  }
};

template <typename index_t>
struct ComputeInverseLTFunctor {
  auto operator()(index_t a, index_t b) const {
    return (a < b);
  }
};

template <>
struct ComputeInverseLTFunctor<float> {
  auto operator()(float a, float b) const {
    return (std::isless(a, b));
  }
};

template <>
struct ComputeInverseLTFunctor<double> {
  auto operator()(double a, double b) const {
    return (std::isless(a, b));
  }
};

template <typename input_t, typename index_t, typename not_equal_t>
Tensor compute_inverse(
    const Tensor& sorted,
    int64_t num_inp,
    const Tensor& sorted_indices,
    const bool return_inverse,
    TensorOptions index_options,
    not_equal_t not_equal) {
  // inverse indices
  Tensor inverse_indices;
  input_t* data = sorted.data_ptr<input_t>();
  auto data_begin = data;
  if (!return_inverse) {
    inverse_indices = at::empty({0}, index_options);
  } else {
    TORCH_INTERNAL_ASSERT(
        sorted_indices.defined(),
        "return_inverse is set to true, but sorted_indices is undefined. Send a bug report!");
    index_t* sorted_indices_ptr = sorted_indices.data_ptr<index_t>();
    Tensor inv_loc = at::empty({num_inp}, index_options);
    inverse_indices = at::empty({num_inp}, index_options);
    index_t* inv_loc_ptr = inv_loc.data_ptr<index_t>();
    auto inv_loc_begin = inv_loc_ptr;
    pstl::adjacent_difference<index_t>(
        data_begin, data_begin + num_inp, inv_loc_begin, not_equal);
    inv_loc[0] = 0;
    scan<INCLUSIVE_TYPE, index_t, index_t>(
        inv_loc,
        inv_loc,
        /*dim*/ 0,
        (index_t)(0.0),
        std::plus<index_t>());
    ComputeInverseLTFunctor<index_t> f;
    pstl::sort<index_t, index_t>(
        sorted_indices_ptr, inv_loc_ptr, sorted_indices.size(0), f);
    inverse_indices = inv_loc;
  }

  return inverse_indices;
}

template <typename input_t, typename index_t, typename equal_t>
std::tuple<Tensor, index_t> compute_unique(
    input_t* data,
    int64_t num_inp,
    const bool return_counts,
    TensorOptions index_options,
    equal_t equal) {
  auto data_begin = data;
  // unique and count
  Tensor counts = at::empty({0}, index_options);
  int64_t num_out;
  if (!return_counts) {
    num_out = pstl::unique<input_t, index_t>(
                  data_begin, data_begin + num_inp, equal) -
        data_begin;
  } else {
    Tensor range = at::empty({num_inp + 1}, index_options);
    index_t* range_begin = range.data_ptr<index_t>();
    pstl::iota(range_begin, range_begin + num_inp + 1, (index_t)0);
    auto data_end = data_begin;
    auto range_end = range_begin;
    std::tie(data_end, range_end) =
        pstl::unique_with_zip<input_t, index_t, index_t>(
            data_begin, data_begin + num_inp, range_begin, equal);
    num_out = std::distance(data_begin, data_end);
    range[num_out] = num_inp;
    counts.resize_(num_out);
    int64_t* counts_ptr = counts.data_ptr<index_t>();
    auto counts_begin = counts_ptr;
    pstl::adjacent_difference<index_t>(
        range_begin + 1, range_begin + num_out + 1, counts_begin);
  }

  return std::tuple<Tensor, index_t>(counts, num_out);
}

template <typename scalar_t>
std::tuple<Tensor, Tensor, Tensor> unique_template(
    const Tensor& self,
    const bool consecutive,
    const bool return_inverse,
    const bool return_counts) {
  int64_t num_inp = self.numel();
  TORCH_CHECK(
      num_inp <= INT_MAX, "num_inp ", num_inp, " is too big to for XPU");
  if (num_inp == 0) {
    Tensor output = at::empty({0}, self.options());
    Tensor inverse_indices =
        at::empty(self.sizes(), self.options().dtype(kLong));
    Tensor counts = at::empty({0}, self.options().dtype(kLong));
    return std::tuple<Tensor, Tensor, Tensor>(output, inverse_indices, counts);
  }

  auto index_options = self.options().dtype(kLong);
  auto self_c = *(self.expect_contiguous());
  Tensor output = self_c.clone().reshape(-1);
  Tensor sorted_indices = at::empty({num_inp}, index_options);
  Tensor inverse_indices = at::empty({num_inp}, index_options);
  Tensor counts = at::empty({num_inp}, index_options);
  auto sorted_indices_begin = sorted_indices.data_ptr<int64_t>();
  at::native::xpu::pstl::iota(
      sorted_indices_begin, sorted_indices_begin + num_inp, (int64_t)0);

  if (!consecutive) {
    at::native::xpu::pstl::sort<scalar_t, int64_t>(
        self_c.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        sorted_indices.data_ptr<int64_t>(),
        num_inp,
        false);
  }

  int64_t num_out;

  scalar_t* output_data = output.data_ptr<scalar_t>();
  UniqueNotEqualFunctor not_equal_cmp_functor;
  inverse_indices = compute_inverse<scalar_t, int64_t>(
      output,
      num_inp,
      sorted_indices,
      return_inverse,
      index_options,
      not_equal_cmp_functor);

  UniqueEqualFunctor equal_cmp_functor;
  std::tie(counts, num_out) = compute_unique<scalar_t, int64_t>(
      output_data, num_inp, return_counts, index_options, equal_cmp_functor);
  output.resize_(num_out);
  if (return_inverse) {
    inverse_indices.resize_(self.sizes());
  }

  return std::tuple<Tensor, Tensor, Tensor>(output, inverse_indices, counts);
}

std::tuple<Tensor, Tensor, Tensor> unique_consecutive_kernel(
    const Tensor& self,
    const bool return_inverse,
    const bool return_counts,
    c10::optional<int64_t> dim) {
  return AT_DISPATCH_V2(
      self.scalar_type(),
      "unique_consecutive",
      AT_WRAP([&] {
        return unique_template<scalar_t>(
            self, true, return_inverse, return_counts);
      }),
      AT_EXPAND(AT_ALL_TYPES),
      kBool,
      kHalf,
      kBFloat16,
      AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
}

std::tuple<Tensor, Tensor> _unique_kernel(
    const Tensor& self,
    const bool return_inverse) {
  return AT_DISPATCH_V2(
      self.scalar_type(),
      "_unique",
      AT_WRAP([&] {
        auto [output, inverse, _] =
            unique_template<scalar_t>(self, false, return_inverse, false);
        return std::make_tuple(output, inverse);
      }),
      AT_EXPAND(AT_ALL_TYPES),
      kBool,
      kHalf,
      kBFloat16,
      AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
}

std::tuple<Tensor, Tensor, Tensor> _unique2_kernel(
    const Tensor& self,
    const bool return_inverse,
    const bool return_counts) {
  return AT_DISPATCH_V2(
      self.scalar_type(),
      "_unique2",
      AT_WRAP([&] {
        return unique_template<scalar_t>(
            self, false, return_inverse, return_counts);
      }),
      AT_EXPAND(AT_ALL_TYPES),
      kBool,
      kHalf,
      kBFloat16,
      AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
}

template <typename scalar_t>
struct UniqueDimLessFunctor {
  bool operator()(int64_t a, int64_t b) const {
    // this is a must to bypass padding comparision in bitonic sort.
    if (a >= num_inp_ || b >= num_inp_)
      return a < b;
    // calculate the dictionary order
    for (int64_t i = 0; i < n_; ++i) {
      scalar_t lhs = c10::load(&input_flat_ptr_[i + a * n_]);
      scalar_t rhs = c10::load(&input_flat_ptr_[i + b * n_]);
      if (lhs < rhs) {
        return true;
      } else if (lhs > rhs) {
        return false;
      }
    }
    return false;
  }

  UniqueDimLessFunctor(int64_t num_inp, int64_t n, scalar_t* input_flat_ptr)
      : num_inp_(num_inp), n_(n), input_flat_ptr_(input_flat_ptr) {}

 private:
  int64_t num_inp_;
  int64_t n_;
  scalar_t* input_flat_ptr_;
};

template <typename scalar_t>
struct UniqueDimEqualFunctor {
  template <typename T>
  bool operator()(T a, T b) const {
    for (int64_t i = 0; i < n_; ++i) {
      scalar_t lhs = c10::load(&input_flat_ptr_[i + a * n_]);
      scalar_t rhs = c10::load(&input_flat_ptr_[i + b * n_]);
      if (lhs != rhs) {
        return false;
      }
    }
    return true;
  }
  UniqueDimEqualFunctor(int64_t n, scalar_t* input_flat_ptr)
      : n_(n), input_flat_ptr_(input_flat_ptr) {}

 private:
  int64_t n_;
  scalar_t* input_flat_ptr_;
};

template <typename scalar_t>
struct UniqueDimNotEqualFunctor {
  template <typename T>
  bool operator()(T a, T b) const {
    for (int64_t i = 0; i < n_; ++i) {
      scalar_t lhs = c10::load(&input_flat_ptr_[i + a * n_]);
      scalar_t rhs = c10::load(&input_flat_ptr_[i + b * n_]);
      if (lhs != rhs) {
        return true;
      }
    }
    return false;
  }
  UniqueDimNotEqualFunctor(int64_t n, scalar_t* input_flat_ptr)
      : n_(n), input_flat_ptr_(input_flat_ptr) {}

 private:
  int64_t n_;
  scalar_t* input_flat_ptr_;
};

template <typename scalar_t>
std::tuple<Tensor, Tensor, Tensor> unique_dim_template(
    const Tensor& self,
    const int64_t dim,
    const bool consecutive,
    const bool return_inverse,
    const bool return_counts) {
  auto sizes = self.sizes().vec();
  auto num_zero_dims = std::count(sizes.begin(), sizes.end(), 0);

  if (self.size(dim) == 0) {
    TORCH_CHECK(
        num_zero_dims == 1,
        "Number of zero sized dimensions is more than one, so unique cannot be applied ")
    Tensor output = at::empty(sizes, self.options());
    Tensor inverse_indices = at::empty({0}, self.options().dtype(kLong));
    Tensor counts = at::empty({0}, self.options().dtype(kLong));

    return std::make_tuple(output, inverse_indices, counts);
  }

  TORCH_CHECK(
      num_zero_dims == 0,
      "There are 0 sized dimensions, and they aren't selected, so unique cannot be applied");

  int64_t num_inp = self.size(dim);
  auto index_options = self.options().dtype(kLong);
  Tensor input_flat = self.moveaxis(dim, 0).contiguous().view({num_inp, -1});
  int64_t n = input_flat.size(1);
  scalar_t* input_flat_ptr = input_flat.data_ptr<scalar_t>();

  Tensor indices = at::arange(0, num_inp, index_options);
  Tensor indices_idx = at::arange(0, num_inp, index_options);
  int64_t* indices_data = indices.data_ptr<int64_t>();
  auto indices_begin = indices_data;

  UniqueDimLessFunctor<scalar_t> less_comp(num_inp, n, input_flat_ptr);
  UniqueDimEqualFunctor<scalar_t> equal_comp(n, input_flat_ptr);
  UniqueDimNotEqualFunctor<scalar_t> not_equal_comp(n, input_flat_ptr);

  if (!consecutive) {
    pstl::sort<int64_t, int64_t>(
        indices_begin, indices_idx.data_ptr<int64_t>(), num_inp, less_comp);
  }
  Tensor origin_indices = indices.clone();
  int64_t* origin_indices_data = origin_indices.data_ptr<int64_t>();

  Tensor inverse_indices, counts;
  inverse_indices = compute_inverse<int64_t, int64_t>(
      indices, num_inp, indices, return_inverse, index_options, not_equal_comp);

  int64_t num_out;
  std::tie(counts, num_out) = compute_unique<int64_t, int64_t>(
      origin_indices_data, num_inp, return_counts, index_options, equal_comp);

  origin_indices.resize_(num_out);
  return std::tuple<Tensor, Tensor, Tensor>(
      self.index_select(dim, origin_indices), inverse_indices, counts);
}

std::tuple<Tensor, Tensor, Tensor> unique_dim_consecutive_kernel(
    const Tensor& self,
    const int64_t dim,
    const bool return_inverse,
    const bool return_counts) {
  return AT_DISPATCH_V2(
      self.scalar_type(),
      "unique_dim_consecutive",
      AT_WRAP([&] {
        return unique_dim_template<scalar_t>(
            self, dim, true, return_inverse, return_counts);
      }),
      AT_EXPAND(AT_ALL_TYPES),
      kBool,
      kHalf,
      AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
}

std::tuple<Tensor, Tensor, Tensor> unique_dim_kernel(
    const Tensor& self,
    const int64_t dim,
    const bool return_inverse,
    const bool return_counts) {
  return AT_DISPATCH_V2(
      self.scalar_type(),
      "unique_dim",
      AT_WRAP([&] {
        return unique_dim_template<scalar_t>(
            self, dim, false, return_inverse, return_counts);
      }),
      AT_EXPAND(AT_ALL_TYPES),
      kBool,
      kHalf,
      kBFloat16,
      AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
}

} // namespace at::native::xpu
