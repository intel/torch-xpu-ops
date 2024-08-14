#include <ATen/native/Resize.h>
#include <ATen/native/SpectralOpsUtils.h>
#include <ATen/native/xpu/mkl/SpectralOps.h>
#include <ATen/ops/mul.h>
#include <comm/SYCLContext.h>
#include <comm/TensorInfo.h>
#include <oneapi/mkl.hpp>

using namespace oneapi::mkl::dft;

namespace at::native::xpu {

namespace impl {

constexpr int64_t mkl_max_ndim = 3;

// Sort transform dimensions by input layout, for best performance
// exclude_last is for onesided transforms where the last dimension cannot be
// reordered
static DimVector _sort_dims(
    const Tensor& self,
    IntArrayRef dim,
    bool exclude_last = false) {
  DimVector sorted_dims(dim.begin(), dim.end());
  auto self_strides = self.strides();
  std::sort(
      sorted_dims.begin(),
      sorted_dims.end() - exclude_last,
      [&](int64_t a, int64_t b) { return self_strides[a] > self_strides[b]; });
  return sorted_dims;
}

class dft_config_t {
 public:
  using config_int64_t = std::unordered_map<config_param, int64_t>;
  using config_float_t = std::unordered_map<config_param, float>;
  using config_double_t = std::unordered_map<config_param, double>;

  dft_config_t() {
    val_int64_.clear();
    val_float_.clear();
    val_double_.clear();
    fwd_strides_.clear();
    bwd_strides_.clear();
  }

  void set_strides(
      std::vector<int64_t>& fwd_strides,
      std::vector<int64_t>& bwd_strides) {
    fwd_strides_ = fwd_strides;
    bwd_strides_ = bwd_strides;
  }

  template <typename T>
  void set_value(config_param key, T value) {
    if (std::is_same<DFTI_CONFIG_VALUE, T>::value ||
        std::is_same<int64_t, T>::value) {
      val_int64_.insert({key, value});
    } else if (std::is_same<T, float>::value) {
      val_float_.insert({key, value});
    } else if (std::is_same<T, double>::value) {
      val_double_.insert({key, value});
    } else {
      TORCH_CHECK(0, "Unsupported value type in FFT config!");
    }
  }

  template <precision prec, domain dom>
  void commit_values(descriptor<prec, dom>& desc) {
#define COMMIT_VAL(val_map)                    \
  for (auto& value : (val_map)) {              \
    desc.set_value(value.first, value.second); \
  }

    COMMIT_VAL(val_int64_);
    COMMIT_VAL(val_float_);
    COMMIT_VAL(val_double_);

    if (!fwd_strides_.empty()) {
      desc.set_value(config_param::FWD_STRIDES, fwd_strides_.data());
    }
    if (!bwd_strides_.empty()) {
      desc.set_value(config_param::BWD_STRIDES, bwd_strides_.data());
    }
  }

 private:
  config_int64_t val_int64_;
  config_float_t val_float_;
  config_double_t val_double_;
  std::vector<int64_t> fwd_strides_;
  std::vector<int64_t> bwd_strides_;
};

template <precision prec, domain dom>
class dft_desc_t {
 public:
  using mkl_desc_t = descriptor<prec, dom>;

  dft_desc_t(
      sycl::queue& q,
      std::vector<std::int64_t>& dimensions,
      std::shared_ptr<dft_config_t> configs)
      : desc_(dimensions), configs_(configs) {
    configs_->commit_values(desc_);
    desc_.set_value(
        oneapi::mkl::dft::config_param::WORKSPACE,
        oneapi::mkl::dft::config_value::WORKSPACE_EXTERNAL);
    desc_.commit(q);
  }

  mkl_desc_t& raw() {
    return desc_;
  }

 private:
  mkl_desc_t desc_;
  std::shared_ptr<dft_config_t> configs_;
};

template <precision prec, domain signal_type, typename scalar_t>
void _mkl_dft(
    const Tensor& input,
    Tensor& output,
    int64_t signal_ndim,
    bool complex_input,
    bool complex_output,
    bool inverse,
    IntArrayRef checked_signal_sizes,
    bool onesided,
    int64_t batch) {
  auto& queue = at::xpu::getCurrentSYCLQueue();
  std::vector<int64_t> mkl_signal_sizes(
      checked_signal_sizes.begin() + 1, checked_signal_sizes.end());

  std::shared_ptr<dft_config_t> desc_config(new dft_config_t);
  desc_config->set_value(config_param::PLACEMENT, DFTI_NOT_INPLACE);
  desc_config->set_value(config_param::NUMBER_OF_TRANSFORMS, batch);

  auto istrides = input.strides();
  auto ostrides = output.strides();
  int64_t idist = istrides[0];
  int64_t odist = ostrides[0];

  if (!inverse) {
    desc_config->set_value(config_param::FWD_DISTANCE, idist);
    desc_config->set_value(config_param::BWD_DISTANCE, odist);
  } else {
    desc_config->set_value(config_param::FWD_DISTANCE, odist);
    desc_config->set_value(config_param::BWD_DISTANCE, idist);
  }

  std::vector<int64_t> fwd_strides(1 + signal_ndim, 0),
      bwd_strides(1 + signal_ndim, 0);

  for (int64_t i = 1; i <= signal_ndim; i++) {
    if (!inverse) {
      fwd_strides[i] = istrides[i];
      bwd_strides[i] = ostrides[i];
    } else {
      fwd_strides[i] = ostrides[i];
      bwd_strides[i] = istrides[i];
    }
  }

  desc_config->set_strides(fwd_strides, bwd_strides);

  if (!complex_input || !complex_output) {
    desc_config->set_value(
        config_param::CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
  }

  auto desc =
      dft_desc_t<prec, signal_type>(queue, mkl_signal_sizes, desc_config);

  // Obtain the size of workspace required after commit.
  size_t workspaceSizeBytes = 0;
  desc.raw().get_value(
      oneapi::mkl::dft::config_param::WORKSPACE_BYTES, &workspaceSizeBytes);

  // Allocate USM workspace and provide it to the descriptor.
  Tensor workspaceBuf = at::empty(
      {(long)(workspaceSizeBytes / sizeof(double))},
      input.options().dtype(at::kDouble),
      c10::nullopt);
  desc.raw().set_workspace((double*)workspaceBuf.data_ptr());

  auto in_data = (scalar_t*)input.data_ptr();
  auto out_data = (scalar_t*)output.data_ptr();

  sycl::event event;
  if (!inverse) {
    event = compute_forward(desc.raw(), in_data, out_data);
  } else {
    event = compute_backward(desc.raw(), in_data, out_data);
  }
  event.wait_and_throw();
  queue.throw_asynchronous();
}

void _fft_with_size(
    Tensor& output,
    const Tensor& self,
    int64_t signal_ndim,
    bool complex_input,
    bool complex_output,
    bool inverse,
    IntArrayRef checked_signal_sizes,
    bool onesided) {
  int64_t batch = self.size(0);
  Tensor input_ = self;
  // real/imag dimension must aligned when viewed as of complex type

  if (complex_input) {
    bool need_contiguous = input_.stride(-1) != 1;

    for (int64_t i = 0; !need_contiguous && i <= signal_ndim; i++) {
      need_contiguous |= input_.stride(i) % 2 != 0;
    }

    if (need_contiguous) {
      input_ = input_.contiguous();
    }
  }

  bool complex_type = inverse ? complex_output : complex_input;

  void (*dft_func)(
      const class at::Tensor&,
      class at::Tensor&,
      int64_t,
      bool,
      bool,
      bool,
      class c10::ArrayRef<int64_t>,
      bool,
      int64_t);
  Tensor input = input_;

  if (input.scalar_type() == ScalarType::Float ||
      input.scalar_type() == ScalarType::ComplexFloat) {
    dft_func = complex_type
        ? _mkl_dft<precision::SINGLE, domain::COMPLEX, float>
        : _mkl_dft<precision::SINGLE, domain::REAL, float>;
  } else if (
      input.scalar_type() == ScalarType::Double ||
      input.scalar_type() == ScalarType::ComplexDouble) {
    dft_func = complex_type
        ? _mkl_dft<precision::DOUBLE, domain::COMPLEX, double>
        : _mkl_dft<precision::DOUBLE, domain::REAL, double>;
  } else {
    AT_ERROR("MKL FFT doesn't support tensor of type");
  }

  dft_func(
      input,
      output,
      signal_ndim,
      complex_input,
      complex_output,
      inverse,
      checked_signal_sizes,
      onesided,
      batch);
}

// Execute a general fft operation (can be c2c, onesided r2c or onesided c2r)
Tensor& _exec_fft(
    Tensor& out,
    Tensor self,
    IntArrayRef out_sizes,
    IntArrayRef dim,
    bool onesided,
    bool forward) {
  const auto ndim = self.dim();
  const int64_t signal_ndim = dim.size();
  const auto batch_dims = ndim - signal_ndim;

  // Permute dimensions so batch dimensions come first, and in stride order
  // This maximizes data locality when collapsing to a single batch dimension
  DimVector dim_permute(ndim);
  std::iota(dim_permute.begin(), dim_permute.end(), int64_t{0});

  c10::SmallVector<bool, kDimVectorStaticSize> is_transformed_dim(ndim);
  for (const auto& d : dim) {
    is_transformed_dim[d] = true;
  }

  auto batch_end =
      std::partition(dim_permute.begin(), dim_permute.end(), [&](int64_t d) {
        return !is_transformed_dim[d];
      });

  auto self_strides = self.strides();
  std::sort(dim_permute.begin(), batch_end, [&](int64_t a, int64_t b) {
    return self_strides[a] > self_strides[b];
  });
  std::copy(dim.cbegin(), dim.cend(), batch_end);

  auto input = self.permute(dim_permute);

  // Collapse batch dimensions into a single dimension
  DimVector batched_sizes(signal_ndim + 1);
  batched_sizes[0] = -1;
  std::copy(
      input.sizes().cbegin() + batch_dims,
      input.sizes().cend(),
      batched_sizes.begin() + 1);
  input = input.reshape(batched_sizes);

  const auto batch_size = input.sizes()[0];
  DimVector signal_size(signal_ndim + 1);
  signal_size[0] = batch_size;

  for (int64_t i = 0; i < signal_ndim; ++i) {
    auto in_size = input.sizes()[i + 1];
    auto out_size = out_sizes[dim[i]];
    signal_size[i + 1] = std::max(in_size, out_size);
    TORCH_INTERNAL_ASSERT(
        in_size == signal_size[i + 1] ||
        in_size == (signal_size[i + 1] / 2) + 1);
    TORCH_INTERNAL_ASSERT(
        out_size == signal_size[i + 1] ||
        out_size == (signal_size[i + 1] / 2) + 1);
  }

  batched_sizes[0] = batch_size;
  DimVector batched_out_sizes(batched_sizes.begin(), batched_sizes.end());

  for (size_t i = 0; i < dim.size(); ++i) {
    batched_out_sizes[i + 1] = out_sizes[dim[i]];
  }

  out.resize_(batched_out_sizes, MemoryFormat::Contiguous);

  // run the FFT
  _fft_with_size(
      out,
      input,
      signal_ndim,
      input.is_complex(),
      out.is_complex(),
      !forward,
      signal_size,
      onesided);

  // Inplace reshaping to original batch shape and inverting the dimension
  // permutation
  DimVector out_strides(ndim);
  int64_t batch_numel = 1;

  for (int64_t i = batch_dims - 1; i >= 0; --i) {
    out_strides[dim_permute[i]] = batch_numel * out.strides()[0];
    batch_numel *= out_sizes[dim_permute[i]];
  }

  for (int64_t i = batch_dims; i < ndim; ++i) {
    out_strides[dim_permute[i]] = out.strides()[1 + (i - batch_dims)];
  }

  out.as_strided_(out_sizes, out_strides, out.storage_offset());

  return out;
}

double _dft_scale(
    IntArrayRef dim,
    IntArrayRef input_sizes,
    IntArrayRef out_sizes,
    int64_t normalization) {
  const auto norm = static_cast<fft_norm_mode>(normalization);
  double double_scale = 1.0;

  if (norm == fft_norm_mode::none) {
    return double_scale;
  }

  const int64_t signal_ndim = dim.size();
  int64_t signal_numel = 1;

  for (int64_t i = 0; i < signal_ndim; ++i) {
    auto in_size = input_sizes[dim[i]];
    auto out_size = out_sizes[dim[i]];
    auto signal_size = std::max(in_size, out_size);

    signal_numel *= signal_size;
    TORCH_INTERNAL_ASSERT(
        in_size == signal_size || in_size == (signal_size / 2) + 1);
    TORCH_INTERNAL_ASSERT(
        out_size == signal_size || out_size == (signal_size / 2) + 1);
  }

  if (norm == fft_norm_mode::by_root_n) {
    double_scale = 1.0 / std::sqrt(signal_numel);
  } else {
    double_scale = 1.0 / static_cast<double>(signal_numel);
  }

  return double_scale;
}

const Tensor& _fft_apply_normalization(
    const Tensor& self,
    int64_t normalization,
    IntArrayRef sizes,
    IntArrayRef dims) {
  auto scale = _dft_scale(dims, sizes, self.sizes(), normalization);
  return (scale == 1.0) ? self : self.mul_(scale);
}

Tensor& _fft_apply_normalization_out(
    Tensor& out,
    const Tensor& self,
    int64_t normalization,
    IntArrayRef sizes,
    IntArrayRef dims) {
  auto scale = _dft_scale(dims, sizes, self.sizes(), normalization);
  return at::mul_out(out, self, c10::scalar_to_tensor(scale));
}

} // namespace impl

Tensor _fft_c2c_mkl(
    const Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    bool forward) {
  if (dim.empty()) {
    return self.clone();
  }

  auto sorted_dims = impl::_sort_dims(self, dim);
  auto out_sizes = self.sizes();
  auto out = at::empty(out_sizes, self.options());
  auto input_sizes = self.sizes();
  auto working_tensor = self;

  while (!sorted_dims.empty()) {
    const auto max_dims =
        std::min(static_cast<size_t>(impl::mkl_max_ndim), sorted_dims.size());
    auto fft_dims =
        IntArrayRef(sorted_dims).slice(sorted_dims.size() - max_dims, max_dims);

    impl::_exec_fft(
        out,
        working_tensor,
        out_sizes,
        fft_dims,
        /*onesided=*/false,
        forward);

    sorted_dims.resize(sorted_dims.size() - max_dims);

    if (sorted_dims.empty()) {
      break;
    }

    sorted_dims = impl::_sort_dims(self, sorted_dims);

    if (working_tensor.is_same(self)) {
      working_tensor = std::move(out);
      out = at::empty(out_sizes, self.options());
    } else {
      std::swap(out, working_tensor);
    }
  }

  return impl::_fft_apply_normalization(out, normalization, input_sizes, dim);
}

Tensor& _fft_c2c_mkl_out(
    const Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    bool forward,
    Tensor& out) {
  auto result = _fft_c2c_mkl(
      self, dim, static_cast<int64_t>(fft_norm_mode::none), forward);
  at::native::resize_output(out, result.sizes());
  return impl::_fft_apply_normalization_out(
      out, result, normalization, result.sizes(), dim);
}

} // namespace at::native::xpu
