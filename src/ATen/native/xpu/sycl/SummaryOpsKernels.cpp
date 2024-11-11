#pragma clang diagnostic push
#pragma GCC diagnostic push
// Avoid SYCL compiler return-type error
#pragma clang diagnostic ignored "-Wreturn-type"
#pragma GCC diagnostic ignored "-Wreturn-type"

#include <ATen/AccumulateType.h>
#include <ATen/native/xpu/sycl/Atomics.h>
#include <comm/Runtime.h>
#include <comm/SYCLHelpers.h>
#include <comm/TensorInfo.h>

#include <ATen/native/xpu/sycl/SummaryOpsKernels.h>

namespace at::native::xpu {
using namespace at::native;
using namespace at::xpu::detail;

template <typename input_t, typename IndexType>
static IndexType get_bin(
    input_t b_val,
    at::acc_type_device<input_t, kXPU> min_value,
    at::acc_type_device<input_t, kXPU> max_value,
    int nbins) {
  IndexType bin = (int)((b_val - min_value) * nbins / (max_value - min_value));
  // (only applicable for histc)
  // while each bin is inclusive at the lower end and exclusive at the higher,
  // i.e. [start, end)
  // the last bin is inclusive at both, i.e. [start, end], in order to include
  // max_value if exists
  // therefore when bin == nbins, adjust bin to the last bin
  if (bin == nbins) {
    bin -= 1;
  }

  return bin;
}

template <
    typename output_t,
    typename input_t,
    typename IndexType,
    int ADims,
    bool has_weight,
    typename Op>
struct Histogram1DKernelFunctor {
  void operator()(sycl::item<1> item_id) const {
    auto out_ptr = a_.data;
    auto in_ptr = b_.data;
    auto weight_ptr = c_.data;

    auto linear_index = item_id.get_id(0);
    // Convert `linear_index` into an offset of `b`
    const IndexType b_offset =
        IndexToOffset<const input_t, IndexType>::get(linear_index, b_);
    const auto b_val = in_ptr[b_offset];
    if (b_val >= min_value_ && b_val <= max_value_) {
      // Use value at `b` as an offset of `a`
      const IndexType bin =
          get_bin<input_t, IndexType>(b_val, min_value_, max_value_, nbins_);
      const IndexType a_offset =
          IndexToOffset<output_t, IndexType>::get(bin, a_);
      atomicAdd(
          (sycl_global_ptr<output_t>)&out_ptr[a_offset],
          get_op_(weight_ptr, linear_index));
    }
  }
  Histogram1DKernelFunctor(
      TensorInfo<output_t, IndexType> a,
      TensorInfo<const input_t, IndexType> b,
      TensorInfo<output_t, IndexType> c,
      int nbins,
      at::acc_type_device<input_t, kXPU> minvalue,
      at::acc_type_device<input_t, kXPU> maxvalue,
      IndexType totalElements,
      Op get_op)
      : a_(a),
        b_(b),
        c_(c),
        nbins_(nbins),
        min_value_(minvalue),
        max_value_(maxvalue),
        total_elements_(totalElements),
        get_op_(get_op) {}

 private:
  TensorInfo<output_t, IndexType> a_;
  TensorInfo<const input_t, IndexType> b_;
  TensorInfo<output_t, IndexType> c_;
  int nbins_;
  at::acc_type_device<input_t, kXPU> min_value_;
  at::acc_type_device<input_t, kXPU> max_value_;
  IndexType total_elements_;
  Op get_op_;
};

/*
  Kernel for computing the histogram of the input.
 */
template <
    typename output_t,
    typename input_t,
    typename IndexType,
    int ADims,
    bool has_weight,
    typename Op>
void histogram_1d_kernel(
    TensorInfo<output_t, IndexType> a, /* output */
    TensorInfo<const input_t, IndexType> b, /* input */
    TensorInfo<output_t, IndexType> c, /* weight */
    int nbins,
    at::acc_type_device<input_t, kXPU> min_value,
    at::acc_type_device<input_t, kXPU> max_value,
    IndexType total_elements,
    Op get_op) {
  auto& sycl_queue = at::xpu::getCurrentSYCLQueue();

  Histogram1DKernelFunctor<output_t, input_t, IndexType, ADims, has_weight, Op>
      kfn(a, b, c, nbins, min_value, max_value, total_elements, get_op);

  sycl_kernel_submit(::sycl::range<1>(total_elements), sycl_queue, kfn);
}

#define HANDLE_CASE(WEIGHTS_OP, WITH_WEIGHT)                         \
  histogram_1d_kernel<output_t, input_t, IndexType, 1, WITH_WEIGHT>( \
      a_info,                                                        \
      b_info,                                                        \
      c_info,                                                        \
      nbins,                                                         \
      min_value,                                                     \
      max_value,                                                     \
      total_elements,                                                \
      WEIGHTS_OP);

template <typename output_t, typename index_type, typename info_t>
struct IndexingFunctor {
  auto operator()(output_t* c_ptr, index_type c_index) const {
    const index_type c_offset =
        IndexToOffset<output_t, index_type>::get(c_index, c_info);
    return c_ptr[c_offset];
  }

  IndexingFunctor(info_t c_info) : c_info(c_info) {}

 private:
  info_t c_info;
};

template <typename output_t, typename index_type>
struct DummyIndexingFunctor {
  auto operator()(output_t*, index_type) const {
    return static_cast<output_t>(1);
  }
};
template <typename output_t, typename input_t, bool has_weights>
void tensor_histogram(
    at::Tensor a, /* output */
    at::Tensor b, /* input */
    at::Tensor c, /* weights(optional) */
    int64_t nbins,
    at::acc_type_device<input_t, kXPU> min_value,
    at::acc_type_device<input_t, kXPU> max_value) {
  checkBackend("tensor_histogram", {a, b}, Backend::XPU);
  if (has_weights) {
    checkBackend("tensor_histogram", {c}, Backend::XPU);
  }
  auto total_elements = b.numel();
  if (total_elements == 0) {
    return;
  }

  using IndexType = int64_t;
  auto a_info = getTensorInfo<output_t, IndexType>(a);
  auto b_info = getTensorInfo<const input_t, IndexType>(b);
  if (has_weights) {
    auto c_info = getTensorInfo<output_t, IndexType>(c);
    const IndexingFunctor<output_t, IndexType, decltype(c_info)> get_weights_op(
        c_info);
    HANDLE_CASE(get_weights_op, true);
  } else {
    TensorInfo<output_t, IndexType> c_info;
    // set the dummy cinfo with the ptr to the output
    c_info.data = a_info.data;
    static const DummyIndexingFunctor<output_t, IndexType> get_dummyOp;
    HANDLE_CASE(get_dummyOp, false);
  }

  return;
}

template <typename input_t>
Tensor _histc_template(
    const Tensor& self,
    int64_t nbins,
    at::acc_type_device<input_t, kXPU> min,
    at::acc_type_device<input_t, kXPU> max) {
  if (nbins <= 0) {
    AT_ERROR("bins must be > 0");
  }
  Tensor output = at::zeros(
      {nbins},
      self.scalar_type(),
      std::nullopt /* layout */,
      DeviceType::XPU,
      std::nullopt /* pin_memory */);

  using bounds_t = at::acc_type_device<input_t, kXPU>;
  bounds_t minvalue = min;
  bounds_t maxvalue = max;

  if (min == max && self.numel() > 0) {
    minvalue = *self.min().cpu().const_data_ptr<input_t>();
    maxvalue = *self.max().cpu().const_data_ptr<input_t>();
  }
  if (minvalue == maxvalue) {
    minvalue = minvalue - 1;
    maxvalue = maxvalue + 1;
  }

  TORCH_CHECK(
      !(std::isinf((float)minvalue) || std::isinf((float)maxvalue) ||
        std::isnan((float)minvalue) || std::isnan((float)maxvalue)),
      "range of [",
      minvalue,
      ", ",
      maxvalue,
      "] is not finite");

  TORCH_CHECK(minvalue < maxvalue, "max must be larger than min");

  tensor_histogram<input_t, input_t, false>(
      output, self, Tensor(), nbins, minvalue, maxvalue);
  return output;
}

Tensor _histc_kernel(
    const Tensor& self,
    int64_t nbins,
    const Scalar& min,
    const Scalar& max) {
  if (self.scalar_type() == ScalarType::Half) {
    AT_ERROR("HalfTensor is not supported");
  }
  return AT_DISPATCH_ALL_TYPES(self.scalar_type(), "_histc_xpu", [&] {
    using bounds_t = at::acc_type_device<scalar_t, kXPU>;
    return _histc_template<scalar_t>(
        self, nbins, min.to<bounds_t>(), max.to<bounds_t>());
  });
}

template <typename input_t, typename weights_t>
Tensor bincount_template(
    const Tensor& self,
    const Tensor& weights,
    int64_t minlength) {
  if (minlength < 0) {
    TORCH_CHECK(0, "minlength should be >= 0");
  }
  if (self.dim() == 1 && self.numel() == 0) {
    return at::zeros({minlength}, device(kXPU).dtype(kLong));
  }
  if (self.dim() != 1 ||
      (!std::is_same<input_t, uint8_t>::value &&
       *self.min().cpu().data_ptr<input_t>() < 0)) {
    TORCH_CHECK(0, "bincount only supports 1-d non-negative integral inputs.");
  }

  bool has_weights = weights.defined();
  if (has_weights && (weights.dim() != 1 || weights.size(0) != self.size(0))) {
    TORCH_CHECK(0, "weights should be 1-d and have the same length as input");
  }

  const int64_t nbins =
      std::max(self.max().item<input_t>() + (int64_t)1, minlength);
  using bounds_t = at::acc_type_device<input_t, kXPU>;
  const bounds_t min_value = 0;
  const bounds_t max_value = nbins;
  // alloc output counter on GPU
  Tensor output;
  if (has_weights) {
    output = at::zeros(
        {nbins},
        optTypeMetaToScalarType(weights.options().dtype_opt()),
        weights.options().layout_opt(),
        weights.options().device_opt(),
        weights.options().pinned_memory_opt());
    tensor_histogram<weights_t, input_t, true>(
        output, self, weights, nbins, min_value, max_value);
  } else {
    output = at::zeros(
        {nbins},
        kLong,
        c10::nullopt /* layout */,
        DeviceType::XPU,
        c10::nullopt /* pin_memory */);
    tensor_histogram<
        typename c10::impl::ScalarTypeToCPPType<kLong>::type,
        input_t,
        false>(output, self, weights, nbins, min_value, max_value);
  }
  return output;
}

Tensor bincount_kernel(
    const Tensor& self,
    const Tensor& weights,
    int64_t minlength) {
  return AT_DISPATCH_INTEGRAL_TYPES(self.scalar_type(), "bincount_xpu", [&] {
    const auto scalar = weights.scalar_type();
    if (scalar == ScalarType::Undefined || scalar == ScalarType::Float)
      return bincount_template<scalar_t, float>(self, weights, minlength);
    return bincount_template<scalar_t, double>(
        self, weights.to(kDouble), minlength);
  });
}
} // namespace at::native::xpu

#pragma GCC diagnostic pop
#pragma clang diagnostic pop
