#pragma clang diagnostic push
#pragma GCC diagnostic push
// Avoid SYCL compiler return-type error
#pragma clang diagnostic ignored "-Wreturn-type"
#pragma GCC diagnostic ignored "-Wreturn-type"

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <aten/sycl/Atomics.h>
#include <comm/Runtime.h>
#include <comm/SYCLContext.h>
#include <comm/SYCLHelpers.h>
#include <comm/TensorInfo.h>

namespace at::native::xpu {
using namespace at::native;
using namespace at::xpu::detail;

template <typename input_t, typename IndexType>
static IndexType getBin(
    input_t bVal,
    at::acc_type<input_t, true> minvalue,
    at::acc_type<input_t, true> maxvalue,
    int nbins) {
  IndexType bin = (int)((bVal - minvalue) * nbins / (maxvalue - minvalue));
  // (only applicable for histc)
  // while each bin is inclusive at the lower end and exclusive at the higher,
  // i.e. [start, end)
  // the last bin is inclusive at both, i.e. [start, end], in order to include
  // maxvalue if exists
  // therefore when bin == nbins, adjust bin to the last bin
  if (bin == nbins){
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
    auto out_ptr = out_data;
    auto in_ptr = in_data;
    auto weight_ptr = weight_data;

    auto linearIndex = item_id.get_id(0);
    // Convert `linearIndex` into an offset of `b`
    const IndexType bOffset =
        IndexToOffset<input_t, IndexType>::get(linearIndex, b);
    const auto bVal = in_ptr[bOffset];
    if (bVal >= minvalue && bVal <= maxvalue) {
      // Use value at `b` as an offset of `a`
      const IndexType bin =
          getBin<input_t, IndexType>(bVal, minvalue, maxvalue, nbins);
      const IndexType aOffset = IndexToOffset<output_t, IndexType>::get(bin, a);
      atomicAdd(
          (sycl_global_ptr<output_t>) &out_ptr[aOffset],
          getOp(weight_ptr, linearIndex));
    }
  }
  Histogram1DKernelFunctor(
      TensorInfo<output_t, IndexType> a_,
      TensorInfo<input_t, IndexType> b_,
      TensorInfo<output_t, IndexType> c_,
      int nbins_,
      at::acc_type<input_t, true> minvalue_,
      at::acc_type<input_t, true> maxvalue_,
      IndexType totalElements_,
      Op getOp_,
      output_t* out_data_,
      input_t* in_data_,
      output_t* weight_data_)
      : a(a_),
        b(b_),
        c(c_),
        nbins(nbins_),
        minvalue(minvalue_),
        maxvalue(maxvalue_),
        totalElements(totalElements_),
        getOp(getOp_),
        out_data(out_data_),
        in_data(in_data_),
        weight_data(weight_data_) {}

 private:
  TensorInfo<output_t, IndexType> a;
  TensorInfo<input_t, IndexType> b;
  TensorInfo<output_t, IndexType> c;
  int nbins;
  at::acc_type<input_t, true> minvalue;
  at::acc_type<input_t, true> maxvalue;
  IndexType totalElements;
  Op getOp;
  output_t* out_data;
  input_t* in_data;
  output_t* weight_data;
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
void kernelHistogram1D(
    TensorInfo<output_t, IndexType> a, /* output */
    TensorInfo<input_t, IndexType> b, /* input */
    TensorInfo<output_t, IndexType> c, /* weight */
    int nbins,
    at::acc_type<input_t, true> minvalue,
    at::acc_type<input_t, true> maxvalue,
    IndexType totalElements,
    Op getOp) {
  auto& sycl_queue = at::xpu::getCurrentSYCLQueue();;
  auto out_data = a.data;
  auto in_data = b.data;
  auto weight_data = c.data;

  Histogram1DKernelFunctor<
      output_t,
      input_t,
      IndexType,
      ADims,
      has_weight,
      Op>
      kfn(a,
          b,
          c,
          nbins,
          minvalue,
          maxvalue,
          totalElements,
          getOp,
          out_data,
          in_data,
          weight_data);

  sycl_kernel_submit(::sycl::range<1>(totalElements), sycl_queue, kfn);
}

#define HANDLE_CASE(WEIGHTS_OP, WITH_WEIGHT)                       \
  kernelHistogram1D<output_t, input_t, IndexType, 1, WITH_WEIGHT>( \
      aInfo,                                                       \
      bInfo,                                                       \
      cInfo,                                                       \
      nbins,                                                       \
      minvalue,                                                    \
      maxvalue,                                                    \
      totalElements,                                               \
      WEIGHTS_OP);

template <typename output_t, typename IndexType, typename info_t>
struct xpu_tensor_histogram_functor {
  auto operator()(output_t* cPtr, IndexType cIndex) const {
    const IndexType cOffset =
        IndexToOffset<output_t, IndexType>::get(cIndex, cInfo);
    return cPtr[cOffset];
  }

  xpu_tensor_histogram_functor(info_t cInfo) : cInfo(cInfo) {}

 private:
  info_t cInfo;
};

template <typename output_t, typename IndexType>
struct xpu_tensor_histogram_functor_2 {
  auto operator()(output_t*, IndexType) const {
    return static_cast<output_t>(1);
  }
};
template <typename output_t, typename input_t, bool HasWeights>
void xpu_tensor_histogram(
  at::Tensor a, /* output */
  at::Tensor b, /* input */
  at::Tensor c, /* weights(optional) */
  int64_t nbins,
  at::acc_type<input_t, true> minvalue,
  at::acc_type<input_t, true> maxvalue) {
  checkBackend("xpu_tensor_histogram", {a, b}, Backend::XPU);
  if (HasWeights) {
    checkBackend("xpu_tensor_histogram", {c}, Backend::XPU);
  }
  auto totalElements = b.numel();
  if (totalElements == 0) {
    return ;
  }

  using IndexType = int64_t;
  auto aInfo = getTensorInfo<output_t, IndexType>(a);
  auto bInfo = getTensorInfo<input_t, IndexType>(b);
  if (HasWeights) {
    auto cInfo = getTensorInfo<output_t, IndexType>(c);
    const xpu_tensor_histogram_functor<output_t, IndexType, decltype(cInfo)>
        getWeightsOp(cInfo);
    HANDLE_CASE(getWeightsOp, true);
  } else {
    TensorInfo<output_t, IndexType> cInfo;
    // set the dummy cinfo with the ptr to the output
    cInfo.data = aInfo.data;
    static const xpu_tensor_histogram_functor_2<output_t, IndexType>
        getDummyOp;
    HANDLE_CASE(getDummyOp, false);
  }

  return ;
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
    TORCH_CHECK(0, "input and weights should have the same length");
  }

  const int64_t nbins =
      std::max(self.max().item<input_t>() + (int64_t)1, minlength);
  using bounds_t = at::acc_type<input_t, true>;
  const bounds_t minvalue = 0;
  const bounds_t maxvalue = nbins;
  // alloc output counter on GPU
  Tensor output;
  if (has_weights) {
    output = at::zeros(
        {nbins},
        optTypeMetaToScalarType(weights.options().dtype_opt()),
        weights.options().layout_opt(),
        weights.options().device_opt(),
        weights.options().pinned_memory_opt());
        xpu_tensor_histogram<weights_t, input_t, true>(
        output, self, weights, nbins, minvalue, maxvalue);
  } else {
    output = at::zeros(
        {nbins},
        kLong,
        c10::nullopt /* layout */,
        DeviceType::XPU,
        c10::nullopt /* pin_memory */);
    xpu_tensor_histogram<
        typename c10::impl::ScalarTypeToCPPType<kLong>::type,
        input_t,
        false>(output, self, weights, nbins, minvalue, maxvalue);
  }
  return output;
}

Tensor bincount_kernel(
const Tensor& self,
const c10::optional<Tensor>& weights_opt,
int64_t minlength) {
  c10::MaybeOwned<Tensor> weights_maybe_owned = at::borrow_from_optional_tensor(weights_opt);
  const Tensor& weights = *weights_maybe_owned;

  if (weights_opt.has_value()) {
      // See Note [Writing Nondeterministic Operations]
      // Nondeterministic if weights are given, because of floating point
      // atomicAdd usage
      globalContext().alertNotDeterministic("_bincount_xpu");
  }

  return AT_DISPATCH_INTEGRAL_TYPES(
    self.scalar_type(), "bincount_kernel", [&] {
      const auto scalar = weights.scalar_type();
      if (scalar == ScalarType::Undefined || scalar == ScalarType::Float)
        return bincount_template<scalar_t, float>(
            self, weights, minlength);
      return bincount_template<scalar_t, double>(
          self, weights.to(kDouble), minlength);
    });
}
} // namespace at::native::xpu



#pragma GCC diagnostic pop
#pragma clang diagnostic pop