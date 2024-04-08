#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/Activation.h>
#include <ATen/native/CanUse32BitIndexMath.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/xpu/XPUGeneratorImpl.h>
#include <aten/sycl/DistributionTemplates.h>
#include <aten/sycl/Loops.h>
#include <aten/sycl/MemoryAccessUtils.h>
#include <aten/sycl/TensorInfo.h>

namespace at {
namespace native {
namespace xpu {

namespace {

using namespace at::native;
using namespace at::xpu::detail;

const int UNROLL = 4;

template <
    typename scalar_t,
    typename accscalar_t,
    typename IndexType,
    int ADims,
    int VEC,
    typename mask_t>
struct FusedDropoutVecFunctor {
  void operator()(sycl::nd_item<1> item) const {
    // make sure we don't break assumption that we can't have > 4 elements /
    // thread
    static_assert(VEC <= 4, "Value of VEC must be in [2, 4]");

    using LoadT = memory::aligned_vector<scalar_t, VEC>;
    using MaskLoadT = memory::aligned_vector<mask_t, VEC>;

    auto seeds = philox_unpack(philox_args);
    IndexType idx = item.get_global_linear_id();
    randStatePhilox4_32_10_t state;
    rand_init(std::get<0>(seeds), idx, std::get<1>(seeds), &state);

    // Helps align the total number of times rand_uniform4 is called by each
    // thread for the same totalElements in the vec=2 and vec=4 cases.
    bool gridxvec_loop_state = 0;
    accscalar_t scale = 1.0 / p;

    float4 rand;

    // Note: Vectorized loads means we'll stride each thread by an additional
    // VEC factor, as we'll load VEC elements at a time
    IndexType full_tile_work_size =
        item.get_group_range(0) * item.get_local_range(0) * VEC;
    for (IndexType linearIndex = idx * VEC; linearIndex < totalElements;
         linearIndex += full_tile_work_size) {
      // local storage
      scalar_t src[VEC];
      // We'll use this to actually cause vectorized loads later
      LoadT* value = reinterpret_cast<LoadT*>(&src);

      // rand_uniform_double was pure evil anyway, not doing what it promises,
      // and there's nothing for halfs, so generate float for everything
      //  Note: need a new set of random values per 4 elements -- we'll handle
      //  VEC elements in this thread, so need ceil(VEC / 4) sets of rand.
      if ((VEC == 4) || (gridxvec_loop_state == 0)) {
        rand = rand_uniform4(&state);
      } else {
        // sets up the last two values we generated last iteration to be used
        // this iteration.
        rand.x = rand.z;
        rand.y = rand.w;
        gridxvec_loop_state ^= 1;
      }

      rand.x = rand.x < p;
      rand.y = rand.y < p;
      if (VEC == 4) {
        rand.z = rand.z < p;
        rand.w = rand.w < p;
      }

      // Note: We explicitly check for is_contiguous() before launching the
      // vectorized kernel and replace IndexToOffset call with linearIndex to
      // allow vectorization of NHWC (or other) ordering. Single vectorized load
      *value = *reinterpret_cast<const LoadT*>(&a.data[linearIndex]);

      scalar_t r[VEC];
      mask_t mask[VEC];

// Perform the actual computation
#pragma unroll
      for (int ii = 0; ii < VEC; ii++) {
        r[ii] = src[ii] * (&rand.x)[ii] * scale;
        mask[ii] = (mask_t)(&rand.x)[ii];
      }
      // Vectorized writes for both mask & result
      *(reinterpret_cast<LoadT*>(&b.data[linearIndex])) =
          *reinterpret_cast<LoadT*>(&r[0]);
      *(reinterpret_cast<MaskLoadT*>(&c.data[linearIndex])) =
          *reinterpret_cast<MaskLoadT*>(&mask[0]);
    }
  }
  FusedDropoutVecFunctor(
      TensorInfo<const scalar_t, IndexType> a,
      TensorInfo<scalar_t, IndexType> b,
      TensorInfo<mask_t, IndexType> c,
      IndexType totalElements,
      accscalar_t p,
      PhiloxState philox_args)
      : a(a),
        b(b),
        c(c),
        totalElements(totalElements),
        p(p),
        philox_args(philox_args) {}

 private:
  TensorInfo<const scalar_t, IndexType> a;
  TensorInfo<scalar_t, IndexType> b;
  TensorInfo<mask_t, IndexType> c;
  IndexType totalElements;
  accscalar_t p;
  PhiloxState philox_args;
};

template <
    typename scalar_t,
    typename accscalar_t,
    typename IndexType,
    int ADims,
    int BDims,
    typename mask_t>
struct FusedDropoutUnrollFunctor {
  void operator()(sycl::nd_item<1> item) const {
    auto seeds = philox_unpack(philox_args);
    IndexType idx = item.get_global_linear_id();
    randStatePhilox4_32_10_t state;
    rand_init(std::get<0>(seeds), idx, std::get<1>(seeds), &state);
    accscalar_t scale = 1.0 / p;
    IndexType group_work_size =
        item.get_group_range(0) * item.get_local_range(0);
    IndexType full_tile_work_size = group_work_size * UNROLL;
    IndexType rounded_size =
        ((totalElements - 1) / full_tile_work_size + 1) * full_tile_work_size;

    for (IndexType linearIndex = idx; linearIndex < rounded_size;
         linearIndex += full_tile_work_size) {
      // rand_uniform_double was pure evil anyway, not doing what it promises,
      // and there's nothing for halfs, so generate float for everything
      float4 rand = rand_uniform4(&state);
      scalar_t src[UNROLL];
      rand.x = rand.x < p;
      rand.y = rand.y < p;
      rand.z = rand.z < p;
      rand.w = rand.w < p;
      for (int ii = 0; ii < UNROLL; ii++) {
        IndexType li = linearIndex + group_work_size * ii;
        if (li < totalElements) {
          // Convert `linearIndex` into an offset of `a`
          const IndexType aOffset =
              IndexToOffset<const scalar_t, IndexType, ADims>::get(li, a);
          src[ii] = a.data[aOffset];
        }
      }
      for (int ii = 0; ii < UNROLL; ii++) {
        IndexType li = linearIndex + group_work_size * ii;
        if (li < totalElements) {
          // Convert `linearIndex` into an offset of `b`
          const IndexType bOffset =
              IndexToOffset<scalar_t, IndexType, BDims>::get(li, b);
          b.data[bOffset] = src[ii] * (&rand.x)[ii] * scale;
          c.data[bOffset] = (mask_t)(&rand.x)[ii];
        }
      }
    }
  }
  FusedDropoutUnrollFunctor(
      TensorInfo<const scalar_t, IndexType> a,
      TensorInfo<scalar_t, IndexType> b,
      TensorInfo<mask_t, IndexType> c,
      IndexType totalElements,
      accscalar_t p,
      PhiloxState philox_args)
      : a(a),
        b(b),
        c(c),
        totalElements(totalElements),
        p(p),
        philox_args(philox_args) {}

 private:
  TensorInfo<const scalar_t, IndexType> a;
  TensorInfo<scalar_t, IndexType> b;
  TensorInfo<mask_t, IndexType> c;
  IndexType totalElements;
  accscalar_t p;
  PhiloxState philox_args;
};

template <typename scalar_t, typename accscalar_t, typename mask_t>
struct MaskedScaleKernelFunctor {
  scalar_t operator()(const scalar_t src_val, const mask_t mask_val) const {
    return (float)mask_val * src_val * scale_;
  }
  MaskedScaleKernelFunctor(accscalar_t scale) : scale_(scale) {}

 private:
  accscalar_t scale_;
};

template <typename mask_t, typename scalar_t, typename accscalar_t>
void masked_scale_kernel(
    at::Tensor& ret,
    const at::Tensor& src,
    const at::Tensor& mask,
    accscalar_t scale) {
  auto iter = at::TensorIteratorConfig()
                  .check_all_same_dtype(false)
                  .add_output(ret)
                  .add_const_input(src)
                  .add_const_input(mask)
                  .build();

  auto caller = MaskedScaleKernelFunctor<scalar_t, accscalar_t, mask_t>(scale);
  gpu_kernel(iter, caller);
}

template <typename scalar_t>
int get_vector_size(at::Tensor self, at::Tensor ret, at::Tensor mask) {
  int vec_size = 4;
  // get the vector size
  if (!self.is_non_overlapping_and_dense() ||
      !ret.is_non_overlapping_and_dense() ||
      !mask.is_non_overlapping_and_dense()) {
    vec_size = 1;
  } else {
    vec_size =
        memory::can_vectorize_up_to<scalar_t>((char*)self.const_data_ptr());
  }

  // check that we'd have no remainders - prefer a smaller vector size with no
  // remainders over a larger vector and remainder.
  bool can_vectorize = true;
  do {
    can_vectorize = self.numel() % vec_size == 0 &&
        ret.numel() % vec_size == 0 && mask.numel() % vec_size == 0;
    if (!can_vectorize)
      vec_size /= 2;
  } while (vec_size > 1 && !can_vectorize);
  return can_vectorize ? vec_size : 1;
}

template <typename index_type, typename mask_t>
inline void launcher(
    const Tensor& self,
    Tensor& ret,
    Tensor& mask,
    double p,
    const int64_t nelem,
    const PhiloxState rng_engine_inputs,
    uint32_t num_groups,
    uint32_t group_size) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "fused_dropout",
      [&] {
        using accscalar_t = acc_type<scalar_t, true>;
        accscalar_t pa = (accscalar_t)(p);
        auto self_info = getTensorInfo<const scalar_t, index_type>(self);
        auto ret_info = getTensorInfo<scalar_t, index_type>(ret);
        auto mask_info = getTensorInfo<mask_t, index_type>(mask);
        self_info.collapseDims();
        ret_info.collapseDims();
        mask_info.collapseDims(); // ret and mask are collapsed to 1d
                                  // contiguous tensor

        int vec_size = get_vector_size<scalar_t>(self, ret, mask);

        if (vec_size > 1) {
          switch (vec_size) {
            case 4: {
              auto caller = FusedDropoutVecFunctor<
                  scalar_t,
                  accscalar_t,
                  index_type,
                  1,
                  4,
                  mask_t>(
                  self_info, ret_info, mask_info, nelem, pa, rng_engine_inputs);
              sycl_kernel_submit(
                  num_groups * group_size,
                  group_size,
                  getCurrentSYCLQueue(),
                  caller);
            } break;
            case 2: {
              auto caller = FusedDropoutVecFunctor<
                  scalar_t,
                  accscalar_t,
                  index_type,
                  1,
                  2,
                  mask_t>(
                  self_info, ret_info, mask_info, nelem, pa, rng_engine_inputs);
              sycl_kernel_submit(
                  num_groups * group_size,
                  group_size,
                  getCurrentSYCLQueue(),
                  caller);
            } break;
          }
        } else {
          switch (self_info.dims) {
            case 1: {
              auto caller = FusedDropoutUnrollFunctor<
                  scalar_t,
                  accscalar_t,
                  index_type,
                  1,
                  1,
                  mask_t>(
                  self_info, ret_info, mask_info, nelem, pa, rng_engine_inputs);
              sycl_kernel_submit(
                  num_groups * group_size,
                  group_size,
                  getCurrentSYCLQueue(),
                  caller);
            } break;
            default:
              if (!self.is_contiguous() && ret.is_contiguous() &&
                  mask.is_contiguous()) {
                auto caller = FusedDropoutUnrollFunctor<
                    scalar_t,
                    accscalar_t,
                    index_type,
                    -1,
                    1,
                    mask_t>(
                    self_info,
                    ret_info,
                    mask_info,
                    nelem,
                    pa,
                    rng_engine_inputs);
                sycl_kernel_submit(
                    num_groups * group_size,
                    group_size,
                    getCurrentSYCLQueue(),
                    caller);
              } else {
                auto caller = FusedDropoutUnrollFunctor<
                    scalar_t,
                    accscalar_t,
                    index_type,
                    -1,
                    -1,
                    mask_t>(
                    self_info,
                    ret_info,
                    mask_info,
                    nelem,
                    pa,
                    rng_engine_inputs);
                sycl_kernel_submit(
                    num_groups * group_size,
                    group_size,
                    getCurrentSYCLQueue(),
                    caller);
              }
          }
        }
      });
}

} // namespace

template <typename mask_t>
std::tuple<Tensor, Tensor> dropout(
    XPUGeneratorImpl* gen,
    const Tensor& self,
    double p) {
  Tensor mask = at::empty_like(
      self, self.options().dtype(c10::CppTypeToScalarType<mask_t>::value));
  const int64_t nelem = self.numel();
  // empty tensors should not get here, but just in case, avoid FPE
  // non-training shot-cut
  if (nelem == 0)
    return std::tuple<Tensor, Tensor>(self.clone(), mask);

  Tensor ret = at::empty_like(self);

  auto group_size = std::min(
      syclGpuHWThreadsPerEU() * syclMaxSubGroupSize(), syclMaxWorkGroupSize());
  auto num_groups = (nelem + group_size - 1) / group_size;
  auto hw_max_groups = syclMaxWorkItemsPerTile() / group_size;
  num_groups = num_groups > hw_max_groups ? hw_max_groups : num_groups;

  // number of times random will be generated per thread, to offset philox
  // counter in thc random state
  int64_t counter_offset =
      ((nelem - 1) / (group_size * num_groups * UNROLL) + 1) * UNROLL;

  std::pair<uint64_t, uint64_t> rng_engine_inputs_;
  {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs_ = gen->philox_engine_inputs(counter_offset);
  }
  PhiloxState rng_engine_inputs(
      std::get<0>(rng_engine_inputs_), std::get<1>(rng_engine_inputs_));

  if (canUse32BitIndexMath(self)) {
    launcher<unsigned int, mask_t>(
        self, ret, mask, p, nelem, rng_engine_inputs, num_groups, group_size);
  } else {
    launcher<uint64_t, mask_t>(
        self, ret, mask, p, nelem, rng_engine_inputs, num_groups, group_size);
  }
  return std::tuple<Tensor, Tensor>(ret, mask);
}

std::tuple<Tensor, Tensor> native_dropout(
    const Tensor& self,
    double p,
    c10::optional<bool> train) {
  // short-cut for train == false
  if (train.has_value() && !train.value()) {
    return std::make_tuple(
        self.clone(),
        at::ones_like(
            self, self.options().dtype(c10::CppTypeToScalarType<bool>::value)));
  }
  // short-cut
  if (p == 1) {
    auto ret = at::zeros_like(self);
    auto mask = at::zeros_like(
        self, self.options().dtype(c10::CppTypeToScalarType<bool>::value));
    return std::tuple<Tensor, Tensor>(ret, mask);
  }
  auto gen = get_generator_or_default<at::XPUGeneratorImpl>(
      c10::nullopt, at::xpu::detail::getDefaultXPUGenerator());
  double p1m = 1. - p;
  return dropout<bool>(gen, self, p1m);
}

std::tuple<Tensor, Tensor> fused_dropout(
    const Tensor& self,
    double p,
    c10::optional<Generator> gen_) {
  auto gen = get_generator_or_default<at::XPUGeneratorImpl>(
      gen_, at::xpu::detail::getDefaultXPUGenerator());
  return dropout<uint8_t>(gen, self, p);
}

template <typename mask_t>
Tensor dropout_backward(const Tensor& grad, const Tensor& mask, double scale) {
  Tensor ret = at::empty_like(grad, grad.suggest_memory_format());
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      ret.scalar_type(),
      "masked_scale",
      [&] {
        using accscalar_t = acc_type<scalar_t, true>;
        masked_scale_kernel<mask_t, scalar_t>(
            ret, grad, mask, (accscalar_t)scale);
      });
  return ret;
}

Tensor native_dropout_backward(
    const Tensor& grad,
    const Tensor& mask,
    double scale) {
  TORCH_CHECK(
      mask.scalar_type() == at::ScalarType::Bool,
      "Mask should be Bool Scalar Type",
      mask.scalar_type());
  return dropout_backward<bool>(grad, mask, scale);
}

Tensor masked_scale(const Tensor& self, const Tensor& mask, double scale) {
  TORCH_CHECK(
      mask.scalar_type() == at::ScalarType::Byte,
      "mask should be torch.uint8 dtype");
  return dropout_backward<uint8_t>(self, mask, scale);
}

} // namespace xpu
} // namespace native
} // namespace at
