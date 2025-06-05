#include <ATen/WrapDimUtils.h>
#include <ATen/native/SpectralOpsUtils.h>
#include <ATen/native/xpu/sycl/OffsetCalculator.h>
#include <comm/SYCLContext.h>
#include <comm/TensorInfo.h>

namespace at {
namespace native {
namespace xpu {

template <typename index_t>
struct HermitianSymmetryOffsetCalculator {
  using offset_type = at::detail::Array<index_t, 1>;
  using dim_type = std::remove_cv_t<decltype(XPU_MAX_TENSORINFO_DIMS)>;

  dim_type dims;
  at::detail::IntDivider<index_t> sizes_[XPU_MAX_TENSORINFO_DIMS];
  index_t strides_[XPU_MAX_TENSORINFO_DIMS];
  uint32_t mirror_dim_; // bit mask
  static_assert(XPU_MAX_TENSORINFO_DIMS < 32, "Need a bigger mask type");

  HermitianSymmetryOffsetCalculator(
      IntArrayRef sizes,
      IntArrayRef strides,
      IntArrayRef dim,
      const int64_t element_size) {
    TORCH_INTERNAL_ASSERT(sizes.size() == strides.size());
    TORCH_INTERNAL_ASSERT(sizes.size() <= XPU_MAX_TENSORINFO_DIMS);
    dims = sizes.size();

    {
      dim_type i;
      for (i = 0; i < dims; ++i) {
        sizes_[i] = at::detail::IntDivider<index_t>(sizes[i]);
        strides_[i] = strides[i] / element_size;
      }
      for (; i < XPU_MAX_TENSORINFO_DIMS; ++i) {
        sizes_[i] = at::detail::IntDivider<index_t>(1);
        strides_[i] = 0;
      }
    }

    mirror_dim_ = 0;
    for (int64_t i = 0; i < dim.size(); ++i) {
      mirror_dim_ |= (uint32_t{1} << dim[i]);
    }
  }

  offset_type get(index_t linear_idx) const {
    index_t offset = 0;

    for (dim_type dim = 0; dim < dims; ++dim) {
      auto divmod = sizes_[dim].divmod(linear_idx);
      linear_idx = divmod.div;

      if ((mirror_dim_ & (uint32_t{1} << dim)) == 0) {
        offset += divmod.mod * strides_[dim];
      } else if (divmod.mod != 0) {
        offset += (sizes_[dim].divisor - divmod.mod) * strides_[dim];
      }
    }
    offset_type offsets;
    offsets[0] = offset;

    return offsets;
  }
};

template <typename scalar_t, typename inp_calc_t, typename out_calc_t>
struct FFTConjugateCopyKernelFunctor {
  void operator()(sycl::item<1> item_id) const {
    auto in_offset = ic_.get(item_id)[0];
    auto out_offset = oc_.get(item_id)[0];
    out_data_[out_offset] = std::conj(in_data_[in_offset]);
  }

  FFTConjugateCopyKernelFunctor(
      int64_t numel,
      scalar_t* out_data,
      const scalar_t* in_data,
      inp_calc_t ic,
      out_calc_t oc)
      : numel_(numel),
        out_data_(out_data),
        in_data_(in_data),
        ic_(ic),
        oc_(oc) {}

 private:
  int64_t numel_;
  scalar_t* out_data_;
  const scalar_t* in_data_;
  inp_calc_t ic_;
  out_calc_t oc_;
};

template <typename scalar_t, typename inp_calc_t, typename out_calc_t>
void _fft_conjugate_copy_kernel(
    int64_t numel,
    scalar_t* out_data,
    const scalar_t* in_data,
    inp_calc_t ic,
    out_calc_t oc) {
  auto& queue = at::xpu::getCurrentSYCLQueue();
  int thread_num = numel;

  auto ker = FFTConjugateCopyKernelFunctor<scalar_t, inp_calc_t, out_calc_t>(
      numel, out_data, in_data, ic, oc);

  sycl_kernel_submit(sycl::range<1>(thread_num), queue, ker);
}

void _fft_fill_with_conjugate_symmetry_xpu(
    ScalarType dtype,
    IntArrayRef mirror_dims,
    IntArrayRef signal_half_sizes,
    IntArrayRef in_strides,
    const void* in_data,
    IntArrayRef out_strides,
    void* out_data) {
  // Do the actual conjugate mirroring.
  auto* in_strides_ptr = in_strides.data();
  const int ndim = in_strides.size();
  const int64_t element_size = scalarTypeToTypeMeta(dtype).itemsize();

  OffsetCalculator<1, int64_t> input_offset_calculator(
      ndim, signal_half_sizes.data(), &in_strides_ptr, &element_size);
  HermitianSymmetryOffsetCalculator<int64_t> output_offset_calculator(
      signal_half_sizes, out_strides, mirror_dims, element_size);

  const auto numel = c10::multiply_integers(signal_half_sizes);
  AT_DISPATCH_COMPLEX_TYPES(dtype, "_fft_fill_with_conjugate_symmetry_", [&] {
    _fft_conjugate_copy_kernel(
        numel,
        static_cast<scalar_t*>(out_data),
        static_cast<const scalar_t*>(in_data),
        input_offset_calculator,
        output_offset_calculator);
  });
}

} // namespace xpu
} // namespace native
} // namespace at
