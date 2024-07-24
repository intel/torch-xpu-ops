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

    for (dim_type i = 0; i < XPU_MAX_TENSORINFO_DIMS; ++i) {
      if (i < dims) {
        sizes_[i] = at::detail::IntDivider<index_t>(sizes[i]);
        strides_[i] = strides[i] / element_size;
      } else {
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
    auto in_offset = ic.get(item_id)[0];
    auto out_offset = oc.get(item_id)[0];
    out_data[out_offset] = std::conj(in_data[in_offset]);
  }

  FFTConjugateCopyKernelFunctor(
      int64_t numel_,
      scalar_t* out_data_,
      const scalar_t* in_data_,
      inp_calc_t ic_,
      out_calc_t oc_)
      : numel(numel_),
        out_data(out_data_),
        in_data(in_data_),
        ic(ic_),
        oc(oc_) {}

 private:
  int64_t numel;
  scalar_t* out_data;
  const scalar_t* in_data;
  inp_calc_t ic;
  out_calc_t oc;
};

template <typename scalar_t, typename inp_calc_t, typename out_calc_t>
void _fft_conjugate_copy_kernel(
    int64_t numel,
    scalar_t* out_data,
    const scalar_t* in_data,
    inp_calc_t ic,
    out_calc_t oc) {
  printf("Enter _fft_conjugate_copy_kernel\n");
  auto& queue = at::xpu::getCurrentSYCLQueue();
  int thread_num = numel;

  auto ker = FFTConjugateCopyKernelFunctor<scalar_t, inp_calc_t, out_calc_t>(
      numel, out_data, in_data, ic, oc);

  printf("Executing sycl_kernel_submit\n");
  sycl_kernel_submit(sycl::range<1>(thread_num), queue, ker);
  printf("Exit _fft_conjugate_copy_kernel\n");
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

void _fft_fill_with_conjugate_symmetry_(const Tensor& input, IntArrayRef dim_) {
  const auto input_sizes = input.sizes();
  const auto input_strides = input.strides();
  TORCH_CHECK(dim_.size() > 0);
  DimVector dim(dim_.begin(), dim_.end());
  at::maybe_wrap_dims(dim, input_strides.size(), /*wrap_scalars=*/false);

  if (input.numel() == 0 || input_sizes[dim.back()] <= 2) {
    return; // No elements need writing
  }

  // Small dimensions may be treated as batch dims since they don't get mirrored
  dim.erase(
      std::remove_if(
          dim.begin(),
          dim.end(),
          [&](int64_t dim) { return (input_sizes[dim] <= 2); }),
      dim.end());

  // Use TensorIterator to coalesce batch dimensions
  // NOTE: Can't use TensorIterator loops because we need negative strides
  auto iter = TensorIteratorConfig()
                  .add_output(input)
                  .add_input(input)
                  .resize_outputs(false)
                  .declare_static_shape(input_sizes, dim)
                  .build();

  const auto iter_strides = iter.strides(0);
  const auto iter_sizes = iter.shape();
  const auto ndim = static_cast<int64_t>(iter_strides.size() + dim.size());
  DimVector in_strides(ndim), signal_half_sizes(ndim);
  // Take coalesced batch dimensions from TensorIterator
  std::copy(iter_strides.begin(), iter_strides.end(), in_strides.begin());
  std::copy(iter_sizes.begin(), iter_sizes.end(), signal_half_sizes.begin());

  // Take transformed dimensions directly from the input
  const auto element_size = iter.element_size(0);
  for (const auto i : c10::irange(dim.size())) {
    // Convert to byte strides to match TensorIterator
    in_strides[iter_strides.size() + i] = input_strides[dim[i]] * element_size;
    signal_half_sizes[iter_strides.size() + i] = input_sizes[dim[i]];
  }

  // For the last dimension, use negative strides to perform the mirroring
  signal_half_sizes.back() = (input_sizes[dim.back()] - 1) / 2;
  auto out_strides = in_strides;
  out_strides.back() *= -1;

  auto* data_ptr = static_cast<char*>(input.data_ptr());
  const auto* in_data = data_ptr + input_strides[dim.back()] * element_size;
  auto* out_data = data_ptr +
      (input_strides[dim.back()] * (input_sizes[dim.back()] - 1) *
       element_size);

  // Reorder dimensions by stride to maximize data locality
  DimVector dim_permute(ndim);
  std::iota(dim_permute.begin(), dim_permute.end(), 0);
  std::sort(dim_permute.begin(), dim_permute.end(), [&](auto dim1, auto dim2) {
    return in_strides[dim1] < in_strides[dim2];
  });
  DimVector temp(ndim);
  auto apply_permutation = [&](DimVector& vec) {
    // Do permuted index copy into a temporary, then copy back
    for (const auto i : c10::irange(ndim)) {
      temp[i] = vec[dim_permute[i]];
    }
    vec = temp;
  };
  apply_permutation(in_strides);
  apply_permutation(out_strides);
  apply_permutation(signal_half_sizes);

  // Find dims.slice(dims.size() - 1) in the new permuted order.
  // These are the dimensions that need explicit Hermitian mirroring
  DimVector mirror_dims;
  mirror_dims.reserve(dim.size() - 1);
  for (const auto i : c10::irange(ndim)) {
    if (dim_permute[i] >= static_cast<int64_t>(
                              iter_strides.size()) && // Not a batch dimension
        dim_permute[i] != ndim - 1) { // Not the last dim, which is mirrored
                                      // separately with negative strides
      mirror_dims.push_back(i);
    }
  }
  TORCH_INTERNAL_ASSERT(mirror_dims.size() == dim.size() - 1);

  _fft_fill_with_conjugate_symmetry_xpu(
      input.scalar_type(),
      mirror_dims,
      signal_half_sizes,
      in_strides,
      in_data,
      out_strides,
      out_data);
}

} // namespace xpu
} // namespace native
} // namespace at
