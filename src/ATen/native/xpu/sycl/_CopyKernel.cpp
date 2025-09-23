#include <ATen/native/xpu/sycl/Loops.h>
#include <c10/core/ScalarType.h>

namespace at {
namespace native {
namespace xpu {

template <typename scalar_t>
struct CopyFunctor {
  scalar_t operator()(scalar_t x) const {
    return x;
  }
};

// We have got aten::copy, which uses PyTorch Tensor as inputs. But
// in some case, we need raw pointers inputted.
// Fixing compilation error: multiple definitions, when the kernel is
// inlined. See the below case,
// func<dtype1, dtype2> -> kernel<dtype1, type2> -> copy_kernel<dtype1>
// When the func extends for multiple dtype2 in a same source file,
// copy_kernel will be defined multiple times for a same dtype1.
template <typename scalar_t>
void _copy_kernel(scalar_t* dst, scalar_t* src, int numel) {
  auto input_calc = TrivialOffsetCalculator<2>();
  at::detail::Array<char*, 2> data;
  data[0] = (char*)dst;
  data[1] = (char*)src;

  auto vec_size = memory::can_vectorize_up_to<CopyFunctor<scalar_t>>(data);
  launch_vectorized_kernel(
      numel, CopyFunctor<scalar_t>(), data, input_calc, vec_size);
}

// Add it on demand.
template void _copy_kernel<bool>(bool*, bool*, int);
template void _copy_kernel<int8_t>(int8_t*, int8_t*, int);
template void _copy_kernel<uint8_t>(uint8_t*, uint8_t*, int);
template void _copy_kernel<short>(short*, short*, int);
template void _copy_kernel<int>(int*, int*, int);
template void _copy_kernel<long>(long*, long*, int);

template void _copy_kernel<double>(double*, double*, int);
template void _copy_kernel<float>(float*, float*, int);
template void _copy_kernel<c10::BFloat16>(c10::BFloat16*, c10::BFloat16*, int);
template void _copy_kernel<c10::Half>(c10::Half*, c10::Half*, int);
template void _copy_kernel<c10::Float8_e5m2>(
    c10::Float8_e5m2*,
    c10::Float8_e5m2*,
    int);
template void _copy_kernel<c10::Float8_e4m3fn>(
    c10::Float8_e4m3fn*,
    c10::Float8_e4m3fn*,
    int);

} // namespace xpu
} // namespace native
} // namespace at
