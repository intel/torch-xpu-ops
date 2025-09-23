#pragma once

namespace at {
namespace native {
namespace xpu {

template <typename scalar_t>
void _copy_kernel(scalar_t* dst, scalar_t* src, int n);

extern template void _copy_kernel<bool>(bool*, bool*, int);
extern template void _copy_kernel<int8_t>(int8_t*, int8_t*, int);
extern template void _copy_kernel<uint8_t>(uint8_t*, uint8_t*, int);
extern template void _copy_kernel<short>(short*, short*, int);
extern template void _copy_kernel<int>(int*, int*, int);
extern template void _copy_kernel<long>(long*, long*, int);
extern template void _copy_kernel<double>(double*, double*, int);
extern template void _copy_kernel<float>(float*, float*, int);
extern template void _copy_kernel<c10::BFloat16>(
    c10::BFloat16*,
    c10::BFloat16*,
    int);
extern template void _copy_kernel<c10::Half>(c10::Half*, c10::Half*, int);
extern template void _copy_kernel<c10::Float8_e5m2>(
    c10::Float8_e5m2*,
    c10::Float8_e5m2*,
    int);
extern template void _copy_kernel<c10::Float8_e4m3fn>(
    c10::Float8_e4m3fn*,
    c10::Float8_e4m3fn*,
    int);

} // namespace xpu
} // namespace native
} // namespace at
