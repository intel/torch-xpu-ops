#pragma once
#include <type_traits>

namespace at::native::xpu {
template <typename T, std::enable_if_t<std::is_integral_v<T>, bool> = true>
inline T CeilDiv(T a, T b) {
  return (a - 1) / b + 1;
}
} // namespace at::native::xpu
