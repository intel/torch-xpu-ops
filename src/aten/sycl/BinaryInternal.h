#pragma once

namespace at::native::xpu {

template <typename scalar_t>
struct DivFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    return a / b;
  }
};

template <typename scalar_t>
struct MulFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    return a * b;
  }
};

// Workaround for the error: '*' in boolean context, suggest '&&' instead
// [-Werror=int-in-bool-context]
template <>
struct MulFunctor<bool> {
  bool operator()(bool a, bool b) const {
    return a && b;
  }
};

} // namespace at::native::xpu
