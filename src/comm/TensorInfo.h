// Porting from ipex
// will upstream to pytorch when in tree
#pragma once

#include <ATen/ATen.h>
#include <ATen/CPUApplyUtils.h>

#include <ATen/native/xpu/sycl/IntegerDivider.h>

namespace at {
namespace xpu {
namespace detail {

#define XPU_MAX_TENSORINFO_DIMS 12

template <typename T, typename IndexType>
struct TensorInfo {
  using scalar_t = T;

  TensorInfo();
  TensorInfo(
      T* p,
      int dim,
      IndexType sz[XPU_MAX_TENSORINFO_DIMS],
      IndexType st[XPU_MAX_TENSORINFO_DIMS]);

  // Set the sive of given dimension to 1, as if were a
  // reduction dim (allow you to calculate offsets of the
  // reduction slice)
  void reduceDim(int dim);

  // See note on [collapse dims].
  int collapseDims(const int excludeDim = -1);

  int outerSize(const int dim);

  int innerSize(const int dim);

  // Contiguous tensors of more than one dimension are collapsed down
  // to one tensor
  inline bool isContiguous() const {
    return dims == 1 && strides[0] == 1;
  }

  inline bool isContiguousCheckStrict(bool strict_contiguous) const {
    if (strict_contiguous)
      return is_strict_contiguous;
    else
      return is_contiguous;
  }

  T* data = nullptr;
  IndexType sizes[XPU_MAX_TENSORINFO_DIMS];
  IndexType strides[XPU_MAX_TENSORINFO_DIMS];
  int dims = 0;
  bool is_contiguous;
  bool is_strict_contiguous;
};

template <typename T, typename IndexType>
TensorInfo<T, IndexType>::TensorInfo() {
  data = nullptr;
  dims = 0;
  is_contiguous = true;
  is_strict_contiguous = true;
}

template <typename T, typename IndexType>
TensorInfo<T, IndexType>::TensorInfo(
    T* p,
    int dim,
    IndexType sz[XPU_MAX_TENSORINFO_DIMS],
    IndexType st[XPU_MAX_TENSORINFO_DIMS]) {
  data = p;
  dims = dim;
  TORCH_INTERNAL_ASSERT(dims <= XPU_MAX_TENSORINFO_DIMS);

  is_contiguous = true;
  int z = 1;
  for (int i = dim - 1; i >= 0; i--) {
    sizes[i] = sz[i];
    strides[i] = st[i];

    if (is_contiguous && strides[i] == z) {
      z *= sizes[i];
    } else {
      is_contiguous = false;
    }
  }

  is_strict_contiguous = dims == 1 && strides[0] == 1;
}

template <typename T, typename IndexType>
void TensorInfo<T, IndexType>::reduceDim(int dim) {
  TORCH_CHECK(dim < dims && dim >= 0, "expect dim between 0 and dims - 1");
  sizes[dim] = 1;
}

template <typename T, typename IndexType>
int TensorInfo<T, IndexType>::collapseDims(const int excludeDim) {
  auto result = at::collapse_dims(sizes, strides, dims, excludeDim);
  dims = std::get<1>(result);
  return std::get<0>(result);
}

template <typename T, typename IndexType>
int TensorInfo<T, IndexType>::innerSize(const int exclusive) {
  int size = 1;
  for (int i = dims - 1; i > exclusive; i--) {
    size *= sizes[i];
  }
  return size;
}

template <typename T, typename IndexType>
int TensorInfo<T, IndexType>::outerSize(const int exclusive) {
  int size = 1;
  for (int i = 0; i < exclusive; i++) {
    size *= sizes[i];
  }
  return size;
}

// Translate a linear index for the apply to a T* offset;
template <typename T, typename IndexType, bool Trivial = false>
struct IndexToOffset {
  static constexpr bool STRICT_CONTIGUOUS = true;
  static constexpr bool NON_STRICT_CONTIGUOUS = false;
  static inline IndexType get(
      IndexType linearId,
      const TensorInfo<T, IndexType>& info,
      bool strict_contiguous = true) {
    IndexType offset = 0;

    if (info.isContiguousCheckStrict(strict_contiguous)) {
      return linearId;
    }

#pragma unroll
    for (int dim = XPU_MAX_TENSORINFO_DIMS - 1; dim >= 0; --dim) {
      if (dim < info.dims) {
        auto divider = at::detail::IntDivider<IndexType>(info.sizes[dim]);
        auto divmod = divider.divmod(linearId);
        linearId = divmod.div;
        offset += divmod.mod * info.strides[dim];
      }
    }
    return offset;
  }
};

// To isolate unnecessary code, even the code is not involved in
// contiguouse case. Additional unnecessary code impacts efficiency of
// generated code.
template <typename T, typename IndexType>
struct IndexToOffset<T, IndexType, true> {
  static constexpr bool STRICT_CONTIGUOUS = true;
  static constexpr bool NON_STRICT_CONTIGUOUS = false;
  static inline IndexType get(
      IndexType linearId,
      const TensorInfo<T, IndexType>& info,
      bool strict_contiguous = true) {
    return linearId;
  }
};

template <typename scalar, typename IndexType>
TensorInfo<scalar, IndexType> getTensorInfo(const at::Tensor& t) {
  IndexType sz[XPU_MAX_TENSORINFO_DIMS];
  IndexType st[XPU_MAX_TENSORINFO_DIMS];

  TORCH_CHECK(
      t.dim() <= XPU_MAX_TENSORINFO_DIMS,
      "dim:",
      t.dim(),
      " exceed max allowed dim:",
      XPU_MAX_TENSORINFO_DIMS);

  int dims;
  if (t.dim()) {
    dims = t.dim();
    for (int i = 0; i < dims; ++i) {
      sz[i] = t.size(i);
      st[i] = t.stride(i);
    }
  } else {
    dims = 1;
    sz[0] = 1;
    st[0] = 1;
  }

  return TensorInfo<scalar, IndexType>(t.data_ptr<scalar>(), dims, sz, st);
}

} // namespace detail
} // namespace xpu
} // namespace at
