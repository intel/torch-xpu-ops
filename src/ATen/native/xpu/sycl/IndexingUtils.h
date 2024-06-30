#pragma once

#include <ATen/ExpandUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/IndexingUtils.h>
#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

static Tensor wrapIndexOnce(
    const Tensor& index,
    int64_t dim,
    int64_t dim_size,
    bool check_range = true) {
  // we don't need to check range in backward - if there were out of bounds
  // indices forward should already have errored out
  if (index.numel() != 0 && check_range) {
    auto max_idx = index.max().item<int64_t>();
    auto min_idx = index.min().item<int64_t>();
    if (max_idx >= dim_size) {
      TORCH_CHECK_INDEX(
          false,
          "index ",
          max_idx,
          " is out of bounds for dimension ",
          dim,
          " with size ",
          dim_size);
    }
    if (min_idx < -dim_size) {
      TORCH_CHECK_INDEX(
          false,
          "index ",
          min_idx,
          " is out of bounds for dimension ",
          dim,
          " with size ",
          dim_size);
    }
  }
  return index.remainder(dim_size);
}

static std::vector<int64_t> computeLinearStride(const Tensor& tensor) {
  // computes the stride as if tensor were contiguous
  auto sizes = tensor.sizes();
  std::vector<int64_t> stride(tensor.dim());
  if (stride.empty()) {
    return stride;
  }
  stride[tensor.dim() - 1] = 1;
  std::partial_sum(
      sizes.rbegin(),
      sizes.rend() - 1,
      stride.rbegin() + 1,
      std::multiplies<int64_t>());
  return stride;
}

static std::tuple<Tensor, int64_t, int64_t, int64_t> computeLinearIndex(
    const Tensor& src,
    TensorList indices,
    bool check_range) {
  auto strides = computeLinearStride(src);
  const auto& device = src.options().device();

  // Compute the linear index by multiplying the indexing tensors by the
  // stride and summing them. All the indexing tensors have the same shape at
  // this point. We also compute the number of dimensions before and after that
  // are not being index.
  Tensor linearIndex;
  int64_t nElemBefore = 1, nElemAfter = 1, strideBefore = 0;
  for (const auto i : c10::irange(src.dim())) {
    if (indices[i].defined()) {
      // Cast index to the longType matching src's device
      // This allows us to support ie indexing a xpu tensor with a cpu tensor
      Tensor index =
          (wrapIndexOnce(indices[i], i, src.size(i), check_range) * strides[i])
              .to(device);
      if (linearIndex.defined()) {
        linearIndex += index;
      } else {
        linearIndex = index;
        if (i > 0) {
          strideBefore = src.stride(i - 1); // stride after undefined dimensions
        }
      }
    } else if (linearIndex.defined()) {
      nElemAfter *= src.size(i);
    } else {
      nElemBefore *= src.size(i);
    }
  }

  return std::make_tuple(
      std::move(linearIndex), nElemBefore, strideBefore, nElemAfter);
}

static std::
    tuple<Tensor, Tensor, int64_t, int64_t, int64_t, std::vector<int64_t>>
    makeLinearIndex(Tensor self, IOptTensorListRef orig, bool check_range) {
  checkIndexTensorTypes(orig, /*allow_int*/ true);
  // first expand BoolTensor (masks) or ByteTensor (masks) into 1 or more
  // LongTensors
  auto indices = expandTensors(self, orig);
  for (auto& i : indices) {
    if (i.defined() && i.dtype() == at::kInt) {
      i = i.to(at::kLong);
    }
  }
  // next broadcast all index tensors together
  indices = expand_outplace(indices);
  // add missing null Tensors so that it matches self.dim()
  while (indices.size() < (size_t)self.dim()) {
    indices.emplace_back();
  }
  // if the non-null indices are not all adjacent, transpose self and indices
  // together so that they're adjacent at the front
  std::vector<int64_t> inversePerm;
  if (!hasContiguousSubspace(indices)) {
    std::tie(self, indices, inversePerm) =
        transposeToFrontAndInvPerm(self, indices);
  }
  auto [linearIndex, nElemBefore, strideBefore, nElemAfter] =
      computeLinearIndex(self, indices, check_range);
  return std::make_tuple(
      linearIndex, self, nElemBefore, strideBefore, nElemAfter, inversePerm);
}

} // namespace at::native::xpu
