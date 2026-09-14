/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

/*
 * Subgroup top-k kernel -- K=4 instantiations.
 *
 * Splitting per-K into separate files to parallelize AOT compilation.
 * See TensorTopKSbtopkKernelImpl.h for the shared template code.
 */

#include <ATen/native/xpu/sycl/TensorTopKSbtopkKernelImpl.h>

namespace at::native::xpu {

template <typename scalar_t, typename IndexT>
static void sbtopk_k4_typed(
    const scalar_t* input,
    scalar_t* topK,
    int64_t* indices,
    IndexT numSlices,
    int64_t sliceSize,
    int k,
    bool largest) {
  if (largest) {
    sbtopk_launch_vec_dispatch<scalar_t, 4, true, IndexT>(
        input, topK, indices, numSlices, sliceSize, k);
  } else {
    sbtopk_launch_vec_dispatch<scalar_t, 4, false, IndexT>(
        input, topK, indices, numSlices, sliceSize, k);
  }
}

void sbtopk_k4_launch(
    const at::Tensor& self,
    int64_t nsegments,
    int64_t nelements,
    int k,
    bool largest,
    const at::Tensor& values,
    const at::Tensor& indices) {
  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "sbtopk_k4",
      [&]() {
        const auto* input = static_cast<const scalar_t*>(self.const_data_ptr());
        auto* topK = static_cast<scalar_t*>(values.data_ptr());
        auto* idx = static_cast<int64_t*>(indices.data_ptr());

        if (nsegments <=
            static_cast<int64_t>(std::numeric_limits<int>::max())) {
          sbtopk_k4_typed<scalar_t, int>(
              input,
              topK,
              idx,
              static_cast<int>(nsegments),
              nelements,
              k,
              largest);
        } else {
          sbtopk_k4_typed<scalar_t, int64_t>(
              input, topK, idx, nsegments, nelements, k, largest);
        }
      });
}

} // namespace at::native::xpu
