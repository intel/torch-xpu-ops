/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>

#include <cutlass/util/device_memory.h>
#include <cutlass/util/packed_stride.hpp>

#include <vector>

namespace at::xpu::sycltla::detail {

template <
    typename ProblemShape,
    typename ElemA,
    typename ElemB,
    typename ElemOutput,
    typename StrideAType,
    typename StrideBType,
    typename StrideCType,
    typename StrideDType>
struct GroupedGemmData {
  using ElementA = ElemA;
  using ElementB = ElemB;
  using ElementOutput = ElemOutput;
  using StrideA = StrideAType;
  using StrideB = StrideBType;
  using StrideC = StrideCType;
  using StrideD = StrideDType;

  std::vector<typename ProblemShape::UnderlyingProblemShape> problem_sizes_host;
  std::vector<const ElementA*> ptr_A_host;
  std::vector<const ElementB*> ptr_B_host;
  std::vector<const ElementOutput*> ptr_C_host;
  std::vector<ElementOutput*> ptr_D_host;
  std::vector<StrideA> stride_A_host;
  std::vector<StrideB> stride_B_host;
  std::vector<StrideC> stride_C_host;
  std::vector<StrideD> stride_D_host;
  std::vector<at::Tensor> packed_a;
  std::vector<at::Tensor> packed_b;
  std::vector<at::Tensor> packed_c;
  std::vector<at::Tensor> packed_d;

  cutlass::DeviceAllocation<typename ProblemShape::UnderlyingProblemShape>
      problem_sizes;
  cutlass::DeviceAllocation<const ElemA*> ptr_A;
  cutlass::DeviceAllocation<const ElemB*> ptr_B;
  cutlass::DeviceAllocation<const ElemOutput*> ptr_C;
  cutlass::DeviceAllocation<ElemOutput*> ptr_D;
  cutlass::DeviceAllocation<StrideAType> stride_A;
  cutlass::DeviceAllocation<StrideBType> stride_B;
  cutlass::DeviceAllocation<StrideCType> stride_C;
  cutlass::DeviceAllocation<StrideDType> stride_D;
};

template <typename GroupDesc, typename Data>
bool prepare_grouped_gemm_data(const std::vector<GroupDesc>& groups, Data& data) {
  const auto group_count = groups.size();
  data.problem_sizes_host.reserve(group_count);
  data.ptr_A_host.reserve(group_count);
  data.ptr_B_host.reserve(group_count);
  data.ptr_C_host.reserve(group_count);
  data.ptr_D_host.reserve(group_count);
  data.stride_A_host.reserve(group_count);
  data.stride_B_host.reserve(group_count);
  data.stride_C_host.reserve(group_count);
  data.stride_D_host.reserve(group_count);
  data.packed_a.reserve(group_count);
  data.packed_b.reserve(group_count);
  data.packed_c.reserve(group_count);
  data.packed_d.reserve(group_count);

  using StrideA = typename Data::StrideA;
  using StrideB = typename Data::StrideB;
  using StrideC = typename Data::StrideC;
  using StrideD = typename Data::StrideD;

  for (const auto& group : groups) {
    auto a_contig = group.a.contiguous();
    auto b_contig = group.b.contiguous();
    int64_t m = a_contig.size(0);
    int64_t k = a_contig.size(1);
    int64_t n = b_contig.size(1);
    if (m == 0 || k == 0 || n == 0) {
      return false;
    }

    int m_int = static_cast<int>(m);
    int n_int = static_cast<int>(n);
    int k_int = static_cast<int>(k);
    auto c_contig = at::zeros({m, n}, a_contig.options().dtype(at::kFloat));
    auto d_contig = at::empty({m, n}, a_contig.options().dtype(at::kFloat));

    data.problem_sizes_host.push_back({m_int, n_int, k_int});
    data.ptr_A_host.push_back(
        reinterpret_cast<const typename Data::ElementA*>(
        a_contig.template data_ptr<c10::BFloat16>()));
    data.ptr_B_host.push_back(
        reinterpret_cast<const typename Data::ElementB*>(
        b_contig.template data_ptr<c10::BFloat16>()));
    data.ptr_C_host.push_back(c_contig.template data_ptr<float>());
    data.ptr_D_host.push_back(d_contig.template data_ptr<float>());
    data.stride_A_host.push_back(
        cutlass::make_cute_packed_stride(StrideA{}, {m_int, k_int, 1}));
    data.stride_B_host.push_back(
        cutlass::make_cute_packed_stride(StrideB{}, {n_int, k_int, 1}));
    data.stride_C_host.push_back(
        cutlass::make_cute_packed_stride(StrideC{}, {m_int, n_int, 1}));
    data.stride_D_host.push_back(
        cutlass::make_cute_packed_stride(StrideD{}, {m_int, n_int, 1}));

    data.packed_a.push_back(std::move(a_contig));
    data.packed_b.push_back(std::move(b_contig));
    data.packed_c.push_back(std::move(c_contig));
    data.packed_d.push_back(std::move(d_contig));
  }

  data.problem_sizes.reset(group_count);
  data.problem_sizes.copy_from_host(data.problem_sizes_host.data());
  data.ptr_A.reset(group_count);
  data.ptr_A.copy_from_host(data.ptr_A_host.data());
  data.ptr_B.reset(group_count);
  data.ptr_B.copy_from_host(data.ptr_B_host.data());
  data.ptr_C.reset(group_count);
  data.ptr_C.copy_from_host(data.ptr_C_host.data());
  data.ptr_D.reset(group_count);
  data.ptr_D.copy_from_host(data.ptr_D_host.data());
  data.stride_A.reset(group_count);
  data.stride_A.copy_from_host(data.stride_A_host.data());
  data.stride_B.reset(group_count);
  data.stride_B.copy_from_host(data.stride_B_host.data());
  data.stride_C.reset(group_count);
  data.stride_C.copy_from_host(data.stride_C_host.data());
  data.stride_D.reset(group_count);
  data.stride_D.copy_from_host(data.stride_D_host.data());
  return true;
}

} // namespace at::xpu::sycltla::detail
