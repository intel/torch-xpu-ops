// Copyright 2025 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// AllgatherGemmOps.hpp
// -----------------------------------------------------------------------
// Public API for CUTLASS-based pipelined allgather+GEMM and
// GEMM+reduce-scatter on XPU.
//
// This header deliberately does NOT include any CUTLASS headers so that
// it can be included from XPUSymmetricMemoryOps.cpp, which is compiled
// without the sycl-tla / CUTLASS include paths.
//
// The implementation lives in AllgatherGemmOps.cpp, which IS compiled
// with ATen_XPU_SYCLTLA_SRCS (CUTLASS + sycl-tla includes available).
// -----------------------------------------------------------------------

#pragma once

#include <sycl/sycl.hpp>
#include <functional>
#include <cstdint>

namespace xpu_symm_gemm {

// -----------------------------------------------------------------------
// run_allgather_gemm_bf16
// -----------------------------------------------------------------------
// Pipelined allgather + GEMM using P2P symmetric memory.
// Matches the algorithm in allgather_gemm.hpp::ExampleRunner::allgather_gemm.
//
// The CUTLASS GEMM operates on bfloat16 A and B matrices and produces a
// bfloat16 output.  Only gather_dim=0 (row-major M-split) is supported.
//
// Workspace layout (peer_ptrs):
//   peer_ptrs[r] = XPUSymmetricMemory::get_buffer_ptrs()[r]
//   Each rank's buffer must be at least world_size * shard_m * k * 2 bytes.
//   Rank r writes local_A into peer_ptrs[rank][rank * shard_m * k].
//   Other ranks P2P-read from peer_ptrs[remote_rank][remote_rank * shard_m * k].
//
// @param local_A        [shard_m, k] bfloat16 — local A shard, contiguous
// @param B              [k, n] bfloat16 — weight matrix, contiguous
// @param out_C          [world_size * shard_m, n] bfloat16 — pre-allocated output
// @param peer_ptrs      void** — XPUSymmetricMemory::get_buffer_ptrs()
// @param rank           local rank
// @param world_size     total number of ranks
// @param shard_m        rows per rank (= global_m / world_size)
// @param n              GEMM N dimension
// @param k              GEMM K dimension
// @param alpha, beta    epilogue scaling factors (typically 1.0 / 0.0)
// @param device_id      XPU device index
// @param primary_q      primary in-order SYCL queue (PyTorch current stream)
// @param secondary_q    secondary in-order SYCL queue (for pipeline overlap)
// @param barrier_fn     distributed barrier — must synchronise all ranks and
//                       submit to primary_q (e.g. calls symm_mem->barrier(0,0))
void run_allgather_gemm_bf16(
    void* local_A,
    void* B,
    void* out_C,
    void** peer_ptrs,
    int rank,
    int world_size,
    int shard_m,
    int n,
    int k,
    float alpha,
    float beta,
    int device_id,
    sycl::queue& primary_q,
    sycl::queue& secondary_q,
    const std::function<void()>& barrier_fn);

// -----------------------------------------------------------------------
// run_reduce_scatter_gemm_bf16
// -----------------------------------------------------------------------
// Pipelined GEMM + reduce-scatter using P2P symmetric memory.
// Matches the algorithm in gemm_reducescatter.hpp::ExampleRunner::reduce_scatter.
//
// Workspace layout (peer_ptrs):
//   peer_ptrs[r] = XPUSymmetricMemory::get_buffer_ptrs()[r]
//   Each rank's buffer must be at least world_size * shard_m * n * 2 bytes.
//   Each rank stores world_size partial GEMM results in its buffer, where
//   peer_ptrs[rank][s * shard_m * n] holds the contribution from rank s for
//   this rank's output shard.
//
// @param A              [world_size * shard_m, k] bfloat16 — full A matrix, contiguous
// @param B              [k, n] bfloat16 — weight matrix, contiguous
// @param out_C          [shard_m, n] bfloat16 — pre-allocated output shard
// @param peer_ptrs      void** — XPUSymmetricMemory::get_buffer_ptrs()
// @param rank           local rank
// @param world_size     total number of ranks
// @param shard_m        output rows per rank (= global_m / world_size)
// @param n              GEMM N dimension
// @param k              GEMM K dimension
// @param alpha, beta    epilogue scaling factors (typically 1.0 / 0.0)
// @param device_id      XPU device index
// @param primary_q      primary in-order SYCL queue (PyTorch current stream)
// @param secondary_q    secondary in-order SYCL queue (for pipeline overlap)
// @param barrier_fn     distributed barrier callable
void run_reduce_scatter_gemm_bf16(
    void* A,
    void* B,
    void* out_C,
    void** peer_ptrs,
    int rank,
    int world_size,
    int shard_m,
    int n,
    int k,
    float alpha,
    float beta,
    int device_id,
    sycl::queue& primary_q,
    sycl::queue& secondary_q,
    const std::function<void()>& barrier_fn);

} // namespace xpu_symm_gemm
