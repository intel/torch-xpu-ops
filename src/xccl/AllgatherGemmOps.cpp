// Copyright 2025 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// AllgatherGemmOps.cpp
// -----------------------------------------------------------------------
// CUTLASS-based implementations of:
//   - run_allgather_gemm_bf16   (pipelined allgather + GEMM)
//   - run_reduce_scatter_gemm_bf16 (pipelined GEMM + reduce-scatter)
//
// This file must be compiled with ATen_XPU_SYCLTLA_SRCS so that the
// CUTLASS / sycl-tla include paths and compile definitions are available.
//
// The algorithms mirror exactly:
//   allgather_gemm.hpp::ExampleRunner<Gemm>::allgather_gemm::operator()
//   gemm_reducescatter.hpp::ExampleRunner<Gemm>::reduce_scatter::operator()
// but replace the MPI-based SymmMemory with raw void** peer_ptrs from
// XPUSymmetricMemory, and replace the MPI barrier with a callable.
// -----------------------------------------------------------------------

// CUTLASS / sycl-tla headers (available only when compiled with SYCLTLA flags)
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/collective/xe_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/util/packed_stride.hpp"
#include "sycl_common.hpp"   // TiledMMAHelper, XE_DPAS_TT — from sycl-tla/applications/

#include <cute/tensor.hpp>
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>

#include <functional>
#include <mutex>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

#include "xccl/AllgatherGemmOps.hpp"

using namespace cute;  // CUTLASS uses this pervasively

// ============================================================================
// CUTLASS Gemm type — bf16 input/output, RowMajor, Xe BMG tile config.
// Identical to the Gemm type instantiated in allgather_gemm.hpp::main() and
// gemm_reducescatter.hpp::main().
// ============================================================================
namespace {

using ElementAccumulator     = float;
using ElementComputeEpilogue = float;
using ElementInputA          = bfloat16_t;
using ElementInputB          = bfloat16_t;
using ElementOutput          = bfloat16_t;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

using GmemTiledCopyA = void;
using GmemTiledCopyB = void;
using TileShape = Shape<_256, _256, _32>;
using TiledMma = typename TiledMMAHelper<
    MMA_Atom<XE_DPAS_TT<8, float, cute::bfloat16_t>>,
    Layout<TileShape>,
    Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;

constexpr int PipelineStages = 2;
using GEMMDispatchPolicy    = cutlass::gemm::MainloopXeL1Staged<PipelineStages>;
using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeGeneric;

using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
    ElementOutput,
    ElementComputeEpilogue,
    ElementAccumulator,
    ElementAccumulator,
    cutlass::FloatRoundStyle::round_to_nearest>;

using FusionCallbacks = cutlass::epilogue::fusion::FusionCallbacks<
    EpilogueDispatchPolicy,
    EpilogueOp,
    TileShape,
    decltype(tile_shape(TiledMma()))>;

using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
    EpilogueDispatchPolicy,
    TileShape,
    void,
    ElementAccumulator,
    cutlass::gemm::TagToStrideC_t<LayoutC>,
    ElementOutput,
    cutlass::gemm::TagToStrideC_t<LayoutD>,
    FusionCallbacks,
    void,
    void>;

using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
    GEMMDispatchPolicy,
    TileShape,
    ElementInputA,
    cutlass::gemm::TagToStrideA_t<LayoutA>,
    ElementInputB,
    cutlass::gemm::TagToStrideB_t<LayoutB>,
    TiledMma,
    GmemTiledCopyA,
    void,
    void,
    cute::identity,
    GmemTiledCopyB,
    void,
    void,
    cute::identity>;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// ============================================================================
// Lightweight CUTLASS GEMM runner.
// Holds the initialized Gemm operator and shard strides.
// Call initialize() once (or on shape change), then run_shard_gemm() per step.
// ============================================================================
struct GemmRunner {
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;
    using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;

    StrideA shard_stride_A{};
    StrideB stride_B{};
    StrideC shard_stride_C{};
    StrideD shard_stride_D{};
    ProblemShapeType shard_problem_{};
    cutlass::KernelHardwareInfo hw_info_{};
    Gemm gemm_op_;
    bool initialized_ = false;

    void initialize(
            int shard_m,
            int n,
            int k,
            float alpha,
            float beta,
            int device_id,
            ElementInputA* a_ptr,
            ElementInputB* b_ptr,
            ElementOutput* d_ptr,
            sycl::queue& q) {
        hw_info_.device_id = device_id;
        hw_info_.sm_count =
            cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);

        shard_problem_ = ProblemShapeType{shard_m, n, k, 1};
        shard_stride_A = cutlass::make_cute_packed_stride(
            StrideA{}, cute::make_shape(shard_m, k, 1));
        stride_B = cutlass::make_cute_packed_stride(
            StrideB{}, cute::make_shape(n, k, 1));
        shard_stride_C = cutlass::make_cute_packed_stride(
            StrideC{}, cute::make_shape(shard_m, n, 1));
        shard_stride_D = cutlass::make_cute_packed_stride(
            StrideD{}, cute::make_shape(shard_m, n, 1));

        typename Gemm::GemmKernel::Arguments args{
            cutlass::gemm::GemmUniversalMode::kGemm,
            shard_problem_,
            {a_ptr, shard_stride_A, b_ptr, stride_B},
            {{alpha, beta},
             static_cast<typename Gemm::ElementC const*>(nullptr),
             shard_stride_C,
             d_ptr,
             shard_stride_D},
            hw_info_};

        auto st = gemm_op_.can_implement(args);
        if (st != cutlass::Status::kSuccess) {
            throw std::runtime_error("GemmRunner: GEMM can_implement failed.");
        }
        st = gemm_op_.initialize(args, /*workspace=*/nullptr, &q);
        if (st != cutlass::Status::kSuccess) {
            throw std::runtime_error("GemmRunner: GEMM initialize failed.");
        }
        initialized_ = true;
    }

    cutlass::Status run_shard_gemm(
            sycl::queue& queue,
            ElementInputA* a_ptr,
            ElementInputB* b_ptr,
            ElementOutput* d_ptr,
            float alpha,
            float beta) {
        if (!initialized_) {
            return cutlass::Status::kErrorInternal;
        }
        typename Gemm::GemmKernel::Arguments args{
            cutlass::gemm::GemmUniversalMode::kGemm,
            shard_problem_,
            {a_ptr, shard_stride_A, b_ptr, stride_B},
            {{alpha, beta},
             static_cast<typename Gemm::ElementC const*>(nullptr),
             shard_stride_C,
             d_ptr,
             shard_stride_D},
            hw_info_};

        auto st = gemm_op_.update(args, /*workspace=*/nullptr);
        if (st != cutlass::Status::kSuccess) {
            return st;
        }
        return gemm_op_.run(&queue);
    }
};

// ============================================================================
// Per-call-site runner caches.
// Key: (shard_m, n, k, device_id)
// Two separate caches so allgather_gemm and reduce_scatter_gemm can coexist
// with different shapes for the same group.
// ============================================================================
using ShapeKey = std::tuple<int, int, int, int>; // (shard_m, n, k, device_id)

struct ShapeKeyHash {
    std::size_t operator()(const ShapeKey& key) const noexcept {
        std::size_t h = 17;
        h = h * 31 + std::hash<int>{}(std::get<0>(key));
        h = h * 31 + std::hash<int>{}(std::get<1>(key));
        h = h * 31 + std::hash<int>{}(std::get<2>(key));
        h = h * 31 + std::hash<int>{}(std::get<3>(key));
        return h;
    }
};

static GemmRunner& get_ag_runner(int shard_m, int n, int k, int device_id) {
    static std::mutex mu;
    static std::unordered_map<ShapeKey, GemmRunner, ShapeKeyHash> cache;
    std::lock_guard<std::mutex> lk(mu);
    return cache[{shard_m, n, k, device_id}];
}

static GemmRunner& get_rs_runner(int shard_m, int n, int k, int device_id) {
    static std::mutex mu;
    static std::unordered_map<ShapeKey, GemmRunner, ShapeKeyHash> cache;
    std::lock_guard<std::mutex> lk(mu);
    return cache[{shard_m, n, k, device_id}];
}

} // anonymous namespace

// ============================================================================
// Public API implementation
// ============================================================================
namespace xpu_symm_gemm {

// ---------------------------------------------------------------------------
// run_allgather_gemm_bf16
// ---------------------------------------------------------------------------
// Mirrors ExampleRunner<Gemm>::allgather_gemm::operator() in allgather_gemm.hpp.
//
// Algorithm:
//   1. memcpy local_A → peer_ptrs[rank][rank * shard_a_elems]    (publish)
//   2. barrier_fn()                                               (all ranks published)
//   3. run_shard_gemm on local shard (no need to wait for remote copies)
//   4. For step in 1..world_size-1, alternating primary/secondary queue:
//        P2P memcpy: peer_ptrs[remote_rank][remote_rank * shard_a_elems] → local_ws
//        run_shard_gemm on that slice
//   5. primary_q.ext_oneapi_submit_barrier(secondary_q.ext_oneapi_submit_barrier())
//   6. barrier_fn()
// ---------------------------------------------------------------------------
void run_allgather_gemm_bf16(
        void* local_A_v,
        void* B_v,
        void* out_C_v,
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
        const std::function<void()>& barrier_fn) {
    auto* local_A = reinterpret_cast<ElementInputA*>(local_A_v);
    auto* B       = reinterpret_cast<ElementInputB*>(B_v);
    auto* out_C   = reinterpret_cast<ElementOutput*>(out_C_v);

    GemmRunner& runner = get_ag_runner(shard_m, n, k, device_id);
    if (!runner.initialized_) {
        runner.initialize(
            shard_m, n, k, alpha, beta, device_id,
            local_A, B, out_C, primary_q);
    }

    const size_t shard_a_elems = static_cast<size_t>(shard_m) * k;
    const size_t shard_c_elems = static_cast<size_t>(shard_m) * n;
    const size_t shard_a_bytes = shard_a_elems * sizeof(ElementInputA);

    // Gathered A workspace lives in our symm buffer (peer_ptrs[rank]).
    ElementInputA* gathered_A =
        reinterpret_cast<ElementInputA*>(peer_ptrs[rank]);

    // Step 1: publish local shard into our slot of the symm workspace
    primary_q.memcpy(
        gathered_A + static_cast<size_t>(rank) * shard_a_elems,
        local_A,
        shard_a_bytes);

    // Step 2: barrier — all ranks have written their local shard
    barrier_fn();

    // Step 3: local GEMM first (immediately, before any remote P2P copies)
    {
        auto st = runner.run_shard_gemm(
            primary_q,
            gathered_A + static_cast<size_t>(rank) * shard_a_elems,
            B,
            out_C + static_cast<size_t>(rank) * shard_c_elems,
            alpha, beta);
        if (st != cutlass::Status::kSuccess) {
            throw std::runtime_error(
                "run_allgather_gemm_bf16: local shard GEMM submission failed.");
        }
    }

    // Step 4: remote shards — alternate primary/secondary queue
    for (int step = 1; step < world_size; ++step) {
        const int remote_rank = (rank + step) % world_size;
        sycl::queue& work_q = (step % 2 == 0) ? primary_q : secondary_q;

        // P2P read: remote_rank's gathered_A at remote_rank's slot
        ElementInputA* remote_buf =
            reinterpret_cast<ElementInputA*>(peer_ptrs[remote_rank]);
        ElementInputA* remote_src =
            remote_buf + static_cast<size_t>(remote_rank) * shard_a_elems;
        ElementInputA* local_dst =
            gathered_A + static_cast<size_t>(remote_rank) * shard_a_elems;

        work_q.memcpy(local_dst, remote_src, shard_a_bytes);

        auto st = runner.run_shard_gemm(
            work_q,
            local_dst,
            B,
            out_C + static_cast<size_t>(remote_rank) * shard_c_elems,
            alpha, beta);
        if (st != cutlass::Status::kSuccess) {
            throw std::runtime_error(
                "run_allgather_gemm_bf16: remote shard GEMM submission failed.");
        }
    }

    // Step 5: merge secondary into primary, then final barrier
    primary_q.ext_oneapi_submit_barrier(
        {secondary_q.ext_oneapi_submit_barrier()});
    barrier_fn();
}

// ---------------------------------------------------------------------------
// run_reduce_scatter_gemm_bf16
// ---------------------------------------------------------------------------
// Mirrors ExampleRunner<Gemm>::reduce_scatter::operator() in gemm_reducescatter.hpp.
//
// Algorithm:
//   1. barrier_fn()                        (peers ready to receive)
//   2. For step in 1..world_size-1, alternating queue:
//        run_shard_gemm for dst_rank's A rows → local_ws[dst_rank * shard_c_elems]
//        P2P push → peer_ptrs[dst_rank][rank * shard_c_elems]
//   3. run_shard_gemm for own rows → local_ws[rank * shard_c_elems]
//   4. merge queues + barrier_fn()
//   5. local reduction: sum world_size copies in local_ws → out_C
// ---------------------------------------------------------------------------
void run_reduce_scatter_gemm_bf16(
        void* A_v,
        void* B_v,
        void* out_C_v,
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
        const std::function<void()>& barrier_fn) {
    auto* A     = reinterpret_cast<ElementInputA*>(A_v);
    auto* B     = reinterpret_cast<ElementInputB*>(B_v);
    auto* out_C = reinterpret_cast<ElementOutput*>(out_C_v);

    GemmRunner& runner = get_rs_runner(shard_m, n, k, device_id);
    if (!runner.initialized_) {
        runner.initialize(
            shard_m, n, k, alpha, beta, device_id,
            A, B, out_C, primary_q);
    }

    const size_t shard_a_elems = static_cast<size_t>(shard_m) * k;
    const size_t shard_c_elems = static_cast<size_t>(shard_m) * n;
    const size_t shard_c_bytes = shard_c_elems * sizeof(ElementOutput);

    // Our local symm workspace (world_size * shard_c_bytes).
    ElementOutput* local_ws =
        reinterpret_cast<ElementOutput*>(peer_ptrs[rank]);

    // Step 1: barrier — peers are ready to receive
    barrier_fn();

    // Step 2: remote partial GEMMs + P2P push
    for (int step = 1; step < world_size; ++step) {
        const int dst_rank = (rank + step) % world_size;
        sycl::queue& work_q = (step % 2 == 0) ? primary_q : secondary_q;

        // Compute partial GEMM for dst_rank's A rows, write into local_ws at
        // dst_rank's slot (will be read back during local reduction for our rank
        // if dst_rank == rank, which doesn't happen here since dst_rank != rank).
        ElementOutput* local_shard_out =
            local_ws + static_cast<size_t>(dst_rank) * shard_c_elems;

        auto st = runner.run_shard_gemm(
            work_q,
            A + static_cast<size_t>(dst_rank) * shard_a_elems,
            B,
            local_shard_out,
            alpha, beta);
        if (st != cutlass::Status::kSuccess) {
            throw std::runtime_error(
                "run_reduce_scatter_gemm_bf16: remote shard GEMM submission failed.");
        }

        // P2P push: send our contribution for dst_rank to dst_rank's buffer
        // at my rank's slot (peer_ptrs[dst_rank][rank * shard_c_elems]).
        ElementOutput* remote_dst =
            reinterpret_cast<ElementOutput*>(peer_ptrs[dst_rank]) +
            static_cast<size_t>(rank) * shard_c_elems;
        work_q.memcpy(remote_dst, local_shard_out, shard_c_bytes);
    }

    // Step 3: local shard GEMM (own rows)
    {
        auto st = runner.run_shard_gemm(
            primary_q,
            A + static_cast<size_t>(rank) * shard_a_elems,
            B,
            local_ws + static_cast<size_t>(rank) * shard_c_elems,
            alpha, beta);
        if (st != cutlass::Status::kSuccess) {
            throw std::runtime_error(
                "run_reduce_scatter_gemm_bf16: local shard GEMM submission failed.");
        }
    }

    // Step 4: merge secondary into primary + barrier
    primary_q.ext_oneapi_submit_barrier(
        {secondary_q.ext_oneapi_submit_barrier()});
    barrier_fn();

    // Step 5: local reduction — sum world_size copies in local_ws → out_C.
    // Vectorised bf16 kernel (matches gemm_reducescatter.hpp Phase 3 style).
    const size_t n_elems = shard_c_elems;
    constexpr int NUM_PER_TH = 8; // 8 * bf16 = 16 bytes
    using SyclBF16 = sycl::ext::oneapi::bfloat16;
    using VecT = sycl::vec<SyclBF16, NUM_PER_TH>;

    const size_t vec_elems = n_elems / NUM_PER_TH;
    constexpr size_t WG_SIZE = 256;
    const size_t n_groups = (vec_elems + WG_SIZE - 1) / WG_SIZE;

    // Build pointer array — struct must be trivially copyable for SYCL capture.
    struct SrcPtrs {
        const VecT* ptrs[8];
        int count;
    } srcs{};
    srcs.count = world_size;
    for (int r = 0; r < world_size && r < 8; ++r) {
        srcs.ptrs[r] = reinterpret_cast<const VecT*>(
            local_ws + static_cast<size_t>(r) * shard_c_elems);
    }

    VecT* out_vec = reinterpret_cast<VecT*>(out_C);

    if (vec_elems > 0) {
        primary_q.submit([=](sycl::handler& h) {
            h.parallel_for(
                sycl::nd_range<1>(n_groups * WG_SIZE, WG_SIZE),
                [=](sycl::nd_item<1> item) {
                    const size_t vi = item.get_global_linear_id();
                    if (vi >= vec_elems) return;
                    VecT acc = srcs.ptrs[0][vi];
                    for (int r = 1; r < srcs.count; ++r) {
                        acc += srcs.ptrs[r][vi];
                    }
                    out_vec[vi] = acc;
                });
        });
    }

    // Tail: remaining elements not covered by the vectorised kernel
    const size_t tail_start = vec_elems * NUM_PER_TH;
    const size_t tail = n_elems - tail_start;
    if (tail > 0) {
        struct TailSrcs {
            const SyclBF16* ptrs[8];
            int count;
        } tail_srcs{};
        tail_srcs.count = world_size;
        for (int r = 0; r < world_size && r < 8; ++r) {
            tail_srcs.ptrs[r] = reinterpret_cast<const SyclBF16*>(
                local_ws + static_cast<size_t>(r) * shard_c_elems);
        }
        SyclBF16* tail_out = reinterpret_cast<SyclBF16*>(out_C) + tail_start;
        primary_q.submit([=](sycl::handler& h) {
            h.parallel_for(
                sycl::range<1>(tail),
                [=](sycl::id<1> i) {
                    const size_t idx = tail_start + i[0];
                    SyclBF16 acc = tail_srcs.ptrs[0][idx];
                    for (int r = 1; r < tail_srcs.count; ++r) {
                        acc += tail_srcs.ptrs[r][idx];
                    }
                    tail_out[i[0]] = acc;
                });
        });
    }
}

} // namespace xpu_symm_gemm
