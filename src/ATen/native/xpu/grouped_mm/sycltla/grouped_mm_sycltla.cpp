#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/collective/xe_array_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/util/GPU_Clock.hpp"

#include <cute/tensor.hpp>
#include <c10/xpu/XPUCachingAllocator.h>
#include <random>

#include <ATen/core/Tensor.h>
#include <ATen/TensorUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/CPUFunctions.h>
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/bmm.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_strided.h>
#include <ATen/ops/mm.h>
#endif

#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "groupmm_common.h"
#include "grouped_mm_sycltla.h"

using namespace cute;
using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>; // <M,N,K> per group

namespace sycltla
{
    int round_up_to_nearest_multiple(int a, int b)
    {
        return (a + b - 1) / b * b;
    }  

    // Function to perform matrix multiplication
    // similar to https://github.com/pytorch/pytorch/blob/f9875166a953a51bbd454d963ee03d41818a27e8/aten/src/ATen/native/cuda/Blas.cpp#L1661
    template <
        bool a_row_major,
        bool b_row_major,
        typename TB_M,
        typename TB_N,
        typename TB_K>
    void bf16bf16_grouped_gemm_impl(
        at::Tensor mat_a, // bf16
        at::Tensor mat_b, // bf16
        std::optional<at::Tensor> offs,
        std::optional<at::Tensor> bias, // BF16
        at::Tensor &output)
    {

        // The KernelHardwareInfo struct holds the number of EUs on the GPU with a given device ID. This
        // information is used by the underlying kernel.
        cutlass::KernelHardwareInfo hw_info;

        // Change device_id to another value if you are running on a machine with multiple GPUs and wish
        // to use a GPU other than that with device ID 0.
        hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

        // The code section below describes datatype for input, output matrices and computation between
        // elements in input matrices.
        using DtypeA = cutlass::bfloat16_t;
        using DtypeB = cutlass::bfloat16_t;
        using DtypeOutput = float;
        using DtypeAccum = float;
        using ElementComputeEpilogue = float;
        using LayoutA = cute::conditional_t<
            a_row_major,
            cutlass::layout::RowMajor,
            cutlass::layout::ColumnMajor>;

        constexpr int AlignmentA = 16 / sizeof(DtypeA);

        using LayoutB = cute::conditional_t<
            b_row_major,
            cutlass::layout::RowMajor,
            cutlass::layout::ColumnMajor>;
        constexpr int AlignmentB = 16 / sizeof(DtypeB);
        using LayoutOutput = cutlass::layout::RowMajor;

        using GmemTiledCopyA = XE_2D_U16x32x32_LD_N;
        using GmemTiledCopyB = XE_2D_U16x32x32_LD_V;

        // Workgroup-level tile
        using TileShape = Shape<_256, _256, _32>;

        using TiledMma =
            TiledMMA<MMA_Atom<XE_8x16x16_F32BF16BF16F32_TT>,
                     Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>,
                     Tile<Layout<Shape<_8, _8, _4>, Stride<_1, _32, _8>>,
                          Layout<Shape<_16, _4, _4>, Stride<_1, _64, _16>>, _32>>;

        constexpr int PipelineStages = 2;
        // Dispatch to grouped gemm algorithm
        using GEMMDispatchPolicy = cutlass::gemm::MainloopIntelXeXMX16Group<PipelineStages>;
        using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeXMX16Group;

        using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<DtypeOutput, ElementComputeEpilogue,
                                                                        DtypeAccum, DtypeAccum, cutlass::FloatRoundStyle::round_to_nearest>;

        using FusionCallBacks = cutlass::epilogue::fusion::FusionCallbacks<EpilogueDispatchPolicy, EpilogueOp, TileShape,
                                                                           decltype(tile_shape(TiledMma()))>;
        using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
            EpilogueDispatchPolicy,
            TileShape,
            DtypeAccum,
            cutlass::gemm::TagToStrideC_t<LayoutOutput *>,
            DtypeOutput,
            cutlass::gemm::TagToStrideC_t<LayoutOutput *>,
            FusionCallBacks,
            XE_2D_U32x8x16_LD_N,
            void, void,
            XE_2D_U32x8x16_ST_N,
            void, void>;

        // Mainloop
        using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
            GEMMDispatchPolicy,
            TileShape,
            DtypeA,
            cutlass::gemm::TagToStrideA_t<LayoutA *>,
            DtypeB,
            cutlass::gemm::TagToStrideB_t<LayoutB *>,
            TiledMma,
            GmemTiledCopyA, void, void, cute::identity, // A
            GmemTiledCopyB, void, void, cute::identity  // B
            >;

        using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
            ProblemShape,
            CollectiveMainloop,
            CollectiveEpilogue,
            cutlass::gemm::GroupScheduler>;

        using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
        int32_t M, N, K, group_count;
        using StrideA = typename Gemm::GemmKernel::InternalStrideA;
        using StrideB = typename Gemm::GemmKernel::InternalStrideB;
        using Strideout = typename Gemm::GemmKernel::InternalStrideD;
        using Strides = std::array<int64_t, 3>;

        M = mat_a.size(-2);
        K = mat_a.size(-1);
        N = mat_b.size(-1);

        if (mat_a.dim() == 2 && mat_b.dim() == 2)
        {
            // if both inputs are ragged, K is dynamic, M and N come from inputs
            group_count = offs->size(0);
            K = -1;
        }
        else if (mat_a.dim() == 2)
        {
            group_count = mat_b.size(0);
            M = -1;
        }
        else if (mat_b.dim() == 2)
        {
            group_count = mat_a.size(0);
            N = -1;
        }
        else
        {
            // regular bmm
            group_count = mat_a.size(0);
        }

        TORCH_CHECK(group_count < 1024, "Can't process more than 1024 groups");
        const int64_t problem_shape_size =
            group_count * ((int64_t)sizeof(ProblemShape::UnderlyingProblemShape));

        const int64_t stride_size = 3 * group_count * ((int64_t)sizeof(StrideA));

        // dummy tmas are created based on these pointer-to-pointers
        // the actual values are never used, they are replaced
        // by real addresses, but for dummy tma creation to succeed
        // due to bug in cuda < 12.4 the pointers have to be aligned to 128 bits
        const int group_alignment = 16 / sizeof(void *);
        const int aligned_group_count =
            round_up_to_nearest_multiple(group_count, group_alignment);
        int64_t input_args_size = aligned_group_count * 3 * sizeof(void *) +
                                  problem_shape_size + stride_size;
        auto &allocator = *c10::xpu::XPUCachingAllocator::get();
        auto input_buf = allocator.allocate(input_args_size);
        void *buf_ptr = input_buf.get();
        DtypeA **inputA_ptrs = reinterpret_cast<DtypeA **>(buf_ptr);
        DtypeB **inputB_ptrs =
            reinterpret_cast<DtypeB **>(inputA_ptrs + aligned_group_count);
        DtypeOutput **out_ptrs =
            reinterpret_cast<DtypeOutput **>(inputB_ptrs + aligned_group_count);
        static_assert(
            sizeof(StrideA) == 8, "expected StrideA to be 8 bytes for alignment");
        StrideA *stride_A =
            reinterpret_cast<StrideA *>(out_ptrs + aligned_group_count);
        StrideB *stride_B = reinterpret_cast<StrideB *>(stride_A + group_count);
        Strideout *stride_out =
            reinterpret_cast<Strideout *>(stride_B + group_count);
        ProblemShape::UnderlyingProblemShape *problem_sizes =
            reinterpret_cast<ProblemShape::UnderlyingProblemShape *>(
                stride_out + group_count);

        auto make_strides = [](at::IntArrayRef strides) -> Strides
        {
            Strides stride_values;
            std::copy(strides.begin(), strides.end(), stride_values.begin());
            return stride_values;
        };

        Strides tensor_StrideA = make_strides(mat_a.strides());
        Strides tensor_StrideB = make_strides(mat_b.strides());
        Strides tensor_Strideout = make_strides(output.strides());
        Strides tensor_ShapeA = make_strides(mat_a.sizes());
        Strides tensor_ShapeB = make_strides(mat_b.sizes());
        printf("Launching sycl kernel to prepare grouped gemm data...\n");
        auto prepared_data_kernel = at::xpu::detail::PreparedGroupedGEMMDataKernel<DtypeA, DtypeB, DtypeOutput, float, ProblemShape::UnderlyingProblemShape, StrideA, StrideB, Strideout>(
            reinterpret_cast<DtypeA *>(mat_a.data_ptr()),
            reinterpret_cast<DtypeB *>(mat_b.data_ptr()),
            reinterpret_cast<DtypeOutput *>(output.data_ptr()),
            static_cast<float *>(nullptr),
            static_cast<float *>(nullptr),
            inputA_ptrs,
            inputB_ptrs,
            out_ptrs,
            static_cast<float **>(nullptr),
            static_cast<float **>(nullptr),
            problem_sizes,
            stride_A,
            stride_B,
            stride_out,
            offs.has_value() ? reinterpret_cast<int32_t *>(offs->data_ptr()) : nullptr,
            M,
            N,
            K,
            tensor_StrideA,
            tensor_StrideB,
            tensor_Strideout,
            tensor_ShapeA,
            tensor_ShapeB,
            0, // offsetA
            0, // offsetB
            a_row_major,
            b_row_major);
        
        // launch the sycl kernel of preparing data
        sycl::queue q = c10::xpu::getCurrentXPUStream().queue();
        auto global_range = sycl::range<1>(group_count);
        auto local_range = sycl::range<1>(group_count);
        auto cgf = [&](::sycl::handler &cgh)
        {
            cgh.parallel_for(
                ::sycl::nd_range<1>(global_range, local_range), prepared_data_kernel);
        };
        q.submit(cgf).wait();
        printf("Data prepared, running cutlass grouped gemm...\n");
        typename Gemm::Arguments arguments{
            cutlass::gemm::GemmUniversalMode::kGrouped,
            {group_count, problem_sizes, nullptr},
            {(const DtypeA **)inputA_ptrs,
             stride_A,
             (const DtypeB **)inputB_ptrs,
             stride_B},
            {{},
             (const DtypeOutput **)out_ptrs,
             stride_out,
             out_ptrs,
             stride_out}};
        arguments.epilogue.thread.alpha = 1.0;
        arguments.epilogue.thread.dAlpha = {cute::_0{}, cute::_0{}, 0};
        arguments.hw_info = hw_info;

        size_t workspace_size = Gemm::get_workspace_size(arguments);
        auto workspace = allocator.allocate(workspace_size);
        Gemm gemm;
        TORCH_CHECK(
            gemm.can_implement(arguments) == cutlass::Status::kSuccess,
            "cutlass cannot implement");
        TORCH_CHECK(
            gemm.initialize(arguments, workspace.get()) == cutlass::Status::kSuccess,
            "cutlass cannot initialize");
        printf("Initialized finish, Running cutlass grouped gemm kernel...\n");
        auto status = gemm.run(at::xpu::getCurrentXPUStream());
        TORCH_CHECK(
            status == cutlass::Status::kSuccess,
            "cutlass cannot run,  at::Tensor ",
            int(status));
    }

    template <bool a_row_major, bool b_row_major>
    void dispatch_bf16_grouped_kernel_on_tile_size(
        at::Tensor mat_a, // bf16
        at::Tensor mat_b, // bf16
        std::optional<at::Tensor> offs,
        std::optional<at::Tensor> bias, // BF16
        at::Tensor &out)
    {
        int32_t M, N, K, group_count;

        M = mat_a.size(-2);
        K = mat_a.size(-1);
        N = mat_b.size(-1);

        // below we assume that gemms are approx same size
        if (mat_a.dim() == 2 && mat_b.dim() == 2)
        {
            // if both inputs are ragged, K is dynamic, M and N come from inputs
            group_count = offs->size(0);
            K = K / group_count;
        }
        else if (mat_a.dim() == 2)
        {
            group_count = mat_b.size(0);
            M = M / group_count;
        }
        else if (mat_b.dim() == 2)
        {
            group_count = mat_a.size(0);
            N = N / group_count;
        }
        bool small = (M <= 128 || N <= 128);
        if (small)
        {
            bf16bf16_grouped_gemm_impl<
                a_row_major,
                b_row_major,
                cute::_256,
                cute::_256,
                cute::_32>(mat_a, mat_b, offs, bias, out); // Tile shape taken from CUTLASS examples, 64 = 128/sizeof(bfloat16)
        }
        else
        {
            bf16bf16_grouped_gemm_impl<
                a_row_major,
                b_row_major,
                cute::_256,
                cute::_256,
                cute::_32>(mat_a, mat_b, offs, bias, out); // Same as above ^
        }
    }

    void grouped_mm_moe_forward_sycltla(
        at::Tensor mat_a, // bf16
        at::Tensor mat_b, // bf16
        std::optional<at::Tensor> offs,
        std::optional<at::Tensor> bias, // BF16
        at::Tensor &out)
    {
        // we already checked that one of the strides is 1
        bool a_row_major = mat_a.stride(-1) == 1;
        bool b_row_major = mat_b.stride(-1) == 1;
        TORCH_CHECK(a_row_major, "mat_a has to be row major on XPU");
        TORCH_CHECK(b_row_major, "mat_b has to be row major on XPU");
        if (a_row_major && b_row_major)
        {
            dispatch_bf16_grouped_kernel_on_tile_size<true, true>(
                mat_a, mat_b, offs, bias, out);
        }
        else if (a_row_major && !b_row_major)
        {
            dispatch_bf16_grouped_kernel_on_tile_size<true, false>(
                mat_a, mat_b, offs, bias, out);
        }
        else
        {
            TORCH_CHECK(false, "mat_a has to be row major on XPU");
        }
        // else if (!a_row_major && b_row_major)
        // {
        //     dispatch_bf16_grouped_kernel_on_tile_size<false, true>(
        //         mat_a, mat_b, offs, bias, out);
        // }
        // else
        // {
        //     dispatch_bf16_grouped_kernel_on_tile_size<false, false>(
        //         mat_a, mat_b, offs, bias, out);
        // }
     }
} // namespace sytla