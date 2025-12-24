#pragma once
#include <sycl/sycl.hpp>

namespace at::xpu::detail
{
    using Strides = std::array<int64_t, 3>;

    template <
            typename DtypeA,
            typename DtypeB,
            typename DtypeOutput,
            typename DtypeScale,
            typename ProblemShape,
            typename StrideA,
            typename StrideB,
            typename StrideOutput>
    struct PreparedGroupedGEMMDataKernel
    { 

        void operator()(sycl::nd_item<1> item) const
        {
            int32_t tid = item.get_global_id(0);
            int32_t delta = 0;
            int32_t offset = 0;
            auto real_M = M;
            auto real_N = N;
            auto real_K = K;
            if (offs != nullptr)
            {
                int32_t start = tid == 0 ? 0 : offs[tid - 1];
                offset = offs[tid];
                delta = offset - start;
                SYCL_KERNEL_ASSERT(delta >= 0 && "expected gemm dimension to be greater or equal 0\n");

                // TMA transfers require global memory tensor addresses to be
                // aligned to 16 bytes.
                if (tid < item.get_local_range().size() - 1)
                {
                    // Check this requirement for input tensors, in case group
                    // addresses are increased along the dynamic dimension.
                    if ((K < 0 && a_row_major) || // 2D/2D: check along K dimension
                        (M < 0 && !a_row_major))
                    { // 3D/2D: check along N dimension
                        int align = 128 / cutlass::sizeof_bits<DtypeA>::value;
                        SYCL_KERNEL_ASSERT(
                            delta % align == 0 &&
                            "expected input tensor dynamic dimension byte size to be non-negative multiple of 16\n");
                    }
                    if ((K < 0 && !b_row_major) || // 2D/2D: check along K dimension
                        (N < 0 && b_row_major))
                    { // 3D/2D: check along N dimension
                        int align = 128 / cutlass::sizeof_bits<DtypeB>::value;
                        SYCL_KERNEL_ASSERT(
                            delta % align == 0 &&
                            "expected input tensor dynamic dimension byte size to be non-negative multiple of 16\n");
                    }

                    // Check the same requirement for output tensor (that is always
                    // contiguous, and in row-major layout).
                    if (N < 0)
                    {
                        int align = 128 / cutlass::sizeof_bits<DtypeOutput>::value;
                        SYCL_KERNEL_ASSERT(
                            delta % align == 0 &&
                            "expected output tensor dynamic dimension byte size to be non-negative multiple of 16\n");
                    }
                }
            }
            int32_t lda, ldb, ldoutput;
            if (M < 0)
            {
                // A and output is 2d
                SYCL_KERNEL_ASSERT(offset <= tensor_ShapeA[0] && "expected offset to be less than tensor size\n");
                real_M = delta;
                lda = a_row_major ? tensor_StrideA[0] : tensor_StrideA[1];
                ldb = b_row_major ? tensor_StrideB[1] : tensor_StrideB[2];
                ldoutput = tensor_StrideOutput[0];
                A_ptrs[tid] = tid == 0 ? A : A + offs[tid - 1] * tensor_StrideA[0];
                if (scale_A != nullptr)
                {
                    inputA_scale_ptrs[tid] = tid == 0 ? scale_A : scale_A + offs[tid - 1];
                    inputB_scale_ptrs[tid] = scale_B + tid * b_scale_stride;
                }
                output_ptrs[tid] = tid == 0 ? output : output + offs[tid - 1] * ldoutput;
                B_ptrs[tid] = B + tid * tensor_StrideB[0];
            }
            else if (N < 0)
            {
                SYCL_KERNEL_ASSERT(offset <= tensor_ShapeB[1] && "expected offset to be less than tensor size\n");
                real_N = delta;
                lda = a_row_major ? tensor_StrideA[1] : tensor_StrideA[2];
                ldb = b_row_major ? tensor_StrideB[0] : tensor_StrideB[1]; // B is transposed
                ldoutput = tensor_StrideOutput[0];
                A_ptrs[tid] = A + tid * tensor_StrideA[0];
                output_ptrs[tid] = tid == 0 ? output : output + offs[tid - 1];
                B_ptrs[tid] = tid == 0 ? B : B + offs[tid - 1] * tensor_StrideB[1];
                if (scale_A != nullptr)
                {
                    inputA_scale_ptrs[tid] = scale_A + tid * a_scale_stride;
                    inputB_scale_ptrs[tid] = tid == 0 ? scale_B : scale_B + offs[tid - 1];
                }
            }
            else if (K < 0)
            {
                SYCL_KERNEL_ASSERT(offset <= tensor_ShapeA[1] && offset <= tensor_ShapeB[0] && "expected offset to be less than tensor size\n");
                // A, B is 2d, output is 3d
                real_K = delta;
                lda = a_row_major ? tensor_StrideA[0] : tensor_StrideA[1];
                ldb = b_row_major ? tensor_StrideB[0] : tensor_StrideB[1];
                ldoutput = tensor_StrideOutput[1];
                A_ptrs[tid] = tid == 0 ? A : A + offs[tid - 1] * tensor_StrideA[1];
                B_ptrs[tid] = tid == 0 ? B : B + offs[tid - 1] * tensor_StrideB[0];
                output_ptrs[tid] = output + tid * tensor_StrideOutput[0];
                if (scale_A != nullptr)
                {
                    inputA_scale_ptrs[tid] = scale_A + tid * M;
                    inputB_scale_ptrs[tid] = scale_B + tid * N;
                }
            }
            else
            {
                // A, B, output are 3D
                lda = a_row_major ? tensor_StrideA[1] : tensor_StrideA[2];
                ldb = b_row_major ? tensor_StrideB[1] : tensor_StrideB[2];
                ldoutput = tensor_StrideOutput[1];
                A_ptrs[tid] = A + tid * tensor_StrideA[0];
                B_ptrs[tid] = B + tid * tensor_StrideB[0];
                output_ptrs[tid] = output + tid * tensor_StrideOutput[0];
                if (scale_A != nullptr)
                {
                    inputA_scale_ptrs[tid] = scale_A + tid * a_scale_stride;
                    inputB_scale_ptrs[tid] = scale_B + tid * b_scale_stride;
                }
            }

            problem_sizes[tid] = ProblemShape(real_M, real_N, real_K);

            // make_cute_packed_stride only replaces one of the stride elements with
            // one the provided values in the shape arguments
            // the indices of the src/dst depend on whether A/B are row-major
            // so constructing shape argument with two similar lda values
            // while it looks non-sensical (and it is a nonsensical shape)
            // is fine for these stride construction purposes - the one that will be used
            // for replacement is correct, the other one is ignored, and we don't have to
            // branch on whether A/B are row-major
            stride_A[tid] = cutlass::make_cute_packed_stride(StrideA{}, {lda, lda, 1});
            stride_B[tid] = cutlass::make_cute_packed_stride(StrideB{}, {ldb, ldb, 1});
            stride_output[tid] =
                cutlass::make_cute_packed_stride(StrideOutput{}, {real_M, ldoutput, 1});
        }
        PreparedGroupedGEMMDataKernel(DtypeA *A,
            DtypeB *B,
            DtypeOutput *output,
            DtypeScale *scale_A,
            DtypeScale *scale_B,
            DtypeA **A_ptrs,
            DtypeB **B_ptrs,
            DtypeOutput **output_ptrs,
            DtypeScale **inputA_scale_ptrs,
            DtypeScale **inputB_scale_ptrs,
            ProblemShape *problem_sizes,
            // Strides for cutlass, cute::Stride
            StrideA *stride_A,
            StrideB *stride_B,
            StrideOutput *stride_output,
            const int32_t *offs,
            int32_t M,
            int32_t N,
            int32_t K,
            // Original strides of the input tensors
            Strides tensor_StrideA,
            Strides tensor_StrideB,
            Strides tensor_StrideOutput,
            Strides tensor_ShapeA,
            Strides tensor_ShapeB,
            int64_t a_scale_stride,
            int64_t b_scale_stride,
            bool a_row_major = true,
            bool b_row_major = false)
            : A(A), B(B), output(output), scale_A(scale_A), scale_B(scale_B),
              A_ptrs(A_ptrs), B_ptrs(B_ptrs), output_ptrs(output_ptrs),
              inputA_scale_ptrs(inputA_scale_ptrs), inputB_scale_ptrs(inputB_scale_ptrs),
              problem_sizes(problem_sizes), stride_A(stride_A), stride_B(stride_B),
              stride_output(stride_output), offs(offs), M(M), N(N), K(K),
              tensor_StrideA(tensor_StrideA), tensor_StrideB(tensor_StrideB),
              tensor_StrideOutput(tensor_StrideOutput), tensor_ShapeA(tensor_ShapeA),
              tensor_ShapeB(tensor_ShapeB), a_scale_stride(a_scale_stride),
              b_scale_stride(b_scale_stride), a_row_major(a_row_major),
              b_row_major(b_row_major)
        {

        }
    private:
        DtypeA *A;
        DtypeB *B;
        DtypeOutput *output;
        DtypeScale *scale_A;
        DtypeScale *scale_B;
        DtypeA **A_ptrs;
        DtypeB **B_ptrs;
        DtypeOutput **output_ptrs;
        DtypeScale **inputA_scale_ptrs;
        DtypeScale **inputB_scale_ptrs;
        ProblemShape *problem_sizes;
        StrideA *stride_A;
        StrideB *stride_B;
        StrideOutput *stride_output;
        const int32_t *offs;
        int32_t M;
        int32_t N;
        int32_t K; 
        Strides tensor_StrideA;
        Strides tensor_StrideB;
        Strides tensor_StrideOutput;
        Strides tensor_ShapeA;
        Strides tensor_ShapeB;
        int64_t a_scale_stride;
        int64_t b_scale_stride;
        bool a_row_major;
        bool b_row_major;

};
} // namespace 