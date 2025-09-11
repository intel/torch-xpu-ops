#include <cstdio>
#include <cstddef>
#include <cassert>
#include <cstdlib>
#include <math.h>
#include <string>

#include <sycl/sycl.hpp>
#include <cutlasscompat.hpp>

#include <cute/tensor.hpp>

#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/print_error.hpp"
#include "cutlass/util/sycl_event_manager.hpp"
#include "cutlass/util/GPU_Clock.hpp"

#include "AttentionKernels.h"

using namespace cute;

/***************************************************************************************************
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (c) 2024 - 2024 Codeplay Software Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#include <sycl/sycl.hpp>
#include <cutlasscompat.hpp>

#include <cute/tensor.hpp>

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/sycl_event_manager.hpp"
#include "cutlass/util/GPU_Clock.hpp"

template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class TiledCopyA,
          class TB, class BStride, class BSmemLayout, class TiledCopyB,
          class TC, class CStride, class CSmemLayout, class TiledMma,
          class Alpha, class Beta> class GemmName;

template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class TiledCopyA,
          class TB, class BStride, class BSmemLayout, class TiledCopyB,
          class TC, class CStride, class CSmemLayout, class TiledMma,
          class Alpha, class Beta>
void
gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
            TA const* A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a,
            TB const* B, BStride dB, BSmemLayout sB_layout, TiledCopyB copy_b,
            TC      * C, CStride dC, CSmemLayout          , TiledMma mma,
            Alpha alpha, Beta beta)
{
  using namespace cute;

  // Preconditions
  CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});                   // (M, N, K)
  CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});                   // (BLK_M, BLK_N, BLK_K)

  CUTE_STATIC_ASSERT_V(size(copy_a) == size(mma));                     // NumThreads
  CUTE_STATIC_ASSERT_V(size(copy_b) == size(mma));                     // NumThreads

  static_assert(is_static<ASmemLayout>::value);
  static_assert(is_static<BSmemLayout>::value);
  static_assert(is_static<CSmemLayout>::value);

  CUTE_STATIC_ASSERT_V(size<0>(ASmemLayout{}) == size<0>(cta_tiler));  // BLK_M
  CUTE_STATIC_ASSERT_V(size<0>(CSmemLayout{}) == size<0>(cta_tiler));  // BLK_M
  CUTE_STATIC_ASSERT_V(size<0>(BSmemLayout{}) == size<1>(cta_tiler));  // BLK_N
  CUTE_STATIC_ASSERT_V(size<1>(CSmemLayout{}) == size<1>(cta_tiler));  // BLK_N
  CUTE_STATIC_ASSERT_V(size<1>(ASmemLayout{}) == size<2>(cta_tiler));  // BLK_K
  CUTE_STATIC_ASSERT_V(size<1>(BSmemLayout{}) == size<2>(cta_tiler));  // BLK_K

  CUTE_STATIC_ASSERT_V(congruent(select<0,2>(shape_MNK), dA));         // dA strides for shape MK
  CUTE_STATIC_ASSERT_V(congruent(select<1,2>(shape_MNK), dB));         // dB strides for shape NK
  CUTE_STATIC_ASSERT_V(congruent(select<0,1>(shape_MNK), dC));         // dC strides for shape MN

  //
  // Full and Tiled Tensors
  //

  // Represent the full tensors
  Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA); // (M,K)
  Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB); // (N,K)
  Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC); // (M,N)

  // Get the appropriate blocks for this thread block
  auto cta_coord = make_coord(cutlasscompat::work_group_id::x(), cutlasscompat::work_group_id::y(), _);  // (m,n,k)
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)

  // Shared memory buffers
  auto smemA = cutlasscompat::local_mem<TA[cosize_v<ASmemLayout>]>();
  auto smemB = cutlasscompat::local_mem<TB[cosize_v<BSmemLayout>]>();
  Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);            // (BLK_M,BLK_K)
  Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);            // (BLK_N,BLK_K)

  //
  // Partition the copying of A and B tiles across the threads
  //

  // TUTORIAL: Example of partitioning via a TiledCopy

  ThrCopy thr_copy_a = copy_a.get_slice(cutlasscompat::local_id::x());
  Tensor tAgA = thr_copy_a.partition_S(gA);                            // (CPY,CPY_M,CPY_K,k)
  Tensor tAsA = thr_copy_a.partition_D(sA);                            // (CPY,CPY_M,CPY_K)
  Tensor tArA = make_fragment_like(tAsA);                              // (CPY,CPY_M,CPY_K)

  ThrCopy thr_copy_b = copy_b.get_slice(cutlasscompat::local_id::x());
  Tensor tBgB = thr_copy_b.partition_S(gB);                            // (CPY,CPY_N,CPY_K,k)
  Tensor tBsB = thr_copy_b.partition_D(sB);                            // (CPY,CPY_N,CPY_K)
  Tensor tBrB = make_fragment_like(tBsB);                              // (CPY,CPY_N,CPY_K)

  CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA));                // CPY_M
  CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tArA));                // CPY_M
  CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tAsA));                // CPY_K
  CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tArA));                // CPY_K
  CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB));                // CPY_N
  CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBrB));                // CPY_N
  CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBsB));                // CPY_K
  CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBrB));                // CPY_K

  // Copy gmem to rmem for k_tile=0
  copy(copy_a, tAgA(_,_,_,0), tArA);
  copy(copy_b, tBgB(_,_,_,0), tBrB);
  //
  // Define A/B partitioning and C accumulators
  //

  // TUTORIAL: Example of partitioning via a TiledMMA

  ThrMMA thr_mma = mma.get_slice(cutlasscompat::local_id::x());
  Tensor tCsA = thr_mma.partition_A(sA);                               // (MMA,MMA_M,MMA_K)
  Tensor tCsB = thr_mma.partition_B(sB);                               // (MMA,MMA_N,MMA_K)
  Tensor tCgC = thr_mma.partition_C(gC);                               // (MMA,MMA_M,MMA_N)

  // Allocate registers for pipelining
  Tensor tCrA = thr_mma.make_fragment_A(tCsA);                         // (MMA,MMA_M,MMA_K)
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);                         // (MMA,MMA_N,MMA_K)
  // Allocate the accumulators -- same size as the projected data
  Tensor tCrC = thr_mma.make_fragment_C(tCgC);                         // (MMA,MMA_M,MMA_N)

  CUTE_STATIC_ASSERT_V(  shape(tCrA) ==   shape(tCsA));                // (MMA,MMA_M,MMA_K)
  CUTE_STATIC_ASSERT_V(  shape(tCrB) ==   shape(tCsB));                // (MMA,MMA_N,MMA_K)
  CUTE_STATIC_ASSERT_V(  shape(tCrC) ==   shape(tCgC));                // (MMA,MMA_M,MMA_N)
  CUTE_STATIC_ASSERT_V(size<1>(tCgC) == size<1>(tCsA));                // MMA_M
  CUTE_STATIC_ASSERT_V(size<2>(tCgC) == size<1>(tCsB));                // MMA_N
  CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB));                // MMA_K

  // Clear the accumulators
  clear(tCrC);

#if 0
  if(thread0()) {
    print("  mA : "); print(  mA); print("\n");
    print("  gA : "); print(  gA); print("\n");
    print("  sA : "); print(  sA); print("\n");
    print("tAgA : "); print(tAgA); print("\n");
    print("tAsA : "); print(tAsA); print("\n");
    print("tArA : "); print(tArA); print("\n");
  }
#endif

#if 0
  if(thread0()) {
    print("  mB : "); print(  mB); print("\n");
    print("  gB : "); print(  gB); print("\n");
    print("  sB : "); print(  sB); print("\n");
    print("tBgB : "); print(tBgB); print("\n");
    print("tBsB : "); print(tBsB); print("\n");
    print("tArA : "); print(tArA); print("\n");
  }
#endif

#if 0
  if(thread0()) {
    print("  mC : "); print(  mC); print("\n");
    print("  gC : "); print(  gC); print("\n");
    print("tCsA : "); print(tCsA); print("\n");
    print("tCsB : "); print(tCsB); print("\n");
    print("tCgC : "); print(tCgC); print("\n");
    print("tCrC : "); print(tCrC); print("\n");
  }
#endif

#if 1

  // Copy rmem to smem
  copy(tArA, tAsA);
  copy(tBrB, tBsB);
  cutlasscompat::wg_barrier();

  //
  // PIPELINED MAIN LOOP
  // TUTORIAL: Example of a gemm loop that pipelines shared memory AND register memory
  //   Data is read from global to registers, then to shared via the tA|tB partitions
  //   Data is then copied from shared to registers in multiple waves via the tC partitions
  //     and gemm(.) operates on the current register wave
  //

  // Load A, B shmem->regs for k_block=0
  copy(tCsA(_,_,0), tCrA(_,_,0));
  copy(tCsB(_,_,0), tCrB(_,_,0));
  auto K_TILE_MAX  = size<3>(tAgA);
  auto K_BLOCK_MAX = size<2>(tCrA);

  CUTE_NO_UNROLL
  for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile)
  {
    // Pipeline the k-mode of the block registers
    CUTE_UNROLL
    for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block)
    {
      if (k_block == K_BLOCK_MAX - 1)
      {
        // Copy rmem to smem
        cutlasscompat::wg_barrier();
        copy(tArA, tAsA);
        copy(tBrB, tBsB);
        cutlasscompat::wg_barrier();
      }

      // Copy smem to rmem for k_block+1
      int k_block_next = (k_block + 1) % K_BLOCK_MAX;
      copy(tCsA(_,_,k_block_next), tCrA(_,_,k_block_next));
      copy(tCsB(_,_,k_block_next), tCrB(_,_,k_block_next));
      if (k_block == 0)
      {
        // Copy gmem to rmem for k_tile+1
        int k_tile_next = (k_tile + 1 < K_TILE_MAX) ? k_tile + 1 : k_tile;
        copy(copy_a, tAgA(_,_,_,k_tile_next), tArA);
        copy(copy_b, tBgB(_,_,_,k_tile_next), tBrB);
      }
      // Thread-level register gemm for k_block
      gemm(mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);
    } // k_block
  } // k_tile

#endif

  //
  // Epilogue
  //

  axpby(alpha, tCrC, beta, tCgC);
}

// Setup params for a NT GEMM
template <class TA, class TB, class TC,
          class Alpha, class Beta>
void
gemm_nt(int m, int n, int k,
        Alpha alpha,
        TA const* A, int ldA,
        TB const* B, int ldB,
        Beta beta,
        TC      * C, int ldC)
{
  using namespace cute;

  // Define shapes (dynamic)
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(M, N, K);                     // (M, N, K)

  // Define NT strides (mixed)
  auto dA = make_stride(Int<1>{}, ldA);                      // (dM, dK)
  auto dB = make_stride(Int<1>{}, ldB);                      // (dN, dK)
  auto dC = make_stride(Int<1>{}, ldC);                      // (dM, dN)

  // Define CTA tile sizes (static)
  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int<  8>{};
  auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)

  // Define the smem layouts (static)
  auto sA = make_layout(make_shape(bM, bK));                 // (m,k) -> smem_idx; m-major
  auto sB = make_layout(make_shape(bN, bK));                 // (n,k) -> smem_idx; n-major
  auto sC = make_layout(make_shape(bM, bN));                 // (m,n) -> smem_idx; m-major

  // Define the thread layouts (static)
  TiledCopy copyA = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, TA>{},
                                    Layout<Shape<_32,_8>>{},  // Thr layout 32x8 m-major
                                    Layout<Shape< _4,_1>>{}); // Val layout  4x1 m-major
  TiledCopy copyB = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, TB>{},
                                    Layout<Shape<_32,_8>>{},  // Thr layout 32x8 n-major
                                    Layout<Shape< _4,_1>>{}); // Val layout  4x1 n-major

  TiledMMA mmaC = make_tiled_mma(UniversalFMA<TC,TA,TB>{},
                                 Layout<Shape<_16,_16,_1>>{});  // 16x16x1 TiledMMA

#if 0
  print(copyA);
  print(copyB);
  print(mmaC);
#endif

#if 0
  print_latex(copyA);
  print_latex(copyB);
  print_latex(mmaC);
#endif

  auto dimBlock = cutlasscompat::dim3(size(mmaC));
  auto dimGrid  = cutlasscompat::dim3(size(ceil_div(M, bM)), size(ceil_div(N, bN)));
  auto event = cutlasscompat::launch<
      gemm_device<decltype(prob_shape), decltype(cta_tiler),
                  TA, decltype(dA), decltype(sA), decltype(copyA),
                  TB, decltype(dB), decltype(sB), decltype(copyB),
                  TC, decltype(dC), decltype(sC), decltype(mmaC),
                  Alpha, Beta>,
                  GemmName<decltype(prob_shape), decltype(cta_tiler),
                  TA, decltype(dA), decltype(sA), decltype(copyA),
                  TB, decltype(dB), decltype(sB), decltype(copyB),
                  TC, decltype(dC), decltype(sC), decltype(mmaC),
                  Alpha, Beta>>(dimGrid, dimBlock, prob_shape, cta_tiler,
                    A, dA, sA, copyA,
                    B, dB, sB, copyB,
                    C, dC, sC, mmaC,
                    alpha, beta);
  EventManager::getInstance().addEvent(event);
}

// Setup params for a TN GEMM
template <class TA, class TB, class TC,
          class Alpha, class Beta>
void
gemm_tn(int m, int n, int k,
        Alpha alpha,
        TA const* A, int ldA,
        TB const* B, int ldB,
        Beta beta,
        TC      * C, int ldC)
{
  using namespace cute;

  // Define shapes (dynamic)
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(M, N, K);                     // (M, N, K)

  // Define TN strides (mixed)
  auto dA = make_stride(ldA, Int<1>{});                      // (dM, dK)
  auto dB = make_stride(ldB, Int<1>{});                      // (dN, dK)
  auto dC = make_stride(Int<1>{}, ldC);                      // (dM, dN)

  // Define CTA tile sizes (static)
  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int<  8>{};
  auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)

  // Define the smem layouts (static)
  auto sA = make_layout(make_shape (      bM,          bK),
                        make_stride(Int<1>{}, bM+Int<1>{}));        // (m,k) -> smem_idx; padded m-major
  auto sB = make_layout(make_shape (      bN,          bK),
                        make_stride(Int<1>{}, bN+Int<1>{}));        // (n,k) -> smem_idx; padded n-major
  auto sC = make_layout(make_shape(bM, bN));                        // (m,n) -> smem_idx

  // Define the thread layouts (static)

  TiledCopy copyA = make_tiled_copy(Copy_Atom<UniversalCopy<TA>, TA>{},
                                    Layout<Shape<_32,_8>,Stride<_8,_1>>{}, // Thr layout 32x8 k-major
                                    Layout<Shape< _1,_1>>{});              // Val layout  1x1
  TiledCopy copyB = make_tiled_copy(Copy_Atom<UniversalCopy<TB>, TB>{},
                                    Layout<Shape<_32,_8>,Stride<_8,_1>>{}, // Thr layout 32x8 k-major
                                    Layout<Shape< _1,_1>>{});              // Val layout  1x1

  TiledMMA mmaC = make_tiled_mma(UniversalFMA<TC,TA,TB>{},
                                 Layout<Shape<_16,_16,_1>>{});  // 16x16x1 TiledMMA

#if 0
  print(copyA);
  print(copyB);
  print(mmaC);
#endif

#if 0
  print_latex(copyA);
  print_latex(copyB);
  print_latex(mmaC);
#endif

  auto dimBlock = cutlasscompat::dim3(size(mmaC));
  auto dimGrid  = cutlasscompat::dim3(size(ceil_div(M, bM)), size(ceil_div(N, bN)));
  auto event = cutlasscompat::launch<
      gemm_device<decltype(prob_shape), decltype(cta_tiler),
                  TA, decltype(dA), decltype(sA), decltype(copyA),
                  TB, decltype(dB), decltype(sB), decltype(copyB),
                  TC, decltype(dC), decltype(sC), decltype(mmaC),
                  Alpha, Beta>, GemmName<decltype(prob_shape), decltype(cta_tiler),
                  TA, decltype(dA), decltype(sA), decltype(copyA),
                  TB, decltype(dB), decltype(sB), decltype(copyB),
                  TC, decltype(dC), decltype(sC), decltype(mmaC),
                  Alpha, Beta>>(dimGrid, dimBlock, prob_shape, cta_tiler,
                    A, dA, sA, copyA,
                    B, dB, sB, copyB,
                    C, dC, sC, mmaC,
                    alpha, beta);
  EventManager::getInstance().addEvent(event);
}

template <class TA, class TB, class TC,
          class Alpha, class Beta>
void
gemm(char transA, char transB, int m, int n, int k,
     Alpha alpha,
     TA const* A, int ldA,
     TB const* B, int ldB,
     Beta beta,
     TC      * C, int ldC)
{
  if (transA == 'N' && transB == 'T') {
    return gemm_nt(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC);
  } else
  if (transA == 'T' && transB == 'N') {
    return gemm_tn(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC);
  }
  assert(false && "Not implemented");
}


int temp_entry()
{
  int m = 5120;
  int n = 5120;
  int k = 4096;
  char transA = 'N';
  char transB = 'T';

  using TA = float;
  using TB = float;
  using TC = float;
  using TI = float;

  TI alpha = 1.0;
  TI beta  = 0.0;

  std::cout << "M = " << m << std::endl;
  std::cout << "N = " << n << std::endl;
  std::cout << "K = " << k << std::endl;
  std::cout << "C = A^" << transA << " B^" << transB << std::endl;

  std::vector<TA> h_A(m*k);
  std::vector<TB> h_B(n*k);
  std::vector<TC> h_C(m*n);

  for (int j = 0; j < m*k; ++j) h_A[j] = static_cast<TA>( 2*(rand() / double(RAND_MAX)) - 1 );
  for (int j = 0; j < n*k; ++j) h_B[j] = static_cast<TB>( 2*(rand() / double(RAND_MAX)) - 1 );
  for (int j = 0; j < m*n; ++j) h_C[j] = static_cast<TC>(-1);

  auto d_A = cutlasscompat::malloc<TA>(m*k);
  auto d_B = cutlasscompat::malloc<TB>(k*n);
  auto d_C = cutlasscompat::malloc<TC>(m*n);

  cutlasscompat::memcpy<TA>(d_A, h_A.data(), m*k);
  cutlasscompat::memcpy<TB>(d_B, h_B.data(), k*n);
  cutlasscompat::memcpy<TC>(d_C, h_C.data(), m*n);

  double gflops = (2.0*m*n*k) * 1e-9;

  const int timing_iterations = 100;
  GPU_Clock timer;

  int ldA = 0, ldB = 0, ldC = m;

  if (transA == 'N') {
    ldA = m;
  } else if (transA == 'T') {
    ldA = k;
  } else {
    assert(false);
  }

  if (transB == 'N') {
    ldB = k;
  } else if (transB == 'T') {
    ldB = n;
  } else {
    assert(false);
  }

  // Run once
  gemm(transA, transB, m, n, k,
       alpha,
       d_A, ldA,
       d_B, ldB,
       beta,
       d_C, ldC);
  cutlasscompat::wait_and_throw();

  // Timing iterations
  timer.start();
  for (int i = 0; i < timing_iterations; ++i) {
    gemm(transA, transB, m, n, k,
         alpha,
         d_A, ldA,
         d_B, ldB,
         beta,
         d_C, ldC);
  }
  double cute_time = timer.seconds() / timing_iterations;
  printf("CUTE_GEMM:     [%6.1f]GFlop/s  (%6.4f)ms\n", gflops / cute_time, cute_time*1000);

  return 0;
}


// using ProblemShapeRegular = cute::tuple<int, int, int, int, int, int, int>; // batch, num_head_q,num_head_kv,seq_len_qo,seq_len_kv,head_size_qk,head_size_vo
// template <typename T>
// struct OPS_tobf16{
//     template <class Tensor>
//     auto operator()(Tensor &src){
//         cutlass::NumericConverter<
//             T, float, cutlass::FloatRoundStyle::round_toward_zero> converter;
//         auto dst = make_tensor_like<T>(src);
//         CUTLASS_PRAGMA_UNROLL
//         for (int i = 0; i < size(src); ++i) {
//             dst(i) = converter(src(i));
//         }
//         return dst;
//     }
// };

// constexpr int tid = 0;
// constexpr int bid = 16;
// using MASKType = uint16_t;
// template <class T_, class ProblemShape_>
// struct MHA_TYPE {
//     using ProblemShape = ProblemShape_;
//     /*
//       Q BATCH,NUM_HEAD_Q,SEQ_LEN_QO,HEAD_SIZE_QK
//       K BATCH,NUM_HEAD_KV,SEQ_LEN_KV,HEAD_SIZE_QK
//       V BATCH,NUM_HEAD_KV,SEQ_LEN_KV,HEAD_SIZE_VO
//       P BATCH,NUM_HEAD_Q,SEQ_LEN_QO,SEQ_LEN_KV
//       O BATCH,NUM_HEAD_Q,SEQ_LEN_QO,HEAD_SIZE_VO
//     */
//     /*
//       dPs BATCH,NUM_HEAD_Q,SEQ_LEN_QO,SEQ_LEN_KV
//       dP BATCH,NUM_HEAD_Q,SEQ_LEN_QO,SEQ_LEN_KV
//       dP=softmax_backward(softmax, dPs)
//     */
//     using DType = T_;
//     using Stride1 = cute::tuple<long, cute::C<1>, long>;
//     using Stride0 = cute::tuple<cute::C<1>, long, long>;

//     static constexpr int bMi = 256;
//     static constexpr int bNi = 256;
//     static constexpr int bKi = 32;
//     static constexpr auto bM = Int<bMi>{};
//     static constexpr auto bN = Int<bNi>{};
//     static constexpr auto bK = Int<bKi>{};
//     static constexpr auto tile_mnk = make_shape(bM, bN, bK);
//     static constexpr auto bP = Int<2>{}; // Pipeline

//     /*
//       shape
//       Pt BATCH,NUM_HEAD_Q,SEQ_LEN_KV,SEQ_LEN_QO
//       dO BATCH,NUM_HEAD_Q,SEQ_LEN_QO,HEAD_SIZE_VO
//       dV BATCH,NUM_HEAD_KV,SEQ_LEN_KV,HEAD_SIZE_VO
//       M SEQ_LEN_KV
//       N HEAD_SIZE_VO
//       K SEQ_LEN_QO
//       dV=Pt*dO
//     */
//     using CopyPt = decltype(
//         make_tiled_copy(Copy_Atom<Copy_Traits<XE_2D_U16x16x16_LD_T, Stride0>, DType>{},
//                         Layout<Shape<_1,_16>>{}, // Thr layout 1x16 k-major
//                         Layout<Shape<_16,_1>>{})); // Val layout  16x1
//     using CopygOB = decltype(
//         make_tiled_copy(Copy_Atom<Copy_Traits<XE_2D_U16x32x32_LD_N, Stride0>, DType>{},
//                         Layout<Shape<_1,_16>>{}, // Thr layout 1x16 n-major
//                         Layout<Shape<_32,_2>>{})); // Val layout  32x2);
//     using CopygV = decltype(
//         make_tiled_copy(Copy_Atom<Copy_Traits<XE_2D_U16x8x16_ST_N, Stride1>, DType>{},
//                         Layout<Shape<_1,_16>>{}, // Thr layout 1x16 n-major
//                         Layout<Shape<_8,_1>>{})); // Val layout  8x1
//     /*
//       dO BATCH,NUM_HEAD_Q,SEQ_LEN_QO,HEAD_SIZE_VO
//       dV BATCH,NUM_HEAD_KV,HEAD_SIZE_VO,SEQ_LEN_KV
//       dP BATCH,NUM_HEAD_Q,SEQ_LEN_QO,SEQ_LEN_KV
//       M SEQ_LEN_QO
//       N SEQ_LEN_KV
//       K HEAD_SIZE_VO
//       dP=dO*Vt
//     */

//     using CopygOA = decltype(
//         make_tiled_copy(Copy_Atom<Copy_Traits<XE_2D_U16x32x32_LD_N, Stride1>, DType>{},
//                         Layout<Shape<_1,_16>>{}, // Thr layout 1x16 k-major
//                         Layout<Shape<_32,_2>>{}));              // Val layout  32x2
//     using CopyVt = decltype(
//         make_tiled_copy(Copy_Atom<Copy_Traits<XE_2D_U16x16x16_LD_T, Stride1>, DType>{},
//                         Layout<Shape<_1,_16>>{}, // Thr layout 1x16 n-major
//                         Layout<Shape<_16,_1>>{})); //Val layout 16x1
//     using CopygP = decltype(
//         make_tiled_copy(Copy_Atom<Copy_Traits<XE_2D_U16x8x16_ST_N, Stride1>, DType>{},
//                         Layout<Shape<_1,_16>>{}, // Thr layout 1x16 n-major
//                         Layout<Shape<_8,_1>>{}));              // Val layout  8x1
//     using CopyP = decltype(
//         make_tiled_copy(Copy_Atom<Copy_Traits<XE_2D_U16x8x16_LD_N, Stride1>, DType>{},
//                         Layout<Shape<_1, _16>>{},
//                         Layout<Shape<_8, _1>> {}));
//     /*
//      * dS BATCH,NUM_HEAD_Q,SEQ_LEN_QO,SEQ_LEN_KV
//      * Q BATCH,NUM_HEAD_Q,SEQ_LEN_QO,HEAD_SIZE_QK
//      * dK BATCH,NUM_HEAD_KV,SEQ_LEN_KV,HEAD_SIZE_QK
//      * M SEQ_LEN_KV
//      * N HEAD_SIZE_QK
//      * K SEQ_LEN_QO
//      * dK=dSt*Q
//      */
//     using CopygSt = decltype(
//         make_tiled_copy(Copy_Atom<Copy_Traits<XE_2D_U16x16x16_LD_T, Stride0>, DType>{},
//                         Layout<Shape<_1,_16>>{}, // Thr layout 1x16 k-major
//                         Layout<Shape<_16,_1>>{})); // Val layout  16x1
//     using CopyQ = decltype(
//         make_tiled_copy(Copy_Atom<Copy_Traits<XE_2D_U16x32x32_LD_N, Stride0>, DType>{},
//                         Layout<Shape<_1,_16>>{}, // Thr layout 1x16 n-major
//                         Layout<Shape<_32,_2>>{}));              // Val layout  16x1
//     using CopygK = decltype(
//         make_tiled_copy(Copy_Atom<Copy_Traits<XE_2D_U16x8x16_ST_N, Stride1>, DType>{},
//                         Layout<Shape<_1,_16>>{}, // Thr layout 1x16 n-major
//                         Layout<Shape<_8,_1>>{}));              // Val layout  8x1
//     /*
//      * dS BATCH,NUM_HEAD_Q,SEQ_LEN_QO,SEQ_LEN_KV
//      * K BATCH,NUM_HEAD_KV,SEQ_LEN_KV,HEAD_SIZE_QK
//      * dQ BATCH,NUM_HEAD_Q,SEQ_LEN_QO,HEAD_SIZE_QK
//      * M SEQ_LEN_QO
//      * N HEAD_SIZE_QK
//      * K SEQ_LEN_KV
//      * dQ=dS*K
//      */
//     using CopygS = decltype(
//         make_tiled_copy(Copy_Atom<Copy_Traits<XE_2D_U16x32x32_LD_N, Stride1>, DType>{},
//                         Layout<Shape<_1,_16>>{}, // Thr layout 1x16 k-major
//                         Layout<Shape<_32,_2>>{}));              // Val layout  32x2
//     using CopyK = decltype(
//         make_tiled_copy(Copy_Atom<Copy_Traits<XE_2D_U16x32x32_LD_N, Stride0>, DType>{},
//                         Layout<Shape<_1,_16>>{}, // Thr layout 1x16 n-major
//                         Layout<Shape<_32,_2>>{}));              // Val layout  32x2
//     using CopygQ = decltype(
//         make_tiled_copy(Copy_Atom<Copy_Traits<XE_2D_U16x8x16_ST_N, Stride1>, DType>{},
//                         Layout<Shape<_1,_16>>{}, // Thr layout 1x16 n-major
//                         Layout<Shape<_8,_1>>{}));              // Val layout  8x1
//     using MMA_Atom_Arch = std::conditional_t<
//         std::is_same_v<T_, cutlass::half_t>,
//         MMA_Atom<XE_8x16x16_F32F16F16F32_TT>,
//         MMA_Atom<XE_8x16x16_F32BF16BF16F32_TT>>;
//     static constexpr TiledMMA mmaC = typename TiledMMAHelper<MMA_Atom_Arch, Layout<decltype(tile_mnk)>,
//                                                              Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA{};  // 256x128x16 TiledMMA
//     using TiledMma = decltype(mmaC);
//     static constexpr int SubgroupSize = 16;
//     static constexpr int smem_size = 0;

//     MHA_TYPE(ProblemShape_ mha_shape_)
//         :mha_shape(mha_shape_),
//          BATCH(get<0>(mha_shape)),
//          NUM_HEAD_Q(get<1>(mha_shape)),
//          NUM_HEAD_KV(get<2>(mha_shape)),
//          SEQ_LEN_QO(get<3>(mha_shape)),
//          SEQ_LEN_KV(get<4>(mha_shape)),
//          HEAD_SIZE_QK(get<5>(mha_shape)),
//          HEAD_SIZE_VO(get<6>(mha_shape)),
//          max_block_m(std::max(SEQ_LEN_KV, SEQ_LEN_QO)),
//          max_block_n(NUM_HEAD_KV),
//          block_n_chunk(NUM_HEAD_Q / NUM_HEAD_KV)
//         {}

//     // variables
//     ProblemShape mha_shape;
//     TiledMma tiled_mma;
//     const int64_t BATCH;
//     const int64_t NUM_HEAD_Q;
//     const int64_t NUM_HEAD_KV;
//     const int64_t SEQ_LEN_QO;
//     const int64_t SEQ_LEN_KV;
//     const int64_t HEAD_SIZE_QK;
//     const int64_t HEAD_SIZE_VO;
//     const int64_t max_block_m;
//     const int64_t max_block_n;
//     const int64_t block_n_chunk;
// };

// template<class T, class ProblemShape, class ThrMma,
//          class YStride, class TiledCopyY, class RTensor>
// void load_y(T &trait, ProblemShape &shape_mnkl, ThrMma &thr_mma,
//             typename T::DType const *Y, YStride dY, TiledCopyY,
//             RTensor &t,
//             const int m_coord, const int n_coord, const int lq_coord) {
//     auto Y_shape = select<0,1,3>(shape_mnkl);
//     auto mY = make_tensor(make_gmem_ptr(Y), make_layout(Y_shape, dY));
//     auto copy_caux = TiledCopyY{mY};
//     Tensor mY_coord = cute::get_xe_tensor(Y_shape);   //(m,n,l)
//     Tensor gY = local_tile(mY_coord, select<0, 1>(trait.tile_mnk),
//                            make_coord(m_coord, n_coord, lq_coord)); // aux
//     // copy y
//     Tensor tCgy = thr_mma.partition_C(gY);
//     copy(copy_caux, tCgy, t);
// }

// template<class FragTensor,
//          class AuxTensor,
//          class SumTensor>
// void softmax_bwd_partial_sum(FragTensor &tCrC,
//                              AuxTensor &tCrCy,
//                              SumTensor &sum_row) {
//     // calculate sum of dy*y
//     CUTLASS_PRAGMA_UNROLL
//     for (int i = 0; i < size<2>(tCrC); ++i) {
//         auto dy_col = tCrC(_, _, i);
//         auto y_col = tCrCy(_, _, i);
//         CUTLASS_PRAGMA_UNROLL
//         for (int j = 0; j < size(dy_col); ++j) {
//             dy_col(j) = dy_col(j) * y_col(j);
//             sum_row(j) = sum_row(j) + dy_col(j);
//         }
//     }
// }

// template<class T,
//          class FragTensor, class AuxTensor,
//          class STensor>
// void softmax_bwd_last(T &trait,
//                       FragTensor &tCrC,
//                       AuxTensor &tCrCy,
//                       STensor &sum_buf,
//                       const float softmax_coef) {
//     // copy y
//     for (int i = 0; i < size<2>(tCrC); ++i) {
//         auto y_col = tCrCy(_, _, i);
//         auto ydy_col = tCrC(_, _, i);
//         for (int j = 0; j < size(y_col); ++j) {
//             auto sum_val = sum_buf(j);
//             ydy_col(j) = (ydy_col(j) - y_col(j) * sum_val) * softmax_coef;
//         }
//     }
// }

// template<typename T, class ProblemShape, class ThrMma,
//          class AStride, class TiledCopyA,
//          class BStride, class TiledCopyB,
//          class AccT>
// void gemm_kernel(T &trait, ProblemShape &shape_mnkl,
//                  ThrMma &thr_mma,
//                  typename T::DType const *A, AStride dA, TiledCopyA,
//                  typename T::DType const *B, BStride dB, TiledCopyB,
//                  AccT &tCrC, const int m_coord, const int n_coord, const int l_coord,
//                  bool debug) {

//     auto A_shape = select<0,2,3>(shape_mnkl);
//     auto B_shape = select<1,2,3>(shape_mnkl);

//     // Represent the full tensors
//     auto mA = make_tensor(make_gmem_ptr(A), make_layout(A_shape, dA));
//     auto mB = make_tensor(make_gmem_ptr(B), make_layout(B_shape, dB));

//     auto copy_a = TiledCopyA{mA};
//     auto copy_b = TiledCopyB{mB};

//     Tensor mA_coord = cute::get_xe_tensor(A_shape);   //(m,k,l)
//     Tensor mB_coord = cute::get_xe_tensor(B_shape);   //(n,k,l)

//     Tensor gA = local_tile(mA_coord, select<0, 2>(trait.tile_mnk), make_coord(m_coord, _, l_coord));  // (BLK_M,BLK_K,k)
//     Tensor gB = local_tile(mB_coord, select<1, 2>(trait.tile_mnk), make_coord(n_coord, _, l_coord));  // (BLK_N,BLK_K,k)

//     // Partition global counting tensors for MMA
//     Tensor tCgA = thr_mma.partition_A(gA);
//     Tensor tCgB = thr_mma.partition_B(gB);

//     Tensor tCrA = make_tensor<typename T::DType>(make_fragment_layout(copy_a, tCgA(_,_,_,0).shape()));
//     Tensor tCrB = make_tensor<typename T::DType>(make_fragment_layout(copy_b, tCgB(_,_,_,0).shape()));

//     ThrCopy thr_copy_a = copy_a.get_slice(cutlasscompat::local_id::x());
//     ThrCopy thr_copy_b = copy_b.get_slice(cutlasscompat::local_id::x());

//     // Retile registers for copies
//     Tensor tArA = thr_copy_a.retile_D(tCrA);
//     Tensor tBrB = thr_copy_b.retile_D(tCrB);

//     // Retile global counting tensors for copies
//     Tensor tAgA = thr_copy_a.retile_S(tCgA);
//     Tensor tBgB = thr_copy_b.retile_S(tCgB);

//     //
//     // PREFETCH
//     //

//     // constexpr int Num_SGs = size(tiled_mma);
//     static constexpr auto ATOM_M = get<1>(typename T::TiledMma::ThrLayoutVMNK{}.shape());
//     static constexpr auto ATOM_N = get<2>(typename T::TiledMma::ThrLayoutVMNK{}.shape());
//     static constexpr auto ATOM_K = get<3>(typename T::TiledMma::ThrLayoutVMNK{}.shape());
//     static constexpr auto Num_SGs = ATOM_N * ATOM_M * ATOM_K;

//     static constexpr auto BLK_M = get<0>(T::tile_mnk);
//     static constexpr auto BLK_N = get<1>(T::tile_mnk);
//     static constexpr auto BLK_K = get<2>(T::tile_mnk);

//     auto prefetch_a = cute::prefetch_selector<Shape<Int<BLK_M>,Int<BLK_K>>, Num_SGs>(copy_a);
//     auto prefetch_b = cute::prefetch_selector<Shape<Int<BLK_N>,Int<BLK_K>>, Num_SGs>(copy_b);
//     int thread_idx = int(ThreadIdxX());
//     auto thr_prefetch_A = prefetch_a.get_slice(thread_idx);
//     auto thr_prefetch_B = prefetch_b.get_slice(thread_idx);

//     // Partition global tile for prefetch
//     auto pAgA = thr_prefetch_A.partition_S(gA);
//     auto pBgB = thr_prefetch_B.partition_S(gB);

//     int prefetch_k = 0;

//     constexpr int barrier_scope = 2;
//     int k_tile_count = ceil_div(get<2>(shape_mnkl), get<2>(trait.tile_mnk));
//     auto stages = trait.bP;
//     CUTLASS_PRAGMA_UNROLL
//     for (; prefetch_k < stages; prefetch_k++) {
//         if (prefetch_k < k_tile_count) {
//             prefetch(prefetch_a, pAgA(_, _, _, prefetch_k));
//             prefetch(prefetch_b, pBgB(_, _, _, prefetch_k));
//         }
//     }

//     CUTLASS_PRAGMA_UNROLL
//     for (int k_tile = 0; k_tile < k_tile_count; k_tile++, prefetch_k++) {
//         barrier_arrive(barrier_scope);
//         // Copy gmem to rmem for the first k_tile
//         copy(copy_a, tAgA(_,_,_,k_tile), tArA);
//         copy(copy_b, tBgB(_,_,_,k_tile), tBrB);

//         if (prefetch_k < k_tile_count) {
//             prefetch(prefetch_a, pAgA(_, _, _, prefetch_k));
//             prefetch(prefetch_b, pBgB(_, _, _, prefetch_k));
//         }

//         cute::gemm(trait.tiled_mma, tCrA, tCrB, tCrC);
//         barrier_wait(barrier_scope);

//     }
// }

// template<typename T, class ProblemShape, class ThrMma,
//          class AStride, class TiledCopyA,
//          class BStride, class TiledCopyB,
//          class AccT>
// void gemm_kernel_bb(T &trait, ProblemShape &shape_mnkl,
//                     ThrMma &thr_mma,
//                     typename T::DType const *A, AStride dA, TiledCopyA,
//                     typename T::DType const *B, BStride dB, TiledCopyB,
//                     AccT &tCrC, const int m_coord, const int n_coord, const int lq_coord, const int lk_coord,
//                     bool debug) {

//     auto A_shape = select<0,2,3>(shape_mnkl);
//     auto B_shape = select<1,2,4>(shape_mnkl);

//     // Represent the full tensors
//     auto mA = make_tensor(make_gmem_ptr(A), make_layout(A_shape, dA));
//     auto mB = make_tensor(make_gmem_ptr(B), make_layout(B_shape, dB));

//     auto copy_a = TiledCopyA{mA};
//     auto copy_b = TiledCopyB{mB};

//     Tensor mA_coord = cute::get_xe_tensor(A_shape);   //(m,k,l)
//     Tensor mB_coord = cute::get_xe_tensor(B_shape);   //(n,k,lh)


//     Tensor gA = local_tile(mA_coord, select<0, 2>(trait.tile_mnk), make_coord(m_coord, _, lq_coord));  // (BLK_M,BLK_K,k)
//     Tensor gB = local_tile(mB_coord, select<1, 2>(trait.tile_mnk), make_coord(n_coord, _, lk_coord));  // (BLK_N,BLK_K,k)

//     // Partition global counting tensors for MMA
//     Tensor tCgA = thr_mma.partition_A(gA);
//     Tensor tCgB = thr_mma.partition_B(gB);

//     Tensor tCrA = make_tensor<typename T::DType>(make_fragment_layout(copy_a, tCgA(_,_,_,0).shape()));
//     Tensor tCrB = make_tensor<typename T::DType>(make_fragment_layout(copy_b, tCgB(_,_,_,0).shape()));

//     ThrCopy thr_copy_a = copy_a.get_slice(cutlasscompat::local_id::x());
//     ThrCopy thr_copy_b = copy_b.get_slice(cutlasscompat::local_id::x());

//     // Retile registers for copies
//     Tensor tArA = thr_copy_a.retile_D(tCrA);
//     Tensor tBrB = thr_copy_b.retile_D(tCrB);

//     // Retile global counting tensors for copies
//     Tensor tAgA = thr_copy_a.retile_S(tCgA);
//     Tensor tBgB = thr_copy_b.retile_S(tCgB);

//     //
//     // PREFETCH
//     //

//     // constexpr int Num_SGs = size(tiled_mma);
//     static constexpr auto ATOM_M = get<1>(typename T::TiledMma::ThrLayoutVMNK{}.shape());
//     static constexpr auto ATOM_N = get<2>(typename T::TiledMma::ThrLayoutVMNK{}.shape());
//     static constexpr auto ATOM_K = get<3>(typename T::TiledMma::ThrLayoutVMNK{}.shape());
//     static constexpr auto Num_SGs = ATOM_N * ATOM_M * ATOM_K;

//     static constexpr auto BLK_M = get<0>(T::tile_mnk);
//     static constexpr auto BLK_N = get<1>(T::tile_mnk);
//     static constexpr auto BLK_K = get<2>(T::tile_mnk);

//     auto prefetch_a = cute::prefetch_selector<Shape<Int<BLK_M>,Int<BLK_K>>, Num_SGs>(copy_a);
//     auto prefetch_b = cute::prefetch_selector<Shape<Int<BLK_N>,Int<BLK_K>>, Num_SGs>(copy_b);
//     int thread_idx = int(ThreadIdxX());
//     auto thr_prefetch_A = prefetch_a.get_slice(thread_idx);
//     auto thr_prefetch_B = prefetch_b.get_slice(thread_idx);

//     // Partition global tile for prefetch
//     auto pAgA = thr_prefetch_A.partition_S(gA);
//     auto pBgB = thr_prefetch_B.partition_S(gB);

//     int prefetch_k = 0;

//     constexpr int barrier_scope = 2;
//     int k_tile_count = ceil_div(get<2>(shape_mnkl), get<2>(trait.tile_mnk));
//     auto stages = trait.bP;
//     CUTLASS_PRAGMA_UNROLL
//     for (; prefetch_k < stages; prefetch_k++) {
//         if (prefetch_k < k_tile_count) {
//             prefetch(prefetch_a, pAgA(_, _, _, prefetch_k));
//             prefetch(prefetch_b, pBgB(_, _, _, prefetch_k));
//         }
//     }

//     CUTLASS_PRAGMA_UNROLL
//     for (int k_tile = 0; k_tile < k_tile_count; k_tile++, prefetch_k++) {
//         barrier_arrive(barrier_scope);
//         // Copy gmem to rmem for the first k_tile
//         copy(copy_a, tAgA(_,_,_,k_tile), tArA);
//         copy(copy_b, tBgB(_,_,_,k_tile), tBrB);

//         if (prefetch_k < k_tile_count) {
//             prefetch(prefetch_a, pAgA(_, _, _, prefetch_k));
//             prefetch(prefetch_b, pBgB(_, _, _, prefetch_k));
//         }

//         cute::gemm(trait.tiled_mma, tCrA, tCrB, tCrC);
//         barrier_wait(barrier_scope);

//     }
// }

// template<typename T, class ProblemShape, class ThrMma,
//          class AStride, class TiledCopyA,
//          class BStride, class TiledCopyB,
//          class CStride, class TiledCopyC>
// void gemm(T &trait, ProblemShape &shape_mnkl, ThrMma &thr_mma,
//           typename T::DType const *A, AStride dA, TiledCopyA tiledcopy_A,
//           typename T::DType const *B, BStride dB, TiledCopyB tiledcopy_B,
//           typename T::DType const *C, CStride dC, TiledCopyC tiledcopy_C,
//           const int m_coord, const int l_coord,
//           bool debug) {
//     auto C_shape = select<0,1,3>(shape_mnkl);
//     auto mC = make_tensor(make_gmem_ptr(C), make_layout(C_shape, dC));
//     auto copy_c = TiledCopyC{mC};
//     Tensor mC_coord = cute::get_xe_tensor(C_shape);   //(m,n,l)
//     OPS_tobf16<typename T::DType> op;
//     if (m_coord < ceil_div(size<0>(shape_mnkl), trait.bM)) {
//         for (int n_coord = 0; n_coord < ceil_div(size<1>(shape_mnkl), trait.bN); ++n_coord) {
//             Tensor gC = local_tile(mC_coord, select<0, 1>(trait.tile_mnk), make_coord(m_coord, n_coord, l_coord));  // (BLK_M,BLK_N)
//             Tensor tCgC = thr_mma.partition_C(gC);
//             Tensor tCrC = partition_fragment_C(trait.tiled_mma, take<0,2>(trait.tile_mnk));
//             clear(tCrC);
//             gemm_kernel(trait, shape_mnkl, thr_mma,
//                         A, dA, tiledcopy_A,
//                         B, dB, tiledcopy_B,
//                         tCrC,
//                         m_coord, n_coord, l_coord,
//                         debug);
//             auto tCrC_bf16 = op(tCrC);
//             copy(copy_c, tCrC_bf16, tCgC);
//         }
//     }
// }

// template<typename T, class ProblemShape, class ThrMma,
//          class AStride, class TiledCopyA,
//          class BStride, class TiledCopyB,
//          class CStride, class TiledCopyC>
// void gemm_dq(T &trait, ProblemShape &shape_mnkl, ThrMma &thr_mma,
//              typename T::DType const *A, AStride dA, TiledCopyA tiledcopy_A,
//              typename T::DType const *B, BStride dB, TiledCopyB tiledcopy_B,
//              typename T::DType const *C, CStride dC, TiledCopyC tiledcopy_C,
//              const int m_coord, const int lq_coord, const int lk_coord,
//              bool debug) {
//     auto C_shape = select<0,1,3>(shape_mnkl);
//     auto mC = make_tensor(make_gmem_ptr(C), make_layout(C_shape, dC));
//     auto copy_c = TiledCopyC{mC};
//     Tensor mC_coord = cute::get_xe_tensor(C_shape);   //(m,n,l)
//     OPS_tobf16<typename T::DType> op;
//     if (m_coord < ceil_div(size<0>(shape_mnkl), trait.bM)) {
//         for (int n_coord = 0; n_coord < ceil_div(size<1>(shape_mnkl), trait.bN); ++n_coord) {
//             Tensor gC = local_tile(mC_coord, select<0, 1>(trait.tile_mnk), make_coord(m_coord, n_coord, lq_coord));  // (BLK_M,BLK_N)
//             Tensor tCgC = thr_mma.partition_C(gC);
//             Tensor tCrC = partition_fragment_C(trait.tiled_mma, take<0,2>(trait.tile_mnk));
//             clear(tCrC);
//             gemm_kernel_bb(trait, shape_mnkl, thr_mma,
//                            A, dA, tiledcopy_A,
//                            B, dB, tiledcopy_B,
//                            tCrC,
//                            m_coord, n_coord, lq_coord, lk_coord,
//                            debug);
//             auto tCrC_bf16 = op(tCrC);
//             copy(copy_c, tCrC_bf16, tCgC);
//         }
//     }
// }

// template<typename T, class ProblemShape, class ThrMma,
//          class AStride, class TiledCopyA,
//          class BStride, class TiledCopyB,
//          class CStride, class TiledCopyC>
// void gemm_dkv(T &trait, ProblemShape &shape_mnkl, ThrMma &thr_mma,
//               typename T::DType const *A, AStride dA, TiledCopyA tiledcopy_A,
//               typename T::DType const *B, BStride dB, TiledCopyB tiledcopy_B,
//               typename T::DType const *C, CStride dC, TiledCopyC tiledcopy_C,
//               const int m_coord, const int lh_coord, const int lb_coord,
//               bool debug) {
//     auto C_shape = select<0,1,3>(shape_mnkl);
//     auto mC = make_tensor(make_gmem_ptr(C), make_layout(C_shape, dC));
//     auto copy_c = TiledCopyC{mC};
//     Tensor mC_coord = cute::get_xe_tensor(C_shape);   //(m,n,l)
//     OPS_tobf16<typename T::DType> op;
//     if (m_coord < ceil_div(size<0>(shape_mnkl), trait.bM)) {
//         for (int n_coord = 0; n_coord < ceil_div(size<1>(shape_mnkl), trait.bN); ++n_coord) {
//             Tensor gC = local_tile(mC_coord, select<0, 1>(trait.tile_mnk), make_coord(m_coord, n_coord, lh_coord + lb_coord * trait.max_block_n));  // (BLK_M,BLK_N)
//             Tensor tCgC = thr_mma.partition_C(gC);
//             Tensor tCrC = partition_fragment_C(trait.tiled_mma, take<0,2>(trait.tile_mnk));
//             clear(tCrC);
//             for (int h_q = lh_coord * trait.block_n_chunk; h_q < (lh_coord + 1) * trait.block_n_chunk; ++h_q) {
//                 int l_coord = h_q + lb_coord * trait.NUM_HEAD_Q;
//                 gemm_kernel(trait, shape_mnkl, thr_mma,
//                             A, dA, tiledcopy_A,
//                             B, dB, tiledcopy_B,
//                             tCrC,
//                             m_coord, n_coord, l_coord,
//                             debug);
//             }
//             auto tCrC_bf16 = op(tCrC);
//             copy(copy_c, tCrC_bf16, tCgC);
//         }
//     }
// }

// template<typename T, class ProblemShape, class ThrMma,
//          class AStride, class TiledCopyA,
//          class BStride, class TiledCopyB,
//          class FragTensor>
// void gemm_dp(T &trait, ProblemShape &shape_mnkl, ThrMma &thr_mma,
//              typename T::DType const *A, AStride dA, TiledCopyA tiledcopy_A,
//              typename T::DType const *B, BStride dB, TiledCopyB tiledcopy_B,
//              FragTensor &tCrC,
//              const int m_coord, const int n_coord, const int lq_coord, const int lk_coord,
//              bool debug) {
//     gemm_kernel_bb(trait, shape_mnkl, thr_mma,
//                    A, dA, tiledcopy_A,
//                    B, dB, tiledcopy_B,
//                    tCrC,
//                    m_coord, n_coord, lq_coord, lk_coord, true);
// }

// template<typename T, class ProblemShape, class ThrMma,
//          class CStride, class TiledCopyC,
//          class FragTensor,
//          class AuxTensor, class SumTensor>
// void softmax_bwd(T &trait, ProblemShape &shape_mnkl, ThrMma &thr_mma,
//                  typename T::DType *C, CStride dC, TiledCopyC tiledcopy_C,
//                  FragTensor &tCrC,
//                  AuxTensor &tCrCy, SumTensor &sum_row,
//                  const int m_coord, const int n_coord, const int l_coord, const float softmax_coef) {
//     OPS_tobf16<typename T::DType> op;
//     auto C_shape = select<0,1,3>(shape_mnkl);
//     auto mC = make_tensor(make_gmem_ptr(C), make_layout(C_shape, dC));
//     auto copy_c = TiledCopyC{mC};
//     Tensor mC_coord = cute::get_xe_tensor(C_shape);   //(m,n,l)
//     Tensor gC = local_tile(mC_coord, select<0, 1>(trait.tile_mnk),
//                            make_coord(m_coord, n_coord, l_coord)); // dy
//     Tensor tCgC = thr_mma.partition_C(gC);
//     // y*dy

//     softmax_bwd_last(trait, tCrC, tCrCy, sum_row, softmax_coef);
//     auto tCrC_bf16 = op(tCrC);
//     copy(copy_c, tCrC_bf16, tCgC);
// }

// template<int NUM_SG, class Tensor, class STensor>
// void reduce_row(Tensor &t, STensor &sram) {
//     auto group = cutlasscompat::get_nd_item<1>().get_group();
//     auto sg = cutlasscompat::get_nd_item<1>().get_sub_group();
//     const auto sg_local_id = sg.get_local_id();
//     const auto sg_group_id = sg.get_group_id();
//     const auto sg_group_id_N = sg_group_id % NUM_SG;
//     const auto sg_group_id_M = sg_group_id / NUM_SG;
//     auto stensor = sram(_, _, sg_group_id_M);
//     CUTLASS_PRAGMA_UNROLL
//     for (int i = 0; i< size(t); ++i) {
//         t(i) = reduce_over_group(sg, t(i), sycl::plus<>());
//     }

//     if (sg_local_id == 0) {
//         CUTLASS_PRAGMA_UNROLL
//         for (int i = 0; i < size(t); ++i) {
//             stensor(i, sg_group_id_N) = t(i);
//         }
//     }
//     // have to wait here
//     sycl::group_barrier(group);
//     if (sg_local_id == 0) {
//         for (int i = 0; i < size(t); ++i) {
//             t(i) = 0.0f;
//             CUTLASS_PRAGMA_UNROLL
//             for (int j = 0; j < NUM_SG; ++j) {
//                 t(i) += stensor(i, j);
//             }
//         }
//     }
//     CUTLASS_PRAGMA_UNROLL
//     for (int i = 0; i < size(t); ++i) {
//         t(i) = sycl::group_broadcast(sg, t(i), 0);
//     }
// }

// template<typename T, typename V>
// void copy_tensor(T &src, V &dst) {
//     // static_assert(size(src) == size(dst));
//     CUTLASS_PRAGMA_UNROLL
//     for (int i = 0; i < size(src); ++i) {
//         dst(i) = src(i);
//     }
// }

// template <class, int, bool, bool> class MhaBackwardName;

// template<class T, int Nblk, bool dropout, bool first>
// void
// mha_backward(T trait,
//              typename T::DType const *go_d,
//              typename T::DType const *q_d,
//              typename T::DType const *k_d,
//              typename T::DType const *v_d,
//              typename T::DType const *ps_d,
//              typename T::DType const *psd_d,
//              typename T::DType *gq_d,
//              typename T::DType *gk_d,
//              typename T::DType *gv_d,
//              typename T::DType *gps_d) {
//     float softmax_coef = 1.0f / sqrtf(static_cast<float>(trait.HEAD_SIZE_QK));
//     auto sg = cutlasscompat::get_nd_item<1>().get_sub_group();
//     auto first_thread_in_sg_idx = sg.get_group_linear_id() * trait.SubgroupSize;
//     auto thr_mma = trait.tiled_mma.get_slice(first_thread_in_sg_idx);
//     // std::is_same_v<decltype(copy_Pst), int>;
//     int m_coord = BlockIdxX();
//     int lh_coord = BlockIdxY();
//     int lb_coord = BlockIdxZ();
//     using ProblemShape = cute::tuple<int, int, int, int>;
//     using ProblemShapeEx = cute::tuple<int, int, int, int, int>;
//     OPS_tobf16<typename T::DType> op;

//     if constexpr(first) {
//     {
//         /*
//           dV=Pt * dO
//           dV BATCH,NUM_HEAD_KV,SEQ_LEN_KV,HEAD_SIZE_VO
//           Pt BATCH,NUM_HEAD_Q,SEQ_LEN_KV,SEQ_LEN_QO
//           dO BATCH,NUM_HEAD_Q,SEQ_LEN_QO,HEAD_SIZE_VO
//           M SEQ_LEN_KV
//           N HEAD_SIZE_VO
//           K SEQ_LEN_QO
//         */
//         auto dPt = make_stride(Int<1>{}, trait.SEQ_LEN_KV,
//                                trait.SEQ_LEN_KV *trait.SEQ_LEN_QO); // A SEQ_LEN_KV,SEQ_LEN_QO
//         auto dO = make_stride(Int<1>{}, trait.HEAD_SIZE_VO,
//                               trait.HEAD_SIZE_VO *trait.SEQ_LEN_QO); // B HEAD_SIZE_VO,SEQ_LEN_QO
//         auto dV = make_stride(trait.HEAD_SIZE_VO, Int<1>{},
//                               trait.SEQ_LEN_KV *trait.HEAD_SIZE_VO); // C SEQ_LEN_KV,HEAD_SIZE_VO
//         ProblemShape problem_shape = ProblemShape(
//             trait.SEQ_LEN_KV,
//             trait.HEAD_SIZE_VO,
//             trait.SEQ_LEN_QO,
//             trait.BATCH * trait.NUM_HEAD_Q);
//         if constexpr(dropout) {
//             gemm_dkv(trait, problem_shape, thr_mma,
//                      psd_d, dPt, typename T::CopyPt{},
//                      go_d, dO, typename T::CopygOB{},
//                      gv_d, dV, typename T::CopygV{},
//                      m_coord, lh_coord, lb_coord, false);
//         } else {
//             gemm_dkv(trait, problem_shape, thr_mma,
//                      ps_d, dPt, typename T::CopyPt{},
//                      go_d, dO, typename T::CopygOB{},
//                      gv_d, dV, typename T::CopygV{},
//                      m_coord, lh_coord, lb_coord, false);
//         }
//     }
//     {
//         /*
//           shape
//           dO BATCH,NUM_HEAD_Q,SEQ_LEN_QO,HEAD_SIZE_VO
//           V  BATCH,NUM_HEAD_Q,SEQ_LEN_KV,HEAD_SIZE_VO
//           dP BATCH,NUM_HEAD_Q,SEQ_LEN_QO,SEQ_LEN_KV
//           M SEQ_LEN_KV
//           N HEAD_SIZE_VO
//           K SEQ_LEN_QO
//           dP=dO*Vt
//          */
//         auto dO = make_stride(trait.HEAD_SIZE_VO, Int<1>{},
//                               trait.SEQ_LEN_QO *trait.HEAD_SIZE_VO); // A SEQ_LEN_QO, HEAD_SIZE_VO
//         auto dV = make_stride(trait.HEAD_SIZE_VO, Int<1>{},
//                               trait.HEAD_SIZE_VO * trait.SEQ_LEN_KV); // B SEQ_LEN_KV, HEAD_SIZE_VO
//         auto dP = make_stride(trait.SEQ_LEN_KV, Int<1>{},
//                               trait.SEQ_LEN_QO *trait.SEQ_LEN_KV); // C SEQ_LEN_QO,SEQ_LEN_KV

//         auto tCrC = partition_fragment_C(trait.tiled_mma, take<0, 2>(trait.tile_mnk)); // dy

//         constexpr auto dimN = Int<Nblk>{};
//         constexpr auto dim0 = size<0>(tCrC);
//         constexpr auto dim1 = size<1>(tCrC);
//         constexpr auto dim2 = size<2>(tCrC);
//         auto dProw = make_tensor<float>(make_shape(dim0, dim1, dim2, dimN));
//         auto tCrCy = make_tensor<typename T::DType>(make_shape(dim0, dim1, dim2, dimN));
// // y
//         auto tCrCyd = make_tensor_like<typename T::DType>(tCrCy);
//         // init sum_row
//         auto sum_row = make_tensor_like(tCrC(_, _, 0));
//         constexpr auto NUM_COL_PER_THD = size<2>(tCrC);
//         constexpr auto NUM_ROW_PER_THD = size<0>(tCrC) * size<1>(tCrC);
//         constexpr auto NUM_SG_PER_ROW = trait.bN / (Int<trait.SubgroupSize>{} * NUM_COL_PER_THD);
//         constexpr auto NUM_SG_PER_BLK_M = trait.bM / NUM_ROW_PER_THD;
//         constexpr auto NUM_SG = NUM_SG_PER_ROW * NUM_SG_PER_BLK_M;

//         // leverage share memory to reduce over 1 block
//         auto smem = cutlasscompat::local_mem<float[NUM_SG * NUM_ROW_PER_THD]>();
//         Tensor stensor = make_tensor(make_smem_ptr(smem), make_shape(Int<NUM_ROW_PER_THD>{}, Int<NUM_SG_PER_ROW>{}, Int<NUM_SG_PER_BLK_M>{})); // bank conflict
//         ProblemShapeEx problem_shape = ProblemShapeEx(
//             trait.SEQ_LEN_QO,
//             trait.SEQ_LEN_KV,
//             trait.HEAD_SIZE_VO,
//             trait.BATCH * trait.NUM_HEAD_Q,
//             trait.BATCH * trait.NUM_HEAD_KV);
//         if (m_coord < ceil_div(trait.SEQ_LEN_QO, trait.bM)) {
//             int lk_coord = lh_coord + lb_coord * trait.max_block_n;
//             for (int h_q = lh_coord * trait.block_n_chunk; h_q < (lh_coord + 1)* trait.block_n_chunk; ++h_q) {
//                 int lq_coord = h_q + lb_coord * trait.NUM_HEAD_Q;
//                 clear(tCrCy);
//                 clear(sum_row);
//                 for (int n_coord = 0; n_coord<ceil_div(size<1>(problem_shape), trait.bN); ++n_coord) {
//                     clear(tCrC);
//                     gemm_dp(trait, problem_shape, thr_mma,
//                             go_d, dO, typename T::CopygOA{},
//                             v_d, dV, typename T::CopyVt{},
//                             tCrC,
//                             m_coord, n_coord, lq_coord, lk_coord, false);
//                     auto dPn = dProw(_, _, _, n_coord);
//                     // copy dP
//                     copy_tensor(tCrC, dPn);
//                     // softmax partial reduce
//                     auto Pn = tCrCy(_, _, _, n_coord);
//                     load_y(trait, problem_shape, thr_mma, ps_d, dP, typename T::CopyP{}, Pn, m_coord, n_coord, lq_coord); // load attn without dropout
//                     if constexpr(dropout) {
//                         auto Pdn = tCrCyd(_, _, _, n_coord);
//                         load_y(trait, problem_shape, thr_mma, psd_d, dP, typename T::CopyP{}, Pdn, m_coord, n_coord, lq_coord); // load attn with dropout
//                         softmax_bwd_partial_sum(dPn, Pdn, sum_row);
//                     } else {
//                         softmax_bwd_partial_sum(dPn, Pn, sum_row);
//                     }
//                 }
//                 reduce_row<NUM_SG_PER_ROW, decltype(sum_row), decltype(stensor)>(sum_row, stensor);
//                 for (int n_coord = 0; n_coord<ceil_div(size<1>(problem_shape), trait.bN); ++n_coord) {
//                     auto dPn = dProw(_, _, _, n_coord);
//                     auto Pn = tCrCy(_, _, _, n_coord);
//                     softmax_bwd(trait, problem_shape, thr_mma,
//                                 gps_d, dP, typename T::CopygP{},
//                                 dPn, Pn, sum_row,
//                                 m_coord, n_coord, lq_coord, softmax_coef);
//                 }
//             }
//         }
//     }
//     } else {
//     // store in smem for transpose syk
//     // auto group = cutlasscompat::get_nd_item<1>().get_group();
//     // sycl::group_barrier(group);
//     {
//         /*
//           dS BATCH,NUM_HEAD_Q,SEQ_LEN_QO,HEAD_SIZE_VO
//           K  BATCH,NUM_HEAD_KV,SEQ_LEN_KV,HEAD_SIZE_QK
//           dQ BATCH,NUM_HEAD_Q,SEQ_LEN_QO,HEAD_SIZE_QK
//           M  SEQ_LEN_QO
//           N  HEAD_SIZE_QK
//           K  SEQ_LEN_KV
//           dQ=dS*K
//          */
//         auto dS = make_stride(trait.SEQ_LEN_KV, Int<1>{},
//                               trait.SEQ_LEN_KV *trait.SEQ_LEN_QO); // A SEQ_LEN_QO, SEQ_LEN_KV
//         auto dK = make_stride(Int<1>{}, trait.HEAD_SIZE_QK,
//                               trait.HEAD_SIZE_QK *trait.SEQ_LEN_KV); // B HEAD_SIZE_QK, SEQ_LEN_KV
//         auto dQ = make_stride(trait.HEAD_SIZE_QK, Int<1>{},
//                               trait.SEQ_LEN_QO * trait.HEAD_SIZE_QK); // C SEQ_LEN_QO, HEAD_SIZE_QK
//         ProblemShapeEx problem_shape = ProblemShapeEx(
//             trait.SEQ_LEN_QO,
//             trait.HEAD_SIZE_QK,
//             trait.SEQ_LEN_KV,
//             trait.BATCH * trait.NUM_HEAD_Q,
//             trait.BATCH * trait.NUM_HEAD_KV);
//         int lk_coord = lh_coord + lb_coord * trait.max_block_n;
//         for (int h_q = lh_coord * trait.block_n_chunk; h_q < (lh_coord + 1) * trait.block_n_chunk; ++h_q) {
//             int lq_coord = h_q + lb_coord * trait.NUM_HEAD_Q;
//             gemm_dq(trait, problem_shape, thr_mma,
//                     gps_d, dS, typename T::CopygS{},
//                     k_d, dK, typename T::CopyK{},
//                     gq_d, dQ, typename T::CopygQ{},
//                     m_coord, lq_coord, lk_coord, false);
//         }
//     }
//     {
//         /*
//           dS BATCH,NUM_HEAD_Q,SEQ_LEN_QO,SEQ_LEN_KV
//           Q  BATCH,NUM_HEAD_Q,SEQ_LEN_QO,HEAD_SIZE_QK
//           dK BATCH,NUM_HEAD_KV,SEQ_LEN_KV,HEAD_SIZE_QK
//           M SEQ_LEN_KV
//           N HEAD_SIZE_QK
//           K SEQ_LEN_QO
//           dK=dSt*Q
//          */
//         auto dS = make_stride(Int<1>{}, trait.SEQ_LEN_KV,
//                               trait.SEQ_LEN_KV *trait.SEQ_LEN_QO); // A SEQ_LEN_KV,SEQ_LEN_QO
//         auto dQ = make_stride(Int<1>{}, trait.HEAD_SIZE_QK,
//                               trait.SEQ_LEN_QO *trait.HEAD_SIZE_QK); // B SEQ_LEN_QO,HEAD_SIZE_QK
//         auto dK = make_stride(trait.HEAD_SIZE_QK, Int<1>{},
//                               trait.HEAD_SIZE_QK *trait.SEQ_LEN_KV); // C SEQ_LEN_KV,HEAD_SIZE_QK

//         ProblemShape problem_shape = ProblemShape(
//             trait.SEQ_LEN_KV,
//             trait.HEAD_SIZE_QK,
//             trait.SEQ_LEN_QO,
//             trait.BATCH * trait.NUM_HEAD_Q);
//         gemm_dkv(trait, problem_shape, thr_mma,
//                  gps_d, dS, typename T::CopygSt{},
//                  q_d, dQ, typename T::CopyQ{},
//                  gk_d, dK, typename T::CopygK{},
//                  m_coord, lh_coord, lb_coord, false);
//     }
//     }
// }

// template<typename T, class ProblemShape, int nBlk, bool dropout>
// void launch_mha_backward(ProblemShape problem_shape,
//                          T *go_d,
//                          const T *q_d,
//                          const T *k_d,
//                          const T *v_d,
//                          const T *ps_d,
//                          const T *psd_d,
//                          T *dq_d,
//                          T *dk_d,
//                          T *dv_d,
//                          T *dps_d) {

//     auto trait = MHA_TYPE<T, ProblemShape>(problem_shape);

//     auto dimGrid = cutlasscompat::dim3(size(ceil_div(trait.max_block_m, trait.bM)), size(trait.max_block_n), size(trait.BATCH));
//     assert((trait.NUM_HEAD_Q % trait.NUM_HEAD_KV == 0) && "num_head_q must be dividable by num_head_kv");
//     assert((trait.NUM_HEAD_Q >= trait.NUM_HEAD_KV) && "num_head_q must be bigger than or equal to num_head_kv");
//     assert((trait.bNi <= trait.SEQ_LEN_KV) && "tile_N must be larger than SEQ_LEN_KV");
//     auto dimBlock = cutlasscompat::dim3(size(trait.mmaC), size(1), size(1));

//     std::cout << "Launch mha bwd kernel with: " <<
//         "batch_size: " << trait.BATCH <<
//         ", num_head_q: " << trait.NUM_HEAD_Q <<
//         ", num_head_kv: " << trait.NUM_HEAD_KV <<
//         ", head_dim_qk: " << trait.HEAD_SIZE_QK <<
//         ", head_dim_v: " << trait.HEAD_SIZE_VO <<
//         ", seq_len_q: " << trait.SEQ_LEN_QO <<
//         ", seq_len_kv: " << trait.SEQ_LEN_KV << std::endl;

//     cutlasscompat::experimental::launch_properties launch_props{
//         // sycl::ext::oneapi::experimental::work_group_scratch_size(0),
//     };
//     cutlasscompat::experimental::kernel_properties kernel_props{
//         sycl::ext::oneapi::experimental::sub_group_size<trait.SubgroupSize>};
//     cutlasscompat::experimental::launch_policy policy{dimGrid, dimBlock, launch_props, kernel_props};
//     auto event1 = cutlasscompat::experimental::launch<
//         mha_backward<decltype(trait), nBlk, dropout, true>,
//         MhaBackwardName<decltype(trait), nBlk, dropout, true>>(policy,
//                                                             trait,
//                                                             go_d,
//                                                             q_d, k_d, v_d,
//                                                             ps_d, psd_d,
//                                                             dq_d, dk_d, dv_d,
//                                                             dps_d);
//     EventManager::getInstance().addEvent(event1);
//     auto event2 = cutlasscompat::experimental::launch<
//         mha_backward<decltype(trait), nBlk, dropout, false>,
//         MhaBackwardName<decltype(trait), nBlk, dropout, false>>(policy,
//                                                              trait,
//                                                              go_d,
//                                                              q_d, k_d, v_d,
//                                                              ps_d, psd_d,
//                                                              dq_d, dk_d, dv_d,
//                                                              dps_d);
//     EventManager::getInstance().addEvent(event2);
// }

// template<typename T, class ProblemShape>
// void launch_mha_bwd_wrapper(ProblemShape problem_shape,
//                             T *go_d,
//                             const T *q_d,
//                             const T *k_d,
//                             const T *v_d,
//                             const T *ps_d,
//                             const T *psd_d,
//                             T *dq_d,
//                             T *dk_d,
//                             T *dv_d,
//                             T *dps_d) {
//     int SEQ_LEN_KV = get<4>(problem_shape);
//     if (psd_d == nullptr) {
//         if (SEQ_LEN_KV <= 1024) {
//             constexpr int nBlk = 4;
//             launch_mha_backward<T, ProblemShape, nBlk, false>(problem_shape,
//                                              go_d,
//                                              q_d, k_d, v_d,
//                                              ps_d, psd_d,
//                                              dq_d, dk_d, dv_d,
//                                              dps_d);
//         } else if (SEQ_LEN_KV <= 512) {
//             constexpr int nBlk = 2;
//             launch_mha_backward<T, ProblemShape, nBlk, false>(problem_shape,
//                                              go_d,
//                                              q_d, k_d, v_d,
//                                              ps_d, psd_d,
//                                              dq_d, dk_d, dv_d,
//                                              dps_d);
//         } else if (SEQ_LEN_KV <= 256) {
//             constexpr int nBlk = 1;
//             launch_mha_backward<T, ProblemShape, nBlk, false>(problem_shape,
//                                              go_d,
//                                              q_d, k_d, v_d,
//                                              ps_d, psd_d,
//                                              dq_d, dk_d, dv_d,
//                                              dps_d);
//         }
//     } else {
//         if (SEQ_LEN_KV <= 1024) {
//             constexpr int nBlk = 4;
//             launch_mha_backward<T, ProblemShape, nBlk, true>(problem_shape,
//                                              go_d,
//                                              q_d, k_d, v_d,
//                                              ps_d, psd_d,
//                                              dq_d, dk_d, dv_d,
//                                              dps_d);
//         } else if (SEQ_LEN_KV <= 512) {
//             constexpr int nBlk = 2;
//             launch_mha_backward<T, ProblemShape, nBlk, true>(problem_shape,
//                                              go_d,
//                                              q_d, k_d, v_d,
//                                              ps_d, psd_d,
//                                              dq_d, dk_d, dv_d,
//                                              dps_d);
//         } else if (SEQ_LEN_KV <= 256) {
//             constexpr int nBlk = 1;
//             launch_mha_backward<T, ProblemShape, nBlk, true>(problem_shape,
//                                              go_d,
//                                              q_d, k_d, v_d,
//                                              ps_d, psd_d,
//                                              dq_d, dk_d, dv_d,
//                                              dps_d);
//         }
//     }
// }

void cutlass_sdpa_backward(
    int batch_size,
    int num_head_q,
    int num_head_kv,
    int seq_len_q,
    int seq_len_kv,
    int head_dim_qk,
    int head_dim_v,
    const void* grad_out,
    const void* query,
    const void* key,
    const void* value,
    const void* ps,
    const void* psd,
    void* grad_query,
    void* grad_key,
    void* grad_value,
    void* dps) {
    temp_entry();
    // using T = cute::bfloat16_t;
    // ProblemShapeRegular problem_shape{batch_size, num_head_q, num_head_kv, seq_len_q, seq_len_kv, head_dim_qk, head_dim_v};
    // launch_mha_bwd_wrapper<T, ProblemShapeRegular>(problem_shape,
    //                                               (T*)grad_out,
    //                                               (const T*)query,
    //                                               (const T*)key,
    //                                               (const T*)value,
    //                                               (const T*) ps,
    //                                               (const T*)psd,
    //                                                 (T*)grad_query,
    //                                                 (T*)grad_key,
    //                                                 (T*)grad_value,
    //                                                 (T*)dps);
}

// template void launch_mha_bwd_wrapper<cute::bfloat16_t, ProblemShapeRegular>(ProblemShapeRegular problem_shape,
//                             cute::bfloat16_t *go_d,
//                             const cute::bfloat16_t *q_d,
//                             const cute::bfloat16_t *k_d,
//                             const cute::bfloat16_t *v_d,
//                             const cute::bfloat16_t *ps_d,
//                             const cute::bfloat16_t *psd_d,
//                             cute::bfloat16_t *dq_d,
//                             cute::bfloat16_t *dk_d,
//                             cute::bfloat16_t *dv_d,
//                             cute::bfloat16_t *dps_d);