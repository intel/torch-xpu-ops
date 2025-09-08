#include <cstdio>
#include <cstddef>
#include <cassert>
#include <cstdlib>
#include <math.h>
#include <string>

#include <sycl/sycl.hpp>
#include <syclcompat.hpp>

#include <cute/tensor.hpp>

#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/print_error.hpp"
#include "cutlass/util/sycl_event_manager.hpp"
#include "cutlass/util/GPU_Clock.hpp"

#include "AttentionKernels.h"

using namespace cute;

using ProblemShapeRegular = cute::tuple<int, int, int, int, int, int, int>; // batch, num_head_q,num_head_kv,seq_len_qo,seq_len_kv,head_size_qk,head_size_vo
template <typename T>
struct OPS_tobf16{
    template <class Tensor>
    auto operator()(Tensor &src){
        cutlass::NumericConverter<
            T, float, cutlass::FloatRoundStyle::round_toward_zero> converter;
        auto dst = make_tensor_like<T>(src);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(src); ++i) {
            dst(i) = converter(src(i));
        }
        return dst;
    }
};

constexpr int tid = 0;
constexpr int bid = 16;
using MASKType = uint16_t;
template <class T_, class ProblemShape_>
struct MHA_TYPE {
    using ProblemShape = ProblemShape_;
    /*
      Q BATCH,NUM_HEAD_Q,SEQ_LEN_QO,HEAD_SIZE_QK
      K BATCH,NUM_HEAD_KV,SEQ_LEN_KV,HEAD_SIZE_QK
      V BATCH,NUM_HEAD_KV,SEQ_LEN_KV,HEAD_SIZE_VO
      P BATCH,NUM_HEAD_Q,SEQ_LEN_QO,SEQ_LEN_KV
      O BATCH,NUM_HEAD_Q,SEQ_LEN_QO,HEAD_SIZE_VO
    */
    /*
      dPs BATCH,NUM_HEAD_Q,SEQ_LEN_QO,SEQ_LEN_KV
      dP BATCH,NUM_HEAD_Q,SEQ_LEN_QO,SEQ_LEN_KV
      dP=softmax_backward(softmax, dPs)
    */
    using DType = T_;
    using Stride1 = cute::tuple<long, cute::C<1>, long>;
    using Stride0 = cute::tuple<cute::C<1>, long, long>;

    static constexpr int bMi = 256;
    static constexpr int bNi = 256;
    static constexpr int bKi = 32;
    static constexpr auto bM = Int<bMi>{};
    static constexpr auto bN = Int<bNi>{};
    static constexpr auto bK = Int<bKi>{};
    static constexpr auto tile_mnk = make_shape(bM, bN, bK);
    static constexpr auto bP = Int<2>{}; // Pipeline

    /*
      shape
      Pt BATCH,NUM_HEAD_Q,SEQ_LEN_KV,SEQ_LEN_QO
      dO BATCH,NUM_HEAD_Q,SEQ_LEN_QO,HEAD_SIZE_VO
      dV BATCH,NUM_HEAD_KV,SEQ_LEN_KV,HEAD_SIZE_VO
      M SEQ_LEN_KV
      N HEAD_SIZE_VO
      K SEQ_LEN_QO
      dV=Pt*dO
    */
    using CopyPt = decltype(
        make_tiled_copy(Copy_Atom<Copy_Traits<XE_2D_U16x16x16_LD_T, Stride0>, DType>{},
                        Layout<Shape<_1,_16>>{}, // Thr layout 1x16 k-major
                        Layout<Shape<_16,_1>>{})); // Val layout  16x1
    using CopygOB = decltype(
        make_tiled_copy(Copy_Atom<Copy_Traits<XE_2D_U16x32x32_LD_N, Stride0>, DType>{},
                        Layout<Shape<_1,_16>>{}, // Thr layout 1x16 n-major
                        Layout<Shape<_32,_2>>{})); // Val layout  32x2);
    using CopygV = decltype(
        make_tiled_copy(Copy_Atom<Copy_Traits<XE_2D_U16x8x16_ST_N, Stride1>, DType>{},
                        Layout<Shape<_1,_16>>{}, // Thr layout 1x16 n-major
                        Layout<Shape<_8,_1>>{})); // Val layout  8x1
    /*
      dO BATCH,NUM_HEAD_Q,SEQ_LEN_QO,HEAD_SIZE_VO
      dV BATCH,NUM_HEAD_KV,HEAD_SIZE_VO,SEQ_LEN_KV
      dP BATCH,NUM_HEAD_Q,SEQ_LEN_QO,SEQ_LEN_KV
      M SEQ_LEN_QO
      N SEQ_LEN_KV
      K HEAD_SIZE_VO
      dP=dO*Vt
    */

    using CopygOA = decltype(
        make_tiled_copy(Copy_Atom<Copy_Traits<XE_2D_U16x32x32_LD_N, Stride1>, DType>{},
                        Layout<Shape<_1,_16>>{}, // Thr layout 1x16 k-major
                        Layout<Shape<_32,_2>>{}));              // Val layout  32x2
    using CopyVt = decltype(
        make_tiled_copy(Copy_Atom<Copy_Traits<XE_2D_U16x16x16_LD_T, Stride1>, DType>{},
                        Layout<Shape<_1,_16>>{}, // Thr layout 1x16 n-major
                        Layout<Shape<_16,_1>>{})); //Val layout 16x1
    using CopygP = decltype(
        make_tiled_copy(Copy_Atom<Copy_Traits<XE_2D_U16x8x16_ST_N, Stride1>, DType>{},
                        Layout<Shape<_1,_16>>{}, // Thr layout 1x16 n-major
                        Layout<Shape<_8,_1>>{}));              // Val layout  8x1
    using CopyP = decltype(
        make_tiled_copy(Copy_Atom<Copy_Traits<XE_2D_U16x8x16_LD_N, Stride1>, DType>{},
                        Layout<Shape<_1, _16>>{},
                        Layout<Shape<_8, _1>> {}));
    /*
     * dS BATCH,NUM_HEAD_Q,SEQ_LEN_QO,SEQ_LEN_KV
     * Q BATCH,NUM_HEAD_Q,SEQ_LEN_QO,HEAD_SIZE_QK
     * dK BATCH,NUM_HEAD_KV,SEQ_LEN_KV,HEAD_SIZE_QK
     * M SEQ_LEN_KV
     * N HEAD_SIZE_QK
     * K SEQ_LEN_QO
     * dK=dSt*Q
     */
    using CopygSt = decltype(
        make_tiled_copy(Copy_Atom<Copy_Traits<XE_2D_U16x16x16_LD_T, Stride0>, DType>{},
                        Layout<Shape<_1,_16>>{}, // Thr layout 1x16 k-major
                        Layout<Shape<_16,_1>>{})); // Val layout  16x1
    using CopyQ = decltype(
        make_tiled_copy(Copy_Atom<Copy_Traits<XE_2D_U16x32x32_LD_N, Stride0>, DType>{},
                        Layout<Shape<_1,_16>>{}, // Thr layout 1x16 n-major
                        Layout<Shape<_32,_2>>{}));              // Val layout  16x1
    using CopygK = decltype(
        make_tiled_copy(Copy_Atom<Copy_Traits<XE_2D_U16x8x16_ST_N, Stride1>, DType>{},
                        Layout<Shape<_1,_16>>{}, // Thr layout 1x16 n-major
                        Layout<Shape<_8,_1>>{}));              // Val layout  8x1
    /*
     * dS BATCH,NUM_HEAD_Q,SEQ_LEN_QO,SEQ_LEN_KV
     * K BATCH,NUM_HEAD_KV,SEQ_LEN_KV,HEAD_SIZE_QK
     * dQ BATCH,NUM_HEAD_Q,SEQ_LEN_QO,HEAD_SIZE_QK
     * M SEQ_LEN_QO
     * N HEAD_SIZE_QK
     * K SEQ_LEN_KV
     * dQ=dS*K
     */
    using CopygS = decltype(
        make_tiled_copy(Copy_Atom<Copy_Traits<XE_2D_U16x32x32_LD_N, Stride1>, DType>{},
                        Layout<Shape<_1,_16>>{}, // Thr layout 1x16 k-major
                        Layout<Shape<_32,_2>>{}));              // Val layout  32x2
    using CopyK = decltype(
        make_tiled_copy(Copy_Atom<Copy_Traits<XE_2D_U16x32x32_LD_N, Stride0>, DType>{},
                        Layout<Shape<_1,_16>>{}, // Thr layout 1x16 n-major
                        Layout<Shape<_32,_2>>{}));              // Val layout  32x2
    using CopygQ = decltype(
        make_tiled_copy(Copy_Atom<Copy_Traits<XE_2D_U16x8x16_ST_N, Stride1>, DType>{},
                        Layout<Shape<_1,_16>>{}, // Thr layout 1x16 n-major
                        Layout<Shape<_8,_1>>{}));              // Val layout  8x1
    using MMA_Atom_Arch = std::conditional_t<
        std::is_same_v<T_, cutlass::half_t>,
        MMA_Atom<XE_8x16x16_F32F16F16F32_TT>,
        MMA_Atom<XE_8x16x16_F32BF16BF16F32_TT>>;
    static constexpr TiledMMA mmaC = typename TiledMMAHelper<MMA_Atom_Arch, Layout<decltype(tile_mnk)>,
                                                             Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA{};  // 256x128x16 TiledMMA
    using TiledMma = decltype(mmaC);
    static constexpr int SubgroupSize = 16;
    static constexpr int smem_size = 0;

    MHA_TYPE(ProblemShape_ mha_shape_)
        :mha_shape(mha_shape_),
         BATCH(get<0>(mha_shape)),
         NUM_HEAD_Q(get<1>(mha_shape)),
         NUM_HEAD_KV(get<2>(mha_shape)),
         SEQ_LEN_QO(get<3>(mha_shape)),
         SEQ_LEN_KV(get<4>(mha_shape)),
         HEAD_SIZE_QK(get<5>(mha_shape)),
         HEAD_SIZE_VO(get<6>(mha_shape)),
         max_block_m(std::max(SEQ_LEN_KV, SEQ_LEN_QO)),
         max_block_n(NUM_HEAD_KV),
         block_n_chunk(NUM_HEAD_Q / NUM_HEAD_KV)
        {}

    // variables
    ProblemShape mha_shape;
    TiledMma tiled_mma;
    const int64_t BATCH;
    const int64_t NUM_HEAD_Q;
    const int64_t NUM_HEAD_KV;
    const int64_t SEQ_LEN_QO;
    const int64_t SEQ_LEN_KV;
    const int64_t HEAD_SIZE_QK;
    const int64_t HEAD_SIZE_VO;
    const int64_t max_block_m;
    const int64_t max_block_n;
    const int64_t block_n_chunk;
};

template<class T, class ProblemShape, class ThrMma,
         class YStride, class TiledCopyY, class RTensor>
void load_y(T &trait, ProblemShape &shape_mnkl, ThrMma &thr_mma,
            typename T::DType const *Y, YStride dY, TiledCopyY,
            RTensor &t,
            const int m_coord, const int n_coord, const int lq_coord) {
    auto Y_shape = select<0,1,3>(shape_mnkl);
    auto mY = make_tensor(make_gmem_ptr(Y), make_layout(Y_shape, dY));
    auto copy_caux = TiledCopyY{mY};
    Tensor mY_coord = cute::get_xe_tensor(Y_shape);   //(m,n,l)
    Tensor gY = local_tile(mY_coord, select<0, 1>(trait.tile_mnk),
                           make_coord(m_coord, n_coord, lq_coord)); // aux
    // copy y
    Tensor tCgy = thr_mma.partition_C(gY);
    copy(copy_caux, tCgy, t);
}

template<class FragTensor,
         class AuxTensor,
         class SumTensor>
void softmax_bwd_partial_sum(FragTensor &tCrC,
                             AuxTensor &tCrCy,
                             SumTensor &sum_row) {
    // calculate sum of dy*y
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size<2>(tCrC); ++i) {
        auto dy_col = tCrC(_, _, i);
        auto y_col = tCrCy(_, _, i);
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < size(dy_col); ++j) {
            dy_col(j) = dy_col(j) * y_col(j);
            sum_row(j) = sum_row(j) + dy_col(j);
        }
    }
}

template<class T,
         class FragTensor, class AuxTensor,
         class STensor>
void softmax_bwd_last(T &trait,
                      FragTensor &tCrC,
                      AuxTensor &tCrCy,
                      STensor &sum_buf,
                      const float softmax_coef) {
    // copy y
    for (int i = 0; i < size<2>(tCrC); ++i) {
        auto y_col = tCrCy(_, _, i);
        auto ydy_col = tCrC(_, _, i);
        for (int j = 0; j < size(y_col); ++j) {
            auto sum_val = sum_buf(j);
            ydy_col(j) = (ydy_col(j) - y_col(j) * sum_val) * softmax_coef;
        }
    }
}

template<typename T, class ProblemShape, class ThrMma,
         class AStride, class TiledCopyA,
         class BStride, class TiledCopyB,
         class AccT>
void gemm_kernel(T &trait, ProblemShape &shape_mnkl,
                 ThrMma &thr_mma,
                 typename T::DType const *A, AStride dA, TiledCopyA,
                 typename T::DType const *B, BStride dB, TiledCopyB,
                 AccT &tCrC, const int m_coord, const int n_coord, const int l_coord,
                 bool debug) {

    auto A_shape = select<0,2,3>(shape_mnkl);
    auto B_shape = select<1,2,3>(shape_mnkl);

    // Represent the full tensors
    auto mA = make_tensor(make_gmem_ptr(A), make_layout(A_shape, dA));
    auto mB = make_tensor(make_gmem_ptr(B), make_layout(B_shape, dB));

    auto copy_a = TiledCopyA{mA};
    auto copy_b = TiledCopyB{mB};

    Tensor mA_coord = cute::get_xe_tensor(A_shape);   //(m,k,l)
    Tensor mB_coord = cute::get_xe_tensor(B_shape);   //(n,k,l)

    Tensor gA = local_tile(mA_coord, select<0, 2>(trait.tile_mnk), make_coord(m_coord, _, l_coord));  // (BLK_M,BLK_K,k)
    Tensor gB = local_tile(mB_coord, select<1, 2>(trait.tile_mnk), make_coord(n_coord, _, l_coord));  // (BLK_N,BLK_K,k)

    // Partition global counting tensors for MMA
    Tensor tCgA = thr_mma.partition_A(gA);
    Tensor tCgB = thr_mma.partition_B(gB);

    Tensor tCrA = make_tensor<typename T::DType>(make_fragment_layout(copy_a, tCgA(_,_,_,0).shape()));
    Tensor tCrB = make_tensor<typename T::DType>(make_fragment_layout(copy_b, tCgB(_,_,_,0).shape()));

    ThrCopy thr_copy_a = copy_a.get_slice(syclcompat::local_id::x());
    ThrCopy thr_copy_b = copy_b.get_slice(syclcompat::local_id::x());

    // Retile registers for copies
    Tensor tArA = thr_copy_a.retile_D(tCrA);
    Tensor tBrB = thr_copy_b.retile_D(tCrB);

    // Retile global counting tensors for copies
    Tensor tAgA = thr_copy_a.retile_S(tCgA);
    Tensor tBgB = thr_copy_b.retile_S(tCgB);

    //
    // PREFETCH
    //

    // constexpr int Num_SGs = size(tiled_mma);
    static constexpr auto ATOM_M = get<1>(typename T::TiledMma::ThrLayoutVMNK{}.shape());
    static constexpr auto ATOM_N = get<2>(typename T::TiledMma::ThrLayoutVMNK{}.shape());
    static constexpr auto ATOM_K = get<3>(typename T::TiledMma::ThrLayoutVMNK{}.shape());
    static constexpr auto Num_SGs = ATOM_N * ATOM_M * ATOM_K;

    static constexpr auto BLK_M = get<0>(T::tile_mnk);
    static constexpr auto BLK_N = get<1>(T::tile_mnk);
    static constexpr auto BLK_K = get<2>(T::tile_mnk);

    auto prefetch_a = cute::prefetch_selector<Shape<Int<BLK_M>,Int<BLK_K>>, Num_SGs>(copy_a);
    auto prefetch_b = cute::prefetch_selector<Shape<Int<BLK_N>,Int<BLK_K>>, Num_SGs>(copy_b);
    int thread_idx = int(ThreadIdxX());
    auto thr_prefetch_A = prefetch_a.get_slice(thread_idx);
    auto thr_prefetch_B = prefetch_b.get_slice(thread_idx);

    // Partition global tile for prefetch
    auto pAgA = thr_prefetch_A.partition_S(gA);
    auto pBgB = thr_prefetch_B.partition_S(gB);

    int prefetch_k = 0;

    constexpr int barrier_scope = 2;
    int k_tile_count = ceil_div(get<2>(shape_mnkl), get<2>(trait.tile_mnk));
    auto stages = trait.bP;
    CUTLASS_PRAGMA_UNROLL
    for (; prefetch_k < stages; prefetch_k++) {
        if (prefetch_k < k_tile_count) {
            prefetch(prefetch_a, pAgA(_, _, _, prefetch_k));
            prefetch(prefetch_b, pBgB(_, _, _, prefetch_k));
        }
    }

    CUTLASS_PRAGMA_UNROLL
    for (int k_tile = 0; k_tile < k_tile_count; k_tile++, prefetch_k++) {
        barrier_arrive(barrier_scope);
        // Copy gmem to rmem for the first k_tile
        copy(copy_a, tAgA(_,_,_,k_tile), tArA);
        copy(copy_b, tBgB(_,_,_,k_tile), tBrB);

        if (prefetch_k < k_tile_count) {
            prefetch(prefetch_a, pAgA(_, _, _, prefetch_k));
            prefetch(prefetch_b, pBgB(_, _, _, prefetch_k));
        }

        cute::gemm(trait.tiled_mma, tCrA, tCrB, tCrC);
        barrier_wait(barrier_scope);

    }
}

template<typename T, class ProblemShape, class ThrMma,
         class AStride, class TiledCopyA,
         class BStride, class TiledCopyB,
         class AccT>
void gemm_kernel_bb(T &trait, ProblemShape &shape_mnkl,
                    ThrMma &thr_mma,
                    typename T::DType const *A, AStride dA, TiledCopyA,
                    typename T::DType const *B, BStride dB, TiledCopyB,
                    AccT &tCrC, const int m_coord, const int n_coord, const int lq_coord, const int lk_coord,
                    bool debug) {

    auto A_shape = select<0,2,3>(shape_mnkl);
    auto B_shape = select<1,2,4>(shape_mnkl);

    // Represent the full tensors
    auto mA = make_tensor(make_gmem_ptr(A), make_layout(A_shape, dA));
    auto mB = make_tensor(make_gmem_ptr(B), make_layout(B_shape, dB));

    auto copy_a = TiledCopyA{mA};
    auto copy_b = TiledCopyB{mB};

    Tensor mA_coord = cute::get_xe_tensor(A_shape);   //(m,k,l)
    Tensor mB_coord = cute::get_xe_tensor(B_shape);   //(n,k,lh)


    Tensor gA = local_tile(mA_coord, select<0, 2>(trait.tile_mnk), make_coord(m_coord, _, lq_coord));  // (BLK_M,BLK_K,k)
    Tensor gB = local_tile(mB_coord, select<1, 2>(trait.tile_mnk), make_coord(n_coord, _, lk_coord));  // (BLK_N,BLK_K,k)

    // Partition global counting tensors for MMA
    Tensor tCgA = thr_mma.partition_A(gA);
    Tensor tCgB = thr_mma.partition_B(gB);

    Tensor tCrA = make_tensor<typename T::DType>(make_fragment_layout(copy_a, tCgA(_,_,_,0).shape()));
    Tensor tCrB = make_tensor<typename T::DType>(make_fragment_layout(copy_b, tCgB(_,_,_,0).shape()));

    ThrCopy thr_copy_a = copy_a.get_slice(syclcompat::local_id::x());
    ThrCopy thr_copy_b = copy_b.get_slice(syclcompat::local_id::x());

    // Retile registers for copies
    Tensor tArA = thr_copy_a.retile_D(tCrA);
    Tensor tBrB = thr_copy_b.retile_D(tCrB);

    // Retile global counting tensors for copies
    Tensor tAgA = thr_copy_a.retile_S(tCgA);
    Tensor tBgB = thr_copy_b.retile_S(tCgB);

    //
    // PREFETCH
    //

    // constexpr int Num_SGs = size(tiled_mma);
    static constexpr auto ATOM_M = get<1>(typename T::TiledMma::ThrLayoutVMNK{}.shape());
    static constexpr auto ATOM_N = get<2>(typename T::TiledMma::ThrLayoutVMNK{}.shape());
    static constexpr auto ATOM_K = get<3>(typename T::TiledMma::ThrLayoutVMNK{}.shape());
    static constexpr auto Num_SGs = ATOM_N * ATOM_M * ATOM_K;

    static constexpr auto BLK_M = get<0>(T::tile_mnk);
    static constexpr auto BLK_N = get<1>(T::tile_mnk);
    static constexpr auto BLK_K = get<2>(T::tile_mnk);

    auto prefetch_a = cute::prefetch_selector<Shape<Int<BLK_M>,Int<BLK_K>>, Num_SGs>(copy_a);
    auto prefetch_b = cute::prefetch_selector<Shape<Int<BLK_N>,Int<BLK_K>>, Num_SGs>(copy_b);
    int thread_idx = int(ThreadIdxX());
    auto thr_prefetch_A = prefetch_a.get_slice(thread_idx);
    auto thr_prefetch_B = prefetch_b.get_slice(thread_idx);

    // Partition global tile for prefetch
    auto pAgA = thr_prefetch_A.partition_S(gA);
    auto pBgB = thr_prefetch_B.partition_S(gB);

    int prefetch_k = 0;

    constexpr int barrier_scope = 2;
    int k_tile_count = ceil_div(get<2>(shape_mnkl), get<2>(trait.tile_mnk));
    auto stages = trait.bP;
    CUTLASS_PRAGMA_UNROLL
    for (; prefetch_k < stages; prefetch_k++) {
        if (prefetch_k < k_tile_count) {
            prefetch(prefetch_a, pAgA(_, _, _, prefetch_k));
            prefetch(prefetch_b, pBgB(_, _, _, prefetch_k));
        }
    }

    CUTLASS_PRAGMA_UNROLL
    for (int k_tile = 0; k_tile < k_tile_count; k_tile++, prefetch_k++) {
        barrier_arrive(barrier_scope);
        // Copy gmem to rmem for the first k_tile
        copy(copy_a, tAgA(_,_,_,k_tile), tArA);
        copy(copy_b, tBgB(_,_,_,k_tile), tBrB);

        if (prefetch_k < k_tile_count) {
            prefetch(prefetch_a, pAgA(_, _, _, prefetch_k));
            prefetch(prefetch_b, pBgB(_, _, _, prefetch_k));
        }

        cute::gemm(trait.tiled_mma, tCrA, tCrB, tCrC);
        barrier_wait(barrier_scope);

    }
}

template<typename T, class ProblemShape, class ThrMma,
         class AStride, class TiledCopyA,
         class BStride, class TiledCopyB,
         class CStride, class TiledCopyC>
void gemm(T &trait, ProblemShape &shape_mnkl, ThrMma &thr_mma,
          typename T::DType const *A, AStride dA, TiledCopyA tiledcopy_A,
          typename T::DType const *B, BStride dB, TiledCopyB tiledcopy_B,
          typename T::DType const *C, CStride dC, TiledCopyC tiledcopy_C,
          const int m_coord, const int l_coord,
          bool debug) {
    auto C_shape = select<0,1,3>(shape_mnkl);
    auto mC = make_tensor(make_gmem_ptr(C), make_layout(C_shape, dC));
    auto copy_c = TiledCopyC{mC};
    Tensor mC_coord = cute::get_xe_tensor(C_shape);   //(m,n,l)
    OPS_tobf16<typename T::DType> op;
    if (m_coord < ceil_div(size<0>(shape_mnkl), trait.bM)) {
        for (int n_coord = 0; n_coord < ceil_div(size<1>(shape_mnkl), trait.bN); ++n_coord) {
            Tensor gC = local_tile(mC_coord, select<0, 1>(trait.tile_mnk), make_coord(m_coord, n_coord, l_coord));  // (BLK_M,BLK_N)
            Tensor tCgC = thr_mma.partition_C(gC);
            Tensor tCrC = partition_fragment_C(trait.tiled_mma, take<0,2>(trait.tile_mnk));
            clear(tCrC);
            gemm_kernel(trait, shape_mnkl, thr_mma,
                        A, dA, tiledcopy_A,
                        B, dB, tiledcopy_B,
                        tCrC,
                        m_coord, n_coord, l_coord,
                        debug);
            auto tCrC_bf16 = op(tCrC);
            copy(copy_c, tCrC_bf16, tCgC);
        }
    }
}

template<typename T, class ProblemShape, class ThrMma,
         class AStride, class TiledCopyA,
         class BStride, class TiledCopyB,
         class CStride, class TiledCopyC>
void gemm_dq(T &trait, ProblemShape &shape_mnkl, ThrMma &thr_mma,
             typename T::DType const *A, AStride dA, TiledCopyA tiledcopy_A,
             typename T::DType const *B, BStride dB, TiledCopyB tiledcopy_B,
             typename T::DType const *C, CStride dC, TiledCopyC tiledcopy_C,
             const int m_coord, const int lq_coord, const int lk_coord,
             bool debug) {
    auto C_shape = select<0,1,3>(shape_mnkl);
    auto mC = make_tensor(make_gmem_ptr(C), make_layout(C_shape, dC));
    auto copy_c = TiledCopyC{mC};
    Tensor mC_coord = cute::get_xe_tensor(C_shape);   //(m,n,l)
    OPS_tobf16<typename T::DType> op;
    if (m_coord < ceil_div(size<0>(shape_mnkl), trait.bM)) {
        for (int n_coord = 0; n_coord < ceil_div(size<1>(shape_mnkl), trait.bN); ++n_coord) {
            Tensor gC = local_tile(mC_coord, select<0, 1>(trait.tile_mnk), make_coord(m_coord, n_coord, lq_coord));  // (BLK_M,BLK_N)
            Tensor tCgC = thr_mma.partition_C(gC);
            Tensor tCrC = partition_fragment_C(trait.tiled_mma, take<0,2>(trait.tile_mnk));
            clear(tCrC);
            gemm_kernel_bb(trait, shape_mnkl, thr_mma,
                           A, dA, tiledcopy_A,
                           B, dB, tiledcopy_B,
                           tCrC,
                           m_coord, n_coord, lq_coord, lk_coord,
                           debug);
            auto tCrC_bf16 = op(tCrC);
            copy(copy_c, tCrC_bf16, tCgC);
        }
    }
}

template<typename T, class ProblemShape, class ThrMma,
         class AStride, class TiledCopyA,
         class BStride, class TiledCopyB,
         class CStride, class TiledCopyC>
void gemm_dkv(T &trait, ProblemShape &shape_mnkl, ThrMma &thr_mma,
              typename T::DType const *A, AStride dA, TiledCopyA tiledcopy_A,
              typename T::DType const *B, BStride dB, TiledCopyB tiledcopy_B,
              typename T::DType const *C, CStride dC, TiledCopyC tiledcopy_C,
              const int m_coord, const int lh_coord, const int lb_coord,
              bool debug) {
    auto C_shape = select<0,1,3>(shape_mnkl);
    auto mC = make_tensor(make_gmem_ptr(C), make_layout(C_shape, dC));
    auto copy_c = TiledCopyC{mC};
    Tensor mC_coord = cute::get_xe_tensor(C_shape);   //(m,n,l)
    OPS_tobf16<typename T::DType> op;
    if (m_coord < ceil_div(size<0>(shape_mnkl), trait.bM)) {
        for (int n_coord = 0; n_coord < ceil_div(size<1>(shape_mnkl), trait.bN); ++n_coord) {
            Tensor gC = local_tile(mC_coord, select<0, 1>(trait.tile_mnk), make_coord(m_coord, n_coord, lh_coord + lb_coord * trait.max_block_n));  // (BLK_M,BLK_N)
            Tensor tCgC = thr_mma.partition_C(gC);
            Tensor tCrC = partition_fragment_C(trait.tiled_mma, take<0,2>(trait.tile_mnk));
            clear(tCrC);
            for (int h_q = lh_coord * trait.block_n_chunk; h_q < (lh_coord + 1) * trait.block_n_chunk; ++h_q) {
                int l_coord = h_q + lb_coord * trait.NUM_HEAD_Q;
                gemm_kernel(trait, shape_mnkl, thr_mma,
                            A, dA, tiledcopy_A,
                            B, dB, tiledcopy_B,
                            tCrC,
                            m_coord, n_coord, l_coord,
                            debug);
            }
            auto tCrC_bf16 = op(tCrC);
            copy(copy_c, tCrC_bf16, tCgC);
        }
    }
}

template<typename T, class ProblemShape, class ThrMma,
         class AStride, class TiledCopyA,
         class BStride, class TiledCopyB,
         class FragTensor>
void gemm_dp(T &trait, ProblemShape &shape_mnkl, ThrMma &thr_mma,
             typename T::DType const *A, AStride dA, TiledCopyA tiledcopy_A,
             typename T::DType const *B, BStride dB, TiledCopyB tiledcopy_B,
             FragTensor &tCrC,
             const int m_coord, const int n_coord, const int lq_coord, const int lk_coord,
             bool debug) {
    gemm_kernel_bb(trait, shape_mnkl, thr_mma,
                   A, dA, tiledcopy_A,
                   B, dB, tiledcopy_B,
                   tCrC,
                   m_coord, n_coord, lq_coord, lk_coord, true);
}

template<typename T, class ProblemShape, class ThrMma,
         class CStride, class TiledCopyC,
         class FragTensor,
         class AuxTensor, class SumTensor>
void softmax_bwd(T &trait, ProblemShape &shape_mnkl, ThrMma &thr_mma,
                 typename T::DType *C, CStride dC, TiledCopyC tiledcopy_C,
                 FragTensor &tCrC,
                 AuxTensor &tCrCy, SumTensor &sum_row,
                 const int m_coord, const int n_coord, const int l_coord, const float softmax_coef) {
    OPS_tobf16<typename T::DType> op;
    auto C_shape = select<0,1,3>(shape_mnkl);
    auto mC = make_tensor(make_gmem_ptr(C), make_layout(C_shape, dC));
    auto copy_c = TiledCopyC{mC};
    Tensor mC_coord = cute::get_xe_tensor(C_shape);   //(m,n,l)
    Tensor gC = local_tile(mC_coord, select<0, 1>(trait.tile_mnk),
                           make_coord(m_coord, n_coord, l_coord)); // dy
    Tensor tCgC = thr_mma.partition_C(gC);
    // y*dy

    softmax_bwd_last(trait, tCrC, tCrCy, sum_row, softmax_coef);
    auto tCrC_bf16 = op(tCrC);
    copy(copy_c, tCrC_bf16, tCgC);
}

template<int NUM_SG, class Tensor, class STensor>
void reduce_row(Tensor &t, STensor &sram) {
    auto group = syclcompat::get_nd_item<1>().get_group();
    auto sg = syclcompat::get_nd_item<1>().get_sub_group();
    const auto sg_local_id = sg.get_local_id();
    const auto sg_group_id = sg.get_group_id();
    const auto sg_group_id_N = sg_group_id % NUM_SG;
    const auto sg_group_id_M = sg_group_id / NUM_SG;
    auto stensor = sram(_, _, sg_group_id_M);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i< size(t); ++i) {
        t(i) = reduce_over_group(sg, t(i), sycl::plus<>());
    }

    if (sg_local_id == 0) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(t); ++i) {
            stensor(i, sg_group_id_N) = t(i);
        }
    }
    // have to wait here
    sycl::group_barrier(group);
    if (sg_local_id == 0) {
        for (int i = 0; i < size(t); ++i) {
            t(i) = 0.0f;
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < NUM_SG; ++j) {
                t(i) += stensor(i, j);
            }
        }
    }
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(t); ++i) {
        t(i) = sycl::group_broadcast(sg, t(i), 0);
    }
}

template<typename T, typename V>
void copy_tensor(T &src, V &dst) {
    // static_assert(size(src) == size(dst));
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(src); ++i) {
        dst(i) = src(i);
    }
}

template<class T, int Nblk, bool dropout, bool first>
void
mha_backward(T trait,
             typename T::DType const *go_d,
             typename T::DType const *q_d,
             typename T::DType const *k_d,
             typename T::DType const *v_d,
             typename T::DType const *ps_d,
             typename T::DType const *psd_d,
             typename T::DType *gq_d,
             typename T::DType *gk_d,
             typename T::DType *gv_d,
             typename T::DType *gps_d) {
    float softmax_coef = 1.0f / sqrtf(static_cast<float>(trait.HEAD_SIZE_QK));
    auto sg = syclcompat::get_nd_item<1>().get_sub_group();
    auto first_thread_in_sg_idx = sg.get_group_linear_id() * trait.SubgroupSize;
    auto thr_mma = trait.tiled_mma.get_slice(first_thread_in_sg_idx);
    // std::is_same_v<decltype(copy_Pst), int>;
    int m_coord = BlockIdxX();
    int lh_coord = BlockIdxY();
    int lb_coord = BlockIdxZ();
    using ProblemShape = cute::tuple<int, int, int, int>;
    using ProblemShapeEx = cute::tuple<int, int, int, int, int>;
    OPS_tobf16<typename T::DType> op;

    if constexpr(first) {
    {
        /*
          dV=Pt * dO
          dV BATCH,NUM_HEAD_KV,SEQ_LEN_KV,HEAD_SIZE_VO
          Pt BATCH,NUM_HEAD_Q,SEQ_LEN_KV,SEQ_LEN_QO
          dO BATCH,NUM_HEAD_Q,SEQ_LEN_QO,HEAD_SIZE_VO
          M SEQ_LEN_KV
          N HEAD_SIZE_VO
          K SEQ_LEN_QO
        */
        auto dPt = make_stride(Int<1>{}, trait.SEQ_LEN_KV,
                               trait.SEQ_LEN_KV *trait.SEQ_LEN_QO); // A SEQ_LEN_KV,SEQ_LEN_QO
        auto dO = make_stride(Int<1>{}, trait.HEAD_SIZE_VO,
                              trait.HEAD_SIZE_VO *trait.SEQ_LEN_QO); // B HEAD_SIZE_VO,SEQ_LEN_QO
        auto dV = make_stride(trait.HEAD_SIZE_VO, Int<1>{},
                              trait.SEQ_LEN_KV *trait.HEAD_SIZE_VO); // C SEQ_LEN_KV,HEAD_SIZE_VO
        ProblemShape problem_shape = ProblemShape(
            trait.SEQ_LEN_KV,
            trait.HEAD_SIZE_VO,
            trait.SEQ_LEN_QO,
            trait.BATCH * trait.NUM_HEAD_Q);
        if constexpr(dropout) {
            gemm_dkv(trait, problem_shape, thr_mma,
                     psd_d, dPt, typename T::CopyPt{},
                     go_d, dO, typename T::CopygOB{},
                     gv_d, dV, typename T::CopygV{},
                     m_coord, lh_coord, lb_coord, false);
        } else {
            gemm_dkv(trait, problem_shape, thr_mma,
                     ps_d, dPt, typename T::CopyPt{},
                     go_d, dO, typename T::CopygOB{},
                     gv_d, dV, typename T::CopygV{},
                     m_coord, lh_coord, lb_coord, false);
        }
    }
    {
        /*
          shape
          dO BATCH,NUM_HEAD_Q,SEQ_LEN_QO,HEAD_SIZE_VO
          V  BATCH,NUM_HEAD_Q,SEQ_LEN_KV,HEAD_SIZE_VO
          dP BATCH,NUM_HEAD_Q,SEQ_LEN_QO,SEQ_LEN_KV
          M SEQ_LEN_KV
          N HEAD_SIZE_VO
          K SEQ_LEN_QO
          dP=dO*Vt
         */
        auto dO = make_stride(trait.HEAD_SIZE_VO, Int<1>{},
                              trait.SEQ_LEN_QO *trait.HEAD_SIZE_VO); // A SEQ_LEN_QO, HEAD_SIZE_VO
        auto dV = make_stride(trait.HEAD_SIZE_VO, Int<1>{},
                              trait.HEAD_SIZE_VO * trait.SEQ_LEN_KV); // B SEQ_LEN_KV, HEAD_SIZE_VO
        auto dP = make_stride(trait.SEQ_LEN_KV, Int<1>{},
                              trait.SEQ_LEN_QO *trait.SEQ_LEN_KV); // C SEQ_LEN_QO,SEQ_LEN_KV

        auto tCrC = partition_fragment_C(trait.tiled_mma, take<0, 2>(trait.tile_mnk)); // dy

        constexpr auto dimN = Int<Nblk>{};
        constexpr auto dim0 = size<0>(tCrC);
        constexpr auto dim1 = size<1>(tCrC);
        constexpr auto dim2 = size<2>(tCrC);
        auto dProw = make_tensor<float>(make_shape(dim0, dim1, dim2, dimN));
        auto tCrCy = make_tensor<typename T::DType>(make_shape(dim0, dim1, dim2, dimN));
// y
        auto tCrCyd = make_tensor_like<typename T::DType>(tCrCy);
        // init sum_row
        auto sum_row = make_tensor_like(tCrC(_, _, 0));
        constexpr auto NUM_COL_PER_THD = size<2>(tCrC);
        constexpr auto NUM_ROW_PER_THD = size<0>(tCrC) * size<1>(tCrC);
        constexpr auto NUM_SG_PER_ROW = trait.bN / (Int<trait.SubgroupSize>{} * NUM_COL_PER_THD);
        constexpr auto NUM_SG_PER_BLK_M = trait.bM / NUM_ROW_PER_THD;
        constexpr auto NUM_SG = NUM_SG_PER_ROW * NUM_SG_PER_BLK_M;

        // leverage share memory to reduce over 1 block
        auto smem = syclcompat::local_mem<float[NUM_SG * NUM_ROW_PER_THD]>();
        Tensor stensor = make_tensor(make_smem_ptr(smem), make_shape(Int<NUM_ROW_PER_THD>{}, Int<NUM_SG_PER_ROW>{}, Int<NUM_SG_PER_BLK_M>{})); // bank conflict
        ProblemShapeEx problem_shape = ProblemShapeEx(
            trait.SEQ_LEN_QO,
            trait.SEQ_LEN_KV,
            trait.HEAD_SIZE_VO,
            trait.BATCH * trait.NUM_HEAD_Q,
            trait.BATCH * trait.NUM_HEAD_KV);
        if (m_coord < ceil_div(trait.SEQ_LEN_QO, trait.bM)) {
            int lk_coord = lh_coord + lb_coord * trait.max_block_n;
            for (int h_q = lh_coord * trait.block_n_chunk; h_q < (lh_coord + 1)* trait.block_n_chunk; ++h_q) {
                int lq_coord = h_q + lb_coord * trait.NUM_HEAD_Q;
                clear(tCrCy);
                clear(sum_row);
                for (int n_coord = 0; n_coord<ceil_div(size<1>(problem_shape), trait.bN); ++n_coord) {
                    clear(tCrC);
                    gemm_dp(trait, problem_shape, thr_mma,
                            go_d, dO, typename T::CopygOA{},
                            v_d, dV, typename T::CopyVt{},
                            tCrC,
                            m_coord, n_coord, lq_coord, lk_coord, false);
                    auto dPn = dProw(_, _, _, n_coord);
                    // copy dP
                    copy_tensor(tCrC, dPn);
                    // softmax partial reduce
                    auto Pn = tCrCy(_, _, _, n_coord);
                    load_y(trait, problem_shape, thr_mma, ps_d, dP, typename T::CopyP{}, Pn, m_coord, n_coord, lq_coord); // load attn without dropout
                    if constexpr(dropout) {
                        auto Pdn = tCrCyd(_, _, _, n_coord);
                        load_y(trait, problem_shape, thr_mma, psd_d, dP, typename T::CopyP{}, Pdn, m_coord, n_coord, lq_coord); // load attn with dropout
                        softmax_bwd_partial_sum(dPn, Pdn, sum_row);
                    } else {
                        softmax_bwd_partial_sum(dPn, Pn, sum_row);
                    }
                }
                reduce_row<NUM_SG_PER_ROW, decltype(sum_row), decltype(stensor)>(sum_row, stensor);
                for (int n_coord = 0; n_coord<ceil_div(size<1>(problem_shape), trait.bN); ++n_coord) {
                    auto dPn = dProw(_, _, _, n_coord);
                    auto Pn = tCrCy(_, _, _, n_coord);
                    softmax_bwd(trait, problem_shape, thr_mma,
                                gps_d, dP, typename T::CopygP{},
                                dPn, Pn, sum_row,
                                m_coord, n_coord, lq_coord, softmax_coef);
                }
            }
        }
    }
    } else {
    // store in smem for transpose syk
    // auto group = syclcompat::get_nd_item<1>().get_group();
    // sycl::group_barrier(group);
    {
        /*
          dS BATCH,NUM_HEAD_Q,SEQ_LEN_QO,HEAD_SIZE_VO
          K  BATCH,NUM_HEAD_KV,SEQ_LEN_KV,HEAD_SIZE_QK
          dQ BATCH,NUM_HEAD_Q,SEQ_LEN_QO,HEAD_SIZE_QK
          M  SEQ_LEN_QO
          N  HEAD_SIZE_QK
          K  SEQ_LEN_KV
          dQ=dS*K
         */
        auto dS = make_stride(trait.SEQ_LEN_KV, Int<1>{},
                              trait.SEQ_LEN_KV *trait.SEQ_LEN_QO); // A SEQ_LEN_QO, SEQ_LEN_KV
        auto dK = make_stride(Int<1>{}, trait.HEAD_SIZE_QK,
                              trait.HEAD_SIZE_QK *trait.SEQ_LEN_KV); // B HEAD_SIZE_QK, SEQ_LEN_KV
        auto dQ = make_stride(trait.HEAD_SIZE_QK, Int<1>{},
                              trait.SEQ_LEN_QO * trait.HEAD_SIZE_QK); // C SEQ_LEN_QO, HEAD_SIZE_QK
        ProblemShapeEx problem_shape = ProblemShapeEx(
            trait.SEQ_LEN_QO,
            trait.HEAD_SIZE_QK,
            trait.SEQ_LEN_KV,
            trait.BATCH * trait.NUM_HEAD_Q,
            trait.BATCH * trait.NUM_HEAD_KV);
        int lk_coord = lh_coord + lb_coord * trait.max_block_n;
        for (int h_q = lh_coord * trait.block_n_chunk; h_q < (lh_coord + 1) * trait.block_n_chunk; ++h_q) {
            int lq_coord = h_q + lb_coord * trait.NUM_HEAD_Q;
            gemm_dq(trait, problem_shape, thr_mma,
                    gps_d, dS, typename T::CopygS{},
                    k_d, dK, typename T::CopyK{},
                    gq_d, dQ, typename T::CopygQ{},
                    m_coord, lq_coord, lk_coord, false);
        }
    }
    {
        /*
          dS BATCH,NUM_HEAD_Q,SEQ_LEN_QO,SEQ_LEN_KV
          Q  BATCH,NUM_HEAD_Q,SEQ_LEN_QO,HEAD_SIZE_QK
          dK BATCH,NUM_HEAD_KV,SEQ_LEN_KV,HEAD_SIZE_QK
          M SEQ_LEN_KV
          N HEAD_SIZE_QK
          K SEQ_LEN_QO
          dK=dSt*Q
         */
        auto dS = make_stride(Int<1>{}, trait.SEQ_LEN_KV,
                              trait.SEQ_LEN_KV *trait.SEQ_LEN_QO); // A SEQ_LEN_KV,SEQ_LEN_QO
        auto dQ = make_stride(Int<1>{}, trait.HEAD_SIZE_QK,
                              trait.SEQ_LEN_QO *trait.HEAD_SIZE_QK); // B SEQ_LEN_QO,HEAD_SIZE_QK
        auto dK = make_stride(trait.HEAD_SIZE_QK, Int<1>{},
                              trait.HEAD_SIZE_QK *trait.SEQ_LEN_KV); // C SEQ_LEN_KV,HEAD_SIZE_QK

        ProblemShape problem_shape = ProblemShape(
            trait.SEQ_LEN_KV,
            trait.HEAD_SIZE_QK,
            trait.SEQ_LEN_QO,
            trait.BATCH * trait.NUM_HEAD_Q);
        gemm_dkv(trait, problem_shape, thr_mma,
                 gps_d, dS, typename T::CopygSt{},
                 q_d, dQ, typename T::CopyQ{},
                 gk_d, dK, typename T::CopygK{},
                 m_coord, lh_coord, lb_coord, false);
    }
    }
}

template<typename T, class ProblemShape, int nBlk, bool dropout>
void launch_mha_backward(ProblemShape problem_shape,
                         T *go_d,
                         const T *q_d,
                         const T *k_d,
                         const T *v_d,
                         const T *ps_d,
                         const T *psd_d,
                         T *dq_d,
                         T *dk_d,
                         T *dv_d,
                         T *dps_d) {

    auto trait = MHA_TYPE<T, ProblemShape>(problem_shape);

    auto dimGrid = syclcompat::dim3(size(ceil_div(trait.max_block_m, trait.bM)), size(trait.max_block_n), size(trait.BATCH));
    assert((trait.NUM_HEAD_Q % trait.NUM_HEAD_KV == 0) && "num_head_q must be dividable by num_head_kv");
    assert((trait.NUM_HEAD_Q >= trait.NUM_HEAD_KV) && "num_head_q must be bigger than or equal to num_head_kv");
    assert((trait.bNi <= trait.SEQ_LEN_KV) && "tile_N must be larger than SEQ_LEN_KV");
    auto dimBlock = syclcompat::dim3(size(trait.mmaC), size(1), size(1));

    std::cout << "Launch mha bwd kernel with: " <<
        "batch_size: " << trait.BATCH <<
        ", num_head_q: " << trait.NUM_HEAD_Q <<
        ", num_head_kv: " << trait.NUM_HEAD_KV <<
        ", head_dim_qk: " << trait.HEAD_SIZE_QK <<
        ", head_dim_v: " << trait.HEAD_SIZE_VO <<
        ", seq_len_q: " << trait.SEQ_LEN_QO <<
        ", seq_len_kv: " << trait.SEQ_LEN_KV << std::endl;

    syclcompat::experimental::launch_properties launch_props{
        // sycl::ext::oneapi::experimental::work_group_scratch_size(0),
    };
    syclcompat::experimental::kernel_properties kernel_props{
        sycl::ext::oneapi::experimental::sub_group_size<trait.SubgroupSize>};
    syclcompat::experimental::launch_policy policy{dimGrid, dimBlock, launch_props, kernel_props};
    auto event1 = syclcompat::experimental::launch<
        mha_backward<decltype(trait), nBlk, dropout, true>>(policy,
                                                            trait,
                                                            go_d,
                                                            q_d, k_d, v_d,
                                                            ps_d, psd_d,
                                                            dq_d, dk_d, dv_d,
                                                            dps_d);
    EventManager::getInstance().addEvent(event1);
    auto event2 = syclcompat::experimental::launch<
        mha_backward<decltype(trait), nBlk, dropout, false>>(policy,
                                                             trait,
                                                             go_d,
                                                             q_d, k_d, v_d,
                                                             ps_d, psd_d,
                                                             dq_d, dk_d, dv_d,
                                                             dps_d);
    EventManager::getInstance().addEvent(event2);
}

template<typename T, class ProblemShape>
void launch_mha_bwd_wrapper(ProblemShape problem_shape,
                            T *go_d,
                            const T *q_d,
                            const T *k_d,
                            const T *v_d,
                            const T *ps_d,
                            const T *psd_d,
                            T *dq_d,
                            T *dk_d,
                            T *dv_d,
                            T *dps_d) {
    int SEQ_LEN_KV = get<4>(problem_shape);
    if (psd_d == nullptr) {
        if (SEQ_LEN_KV <= 1024) {
            constexpr int nBlk = 4;
            launch_mha_backward<T, ProblemShape, nBlk, false>(problem_shape,
                                             go_d,
                                             q_d, k_d, v_d,
                                             ps_d, psd_d,
                                             dq_d, dk_d, dv_d,
                                             dps_d);
        } else if (SEQ_LEN_KV <= 512) {
            constexpr int nBlk = 2;
            launch_mha_backward<T, ProblemShape, nBlk, false>(problem_shape,
                                             go_d,
                                             q_d, k_d, v_d,
                                             ps_d, psd_d,
                                             dq_d, dk_d, dv_d,
                                             dps_d);
        } else if (SEQ_LEN_KV <= 256) {
            constexpr int nBlk = 1;
            launch_mha_backward<T, ProblemShape, nBlk, false>(problem_shape,
                                             go_d,
                                             q_d, k_d, v_d,
                                             ps_d, psd_d,
                                             dq_d, dk_d, dv_d,
                                             dps_d);
        }
    } else {
        if (SEQ_LEN_KV <= 1024) {
            constexpr int nBlk = 4;
            launch_mha_backward<T, ProblemShape, nBlk, true>(problem_shape,
                                             go_d,
                                             q_d, k_d, v_d,
                                             ps_d, psd_d,
                                             dq_d, dk_d, dv_d,
                                             dps_d);
        } else if (SEQ_LEN_KV <= 512) {
            constexpr int nBlk = 2;
            launch_mha_backward<T, ProblemShape, nBlk, true>(problem_shape,
                                             go_d,
                                             q_d, k_d, v_d,
                                             ps_d, psd_d,
                                             dq_d, dk_d, dv_d,
                                             dps_d);
        } else if (SEQ_LEN_KV <= 256) {
            constexpr int nBlk = 1;
            launch_mha_backward<T, ProblemShape, nBlk, true>(problem_shape,
                                             go_d,
                                             q_d, k_d, v_d,
                                             ps_d, psd_d,
                                             dq_d, dk_d, dv_d,
                                             dps_d);
        }
    }
}

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
    
    using T = cute::bfloat16_t;
    ProblemShapeRegular problem_shape{batch_size, num_head_q, num_head_kv, seq_len_q, seq_len_kv, head_dim_qk, head_dim_v};
    launch_mha_bwd_wrapper<T, ProblemShapeRegular>(problem_shape,
                                                  (T*)grad_out,
                                                  (const T*)query,
                                                  (const T*)key,
                                                  (const T*)value,
                                                  (const T*) ps,
                                                  (const T*)psd,
                                                    (T*)grad_query,
                                                    (T*)grad_key,
                                                    (T*)grad_value,
                                                    (T*)dps);
}

template void launch_mha_bwd_wrapper<cute::bfloat16_t, ProblemShapeRegular>(ProblemShapeRegular problem_shape,
                            cute::bfloat16_t *go_d,
                            const cute::bfloat16_t *q_d,
                            const cute::bfloat16_t *k_d,
                            const cute::bfloat16_t *v_d,
                            const cute::bfloat16_t *ps_d,
                            const cute::bfloat16_t *psd_d,
                            cute::bfloat16_t *dq_d,
                            cute::bfloat16_t *dk_d,
                            cute::bfloat16_t *dv_d,
                            cute::bfloat16_t *dps_d);