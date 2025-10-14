#include <cstdio>
#include <cstddef>
#include <cassert>
#include <cstdlib>
#include <math.h>
#include <string>

#include <sycl/sycl.hpp>
#include <cute/util/compat.hpp>
#include <cute/tensor.hpp>

#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/print_error.hpp"
#include "cutlass/util/sycl_event_manager.hpp"
#include "cutlass/util/GPU_Clock.hpp"

#include "AttentionKernels.h"

using namespace cute;

#include <sycl/sycl.hpp>
#include <cute/util/compat.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <cute/tensor.hpp>

#include "cutlass/kernel_hardware_info.h"
#include "cutlass/platform/platform.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/util/sycl_event_manager.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"

#include "sycl_cute_common.hpp"

#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

using namespace cute;


template <class ATensor, class BTensor, class CTensor,
          class TiledMMA>
void
gemm_device(ATensor   const& A,         // (M,K)
            BTensor   const& B,         // (N,K)
            CTensor        & C,         // (M,N)
            TiledMMA const & mma)
{
  // -----
  // Setup
  // -----

  /* Get workgroup and local IDs */
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<2>();
  auto wg_m = int(item.get_group(1));
  auto wg_n = int(item.get_group(0));
  auto local_id = int(item.get_local_id(0));

  /* Create proxy coordinate tensors for each global tensor */
  Tensor cA = make_identity_tensor(A.shape());   // (M,K)
  Tensor cB = make_identity_tensor(B.shape());   // (N,K)
  Tensor cC = make_identity_tensor(C.shape());   // (M,N)

  /* Split GEMM into workgroup tiles, and identify our workgroup's tile (wg_coord) */
  auto wg_tile = mma.tile_mnk();
  auto wg_coord = make_coord(wg_m, wg_n, 0);

  Tensor gA = local_tile(cA, select<0,2>(wg_tile), make_coord(wg_m,_));  // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(cB, select<1,2>(wg_tile), make_coord(wg_n,_));  // (BLK_N,BLK_K,k)
  Tensor gC = local_tile(cC, wg_tile, wg_coord, Step<_1,_1, X>{});       // (BLK_M,BLK_N)

  /* Create block 2D TiledCopies */
  auto copy_a = make_block_2d_copy_A(mma, A);
  auto copy_b = make_block_2d_copy_B(mma, B);
  auto copy_c = make_block_2d_copy_C(mma, C);
  if (item.get_global_linear_id() == 0){
    print("\nlfq debug \n");
    print(copy_a);
  }
    

  /* Slice TiledCopy/TiledMMA operations to thread (work-item) level */
  auto thr_mma    =    mma.get_slice(local_id);
  auto thr_copy_a = copy_a.get_slice(local_id);
  auto thr_copy_b = copy_b.get_slice(local_id);

  /* Register fragments for MMA */
  auto tCrA = thr_mma.partition_sg_fragment_A(gA(_,_,0));
  auto tCrB = thr_mma.partition_sg_fragment_B(gB(_,_,0));

  /* Register fragments for copies */
  auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_,_,0));
  auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_,_,0));

  /* Partition global tensor (proxies) for copies */
  Tensor tAgA = thr_copy_a.partition_S(gA);
  Tensor tBgB = thr_copy_b.partition_S(gB);

  /* Partition C */
  Tensor tCrC = partition_fragment_C(mma, select<0,1>(wg_tile));
  Tensor tCgC = thr_mma.partition_C(gC);    /* also matches copy_c's source layout */

  /* Create prefetch TiledCopy instances */
  auto prefetch_a = make_block_2d_prefetch(copy_a);
  auto prefetch_b = make_block_2d_prefetch(copy_b);

  auto thr_prefetch_A = prefetch_a.get_slice(local_id);
  auto thr_prefetch_B = prefetch_b.get_slice(local_id);

  /* Partition global tensor (proxies) for prefetch */
  auto pAgA = thr_prefetch_A.partition_S(gA);
  auto pBgB = thr_prefetch_B.partition_S(gB);

  /* Prefetch distance, in units of k tiles */
  const int prefetch_dist = 3;

  // ------
  // Kernel
  // ------

  constexpr int barrier_scope = 2;

  int k_tile_count = ceil_div(shape<1>(A), get<2>(wg_tile));
  int k_tile_prefetch = 0;

  /* Clear the accumulators */
  clear(tCrC);

  /* Warm up loops with prefetch to L1 */
  CUTE_UNROLL
  for (; k_tile_prefetch < prefetch_dist; k_tile_prefetch++) {
    prefetch(prefetch_a, pAgA(_,_,_,k_tile_prefetch));
    prefetch(prefetch_b, pBgB(_,_,_,k_tile_prefetch));
  }

  /* Main loop */
  for (int k_tile = 0; k_tile < k_tile_count; k_tile++, k_tile_prefetch++) {
    /* Split barrier keeping threads loosely together */
    barrier_arrive(barrier_scope);

    /* Copy A/B from global memory (ideally L1 cache) to registers */
    copy(copy_a, tAgA(_,_,_,k_tile), tArA);
    copy(copy_b, tBgB(_,_,_,k_tile), tBrB);

    /* Prefetch A/B tiles to L1 */
    prefetch(prefetch_a, pAgA(_,_,_,k_tile_prefetch));
    prefetch(prefetch_b, pBgB(_,_,_,k_tile_prefetch));

    /* Shuffle data from copy fragments to MMA fragments */
    reorder(tArA, tCrA);
    reorder(tBrB, tCrB);

    /* Accumulate C += A * B */
    gemm(mma, tCrA, tCrB, tCrC);

    /* Other half of split barrier */
    barrier_wait(barrier_scope);
  }

  /* Write C to global memory */
  copy(copy_c, tCrC, tCgC);
}



template <typename T, size_t = 0>
struct is_complete : std::false_type {};

template <typename T>
struct is_complete<T, 0 * sizeof(T)> : std::true_type {};

template <typename T>
static constexpr bool is_complete_v = is_complete<T>::value;


template <typename TA, typename TB, typename TC>
auto
choose_mma_op()
{
  if constexpr (is_complete_v<XE_DPAS_TT<8, TC, TA, TB>>)
    return XE_DPAS_TT<8, TC, TA, TB>{};
  else if constexpr (is_same_v<TA, cute::bfloat16_t>)
    return XE_DPAS_TT<8, float, cute::bfloat16_t>{};
  else  /* Use f16 by default as upconversion sequences are typically faster */
    return XE_DPAS_TT<8, float, cute::half_t>{};
}

template <class ATensor, class BTensor, class CTensor>
auto
choose_tiled_mma(ATensor const& A, BTensor const& B, CTensor const&)
{
  using TA = typename ATensor::element_type;
  using TB = typename BTensor::element_type;
  using TC = typename CTensor::element_type;

  auto op = choose_mma_op<TA,TB,TC>();

  constexpr bool byte = (cute::max(sizeof_bits_v<TA>, sizeof_bits_v<TB>) <= 8);
  constexpr bool use_1x_dpas_per_k = is_constant_v<1, decltype(stride<0>(A))>             // Use one DPAS in k dimension for A^T case
                                  || (byte && is_constant_v<1, decltype(stride<0>(B))>);  //  pending compiler improvements (also int8 B^N)

  using _K = conditional_t<use_1x_dpas_per_k,
                           C<op.K>, C<op.K*2>>;

  using WGTile = Shape<_256, _256, _K>;                               // 256x256 WG tile size
  using SGLayout = Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>;     // 8x4 SG tiling, n-major

  using MMA = typename TiledMMAHelper<MMA_Atom<decltype(op)>, Layout<WGTile>, SGLayout>::TiledMMA;

  return MMA{};
}

template <class, class, char, char> class GemmCuteName;
template <class ATensor, class BTensor, class CTensor, typename TA, typename TB, char layoutA, char layoutB>
void
gemm_cute(sycl::queue &Q,
          ATensor   const& A,         // (M,K)
          BTensor   const& B,         // (N,K)
          CTensor        & C)         // (M,N)
{
  auto mma = choose_tiled_mma(A, B, C);

  sycl::range<2> local = {size(mma), 1};
  sycl::range<2> global = {local[0] * ceil_div(shape<0>(B), get<1>(mma.tile_mnk())),
                           local[1] * ceil_div(shape<0>(A), get<0>(mma.tile_mnk()))};

  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;

  syclex::properties kernel_props {
    syclex::sub_group_size<16>,
    intelex::grf_size<256>
  };

  auto event = Q.parallel_for<GemmCuteName<TA, TB, layoutA, layoutB>>(sycl::nd_range<2>(global, local), kernel_props,
    [=](auto) {
      gemm_device(A, B, C, mma);
    }
  );

  EventManager::getInstance().addEvent(event);
}

template <class...> class GemmVerifyKernelName;
template <class ATensor, class BTensor, class CTensor>
bool
gemm_verify(sycl::queue &Q,
            ATensor const& A,         // (M,K)
            BTensor const& B,         // (N,K)
            CTensor const& C)         // (M,N)
{
  int m = size<0>(A);
  int n = size<0>(B);
  int k = size<1>(A);

  auto ok = sycl::malloc_shared<bool>(1, Q);
  *ok = true;

  Q.parallel_for<GemmVerifyKernelName<ATensor, BTensor, CTensor>>(sycl::range<2>(m, n), [=](sycl::item<2> id) {
    int i = id[0], j = id[1];

    using AccType = typename CTensor::element_type;
    using SignedAccType = ensure_signed_t<AccType>;

    auto c = AccType(0);
    for (int h = 0; h < k; h++)
      c += AccType(A(i,h)) * AccType(B(j,h));

    auto tol = AccType(1e-5f * k);
    if (std::abs(SignedAccType(c - AccType(C(i,j)))) > tol) {
#ifdef SHOW_DIFF
      printf("Error at (%d,%d): got %f, expected %f\n", i, j, double(C(i,j)), double(c));
#endif
      *ok = false;
    }
  }).wait();

  bool read_ok = *ok;

  sycl::free(ok, Q);

  return read_ok;
}

template <typename TA, typename TB, typename TC,
          char layoutA = 'R', char layoutB = 'R'>
void
test_case(sycl::queue &Q, int m, int n, int k)
{
  std::cout << type_str<TA>() << " (" << layoutA << ") x "
            << type_str<TB>() << " (" << layoutB << ") -> "
            << type_str<TC>() << ": \t";

  // Transpose B to match CuTe conventions
  constexpr char tlayoutB = layoutB ^ ('R' ^ 'C');

  // Prepare data:
  auto A = make_shared_usm_tensor<TA,  layoutA>(Q, m, k);
  auto B = make_shared_usm_tensor<TB, tlayoutB>(Q, n, k);
  auto C = make_shared_usm_tensor<TC,      'R'>(Q, m, n);

  random_fill(A);
  random_fill(B);
  zero_fill(C);

#ifndef SKIP_VERIFY
  auto A_ref = make_shared_usm_tensor<float,  layoutA>(Q, m, k);
  auto B_ref = make_shared_usm_tensor<float, tlayoutB>(Q, n, k);

  copy(A, A_ref);
  copy(B, B_ref);
#endif

  subbyte_pack(A);
  subbyte_pack(B);

  // Test accuracy:
  gemm_cute<decltype(A), decltype(B), decltype(C), TA, TB, layoutA, layoutB>(Q, A, B, C);
  Q.wait_and_throw();

#ifdef SKIP_VERIFY
  const bool ok = true;
  std::cout << "verification skipped";
#else
  bool ok = gemm_verify(Q, A_ref, B_ref, C);
  std::cout << (ok ? "passed" : "failed");
#endif

  if (ok) {
    // Test performance:
    const int timing_iterations = 100;
    GPU_Clock timer;

    timer.start();
    for (int i = 0; i < timing_iterations; ++i)
      gemm_cute<decltype(A), decltype(B), decltype(C), TA, TB, layoutA, layoutB>(Q, A, B, C);
    Q.wait_and_throw();

    double avg = timer.seconds() / timing_iterations;
    double tops = (2.0*m*n*k) * 1e-12;

    printf(", %4.3f TF/s", tops / avg, avg*1000);
  }

  free_usm_tensor(A, Q);
  free_usm_tensor(B, Q);
  free_usm_tensor(C, Q);

#ifndef SKIP_VERIFY
  free_usm_tensor(A_ref, Q);
  free_usm_tensor(B_ref, Q);
#endif

  std::cout << '\n';

  // Pause for a short period of time to allow the GPU to cool.
  static bool first = true;
  if (first)
    first = false;
  else
    sleep(1);
}


void temp_entry()
{
  auto m = 128;
  auto n = 128;
  auto k = 128;

  sycl::queue Q;

  test_case<half_t, half_t, float, 'R', 'R'>(Q, m, n, k);
  test_case<half_t, half_t, float, 'R', 'C'>(Q, m, n, k);
  test_case<half_t, half_t, float, 'C', 'R'>(Q, m, n, k);
  return;
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
    temp_entry();
}
