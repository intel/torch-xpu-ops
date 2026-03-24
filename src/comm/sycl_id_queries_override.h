/*
 * Override __SYCL_ID_QUERIES_FIT_IN_INT__ for host compilation only.
 *
 * In pure icpx mode (no -fsycl-host-compiler), icpx predefines
 * __SYCL_ID_QUERIES_FIT_IN_INT__ to 1 which injects a host-side
 * runtime check that throws when global_range > INT_MAX.
 *
 * Setting the macro to 0 on the host pass disables that check.
 * The device pass (__SYCL_DEVICE_ONLY__) keeps the original value
 * so IGC retains 32-bit ID optimizations and avoids exit code 245.
 *
 * This header is force-included via -include before any source code,
 * ensuring the override takes effect before any SYCL header is parsed.
 */
#ifndef __SYCL_DEVICE_ONLY__
#undef __SYCL_ID_QUERIES_FIT_IN_INT__
#define __SYCL_ID_QUERIES_FIT_IN_INT__ 0
#endif
