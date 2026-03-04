/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * The Philox4x32 algorithm implemented in this file is based on:
 *   "Parallel Random Numbers: As Easy as 1, 2, 3"
 *   Salmon, Moraes, Dror, Shaw - D.E. Shaw Research, SC'11
 *   http://www.thesalmons.org/john/random123/papers/random123sc11.pdf
 *
 * Copyright 2010-2011, D. E. Shaw Research.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 * * Redistributions of source code must retain the above copyright
 *   notice, this list of conditions, and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions, and the following disclaimer in the
 *   documentation and/or other materials provided with the distribution.
 *
 * * Neither the name of D. E. Shaw Research nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <ATen/core/DistributionsHelper.h>
#include <ATen/native/xpu/sycl/MemoryAccess.h>

namespace at {
namespace native {
namespace xpu {

#define EXTRA_FLAG_NORMAL 0x00000001

template <typename T>
struct alignas(sizeof(T) * 2) rand_vec2 {
  union {
    T val[2];
    struct {
      T x, y;
    };
  };
  inline rand_vec2() {}
  inline rand_vec2(T x_, T y_) : x(x_), y(y_) {}
};

template <typename T>
struct alignas(sizeof(T) * 4) rand_vec4 {
  union {
    T val[4];
    struct {
      T x, y, z, w;
    };
  };
  inline rand_vec4() {}
  inline rand_vec4(T x_, T y_, T z_, T w_) : x(x_), y(y_), z(z_), w(w_) {}
};

typedef rand_vec4<float> float4;
typedef rand_vec2<float> float2;
typedef rand_vec2<double> double2;
typedef rand_vec4<uint32_t> uint4;
typedef rand_vec2<uint32_t> uint2;
typedef rand_vec2<uint64_t> ulonglong2;

constexpr uint32_t kPhiloxWeylSequence0 = 0x9E3779B9;
constexpr uint32_t kPhiloxWeylSequence1 = 0xBB67AE85;
constexpr uint32_t kPhiloxMultiplier0 = 0xD2511F53;
constexpr uint32_t kPhiloxMultiplier1 = 0xCD9E8D57;
constexpr unsigned int kPhilox4x32Rounds = 10;

struct PhiloxState {
  uint4 counter;
  uint4 result;
  uint2 key;
  unsigned int index;
  int boxmuller_flag;
  int boxmuller_flag_double;
  float boxmuller_extra;
};

using randStatePhilox4_32_10_t = PhiloxState;

static inline void philox_counter_incr(PhiloxState* state, uint64_t n) {
  const uint32_t low_word = static_cast<uint32_t>(n);
  const uint32_t high_word = static_cast<uint32_t>(n >> 32);

  const uint32_t old_c0 = state->counter.val[0];
  state->counter.val[0] += low_word;
  uint32_t carry = (state->counter.val[0] < old_c0) ? 1u : 0u;

  const uint32_t old_c1 = state->counter.val[1];
  state->counter.val[1] += high_word + carry;
  carry = (state->counter.val[1] < old_c1 ||
           (carry && state->counter.val[1] == old_c1))
      ? 1u
      : 0u;

  if (carry == 0)
    return;

  const uint32_t old_c2 = state->counter.val[2];
  state->counter.val[2] += carry;
  if (state->counter.val[2] >= old_c2)
    return;

  state->counter.val[3] += 1u;
}

static inline void philox_counter_incr_single(PhiloxState* state) {
  if (++state->counter.val[0] != 0)
    return;
  if (++state->counter.val[1] != 0)
    return;
  if (++state->counter.val[2] != 0)
    return;
  ++state->counter.val[3];
}

static inline void philox_counter_incr_high(PhiloxState* state, uint64_t n) {
  const uint32_t low_word = static_cast<uint32_t>(n);
  const uint32_t high_word = static_cast<uint32_t>(n >> 32);

  const uint32_t old_c2 = state->counter.val[2];
  state->counter.val[2] += low_word;
  const uint32_t carry = (state->counter.val[2] < old_c2) ? 1u : 0u;

  state->counter.val[3] += high_word + carry;
}

static inline uint32_t philox_multiply_high_low(
    uint32_t multiplier,
    uint32_t multiplicand,
    uint32_t* high_product) {
  *high_product = sycl::mul_hi(multiplier, multiplicand);
  return multiplier * multiplicand;
}

static inline uint4 philox4x32_single_round(uint4 input, uint2 round_key) {
  uint32_t product_hi_0, product_hi_1;
  const uint32_t product_lo_0 =
      philox_multiply_high_low(kPhiloxMultiplier0, input.val[0], &product_hi_0);
  const uint32_t product_lo_1 =
      philox_multiply_high_low(kPhiloxMultiplier1, input.val[2], &product_hi_1);

  return uint4{
      product_hi_1 ^ input.val[1] ^ round_key.val[0],
      product_lo_1,
      product_hi_0 ^ input.val[3] ^ round_key.val[1],
      product_lo_0};
}

static inline uint2 philox4x32_bump_key(uint2 key) {
  key.val[0] += kPhiloxWeylSequence0;
  key.val[1] += kPhiloxWeylSequence1;
  return key;
}

static inline uint4 philox4x32_rounds(
    uint4 counter,
    uint2 key,
    unsigned int rounds) {
  if (rounds > 0)
    counter = philox4x32_single_round(counter, key);
  if (rounds > 1) {
    key = philox4x32_bump_key(key);
    counter = philox4x32_single_round(counter, key);
  }
  if (rounds > 2) {
    key = philox4x32_bump_key(key);
    counter = philox4x32_single_round(counter, key);
  }
  if (rounds > 3) {
    key = philox4x32_bump_key(key);
    counter = philox4x32_single_round(counter, key);
  }
  if (rounds > 4) {
    key = philox4x32_bump_key(key);
    counter = philox4x32_single_round(counter, key);
  }
  if (rounds > 5) {
    key = philox4x32_bump_key(key);
    counter = philox4x32_single_round(counter, key);
  }
  if (rounds > 6) {
    key = philox4x32_bump_key(key);
    counter = philox4x32_single_round(counter, key);
  }
  if (rounds > 7) {
    key = philox4x32_bump_key(key);
    counter = philox4x32_single_round(counter, key);
  }
  if (rounds > 8) {
    key = philox4x32_bump_key(key);
    counter = philox4x32_single_round(counter, key);
  }
  if (rounds > 9) {
    key = philox4x32_bump_key(key);
    counter = philox4x32_single_round(counter, key);
  }
  return counter;
}

static inline uint4 philox4x32_10(uint4 counter, uint2 key) {
  return philox4x32_rounds(counter, key, kPhilox4x32Rounds);
}

static inline void skipahead_sequence(
    unsigned long long n,
    randStatePhilox4_32_10_t* state) {
  philox_counter_incr_high(state, n);
  state->result = philox4x32_10(state->counter, state->key);
}

static inline void skipahead(
    unsigned long long n,
    randStatePhilox4_32_10_t* state) {
  state->index += (n & 3);
  n /= 4;
  if (state->index > 3) {
    n += 1;
    state->index -= 4;
  }
  philox_counter_incr(state, n);
  state->result = philox4x32_10(state->counter, state->key);
}

static inline void rand_init(
    unsigned long long seed,
    unsigned long long subsequence,
    unsigned long long offset,
    randStatePhilox4_32_10_t* state) {
  state->counter.x = 0;
  state->counter.y = 0;
  state->counter.z = 0;
  state->counter.w = 0;
  state->key.x = (unsigned int)seed;
  state->key.y = (unsigned int)(seed >> 32);
  state->index = 0;
  skipahead_sequence(subsequence, state);
  skipahead(offset, state);
}

static inline unsigned int rand(randStatePhilox4_32_10_t* state) {
  unsigned int ret;
  switch (state->index++) {
    default:
      ret = state->result.x;
      break;
    case 1:
      ret = state->result.y;
      break;
    case 2:
      ret = state->result.z;
      break;
    case 3:
      ret = state->result.w;
      break;
  }
  if (state->index == 4) {
    philox_counter_incr_single(state);
    state->result = philox4x32_10(state->counter, state->key);
    state->index = 0;
  }
  return ret;
}

static inline uint4 rand4(randStatePhilox4_32_10_t* state) {
  uint4 r;
  uint4 tmp = state->result;
  philox_counter_incr_single(state);
  state->result = philox4x32_10(state->counter, state->key);
  switch (state->index) {
    case 0:
      return tmp;
    case 1:
      r.x = tmp.y;
      r.y = tmp.z;
      r.z = tmp.w;
      r.w = state->result.x;
      break;
    case 2:
      r.x = tmp.z;
      r.y = tmp.w;
      r.z = state->result.x;
      r.w = state->result.y;
      break;
    case 3:
      r.x = tmp.w;
      r.y = state->result.x;
      r.z = state->result.y;
      r.w = state->result.z;
      break;
    default:
      // NOT possible but needed to avoid compiler warnings
      return tmp;
  }
  return r;
}

#define RAND_2POW32_INV (2.3283064e-10f)
#define RAND_2POW32_INV_2PI (2.3283064e-10f * 6.2831855f)
#define RAND_2POW53_INV_DOUBLE (1.1102230246251565e-16)
#define RAND_PI_DOUBLE (3.1415926535897932)

// =================== uniform ===================

static inline float _rand_uniform(unsigned int x) {
  return x * RAND_2POW32_INV + (RAND_2POW32_INV / 2.0f);
}

static inline float _rand_uniform(unsigned long long x) {
  unsigned int t;
  t = (unsigned int)(x >> 32);
  return t * RAND_2POW32_INV + (RAND_2POW32_INV / 2.0f);
}

static inline float rand_uniform(randStatePhilox4_32_10_t* state) {
  return _rand_uniform(rand(state));
}

static inline float4 rand_uniform4(randStatePhilox4_32_10_t* state) {
  auto x = rand4(state);
  float4 y;
  y.x = x.x * RAND_2POW32_INV + (RAND_2POW32_INV / 2.0f);
  y.y = x.y * RAND_2POW32_INV + (RAND_2POW32_INV / 2.0f);
  y.z = x.z * RAND_2POW32_INV + (RAND_2POW32_INV / 2.0f);
  y.w = x.w * RAND_2POW32_INV + (RAND_2POW32_INV / 2.0f);
  return y;
}

static inline double _rand_uniform_double_hq(unsigned int x, unsigned int y) {
  unsigned long long z =
      (unsigned long long)x ^ ((unsigned long long)y << (53 - 32));
  return z * RAND_2POW53_INV_DOUBLE + (RAND_2POW53_INV_DOUBLE / 2.0);
}

static inline double2 rand_uniform2_double(randStatePhilox4_32_10_t* state) {
  auto _x = rand4(state);
  double2 result;
  result.x = _rand_uniform_double_hq(_x.x, _x.y);
  result.y = _rand_uniform_double_hq(_x.z, _x.w);
  return result;
}

// =================== normal ===================

static inline float2 _rand_box_muller(unsigned int x, unsigned int y) {
  float2 result;
  float u = x * RAND_2POW32_INV + (RAND_2POW32_INV / 2);
  float v = y * RAND_2POW32_INV_2PI + (RAND_2POW32_INV_2PI / 2);
  float s = std::sqrt(-2.0f * std::log(u));
  result.x = std::sin(v);
  result.y = std::cos(v);
  result.x *= s;
  result.y *= s;
  return result;
}

template <typename R>
static inline float4 rand_box_muller4(R* state) {
  float4 result;
  float2 _result;
  auto x = rand4(state);
  _result = _rand_box_muller(x.x, x.y);
  result.x = _result.x;
  result.y = _result.y;
  _result = _rand_box_muller(x.z, x.w);
  result.z = _result.x;
  result.w = _result.y;
  return result;
}

static inline double2 _rand_box_muller_double(
    unsigned int x0,
    unsigned int x1,
    unsigned int y0,
    unsigned int y1) {
  double2 result;
  unsigned long long zx =
      (unsigned long long)x0 ^ ((unsigned long long)x1 << (53 - 32));
  double u = zx * RAND_2POW53_INV_DOUBLE + (RAND_2POW53_INV_DOUBLE / 2.0);
  unsigned long long zy =
      (unsigned long long)y0 ^ ((unsigned long long)y1 << (53 - 32));
  double v = zy * (RAND_2POW53_INV_DOUBLE * 2.0) + RAND_2POW53_INV_DOUBLE;
  double s = std::sqrt(-2.0 * std::log(u));

  result.x = std::sin(v * RAND_PI_DOUBLE);
  result.y = std::cos(v * RAND_PI_DOUBLE);
  result.x *= s;
  result.y *= s;

  return result;
}

static inline float rand_normal(randStatePhilox4_32_10_t* state) {
  if (state->boxmuller_flag != EXTRA_FLAG_NORMAL) {
    unsigned int x, y;
    x = rand(state);
    y = rand(state);
    float2 v = _rand_box_muller(x, y);
    state->boxmuller_extra = v.y;
    state->boxmuller_flag = EXTRA_FLAG_NORMAL;
    return v.x;
  }
  state->boxmuller_flag = 0;
  return state->boxmuller_extra;
}

static inline float4 rand_normal4(randStatePhilox4_32_10_t* state) {
  return rand_box_muller4(state);
}

static inline double2 rand_normal2_double(randStatePhilox4_32_10_t* state) {
  double2 result;
  auto x = rand4(state);
  result = _rand_box_muller_double(x.x, x.y, x.z, x.w);
  return result;
}

static inline double rand_normal_double(randStatePhilox4_32_10_t* state) {
  if (state->boxmuller_flag_double != EXTRA_FLAG_NORMAL) {
    uint4 _x;
    _x = rand4(state);
    double2 v = _rand_box_muller_double(_x.x, _x.y, _x.z, _x.w);
    state->boxmuller_extra = v.y;
    state->boxmuller_flag_double = EXTRA_FLAG_NORMAL;
    return v.x;
  }
  state->boxmuller_flag_double = 0;
  return state->boxmuller_extra;
}

// Compute log(Gamma(a)) for positive integer a
// Uses precomputed values for small a, Stirling series for large a
static inline double lgamma_integer(int a) {
  // Precomputed log((n-1)!) values for n = 0..9
  constexpr double kLgammaTable[] = {
      0.0, // a=0: undefined, but return 0
      0.0, // a=1: log(Gamma(1)) = log(0!) = 0
      0.0, // a=2: log(Gamma(2)) = log(1!) = 0
      0.6931471805599453, // a=3: log(Gamma(3)) = log(2!) = log(2)
      1.7917594692280550, // a=4: log(Gamma(4)) = log(3!) = log(6)
      3.1780538303479458, // a=5: log(Gamma(5)) = log(4!) = log(24)
      4.7874917427820458, // a=6: log(Gamma(6)) = log(5!) = log(120)
      6.5792512120101012, // a=7: log(Gamma(7)) = log(6!) = log(720)
      8.5251613610654147, // a=8: log(Gamma(8)) = log(7!) = log(5040)
      10.604602902745251 // a=9: log(Gamma(9)) = log(8!) = log(40320)
  };

  if (a <= 9) {
    return (a >= 0) ? kLgammaTable[a] : 0.0;
  }

  // Stirling's approximation for a > 9:
  // log(Gamma(x)) ≈ (x - 0.5)*log(x) - x + 0.5*log(2*pi) + 1/(12x) - 1/(360x^3)
  // + ...
  const double x = static_cast<double>(a);
  const double log_x = std::log(x);
  const double inv_x = 1.0 / x;
  const double inv_x2 = inv_x * inv_x;

  // Asymptotic series coefficients (Bernoulli numbers based)
  // B_2/(2*1) = 1/12, -B_4/(4*3) = -1/360, B_6/(6*5) = 1/1260, ...
  double correction = inv_x *
      (1.0 / 12.0 +
       inv_x2 *
           (-1.0 / 360.0 + inv_x2 * (1.0 / 1260.0 + inv_x2 * (-1.0 / 1680.0))));

  // log(sqrt(2*pi)) = 0.9189385332046727
  constexpr double kLogSqrt2Pi = 0.9189385332046727;

  return (x - 0.5) * log_x - x + kLogSqrt2Pi + correction;
}

/* Computes regularized gamma function:  gammainc(a,x)/gamma(a) */
static inline float pgammainc(float a, float x) {
  float t, alpha, beta;

  /* First level parametrization constants */
  float ma1 = 1.43248035075540910f, ma2 = 0.12400979329415655f,
        ma3 = 0.00025361074907033f, mb1 = 0.21096734870196546f,
        mb2 = 1.97381164089999420f, mb3 = 0.94201734077887530f;

  /* Second level parametrization constants (depends only on a) */

  alpha = 1 / sqrtf(a - ma2);
  alpha = ma1 * alpha + ma3;
  beta = 1 / sqrtf(a - mb2);
  beta = mb1 * beta + mb3;

  /* Final approximation (depends on a and x) */

  t = a - x;
  t = alpha * t - beta;
  t = 1.0f + expf(t);
  t = t * t;
  t = 1 / t;

  /* Negative a,x or a,x=NAN requires special handling */
  // t = !(x > 0 && a >= 0) ? 0.0 : t;
  return t;
}

/* Computes inverse of pgammainc */
static inline float pgammaincinv(float a, float y) {
  float t, alpha, beta;

  /* First level parametrization constants */

  float ma1 = 1.43248035075540910f, ma2 = 0.12400979329415655f,
        ma3 = 0.00025361074907033f, mb1 = 0.21096734870196546f,
        mb2 = 1.97381164089999420f, mb3 = 0.94201734077887530f;

  /* Second level parametrization constants (depends only on a) */

  alpha = 1.0f / sqrtf(a - ma2);
  alpha = ma1 * alpha + ma3;
  beta = 1.0f / sqrtf(a - mb2);
  beta = mb1 * beta + mb3;

  /* Final approximation (depends on a and y) */

  t = 1.0f / sqrtf(y) - 1.0f;
  t = logf(t);
  t = beta + t;
  t = -t * (1 / alpha) + a;
  /* Negative a,x or a,x=NAN requires special handling */
  // t = !(y > 0 && a >= 0) ? 0.0 : t;
  return t;
}

/* Rejection Method for Poisson distribution based on gammainc approximation */
static inline unsigned int rand_poisson_gammainc(
    randStatePhilox4_32_10_t* state,
    float lambda) {
  float y, x, t, z, v;
  float logl = logf(lambda);
  while (true) {
    y = rand_uniform(state);
    x = pgammaincinv(lambda, y);
    x = floorf(x);
    z = rand_uniform(state);
    v = (pgammainc(lambda, x + 1.0f) - pgammainc(lambda, x)) * 1.3f;
    z = z * v;
    t = (float)expf(
        -lambda + x * logl - (float)lgamma_integer((int)(1.0f + x)));
    if ((z < t) && (v >= 1e-20))
      break;
  }
  return (unsigned int)x;
}

// Donald E. Knuth Seminumerical Algorithms. The Art of Computer Programming,
// Volume 2
static inline unsigned int rand_poisson_knuth(
    randStatePhilox4_32_10_t* state,
    float lambda) {
  unsigned int k = 0;
  float p = expf(lambda);
  do {
    k++;
    p *= rand_uniform(state);
  } while (p > 1.0);
  return k - 1;
}

static inline unsigned int rand_poisson(
    randStatePhilox4_32_10_t* state,
    double lambda) {
  if (lambda < 64)
    return rand_poisson_knuth(state, (float)lambda);
  if (lambda > 4000)
    return (unsigned int)((std::sqrt(lambda) * rand_normal_double(state)) +
                          lambda + 0.5); // Round to nearest
  return rand_poisson_gammainc(state, (float)lambda);
}

} // namespace xpu
} // namespace native
} // namespace at
