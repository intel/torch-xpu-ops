// Verification cases for subgroup reduction anti-pattern rule.
// All "BAD" cases should be flagged by the reviewer.
// All "GOOD" cases should pass without complaint.

#include <sycl/sycl.hpp>

using arg_t = float;

// ============================================================
// CASE 1: Direct inline loop — ascending offset (classic)
// Expected: FLAG
// ============================================================
void bad_case_1_inline_ascending(sycl::sub_group sg, arg_t* value, int sg_size) {
  for (int offset = 1; offset < sg_size; offset <<= 1) {
    arg_t other = sycl::shift_group_left(sg, value[0], offset);
    value[0] = value[0] + other;
  }
}

// ============================================================
// CASE 2: Direct inline loop — descending offset
// Expected: FLAG
// ============================================================
void bad_case_2_inline_descending(sycl::sub_group sg, arg_t* value, int sg_size) {
  for (int offset = (sg_size >> 1); offset > 0; offset >>= 1) {
    arg_t other = sycl::shift_group_left(sg, value[0], offset);
    value[0] = value[0] + other;
  }
}

// ============================================================
// CASE 3: Vectorized inner loop
// Expected: FLAG
// ============================================================
void bad_case_3_vectorized(sycl::sub_group sg, arg_t* value, int sg_size, int vec_sz) {
  for (int offset = 1; offset < sg_size; offset <<= 1) {
    for (int i = 0; i < vec_sz; ++i) {
      arg_t other = sycl::shift_group_left(sg, value[i], offset);
      value[i] = value[i] + other;
    }
  }
}

// ============================================================
// CASE 4: Using min as the combiner
// Expected: FLAG
// ============================================================
void bad_case_4_min(sycl::sub_group sg, arg_t* value, int sg_size) {
  for (int offset = 1; offset < sg_size; offset <<= 1) {
    arg_t other = sycl::shift_group_left(sg, value[0], offset);
    value[0] = (value[0] < other) ? value[0] : other;
  }
}

// ============================================================
// CASE 5: Using max as the combiner
// Expected: FLAG
// ============================================================
void bad_case_5_max(sycl::sub_group sg, arg_t* value, int sg_size) {
  for (int offset = 1; offset < sg_size; offset <<= 1) {
    arg_t other = sycl::shift_group_left(sg, value[0], offset);
    value[0] = (value[0] > other) ? value[0] : other;
  }
}

// ============================================================
// CASE 6: Mean reduction (sum then divide)
// Expected: FLAG
// ============================================================
void bad_case_6_mean(sycl::sub_group sg, arg_t* value, int sg_size) {
  for (int offset = 1; offset < sg_size; offset <<= 1) {
    arg_t other = sycl::shift_group_left(sg, value[0], offset);
    value[0] = value[0] + other;
  }
  value[0] = value[0] / static_cast<arg_t>(sg_size);
}

// ============================================================
// CASE 7: Product reduction
// Expected: FLAG
// ============================================================
void bad_case_7_product(sycl::sub_group sg, arg_t* value, int sg_size) {
  for (int offset = 1; offset < sg_size; offset <<= 1) {
    arg_t other = sycl::shift_group_left(sg, value[0], offset);
    value[0] = value[0] * other;
  }
}

// ============================================================
// CASE 8: Bitwise-OR reduction (integer case)
// Expected: FLAG
// ============================================================
void bad_case_8_bitwise_or(sycl::sub_group sg, int* value, int sg_size) {
  for (int offset = 1; offset < sg_size; offset <<= 1) {
    int other = sycl::shift_group_left(sg, value[0], offset);
    value[0] = value[0] | other;
  }
}

// ============================================================
// CASE 9: Functor with shift+combine, loop at call site
// Expected: FLAG (both the functor definition and the call site)
// ============================================================
template <typename BinaryOp>
struct ShiftAndCombineFunctor {
  arg_t operator()(sycl::sub_group sg, arg_t val, int offset) const {
    arg_t other = sycl::shift_group_left(sg, val, offset);
    return op_(val, other);
  }
  BinaryOp op_;
};

struct AddOp {
  arg_t operator()(arg_t a, arg_t b) const { return a + b; }
};

void bad_case_9_functor_caller(sycl::sub_group sg, arg_t* value, int sg_size) {
  ShiftAndCombineFunctor<AddOp> func{AddOp{}};
  for (int offset = 1; offset < sg_size; offset <<= 1) {
    value[0] = func(sg, value[0], offset);
  }
}

// ============================================================
// CASE 10: Lambda with shift+combine, loop at call site
// Expected: FLAG
// ============================================================
void bad_case_10_lambda_caller(sycl::sub_group sg, arg_t* value, int sg_size) {
  auto shift_and_add = [](sycl::sub_group sg, arg_t val, int offset) {
    arg_t other = sycl::shift_group_left(sg, val, offset);
    return val + other;
  };
  for (int offset = 1; offset < sg_size; offset <<= 1) {
    value[0] = shift_and_add(sg, value[0], offset);
  }
}

// ============================================================
// CASE 11: Template function wrapping the shift+combine
// Expected: FLAG
// ============================================================
template <typename T, typename BinaryOp>
T shift_and_reduce_step(sycl::sub_group sg, T val, int offset, BinaryOp op) {
  T other = sycl::shift_group_left(sg, val, offset);
  return op(val, other);
}

void bad_case_11_template_helper(sycl::sub_group sg, arg_t* value, int sg_size) {
  for (int offset = 1; offset < sg_size; offset <<= 1) {
    value[0] = shift_and_reduce_step(sg, value[0], offset, AddOp{});
  }
}

// ============================================================
// CASE 12: Combine via std::plus (recognizable reduction op)
// Expected: FLAG
// ============================================================
void bad_case_12_std_plus(sycl::sub_group sg, arg_t* value, int sg_size) {
  std::plus<arg_t> combine;
  for (int offset = 1; offset < sg_size; offset <<= 1) {
    arg_t other = sycl::shift_group_left(sg, value[0], offset);
    value[0] = combine(value[0], other);
  }
}

// ============================================================
// CASE 13: While-loop instead of for-loop
// Expected: FLAG
// ============================================================
void bad_case_13_while_loop(sycl::sub_group sg, arg_t* value, int sg_size) {
  int offset = 1;
  while (offset < sg_size) {
    arg_t other = sycl::shift_group_left(sg, value[0], offset);
    value[0] = value[0] + other;
    offset <<= 1;
  }
}

// ============================================================
// CASE 14: Unrolled shifts (no explicit loop, but same pattern)
// Expected: FLAG
// ============================================================
void bad_case_14_unrolled(sycl::sub_group sg, arg_t* value) {
  arg_t tmp;
  tmp = sycl::shift_group_left(sg, value[0], 1);
  value[0] += tmp;
  tmp = sycl::shift_group_left(sg, value[0], 2);
  value[0] += tmp;
  tmp = sycl::shift_group_left(sg, value[0], 4);
  value[0] += tmp;
  tmp = sycl::shift_group_left(sg, value[0], 8);
  value[0] += tmp;
  tmp = sycl::shift_group_left(sg, value[0], 16);
  value[0] += tmp;
}

// ============================================================
// GOOD CASE 1: reduce_over_group for sum
// Expected: PASS
// ============================================================
void good_case_1_reduce_sum(sycl::sub_group sg, arg_t* value) {
  value[0] = sycl::reduce_over_group(sg, value[0], sycl::plus<arg_t>());
}

// ============================================================
// GOOD CASE 2: reduce_over_group for min
// Expected: PASS
// ============================================================
void good_case_2_reduce_min(sycl::sub_group sg, arg_t* value) {
  value[0] = sycl::reduce_over_group(sg, value[0], sycl::minimum<arg_t>());
}

// ============================================================
// GOOD CASE 3: reduce_over_group for max
// Expected: PASS
// ============================================================
void good_case_3_reduce_max(sycl::sub_group sg, arg_t* value) {
  value[0] = sycl::reduce_over_group(sg, value[0], sycl::maximum<arg_t>());
}

// ============================================================
// GOOD CASE 4: Single shift_group_left for neighbor access (not a reduction)
// Expected: PASS
// ============================================================
void good_case_4_neighbor_access(sycl::sub_group sg, arg_t* value) {
  arg_t neighbor = sycl::shift_group_left(sg, value[0], 1);
  value[0] = value[0] - neighbor;  // difference with neighbor, not a reduction
}

// ============================================================
// GOOD CASE 5: shift_group_left for exclusive scan (not a reduction)
// Expected: PASS
// ============================================================
void good_case_5_scan(sycl::sub_group sg, arg_t* value, int sg_size) {
  // Inclusive prefix sum using shift — this is a SCAN, not a reduction.
  // Each work-item retains a distinct partial result.
  for (int offset = 1; offset < sg_size; offset <<= 1) {
    arg_t other = sycl::shift_group_left(sg, value[0], offset);
    if (sg.get_local_linear_id() >= (unsigned)offset) {
      value[0] = value[0] + other;
    }
  }
}

// ============================================================
// GOOD CASE 6: shift_group_left used for data shuffle (no combine)
// Expected: PASS
// ============================================================
void good_case_6_shuffle_only(sycl::sub_group sg, arg_t* value) {
  // Just moving data around, no reduction
  value[0] = sycl::shift_group_left(sg, value[0], 4);
}
