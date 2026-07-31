// Verification cases for SYCL kernel vectorization pattern rule.
// All "BAD" cases should be flagged by the reviewer.
// All "GOOD" cases should pass without complaint.

#include <ATen/native/xpu/sycl/Loops.h>

using scalar_t = float;
constexpr int vec_size = 4;

// ============================================================
// CASE 1: Missing RESTRICT on pointer arguments
// Expected: FLAG
// ============================================================
template <typename scalar_t>
struct BadCase1MissingRestrict {
  void operator()(
      sycl::nd_item<1> item,
      scalar_t* output,  // Missing RESTRICT
      const scalar_t* input,  // Missing RESTRICT
      int n) const {
    using vec_t = at::native::memory::aligned_vector<scalar_t, vec_size>;
    int idx = item.get_global_linear_id();
    if (idx < n / vec_size) {
      const vec_t* input_vec = reinterpret_cast<const vec_t*>(input);
      vec_t* output_vec = reinterpret_cast<vec_t*>(output);
      vec_t data = input_vec[idx];
      for (int i = 0; i < vec_size; ++i) {
        data.val[i] = data.val[i] * 2.0f;
      }
      output_vec[idx] = data;
    }
  }
};

// ============================================================
// CASE 2: Raw pointer arithmetic instead of reinterpret_cast vec load
// Expected: FLAG
// ============================================================
template <typename scalar_t>
struct BadCase2RawPointerArithmetic {
  void operator()(
      sycl::nd_item<1> item,
      scalar_t* RESTRICT output,
      const scalar_t* RESTRICT input,
      int n) const {
    int idx = item.get_global_linear_id() * vec_size;
    if (idx + vec_size <= n) {
      // BAD: loading element-by-element instead of vectorized load
      for (int i = 0; i < vec_size; ++i) {
        output[idx + i] = input[idx + i] * 2.0f;
      }
    }
  }
};

// ============================================================
// CASE 3: Computation on raw pointer instead of loaded vec_t local
// Expected: FLAG
// ============================================================
template <typename scalar_t>
struct BadCase3ComputeOnPointer {
  void operator()(
      sycl::nd_item<1> item,
      scalar_t* RESTRICT output,
      const scalar_t* RESTRICT input,
      int n) const {
    using vec_t = at::native::memory::aligned_vector<scalar_t, vec_size>;
    int idx = item.get_global_linear_id();
    if (idx < n / vec_size) {
      const vec_t* input_vec = reinterpret_cast<const vec_t*>(input);
      vec_t* output_vec = reinterpret_cast<vec_t*>(output);
      // BAD: operating directly on pointer dereference instead of a local
      for (int i = 0; i < vec_size; ++i) {
        output_vec[idx].val[i] = input_vec[idx].val[i] * 2.0f;
      }
    }
  }
};

// ============================================================
// CASE 4: Missing aligned_vector type — using plain array
// Expected: FLAG
// ============================================================
template <typename scalar_t>
struct BadCase4PlainArrayLoad {
  void operator()(
      sycl::nd_item<1> item,
      scalar_t* RESTRICT output,
      const scalar_t* RESTRICT input,
      int n) const {
    int idx = item.get_global_linear_id() * vec_size;
    if (idx + vec_size <= n) {
      // BAD: using a plain array instead of aligned_vector
      scalar_t local[vec_size];
      for (int i = 0; i < vec_size; ++i) {
        local[i] = input[idx + i];
      }
      for (int i = 0; i < vec_size; ++i) {
        output[idx + i] = local[i] * 2.0f;
      }
    }
  }
};

// ============================================================
// CASE 5: Using sycl::vec instead of aligned_vector
// Expected: FLAG
// ============================================================
template <typename scalar_t>
struct BadCase5SyclVec {
  void operator()(
      sycl::nd_item<1> item,
      scalar_t* RESTRICT output,
      const scalar_t* RESTRICT input,
      int n) const {
    int idx = item.get_global_linear_id() * vec_size;
    if (idx + vec_size <= n) {
      // BAD: sycl::vec does not guarantee the aligned_vector pattern
      sycl::vec<scalar_t, vec_size> data;
      data.load(idx / vec_size, sycl::global_ptr<const scalar_t>(input));
      for (int i = 0; i < vec_size; ++i) {
        data[i] = data[i] * 2.0f;
      }
      data.store(idx / vec_size, sycl::global_ptr<scalar_t>(output));
    }
  }
};

// ============================================================
// CASE 6: Partial pattern — vec load but scalar store
// Expected: FLAG
// ============================================================
template <typename scalar_t>
struct BadCase6VecLoadScalarStore {
  void operator()(
      sycl::nd_item<1> item,
      scalar_t* RESTRICT output,
      const scalar_t* RESTRICT input,
      int n) const {
    using vec_t = at::native::memory::aligned_vector<scalar_t, vec_size>;
    int idx = item.get_global_linear_id();
    if (idx < n / vec_size) {
      const vec_t* input_vec = reinterpret_cast<const vec_t*>(input);
      vec_t data = input_vec[idx];
      // BAD: storing element-by-element instead of vectorized store
      for (int i = 0; i < vec_size; ++i) {
        output[idx * vec_size + i] = data.val[i] * 2.0f;
      }
    }
  }
};

// ============================================================
// GOOD CASE 1: Complete aligned_vector pattern
// Expected: PASS
// ============================================================
template <typename scalar_t>
struct GoodCase1Complete {
  void operator()(
      sycl::nd_item<1> item,
      scalar_t* RESTRICT output,
      const scalar_t* RESTRICT input,
      int n) const {
    using vec_t = at::native::memory::aligned_vector<scalar_t, vec_size>;
    int idx = item.get_global_linear_id();
    if (idx < n / vec_size) {
      const vec_t* RESTRICT input_vec =
          reinterpret_cast<const vec_t*>(input);
      vec_t* RESTRICT output_vec = reinterpret_cast<vec_t*>(output);
      vec_t data = input_vec[idx];
      for (int i = 0; i < vec_size; ++i) {
        data.val[i] = data.val[i] * 2.0f;
      }
      output_vec[idx] = data;
    }
  }
};

// ============================================================
// GOOD CASE 2: Multiple vec loads with independent computation
// Expected: PASS
// ============================================================
template <typename scalar_t>
struct GoodCase2MultipleVecLoads {
  void operator()(
      sycl::nd_item<1> item,
      scalar_t* RESTRICT output,
      const scalar_t* RESTRICT input_a,
      const scalar_t* RESTRICT input_b,
      int n) const {
    using vec_t = at::native::memory::aligned_vector<scalar_t, vec_size>;
    int idx = item.get_global_linear_id();
    if (idx < n / vec_size) {
      const vec_t* RESTRICT a_vec =
          reinterpret_cast<const vec_t*>(input_a);
      const vec_t* RESTRICT b_vec =
          reinterpret_cast<const vec_t*>(input_b);
      vec_t* RESTRICT out_vec = reinterpret_cast<vec_t*>(output);
      vec_t a = a_vec[idx];
      vec_t b = b_vec[idx];
      vec_t result;
      for (int i = 0; i < vec_size; ++i) {
        result.val[i] = a.val[i] + b.val[i];
      }
      out_vec[idx] = result;
    }
  }
};

// ============================================================
// GOOD CASE 3: Non-vectorized kernel (no vec pattern at all — no issue)
// Expected: PASS
// ============================================================
template <typename scalar_t>
struct GoodCase3ScalarKernel {
  void operator()(
      sycl::nd_item<1> item,
      scalar_t* RESTRICT output,
      const scalar_t* RESTRICT input,
      int n) const {
    int idx = item.get_global_linear_id();
    if (idx < n) {
      output[idx] = input[idx] * 2.0f;
    }
  }
};

// ============================================================
// GOOD CASE 4: Tail handling with scalar fallback (acceptable)
// Expected: PASS
// ============================================================
template <typename scalar_t>
struct GoodCase4TailHandling {
  void operator()(
      sycl::nd_item<1> item,
      scalar_t* RESTRICT output,
      const scalar_t* RESTRICT input,
      int n) const {
    using vec_t = at::native::memory::aligned_vector<scalar_t, vec_size>;
    int idx = item.get_global_linear_id();
    int vec_n = n / vec_size;
    if (idx < vec_n) {
      const vec_t* RESTRICT input_vec =
          reinterpret_cast<const vec_t*>(input);
      vec_t* RESTRICT output_vec = reinterpret_cast<vec_t*>(output);
      vec_t data = input_vec[idx];
      for (int i = 0; i < vec_size; ++i) {
        data.val[i] = data.val[i] * 2.0f;
      }
      output_vec[idx] = data;
    }
    // Scalar tail — acceptable for remainder elements
    int tail_start = vec_n * vec_size;
    int tail_idx = tail_start + item.get_global_linear_id();
    if (tail_idx < n && item.get_global_linear_id() < (n - tail_start)) {
      output[tail_idx] = input[tail_idx] * 2.0f;
    }
  }
};
