#pragma clang diagnostic push
#pragma GCC diagnostic push
// Avoid SYCL compiler return-type error
#pragma clang diagnostic ignored "-Wreturn-type"
#pragma GCC diagnostic ignored "-Wreturn-type"

#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/Atomics.h>
#include <ATen/native/xpu/sycl/LossCTCKernels.h>
#include <comm/SYCLContext.h>

namespace at::native::xpu {

// this ad-hoc converts from targets (l in [1]) to augmented targets (l' in [1])
// so if l is l_0 l_1 ... l_(tl-1) then this looks up idx in
// l' = BLANK l_0 BLANK l_1 BLANK ... BLANK l_(tl-1) BLANK
// - note that no bound-checking is done
// - it is important to only call it with idx == 0 if the target length is 0
template <typename target_t>
static inline int64_t get_target_prime(
    const target_t* RESTRICT target,
    int64_t offset,
    int64_t stride,
    int64_t idx,
    int64_t BLANK) {
  if (idx % 2 == 0) {
    return BLANK;
  } else {
    return target[offset + stride * (idx / 2)];
  }
}

template <typename scalar_t, typename target_t>
struct CTCLossLogAlphaKernelFunctor {
  void operator()(sycl::nd_item<2> item) const {
    constexpr scalar_t neginf = -INFINITY;

    auto tid_x = item.get_local_id(1);
    auto tid_y = item.get_local_id(0);

    // bookkeeping
    int64_t b = tid_y + item.get_group(0) * item.get_local_range(0);
    int64_t input_length = input_lengths_[b];
    int64_t target_length = target_lengths_[b];
    int64_t lp_batch_offset = b * lp_batch_stride_;
    int64_t la_batch_offset = b * la_batch_stride_;
    int64_t tg_batch_offset = tg_batch_offsets_[b];

    bool valid = true;
    if (b >= batch_size_)
      valid = false;

    // Waiting for support for activeThreadsOnlyBarrier
    if (input_length == 0) {
      if (tid_x == 0) {
        scalar_t log_likelihood = target_length == 0 ? 0 : neginf;
        neg_log_likelihood_data_[b] = -log_likelihood;
      }
      valid = false;
    }

    if (valid) {
      // first row (t=0), the three equations for alpha_1 above eq (6)
      for (int64_t block_s = 0; block_s < 2 * max_target_length_ + 1;
           block_s += item.get_local_range(1)) {
        int64_t s = tid_x + block_s;
        scalar_t la;
        switch (s) {
          case 0:
            la = log_probs_data_[lp_batch_offset + lp_char_stride_ * BLANK_];
            break;
          case 1:
            la = target_length == 0 ? neginf
                                    : log_probs_data_
                                          [lp_batch_offset +
                                           lp_char_stride_ *
                                               get_target_prime(
                                                   targets_data_,
                                                   tg_batch_offset,
                                                   tg_target_stride_,
                                                   1,
                                                   BLANK_)];
            break;
          default:
            la = neginf;
        }
        if (s < 2 * max_target_length_ + 1)
          log_alpha_data_
              [la_batch_offset +
               /* la_input_stride_ * 0 */ +la_target_stride_ * s] = la;
      }
    }

    for (int64_t block_s = 0; block_s < 2 * max_target_length_ + 1;
         block_s += item.get_local_range(1)) {
      int64_t s = tid_x + block_s;

      // These two only depend on s, so we can cache them.
      int64_t current_char; // l_s in eq (6)
      bool have_three; // flag which of the two cases in eq (6) we have
      if (valid && s < 2 * target_length + 1 && target_length > 0) {
        current_char = get_target_prime(
            targets_data_, tg_batch_offset, tg_target_stride_, s, BLANK_);
        have_three =
            ((s > 1) &&
             (get_target_prime(
                  targets_data_,
                  tg_batch_offset,
                  tg_target_stride_,
                  s - 2,
                  BLANK_) != current_char));
      } else {
        current_char = BLANK_;
        have_three = false;
      }
      for (int64_t t = 1; t < max_input_length_; t++) {
        item.barrier(sycl_local_fence);
        if (valid && (t < input_length) && (s < 2 * target_length + 1)) {
          // only for valid t, s. This is equation (6) and (7), la1, la2, la3
          // are the three summands, lamax is the maximum for the logsumexp
          // trick.
          scalar_t la1 = log_alpha_data_
              [la_batch_offset + la_input_stride_ * (t - 1) +
               la_target_stride_ * s];
          scalar_t lamax = la1;
          scalar_t la2, la3;
          if (s > 0) {
            la2 = log_alpha_data_
                [la_batch_offset + la_input_stride_ * (t - 1) +
                 la_target_stride_ * (s - 1)];
            if (la2 > lamax)
              lamax = la2;
          } else {
            la2 = neginf;
          }
          if (have_three) {
            la3 = log_alpha_data_
                [la_batch_offset + la_input_stride_ * (t - 1) +
                 la_target_stride_ * (s - 2)];
            if (la3 > lamax)
              lamax = la3;
          } else {
            la3 = neginf;
          }
          if (lamax == neginf) // when all are neginf. (then the whole thing is
                               // neginf, but we can pretend)
            lamax = 0;

          log_alpha_data_
              [la_batch_offset + la_input_stride_ * t + la_target_stride_ * s] =
                  std::log(
                      std::exp(la1 - lamax) + std::exp(la2 - lamax) +
                      std::exp(la3 - lamax)) +
              lamax +
              log_probs_data_
                  [lp_batch_offset + t * lp_input_stride_ +
                   lp_char_stride_ * current_char];
        } else {
          // otherwise we just set to neginf
          if (valid && s < 2 * max_target_length_ + 1)
            log_alpha_data_
                [la_batch_offset + la_input_stride_ * t +
                 la_target_stride_ * s] = neginf;
        }
      }
    }
    item.barrier(sycl_local_fence);

    if (!valid)
      return;

    // compute the loss (eq (8))
    if (tid_x == 0) {
      scalar_t l1 = log_alpha_data_
          [la_batch_offset + la_input_stride_ * (input_length - 1) +
           la_target_stride_ * (target_length * 2)];
      scalar_t l2 = target_length > 0
          ? log_alpha_data_
                [la_batch_offset + la_input_stride_ * (input_length - 1) +
                 la_target_stride_ * (target_length * 2 - 1)]
          : neginf;
      scalar_t m = ((l1 > l2) ? l1 : l2);
      m = ((m == neginf) ? 0 : m);
      scalar_t log_likelihood =
          std::log(std::exp(l1 - m) + std::exp(l2 - m)) + m;
      neg_log_likelihood_data_[b] = -log_likelihood;
    }
  }

  CTCLossLogAlphaKernelFunctor(
      scalar_t* RESTRICT log_alpha_data,
      const scalar_t* log_probs_data,
      const int64_t* RESTRICT input_lengths,
      int64_t max_input_length,
      const target_t* RESTRICT targets_data,
      const int64_t* RESTRICT target_lengths,
      int64_t max_target_length,
      scalar_t* RESTRICT neg_log_likelihood_data,
      int64_t lp_input_stride,
      int64_t lp_batch_stride,
      int64_t lp_char_stride,
      int64_t la_batch_stride,
      int64_t la_input_stride,
      int64_t la_target_stride,
      const int64_t* RESTRICT tg_batch_offsets,
      int64_t tg_target_stride,
      int64_t batch_size,
      int64_t BLANK)
      : log_alpha_data_(log_alpha_data),
        log_probs_data_(log_probs_data),
        input_lengths_(input_lengths),
        max_input_length_(max_input_length),
        targets_data_(targets_data),
        target_lengths_(target_lengths),
        max_target_length_(max_target_length),
        neg_log_likelihood_data_(neg_log_likelihood_data),
        lp_input_stride_(lp_input_stride),
        lp_batch_stride_(lp_batch_stride),
        lp_char_stride_(lp_char_stride),
        la_batch_stride_(la_batch_stride),
        la_input_stride_(la_input_stride),
        la_target_stride_(la_target_stride),
        tg_batch_offsets_(tg_batch_offsets),
        tg_target_stride_(tg_target_stride),
        batch_size_(batch_size),
        BLANK_(BLANK) {}

 private:
  scalar_t* RESTRICT log_alpha_data_;
  const scalar_t* log_probs_data_;
  const int64_t* RESTRICT input_lengths_;
  int64_t max_input_length_;
  const target_t* RESTRICT targets_data_;
  const int64_t* RESTRICT target_lengths_;
  int64_t max_target_length_;
  scalar_t* RESTRICT neg_log_likelihood_data_;
  int64_t lp_input_stride_;
  int64_t lp_batch_stride_;
  int64_t lp_char_stride_;
  int64_t la_batch_stride_;
  int64_t la_input_stride_;
  int64_t la_target_stride_;
  const int64_t* RESTRICT tg_batch_offsets_;
  int64_t tg_target_stride_;
  int64_t batch_size_;
  int64_t BLANK_;
};

// The forward computation. Lot's of admin and a call to the alpha kernel.
// Note: we do not check that the labels are in the valid range. As we use
// them for indexing in the kernels, you'll see memory errors when you
// pass corrupt labels.
// We support both a 2-dimensional tensor as targets (one set of targets in each
// row) and a 1-dimensional tensor where all targets are concatenated (and we
// use target_lengths to figure out where they begin). We return log_alpha
// (currently, might change to (log_alpha+log_beta) to be passed to the
// backward. The dispatch function will only return the loss.
template <typename scalar_t, ScalarType target_scalar_type>
std::tuple<Tensor, Tensor> ctc_loss_kernel_template(
    const Tensor& log_probs,
    const Tensor& targets,
    IntArrayRef input_lengths,
    IntArrayRef target_lengths,
    int64_t BLANK) {
  // log_probs: input_len x batch_size x num_labels
  // targets [int64]: batch_size x target_length OR sum(target_lengths)
  CheckedFrom c = "ctc_loss_kernel";
  using target_t =
      typename std::conditional<target_scalar_type == kInt, int, int64_t>::type;
  auto log_probs_arg = TensorArg(log_probs, "log_probs", 1);
  auto targets_arg = TensorArg(targets, "targets", 2);
  checkAllSameGPU(c, {log_probs_arg, targets_arg});

  checkScalarType(c, targets_arg, target_scalar_type);
  checkDim(c, log_probs_arg, 3);
  checkDimRange(c, targets_arg, 1, 3);

  int64_t batch_size = log_probs.size(1);
  int64_t num_labels = log_probs.size(2);
  TORCH_CHECK(
      (0 <= BLANK) && (BLANK < num_labels), "blank must be in label range");
  TORCH_CHECK(
      input_lengths.size() == static_cast<size_t>(batch_size),
      "input_lengths must be of size batch_size");
  TORCH_CHECK(
      target_lengths.size() == static_cast<size_t>(batch_size),
      "target_lengths must be of size batch_size");

  int64_t tg_target_stride;

  int64_t max_target_length = 0;
  auto tg_batch_offsets =
      at::empty({batch_size}, at::device(at::kCPU).dtype(at::kLong));
  auto tg_batch_offsets_data = tg_batch_offsets.mutable_data_ptr<int64_t>();
  if (targets.dim() == 1) { // concatenated targets
    int64_t pos = 0;
    for (int64_t i = 0; i < batch_size; i++) {
      TORCH_CHECK(
          target_lengths[i] >= 0,
          "Expected target_lengths to have value at least ",
          0,
          ", but got value ",
          target_lengths[i],
          " (while checking arguments for ",
          c,
          ")");
      tg_batch_offsets_data[i] = pos;
      pos += target_lengths[i];
      if (max_target_length < target_lengths[i])
        max_target_length = target_lengths[i];
    }
    tg_target_stride = targets.stride(0);
    checkSize(c, targets_arg, 0, pos);
  } else { // batch x max_target_length
    // dim is 2
    int64_t tg_batch_stride = targets.stride(0);
    for (int64_t i = 0; i < batch_size; i++) {
      TORCH_CHECK(
          target_lengths[i] >= 0,
          "Expected target_lengths to have value at least ",
          0,
          ", but got value ",
          target_lengths[i],
          " (while checking arguments for ",
          c,
          ")");
      tg_batch_offsets_data[i] = i * tg_batch_stride;
      if (max_target_length < target_lengths[i])
        max_target_length = target_lengths[i];
    }
    tg_target_stride = targets.stride(1);
    checkSize(c, targets_arg, 0, batch_size);
    TORCH_CHECK(
        targets.size(1) >= max_target_length,
        "Expected tensor to have size at least ",
        max_target_length,
        " at dimension 1, but got size ",
        targets.size(1),
        " for ",
        targets_arg,
        " (while checking arguments for ",
        c,
        ")");
  }
  int64_t max_input_length = log_probs.size(0);
  for (int64_t b = 0; b < batch_size; b++) {
    TORCH_CHECK(
        input_lengths[b] >= 0,
        "Expected input_lengths to have value at least ",
        0,
        ", but got value ",
        input_lengths[b],
        " (while checking arguments for ",
        c,
        ")");
    TORCH_CHECK(
        input_lengths[b] <= max_input_length,
        "Expected input_lengths to have value at most ",
        max_input_length,
        ", but got value ",
        input_lengths[b],
        " (while checking arguments for ",
        c,
        ")");
  }

  auto target_lengths_t =
      at::tensor(target_lengths, targets.options().dtype(kLong));
  auto input_lengths_t =
      at::tensor(input_lengths, targets.options().dtype(kLong));
  tg_batch_offsets = tg_batch_offsets.to(targets.device());

  Tensor log_alpha = at::empty(
      {batch_size, log_probs.size(0), 2 * max_target_length + 1},
      log_probs.options());
  Tensor neg_log_likelihood = at::empty({batch_size}, log_probs.options());

  using CTCLossLogAlphaKernel =
      CTCLossLogAlphaKernelFunctor<scalar_t, target_t>;
  int max_threads = syclMaxWorkGroupSize<CTCLossLogAlphaKernel>();

  int threads_target = max_threads;
  while (threads_target / 2 >= 2 * max_target_length + 1) {
    threads_target /= 2;
  }
  int threads_batch = std::min(max_threads / threads_target, (int)batch_size);
  size_t group_size_x = threads_target;
  size_t group_size_y = threads_batch;
  size_t ngroups_x = 1;
  size_t ngroups_y = (batch_size + threads_batch - 1) / threads_batch;

  sycl::range<2> local_range{group_size_y, group_size_x};
  sycl::range<2> global_range{
      ngroups_y * group_size_y, ngroups_x * group_size_x};

  auto caller = CTCLossLogAlphaKernel(
      log_alpha.mutable_data_ptr<scalar_t>(),
      log_probs.const_data_ptr<scalar_t>(),
      input_lengths_t.const_data_ptr<int64_t>(),
      log_probs.size(0),
      targets.const_data_ptr<target_t>(),
      target_lengths_t.const_data_ptr<int64_t>(),
      max_target_length,
      neg_log_likelihood.mutable_data_ptr<scalar_t>(),
      log_probs.stride(0),
      log_probs.stride(1),
      log_probs.stride(2),
      log_alpha.stride(0),
      log_alpha.stride(1),
      log_alpha.stride(2),
      tg_batch_offsets.const_data_ptr<int64_t>(),
      tg_target_stride,
      batch_size,
      BLANK);
  sycl_kernel_submit(
      global_range, local_range, at::xpu::getCurrentSYCLQueue(), caller);

  return std::make_tuple(neg_log_likelihood, log_alpha);
}

template <typename scalar_t, typename target_t>
struct CTCLossBackwardLogBetaKernelFunctor {
  void operator()(sycl::nd_item<2> item) const {
    constexpr scalar_t neginf = -INFINITY;

    auto tid_x = item.get_local_id(1);
    auto tid_y = item.get_local_id(0);
    auto group_size_x = item.get_local_range(1);
    auto group_size_y = item.get_local_range(0);

    int64_t b = tid_y + item.get_group(0) * group_size_y;

    int64_t input_length = input_lengths_[b];
    int64_t target_length = target_lengths_[b];
    int64_t lp_batch_offset = b * lp_batch_stride_;
    int64_t lb_batch_offset = b * lb_batch_stride_;
    int64_t tg_batch_offset = tg_batch_offsets_[b];

    bool valid = true;

    if (b >= batch_size_)
      valid = false;

    if (input_length == 0)
      valid = false;

    if (valid) {
      // "first" row, the beta initialization before eq (10) (t=target_length -
      // differes per batch)
      for (int64_t block_s =
               2 * max_target_length_ - (2 * max_target_length_ % group_size_x);
           block_s >= 0;
           block_s -= group_size_x) {
        int64_t s = tid_x + block_s;
        scalar_t lb;
        if (s == 2 * target_length) {
          lb = log_probs_data_
              [lp_batch_offset + (input_length - 1) * lp_input_stride_ +
               lp_char_stride_ * BLANK_];
        } else if (s == 2 * target_length - 1) { // false for target_length == 0
          int64_t current_target_prime = get_target_prime(
              targets_data_, tg_batch_offset, tg_target_stride_, s, BLANK_);
          lb = log_probs_data_
              [lp_batch_offset + (input_length - 1) * lp_input_stride_ +
               lp_char_stride_ * current_target_prime];
        } else {
          lb = neginf;
        }
        if (s < 2 * max_target_length_ + 1) {
          log_beta_data_
              [lb_batch_offset + (input_length - 1) * lb_input_stride_ +
               lb_target_stride_ * s] = lb;
        }
      }
    }

    // go backward in s
    for (int64_t block_s =
             2 * max_target_length_ - (2 * max_target_length_ % group_size_x);
         block_s >= 0;
         block_s -= group_size_x) {
      int64_t s = tid_x + block_s;
      int64_t current_target_prime;
      bool have_three;
      if (valid && s < 2 * target_length + 1 && target_length > 0) {
        current_target_prime = get_target_prime(
            targets_data_, tg_batch_offset, tg_target_stride_, s, BLANK_);
        have_three =
            ((s < 2 * target_length - 1) &&
             (get_target_prime(
                  targets_data_,
                  tg_batch_offset,
                  tg_target_stride_,
                  s + 2,
                  BLANK_) != current_target_prime));
      } else {
        current_target_prime = BLANK_;
        have_three = false;
      }
      // now go backward in t. Note that we need to skip the last timestep that
      // we did above.
      for (int64_t t = max_input_length_ - 2; t >= 0; t--) {
        item.barrier(sycl_local_fence);
        if (valid && (t < input_length - 1) && (s < 2 * target_length + 1)) {
          scalar_t lb1 = log_beta_data_
              [lb_batch_offset + lb_input_stride_ * (t + 1) +
               lb_target_stride_ * s];
          scalar_t lbmax = lb1;
          scalar_t lb2, lb3;

          if (s < 2 * target_length) {
            lb2 = log_beta_data_
                [lb_batch_offset + lb_input_stride_ * (t + 1) +
                 lb_target_stride_ * (s + 1)];
            if (lb2 > lbmax)
              lbmax = lb2;
          } else {
            lb2 = neginf;
          }
          if (have_three) {
            lb3 = log_beta_data_
                [lb_batch_offset + lb_input_stride_ * (t + 1) +
                 lb_target_stride_ * (s + 2)];
            if (lb3 > lbmax)
              lbmax = lb3;
          } else {
            lb3 = neginf;
          }
          if (lbmax == neginf)
            lbmax = 0;

          scalar_t lb = std::log(
                            std::exp(lb1 - lbmax) + std::exp(lb2 - lbmax) +
                            std::exp(lb3 - lbmax)) +
              lbmax +
              log_probs_data_
                  [lp_batch_offset + t * lp_input_stride_ +
                   lp_char_stride_ * current_target_prime];

          log_beta_data_
              [lb_batch_offset + lb_input_stride_ * t + lb_target_stride_ * s] =
                  lb;
        } else if (
            (b < batch_size_) && (s < 2 * max_target_length_ + 1) &&
            (((target_length == 0) && (s > 0)) ||
             (s >= 2 * target_length + 1) || (t >= input_length))) {
          log_beta_data_
              [lb_batch_offset + lb_input_stride_ * t + lb_target_stride_ * s] =
                  neginf;
        }
      }
    }
  }

  CTCLossBackwardLogBetaKernelFunctor(
      scalar_t* RESTRICT log_beta_data,
      const scalar_t* log_probs_data,
      const int64_t* RESTRICT input_lengths,
      int64_t max_input_length,
      const target_t* RESTRICT targets_data,
      const int64_t* RESTRICT target_lengths,
      int64_t max_target_length,
      int64_t lp_input_stride,
      int64_t lp_batch_stride,
      int64_t lp_char_stride,
      int64_t lb_batch_stride,
      int64_t lb_input_stride,
      int64_t lb_target_stride,
      const int64_t* RESTRICT tg_batch_offsets,
      int64_t tg_target_stride,
      int64_t batch_size,
      int64_t BLANK)
      : log_beta_data_(log_beta_data),
        log_probs_data_(log_probs_data),
        input_lengths_(input_lengths),
        max_input_length_(max_input_length),
        targets_data_(targets_data),
        target_lengths_(target_lengths),
        max_target_length_(max_target_length),
        lp_input_stride_(lp_input_stride),
        lp_batch_stride_(lp_batch_stride),
        lp_char_stride_(lp_char_stride),
        lb_batch_stride_(lb_batch_stride),
        lb_input_stride_(lb_input_stride),
        lb_target_stride_(lb_target_stride),
        tg_batch_offsets_(tg_batch_offsets),
        tg_target_stride_(tg_target_stride),
        batch_size_(batch_size),
        BLANK_(BLANK) {}

 private:
  scalar_t* RESTRICT log_beta_data_;
  const scalar_t* log_probs_data_;
  const int64_t* RESTRICT input_lengths_;
  int64_t max_input_length_;
  const target_t* RESTRICT targets_data_;
  const int64_t* RESTRICT target_lengths_;
  int64_t max_target_length_;
  int64_t lp_input_stride_;
  int64_t lp_batch_stride_;
  int64_t lp_char_stride_;
  int64_t lb_batch_stride_;
  int64_t lb_input_stride_;
  int64_t lb_target_stride_;
  const int64_t* RESTRICT tg_batch_offsets_;
  int64_t tg_target_stride_;
  int64_t batch_size_;
  int64_t BLANK_;
};

template <typename scalar_t, typename target_t>
struct CTCLossBackwardCollectNonblankKernelFunctor {
  void operator()(sycl::nd_item<2> item) const {
    int64_t b =
        item.get_local_id(0) + item.get_group(0) * item.get_local_range(0);
    int64_t s = item.get_local_id(1) +
        item.get_group(1) *
            item.get_local_range(1); // note, this directly indexes into
                                     // targets, not targets prime!

    if (b >= batch_size_)
      return;

    int64_t input_length = input_lengths_[b];
    int64_t target_length = target_lengths_[b];
    int64_t gr_batch_offset = b * gr_batch_stride_;
    int64_t lp_batch_offset = b * lp_batch_stride_;
    int64_t la_batch_offset = b * la_batch_stride_;
    int64_t lb_batch_offset = b * lb_batch_stride_;
    int64_t tg_batch_offset = tg_batch_offsets_[b];

    if (s >= target_length)
      return;

    int64_t target = targets_data_[tg_batch_offset + s * tg_target_stride_];
    scalar_t nll = neg_log_likelihood_data_[b];
    scalar_t gr = grad_out_data_[b * grad_out_batch_stride_];

    if (zero_infinity_ && nll == INFINITY)
      return;

    for (int64_t t = 0; t < input_length; t++) {
      scalar_t lp = log_probs_data_
          [lp_batch_offset + t * lp_input_stride_ + lp_char_stride_ * target];
      atomicAdd(
          &gradient_data_
              [gr_batch_offset + t * gr_input_stride_ +
               gr_char_stride_ * target],
          -std::exp(
              log_alpha_data_
                  [la_batch_offset + la_input_stride_ * t +
                   la_target_stride_ * (s * 2 + 1)] +
              log_beta_data_
                  [lb_batch_offset + lb_input_stride_ * t +
                   lb_target_stride_ * (s * 2 + 1)] +
              nll - lp) *
              gr);
    }
  }
  CTCLossBackwardCollectNonblankKernelFunctor(
      scalar_t* RESTRICT gradient_data,
      const scalar_t* RESTRICT grad_out_data,
      int64_t grad_out_batch_stride,
      const scalar_t* RESTRICT log_alpha_data,
      const scalar_t* RESTRICT log_beta_data,
      const scalar_t* log_probs_data,
      const int64_t* RESTRICT input_lengths,
      const target_t* RESTRICT targets_data,
      const int64_t* RESTRICT target_lengths,
      const scalar_t* RESTRICT neg_log_likelihood_data,
      int64_t gr_input_stride,
      int64_t gr_batch_stride,
      int64_t gr_char_stride,
      int64_t lp_input_stride,
      int64_t lp_batch_stride,
      int64_t lp_char_stride,
      int64_t la_batch_stride,
      int64_t la_input_stride,
      int64_t la_target_stride,
      int64_t lb_batch_stride,
      int64_t lb_input_stride,
      int64_t lb_target_stride,
      const int64_t* RESTRICT tg_batch_offsets,
      int64_t tg_target_stride,
      int64_t batch_size,
      bool zero_infinity)
      : gradient_data_(gradient_data),
        grad_out_data_(grad_out_data),
        grad_out_batch_stride_(grad_out_batch_stride),
        log_alpha_data_(log_alpha_data),
        log_beta_data_(log_beta_data),
        log_probs_data_(log_probs_data),
        input_lengths_(input_lengths),
        targets_data_(targets_data),
        target_lengths_(target_lengths),
        neg_log_likelihood_data_(neg_log_likelihood_data),
        gr_input_stride_(gr_input_stride),
        gr_batch_stride_(gr_batch_stride),
        gr_char_stride_(gr_char_stride),
        lp_input_stride_(lp_input_stride),
        lp_batch_stride_(lp_batch_stride),
        lp_char_stride_(lp_char_stride),
        la_batch_stride_(la_batch_stride),
        la_input_stride_(la_input_stride),
        la_target_stride_(la_target_stride),
        lb_batch_stride_(lb_batch_stride),
        lb_input_stride_(lb_input_stride),
        lb_target_stride_(lb_target_stride),
        tg_batch_offsets_(tg_batch_offsets),
        tg_target_stride_(tg_target_stride),
        batch_size_(batch_size),
        zero_infinity_(zero_infinity) {}

 private:
  scalar_t* RESTRICT gradient_data_;
  const scalar_t* RESTRICT grad_out_data_;
  int64_t grad_out_batch_stride_;
  const scalar_t* RESTRICT log_alpha_data_;
  const scalar_t* RESTRICT log_beta_data_;
  const scalar_t* log_probs_data_;
  const int64_t* RESTRICT input_lengths_;
  const target_t* RESTRICT targets_data_;
  const int64_t* RESTRICT target_lengths_;
  const scalar_t* RESTRICT neg_log_likelihood_data_;
  int64_t gr_input_stride_;
  int64_t gr_batch_stride_;
  int64_t gr_char_stride_;
  int64_t lp_input_stride_;
  int64_t lp_batch_stride_;
  int64_t lp_char_stride_;
  int64_t la_batch_stride_;
  int64_t la_input_stride_;
  int64_t la_target_stride_;
  int64_t lb_batch_stride_;
  int64_t lb_input_stride_;
  int64_t lb_target_stride_;
  const int64_t* RESTRICT tg_batch_offsets_;
  int64_t tg_target_stride_;
  int64_t batch_size_;
  bool zero_infinity_;
};

// This is the naive implementation of equation (16). It is parallelised in
// batch and input timestep. It appears to be faster than the above method for
// small batch sizes.
template <typename scalar_t, typename target_t>
struct CTCLossBackwardCollectKernelFunctor {
  void operator()(sycl::nd_item<2> item) const {
    constexpr scalar_t neginf = -INFINITY;
    int64_t b =
        item.get_local_id(0) + item.get_group(0) * item.get_local_range(0);
    int64_t t =
        item.get_local_id(1) + item.get_group(1) * item.get_local_range(1);

    if ((t >= max_input_length_) || (b >= batch_size_))
      return;

    int64_t input_length = input_lengths_[b];
    int64_t target_length = target_lengths_[b];
    int64_t gr_batch_offset = b * gr_batch_stride_;
    int64_t lp_batch_offset = b * lp_batch_stride_;
    int64_t la_batch_offset = b * la_batch_stride_;
    int64_t lb_batch_offset = b * lb_batch_stride_;
    int64_t tg_batch_offset = tg_batch_offsets_[b];

    // collected[b, t, target'[s]] "log+=" log_alpha[t, s]+log_beta[t, s]
    for (int s = 0; s < 2 * max_target_length_ + 1; s++) {
      if (s < 2 * target_length + 1) { // if target_length == 0, s == 0
        int64_t current_target_prime = get_target_prime(
            targets_data_, tg_batch_offset, tg_target_stride_, s, BLANK_);
        scalar_t log_alpha_beta =
            (log_alpha_data_
                 [la_batch_offset + la_input_stride_ * t +
                  la_target_stride_ * s] +
             log_beta_data_
                 [lb_batch_offset + lb_input_stride_ * t +
                  lb_target_stride_ * s]);
        scalar_t& lcab = gradient_data_
            [gr_batch_offset + t * gr_input_stride_ +
             gr_char_stride_ * current_target_prime];
        if (lcab == neginf) {
          lcab = log_alpha_beta;
        } else {
          scalar_t max = ((lcab > log_alpha_beta) ? lcab : log_alpha_beta);
          lcab =
              std::log(std::exp(lcab - max) + std::exp(log_alpha_beta - max)) +
              max;
        }
      }
    }

    scalar_t nll = neg_log_likelihood_data_[b];
    scalar_t gr = grad_out_data_[b * grad_out_batch_stride_];

    for (int64_t c = 0; c < num_labels_; c++) {
      scalar_t& res = gradient_data_
          [gr_batch_offset + t * gr_input_stride_ + gr_char_stride_ * c];
      if (t < input_length && (!zero_infinity_ || nll != INFINITY)) {
        scalar_t lp = log_probs_data_
            [lp_batch_offset + t * lp_input_stride_ + lp_char_stride_ * c];
        res = (std::exp(lp) - std::exp(res + nll - lp)) * gr;
      } else {
        res = 0.;
      }
    }
  }

  CTCLossBackwardCollectKernelFunctor(
      scalar_t* RESTRICT gradient_data,
      const scalar_t* RESTRICT grad_out_data,
      int64_t grad_out_batch_stride,
      const scalar_t* RESTRICT log_alpha_data,
      const scalar_t* RESTRICT log_beta_data,
      const scalar_t* log_probs_data,
      const int64_t* RESTRICT input_lengths,
      int64_t max_input_length,
      const target_t* RESTRICT targets_data,
      const int64_t* RESTRICT target_lengths,
      int64_t max_target_length,
      const scalar_t* RESTRICT neg_log_likelihood_data,
      int64_t gr_input_stride,
      int64_t gr_batch_stride,
      int64_t gr_char_stride,
      int64_t lp_input_stride,
      int64_t lp_batch_stride,
      int64_t lp_char_stride,
      int64_t la_batch_stride,
      int64_t la_input_stride,
      int64_t la_target_stride,
      int64_t lb_batch_stride,
      int64_t lb_input_stride,
      int64_t lb_target_stride,
      const int64_t* RESTRICT tg_batch_offsets,
      int64_t tg_target_stride,
      int64_t batch_size,
      int64_t num_labels,
      int64_t BLANK,
      bool zero_infinity)
      : gradient_data_(gradient_data),
        grad_out_data_(grad_out_data),
        grad_out_batch_stride_(grad_out_batch_stride),
        log_alpha_data_(log_alpha_data),
        log_beta_data_(log_beta_data),
        log_probs_data_(log_probs_data),
        input_lengths_(input_lengths),
        max_input_length_(max_input_length),
        targets_data_(targets_data),
        target_lengths_(target_lengths),
        max_target_length_(max_target_length),
        neg_log_likelihood_data_(neg_log_likelihood_data),
        gr_input_stride_(gr_input_stride),
        gr_batch_stride_(gr_batch_stride),
        gr_char_stride_(gr_char_stride),
        lp_input_stride_(lp_input_stride),
        lp_batch_stride_(lp_batch_stride),
        lp_char_stride_(lp_char_stride),
        la_batch_stride_(la_batch_stride),
        la_input_stride_(la_input_stride),
        la_target_stride_(la_target_stride),
        lb_batch_stride_(lb_batch_stride),
        lb_input_stride_(lb_input_stride),
        lb_target_stride_(lb_target_stride),
        tg_batch_offsets_(tg_batch_offsets),
        tg_target_stride_(tg_target_stride),
        batch_size_(batch_size),
        num_labels_(num_labels),
        BLANK_(BLANK),
        zero_infinity_(zero_infinity) {}

 private:
  scalar_t* RESTRICT gradient_data_;
  const scalar_t* RESTRICT grad_out_data_;
  int64_t grad_out_batch_stride_;
  const scalar_t* RESTRICT log_alpha_data_;
  const scalar_t* RESTRICT log_beta_data_;
  const scalar_t* log_probs_data_;
  const int64_t* RESTRICT input_lengths_;
  int64_t max_input_length_;
  const target_t* RESTRICT targets_data_;
  const int64_t* RESTRICT target_lengths_;
  int64_t max_target_length_;
  const scalar_t* RESTRICT neg_log_likelihood_data_;
  int64_t gr_input_stride_;
  int64_t gr_batch_stride_;
  int64_t gr_char_stride_;
  int64_t lp_input_stride_;
  int64_t lp_batch_stride_;
  int64_t lp_char_stride_;
  int64_t la_batch_stride_;
  int64_t la_input_stride_;
  int64_t la_target_stride_;
  int64_t lb_batch_stride_;
  int64_t lb_input_stride_;
  int64_t lb_target_stride_;
  const int64_t* RESTRICT tg_batch_offsets_;
  int64_t tg_target_stride_;
  int64_t batch_size_;
  int64_t num_labels_;
  int64_t BLANK_;
  bool zero_infinity_;
};

// This is to zero gradients which corresponding to the out-of-sequence position
// Those gradients should not be used in any model update since the input
// elements are padded
template <typename scalar_t>
struct CTCLossZeroPaddedGradients {
  void operator()(sycl::nd_item<2> item) const {
    int64_t b =
        item.get_local_id(0) + item.get_group(0) * item.get_local_range(0);
    int64_t t =
        item.get_local_id(1) + item.get_group(1) * item.get_local_range(1);

    if (b >= batch_size_ || t >= max_input_length_) {
      return;
    }

    scalar_t input_length = input_lengths_[b];
    if (t >= input_length) {
      for (int l = 0; l < num_labels_; l++)
        gradient_data_
            [t * gr_timestep_stride_ + b * gr_batch_stride_ +
             l * gr_label_stride_] = 0.0f;
    }
  }

  CTCLossZeroPaddedGradients(
      scalar_t* RESTRICT gradient_data, /* (T, B, D) layout */
      const int64_t* RESTRICT input_lengths, /* (B, ) layout */
      int64_t gr_timestep_stride,
      int64_t gr_batch_stride,
      int64_t gr_label_stride,
      int64_t max_input_length, /* T */
      int64_t batch_size, /* B */
      int64_t num_labels /* D */
      )
      : gradient_data_(gradient_data),
        input_lengths_(input_lengths),
        gr_timestep_stride_(gr_timestep_stride),
        gr_batch_stride_(gr_batch_stride),
        gr_label_stride_(gr_label_stride),
        max_input_length_(max_input_length),
        batch_size_(batch_size),
        num_labels_(num_labels) {}

 private:
  scalar_t* RESTRICT gradient_data_; /* (T, B, D) layout */
  const int64_t* RESTRICT input_lengths_; /* (B, ) layout */
  int64_t gr_timestep_stride_;
  int64_t gr_batch_stride_;
  int64_t gr_label_stride_;
  int64_t max_input_length_; /* T */
  int64_t batch_size_; /* B */
  int64_t num_labels_; /* D */
};

// The backward. It essentially computes eq 16 by using the above kernels.
// We don't do a lot of checking as we envision this to be called only when
// backpropagating through a (well-checked) forward.
template <typename scalar_t, ScalarType target_scalar_type>
Tensor ctc_loss_backward_kernel_template(
    const Tensor& grad_out,
    const Tensor& log_probs,
    const Tensor& targets,
    IntArrayRef input_lengths,
    IntArrayRef target_lengths,
    const Tensor& neg_log_likelihood,
    const Tensor& log_alpha,
    int64_t BLANK,
    bool zero_infinity) {
  constexpr scalar_t neginf = -INFINITY;
  using target_t =
      typename std::conditional<target_scalar_type == kInt, int, int64_t>::type;
  int64_t batch_size = log_probs.size(1);
  int64_t num_labels = log_probs.size(2);
  int64_t tg_target_stride;

  int64_t max_target_length;
  auto tg_batch_offsets =
      at::empty({batch_size}, TensorOptions(at::CPU(kLong)));
  auto tg_batch_offsets_data = tg_batch_offsets.mutable_data_ptr<int64_t>();
  if (targets.dim() == 1) { // concatenated targets
    int64_t pos = 0;
    max_target_length = 0;
    for (int64_t i = 0; i < batch_size; i++) {
      tg_batch_offsets_data[i] = pos;
      pos += target_lengths[i];
      if (max_target_length < target_lengths[i])
        max_target_length = target_lengths[i];
    }
    tg_target_stride = targets.stride(0);
  } else { // batch x max_target_length
    // dim is 2
    int64_t tg_batch_stride = targets.stride(0);
    for (int64_t i = 0; i < batch_size; i++) {
      tg_batch_offsets_data[i] = i * tg_batch_stride;
    }
    tg_target_stride = targets.stride(1);
    max_target_length =
        log_alpha.size(2) / 2; // targets.size(1) might be larger
  }
  auto target_lengths_t =
      at::tensor(target_lengths, targets.options().dtype(kLong));
  auto input_lengths_t =
      at::tensor(input_lengths, targets.options().dtype(kLong));
  tg_batch_offsets = tg_batch_offsets.to(targets.device());

  Tensor log_beta = at::empty_like(log_alpha, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  log_beta.fill_(neginf);

  Tensor grad = at::full_like(
      log_probs,
      neginf,
      LEGACY_CONTIGUOUS_MEMORY_FORMAT); // initialization for log(sum (alpha
                                        // beta))

  using CTCLossBackwardLogBetaKernel =
      CTCLossBackwardLogBetaKernelFunctor<scalar_t, target_t>;
  int max_threads = syclMaxWorkGroupSize<CTCLossBackwardLogBetaKernel>();
  int threads_target = max_threads;
  while (threads_target / 2 >= 2 * max_target_length + 1) {
    threads_target /= 2;
  }
  int threads_batch = std::min(max_threads / threads_target, (int)batch_size);
  auto queue = at::xpu::getCurrentSYCLQueue();

  {
    auto group_size_x = threads_target;
    auto group_size_y = threads_batch;
    sycl::range<2> local_range(group_size_y, group_size_x);
    sycl::range<2> global_range(
        group_size_y * ((batch_size + threads_batch - 1) / threads_batch),
        group_size_x);
    auto caller = CTCLossBackwardLogBetaKernel(
        log_beta.mutable_data_ptr<scalar_t>(),
        log_probs.const_data_ptr<scalar_t>(),
        input_lengths_t.const_data_ptr<int64_t>(),
        log_probs.size(0),
        targets.const_data_ptr<target_t>(),
        target_lengths_t.const_data_ptr<int64_t>(),
        max_target_length,
        log_probs.stride(0),
        log_probs.stride(1),
        log_probs.stride(2),
        log_beta.stride(0),
        log_beta.stride(1),
        log_beta.stride(2),
        tg_batch_offsets.const_data_ptr<int64_t>(),
        tg_target_stride,
        batch_size,
        BLANK);
    sycl_kernel_submit(global_range, local_range, queue, caller);
  }

  // Very crude heuristic for what is a small problem., based on linearly
  // regressing problem dimensions on the (capped) difference of timings. Note
  // that for OK problems target length <= input length, so we only consider
  // input length.
  bool is_large = (2 * log_probs.size(0) + (24 * batch_size) / 10 +
                   (2 * num_labels) / 10) > 450;
  if (is_large) { // large alphabet, large batch
    // this computes the probs, minuend in (16)
    at::exp_out(grad, log_probs);
    // now we compute the subtrahend for the blanks. It is a straightforward
    // reduction because we know that blanks are in every other position. maybe
    // we should kernelize this, too.
    auto grad_blank = grad.narrow(2, BLANK, 1);
    grad_blank -=
        (at::logsumexp(
             log_alpha.as_strided(
                 {batch_size, log_alpha.size(1), max_target_length + 1},
                 {log_alpha.stride(0),
                  log_alpha.stride(1),
                  log_alpha.stride(2) * 2}) +
                 log_beta.as_strided(
                     {batch_size, log_beta.size(1), max_target_length + 1},
                     {log_beta.stride(0),
                      log_beta.stride(1),
                      log_beta.stride(2) * 2}),
             2,
             true)
             .permute({1, 0, 2})
             .add_(neg_log_likelihood.view({1, batch_size, 1}))
             .sub_(log_probs.narrow(2, BLANK, 1))
             .exp_());
    // scale by output gradient (blanks and first summand of non-blanks)
    grad *= grad_out.view({1, batch_size, 1});
    if (zero_infinity) {
      grad = at::where(
          neg_log_likelihood.view({1, batch_size, 1}) == Scalar(INFINITY),
          at::zeros({}, grad.options()),
          grad);
    }

    using CTCLossBackwardCollectNonblankKernel =
        CTCLossBackwardCollectNonblankKernelFunctor<scalar_t, target_t>;
    max_threads = syclMaxWorkGroupSize<CTCLossBackwardCollectNonblankKernel>();
    int threads_target = max_threads;
    while (threads_target / 2 >= max_target_length && threads_target > 1) {
      threads_target /= 2;
    }
    int threads_batch = std::min(max_threads / threads_target, (int)batch_size);
    auto group_size_x = threads_target;
    auto group_size_y = threads_batch;
    auto nwg_x = std::max<int>(
        (max_target_length + threads_target - 1) / threads_target, 1);
    auto nwg_y = (batch_size + threads_batch - 1) / threads_batch;
    sycl::range<2> local_range(group_size_y, group_size_x);
    sycl::range<2> global_range(nwg_y * group_size_y, nwg_x * group_size_x);
    auto caller = CTCLossBackwardCollectNonblankKernel(
        grad.mutable_data_ptr<scalar_t>(),
        grad_out.const_data_ptr<scalar_t>(),
        grad_out.stride(0),
        log_alpha.const_data_ptr<scalar_t>(),
        log_beta.const_data_ptr<scalar_t>(),
        log_probs.const_data_ptr<scalar_t>(),
        input_lengths_t.const_data_ptr<int64_t>(),
        targets.const_data_ptr<target_t>(),
        target_lengths_t.const_data_ptr<int64_t>(),
        neg_log_likelihood.const_data_ptr<scalar_t>(),
        grad.stride(0),
        grad.stride(1),
        grad.stride(2),
        log_probs.stride(0),
        log_probs.stride(1),
        log_probs.stride(2),
        log_alpha.stride(0),
        log_alpha.stride(1),
        log_alpha.stride(2),
        log_beta.stride(0),
        log_beta.stride(1),
        log_beta.stride(2),
        tg_batch_offsets.const_data_ptr<int64_t>(),
        tg_target_stride,
        batch_size,
        zero_infinity);
    sycl_kernel_submit(global_range, local_range, queue, caller);
  } else { // small problem, use naive algorithm
    using CTCLossBackwardCollectKernel =
        CTCLossBackwardCollectKernelFunctor<scalar_t, target_t>;
    max_threads = syclMaxWorkGroupSize<CTCLossBackwardCollectKernel>();
    int threads_input = max_threads;
    while (threads_input / 2 >= log_probs.size(0) && threads_input > 1) {
      threads_input /= 2;
    }
    threads_batch = std::min(max_threads / threads_input, (int)batch_size);
    auto group_size_x = threads_input;
    auto group_size_y = threads_batch;
    auto nwg_x = (log_probs.size(0) + threads_input - 1) / threads_input;
    auto nwg_y = (batch_size + threads_batch - 1) / threads_batch;
    sycl::range<2> local_range(group_size_y, group_size_x);
    sycl::range<2> global_range(nwg_y * group_size_y, nwg_x * group_size_x);
    auto caller = CTCLossBackwardCollectKernel(
        grad.mutable_data_ptr<scalar_t>(),
        grad_out.const_data_ptr<scalar_t>(),
        grad_out.stride(0),
        log_alpha.const_data_ptr<scalar_t>(),
        log_beta.const_data_ptr<scalar_t>(),
        log_probs.const_data_ptr<scalar_t>(),
        input_lengths_t.const_data_ptr<int64_t>(),
        log_probs.size(0),
        targets.const_data_ptr<target_t>(),
        target_lengths_t.const_data_ptr<int64_t>(),
        max_target_length,
        neg_log_likelihood.const_data_ptr<scalar_t>(),
        grad.stride(0),
        grad.stride(1),
        grad.stride(2),
        log_probs.stride(0),
        log_probs.stride(1),
        log_probs.stride(2),
        log_alpha.stride(0),
        log_alpha.stride(1),
        log_alpha.stride(2),
        log_beta.stride(0),
        log_beta.stride(1),
        log_beta.stride(2),
        tg_batch_offsets.const_data_ptr<int64_t>(),
        tg_target_stride,
        batch_size,
        num_labels,
        BLANK,
        zero_infinity);
    sycl_kernel_submit(global_range, local_range, queue, caller);
  }

  // zero those invalid graident elements due to padding
  {
    using CTCLossZeroPaddedGradientsKernel =
        CTCLossZeroPaddedGradients<scalar_t>;
    max_threads = syclMaxWorkGroupSize<CTCLossZeroPaddedGradientsKernel>();
    int threads_input = max_threads;
    while (threads_input / 2 >= log_probs.size(0)) {
      threads_input /= 2;
    }
    threads_batch = std::min(max_threads / threads_input, (int)batch_size);
    auto group_size_x = threads_input;
    auto group_size_y = threads_batch;
    auto nwg_x = (log_probs.size(0) + threads_input - 1) / threads_input;
    auto nwg_y = (batch_size + threads_batch - 1) / threads_batch;
    sycl::range<2> local_range(group_size_y, group_size_x);
    sycl::range<2> global_range(nwg_y * group_size_y, nwg_x * group_size_x);
    auto caller = CTCLossZeroPaddedGradientsKernel(
        grad.mutable_data_ptr<scalar_t>(),
        input_lengths_t.const_data_ptr<int64_t>(),
        grad.stride(0),
        grad.stride(1),
        grad.stride(2),
        grad.size(0),
        grad.size(1),
        grad.size(2));
    sycl_kernel_submit(global_range, local_range, queue, caller);
  }

  return grad;
}

std::tuple<Tensor, Tensor> ctc_loss_kernel(
    const Tensor& log_probs,
    const Tensor& targets,
    IntArrayRef input_lengths,
    IntArrayRef target_lengths,
    int64_t BLANK,
    bool zero_infinity) {
  (void)zero_infinity; // only used for backward
  return AT_DISPATCH_FLOATING_TYPES(
      log_probs.scalar_type(), "ctc_loss_xpu", [&] {
        if (targets.scalar_type() == kLong) {
          return ctc_loss_kernel_template<scalar_t, kLong>(
              log_probs, targets, input_lengths, target_lengths, BLANK);
        } else {
          return ctc_loss_kernel_template<scalar_t, kInt>(
              log_probs, targets, input_lengths, target_lengths, BLANK);
        }
      });
}

Tensor ctc_loss_backward_kernel(
    const Tensor& grad,
    const Tensor& log_probs,
    const Tensor& targets,
    IntArrayRef input_lengths,
    IntArrayRef target_lengths,
    const Tensor& neg_log_likelihood,
    const Tensor& log_alpha,
    int64_t BLANK,
    bool zero_infinity) {
  // See Note [Writing Nondeterministic Operations]
  // Nondeterministic because of atomicAdd usage
  globalContext().alertNotDeterministic("ctc_loss_backward_kernel");
  return AT_DISPATCH_FLOATING_TYPES(
      log_probs.scalar_type(), "ctc_loss_backward_xpu", [&] {
        if (targets.scalar_type() == kLong) {
          return ctc_loss_backward_kernel_template<scalar_t, kLong>(
              grad,
              log_probs,
              targets,
              input_lengths,
              target_lengths,
              neg_log_likelihood,
              log_alpha,
              BLANK,
              zero_infinity);
        } else {
          return ctc_loss_backward_kernel_template<scalar_t, kInt>(
              grad,
              log_probs,
              targets,
              input_lengths,
              target_lengths,
              neg_log_likelihood,
              log_alpha,
              BLANK,
              zero_infinity);
        }
      });
}

} // namespace at::native::xpu

#pragma GCC diagnostic pop
#pragma clang diagnostic pop
