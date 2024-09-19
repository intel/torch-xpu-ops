#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>
#include <comm/SYCLContext.h>

#include <ATen/native/xpu/sycl/LossCTCKernels.h>

namespace at::native::xpu {

// this ad-hoc converts from targets (l in [1]) to augmented targets (l' in [1])
// so if l is l_0 l_1 ... l_(tl-1) then this looks up idx in
// l' = BLANK l_0 BLANK l_1 BLANK ... BLANK l_(tl-1) BLANK
// - note that no bound-checking is done
// - it is important to only call it with idx == 0 if the target length is 0
// - __restrict__ impact to be measured, see
template <typename target_t>
static inline int64_t get_target_prime(
    const target_t* __restrict__ target,
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

    if (b >= batch_size_)
      return;

    if (input_length == 0) {
      if (tid_x == 0) {
        scalar_t log_likelihood = target_length == 0 ? 0 : neginf;
        neg_log_likelihood_data_[b] = -log_likelihood;
      }
      return;
    }

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

    for (int64_t block_s = 0; block_s < 2 * max_target_length_ + 1;
         block_s += item.get_local_range(1)) {
      int64_t s = tid_x + block_s;

      // These two only depend on s, so we can cache them.
      int64_t current_char; // l_s in eq (6)
      bool have_three; // flag which of the two cases in eq (6) we have
      if (s < 2 * target_length + 1 && target_length > 0) {
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
        if ((t < input_length) && (s < 2 * target_length + 1)) {
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
          if (s < 2 * max_target_length_ + 1)
            log_alpha_data_
                [la_batch_offset + la_input_stride_ * t +
                 la_target_stride_ * s] = neginf;
        }
      }
    }
    item.barrier(sycl_local_fence);

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
      scalar_t* __restrict__ log_alpha_data,
      const scalar_t* log_probs_data,
      const int64_t* __restrict__ input_lengths,
      int64_t max_input_length,
      const target_t* __restrict__ targets_data,
      const int64_t* __restrict__ target_lengths,
      int64_t max_target_length,
      scalar_t* __restrict__ neg_log_likelihood_data,
      int64_t lp_input_stride,
      int64_t lp_batch_stride,
      int64_t lp_char_stride,
      int64_t la_batch_stride,
      int64_t la_input_stride,
      int64_t la_target_stride,
      const int64_t* __restrict__ tg_batch_offsets,
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
  scalar_t* __restrict__ log_alpha_data_;
  const scalar_t* log_probs_data_;
  const int64_t* __restrict__ input_lengths_;
  int64_t max_input_length_;
  const target_t* __restrict__ targets_data_;
  const int64_t* __restrict__ target_lengths_;
  int64_t max_target_length_;
  scalar_t* __restrict__ neg_log_likelihood_data_;
  int64_t lp_input_stride_;
  int64_t lp_batch_stride_;
  int64_t lp_char_stride_;
  int64_t la_batch_stride_;
  int64_t la_input_stride_;
  int64_t la_target_stride_;
  const int64_t* __restrict__ tg_batch_offsets_;
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

  // Very likely, we could be more clever here, e.g. learning (or generalizing
  // and reusing) from SoftMax.cu...
  constexpr int max_threads = std::is_same<scalar_t, float>::value
      ? 1024
      : 768; // we need 72 or so 32 bit registers for double
  int threads_target = max_threads;
  while (threads_target / 2 >= 2 * max_target_length + 1) {
    threads_target /= 2;
  }
  int threads_batch = std::min(max_threads / threads_target, (int)batch_size);
  size_t group_size_x = threads_target;
  size_t group_size_y = threads_batch;
  size_t ngroups_x = 1;
  size_t ngroups_y = (batch_size + threads_batch - 1) / threads_batch;

  sycl::range<2> global_range{group_size_y, group_size_x};
  sycl::range<2> local_range{
      ngroups_y * group_size_y, ngroups_x * group_size_x};

  auto caller = CTCLossLogAlphaKernelFunctor<scalar_t, target_t>(
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

} // namespace at::native::xpu
