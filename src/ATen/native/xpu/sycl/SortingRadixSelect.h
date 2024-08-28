#pragma once
#include <ATen/ceil_div.h>
#include <ATen/native/xpu/sycl/Atomics.h>
#include <comm/SYCLContext.h>
#include <comm/SYCLHelpers.h>

namespace at::native::xpu {

template <typename scalar_t>
struct TopKTypeConfig {};

template <>
struct TopKTypeConfig<float> {
  using RadixType = uint32_t;

  // Converts a float to an integer representation with the same
  // sorting; i.e., for floats f1, f2:
  // if f1 < f2 then convert(f1) < convert(f2)
  // We use this to enable radix selection of floating-point values.
  // This also gives a relative order for NaNs, but that's ok, as they
  // will all be adjacent
  // neg inf: signbit=1 exp=ff fraction=0 --> radix = 0 00 ff..
  // pos inf: signbit=0 exp=ff fraction=0 --> radix = 1 ff 00..
  // pos nan: signbit=0 exp=ff fraction>0 --> radix = 1 ff x>0
  // neg nan: signbit=1 exp=ff fraction>0 --> radix = 0 00 x<ff...
  static inline RadixType convert(float v) {
    RadixType x = *((uint32_t*)&v);
    RadixType mask = (x & 0x80000000) ? 0xffffffff : 0x80000000;
    return (x ^ mask);
  }

  static inline float deconvert(RadixType v) {
    RadixType mask = (v & 0x80000000) ? 0x80000000 : 0xffffffff;
    return __int_as_float(v ^ mask);
  }
};

template <>
struct TopKTypeConfig<uint8_t> {
  using RadixType = uint32_t;

  static inline RadixType convert(uint8_t v) {
    return v;
  }

  static inline uint8_t deconvert(RadixType v) {
    return v;
  }
};

template <>
struct TopKTypeConfig<int8_t> {
  using RadixType = uint32_t;

  static inline RadixType convert(int8_t v) {
    return 128u + v;
  }

  static inline int8_t deconvert(RadixType v) {
    return v - 128;
  }
};

template <>
struct TopKTypeConfig<int16_t> {
  using RadixType = uint32_t;

  static inline RadixType convert(int16_t v) {
    return 32768u + v;
  }

  static inline int16_t deconvert(RadixType v) {
    return v - 32768;
  }
};

template <>
struct TopKTypeConfig<int32_t> {
  using RadixType = uint32_t;

  static inline RadixType convert(int32_t v) {
    return 2147483648u + v;
  }

  static inline int32_t deconvert(RadixType v) {
    return v - 2147483648u;
  }
};

template <>
struct TopKTypeConfig<int64_t> {
  using RadixType = uint64_t;

  static inline RadixType convert(int64_t v) {
    return 9223372036854775808ull + v;
  }

  static inline int64_t deconvert(RadixType v) {
    return v - 9223372036854775808ull;
  }
};

template <>
struct TopKTypeConfig<double> {
  using RadixType = uint64_t;

  static inline RadixType convert(double v) {
    RadixType x = *((uint64_t*)&v);
    RadixType mask = -((x >> 63)) | 0x8000000000000000;
    return (x ^ mask);
  }

  static inline double deconvert(RadixType v) {
    RadixType mask = ((v >> 63) - 1) | 0x8000000000000000;
    return __long_long_as_double(v ^ mask);
  }
};

template <>
struct TopKTypeConfig<at::Half> {
  using RadixType = uint32_t;

  static inline RadixType convert(at::Half v) {
    RadixType x = *((uint16_t*)&v);
    RadixType mask = -((x >> 15)) | 0x8000;
    return (x ^ mask);
  }

  static inline at::Half deconvert(RadixType v) {
    RadixType mask = ((v >> 15) - 1) | 0x8000;
    return __ushort_as_half(v ^ mask);
  }
};

template <>
struct TopKTypeConfig<at::BFloat16> {
  using RadixType = uint32_t;

  static inline RadixType convert(at::BFloat16 v) {
    RadixType x = v.x;
    RadixType mask = (x & 0x00008000) ? 0x0000ffff : 0x00008000;
    return (v == v) ? (x ^ mask) : 0xffff;
  }

  static inline at::BFloat16 deconvert(RadixType v) {
    RadixType mask = (v & 0x00008000) ? 0x00008000 : 0x0000ffff;
    at::BFloat16 r;
    r.x = (v ^ mask);
    return r;
  }
};

template <typename T>
struct Bitfield {};

template <>
struct Bitfield<unsigned int> {
  static inline unsigned int getBitfield(unsigned int val, int pos, int len) {
    pos &= 0xff;
    len &= 0xff;
    unsigned int m = (1u << len) - 1u;
    return (val >> pos) & m;
  }

  static inline unsigned int setBitfield(
      unsigned int val,
      unsigned int toInsert,
      int pos,
      int len) {
    pos &= 0xff;
    len &= 0xff;
    unsigned int m = (1u << len) - 1u;
    toInsert &= m;
    toInsert <<= pos;
    m <<= pos;
    return (val & ~m) | toInsert;
  }
};

template <>
struct Bitfield<uint64_t> {
  static inline uint64_t getBitfield(uint64_t val, int pos, int len) {
    pos &= 0xff;
    len &= 0xff;

    uint64_t m = (1u << len) - 1u;
    return (val >> pos) & m;
  }

  static inline uint64_t setBitfield(
      uint64_t val,
      uint64_t toInsert,
      int pos,
      int len) {
    pos &= 0xff;
    len &= 0xff;

    uint64_t m = (1u << len) - 1u;
    toInsert &= m;
    toInsert <<= pos;
    m <<= pos;
    return (val & ~m) | toInsert;
  }
};

// This function counts the distribution of all input values in a
// slice we are selecting by radix digit at `radixDigitPos`, but only
// those that pass the filter `((v & desiredMask) == desired)`.
// This produces and broadcasts the seen counts for a single block only.
// `smem` must have at least `RadixSize` elements.
template <
    typename scalar_t,
    typename bitwise_t,
    typename index_t,
    typename CountType,
    int RadixSize,
    int RadixBits>
void countRadixUsingMask(
    CountType counts[RadixSize],
    const sycl_local_acc_t<int>& smem,
    bitwise_t desired,
    bitwise_t desiredMask,
    int radixDigitPos,
    index_t sliceSize,
    index_t withinSliceStride,
    const sycl_global_ptr<scalar_t>& data,
    sycl::nd_item<1>& item_id) {
  // Clear out per-thread counts from a previous round
  for (int i = 0; i < RadixSize; ++i) {
    counts[i] = 0;
  }

  auto local_id = item_id.get_local_id(0);
  if (local_id < RadixSize) {
    smem[local_id] = 0;
  }

  item_id.barrier(sycl_local_fence);
  // Scan over all the data. Upon a read, the warp will accumulate
  // counts per each digit in the radix using warp voting.
  for (index_t i = local_id; i < sliceSize; i += item_id.get_local_range(0)) {
    bitwise_t val =
        TopKTypeConfig<scalar_t>::convert(data[i * withinSliceStride]);

    bool hasVal = ((val & desiredMask) == desired);
    bitwise_t digitInRadix =
        Bitfield<bitwise_t>::getBitfield(val, radixDigitPos, RadixBits);
    if (hasVal)
      counts[digitInRadix]++;
  }

  auto smem_ptr = (sycl_local_ptr<int>)(smem.template get_multi_ptr<
                                                sycl::access::decorated::no>()
                                            .get());
  for (uint32_t i = 0; i < RadixSize; ++i) {
    atomicAdd(smem_ptr + i, counts[i]);
  }

  item_id.barrier(sycl_local_fence);

  // For each thread, read in the total counts
  for (uint32_t i = 0; i < RadixSize; ++i) {
    counts[i] = smem[i];
  }

  item_id.barrier(sycl_local_fence);
}

// Over what radix we are selecting values
constexpr int RADIX_BITS = 2; // digits are base-(2 ^ RADIX_BITS)
constexpr int RADIX_SIZE = 4; // 2 ^ RADIX_BITS
constexpr int RADIX_MASK = (RADIX_SIZE - 1);

// This finds the unique value `v` that matches the pattern
// ((v & desired) == desiredMask) in our sorted int format
template <typename scalar_t, typename bitwise_t, typename index_t>
scalar_t findPattern(
    const sycl_local_acc_t<int>& smem,
    const sycl_global_ptr<scalar_t>& data,
    index_t sliceSize,
    index_t withinSliceStride,
    bitwise_t desired,
    bitwise_t desiredMask,
    sycl::nd_item<1>& item_id) {
  auto local_id = item_id.get_local_id(0);
  auto smem_ptr = static_cast<scalar_t*>(static_cast<void*>(
      smem.template get_multi_ptr<sycl::access::decorated::no>().get()));
  if (local_id < RADIX_SIZE) {
    smem_ptr[RADIX_SIZE] = static_cast<scalar_t>(0);
  }

  item_id.barrier(sycl_local_fence);

  // All threads participate in the loop, in order to sync on the flag
  index_t numIterations =
      round_up(sliceSize, (index_t)item_id.get_local_range(0));
  for (index_t i = local_id; i < numIterations;
       i += item_id.get_local_range(0)) {
    bool inRange = (i < sliceSize);
    scalar_t v =
        inRange ? data[i * withinSliceStride] : static_cast<scalar_t>(0);

    if (inRange &&
        ((TopKTypeConfig<scalar_t>::convert(v) & desiredMask) == desired)) {
      // There should not be conflicts if we are using findPattern,
      // since the result is unique
      smem_ptr[0] = static_cast<scalar_t>(1);
      smem_ptr[1] = v; // can't use val as the flag, since it could be 0
    }

    item_id.barrier(sycl_local_fence);

    scalar_t found = smem_ptr[0];
    scalar_t val = smem_ptr[1];

    item_id.barrier(sycl_local_fence);

    // Check to see if a thread found the value
    if (found != static_cast<scalar_t>(0)) {
      // all threads return this value
      return val;
    }
  }

  // should not get here
  return static_cast<scalar_t>(0);
}

// Returns the top-Kth element found in the data using radix selection
template <typename scalar_t, typename bitwise_t, typename index_t, bool Order>
void radixSelect(
    const sycl_global_ptr<scalar_t>& data,
    index_t k,
    index_t sliceSize,
    index_t withinSliceStride,
    const sycl_local_acc_t<int>& smem,
    scalar_t* topK,
    sycl::nd_item<1>& item_id) {
  // Per-thread buckets into which we accumulate digit counts in our
  // radix
  int counts[RADIX_SIZE];

  // We only consider elements x such that (x & desiredMask) == desired
  // Initially, we consider all elements of the array, so the above
  // statement is true regardless of input.
  bitwise_t desired = 0;
  bitwise_t desiredMask = 0;

  // We are looking for the top kToFind-th element when iterating over
  // digits; this count gets reduced by elimination when counting
  // successive digits
  int kToFind = k;

  // We start at the most significant digit in our radix, scanning
  // through to the least significant digit
  for (int digitPos = sizeof(scalar_t) * 8 - RADIX_BITS; digitPos >= 0;
       digitPos -= RADIX_BITS) {
    // Count radix distribution for the current position and reduce
    // across all threads
    countRadixUsingMask<
        scalar_t,
        bitwise_t,
        index_t,
        int,
        RADIX_SIZE,
        RADIX_BITS>(
        counts,
        smem,
        desired,
        desiredMask,
        digitPos,
        sliceSize,
        withinSliceStride,
        data,
        item_id);

    auto found_unique = [&](int i, int count) -> bool {
      /* All threads have the same value in counts here, so all */
      /* threads will return from the function. */
      if (count == 1 && kToFind == 1) {
        /* There is a unique answer. */
        desired =
            Bitfield<bitwise_t>::setBitfield(desired, i, digitPos, RADIX_BITS);
        desiredMask = Bitfield<bitwise_t>::setBitfield(
            desiredMask, RADIX_MASK, digitPos, RADIX_BITS);

        /* The answer is now the unique element v such that: */
        /* (v & desiredMask) == desired */
        /* However, we do not yet know what the actual element is. We */
        /* need to perform a search through the data to find the */
        /* element that matches this pattern. */
        *topK = findPattern<scalar_t, bitwise_t, index_t>(
            smem,
            data,
            sliceSize,
            withinSliceStride,
            desired,
            desiredMask,
            item_id);
        return true;
      }
      return false;
    };
    auto found_non_unique = [&](int i, int count) -> bool {
      if (count >= kToFind) {
        desired =
            Bitfield<bitwise_t>::setBitfield(desired, i, digitPos, RADIX_BITS);
        desiredMask = Bitfield<bitwise_t>::setBitfield(
            desiredMask, RADIX_MASK, digitPos, RADIX_BITS);

        /* The top-Kth element v must now be one such that: */
        /* (v & desiredMask == desired) */
        /* but we haven't narrowed it down; we must check the next */
        /* least-significant digit */
        return true;
      }
      kToFind -= count;
      return false; // continue the loop
    };

    // All threads participate in the comparisons below to know the
    // final result
    if (Order) {
      // Process in descending order
      for (int i = RADIX_SIZE - 1; i >= 0; --i) {
        int count = counts[i];
        if (found_unique(i, count)) {
          return;
        }
        if (found_non_unique(i, count)) {
          break;
        }
      }
    } else {
      // Process in ascending order
      for (int i = 0; i < RADIX_SIZE; ++i) {
        int count = counts[i];
        if (found_unique(i, count)) {
          return;
        }
        if (found_non_unique(i, count)) {
          break;
        }
      }
    }
  } // end digitPos for

  // There is no unique result, but there is a non-unique result
  // matching `desired` exactly
  *topK = TopKTypeConfig<scalar_t>::deconvert(desired);
}

} // namespace at::native::xpu