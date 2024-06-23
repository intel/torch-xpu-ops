#pragma once

#include <ATen/ATen.h>
#include <comm/SYCLContext.h>
#include <comm/Scalar.h>
#include <stdlib.h>

namespace at {
namespace native {
namespace xpu {

struct NullType {
  using value_type = NullType;
  template <typename T>
  inline NullType& operator=(const T&) {
    return *this;
  }
  inline bool operator==(const NullType&) {
    return true;
  }
  inline bool operator!=(const NullType&) {
    return false;
  }
};

template <typename T>
struct KeyTraits {};

template <>
struct KeyTraits<NullType> {
  using Type = uint32_t;
  static inline Type convert(float v) {
    return 0;
  }
  static inline NullType deconvert(Type v) {
    return NullType();
  }
  static inline unsigned int endbit() {
    return 0;
  }
};

template <>
struct KeyTraits<float> {
  using Type = uint32_t;
  static inline Type convert(float v) {
    Type x = *((Type*)&v);
    Type mask = (x & 0x80000000) ? 0xffffffff : 0x80000000;
    return (x ^ mask);
  }
  static inline float deconvert(Type v) {
    Type mask = (v & 0x80000000) ? 0x80000000 : 0xffffffff;
    return __int_as_float(v ^ mask);
  }
  static inline int endbit() {
    return sizeof(Type) << 3;
  }
};

template <>
struct KeyTraits<bool> {
  using Type = bool;
  static inline Type convert(bool v) {
    return v;
  }
  static inline bool deconvert(Type v) {
    return v;
  }
  static inline int endbit() {
    return 1;
  }
};

template <>
struct KeyTraits<uint8_t> {
  using Type = uint8_t;
  static inline Type convert(uint8_t v) {
    return v;
  }
  static inline uint8_t deconvert(Type v) {
    return v;
  }
  static inline int endbit() {
    return sizeof(Type) << 3;
  }
};

template <>
struct KeyTraits<int8_t> {
  using Type = uint8_t;
  static inline Type convert(int8_t v) {
    return 128u + v;
  }
  static inline int8_t deconvert(Type v) {
    return v - 128;
  }
  static inline int endbit() {
    return sizeof(Type) << 3;
  }
};

template <>
struct KeyTraits<int16_t> {
  using Type = uint16_t;
  static inline Type convert(int16_t v) {
    return 32768u + v;
  }
  static inline int16_t deconvert(Type v) {
    return v - 32768;
  }
  static inline int endbit() {
    return sizeof(Type) << 3;
  }
};

template <>
struct KeyTraits<int32_t> {
  using Type = uint32_t;
  static inline Type convert(int32_t v) {
    return 2147483648u + v;
  }
  static inline int32_t deconvert(Type v) {
    return v - 2147483648u;
  }
  static inline int endbit() {
    return sizeof(Type) << 3;
  }
};

template <>
struct KeyTraits<int64_t> {
  using Type = uint64_t;
  static inline Type convert(int64_t v) {
    return 9223372036854775808ull + v;
  }
  static inline int64_t deconvert(Type v) {
    return v - 9223372036854775808ull;
  }
  static inline int endbit() {
    return sizeof(Type) << 3;
  }
};

template <>
struct KeyTraits<double> {
  using Type = uint64_t;
  static inline Type convert(double v) {
    Type x = *((Type*)&v);
    Type mask = -((x >> 63)) | 0x8000000000000000;
    return (x ^ mask);
  }
  static inline double deconvert(Type v) {
    Type mask = ((v >> 63) - 1) | 0x8000000000000000;
    return __long_long_as_double(v ^ mask);
  }
  static inline int endbit() {
    return sizeof(Type) << 3;
  }
};

template <>
struct KeyTraits<at::Half> {
  using Type = uint16_t;
  static inline Type convert(at::Half v) {
    Type x = *((Type*)&v);
    Type mask = -((x >> 15)) | 0x8000;
    return (x ^ mask);
  }
  static inline at::Half deconvert(Type v) {
    Type mask = ((v >> 15) - 1) | 0x8000;
    Type v_de = v ^ mask;
    return reinterpret_cast<at::Half&>(v_de);
  }
  static inline int endbit() {
    return sizeof(Type) << 3;
  }
};

template <>
struct KeyTraits<at::BFloat16> {
  using Type = uint16_t;
  static inline Type convert(at::BFloat16 v) {
    Type x = *((Type*)&v);
    Type mask = -((x >> 15)) | 0x8000;
    return (x ^ mask);
  }
  static inline at::BFloat16 deconvert(Type v) {
    Type mask = ((v >> 15) - 1) | 0x8000;
    Type v_de = v ^ mask;
    return reinterpret_cast<at::BFloat16&>(v_de);
  }
  static inline int endbit() {
    return sizeof(Type) << 3;
  }
};

template <int N, int CURRENT_VAL = N, int COUNT = 0>
struct Log2 {
  enum { VALUE = Log2<N, (CURRENT_VAL >> 1), COUNT + 1>::VALUE };
};

template <int N, int COUNT>
struct Log2<N, 0, COUNT> {
  enum { VALUE = (1 << (COUNT - 1) < N) ? COUNT : COUNT - 1 };
};

template <typename T, int STEPS>
inline void subgroup_cumsum(
    sycl::sub_group& sg,
    const int sgid,
    const T input,
    T& inclusive_sum,
    T& exclusive_sum) {
  inclusive_sum = input;
#pragma unroll
  for (int i = 0, offset = 1; i < STEPS; ++i, offset <<= 1) {
    T temp = sycl::shift_group_right(sg, inclusive_sum, offset);
    if (sgid >= offset)
      inclusive_sum += temp;
  }
  exclusive_sum = inclusive_sum - input;
}

template <
    typename T,
    int COUNTER_LANES,
    int GROUP_SIZE,
    int SUBGROUP_SIZE,
    bool EXCLUSIVE = true>
inline T group_cumsum(T* storage, sycl::nd_item<1>& item) {
  static_assert(
      GROUP_SIZE % SUBGROUP_SIZE == 0,
      "GROUP_SIZE should be n * SUBGROUP_SIZE. (n = 1, 2, 3, ...)");

  const int NUM_SUBGROUPS = GROUP_SIZE / SUBGROUP_SIZE;
  const int SUBGROUP_SCAN_STEPS = Log2<SUBGROUP_SIZE>::VALUE;

  int lid = item.get_local_linear_id();
  auto sg = item.get_sub_group();

  int subgroup_local_id = sg.get_local_id()[0];
  int subgroup_id = sg.get_group_id()[0];
  int lane_temp_values[COUNTER_LANES];

  // Read input lane sum
  auto storage_lanes = storage + lid * COUNTER_LANES;
  T lane_all_sum = 0;

  if (EXCLUSIVE) {
#pragma unroll
    for (int lane = 0; lane < COUNTER_LANES; ++lane) {
      lane_temp_values[lane] = lane_all_sum;
      lane_all_sum += storage_lanes[lane];
    }
  } else {
#pragma unroll
    for (int lane = 0; lane < COUNTER_LANES; ++lane) {
      lane_all_sum += storage_lanes[lane];
      lane_temp_values[lane] = lane_all_sum;
    }
  }

  // Get subgroup level exclusive sum
  T subgroup_inclusive_sum, subgroup_exclusive_sum;
  subgroup_cumsum<T, SUBGROUP_SCAN_STEPS>(
      sg,
      subgroup_local_id,
      lane_all_sum,
      subgroup_inclusive_sum,
      subgroup_exclusive_sum);
  item.barrier(sycl_local_fence);

  // Write to storage
  if (subgroup_local_id == (SUBGROUP_SIZE - 1))
    storage[subgroup_id] = subgroup_inclusive_sum;
  item.barrier(sycl_local_fence);

  // Get group prefix
  T group_all_sum = 0, group_exclusive_sum;
#pragma unroll
  for (int i = 0; i < NUM_SUBGROUPS; ++i) {
    if (subgroup_id == i)
      group_exclusive_sum = group_all_sum;
    group_all_sum += storage[i];
  }
  item.barrier(sycl_local_fence);

  // Write to storage
  subgroup_exclusive_sum += group_exclusive_sum;
#pragma unroll
  for (int lane = 0; lane < COUNTER_LANES; ++lane) {
    storage_lanes[lane] = subgroup_exclusive_sum + lane_temp_values[lane];
  }
  item.barrier(sycl_local_fence);

  return group_all_sum;
}

template <typename T, int COUNTER_LANES, int GROUP_SIZE, int SUBGROUP_SIZE>
inline T group_exclusive_cumsum(T* slm_storage, sycl::nd_item<1>& item) {
  return group_cumsum<T, COUNTER_LANES, GROUP_SIZE, SUBGROUP_SIZE, true>(
      slm_storage, item);
}

template <typename T, int COUNTER_LANES, int GROUP_SIZE, int SUBGROUP_SIZE>
inline T group_inclusive_cumsum(T* slm_storage, sycl::nd_item<1>& item) {
  return group_cumsum<T, COUNTER_LANES, GROUP_SIZE, SUBGROUP_SIZE, false>(
      slm_storage, item);
}

} // namespace xpu
} // namespace native
} // namespace at
