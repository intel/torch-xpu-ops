#pragma once

#include <ATen/native/TensorIterator.h>

// #ifndef AT_PER_OPERATOR_HEADERS
// #include <ATen/Functions.h>
// #include <ATen/NativeFunctions.h>
// #else
// #include <ATen/ops/_chunk_cat_native.h>
// #include <ATen/ops/empty.h>
// #include <ATen/ops/split_with_sizes_copy_native.h>
// #endif

#include <ATen/native/xpu/sycl/Philox4x32.h>

namespace at::native::xpu {

static constexpr int64_t BLOCK_SIZE = 128;
static constexpr int64_t BYTES_PER_THREAD = 16;
static constexpr int64_t BYTES_PER_BLOCK = BYTES_PER_THREAD * BLOCK_SIZE;

inline int64_t div_up(int64_t a, int64_t b) {
  return (a + b - 1) / b;
}

template <typename T>
inline void stream_load128(uint4& val, const T* addr) {
  uint64_t low, high;
  low = reinterpret_cast<const uint64_t*>(addr)[0];
  high = reinterpret_cast<const uint64_t*>(addr)[1];
  reinterpret_cast<uint64_t*>(&val)[0] = low;
  reinterpret_cast<uint64_t*>(&val)[1] = high;
}

template <typename T>
inline void stream_store128(T* addr, const uint4& val) {
  uint64_t low, high;
  low = reinterpret_cast<const uint64_t*>(&val)[0];
  high = reinterpret_cast<const uint64_t*>(&val)[1];
  reinterpret_cast<uint64_t*>(addr)[0] = low;
  reinterpret_cast<uint64_t*>(addr)[1] = high;
}

template <typename T>
static inline bool is_aligned(const void* addr) {
  return reinterpret_cast<uintptr_t>(addr) % sizeof(T) == 0;
}

template <typename T>
static inline void load128(uint4& val, const char* addr) {
  for (size_t i = 0; i < BYTES_PER_THREAD / sizeof(T); ++i) {
    reinterpret_cast<T*>(&val)[i] = reinterpret_cast<const T*>(addr)[i];
  }
}

template <>
inline void load128<uint4>(uint4& val, const char* addr) {
  stream_load128(val, addr);
}

static inline void load128(uint4& val, const char* addr) {
  if (is_aligned<uint4>(addr)) {
    load128<uint4>(val, addr);
  } else if (is_aligned<int64_t>(addr)) {
    load128<uint64_t>(val, addr);
  } else if (is_aligned<uint32_t>(addr)) {
    load128<uint32_t>(val, addr);
  } else {
    load128<uint8_t>(val, addr);
  }
}

static inline void get_aligned_region(
    char* ptr,
    const int64_t chunk_size,
    const int64_t alignment,
    int64_t& align_off,
    int64_t& aligned_size) {
  const int64_t ptr_val = reinterpret_cast<uintptr_t>(ptr);
  align_off = div_up(ptr_val, alignment) * alignment - ptr_val;
  aligned_size = (chunk_size - align_off) / alignment * alignment;
}

static inline void copy_chunk(
    char* dst,
    const char* src,
    int64_t chunk_size,
    int64_t thread_idx,
    int64_t num_threads) {
  if (chunk_size < num_threads) {
    if (thread_idx < chunk_size) {
      dst[thread_idx] = src[thread_idx];
    }
    return;
  }

  // Identify the region in which writes are guaranteed to be 128-bit aligned
  int64_t align_off, aligned_size;
  get_aligned_region(
      dst, chunk_size, BYTES_PER_THREAD, align_off, aligned_size);

  for (int64_t off = align_off + thread_idx * BYTES_PER_THREAD;
       off < align_off + aligned_size;
       off += num_threads * BYTES_PER_THREAD) {
    uint4 val;
    // Oppurtunistically vectorize reads
    load128(val, &src[off]);
    stream_store128(&dst[off], val);
  }

  // Handle unaligned regions
  if (thread_idx < align_off && thread_idx < chunk_size) {
    dst[thread_idx] = src[thread_idx];
  }
  if (align_off + aligned_size + thread_idx < chunk_size) {
    dst[align_off + aligned_size + thread_idx] =
        src[align_off + aligned_size + thread_idx];
  }
}

// Calculate the base addr for each split.
static inline std::vector<int64_t> get_split_base_addrs(
    const at::Tensor& tensor,
    at::IntArrayRef split_sizes,
    int64_t dim) {
  const auto* data_ptr = static_cast<const char*>(tensor.const_data_ptr());
  const auto strides = tensor.strides();
  const auto element_sz = tensor.element_size();
  int64_t off = 0;
  std::vector<int64_t> split_base_addrs;
  split_base_addrs.reserve(split_sizes.size());
  for (const auto& split_size : split_sizes) {
    split_base_addrs.push_back(reinterpret_cast<int64_t>(data_ptr + off));
    off += split_size * strides[dim] * element_sz;
  }
  return split_base_addrs;
}

static inline std::vector<int64_t> get_dst_addrs(at::TensorList out) {
  std::vector<int64_t> addrs;
  addrs.reserve(out.size());
  for (const auto& tensor : out) {
    addrs.push_back(reinterpret_cast<int64_t>(tensor.data_ptr()));
  }
  return addrs;
}

// Calculate the chunk size for each split in bytes.
static inline std::vector<int64_t> get_split_chunk_sizes(
    const at::Tensor& tensor,
    at::IntArrayRef split_sizes,
    int64_t dim) {
  const auto stride = tensor.stride(dim);
  const auto element_sz = tensor.element_size();
  std::vector<int64_t> split_chunk_sizes;
  split_chunk_sizes.reserve(split_sizes.size());
  for (const auto& split_size : split_sizes) {
    split_chunk_sizes.push_back(split_size * stride * element_sz);
  }
  return split_chunk_sizes;
}

// Calculate the chunk stride in bytes. This is the same for all splits.
static inline int64_t get_chunk_stride(const at::Tensor& tensor, int64_t dim) {
  int64_t stride = 1;
  for (int64_t d = dim; d < tensor.dim(); ++d) {
    stride *= tensor.sizes()[d];
  }
  return stride * tensor.element_size();
}

// Calculate the number of chunks. This is the same for all splits.
static inline int64_t get_num_chunks(const at::Tensor& tensor, int64_t dim) {
  int64_t num_chunks = tensor.numel();
  for (int64_t d = dim; d < tensor.dim(); ++d) {
    num_chunks /= tensor.sizes()[d];
  }
  return num_chunks;
}

static inline std::vector<int64_t> get_chunk_cat_out_sizes(
    IntArrayRef input_tensor_sizes,
    int64_t dim,
    int64_t num_chunks,
    int64_t chunk_size,
    int64_t out_element_size) {
  std::vector<int64_t> view_sizes = std::vector<int64_t>(
      input_tensor_sizes.begin(), input_tensor_sizes.begin() + dim);
  view_sizes.insert(
      view_sizes.end(), {num_chunks, chunk_size / out_element_size});
  return view_sizes;
}

// Copy `max_chunk_size` bytes from `src` to `dst` by `num_threads`, and pad
// zero when `src` size (i.e., actual_chunk_size) is less than `max_chunk_size`.
// Assume elements of src and dst have the same data type.
template <typename dst_t, typename src_t>
inline void copy_chunk_with_pad(
    dst_t* dst_ptr,
    src_t* src_ptr,
    int64_t max_chunk_size,
    int64_t actual_chunk_size,
    int64_t thread_idx,
    int64_t num_threads) {
  // Supports type cast
  if (!std::is_same_v<dst_t, src_t>) {
    const int64_t max_num_elems = max_chunk_size / sizeof(dst_t);
    const int64_t actual_num_elems = actual_chunk_size / sizeof(src_t);
    int64_t elem_index = thread_idx;
    while (elem_index < actual_num_elems) {
      dst_ptr[elem_index] =
          static_cast_with_inter_type<dst_t, src_t>::apply(src_ptr[elem_index]);
      elem_index += num_threads;
    }
    while (elem_index < max_num_elems) {
      dst_ptr[elem_index] = static_cast_with_inter_type<dst_t, int>::apply(0);
      elem_index += num_threads;
    }
    return;
  }
  char* dst = reinterpret_cast<char*>(dst_ptr);
  char* src = reinterpret_cast<char*>(src_ptr);
  // Fast path when the number of threads is larger than the number of bytes to
  // be copied (i.e., max_chunk_size). In this case, each thread only copies 1
  // byte. For 0 <= thread_idx < actual_chunk_size, the thread copies data from
  // `src`. For actual_chunk_size <= thread_idx < max_chunk_size, the thread set
  // the val=0 for padding.
  if (max_chunk_size < num_threads) {
    char val = static_cast<char>(0);
    if (thread_idx < actual_chunk_size) {
      val = src[thread_idx];
    }
    if (thread_idx < max_chunk_size) {
      dst[thread_idx] = val;
    }
    return;
  }
  // Split dst array into three parts:
  // [dst, dst+align_off), [dst+align_off, dst+align_end), [dst+align_end,
  // dst+max_chunk_size) The second part is aligned with BYTES_PER_THREAD(=16
  // bytes) to enable `stream_store128`.
  int64_t align_off, aligned_size;
  get_aligned_region(
      dst, actual_chunk_size, BYTES_PER_THREAD, align_off, aligned_size);
  int64_t align_end = align_off + aligned_size;
  for (int64_t i = align_off + thread_idx * BYTES_PER_THREAD; i < align_end;
       i += num_threads * BYTES_PER_THREAD) {
    uint4 val;
    if (is_aligned<uint4>(src + i)) {
      stream_load128(val, src + i);
    } else {
      for (size_t j = 0; j < BYTES_PER_THREAD; ++j) {
        reinterpret_cast<char*>(&val)[j] = src[i + j];
      }
    }
    stream_store128(&dst[i], val);
  }
  // Copy data for the first part of dst array [dst, dst+align_off).
  // Check `thread_idx<max_chunk_sze` for the edge case that max_chunk_size <
  // align_off.
  if (thread_idx < align_off && thread_idx < max_chunk_size) {
    char val = (char)0;
    if (thread_idx < actual_chunk_size) {
      val = src[thread_idx];
    }
    dst[thread_idx] = val;
  }
  // Copy data for the third part of dst array [dst+align_end,
  // dst+max_chunk_size).
  while (align_end + thread_idx < max_chunk_size) {
    char val = (char)0;
    if (align_end + thread_idx < actual_chunk_size) {
      val = src[align_end + thread_idx];
    }
    dst[align_end + thread_idx] = val;
    align_end += num_threads;
  }
}

// Get leading dimensions before `dim`-th dimension.
static inline int64_t get_leading_dim(at::IntArrayRef sizes, int64_t dim) {
  int64_t leading_dim = 1;
  if (dim > 0) {
    leading_dim = c10::multiply_integers(sizes.slice(0, dim));
  }
  return leading_dim;
}

// Get trailing dimensions after `dim`-th dimension and padded size along
// `dim`-th dimension.
static inline std::pair<int64_t, int64_t> get_pad_size(
    at::IntArrayRef sizes,
    int64_t dim,
    int64_t num_chunks) {
  int64_t trailing_numel = 1;
  if (sizes.size() > (uint64_t)dim + 1) {
    trailing_numel =
        c10::multiply_integers(sizes.slice(dim + 1, sizes.size() - dim - 1));
  }
  int64_t pad_size_along_dim = div_up(sizes[dim], num_chunks) * num_chunks;
  return std::make_pair(pad_size_along_dim, trailing_numel);
}

// Get the padded chunk size.
static inline int64_t get_chunk_size(
    TensorList tensors,
    int64_t dim,
    int64_t num_chunks,
    int64_t elem_size) {
  auto num_tensors = tensors.size();
  int64_t chunk_size = 0;
  for (const auto i : c10::irange(num_tensors)) {
    auto [pad_size_along_dim, trailing_numel] =
        get_pad_size(tensors[i].sizes(), dim, num_chunks);
    const int64_t pad_tensor_chunk_size =
        pad_size_along_dim * trailing_numel * elem_size / num_chunks;
    chunk_size += pad_tensor_chunk_size;
  }
  return chunk_size;
}

TORCH_XPU_API void split_with_sizes_copy_out_xpu_kernel(
    const Tensor& self,
    IntArrayRef split_sizes,
    int64_t dim,
    TensorList out);

TORCH_XPU_API Tensor
_chunk_cat_xpu_kernel(TensorList tensors, int64_t dim, int64_t num_chunks);

TORCH_XPU_API Tensor& _chunk_cat_out_xpu_kernel(
    TensorList tensors,
    int64_t dim,
    int64_t num_chunks,
    Tensor& out);

} // namespace at::native::xpu