#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorShape.h>
#include <c10/util/TypeCast.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_chunk_cat_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/split_with_sizes_copy_native.h>
#endif

#include <ATen/native/xpu/sycl/Philox4x32.h>
#include <ATen/native/xpu/sycl/TensorShapeKernels.h>

namespace at::native::xpu {

static constexpr int64_t GROUP_SIZE = 128;
static constexpr int64_t BYTES_PER_THREAD = 16;
static constexpr int64_t BYTES_PER_GROUP = BYTES_PER_THREAD * GROUP_SIZE;

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

// NOTE [Fast path for split_with_sizes_copy.out]
// split_with_sizes_copy.out for contiguous operands has the following
// properties:
// - Each src split consists of multiple chunks that are separated by a fixed
// stride. The number of chunks and the strides are the same across all src
// splits.
// - Each dst split is the concatenation of the chunks in its corresponding src
// splits.
// - The sizes of chunks vary across splits.
// - A (src, dst) chunk pair is not guaranteed to have the
// same alignment.
//
// The following strategies are employed to optimize for this workload:
// - The entire workload is fused into a single kernel to maximize I/O
// throughput and minimize wave quantization.
// - To account for both small and large chunk sizes, a "jagged grid" is used.
// Each chunk is processed by one or more groups depending on its size.
// - Within each chunk, the region in which writes can be vectorized is
// identified. Within this region, writes are always vectorized and reads are
// oppurtunistically vectorized.

bool all_contiguous(TensorList tensors) {
  bool contiguous = true;
  for (const auto& t : tensors) {
    contiguous &= t.is_non_overlapping_and_dense();
  }
  return contiguous;
}

struct SplitWithSizesCopyOutContiguousNoCastFunctor {
  void operator()(sycl::nd_item<3> item) const {
    const int64_t split_idx = group_idx_to_split_idx[item.get_group(2)];
    const int64_t split_groups =
        groups_cumsums[split_idx + 1] - groups_cumsums[split_idx];
    const int64_t split_threads = split_groups * item.get_local_range(2);
    const int64_t split_thread_idx =
        (item.get_group(2) - groups_cumsums[split_idx]) *
            item.get_local_range(2) +
        item.get_local_id(2);
    const int64_t split_chunk_size = split_chunk_sizes[split_idx];

    char* dst_base_addr = dst_base_addrs[split_idx];
    char* src_base_addr = src_base_addrs[split_idx];

    for (int64_t i = item.get_group(1); i < num_chunks;
         i += item.get_group_range(1)) {
      copy_chunk(
          dst_base_addr + i * split_chunk_size,
          src_base_addr + i * src_stride,
          split_chunk_size,
          split_thread_idx,
          split_threads);
    }
  }
  SplitWithSizesCopyOutContiguousNoCastFunctor(
      char** dst_base_addrs,
      char** src_base_addrs,
      int64_t* split_chunk_sizes,
      int64_t* group_idx_to_split_idx,
      int64_t* groups_cumsums,
      int64_t src_stride,
      int64_t num_chunks)
      : dst_base_addrs(dst_base_addrs),
        src_base_addrs(src_base_addrs),
        split_chunk_sizes(split_chunk_sizes),
        group_idx_to_split_idx(group_idx_to_split_idx),
        groups_cumsums(groups_cumsums),
        src_stride(src_stride),
        num_chunks(num_chunks) {}

 private:
  char** dst_base_addrs;
  char** src_base_addrs;
  int64_t* split_chunk_sizes;
  int64_t* group_idx_to_split_idx;
  int64_t* groups_cumsums;
  int64_t src_stride;
  int64_t num_chunks;
};

// Pack multiple std::vector<int64_t> into a single tensor.
std::pair<at::Tensor, std::vector<int64_t*>> pack_vecs(
    std::vector<const std::vector<int64_t>*> vecs,
    const at::Device& device) {
  int64_t numel = 0;
  for (const auto* vec : vecs) {
    numel += vec->size();
  }

  auto packed = at::empty(
      {numel}, at::TensorOptions().dtype(at::kLong).pinned_memory(true));
  size_t offset = 0;
  for (const auto* vec : vecs) {
    memcpy(
        packed.data_ptr<int64_t>() + offset,
        vec->data(),
        sizeof(int64_t) * vec->size());
    offset += vec->size();
  }
  packed = packed.to(device, /*non_blocking=*/true);

  std::vector<int64_t*> ptrs;
  ptrs.reserve(vecs.size());
  offset = 0;
  for (const auto* vec : vecs) {
    ptrs.push_back(packed.data_ptr<int64_t>() + offset);
    offset += vec->size();
  }
  return std::make_pair(std::move(packed), std::move(ptrs));
}

template <typename dst_t, typename src_t>
struct ChunkCatFunctor {
  void operator()(sycl::nd_item<3> item) const {
    const int64_t slice_idx = item.get_group(0);
    const int64_t chunk_idx = item.get_group(1);
    const int64_t tensor_idx = group_idx_to_tensor_idx[item.get_group(2)];
    const int64_t tile_idx =
        item.get_group(2) - start_group_idx_per_tensor_chunk[tensor_idx];
    // Number of threads for the `tensor_idx`-th tensor chunk.
    const int64_t num_threads =
        num_groups_per_tensor_chunk[tensor_idx] * GROUP_SIZE;
    const int64_t thread_idx = tile_idx * GROUP_SIZE + item.get_local_id(2);
    char* src_addr = reinterpret_cast<char**>(src)[tensor_idx] +
        slice_idx * actual_tensor_sizes[tensor_idx] +
        chunk_idx * pad_tensor_chunk_sizes[tensor_idx] / dst_to_src_ratio;
    char* dst_addr = reinterpret_cast<char*>(dst) + slice_idx * slice_size +
        chunk_idx * chunk_size + tensor_idx_to_start_tensor_bytes[tensor_idx];
    // Compute the actual number of bytes to copy from src.
    const int64_t actual_copy_size = std::min(
        pad_tensor_chunk_sizes[tensor_idx] / dst_to_src_ratio,
        std::max(
            (int64_t)0,
            actual_tensor_sizes[tensor_idx] -
                chunk_idx * pad_tensor_chunk_sizes[tensor_idx] /
                    dst_to_src_ratio));
    copy_chunk_with_pad<dst_t, src_t>(
        reinterpret_cast<dst_t*>(dst_addr),
        reinterpret_cast<src_t*>(src_addr),
        pad_tensor_chunk_sizes[tensor_idx],
        actual_copy_size,
        thread_idx,
        num_threads);
  }

  ChunkCatFunctor(
      src_t** src,
      dst_t* dst,
      int64_t* group_idx_to_tensor_idx,
      int64_t* tensor_idx_to_start_tensor_bytes,
      int64_t* start_group_idx_per_tensor_chunk,
      int64_t* actual_tensor_sizes,
      int64_t* pad_tensor_chunk_sizes,
      int64_t* num_groups_per_tensor_chunk,
      int64_t slice_size,
      int64_t chunk_size,
      int64_t dst_to_src_ratio)
      : src(src),
        dst(dst),
        group_idx_to_tensor_idx(group_idx_to_tensor_idx),
        tensor_idx_to_start_tensor_bytes(tensor_idx_to_start_tensor_bytes),
        start_group_idx_per_tensor_chunk(start_group_idx_per_tensor_chunk),
        actual_tensor_sizes(actual_tensor_sizes),
        pad_tensor_chunk_sizes(pad_tensor_chunk_sizes),
        num_groups_per_tensor_chunk(num_groups_per_tensor_chunk),
        slice_size(slice_size),
        chunk_size(chunk_size),
        dst_to_src_ratio(dst_to_src_ratio) {}

 private:
  src_t** src;
  dst_t* dst;
  int64_t* group_idx_to_tensor_idx;
  int64_t* tensor_idx_to_start_tensor_bytes;
  int64_t* start_group_idx_per_tensor_chunk;
  int64_t* actual_tensor_sizes;
  int64_t* pad_tensor_chunk_sizes;
  int64_t* num_groups_per_tensor_chunk;
  int64_t slice_size;
  int64_t chunk_size;
  int64_t dst_to_src_ratio;
};

// Get metadata for chunk_cat.
std::tuple<
    int64_t,
    int64_t,
    int64_t,
    int64_t,
    std::vector<int64_t>,
    std::vector<int64_t>,
    std::vector<int64_t>,
    std::vector<int64_t>,
    std::vector<int64_t>,
    std::vector<int64_t>,
    std::vector<int64_t>>
get_chunk_cat_metadata(
    TensorList tensors,
    int64_t dim,
    int64_t num_chunks,
    int64_t dst_elem_size,
    int64_t src_elem_size) {
  TORCH_CHECK(
      dst_elem_size % src_elem_size == 0,
      "get_chunk_cat_metadata error: only support dst_elem_size % src_elem_size == 0");
  auto num_tensors = tensors.size();
  int64_t leading_dim = get_leading_dim(tensors[0].sizes(), dim);
  std::vector<int64_t> pad_tensor_chunk_sizes;
  std::vector<int64_t> num_groups_per_tensor_chunk;
  std::vector<int64_t> start_group_idx_per_tensor_chunk{0};
  std::vector<int64_t> actual_tensor_sizes;
  std::vector<int64_t> tensor_idx_to_start_tensor_bytes{0};
  std::vector<int64_t> srcs;
  pad_tensor_chunk_sizes.reserve(num_tensors);
  num_groups_per_tensor_chunk.reserve(num_tensors);
  start_group_idx_per_tensor_chunk.reserve(num_tensors + 1);
  actual_tensor_sizes.reserve(num_tensors);
  tensor_idx_to_start_tensor_bytes.reserve(num_tensors + 1);
  srcs.reserve(num_tensors);
  // group_idx_to_tensor_idx cannot be reserved since the number of groups is
  // data dependent
  std::vector<int64_t> group_idx_to_tensor_idx;
  // Inline computing `chunk_size` to avoid redundant computation
  int64_t chunk_size = 0;
  for (const auto i : c10::irange(num_tensors)) {
    at::Tensor tensor = tensors[i];
    srcs.push_back(reinterpret_cast<int64_t>(tensor.data_ptr()));
    auto sizes = tensor.sizes();
    auto [pad_size_along_dim, trailing_numel] =
        get_pad_size(sizes, dim, num_chunks);
    const int64_t pad_tensor_chunk_size =
        pad_size_along_dim * trailing_numel * dst_elem_size / num_chunks;
    pad_tensor_chunk_sizes.push_back(pad_tensor_chunk_size);
    chunk_size += pad_tensor_chunk_size;
    // Number of groups required to process this tensor chunk.
    const int64_t num_groups = div_up(pad_tensor_chunk_size, BYTES_PER_GROUP);
    num_groups_per_tensor_chunk.push_back(num_groups);

    start_group_idx_per_tensor_chunk.push_back(
        start_group_idx_per_tensor_chunk.back() + num_groups);
    group_idx_to_tensor_idx.insert(
        group_idx_to_tensor_idx.end(), num_groups, i);
    tensor_idx_to_start_tensor_bytes.push_back(
        tensor_idx_to_start_tensor_bytes.back() + pad_tensor_chunk_size);
    actual_tensor_sizes.push_back(sizes[dim] * trailing_numel * src_elem_size);
  }

  const int64_t num_groups_per_chunk = start_group_idx_per_tensor_chunk.back();
  const int64_t slice_size = num_chunks * chunk_size;
  return std::make_tuple(
      chunk_size,
      leading_dim,
      num_groups_per_chunk,
      slice_size,
      srcs,
      group_idx_to_tensor_idx,
      tensor_idx_to_start_tensor_bytes,
      start_group_idx_per_tensor_chunk,
      actual_tensor_sizes,
      pad_tensor_chunk_sizes,
      num_groups_per_tensor_chunk);
}

template <typename dst_t, typename src_t>
void _chunk_cat_out_xpu_contiguous(
    TensorList tensors,
    int64_t dim,
    int64_t num_chunks,
    Tensor& out,
    int64_t dst_elem_size,
    int64_t src_elem_size) {
  const auto device = tensors[0].device();

  // `get_chunk_cat_metadata` must return vectors and `pack_vecs` cannot be
  // moved into `get_chunk_cat_metadata`. Otherwise `packed` would point to
  // vectors allocated inside `get_chunk_cat_metadata` which become out of local
  // scope.
  auto
      [chunk_size,
       leading_dim,
       num_groups_per_chunk,
       slice_size,
       srcs,
       group_idx_to_tensor_idx,
       tensor_idx_to_start_tensor_bytes,
       start_group_idx_per_tensor_chunk,
       actual_tensor_sizes,
       pad_tensor_chunk_sizes,
       num_groups_per_tensor_chunk] =
          get_chunk_cat_metadata(
              tensors, dim, num_chunks, dst_elem_size, src_elem_size);
  auto packed = pack_vecs(
      {&srcs,
       &group_idx_to_tensor_idx,
       &tensor_idx_to_start_tensor_bytes,
       &start_group_idx_per_tensor_chunk,
       &actual_tensor_sizes,
       &pad_tensor_chunk_sizes,
       &num_groups_per_tensor_chunk},
      device);
  std::vector<int64_t> view_sizes = get_chunk_cat_out_sizes(
      tensors[0].sizes(), dim, num_chunks, chunk_size, dst_elem_size);
  at::native::resize_output(out, view_sizes);

  sycl::range<3> global_range{
      (size_t)(leading_dim),
      (size_t)(num_chunks),
      (size_t)(num_groups_per_chunk * GROUP_SIZE)};
  sycl::range<3> local_range{(size_t)1, (size_t)1, (size_t)GROUP_SIZE};

  ChunkCatFunctor<dst_t, src_t> kfn(
      /*srcs=*/reinterpret_cast<src_t**>(packed.second[0]),
      reinterpret_cast<dst_t*>(out.data_ptr()),
      /*group_idx_to_tensor_idx=*/packed.second[1],
      /*tensor_idx_to_start_tensor_bytes=*/packed.second[2],
      /*start_group_idx_per_tensor_chunk=*/packed.second[3],
      /*actual_tensor_sizes=*/packed.second[4],
      /*pad_tensor_chunk_sizes=*/packed.second[5],
      /*num_groups_per_tensor_chunk=*/packed.second[6],
      slice_size,
      chunk_size,
      dst_elem_size / src_elem_size);

  sycl_kernel_submit(global_range, local_range, getCurrentSYCLQueue(), kfn);
}

void split_with_sizes_copy_out_xpu_contiguous_no_cast(
    const at::Tensor& self,
    at::IntArrayRef split_sizes,
    int64_t dim,
    at::TensorList out) {
  const auto device = self.device();
  const auto src_base_addrs = get_split_base_addrs(self, split_sizes, dim);
  const auto dst_base_addrs = get_dst_addrs(out);
  const auto src_stride = get_chunk_stride(self, dim);
  const auto split_chunk_sizes = get_split_chunk_sizes(self, split_sizes, dim);
  const auto num_chunks = get_num_chunks(self, dim);

  // Calculate the number of groups required for the first chunk across all
  // splits, assuming each thread only processes BYTES_PER_THREAD bytes.
  int64_t num_groups = 0;
  for (const auto& split_chunk_size : split_chunk_sizes) {
    num_groups += div_up(split_chunk_size, GROUP_SIZE * BYTES_PER_THREAD);
  }

  auto dev_id = getDeviceIndexOfCurrentQueue();
  int64_t tile_size = syclMaxWorkItemsPerTile(dev_id);
  const int64_t max_groups = tile_size / GROUP_SIZE * 2.0;

  // Make each thread process BYTES_PER_THREAD * iter_factor bytes to regulate
  // group size. Spread iter_factor evenly between chunks_per_group and
  // iters_per_chunk.
  int64_t iter_factor = div_up(num_groups * num_chunks, max_groups);
  int64_t chunks_per_group = std::ceil(std::sqrt(iter_factor));
  chunks_per_group = std::min(chunks_per_group, num_chunks);
  const int64_t iters_per_chunk = div_up(iter_factor, chunks_per_group);

  // Launch a logically jagged grid of shape
  // (chunk_size*, num_splits, num_chunks / chunks_per_group)
  // backed by a physical grid of shape
  // (sum(chunk_size), num_chunks / chunks_per_group).
  // A group can find its split_idx via group_idx_to_split_idx.
  std::vector<int64_t> group_idx_to_split_idx;
  std::vector<int64_t> groups_cumsums{0};
  group_idx_to_split_idx.reserve(num_groups);
  for (size_t split_idx = 0; split_idx < split_sizes.size(); ++split_idx) {
    const auto groups = div_up(
        split_chunk_sizes[split_idx],
        GROUP_SIZE * BYTES_PER_THREAD * iters_per_chunk);
    group_idx_to_split_idx.insert(
        group_idx_to_split_idx.end(), groups, split_idx);
    groups_cumsums.push_back(groups_cumsums.back() + groups);
  }

  sycl::range<3> global_range{
      (size_t)(1),
      (size_t)(num_chunks / chunks_per_group),
      (size_t)groups_cumsums.back() * GROUP_SIZE};
  sycl::range<3> local_range{(size_t)1, (size_t)1, (size_t)GROUP_SIZE};

  auto [_, ptrs] = pack_vecs(
      {&dst_base_addrs,
       &src_base_addrs,
       &split_chunk_sizes,
       &group_idx_to_split_idx,
       &groups_cumsums},
      device);

  SplitWithSizesCopyOutContiguousNoCastFunctor kfn(
      /*dst_base_addrs=*/reinterpret_cast<char**>(ptrs[0]),
      /*src_base_addrs=*/reinterpret_cast<char**>(ptrs[1]),
      /*split_chunk_sizes=*/ptrs[2],
      /*group_idx_to_split_idx=*/ptrs[3],
      /*groups_cumsums=*/ptrs[4],
      src_stride,
      num_chunks);

  sycl_kernel_submit(global_range, local_range, getCurrentSYCLQueue(), kfn);
}

void split_with_sizes_copy_out_xpu_kernel(
    const Tensor& self,
    IntArrayRef split_sizes,
    int64_t dim,
    TensorList out) {
  bool contiguous_no_cast = self.is_non_overlapping_and_dense();
  for (const auto& t : out) {
    contiguous_no_cast &= t.is_non_overlapping_and_dense();
    contiguous_no_cast &= (t.dtype() == self.dtype());
  }

  if (contiguous_no_cast) {
    // Perform equivalent checks performed by the composite impl
    if (dim < 0) {
      dim = at::maybe_wrap_dim(dim, self.dim());
    }
    TORCH_CHECK(
        self.dim() != 0, "split expects at least a 1-dimensional tensor")

    const int64_t dim_size = self.size(dim);
    int64_t split_sizes_sum = 0;
    for (const auto i : c10::irange(split_sizes.size())) {
      TORCH_CHECK(
          split_sizes[i] >= 0,
          "split_with_sizes expects split_sizes have only non-negative ",
          "entries, but got split_sizes=",
          split_sizes[i]);
      split_sizes_sum += split_sizes[i];
    }
    TORCH_CHECK(
        split_sizes_sum == dim_size,
        "split_with_sizes expects split_sizes to sum exactly to ",
        dim_size,
        " (input tensor's size at dimension ",
        dim,
        "), ",
        "but got split_sizes=",
        split_sizes);

    TORCH_CHECK(
        out.size() == split_sizes.size(),
        "split_with_sizes_copy_out() expected an out= argument of size ",
        split_sizes.size(),
        ", got size ",
        out.size());

    auto out_shape = self.sizes().vec();
    for (const auto i : c10::irange(split_sizes.size())) {
      out_shape[dim] = split_sizes[i];
      if (resize_output_check(out[i], out_shape)) {
        out[i].resize_(out_shape);
      }
      TORCH_CHECK(
          out[i].dtype() == self.dtype(),
          "Expected out tensor to have dtype ",
          self.dtype(),
          ", but got ",
          out[i].dtype(),
          " instead");
      TORCH_CHECK(
          out[i].device() == self.device(),
          "Expected out tensor to have device ",
          self.device(),
          ", but got ",
          out[i].device(),
          " instead");
    }
    split_with_sizes_copy_out_xpu_contiguous_no_cast(
        self, split_sizes, dim, out);
  } else {
    at::native::split_with_sizes_copy_out(self, split_sizes, dim, out);
  }
}

Tensor _chunk_cat_xpu_kernel(
    TensorList tensors,
    int64_t dim,
    int64_t num_chunks) {
  dim = at::native::preprocess_chunk_cat_inputs(tensors, dim, num_chunks);
  if (all_contiguous(tensors)) {
    // Return a tensor with the same dtype as input tensors
    int64_t elem_size = tensors[0].element_size();
    int64_t chunk_size = get_chunk_size(tensors, dim, num_chunks, elem_size);
    int64_t leading_dim = get_leading_dim(tensors[0].sizes(), dim);
    auto view_sizes = get_chunk_cat_out_sizes(
        tensors[0].sizes(), dim, num_chunks, chunk_size, elem_size);
    Tensor out =
        tensors[0]
            .new_empty(chunk_size * num_chunks * leading_dim / elem_size)
            .view(view_sizes);
    // Type-agnostic copy since out and input tensors have the same type.
    _chunk_cat_out_xpu_contiguous<char, char>(
        tensors, dim, num_chunks, out, elem_size, elem_size);
    return out;
  } else {
    return at::native::_chunk_cat(tensors, dim, num_chunks);
  }
}

Tensor& _chunk_cat_out_xpu_kernel(
    TensorList tensors,
    int64_t dim,
    int64_t num_chunks,
    Tensor& out) {
  dim = at::native::preprocess_chunk_cat_inputs(tensors, dim, num_chunks);
  TORCH_CHECK(
      tensors[0].device() == out.device(),
      "_chunk_cat_out_xpu: mismatch between input and out tensor devices");
  bool both_input_output_contiguous =
      all_contiguous(tensors) && out.is_non_overlapping_and_dense();
  if (both_input_output_contiguous &&
      (tensors[0].dtype() == at::ScalarType::BFloat16) &&
      (out.dtype() == at::ScalarType::Float)) {
    // _chunk_cat_out_xpu_contiguous should also support other types, thanks to
    // static_cast_with_inter_type. Here, we dispatch to BFloat16 in and float32
    // out since it is the only known use case.
    _chunk_cat_out_xpu_contiguous<float, BFloat16>(
        tensors,
        dim,
        num_chunks,
        out,
        out.element_size(),
        tensors[0].element_size());
  } else if (
      both_input_output_contiguous && tensors[0].dtype() == out.dtype()) {
    // Type-agnostic copy since out and input tensors have the same type.
    _chunk_cat_out_xpu_contiguous<char, char>(
        tensors,
        dim,
        num_chunks,
        out,
        out.element_size(),
        tensors[0].element_size());
  } else {
    at::native::_chunk_cat_out(tensors, dim, num_chunks, out);
  }
  return out;
}

} // namespace at::native::xpu
