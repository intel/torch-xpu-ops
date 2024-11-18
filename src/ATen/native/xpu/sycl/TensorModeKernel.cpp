#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/CanUse32BitIndexMath.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/xpu/sycl/TensorModeKernel.h>
#include <ATen/ops/sort.h>
#include <ATen/ops/zeros_like.h>
#include <comm/SYCLContext.h>
#include <comm/SYCLHelpers.h>
#include <comm/TensorInfo.h>

namespace at::native::xpu {

using namespace at::xpu::detail;

constexpr int64_t MAX_GROUP_SIZE = 256;
constexpr int64_t MAX_GRID_SIZE = 65535LL;

// Used for a segmented reduction
struct ModeUnsignedBoolPair {
  unsigned int val;
  bool flag;
};

template <typename integer>
inline integer ceil_div(integer n, integer m) {
  return (n + m - 1) / m;
}

// Returns 2^(ceil(lg(n)) from Stanford bit twiddling hacks
inline uint64_t next_highest_power_of_2(uint64_t n) {
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
#ifndef _MSC_VER
  n |= n >> 32;
#endif
  n++;
  return n;
}

std::tuple<int64_t, int64_t, int64_t> get_workgroup_number_from_tiles(
    int64_t gridTiles) {
  if (gridTiles > MAX_GRID_SIZE * MAX_GRID_SIZE * MAX_GRID_SIZE) {
    TORCH_INTERNAL_ASSERT(false);
  }

  int64_t gridX = gridTiles > MAX_GRID_SIZE ? MAX_GRID_SIZE : gridTiles;
  int64_t gridY = 1;
  int64_t gridZ = 1;

  if (gridTiles > MAX_GRID_SIZE) {
    gridTiles = ceil_div(gridTiles, MAX_GRID_SIZE);
    gridY = gridTiles > MAX_GRID_SIZE ? MAX_GRID_SIZE : gridTiles;

    if (gridTiles > MAX_GRID_SIZE) {
      gridTiles = ceil_div(gridTiles, MAX_GRID_SIZE);
      gridZ = gridTiles > MAX_GRID_SIZE ? MAX_GRID_SIZE : gridTiles;
    }
  }
  return std::make_tuple(gridX, gridY, gridZ);
}

template <typename index_t>
inline index_t get_linear_group_id(sycl::nd_item<3> item) {
  return item.get_group(0) * item.get_group_range(1) * item.get_group_range(2) +
      item.get_group(1) * item.get_group_range(2) + item.get_group(2);
}

template <typename T>
inline void swapVars(T& t1, T& t2) {
  T tmp = t1;
  t1 = t2;
  t2 = tmp;
}

template <typename Comparator, typename K>
inline void bitonicSwapKeys(
    K& kA,
    bool& validA,
    K& kB,
    bool& validB,
    bool dir,
    const Comparator& comp) {
  bool swap = (comp(kA, kB) && validA) || !validB;
  if (swap == dir) {
    swapVars(kA, kB);
    swapVars(validA, validB);
  }
}

template <
    typename K,
    typename IndexType,
    int Power2SortSize,
    typename Comparator>
inline void bitonicSortKeys(
    sycl::nd_item<3> item,
    K keys[Power2SortSize],
    bool valid[Power2SortSize],
    const Comparator& comp) {
  auto tx = item.get_local_id(2);
#pragma unroll
  for (unsigned int size = 2; size < Power2SortSize; size *= 2) {
    bool flag = ((tx & (size / 2)) != 0);
    for (unsigned int stride = size / 2; stride > 0; stride /= 2) {
      item.barrier(sycl_local_fence);
      unsigned int pos = 2 * tx - (tx & (stride - 1));
      bitonicSwapKeys<Comparator, K>(
          keys[pos],
          valid[pos],
          keys[pos + stride],
          valid[pos + stride],
          flag,
          comp);
    }
  }
#pragma unroll
  for (unsigned int stride = Power2SortSize / 2; stride > 0; stride /= 2) {
    item.barrier(sycl_local_fence);
    unsigned int pos = 2 * tx - (tx & (stride - 1));
    bitonicSwapKeys<Comparator, K>(
        keys[pos],
        valid[pos],
        keys[pos + stride],
        valid[pos + stride],
        false,
        comp);
  }
  item.barrier(sycl_local_fence);
}

template <typename T>
struct BitonicSortFn {
  bool operator()(const T& a, const T& b) const {
    return a < b;
  }
};

template <typename T, unsigned int Power2Size>
struct ComputeModeKernelFunctor : public __SYCL_KER_CONFIG_CONVENTION__ {
  [[intel::reqd_sub_group_size(32)]] void operator()(
      sycl::nd_item<3> item) const {
    int tidx = item.get_local_id(2);
    int stidx = item.get_local_range(2) +
        item.get_local_id(2); // Second index this thread responsible for

    // First, we need to calculate the offset into the sorted Tensor that
    // represents the start of the slice for this block to calculate the mode
    // for. This offset is a combination of the gridIndices, and the number of
    // elements in the slice.
    unsigned int groupId = get_linear_group_id<unsigned int>(item);
    unsigned int linearOffset = groupId * sliceSize_;

    if (groupId >= slices_) {
      return;
    }

    // smem represents a proportion of the shared memory buffer that is used to
    // store the elements from the slice:
    T* smem = reinterpret_cast<T*>(
        shmem_.template get_multi_ptr<sycl::access::decorated::no>().get());

    // Each thread loads up to two elements from the Tensor into shared memory
    if (tidx < sliceSize_) {
      smem[tidx] = c10::load(&input_[linearOffset + tidx]);
    }
    if (stidx < sliceSize_) {
      smem[stidx] = c10::load(&input_[linearOffset + stidx]);
    }

    // Next, we initialize a boolean region of the buffer, offset by the loaded
    // element smem region
    bool* bmem = reinterpret_cast<bool*>(&smem[Power2Size]);

    // The first use of this region stores bmem[i] = i < sliceSize to mark the
    // valid components in the smem buffer
    bmem[tidx] = tidx < sliceSize_;
    bmem[stidx] = stidx < sliceSize_;
    item.barrier(sycl_local_fence); // barrier for smem, bmem initialization

    // First, sort the input slice in ascending order. smem contains the input
    // elements, and bmem marks the valid indices
    bitonicSortKeys<T, unsigned int, Power2Size>(
        item, smem, bmem, BitonicSortFn<T>());
    item.barrier(
        sycl_local_fence); // make no assumptions that the sort syncs at end

    // The next step of our algorithm is performing a group-wide comparison of
    // neighboring elements. In particular, given an sorted input slice A, we
    // produce an output slice B, such that B[i] = 1 if A[i-i] != A[i],
    // otherwise 0.
    //
    // Given the input A = [0, 0, 1, 1, 2, 2, 2, 4, 5, 6, 6, 7, 8]
    //                 B = [1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1]
    //
    // In particular, we can think of B[i] true indicating the start of a
    // sequence of equal values in the sorted list. Similarly, we will also
    // store the negation of B, which we'll call C. In particular, we can think
    // of C[i] = true iff A[i-1] == A[i] in our original sorted slice.
    //
    //                 C = [0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0]

    // We overwrite bmem, and treat the rest of shared memory as a buffer of
    // (index, flag) pairs where the index represents values from C, and the
    // flag represents values from B.
    //
    // [smem (sorted slice) | ubpmem (index, flag pairs)]

    struct ModeUnsignedBoolPair* ubpmem =
        reinterpret_cast<struct ModeUnsignedBoolPair*>(&smem[Power2Size]);

    if (tidx == 0) {
      ubpmem[0].flag = true;
      ubpmem[0].val = 0;
    }

    // Compares elements (0, 1), (2, 3), ... and sets 1, 3, ...
    ubpmem[tidx * 2 + 1].flag =
        smem[tidx * 2] != smem[tidx * 2 + 1]; // (0, 1), (1, 2), etc.
    ubpmem[tidx * 2 + 1].val = !ubpmem[tidx * 2 + 1].flag;

    // Compares elements (1, 2), (3, 4), ... and sets 2, 4, ...
    if (((tidx + 1) * 2) < Power2Size) {
      ubpmem[(tidx + 1) * 2].flag =
          smem[((tidx + 1) * 2) - 1] != smem[(tidx + 1) * 2];
      ubpmem[(tidx + 1) * 2].val = !ubpmem[(tidx + 1) * 2].flag;
    }
    item.barrier(sycl_local_fence); // barrier for ubpmem initialization
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    shmem_ = sycl_local_acc_t<char>(memsize_, cgh);
  }

  ComputeModeKernelFunctor(
      const T* input,
      at::xpu::detail::TensorInfo<T, unsigned int> values,
      at::xpu::detail::TensorInfo<int64_t, unsigned int> indices,
      int64_t sliceSize,
      int64_t slices,
      int64_t memsize)
      : input_(input),
        values_(values),
        indices_(indices),
        sliceSize_(sliceSize),
        slices_(slices),
        memsize_(memsize) {}

 private:
  const T* input_;
  at::xpu::detail::TensorInfo<T, unsigned int> values_;
  at::xpu::detail::TensorInfo<int64_t, unsigned int> indices_;
  int64_t sliceSize_;
  int64_t slices_;
  int64_t memsize_;
  sycl_local_acc_t<char> shmem_;
};

template <int64_t size, typename scalar_t>
void handle_fused_mode(
    std::tuple<int64_t, int64_t, int64_t> nwgs,
    const TensorBase& self,
    at::xpu::detail::TensorInfo<scalar_t, unsigned int>& ti_values,
    at::xpu::detail::TensorInfo<int64_t, unsigned int>& ti_indices,
    int64_t slice_size,
    int64_t slices) {
  constexpr int num_threads = size / 2;
  constexpr int sg_size = 32;
  TORCH_INTERNAL_ASSERT(
      num_threads % sg_size == 0 && num_threads <= (sg_size * sg_size), "");
  const auto memsize =
      (sizeof(scalar_t) * size) + (2 * size * sizeof(unsigned int));
  auto gx = std::get<0>(nwgs);
  auto gy = std::get<1>(nwgs);
  auto gz = std::get<2>(nwgs);
  sycl::range<3> local_range(1, 1, num_threads);
  sycl::range<3> global_range(gz, gy, gx * num_threads);
  auto caller = ComputeModeKernelFunctor<scalar_t, size>(
      self.const_data_ptr<scalar_t>(),
      ti_values,
      ti_indices,
      slice_size,
      slices,
      memsize);
  sycl_kernel_submit(global_range, local_range, getCurrentSYCLQueue(), caller);
}

template <typename scalar_t>
void fused_mode(
    const TensorBase& values,
    const TensorBase& indices,
    const TensorBase& self,
    int64_t slice_size,
    int64_t slices) {
  // Set-up TensorInfo structs for passing to kernel
  auto ti_values =
      at::xpu::detail::getTensorInfo<scalar_t, unsigned int>(values);
  auto ti_indices =
      at::xpu::detail::getTensorInfo<int64_t, unsigned int>(indices);

  // The number of work group is the number of slices that we need to calculate
  // the mode for. Each group is responsible for computing a single mode
  auto nwgs = get_workgroup_number_from_tiles(slices);

  // The groupsize is two elements per thread, rounded up to the nearest power
  // of 2
  auto ceilPowerOf2 = next_highest_power_of_2(slice_size);

  // Tradeoff between compilation time and the number of specializations.
  // Ideally we would have one handle_fused_mode for each power of 2
  switch (ceilPowerOf2) {
    case 2048:
      handle_fused_mode<2048, scalar_t>(
          nwgs, self, ti_values, ti_indices, slice_size, slices);
      break;
    case 1024:
    case 512:
    case 256:
      handle_fused_mode<1024, scalar_t>(
          nwgs, self, ti_values, ti_indices, slice_size, slices);
      break;
    case 128:
    case 64:
    case 32:
    case 16:
    case 8:
    case 4:
    case 2:
      handle_fused_mode<128, scalar_t>(
          nwgs, self, ti_values, ti_indices, slice_size, slices);
      break;
    case 1:
    default:
      TORCH_INTERNAL_ASSERT(false);
  }
}

void launch_fused_mode_kernel(
    const TensorBase& values,
    const TensorBase& indices,
    const TensorBase& self,
    int64_t slice_size,
    int64_t slices) {
  AT_DISPATCH_ALL_TYPES_AND3(
      kBool, kBFloat16, kHalf, self.scalar_type(), "xpu_mode", [&] {
        fused_mode<scalar_t>(values, indices, self, slice_size, slices);
      });
}

void mode_kernel(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t dim,
    bool keepdim) {
  auto self_sizes = ensure_nonempty_vec(self.sizes().vec());
  int64_t ndim = ensure_nonempty_dim(self.dim());
  int64_t slice_size = ensure_nonempty_size(self, dim);
  int64_t slices = self.numel() / slice_size;

  // Resize output value, index Tensors to appropriate sizes (i.e. the same as
  // the input Tensor, except at dim=dimension, the size is 1)
  assert(0 <= dim && static_cast<size_t>(dim) < self_sizes.size());
  self_sizes[dim] = 1;

  if (!keepdim) {
    if (values.ndimension() >= dim) {
      values.unsqueeze_(dim);
    }
    if (indices.ndimension() >= dim) {
      indices.unsqueeze_(dim);
    }
  }

  at::native::resize_output(values, self_sizes);
  at::native::resize_output(indices, self_sizes);

  // If sliceSize is 1, copy input to values and set indices
  if (slice_size == 1) {
    values.copy_(self);
    indices.fill_(0);
    if (!keepdim) {
      values.squeeze_(dim);
      indices.squeeze_(dim);
    }
    return;
  }

  // Beginning our optimized implementation. First thing we want to do is to
  // transpose the input Tensor along the sort dimension, and then make it
  // contiguous.
  auto transposed = self.transpose(dim, ndim - 1);
  auto contiguous = transposed.contiguous();

  // We also need to view the values and indices Tensors as transposed in order
  // to properly determine the offset into the underlying storage in which to
  // place the mode and index for a particular set of dimension values.
  auto values_transposed = values.transpose(dim, ndim - 1);
  auto indices_transposed = indices.transpose(dim, ndim - 1);

  // Requirements for fused kernel implementation:
  //
  // 1. sliceSize <= 2 * max threads per block
  // 2. uses one block per slice, so number of slices must be less than the
  // maximum number of blocks for a kernel launch
  // 3. Can use 32-bit index math for indexing (mainly just for implementation
  // conciseness, could be changed)
  //
  if (slice_size <= 2 * MAX_GROUP_SIZE &&
      slices <= MAX_GRID_SIZE * MAX_GRID_SIZE * MAX_GRID_SIZE &&
      canUse32BitIndexMath(self)) {
    launch_fused_mode_kernel(
        values_transposed, indices_transposed, contiguous, slice_size, slices);
  } else {
    if (transposed.is_same(contiguous)) {
      contiguous = contiguous.clone();
    }

    // launch_apply_mode_kernel(
    //     values_transposed, indices_transposed, contiguous, dim, ndim);
  }

  if (!keepdim) {
    values.squeeze_(dim);
    indices.squeeze_(dim);
  }
}

} // namespace at::native::xpu
