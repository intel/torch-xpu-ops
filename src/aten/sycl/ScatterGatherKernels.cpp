#include <ATen/Dispatch.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/native/ScatterGatherChecks.h>
#include <ATen/native/TensorIterator.h>
#include <aten/sycl/Atomics.h>
#include <aten/sycl/OffsetCalculator.h>
#include <comm/SYCLContext.h>

namespace at {
namespace native {
namespace xpu {

class ReduceMultiply {
 public:
  template <typename scalar_t>
  constexpr C10_DEVICE void operator()(
      scalar_t* self_data_start,
      int64_t index,
      int64_t numel,
      const scalar_t* src_data) const {
    atomicMul((sycl_global_ptr<scalar_t>)(self_data_start + index), *src_data);
  }

  template <typename scalar_t>
  constexpr void operator()(scalar_t* self_data, const scalar_t* src_data)
      const {
    atomicMul((sycl_global_ptr<scalar_t>)self_data, *src_data);
  }
};
static ReduceMultiply reduce_multiply;

class ReduceAdd {
 public:
  template <typename scalar_t>
  constexpr C10_DEVICE void operator()(
      scalar_t* self_data_start,
      int64_t index,
      int64_t numel,
      const scalar_t* src_data) const {
    atomicAdd((sycl_global_ptr<scalar_t>)(self_data_start + index), *src_data);
  }

  template <typename scalar_t>
  constexpr void operator()(scalar_t* self_data, const scalar_t* src_data)
      const {
    atomicAdd((sycl_global_ptr<scalar_t>)self_data, *src_data);
  }
};
static ReduceAdd reduce_add;

class ReduceMean {
 public:
  template <typename scalar_t>
  constexpr C10_DEVICE void operator()(
      scalar_t* self_data_start,
      int64_t index,
      int64_t numel,
      const scalar_t* src_data) const {
    atomicAdd((sycl_global_ptr<scalar_t>)(self_data_start + index), *src_data);
  }

  template <typename scalar_t>
  constexpr void operator()(scalar_t* self_data, const scalar_t* src_data)
      const {
    atomicAdd((sycl_global_ptr<scalar_t>)self_data, *src_data);
  }
};
static ReduceMean reduce_mean;

class ReduceMinimum {
 public:
  template <typename scalar_t>
  constexpr C10_DEVICE void operator()(
      scalar_t* self_data_start,
      int64_t index,
      int64_t numel,
      const scalar_t* src_data) const {
    atomicMin((sycl_global_ptr<scalar_t>)(self_data_start + index), *src_data);
  }

  template <typename scalar_t>
  constexpr void operator()(scalar_t* self_data, const scalar_t* src_data)
      const {
    atomicMin((sycl_global_ptr<scalar_t>)self_data, *src_data);
  }
};
static ReduceMinimum reduce_minimum;

class ReduceMaximum {
 public:
  template <typename scalar_t>
  constexpr C10_DEVICE void operator()(
      scalar_t* self_data_start,
      int64_t index,
      int64_t numel,
      const scalar_t* src_data) const {
    atomicMax((sycl_global_ptr<scalar_t>)(self_data_start + index), *src_data);
  }

  template <typename scalar_t>
  constexpr void operator()(scalar_t* self_data, const scalar_t* src_data)
      const {
    atomicMax((sycl_global_ptr<scalar_t>)self_data, *src_data);
  }
};
static ReduceMaximum reduce_maximum;

class TensorAssign {
 public:
  template <typename scalar_t>
  constexpr C10_DEVICE void operator()(
      scalar_t* self_data_start,
      int64_t index,
      int64_t numel,
      const scalar_t* src_data) const {
    *(self_data_start + index) = *src_data;
  }

  template <typename scalar_t>
  constexpr void operator()(scalar_t* self_data, const scalar_t* src_data)
      const {
    *self_data = *src_data;
  }
};
static TensorAssign tensor_assign;

// The kernels are implemented on an opaque,
// self-aligned type of the correct size,
// to avoid redundant kernels for different types
// of the same size.
template <int N>
struct alignas(N) OpaqueType {
  char data[N];
};

template <int work_group_size, int thread_work_size, typename func_t>
struct ScatterGatherElementwiseKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    constexpr int nv = work_group_size * thread_work_size;
    auto wg_id = item.get_group_linear_id();
    auto local_id = item.get_local_linear_id();
    int idx = nv * wg_id + local_id;
#pragma unroll
    for (int i = 0; i < thread_work_size; ++i) {
      if (idx < N_) {
        f_(idx);
        idx += work_group_size;
      }
    }
  }
  ScatterGatherElementwiseKernelFunctor(int N, func_t f) : N_(N), f_(f) {}

 private:
  int N_;
  func_t f_;
};

template <int nt, int vt, typename func_t>
static void launch_scatter_gather_kernel(int64_t N, const func_t& f) {
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }

  sycl::range<1> local_range{(size_t)nt};
  int num_workgroups = (N + nt * vt - 1) / (nt * vt);
  sycl::range<1> global_range{(size_t)(num_workgroups * nt)};

  auto caller =
      ScatterGatherElementwiseKernelFunctor<nt, vt, func_t>((int)N, f);
  sycl_kernel_submit(
      global_range, local_range, at::xpu::getCurrentSYCLQueue(), caller);
}

template <
    bool is_scatter_like,
    typename scalar_t,
    typename offset_calc_t,
    typename func_t>
struct ScatterGatherInternalKernelLoopFunctor {
  void operator()(int i) const {
    auto offsets = offset_calc_.get(i);

    int64_t idx_dim = *(int64_t*)(index_ptr_ + offsets[2]);
    SYCL_KERNEL_ASSERT(
        idx_dim >= 0 && idx_dim < index_size_ && "index out of bounds");

    f_((scalar_t*)(self_ptr_ + offsets[0]),
       is_scatter_like ? idx_dim * index_stride_ : 0,
       numel_,
       (scalar_t*)(src_ptr_ + offsets[1]) +
           (is_scatter_like ? 0 : idx_dim * index_stride_));
  }

  ScatterGatherInternalKernelLoopFunctor(
      char* self_ptr,
      char* src_ptr,
      char* index_ptr,
      offset_calc_t offset_calc,
      int64_t index_size,
      int64_t index_stride,
      int64_t numel,
      func_t f)
      : self_ptr_(self_ptr),
        src_ptr_(src_ptr),
        index_ptr_(index_ptr),
        offset_calc_(offset_calc),
        index_size_(index_size),
        index_stride_(index_stride),
        numel_(numel),
        f_(f) {}

 private:
  char* self_ptr_;
  char* src_ptr_;
  char* index_ptr_;
  offset_calc_t offset_calc_;
  int64_t index_size_;
  int64_t index_stride_;
  int64_t numel_;
  func_t f_;
};

template <bool is_scatter_like, typename scalar_t>
struct ScatterGatherInternalKernel {
  template <typename func_t>
  void operator()(
      TensorIterator& iter,
      int64_t index_size,
      int64_t index_stride,
      int64_t numel,
      func_t f) {
    if (!iter.can_use_32bit_indexing()) {
      for (auto& sub_iter : iter.with_32bit_indexing()) {
        ScatterGatherInternalKernel<is_scatter_like, scalar_t>()(
            sub_iter, index_size, index_stride, numel, f);
      }
      return;
    }

    char* self_ptr = (char*)iter.data_ptr(0);
    char* src_ptr = (char*)iter.data_ptr(1);
    char* index_ptr = (char*)iter.data_ptr(2);

    auto offset_calc = make_offset_calculator<3>(iter);

    auto loop = ScatterGatherInternalKernelLoopFunctor<
        is_scatter_like,
        scalar_t,
        decltype(offset_calc),
        func_t>(
        self_ptr,
        src_ptr,
        index_ptr,
        offset_calc,
        index_size,
        index_stride,
        numel,
        f);

    // TODO: optimize it
    constexpr int group_work_items = 256;
    constexpr int work_size_per_item = 4;
    launch_scatter_gather_kernel<group_work_items, work_size_per_item>(
        iter.numel(), loop);
  }
};

template <bool is_scatter_like = true, bool cast_to_opaque = true>
struct ScatterGatherBaseKernel {
  void operator()(
      const Tensor& self,
      int64_t dim,
      const Tensor& index,
      const Tensor& src,
      const std::string& method_name,
      const ReduceAdd& f) {
    at::assert_no_internal_overlap(self);

    auto index_sizes = ensure_nonempty_vec(index.sizes().vec());
    auto self_strides = ensure_nonempty_vec(self.strides().vec());
    auto src_strides = ensure_nonempty_vec(src.strides().vec());

    // restride self and src such that
    // self.shape = src.shape = index.shape
    //
    // restride stride[dim] such that
    // if (is_scatter_like) self.stride[dim] = 0
    // else src.stride[dim] = 0
    auto self_restrided = is_scatter_like
        ? restride_dim(self, dim, index_sizes)
        : self.as_strided(index_sizes, self_strides);
    auto src_restrided = is_scatter_like
        ? src.as_strided(index_sizes, src_strides)
        : restride_dim(src, dim, index_sizes);

    auto iter = TensorIteratorConfig()
                    .set_check_mem_overlap(false)
                    .check_all_same_dtype(false)
                    .resize_outputs(false)
                    .add_output(self_restrided)
                    .add_const_input(src_restrided)
                    .add_const_input(index)
                    .build();

    auto self_dim_stride = ensure_nonempty_stride(self, dim);
    auto self_dim_size = ensure_nonempty_size(self, dim);

    auto src_dim_stride = ensure_nonempty_stride(src, dim);
    auto src_dim_size = ensure_nonempty_size(src, dim);

    auto index_size = is_scatter_like ? self_dim_size : src_dim_size;
    auto index_stride = is_scatter_like ? self_dim_stride : src_dim_stride;

    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
        at::ScalarType::Half,
        at::ScalarType::Bool,
        at::ScalarType::BFloat16,
        iter.dtype(),
        "scatter_gather_base_kernel_func",
        [&] {
          using dtype = typename std::conditional<
              cast_to_opaque,
              OpaqueType<sizeof(scalar_t)>,
              scalar_t>::type;

          ScatterGatherInternalKernel<is_scatter_like, dtype>()(
              iter, index_size, index_stride, self.numel(), f);
        });
  }

  void operator()(
      const Tensor& self,
      int64_t dim,
      const Tensor& index,
      const Tensor& src,
      const std::string& method_name,
      const TensorAssign& f) {
    at::assert_no_internal_overlap(self);

    auto index_sizes = ensure_nonempty_vec(index.sizes().vec());
    auto self_strides = ensure_nonempty_vec(self.strides().vec());
    auto src_strides = ensure_nonempty_vec(src.strides().vec());

    // restride self and src such that
    // self.shape = src.shape = index.shape
    //
    // restride stride[dim] such that
    // if (is_scatter_like) self.stride[dim] = 0
    // else src.stride[dim] = 0
    auto self_restrided = is_scatter_like
        ? restride_dim(self, dim, index_sizes)
        : self.as_strided(index_sizes, self_strides);
    auto src_restrided = is_scatter_like
        ? src.as_strided(index_sizes, src_strides)
        : restride_dim(src, dim, index_sizes);

    auto iter = TensorIteratorConfig()
                    .set_check_mem_overlap(false)
                    .check_all_same_dtype(false)
                    .resize_outputs(false)
                    .add_output(self_restrided)
                    .add_const_input(src_restrided)
                    .add_const_input(index)
                    .build();

    auto self_dim_stride = ensure_nonempty_stride(self, dim);
    auto self_dim_size = ensure_nonempty_size(self, dim);

    auto src_dim_stride = ensure_nonempty_stride(src, dim);
    auto src_dim_size = ensure_nonempty_size(src, dim);

    auto index_size = is_scatter_like ? self_dim_size : src_dim_size;
    auto index_stride = is_scatter_like ? self_dim_stride : src_dim_stride;

    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
        at::ScalarType::Half,
        at::ScalarType::Bool,
        at::ScalarType::BFloat16,
        iter.dtype(),
        "scatter_gather_base_kernel_func",
        [&] {
          using dtype = typename std::conditional<
              cast_to_opaque,
              OpaqueType<sizeof(scalar_t)>,
              scalar_t>::type;

          ScatterGatherInternalKernel<is_scatter_like, dtype>()(
              iter, index_size, index_stride, self.numel(), f);
        });
  }

  template <typename func_t>
  void operator()(
      const Tensor& self,
      int64_t dim,
      const Tensor& index,
      const Tensor& src,
      const std::string& method_name,
      const func_t& f) {
    at::assert_no_internal_overlap(self);

    auto index_sizes = ensure_nonempty_vec(index.sizes().vec());
    auto self_strides = ensure_nonempty_vec(self.strides().vec());
    auto src_strides = ensure_nonempty_vec(src.strides().vec());

    // restride self and src such that
    // self.shape = src.shape = index.shape
    //
    // restride stride[dim] such that
    // if (is_scatter_like) self.stride[dim] = 0
    // else src.stride[dim] = 0
    auto self_restrided = is_scatter_like
        ? restride_dim(self, dim, index_sizes)
        : self.as_strided(index_sizes, self_strides);
    auto src_restrided = is_scatter_like
        ? src.as_strided(index_sizes, src_strides)
        : restride_dim(src, dim, index_sizes);

    auto iter = TensorIteratorConfig()
                    .set_check_mem_overlap(false)
                    .check_all_same_dtype(false)
                    .resize_outputs(false)
                    .add_output(self_restrided)
                    .add_const_input(src_restrided)
                    .add_const_input(index)
                    .build();

    auto self_dim_stride = ensure_nonempty_stride(self, dim);
    auto self_dim_size = ensure_nonempty_size(self, dim);

    auto src_dim_stride = ensure_nonempty_stride(src, dim);
    auto src_dim_size = ensure_nonempty_size(src, dim);

    auto index_size = is_scatter_like ? self_dim_size : src_dim_size;
    auto index_stride = is_scatter_like ? self_dim_stride : src_dim_stride;

    AT_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        iter.dtype(),
        "scatter_gather_base_kernel_func",
        [&] {
          using dtype = typename std::conditional<
              cast_to_opaque,
              OpaqueType<sizeof(scalar_t)>,
              scalar_t>::type;

          ScatterGatherInternalKernel<is_scatter_like, dtype>()(
              iter, index_size, index_stride, self.numel(), f);
        });
  }
};

} // namespace xpu
} // namespace native
} // namespace at
