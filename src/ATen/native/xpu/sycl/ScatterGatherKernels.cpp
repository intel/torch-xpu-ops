#pragma clang diagnostic push
#pragma GCC diagnostic push
// Avoid SYCL compiler return-type error
#pragma clang diagnostic ignored "-Wreturn-type"
#pragma GCC diagnostic ignored "-Wreturn-type"

#include <ATen/Dispatch.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/ReductionType.h>
#include <ATen/native/ScatterGatherChecks.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/Atomics.h>
#include <ATen/native/xpu/sycl/OffsetCalculator.h>
#include <comm/SYCLContext.h>

#include <ATen/native/xpu/sycl/ScatterGatherKernels.h>

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

template <typename func_t>
struct ScatterGatherElementwiseKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    int nv = work_group_size_ * thread_work_size_;
    auto wg_id = item.get_group_linear_id();
    auto local_id = item.get_local_linear_id();
    int idx = nv * wg_id + local_id;
    for (int i = 0; i < thread_work_size_; ++i) {
      if (idx < N_) {
        f_(idx);
        idx += work_group_size_;
      }
    }
  }
  ScatterGatherElementwiseKernelFunctor(
      int N,
      func_t f,
      int work_group_size,
      int thread_work_size)
      : N_(N),
        f_(f),
        work_group_size_(work_group_size),
        thread_work_size_(thread_work_size) {}

 private:
  int N_;
  func_t f_;
  int work_group_size_;
  int thread_work_size_;
};

template <typename func_t>
static void launch_scatter_gather_kernel(int64_t N, const func_t& f) {
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }

  using KernelFn = ScatterGatherElementwiseKernelFunctor<func_t>;
  int64_t max_wg_size = syclMaxWorkGroupSize<KernelFn>();
  int outputSize = N;
  int work_group_size = outputSize > max_wg_size ? max_wg_size : outputSize;
  const auto target_global_size = syclMaxWorkItemsPerTile();
  // Each work group size is work_group_size, one full device launch is
  // target_global_size, so we can calculate max work group num as below
  const int max_work_group_num = target_global_size / work_group_size;
  int work_group_num = outputSize / work_group_size < max_work_group_num
      ? outputSize / work_group_size
      : max_work_group_num;
  int draft_work_group_num =
      (outputSize + work_group_size - 1) / work_group_size;

  int thread_work_size = draft_work_group_num / work_group_num + 1;

  sycl::range<1> local_range(work_group_size);
  sycl::range<1> global_range(work_group_num * work_group_size);

  auto caller = KernelFn((int)N, f, work_group_size, thread_work_size);
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

    launch_scatter_gather_kernel(iter.numel(), loop);
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

template <typename scalar_t, typename offset_calc_t, typename func_t>
struct ScatterFillInternalKernelLoopFunctor {
  void operator()(int i) const {
    auto offsets = offset_calc_.get(i);
    int64_t idx_dim = *(int64_t*)(index_ptr_ + offsets[1]);
    char* self_data = self_ptr_ + offsets[0];
    f_((scalar_t*)self_data + idx_dim * index_stride_, (scalar_t*)&src_val_);
  }
  ScatterFillInternalKernelLoopFunctor(
      char* self_ptr,
      char* index_ptr,
      offset_calc_t offset_calc,
      int64_t index_stride,
      func_t f,
      scalar_t src_val)
      : self_ptr_(self_ptr),
        index_ptr_(index_ptr),
        offset_calc_(offset_calc),
        index_stride_(index_stride),
        f_(f),
        src_val_(src_val) {}

 private:
  char* self_ptr_;
  char* index_ptr_;
  offset_calc_t offset_calc_;
  int64_t index_stride_;
  func_t f_;
  scalar_t src_val_;
};

template <typename scalar_t>
struct ScatterFillInternalKernel {
  template <typename func_t>
  void operator()(
      TensorIterator& iter,
      scalar_t src_val,
      int64_t index_size,
      int64_t index_stride,
      int64_t numel,
      const func_t& f) {
    if (!iter.can_use_32bit_indexing()) {
      for (auto& sub_iter : iter.with_32bit_indexing()) {
        ScatterFillInternalKernel<scalar_t>()(
            sub_iter, src_val, index_size, index_stride, numel, f);
      }
      return;
    }

    char* self_ptr = (char*)iter.data_ptr(0);
    char* index_ptr = (char*)iter.data_ptr(1);

    auto offset_calc = make_offset_calculator<2>(iter);

    auto loop = ScatterFillInternalKernelLoopFunctor<
        scalar_t,
        decltype(offset_calc),
        func_t>(self_ptr, index_ptr, offset_calc, index_stride, f, src_val);

    launch_scatter_gather_kernel(iter.numel(), loop);
  }
};

template <bool cast_to_opaque = true>
struct ScatterFillBaseKernel {
  template <typename func_t>
  void operator()(
      const Tensor& self,
      int64_t dim,
      const Tensor& index,
      Scalar src,
      const std::string& method_name,
      const func_t& f) {
    at::assert_no_internal_overlap(self);

    auto index_sizes = ensure_nonempty_vec(index.sizes().vec());

    // restride self such that
    // self.shape = index.shape and
    // self.stride[dim] = 0
    auto self_restrided = restride_dim(self, dim, index_sizes);

    auto iter = TensorIteratorConfig()
                    .set_check_mem_overlap(false)
                    .check_all_same_dtype(false)
                    .resize_outputs(false)
                    .add_output(self_restrided)
                    .add_const_input(index)
                    .build();

    auto index_size = ensure_nonempty_size(self, dim);
    auto index_stride = ensure_nonempty_stride(self, dim);

    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
        at::ScalarType::Half,
        at::ScalarType::Bool,
        at::ScalarType::BFloat16,
        iter.dtype(),
        "scatter_fill_base_kernel_func",
        [&] {
          using dtype = typename std::conditional<
              cast_to_opaque,
              OpaqueType<sizeof(scalar_t)>,
              scalar_t>::type;

          auto src_scalar_val = src.to<scalar_t>();
          auto src_val = *(dtype*)&src_scalar_val;

          ScatterFillInternalKernel<dtype>()(
              iter, src_val, index_size, index_stride, self.numel(), f);
        });
  }

  void operator()(
      const Tensor& self,
      int64_t dim,
      const Tensor& index,
      Scalar src,
      const std::string& method_name,
      const ReduceMultiply& f) {
    at::assert_no_internal_overlap(self);

    auto index_sizes = ensure_nonempty_vec(index.sizes().vec());

    // restride self such that
    // self.shape = index.shape and
    // self.stride[dim] = 0
    auto self_restrided = restride_dim(self, dim, index_sizes);

    auto iter = TensorIteratorConfig()
                    .set_check_mem_overlap(false)
                    .check_all_same_dtype(false)
                    .resize_outputs(false)
                    .add_output(self_restrided)
                    .add_const_input(index)
                    .build();

    auto index_size = ensure_nonempty_size(self, dim);
    auto index_stride = ensure_nonempty_stride(self, dim);

    AT_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        iter.dtype(),
        "scatter_fill_base_kernel_reduce_multiply",
        [&] {
          using dtype = typename std::conditional<
              cast_to_opaque,
              OpaqueType<sizeof(scalar_t)>,
              scalar_t>::type;

          auto src_scalar_val = src.to<scalar_t>();
          auto src_val = *(dtype*)&src_scalar_val;

          ScatterFillInternalKernel<dtype>()(
              iter, src_val, index_size, index_stride, self.numel(), f);
        });
  }
};

void gather_kernel(
    const Tensor& result,
    const Tensor& self,
    int64_t dim,
    const Tensor& index) {
  ScatterGatherBaseKernel</*is_scatter_like=*/false>()(
      result, dim, index, self, "gather_kernel", tensor_assign);
}

void scatter_kernel(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src) {
  // When indices are not unique, the behavior is non-deterministic
  globalContext().alertNotDeterministic("scatter_");
  ScatterGatherBaseKernel<>()(
      self, dim, index, src, "scatter_kernel", tensor_assign);
}

void scatter_fill_kernel(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Scalar& src) {
  ScatterFillBaseKernel<>()(
      self, dim, index, src, "scatter_fill_kernel", tensor_assign);
}

void scatter_add_kernel(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src) {
  // See Note [Writing Nondeterministic Operations]
  // Nondeterministic because of atomicAdd usage
  globalContext().alertNotDeterministic("scatter_add_kernel");
  ScatterGatherBaseKernel</*is_scatter_like=*/true, /*cast_to_opaque=*/false>()(
      self, dim, index, src, "scatter_add_kernel", reduce_add);
}

void scatter_reduce_kernel(
    const Tensor& self,
    const int64_t dim,
    const Tensor& index,
    const Tensor& src,
    const ReductionType& reduce) {
  // See Note [Writing Nondeterministic Operations]
  // Nondeterministic because of atomicAdd/AtomicMul usage
  globalContext().alertNotDeterministic("scatter_reduce_kernel");
  switch (reduce) {
    case ReductionType::SUM:
      ScatterGatherBaseKernel<true, false>()(
          self, dim, index, src, "scatter_reduce_kernel_add", reduce_add);
      break;
    case ReductionType::PROD:
      ScatterGatherBaseKernel<true, false>()(
          self,
          dim,
          index,
          src,
          "scatter_reduce_kernel_multiply",
          reduce_multiply);
      break;
    default:
      break;
  }
}

void scatter_reduce_two_kernel(
    const Tensor& self,
    const int64_t dim,
    const Tensor& index,
    const Tensor& src,
    const ReductionType& reduce) {
  switch (reduce) {
    case ReductionType::SUM:
      globalContext().alertNotDeterministic("scatter_reduce_kernel_sum");
      ScatterGatherBaseKernel<true, false>()(
          self, dim, index, src, "scatter_reduce_kernel_sum", reduce_add);
      break;
    case ReductionType::PROD:
      globalContext().alertNotDeterministic("scatter_reduce_kernel_prod");
      ScatterGatherBaseKernel<true, false>()(
          self, dim, index, src, "scatter_reduce_kernel_prod", reduce_multiply);
      break;
    case ReductionType::MAX:
      ScatterGatherBaseKernel<true, false>()(
          self, dim, index, src, "scatter_reduce_kernel_amax", reduce_maximum);
      break;
    case ReductionType::MIN:
      ScatterGatherBaseKernel<true, false>()(
          self, dim, index, src, "scatter_reduce_kernel_amin", reduce_minimum);
      break;
    case ReductionType::MEAN:
      globalContext().alertNotDeterministic("scatter_reduce_kernel_mean");
      ScatterGatherBaseKernel<true, false>()(
          self, dim, index, src, "scatter_reduce_kernel_mean", reduce_mean);
      break;
  }
}

void scatter_scalar_reduce_kernel(
    const Tensor& self,
    const int64_t dim,
    const Tensor& index,
    const Scalar& value,
    const ReductionType& reduce) {
  switch (reduce) {
    case ReductionType::SUM:
      ScatterFillBaseKernel<false>()(
          self, dim, index, value, "scatter_fill_kernel_add", reduce_add);
      break;
    case ReductionType::PROD:
      ScatterFillBaseKernel<false>()(
          self,
          dim,
          index,
          value,
          "scatter_fill_kernel_multiply",
          reduce_multiply);
      break;
    default:
      break;
  }
}

} // namespace xpu
} // namespace native
} // namespace at

#pragma GCC diagnostic pop
#pragma clang diagnostic pop
