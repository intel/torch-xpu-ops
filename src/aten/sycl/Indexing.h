#pragma once

#include <aten/sycl/BatchKernel.h>
#include <comm/TensorInfo.h>
#include <aten/sycl/Loops.h>

using namespace at::xpu::detail;
using namespace at::xpu;

namespace at::native::xpu {

// Pretend that the scalar tensor is in fact a one-element vector.
template <typename T, typename IndexType>
TensorInfo<T, IndexType> tensorInfoIfScalar(TensorInfo<T, IndexType> ti) {
  if (ti.dims == 0) {
    ti.dims = 1;
    ti.sizes[0] = 1;
    ti.strides[0] = 1;
  }
  return ti;
}

template <class SrcInfo, class DstInfo, class IdxInfo, class FuncType>
class IndexKernelConfig : public BatchKernelConfig {
 public:
  using ValType = typename SrcInfo::scalar_t;
  using IdxType = typename IdxInfo::scalar_t;

  IndexKernelConfig() = delete;
  IndexKernelConfig(
      SrcInfo& sinfo,
      DstInfo& dinfo,
      IdxInfo& iinfo,
      ValType alpha,
      int64_t index_num,
      int64_t indexing_dimension_size,
      bool indexing_dst,
      bool problem_inner,
      FuncType func,
      int64_t batch,
      int64_t problem,
      int64_t stride,
      int64_t problem_batch,
      bool problem_along_x)
      : BatchKernelConfig(
            batch,
            problem,
            stride,
            problem_batch,
            problem_along_x,
            Policy::pSegment,
            syclMaxWorkItemsPerEU()),
        sinfo_(sinfo),
        dinfo_(dinfo),
        iinfo_(iinfo),
        alpha_(alpha),
        index_num_(index_num),
        indexing_dimension_size_(indexing_dimension_size),
        indexing_dst_(indexing_dst),
        problem_inner_(problem_inner),
        func_(func) {}

  template <class TarInfo>
  static inline void indexing_problem_mapping(
      TarInfo& tinfo,
      IdxInfo& iinfo,
      int dim,
      int64_t index_num,
      int64_t indexing_dimension_size,
      int64_t& batch,
      int64_t& problem,
      int64_t& stride,
      int64_t& problem_batch,
      bool& problem_along_x,
      bool& problem_inner) {
    int64_t outer = tinfo.outerSize(dim);
    int64_t inner = tinfo.innerSize(dim);

    if (inner == 1) {
      problem = outer;
      stride = indexing_dimension_size;
      batch = 1;
      problem_batch = index_num;
      problem_along_x = tinfo.strides[dim] == 1 ? false : true;
      problem_inner = false;
    } else if (outer == 1) {
      problem = inner;
      stride = 1;
      batch = indexing_dimension_size;
      problem_batch = index_num;
      problem_along_x = tinfo.strides[tinfo.dims - 1] == 1 ? true : false;
      problem_inner = true;
    } else {
      problem = inner;
      stride = 1;
      batch = outer * indexing_dimension_size;
      problem_batch = outer * index_num;
      problem_along_x = tinfo.strides[tinfo.dims - 1] == 1 ? true : false;
      problem_inner = true;
    }
    return;
  }

  static IndexKernelConfig<SrcInfo, DstInfo, IdxInfo, FuncType> make_config(
      SrcInfo& src_info,
      DstInfo& dst_info,
      IdxInfo& index_info,
      ValType alpha,
      int64_t dim,
      bool indexing_dst,
      FuncType func) {
    int64_t index_num = index_info.sizes[0];
    int64_t indexing_dimension_size;

    bool problem_along_x, problem_inner;
    int64_t batch, problem, stride, problem_batch;

    TORCH_CHECK(
        indexing_dst || src_info.data != nullptr,
        "Indexing kernel backbone does not support null src ...");

    if (indexing_dst) {
      indexing_dimension_size = dst_info.sizes[dim];
      indexing_problem_mapping(
          dst_info,
          index_info,
          dim,
          index_num,
          indexing_dimension_size,
          batch,
          problem,
          stride,
          problem_batch,
          problem_along_x,
          problem_inner);
    } else {
      indexing_dimension_size = src_info.sizes[dim];
      indexing_problem_mapping(
          src_info,
          index_info,
          dim,
          index_num,
          indexing_dimension_size,
          batch,
          problem,
          stride,
          problem_batch,
          problem_along_x,
          problem_inner);
    }

    return {
        src_info,
        dst_info,
        index_info,
        alpha,
        index_num,
        indexing_dimension_size,
        indexing_dst,
        problem_inner,
        func,
        batch,
        problem,
        stride,
        problem_batch,
        problem_along_x};
  }

 public:
  SrcInfo sinfo_; // sinfo_.data could be nullptr, while indexing along dst.
  DstInfo dinfo_;
  IdxInfo iinfo_;
  ValType alpha_;
  int64_t index_num_;
  int64_t indexing_dimension_size_;
  bool indexing_dst_;
  bool problem_inner_;
  FuncType func_;
};

template <
    class IdxConfig,
    bool TrivialOffCal = false,
    bool KnownProblemInner = false>
class IndexKernel {
 public:
  using ValType = typename IdxConfig::ValType;
  using IdxType = typename IdxConfig::IdxType;

  IndexKernel() = delete;
  IndexKernel(IdxConfig& cfg) : cfg_(cfg) {}

  void init_global_batch_info(
      BatchKernelConfig::ItemDesc& id,
      int64_t& idx_logical_off,
      int64_t& glb_batch_group,
      int64_t& glb_batch_group_loc_off) const {
    idx_logical_off = id.glb_batch % cfg_.index_num_;
    int64_t idx_off;
    if constexpr (TrivialOffCal) {
      idx_off = idx_logical_off;
    } else {
      idx_off = IndexToOffset<IdxType, int64_t>::get(
          idx_logical_off,
          cfg_.iinfo_,
          IndexToOffset<IdxType, int64_t>::NON_STRICT_CONTIGUOUS);
    }
    glb_batch_group = id.glb_batch / cfg_.index_num_;
    glb_batch_group_loc_off = cfg_.iinfo_.data[idx_off];
    glb_batch_group_loc_off = glb_batch_group_loc_off >= 0
        ? glb_batch_group_loc_off
        : cfg_.indexing_dimension_size_ + glb_batch_group_loc_off;
  }

  int64_t inline indexing_logical_off(
      BatchKernelConfig::ItemDesc& id,
      int64_t glb_batch_group,
      int64_t glb_batch_group_loc_off) const {
    int64_t si, pi, bi;
    int64_t glb_batch_group_glb_off =
        glb_batch_group * cfg_.indexing_dimension_size_ +
        glb_batch_group_loc_off;
    auto stride = cfg_.stride_;
    if constexpr (KnownProblemInner) {
      si = 0;
      pi = id.glb_problem;
      bi = glb_batch_group_glb_off;
      return (pi + bi * cfg_.problem_) * stride;
    } else {
      if (cfg_.problem_inner_) {
        si = 0;
        pi = id.glb_problem;
        bi = glb_batch_group_glb_off;
        return (pi + bi * cfg_.problem_) * stride;
      } else {
        si = glb_batch_group_glb_off;
        pi = id.glb_problem;
        bi = 0;
        return si + pi * stride;
      }
    }
  }

  int64_t inline fixing_logical_off(
      BatchKernelConfig::ItemDesc& id,
      int64_t glb_batch_group,
      int64_t idx_logical_off) const {
    int64_t si, pi, bi, stride;
    int64_t glb_batch_group_glb_off =
        glb_batch_group * cfg_.index_num_ + idx_logical_off;
    if constexpr (KnownProblemInner) {
      si = 0;
      stride = 1;
      pi = id.glb_problem;
      bi = glb_batch_group_glb_off;
      return pi + bi * cfg_.problem_;
    } else {
      if (cfg_.problem_inner_) {
        si = 0;
        stride = 1;
        pi = id.glb_problem;
        bi = glb_batch_group_glb_off;
        return pi + bi * cfg_.problem_;
      } else {
        bi = 0;
        si = glb_batch_group_glb_off;
        pi = id.glb_problem;
        stride = cfg_.index_num_;
        return si + pi * stride;
      }
    }
  }

  void operator()(sycl::nd_item<2> item) const {
    auto id = cfg_.get_item_desc(item);

    if (id.glb_problem >= cfg_.problem_ ||
        id.glb_batch >= cfg_.problem_batch_) {
      return;
    }

    // Indexing kernels have three operands,
    // 1. index operand
    // 2. operand indexing on
    // 3. operand has fixing size as index (optional)
    int64_t idx_logical_off, glb_batch_group, glb_batch_group_loc_off;
    int64_t glb_indexing_logical_off, glb_fixing_logical_off;
    int64_t dst_off, src_off;

    init_global_batch_info(
        id, idx_logical_off, glb_batch_group, glb_batch_group_loc_off);

    glb_indexing_logical_off =
        indexing_logical_off(id, glb_batch_group, glb_batch_group_loc_off);

    if (cfg_.sinfo_.data != nullptr && cfg_.dinfo_.data != nullptr) {
      glb_fixing_logical_off =
          fixing_logical_off(id, glb_batch_group, idx_logical_off);
    }

    if constexpr (TrivialOffCal) {
      if (cfg_.indexing_dst_) {
        dst_off = glb_indexing_logical_off;
        if (cfg_.sinfo_.data != nullptr) {
          src_off = glb_fixing_logical_off;
        }
      } else {
        src_off = glb_indexing_logical_off;
        dst_off = glb_fixing_logical_off;
      }
    } else {
      if (cfg_.indexing_dst_) {
        // index_copy, index_add, index_fill
        dst_off = IndexToOffset<ValType, int64_t>::get(
            glb_indexing_logical_off,
            cfg_.dinfo_,
            IndexToOffset<ValType, int64_t>::NON_STRICT_CONTIGUOUS);
        if (cfg_.sinfo_.data != nullptr) {
          src_off = IndexToOffset<ValType, int64_t>::get(
              glb_fixing_logical_off,
              cfg_.sinfo_,
              IndexToOffset<ValType, int64_t>::NON_STRICT_CONTIGUOUS);
        }
      } else {
        // index_select
        src_off = IndexToOffset<ValType, int64_t>::get(
            glb_indexing_logical_off,
            cfg_.sinfo_,
            IndexToOffset<ValType, int64_t>::NON_STRICT_CONTIGUOUS);
        dst_off = IndexToOffset<ValType, int64_t>::get(
            glb_fixing_logical_off,
            cfg_.dinfo_,
            IndexToOffset<ValType, int64_t>::NON_STRICT_CONTIGUOUS);
      }
    }
    cfg_.func_(
        cfg_.dinfo_.data,
        cfg_.sinfo_.data,
        dst_off,
        src_off,
        glb_batch_group_loc_off,
        cfg_.alpha_);
  }

 private:
  IdxConfig cfg_;
};

template <
    class IdxConfig,
    bool TrivialOffCal = false,
    bool KnownProblemInner = false>
static inline void launch_index_kernel(IdxConfig& cfg) {
  auto& queue = getCurrentSYCLQueue();
  IndexKernel<IdxConfig, TrivialOffCal, KnownProblemInner> idx_ker(cfg);
  sycl_kernel_submit(cfg.global_size(), cfg.group_size(), queue, idx_ker);
}

template <typename func_t, typename index_buf_type>
struct SmallIndexKernelImplFunctor {
  void operator()(sycl::nd_item<1> item_id) const {
    auto local_id = item_id.get_local_id(0);
    auto group_id = item_id.get_group(0);

    // construct a indices_size table on SLM
    for (int64_t local_index = local_id; local_index < indices_size;
         local_index += wgroup_size) {
      int64_t offset = 0;
      for (size_t i = 0; i < num_indices; i++) {
        // handle int32 index tensor according to the indice_size_bytes.
        // we didn't use template parametor to avoid too many kernels' creation
        // with numbers of input datatypes.
        if (indice_size_bytes == 4) {
          int32_t index =
              *(int32_t*)(index_ptrs[i] + local_index * indice_size_bytes);
          SYCL_KERNEL_ASSERT(
              index >= -sizes[i] && index < sizes[i] && "index out of bounds");
          if (index < 0) {
            index += sizes[i];
          }
          offset += index * strides[i];
        } else {
          int64_t index =
              *(int64_t*)(index_ptrs[i] + local_index * indice_size_bytes);
          SYCL_KERNEL_ASSERT(
              index >= -sizes[i] && index < sizes[i] && "index out of bounds");
          if (index < 0) {
            index += sizes[i];
          }
          offset += index * strides[i];
        }
      }
      local_offset[local_index] = offset;
    }

    // calculate the number of workloads on each group
    auto group_linear_id = group_id * group_numel;
    auto group_numel_range = group_numel;
    if (group_num_tail && group_id >= group_num) {
      group_linear_id =
          group_num * group_numel + (group_id - group_num) * group_numel_tail;
      group_numel_range = group_numel_tail;
    }
    auto out_ptr = out_data;
    auto in_ptr = in_data;
    item_id.barrier(sycl::access::fence_space::local_space);

    // compute the in/out/indices offsets and perform memory copy
    for (int64_t local_index = local_id; local_index < group_numel_range;
         local_index += wgroup_size) {
      auto linear_id = group_linear_id + local_index;
      auto out_offset = linear_id * element_size_bytes;
      auto src_linear_id = linear_id / indices_size;
      int64_t in_offset = 0;
      for (int i = num_non_indices - 1; i > 0; --i) {
        in_offset += (src_linear_id % src_sizes[i]) * src_strides[i];
        src_linear_id /= src_sizes[i];
      }
      in_offset += src_linear_id * src_strides0;

      auto offset = local_offset[local_index % indices_size];
      f(out_ptr + out_offset, in_ptr + in_offset, offset);
    }
  }
  SmallIndexKernelImplFunctor(
      const func_t f_,
      int64_t indices_size_,
      int64_t group_num_tail_,
      int64_t group_num_,
      int64_t group_numel_,
      int64_t group_numel_tail_,
      int64_t wgroup_size_,
      size_t num_non_indices_,
      at::detail::Array<int64_t, XPU_MAX_TENSORINFO_DIMS> src_sizes_,
      at::detail::Array<int64_t, XPU_MAX_TENSORINFO_DIMS> src_strides_,
      int64_t src_strides0_,
      size_t num_indices_,
      at::detail::Array<int64_t, XPU_MAX_TENSORINFO_DIMS> sizes_,
      at::detail::Array<int64_t, XPU_MAX_TENSORINFO_DIMS> strides_,
      int64_t element_size_bytes_,
      int64_t indice_size_bytes_,
      char* out_data_,
      char* in_data_,
      at::detail::Array<index_buf_type, XPU_MAX_TENSORINFO_DIMS> index_ptrs_,
      sycl_local_acc_t<int64_t, 1> local_offset_)
      : f(f_),
        indices_size(indices_size_),
        group_num_tail(group_num_tail_),
        group_num(group_num_),
        group_numel(group_numel_),
        group_numel_tail(group_numel_tail_),
        wgroup_size(wgroup_size_),
        num_non_indices(num_non_indices_),
        src_sizes(src_sizes_),
        src_strides(src_strides_),
        src_strides0(src_strides0_),
        num_indices(num_indices_),
        sizes(sizes_),
        strides(strides_),
        element_size_bytes(element_size_bytes_),
        indice_size_bytes(indice_size_bytes_),
        out_data(out_data_),
        in_data(in_data_),
        index_ptrs(index_ptrs_),
        local_offset(local_offset_) {}

 private:
  const func_t f;
  int64_t indices_size;
  int64_t group_num_tail;
  int64_t group_num;
  int64_t group_numel;
  int64_t group_numel_tail;
  int64_t wgroup_size;
  size_t num_non_indices;
  at::detail::Array<int64_t, XPU_MAX_TENSORINFO_DIMS> src_sizes;
  at::detail::Array<int64_t, XPU_MAX_TENSORINFO_DIMS> src_strides;
  int64_t src_strides0;
  size_t num_indices;
  at::detail::Array<int64_t, XPU_MAX_TENSORINFO_DIMS> sizes;
  at::detail::Array<int64_t, XPU_MAX_TENSORINFO_DIMS> strides;
  int64_t element_size_bytes;
  int64_t indice_size_bytes;
  char* out_data;
  char* in_data;
  at::detail::Array<index_buf_type, XPU_MAX_TENSORINFO_DIMS> index_ptrs;
  sycl_local_acc_t<int64_t, 1> local_offset;
};


template <>
SmallIndexKernelCreator {
  SmallIndexKernelCreator();
  SmallIndexKernelImplFunctor<func_t, index_buf_type> operator()(sycl::handler& cgh) {
    
  }
};

// SYCL suggest: it’s possible (and even desirable) to oversubscribe tasks to
// device;
constexpr int OVER_SUBSCRIBE_DSS_FACTOR = 16;

template <typename func_t>
void small_index_kernel_impl(
    TensorIterator& iter,
    IntArrayRef index_size,
    IntArrayRef index_stride,
    IntArrayRef non_index_size,
    IntArrayRef non_index_stride,
    const func_t f) {
  auto numel = iter.numel();
  auto indices_size = iter.tensor(2).size(-1);
  auto& queue = getCurrentSYCLQueue();
  auto dev_id = getDeviceIndexOfCurrentQueue();
  int64_t max_group_num = syclMaxDSSNum(dev_id) * OVER_SUBSCRIBE_DSS_FACTOR;

  auto total_index_iter = numel / indices_size;
  max_group_num = std::min(int64_t(total_index_iter / 2), max_group_num);

  // process the tail
  auto group_index_iter =
      (total_index_iter + max_group_num - 1) / max_group_num;
  auto group_num_tail = group_index_iter * max_group_num - total_index_iter;
  auto group_num = max_group_num - group_num_tail;
  auto group_numel = group_index_iter * indices_size;
  auto group_numel_tail = (group_index_iter - 1) * indices_size;

  auto wgroup_size = syclMaxWorkGroupSize(dev_id);
  wgroup_size = std::min(decltype(wgroup_size)(group_numel), wgroup_size);
  auto global_size = max_group_num * wgroup_size;

  size_t num_non_indices = non_index_size.size();
  at::detail::Array<int64_t, XPU_MAX_TENSORINFO_DIMS> src_sizes(0);
  at::detail::Array<int64_t, XPU_MAX_TENSORINFO_DIMS> src_strides(0);
  for (size_t i = 0; i < num_non_indices; ++i) {
    src_sizes[i] = non_index_size[i];
    src_strides[i] = non_index_stride[i];
  }
  auto src_strides0 = non_index_stride[0];

  size_t num_indices = index_size.size();
  at::detail::Array<int64_t, XPU_MAX_TENSORINFO_DIMS> sizes(0);
  at::detail::Array<int64_t, XPU_MAX_TENSORINFO_DIMS> strides(0);
  for (size_t i = 0; i < num_indices; i++) {
    sizes[i] = index_size[i];
    strides[i] = index_stride[i];
  }

  int64_t element_size_bytes = iter.tensor(1).element_size();
  int64_t indice_size_bytes = iter.tensor(2).element_size();

  auto out_data = (char*)iter.data_ptr(0);
  auto in_data = (char*)iter.data_ptr(1);
  using index_buf_type = decltype((char*)iter.data_ptr(0));
  at::detail::Array<index_buf_type, XPU_MAX_TENSORINFO_DIMS> index_ptrs;
  for (size_t i = 0; i < num_indices; i++) {
    index_ptrs[i] = (char*)iter.data_ptr(i + 2);
  }

  sycl_local_acc_t<int64_t, 1> local_offset(indices_size, __cgh);
  SmallIndexKernelImplFunctor<func_t, index_buf_type> kfn(
      f,
      indices_size,
      group_num_tail,
      group_num,
      group_numel,
      group_numel_tail,
      wgroup_size,
      num_non_indices,
      src_sizes,
      src_strides,
      src_strides0,
      num_indices,
      sizes,
      strides,
      element_size_bytes,
      indice_size_bytes,
      out_data,
      in_data,
      index_ptrs,
      local_offset);
  sycl_kernel_submit(sycl::range<1>(global_size), sycl::range<1>(wgroup_size), queue, kfn);
}

template <
    typename func_t,
    typename index_buf_type,
    typename OffsetCalculatorType>
struct IndexKernelImplFunctor {
  void operator()(sycl::item<1> item_id) const {
    auto linear_idx = item_id.get_linear_id();
    auto offsets = offset_calc.get(linear_idx);
    auto out_ptr = out_data + offsets[0];
    auto in_ptr = in_data + offsets[1];
    int64_t offset = 0;
    //#pragma unroll
    for (size_t i = 0; i < num_indices; i++) {
      // handle int32 index tensor according to the indice_size_bytes.
      // we didn't use template parametor to avoid too many kernels' creation
      // with numbers of input datatypes.
      if (indice_size_bytes == 4) {
        int32_t index = *(int32_t*)(index_ptrs[i] + offsets[2]);
        SYCL_KERNEL_ASSERT(
            index >= -sizes[i] && index < sizes[i] && "index out of bounds");
        if (index < 0) {
          index += sizes[i];
        }
        offset += index * strides[i];
      } else {
        int64_t index = *(int64_t*)(index_ptrs[i] + offsets[2]);
        SYCL_KERNEL_ASSERT(
            index >= -sizes[i] && index < sizes[i] && "index out of bounds");
        if (index < 0) {
          index += sizes[i];
        }
        offset += index * strides[i];
      }
    }
    f(out_ptr, in_ptr, offset);
  }
  IndexKernelImplFunctor(
      const func_t f_,
      OffsetCalculatorType offset_calc_,
      int64_t indice_size_bytes_,
      char* out_data_,
      char* in_data_,
      size_t num_indices_,
      at::detail::Array<index_buf_type, XPU_MAX_TENSORINFO_DIMS> index_ptrs_,
      at::detail::Array<int64_t, XPU_MAX_TENSORINFO_DIMS> sizes_,
      at::detail::Array<int64_t, XPU_MAX_TENSORINFO_DIMS> strides_)
      : f(f_),
        offset_calc(offset_calc_),
        indice_size_bytes(indice_size_bytes_),
        out_data(out_data_),
        in_data(in_data_),
        num_indices(num_indices_),
        index_ptrs(index_ptrs_),
        sizes(sizes_),
        strides(strides_) {}

 private:
  const func_t f;
  OffsetCalculatorType offset_calc;
  int64_t indice_size_bytes;
  char* out_data;
  char* in_data;
  size_t num_indices;
  at::detail::Array<index_buf_type, XPU_MAX_TENSORINFO_DIMS> index_ptrs;
  at::detail::Array<int64_t, XPU_MAX_TENSORINFO_DIMS> sizes;
  at::detail::Array<int64_t, XPU_MAX_TENSORINFO_DIMS> strides;
};

template <typename func_t>
void index_kernel_impl(
    TensorIterator& iter,
    IntArrayRef index_size,
    IntArrayRef index_stride,
    const func_t f) {
  size_t num_indices = index_size.size();
  auto numel = iter.numel();
  at::detail::Array<int64_t, XPU_MAX_TENSORINFO_DIMS> sizes(0);
  at::detail::Array<int64_t, XPU_MAX_TENSORINFO_DIMS> strides(0);
  for (size_t i = 0; i < num_indices; i++) {
    sizes[i] = index_size[i];
    strides[i] = index_stride[i];
  }

  int64_t indice_size_bytes = iter.tensor(2).element_size();

  auto& queue = getCurrentSYCLQueue();

  auto cgf = DPCPP_Q_CGF(__cgh) {
    auto out_data = (char*)iter.data_ptr(0);
    auto in_data = (char*)iter.data_ptr(1);
    using index_buf_type = decltype((char*)iter.data_ptr(0));
    at::detail::Array<index_buf_type, XPU_MAX_TENSORINFO_DIMS> index_ptrs;
    for (size_t i = 0; i < num_indices; i++) {
      index_ptrs[i] = (char*)iter.data_ptr(i + 2);
    }

    auto offset_calc = make_offset_calculator<3>(iter);
    IndexKernelImplFunctor<func_t, index_buf_type, decltype(offset_calc)>
        kfn(f,
            offset_calc,
            indice_size_bytes,
            out_data,
            in_data,
            num_indices,
            index_ptrs,
            sizes,
            strides);
    __cgh.parallel_for<decltype(kfn)>(sycl::range</*dim=*/1>(numel), kfn);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

template <typename func_t>
void _index_kernel(
    TensorIterator& iter,
    IntArrayRef index_size,
    IntArrayRef index_stride,
    IntArrayRef non_index_size,
    IntArrayRef non_index_stride,
    const func_t f) {
  auto numel = iter.numel();

  if (numel == 0) {
    return;
  }

  size_t num_indices = index_size.size();
  TORCH_INTERNAL_ASSERT(num_indices == index_stride.size());
  TORCH_INTERNAL_ASSERT(
      num_indices == static_cast<size_t>(iter.ntensors()) - 2);
  TORCH_INTERNAL_ASSERT(num_indices <= XPU_MAX_TENSORINFO_DIMS);

  // the small_index_kernel_impl is applied for last several
  // successive dims indexing of an input tensor Taking 3-dims tensor input
  // (input.shape=[x,y,z]) for example: input[:,:,idx] or input[:,idx1,idx2]
  // when input tensor satisfies the following conditions, the
  // small_index_kernel path will be selected: 1.there are common indices
  // such as input[:,:,idx] and input[:,idx1,idx2] instead of
  //   input[idx0,idx1,idx2], input[idx0,idx1,:], input[idx0,:,idx2],
  //   input[idx0,:,:], input[:,idx1,:]
  // 2.the common indices numel should larger than 2 times of the
  // syclMaxComputeUnitSize (then we can get memory access benifit) 3.the
  // workloads in each group should larger than the maximum number of
  // workitem (ensure all the workitem activate) 4.the indices_table size
  // should satisfied the SLM limit condition

  // check whether the current case satisfying the condition 1
  // for 3-dims input:
  // Taking input[idx0,:,idx2] for example, the indices_sizes=[sz,1,sz]
  // While the satified case is input[:,idx1,idx2], indices_sizes=[1,sz,sz]
  bool small_index = non_index_size.size() != 0 && iter.tensor(1).dim() == 3 &&
      non_index_size.size() + index_size.size() == 3;
  auto indices_sizes = iter.tensor(2).sizes();
  for (size_t i = 1; i < iter.tensor(2).dim(); ++i) {
    if (indices_sizes[i - 1] > indices_sizes[i]) {
      small_index = false;
      break;
    }
  }
  if (small_index) {
    // auto& dpcpp_queue = getCurrentSYCLQueue();
    auto dev_id = getDeviceIndexOfCurrentQueue();
    int64_t max_group_num = syclMaxDSSNum(dev_id);
    auto wgroup_size = syclMaxWorkGroupSize(dev_id);
    auto indices_size = iter.tensor(2).size(-1);
    auto total_index_iter = numel / indices_size;
    auto local_index = numel / max_group_num;

    // the max_local_mem_size = 65536B (64KB)
    // TODO: Is this right?
    auto max_local_mem_size = syclLocalMemSize(dev_id);
    auto indice_table_size = indices_size * sizeof(int64_t);

    // check whether the current case satisfying conditions 2,3,4
    small_index =
        (total_index_iter > 2 * max_group_num && local_index > wgroup_size &&
         indice_table_size < max_local_mem_size * 0.5);
    if (small_index) {
      small_index_kernel_impl<func_t>(
          iter, index_size, index_stride, non_index_size, non_index_stride, f);
      return;
    }
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      _index_kernel(
          sub_iter, index_size, index_stride, IntArrayRef{}, IntArrayRef{}, f);
    }
    return;
  }

  index_kernel_impl<func_t>(iter, index_size, index_stride, f);
}


} // namespace at::native::xpu
