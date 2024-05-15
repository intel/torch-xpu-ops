#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/ScalarOps.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/IndexingUtils.h>
#include <ATen/native/TensorAdvancedIndexing.h>
#include <ATen/native/TensorAdvancedIndexingUtils.h>
#include <ATen/native/TensorIterator.h>

#include <ATen/XPUNativeFunctions.h>
#include <aten/sycl/IndexingKernel.h>
#include <comm/ReduceOpsUtils.h>

namespace at {

// TODO: Should reuse source in stock PyTorch when in-tree.

static bool all_strides_match(TensorList tensors) {
  TORCH_CHECK(!tensors.empty());
  auto strides = tensors[0].strides();
  for (auto& tensor : tensors.slice(1)) {
    if (!strides.equals(tensor.strides())) {
      return false;
    }
  }
  return true;
}

// Replace indexed dimensions in src with stride 0 and the size of the result
// tensor. The offset in these dimensions is computed by the kernel using the
// index tensor's values and the stride of src. The new shape is not meaningful.
// It's used to make the shape compatible with the result tensor.
static Tensor restride_src(
    const Tensor& src,
    int64_t dims_before,
    int64_t dims_indexed,
    IntArrayRef replacement_shape) {
  auto shape = DimVector(src.sizes());
  auto strides = DimVector(src.strides());
  int64_t end = dims_before + dims_indexed;
  shape.erase(shape.begin() + dims_before, shape.begin() + end);
  strides.erase(strides.begin() + dims_before, strides.begin() + end);
  shape.insert(
      shape.begin() + dims_before,
      replacement_shape.begin(),
      replacement_shape.end());
  strides.insert(strides.begin() + dims_before, replacement_shape.size(), 0);
  return src.as_strided(shape, strides);
}

// Add dimensions of size 1 to an index tensor so that it can be broadcast to
// the result shape and iterated over element-wise like the result tensor and
// the restrided src.
static Tensor reshape_indexer(
    const Tensor& index,
    int64_t dims_before,
    int64_t dims_after) {
  auto orig_shape = index.sizes();
  auto shape = DimVector();
  shape.append(dims_before, 1);
  shape.append(orig_shape.begin(), orig_shape.end());
  shape.append(dims_after, 1);
  return index.reshape(shape);
}

native::AdvancedIndex::AdvancedIndex(
    const Tensor& src,
    TensorList indices_list) {
  int64_t element_size_bytes = src.element_size();
  int64_t dims_before = 0, dims_after = 0, dims_indexed = 0;
  IntArrayRef replacement_shape;
  for (const auto dim : c10::irange(indices_list.size())) {
    if (!indices_list[dim].defined()) {
      if (dims_indexed == 0) {
        dims_before++;
      } else {
        dims_after++;
      }
    } else {
      dims_indexed++;
      replacement_shape = indices_list[dim].sizes();
      indexed_sizes.push_back(src.size(dim));
      indexed_strides.push_back(src.stride(dim) * element_size_bytes);
    }
  }

  // Check if the indexed subspace contains a dim of size 0, but the replacement
  // shape does not. This implies that an index is out of bounds, because there
  // is no number that's a valid index for an empty tensor. Normally, out of
  // bounds is handled in the indexing kernel, but this case fails earlier in
  // restride_src with an unhelpful error message.
  if (std::find(indexed_sizes.begin(), indexed_sizes.end(), 0) !=
          indexed_sizes.end() &&
      std::find(replacement_shape.begin(), replacement_shape.end(), 0) ==
          replacement_shape.end()) {
    TORCH_CHECK_INDEX(
        false, "index is out of bounds for dimension with size 0");
  }

  this->dims_before = dims_before;
  this->dims_after = dims_after;
  this->src = restride_src(src, dims_before, dims_indexed, replacement_shape);

  for (auto& index : indices_list) {
    if (index.defined()) {
      indices.push_back(reshape_indexer(index, dims_before, dims_after));
    }
  }

  if (indices.size() >= 2 && (this->src.device().type() == kXPU)) {
    if (!all_strides_match(indices)) {
      for (auto& indice : indices) {
        indice = indice.contiguous();
      }
    }
  }
}

Tensor& XPUNativeFunctions::masked_fill_(
    Tensor& self,
    const Tensor& mask,
    const Scalar& value) {
  TORCH_CHECK(
      self.device() == mask.device(),
      "expected self and mask to be on the same device, but got mask on ",
      mask.device(),
      " and self on ",
      self.device());
  TORCH_CHECK(
      mask.scalar_type() == kBool,
      "masked_fill only supports boolean masks, but got dtype ",
      mask.scalar_type());
  auto maybe_outnames =
      namedinference::broadcast_to_outnames(self, mask, "masked_fill_");
  if (at::has_internal_overlap(self) == MemOverlap::Yes) {
    TORCH_WARN(
        "Use of masked_fill_ on expanded tensors is deprecated. "
        "Please clone() the tensor before performing this operation. "
        "This also applies to advanced indexing e.g. tensor[mask] = scalar");
  }
  at::assert_no_partial_overlap(self, mask);

  c10::MaybeOwned<Tensor> b_mask = expand_inplace(self, mask, "masked_fill_");

  auto iter = TensorIteratorConfig()
                  .set_check_mem_overlap(false)
                  .check_all_same_dtype(false)
                  .resize_outputs(false)
                  .add_output(self)
                  .add_const_input(self)
                  .add_const_input(*b_mask)
                  .build();

  native::xpu::masked_fill_kernel(iter, value);
  namedinference::propagate_names_if_nonempty(self, maybe_outnames);
  return self;
}

Tensor& XPUNativeFunctions::masked_fill_(
    Tensor& self,
    const Tensor& mask,
    const Tensor& value) {
  TORCH_CHECK(
      value.dim() == 0,
      "masked_fill_ only supports a 0-dimensional value tensor, but got tensor "
      "with ",
      value.dim(),
      " dimension(s).");
  // We hit this function if either of the input tensor lives on XPU.
  // It is ok, if `value` is `CPU` tensor but we should not allow `self` or
  // `mask` to be CPU tensor. Check for `self` and `mask` being on same device
  // exists in `masked_fill_` (Scalar version).
  TORCH_CHECK(
      self.device().is_xpu(),
      "masked_fill_: Expected inputs to be on same device")
  return XPUNativeFunctions::masked_fill_(self, mask, value.item());
}

void index_func_meta_impl(
    Tensor& result,
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& source,
    c10::string_view func) {
  auto numel = index.numel();

  TORCH_CHECK_INDEX(
      index.dim() <= 1,
      func,
      "_(): Index is supposed to be a vector, but got dim: ",
      index.dim(),
      " with type: ",
      index.scalar_type(),
      " and size: ",
      index.sizes());
  TORCH_CHECK(
      index.scalar_type() == ScalarType::Long ||
          index.scalar_type() == ScalarType::Int,
      func,
      "_(): Expected dtype int32/int64 for index but got: ",
      index.scalar_type());
  TORCH_CHECK(
      self.scalar_type() == source.scalar_type(),
      func,
      "_(): self (",
      self.scalar_type(),
      ") and source (",
      source.scalar_type(),
      ") must have the same scalar type");
  TORCH_CHECK(
      dim == 0 || dim < source.dim(),
      func,
      "_(): Indexing dim ",
      dim,
      " is out of bounds of the source tensor with dim ",
      source.dim());
  TORCH_CHECK(
      numel == (source.dim() == 0 ? 1 : source.size(dim)),
      func,
      "_(): Number of indices (",
      numel,
      ") should be equal to source.size(dim): (",
      source.size(dim),
      "), for dim: ",
      dim);

  auto self_sizes = self.sizes().vec();
  auto source_sizes = source.sizes().vec();
  if (source.dim() != 0 && self.dim() != 0) {
    self_sizes.erase(self_sizes.begin() + dim);
    source_sizes.erase(source_sizes.begin() + dim);
  }
  TORCH_CHECK(
      self_sizes == source_sizes,
      "source tensor shape must match self tensor shape, excluding the specified dimension. Got self.shape = ",
      self.sizes(),
      " source.shape = ",
      source.sizes());

  bool is_defined = result.defined();

  // set_output_raw_strided
  auto options = self.options();
  auto sizes = self.sizes();
  if (is_defined) {
    at::xpu::resize_out(result, sizes, {}, options);
  } else {
    result = at::xpu::create_out(sizes, {}, options);
  }
  if (is_defined) {
    at::assert_no_internal_overlap(result);
    at::assert_no_overlap(result, index);
    at::assert_no_overlap(result, source);
  }

  // A hack to run TensorIterator checks in the meta function.
  // See comment:
  // https://github.com/pytorch/pytorch/pull/65993#discussion_r760307417
  // TODO: (@krshrimali) Try inheriting from TensorIteratorBase instead.
  if (result.device() == kMeta && result.dim() > 0) {
    auto selfSlice = result.select(dim, 0);
    auto sourceSlice = source.select(dim, 0);
    auto iter =
        TensorIterator::borrowing_binary_op(selfSlice, selfSlice, sourceSlice);
  }
}

Tensor& XPUNativeFunctions::index_add_out(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& source,
    const Scalar& alpha,
    Tensor& out) {
  std::optional<Device> common_device = std::nullopt;
  c10::impl::check_and_update_common_device(
      common_device, self, "xpu::index_add_out", "self");
  c10::impl::check_and_update_common_device(
      common_device, index, "xpu::index_add_out", "index");
  c10::impl::check_and_update_common_device(
      common_device, source, "xpu::index_add_out", "source");
  dim = maybe_wrap_dim(dim, self.dim());
  index_func_meta_impl(out, self, dim, index, source, "index_add");
  native::xpu::index_add_kernel(self, dim, index, source, alpha, out);
  return out;
}

Tensor& XPUNativeFunctions::index_add_(
    Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& source,
    const Scalar& alpha) {
  return index_add_out(self, dim, index, source, alpha, self);
}

Tensor XPUNativeFunctions::index_add(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& source,
    const Scalar& alpha) {
  Tensor out;
  return index_add_out(self, dim, index, source, alpha, out);
}

void check_indices_on_cpu_or_selfdevice(
    const Tensor& self,
    const c10::List<c10::optional<Tensor>>& indices) {
  auto dev = self.device();
  bool indices_on_cpu_or_dev = std::all_of(
      indices.begin(), indices.end(), [=](const c10::optional<Tensor>& opt) {
        if (opt.has_value()) {
          // for optional<Undefined tensor> cases
          if (!opt->defined()) {
            return true;
          }
          return (opt->is_cpu() || opt->device() == dev);
        } else {
          return true;
        }
      });
  TORCH_CHECK(
      indices_on_cpu_or_dev,
      "indices should be either on ",
      at::kCPU,
      " or on the same device as the indexed tensor (",
      dev,
      ")");
}

static void build_index_op(
    TensorIteratorBase& iter,
    const native::AdvancedIndex& info,
    Tensor& result) {
  TensorIteratorConfig config;
  // info.src is a restrided view of result
  config.set_check_mem_overlap(false)
      .check_all_same_dtype(false)
      .add_output(result)
      .add_input(info.src);
  for (auto& index : info.indices) {
    config.add_owned_const_input(index);
  }
  if (!result.defined()) {
    config.declare_static_dtype_and_device(
        info.src.scalar_type(), info.src.device());
  }
  iter.build(config);
}

Tensor& XPUNativeFunctions::index_out(
    const Tensor& self,
    const c10::List<c10::optional<Tensor>>& indices,
    Tensor& result) {
  TORCH_CHECK(
      indices.size() <= (size_t)self.dim(),
      "too many indices for tensor of dimension ",
      self.dim(),
      " (got ",
      indices.size(),
      ")");

  check_indices_on_cpu_or_selfdevice(self, indices);

  if (result.defined()) {
    TORCH_CHECK(
        self.scalar_type() == result.scalar_type(),
        "index_out: self (",
        self.scalar_type(),
        ") and result (",
        result.scalar_type(),
        ") must have the same scalar type");
    at::assert_no_internal_overlap(result);
    at::assert_no_overlap(result, self);
    for (const c10::optional<Tensor>& index : indices) {
      if (index.has_value()) {
        at::assert_no_overlap(result, *index);
      }
    }
  }
  auto info = native::make_info(self, std::move(indices));
  TensorIterator iter;
  build_index_op(iter, info, result);

  native::xpu::index_kernel(
      iter,
      info.indexed_sizes,
      info.indexed_strides,
      IntArrayRef{},
      IntArrayRef{});

  return result;
}

Tensor XPUNativeFunctions::index(
    const Tensor& self,
    const c10::List<c10::optional<Tensor>>& indices) {
  Tensor result;
  TORCH_CHECK(
      indices.size() <= (size_t)self.dim(),
      "too many indices for tensor of dimension ",
      self.dim(),
      " (got ",
      indices.size(),
      ")");

  check_indices_on_cpu_or_selfdevice(self, indices);

  auto info = native::make_info(self, std::move(indices));
  TensorIterator iter;
  build_index_op(iter, info, result);

  native::xpu::index_kernel(
      iter,
      info.indexed_sizes,
      info.indexed_strides,
      IntArrayRef{},
      IntArrayRef{});

  return iter.output();
}

// PyTorch defines it in cpp source. Copy it.
static TensorIterator make_index_put_iterator(
    const native::AdvancedIndex& info,
    const Tensor& value) {
  TORCH_CHECK(
      is_expandable_to(value.sizes(), info.src.sizes()),
      "shape mismatch: value tensor of shape ",
      value.sizes(),
      " cannot be broadcast to indexing result of shape ",
      info.src.sizes());
  TORCH_CHECK(
      value.scalar_type() == info.src.scalar_type(),
      "Index put requires the source and destination dtypes match, "
      "got ",
      info.src.scalar_type(),
      " for the destination "
      "and ",
      value.scalar_type(),
      " for the source.");
  TensorIteratorConfig config;
  // info.src is restrided by restride_src with 0 strided dimensions
  config.set_check_mem_overlap(false);
  config.resize_outputs(false);
  config.check_all_same_dtype(false);
  config.add_output(info.src);
  config.add_input(value);
  for (auto& index : info.indices) {
    config.add_input(index);
  }
  return config.build();
}

Tensor& XPUNativeFunctions::_index_put_impl_(
    Tensor& self,
    const torch::List<c10::optional<Tensor>>& indices,
    const Tensor& value,
    const bool accumulate,
    const bool unsafe) {
  TORCH_CHECK_INDEX(
      indices.size() <= (size_t)self.dim(),
      "too many indices for tensor of dimension ",
      self.dim(),
      " (got ",
      indices.size(),
      ")");
  if (at::has_internal_overlap(self) == MemOverlap::Yes) {
    TORCH_WARN(
        "Use of index_put_ on expanded tensors is deprecated. "
        "Please clone() the tensor before performing this operation. "
        "This also applies to advanced indexing e.g. tensor[indices] = tensor");
  }
  if (!accumulate) {
    auto masked_fill_dispatch =
        native::canDispatchToMaskedFill(self, indices, value);
    if (std::get<0>(masked_fill_dispatch)) {
      return self.masked_fill_(std::get<1>(masked_fill_dispatch), value.item());
    }
  }
  auto value_ = value;
  if (value.device() != self.device() && value.numel() == 1 &&
      value.dim() == 0) {
    value_ = value.to(self.device());
  }
  at::assert_no_overlap(self, value);
  // NOLINTNEXTLINE(performance-implicit-conversion-in-loop)
  for (const c10::optional<Tensor>& index : indices) {
    if (index.has_value()) {
      at::assert_no_overlap(self, *index);
    }
  }

  // Performance consideration:
  // Avoid atomic operations when accumulating bf16 and hf16. No efficient
  // atomic operation hardware support. We have to do CAS, whose performance
  // is worse than deterministic implementation.
  bool need_use_deterministic = (accumulate &&
                                 (self.scalar_type() == at::kBFloat16 ||
                                  self.scalar_type() == at::kHalf)) ||
      globalContext().deterministicAlgorithms();

  if (need_use_deterministic) {
    TORCH_CHECK(
        value_.device() == self.device(),
        "expected device ",
        self.device(),
        " but got device ",
        value_.device(),
        " for value tensor");
    native::xpu::index_put_deterministic_kernel(
        self, indices, value_, accumulate, unsafe);
    return self;
  }

  auto info = native::make_info(self, indices);
  auto iter = make_index_put_iterator(info, value_);
  native::xpu::index_put_kernel(
      iter,
      info.indexed_sizes,
      info.indexed_strides,
      IntArrayRef{},
      IntArrayRef{},
      accumulate);
  return self;
}

} // namespace at
