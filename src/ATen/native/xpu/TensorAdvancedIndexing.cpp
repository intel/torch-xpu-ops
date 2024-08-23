#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/ScalarOps.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/IndexingUtils.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/ReductionType.h>
#include <ATen/native/ScatterGatherChecks.h>
#include <ATen/native/TensorAdvancedIndexing.h>
#include <ATen/native/TensorAdvancedIndexingUtils.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/IndexingKernels.h>
#include <ATen/native/xpu/sycl/ScatterGatherKernels.h>
#include <ATen/xpu/XPUNativeFunctions.h>
#include <comm/ReduceOpsUtils.h>

namespace at {

using namespace at::native;
using namespace at::native::xpu;

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

Tensor& XPUNativeFunctions::index_fill_(
    Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Scalar& source) {
  at::NoNamesGuard guard;

  TORCH_CHECK_INDEX(
      index.scalar_type() == ScalarType::Long,
      "index_fill_(): Expected dtype int64 for index.");

  at::assert_no_overlap(self, index);
  if (at::has_internal_overlap(self) == at::MemOverlap::Yes) {
    TORCH_WARN(
        "Use of index_fill_ on expanded tensors is deprecated. "
        "Please clone() the tensor before performing this operation. "
        "This also applies to advanced indexing e.g. tensor[mask] = scalar");
  }

  if (!self.is_complex() && source.isComplex()) {
    TORCH_CHECK(
        false,
        "index_fill_(): Converting complex Scalar to non-complex type is not supported");
  }

  TORCH_CHECK(
      self.device() == index.device(),
      "index_fill_(): self and index value tensors ",
      "should have same device type, but got self tensor device type ",
      self.device(),
      " and index value ",
      "tensor device type ",
      index.device());

  // Handle the case when `self` is 0-dim
  Tensor self_nonzero_dim = (self.dim() == 0) ? self.unsqueeze(-1) : self;
  dim = at::maybe_wrap_dim(dim, self_nonzero_dim);

  native::xpu::index_fill_kernel(self, dim, index, source);
  return self;
}

Tensor& XPUNativeFunctions::index_fill_(
    Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& source) {
  TORCH_CHECK(
      source.dim() == 0,
      "index_fill_ only supports a 0-dimensional value tensor, but got tensor "
      "with ",
      source.dim(),
      " dimension(s).");
  return self.index_fill_(dim, index, source.item());
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

  if (accumulate || globalContext().deterministicAlgorithms()) {
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

// ============================= scatter =============================

static void scatter_reduce_exclude_self_helper(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const ReductionType& op) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Bool,
      self.scalar_type(),
      "scatter_reduce_exclude_input_init",
      [&] {
        scalar_t init_val;
        switch (op) {
          case ReductionType::SUM:
            init_val = (scalar_t)0;
            break;
          case ReductionType::PROD:
            init_val = (scalar_t)1;
            break;
          case ReductionType::MAX:
            init_val = std::numeric_limits<scalar_t>::has_infinity
                ? -std::numeric_limits<scalar_t>::infinity()
                : std::numeric_limits<scalar_t>::lowest();
            break;
          case ReductionType::MIN:
            init_val = std::numeric_limits<scalar_t>::has_infinity
                ? std::numeric_limits<scalar_t>::infinity()
                : std::numeric_limits<scalar_t>::max();
            break;
          case ReductionType::MEAN:
            init_val = (scalar_t)0;
            break;
        }
        self.scatter_(dim, index, init_val);
      });
}

static void _scatter_via_index_put(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src,
    const Tensor& mut_out,
    bool accumulate) {
  if (self.dim() == 1) {
    torch::List<c10::optional<Tensor>> indices;
    indices.reserve(1);
    indices.push_back(index);
    mut_out.index_put_(indices, src, accumulate);
  } else {
    Tensor mut_out_contig = mut_out.contiguous();

    auto index_coords_sizes = index.sizes().vec();
    index_coords_sizes.push_back(self.dim());
    auto index_coords = at::empty(
        index_coords_sizes,
        at::TensorOptions().dtype(at::ScalarType::Long).device(self.device()));

    for (int64_t dim_other = 0; dim_other < self.dim(); dim_other++) {
      if (dim_other == dim) {
        continue;
      }
      auto dim_coord_vals = at::arange(
          index.size(dim_other), at::TensorOptions().device(self.device()));

      for (int64_t dim_unsqueeze = 0; dim_unsqueeze < self.dim() - 1;
           dim_unsqueeze++) {
        dim_coord_vals =
            dim_coord_vals.unsqueeze((dim_unsqueeze >= dim_other) ? -1 : 0);
      }

      auto view_sizes = index.sizes().vec();
      view_sizes.push_back(1);
      auto view_strides = index_coords.strides().vec();
      view_strides[self.dim()] = self.dim();

      at::as_strided(index_coords, view_sizes, view_strides, dim_other)
          .copy_(dim_coord_vals.unsqueeze(-1));
    }

    auto view_sizes = index.sizes().vec();
    view_sizes.push_back(1);
    auto view_strides = index_coords.strides().vec();
    view_strides[self.dim()] = self.dim();

    at::as_strided(index_coords, view_sizes, view_strides, dim)
        .copy_(index.unsqueeze(-1));

    Tensor index_coords_flat = index_coords.flatten(0, -2);

    // Copy mut_out_contig's strides into a tensor
    // TODO: Is there a utility function that already does this?
    IntArrayRef mut_out_contig_strides = mut_out_contig.strides();
    Tensor coord_strides = at::empty(
        {mut_out_contig.dim()},
        TensorOptions().dtype(at::ScalarType::Long).device(at::kCPU));
    std::memcpy(
        coord_strides.mutable_data_ptr(),
        mut_out_contig_strides.data(),
        coord_strides.nbytes());
    coord_strides = coord_strides.to(mut_out_contig.device());

    // `index_flat` contains the 1-D indices corresponding with the
    // flattened `mut_out`
    Tensor index_flat = (index_coords_flat * coord_strides).sum({-1});
    Tensor mut_out_flat = mut_out_contig.flatten();
    Tensor src_flat =
        at::as_strided(src, index.sizes(), src.strides()).flatten();

    torch::List<c10::optional<Tensor>> indices;
    indices.reserve(1);
    indices.push_back(index_flat);

    mut_out_flat.index_put_(indices, src_flat, accumulate);

    if (!mut_out.is_contiguous()) {
      mut_out.copy_(mut_out_flat.reshape(mut_out.sizes()));
    }
  }
}

template <
    bool use_new_options = false,
    typename T,
    typename ReduceStub,
    typename FillStub>
void scatter_impl(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const T& src,
    const Tensor& out,
    ReduceStub& reduce_stub,
    FillStub& fill_stub,
    const c10::optional<c10::string_view> reduce = nullopt,
    bool reduce_includes_self = true) {
  dim = at::maybe_wrap_dim(dim, self.dim());
  auto mut_out = const_cast<Tensor&>(out);

  if (!self.is_same(mut_out)) {
    mut_out.copy_(self);
  }

  if (index.numel() == 0)
    return;

  auto op = ReductionType::SUM;
  bool deterministic = globalContext().deterministicAlgorithms() &&
      self.device().type() == DeviceType::XPU;

  if (reduce.has_value()) {
    op = get_operator_enum(reduce.value(), use_new_options);
    if (!reduce_includes_self) {
      // scatter inits for reduction to appropriate indices (used by
      // scatter_reduce.two)
      scatter_reduce_exclude_self_helper(mut_out, dim, index, op);
    }
    // _scatter_via_index_put can only handle sum and mean reduction type
    deterministic = deterministic &&
        (op == ReductionType::SUM || op == ReductionType::MEAN);
  }

  // Scalar src should already be deterministic
  if (deterministic && std::is_same_v<T, Tensor>) {
    // both runtime and compile check are required
    if constexpr (std::is_same_v<T, Tensor>) {
      bool accumulate = reduce.has_value();
      _scatter_via_index_put(self, dim, index, src, mut_out, accumulate);
      return;
    }
  }

  if (reduce.has_value()) {
    reduce_stub(mut_out, dim, index, src, op);
  } else {
    fill_stub(mut_out, dim, index, src);
  }
}

template <bool use_new_options = false>
Tensor& scatter_meta_impl(
    Tensor& output,
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const c10::optional<Tensor>& src = nullopt,
    const c10::optional<c10::string_view> reduce = nullopt) {
  int64_t wrapped_dim = at::maybe_wrap_dim(dim, self.dim());
  at::native::scatter_gather_dtype_check("scatter", self, index, src);
  at::native::scatter_shape_check(self, wrapped_dim, index, src);

  if (output.defined()) {
    at::assert_no_internal_overlap(output);
    at::assert_no_overlap(output, index);
    if (src.has_value()) {
      at::assert_no_overlap(output, src.value());
    }
  }

  if (output.defined()) {
    at::xpu::resize_out(output, self.sizes(), {}, self.options());
  } else {
    output = at::xpu::create_out(self.sizes(), {}, self.options());
  }

  if (reduce.has_value()) {
    // Check if we have a valid reduce operator.
    at::native::get_operator_enum(reduce.value(), use_new_options);
  }

  return output;
}

Tensor& scatter_src_meta(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src,
    Tensor& out) {
  return scatter_meta_impl(out, self, dim, index, src);
}

Tensor& scatter_value_meta(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Scalar& value,
    Tensor& out) {
  return scatter_meta_impl(out, self, dim, index);
}

Tensor& scatter_reduce_meta(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src,
    const c10::string_view reduce,
    Tensor& out) {
  TORCH_WARN_ONCE(
      "The reduce argument of torch.scatter with Tensor src is deprecated and will be removed ",
      "in a future PyTorch release. Use torch.scatter_reduce instead for more reduction options.");
  return scatter_meta_impl(out, self, dim, index, src, reduce);
}

Tensor& scatter_value_reduce_meta(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Scalar& src,
    const c10::string_view reduce,
    Tensor& out) {
  return scatter_meta_impl(out, self, dim, index, nullopt, reduce);
}

Tensor& scatter_add_meta(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src,
    Tensor& out) {
  return scatter_meta_impl(out, self, dim, index, src, "add");
}

Tensor& scatter_reduce_two_meta(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src,
    const c10::string_view reduce,
    bool include_self,
    Tensor& out) {
  (void)include_self;
  return scatter_meta_impl</*use_new_options=*/true>(
      out, self, dim, index, src, reduce);
}

Tensor XPUNativeFunctions::scatter(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src) {
  std::optional<Device> common_device = std::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, self, "xpu::scatter_src", "self");
  c10::impl::check_and_update_common_device(
      common_device, index, "xpu::scatter_src", "index");
  c10::impl::check_and_update_common_device(
      common_device, src, "xpu::scatter_src", "src");
  Tensor out;
  out = scatter_src_meta(self, dim, index, src, out);
  scatter_impl(
      self, dim, index, src, out, scatter_reduce_kernel, scatter_kernel);
  return out;
}

Tensor& XPUNativeFunctions::scatter_out(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src,
    Tensor& out) {
  std::optional<Device> common_device = std::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, out, "xpu::scatter_out_src_out", "out");
  c10::impl::check_and_update_common_device(
      common_device, self, "xpu::scatter_out_src_out", "self");
  c10::impl::check_and_update_common_device(
      common_device, index, "xpu::scatter_out_src_out", "index");
  c10::impl::check_and_update_common_device(
      common_device, src, "xpu::scatter_out_src_out", "src");
  out = scatter_src_meta(self, dim, index, src, out);
  scatter_impl(
      self, dim, index, src, out, scatter_reduce_kernel, scatter_kernel);
  return out;
}

Tensor& XPUNativeFunctions::scatter_(
    Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src) {
  std::optional<Device> common_device = std::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, self, "xpu::scatter__src", "self");
  c10::impl::check_and_update_common_device(
      common_device, index, "xpu::scatter__src", "index");
  c10::impl::check_and_update_common_device(
      common_device, src, "xpu::scatter__src", "src");
  self = scatter_src_meta(self, dim, index, src, self);
  scatter_impl(
      self, dim, index, src, self, scatter_reduce_kernel, scatter_kernel);
  return self;
}

Tensor XPUNativeFunctions::scatter(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Scalar& value) {
  std::optional<Device> common_device = std::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, self, "xpu::scatter_value", "self");
  c10::impl::check_and_update_common_device(
      common_device, index, "xpu::scatter_value", "index");
  Tensor out;
  out = scatter_value_meta(self, dim, index, value, out);
  scatter_impl(
      self,
      dim,
      index,
      value,
      out,
      scatter_scalar_reduce_kernel,
      scatter_fill_kernel);
  return out;
}

Tensor& XPUNativeFunctions::scatter_out(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Scalar& value,
    Tensor& out) {
  std::optional<Device> common_device = std::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, out, "xpu::scatter_out_value_out", "out");
  c10::impl::check_and_update_common_device(
      common_device, self, "xpu::scatter_out_value_out", "self");
  c10::impl::check_and_update_common_device(
      common_device, index, "xpu::scatter_out_value_out", "index");
  out = scatter_value_meta(self, dim, index, value, out);
  scatter_impl(
      self,
      dim,
      index,
      value,
      out,
      scatter_scalar_reduce_kernel,
      scatter_fill_kernel);
  return out;
}

Tensor& XPUNativeFunctions::scatter_(
    Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Scalar& value) {
  std::optional<Device> common_device = std::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, self, "xpu::scatter__value", "self");
  c10::impl::check_and_update_common_device(
      common_device, index, "xpu::scatter__value", "index");
  self = scatter_value_meta(self, dim, index, value, self);
  scatter_impl(
      self,
      dim,
      index,
      value,
      self,
      scatter_scalar_reduce_kernel,
      scatter_fill_kernel);
  return self;
}

Tensor XPUNativeFunctions::scatter(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src,
    c10::string_view reduce) {
  std::optional<Device> common_device = std::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, self, "xpu::scatter_reduce", "self");
  c10::impl::check_and_update_common_device(
      common_device, index, "xpu::scatter_reduce", "index");
  c10::impl::check_and_update_common_device(
      common_device, src, "xpu::scatter_reduce", "src");
  Tensor out;
  out = scatter_reduce_meta(self, dim, index, src, reduce, out);
  scatter_impl(
      self,
      dim,
      index,
      src,
      out,
      scatter_reduce_kernel,
      scatter_kernel,
      reduce);
  return out;
}

Tensor& XPUNativeFunctions::scatter_out(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src,
    c10::string_view reduce,
    Tensor& out) {
  std::optional<Device> common_device = std::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, out, "xpu::scatter_out_reduce_out", "out");
  c10::impl::check_and_update_common_device(
      common_device, self, "xpu::scatter_out_reduce_out", "self");
  c10::impl::check_and_update_common_device(
      common_device, index, "xpu::scatter_out_reduce_out", "index");
  c10::impl::check_and_update_common_device(
      common_device, src, "xpu::scatter_out_reduce_out", "src");
  out = scatter_reduce_meta(self, dim, index, src, reduce, out);
  scatter_impl(
      self,
      dim,
      index,
      src,
      out,
      scatter_reduce_kernel,
      scatter_kernel,
      reduce);
  return out;
}

Tensor& XPUNativeFunctions::scatter_(
    Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src,
    c10::string_view reduce) {
  std::optional<Device> common_device = std::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, self, "xpu::scatter__reduce", "self");
  c10::impl::check_and_update_common_device(
      common_device, index, "xpu::scatter__reduce", "index");
  c10::impl::check_and_update_common_device(
      common_device, src, "xpu::scatter__reduce", "src");
  self = scatter_reduce_meta(self, dim, index, src, reduce, self);
  scatter_impl(
      self,
      dim,
      index,
      src,
      self,
      scatter_reduce_kernel,
      scatter_kernel,
      reduce);
  return self;
}

Tensor XPUNativeFunctions::scatter(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Scalar& value,
    c10::string_view reduce) {
  std::optional<Device> common_device = std::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, self, "xpu::scatter_value_reduce", "self");
  c10::impl::check_and_update_common_device(
      common_device, index, "xpu::scatter_value_reduce", "index");
  Tensor out;
  out = scatter_value_reduce_meta(self, dim, index, value, reduce, out);
  scatter_impl(
      self,
      dim,
      index,
      value,
      out,
      scatter_scalar_reduce_kernel,
      scatter_fill_kernel,
      reduce);
  return out;
}

Tensor& XPUNativeFunctions::scatter_out(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Scalar& value,
    c10::string_view reduce,
    Tensor& out) {
  std::optional<Device> common_device = std::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, out, "xpu::scatter_out_value_reduce_out", "out");
  c10::impl::check_and_update_common_device(
      common_device, self, "xpu::scatter_out_value_reduce_out", "self");
  c10::impl::check_and_update_common_device(
      common_device, index, "xpu::scatter_out_value_reduce_out", "index");
  out = scatter_value_reduce_meta(self, dim, index, value, reduce, out);
  scatter_impl(
      self,
      dim,
      index,
      value,
      out,
      scatter_scalar_reduce_kernel,
      scatter_fill_kernel,
      reduce);
  return out;
}

Tensor& XPUNativeFunctions::scatter_(
    Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Scalar& value,
    c10::string_view reduce) {
  std::optional<Device> common_device = std::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, self, "xpu::scatter__value_reduce", "self");
  c10::impl::check_and_update_common_device(
      common_device, index, "xpu::scatter__value_reduce", "index");
  self = scatter_value_reduce_meta(self, dim, index, value, reduce, self);
  scatter_impl(
      self,
      dim,
      index,
      value,
      self,
      scatter_scalar_reduce_kernel,
      scatter_fill_kernel,
      reduce);
  return self;
}

Tensor& scatter_add_impl(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src,
    Tensor& out) {
  auto mut_out = const_cast<Tensor&>(out);
  dim = maybe_wrap_dim(dim, self.dim());

  if (!self.is_same(mut_out)) {
    mut_out.copy_(self);
  }

  if (index.numel() == 0)
    return out;

  // See Note [Enabling Deterministic Operations]
  // Avoid gpuAtomicAdd for XPU if deterministic mode is turned on
  if (globalContext().deterministicAlgorithms() &&
      self.device().type() == DeviceType::XPU) {
    _scatter_via_index_put(self, dim, index, src, mut_out, /*accumulate*/ true);
  } else {
    // TODO: enable fast paths for GNN usage (scatter_add_expanded_index_kernel)
    scatter_add_kernel(mut_out, dim, index, src);
  }
  return out;
}

Tensor XPUNativeFunctions::scatter_add(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src) {
  std::optional<Device> common_device = std::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, self, "xpu::scatter_add", "self");
  c10::impl::check_and_update_common_device(
      common_device, index, "xpu::scatter_add", "index");
  c10::impl::check_and_update_common_device(
      common_device, src, "xpu::scatter_add", "src");
  Tensor out;
  out = scatter_add_meta(self, dim, index, src, out);
  out = scatter_add_impl(self, dim, index, src, out);
  return out;
}

Tensor& XPUNativeFunctions::scatter_add_out(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src,
    Tensor& out) {
  std::optional<Device> common_device = std::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, out, "xpu::scatter_add_out_out", "out");
  c10::impl::check_and_update_common_device(
      common_device, self, "xpu::scatter_add_out_out", "self");
  c10::impl::check_and_update_common_device(
      common_device, index, "xpu::scatter_add_out_out", "index");
  c10::impl::check_and_update_common_device(
      common_device, src, "xpu::scatter_add_out_out", "src");
  out = scatter_add_meta(self, dim, index, src, out);
  out = scatter_add_impl(self, dim, index, src, out);
  return out;
}

Tensor& XPUNativeFunctions::scatter_add_(
    Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src) {
  std::optional<Device> common_device = std::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, self, "xpu::scatter_add_", "self");
  c10::impl::check_and_update_common_device(
      common_device, index, "xpu::scatter_add_", "index");
  c10::impl::check_and_update_common_device(
      common_device, src, "xpu::scatter_add_", "src");
  self = scatter_add_meta(self, dim, index, src, self);
  self = scatter_add_impl(self, dim, index, src, self);
  return self;
}

Tensor& scatter_reduce_two_impl(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src,
    const c10::string_view reduce,
    bool include_self,
    Tensor& out) {
  dim = at::maybe_wrap_dim(dim, self.dim());

  if (!self.is_same(out)) {
    out.copy_(self);
  }

  const auto op = get_operator_enum(reduce, true);

  // TODO: enable scatter_reduce_expanded_index_kernel

  scatter_impl</*use_new_options=*/true>(
      self,
      dim,
      index,
      src,
      out,
      scatter_reduce_two_kernel,
      scatter_kernel,
      reduce,
      include_self);

  if (op == ReductionType::MEAN) {
    auto ones = at::ones_like(src);
    auto count = include_self ? at::ones_like(out) : at::zeros_like(out);
    count.scatter_add_(dim, index, ones);
    count.masked_fill_(count == 0, 1);

    if (out.is_floating_point() || out.is_complex()) {
      out.div_(count);
    } else {
      out.div_(count, "floor");
    }
  }

  return out;
}

Tensor XPUNativeFunctions::scatter_reduce(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src,
    c10::string_view reduce,
    bool include_self) {
  std::optional<Device> common_device = std::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, self, "xpu::scatter_reduce_two", "self");
  c10::impl::check_and_update_common_device(
      common_device, index, "xpu::scatter_reduce_two", "index");
  c10::impl::check_and_update_common_device(
      common_device, src, "xpu::scatter_reduce_two", "src");
  Tensor out;
  out =
      scatter_reduce_two_meta(self, dim, index, src, reduce, include_self, out);
  out =
      scatter_reduce_two_impl(self, dim, index, src, reduce, include_self, out);
  return out;
}

Tensor& XPUNativeFunctions::scatter_reduce_out(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src,
    c10::string_view reduce,
    bool include_self,
    Tensor& out) {
  std::optional<Device> common_device = std::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, out, "xpu::scatter_reduce_out_two_out", "out");
  c10::impl::check_and_update_common_device(
      common_device, self, "xpu::scatter_reduce_out_two_out", "self");
  c10::impl::check_and_update_common_device(
      common_device, index, "xpu::scatter_reduce_out_two_out", "index");
  c10::impl::check_and_update_common_device(
      common_device, src, "xpu::scatter_reduce_out_two_out", "src");
  out =
      scatter_reduce_two_meta(self, dim, index, src, reduce, include_self, out);
  out =
      scatter_reduce_two_impl(self, dim, index, src, reduce, include_self, out);
  return out;
}

Tensor& XPUNativeFunctions::scatter_reduce_(
    Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src,
    c10::string_view reduce,
    bool include_self) {
  std::optional<Device> common_device = std::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, self, "xpu::scatter_reduce__two", "self");
  c10::impl::check_and_update_common_device(
      common_device, index, "xpu::scatter_reduce__two", "index");
  c10::impl::check_and_update_common_device(
      common_device, src, "xpu::scatter_reduce__two", "src");
  self = scatter_reduce_two_meta(
      self, dim, index, src, reduce, include_self, self);
  self = scatter_reduce_two_impl(
      self, dim, index, src, reduce, include_self, self);
  return self;
}

// ============================= gather =============================

Tensor& gather_meta(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    bool sparse_grad,
    Tensor& result) {
  int64_t wrapped_dim = at::maybe_wrap_dim(dim, self.dim());

  // Memory overlap checks need to be done after resizing (if required) is done.
  // But it only makes sense to do these checks when result was defined, hence
  // the boolean variable `check_result` here.
  // For more details, see:
  // https://github.com/pytorch/pytorch/pull/63312#discussion_r694794832 and
  // https://github.com/pytorch/pytorch/issues/63837
  bool check_result = result.defined();

  if (result.defined()) {
    at::xpu::resize_out(result, index.sizes(), {}, self.options());
  } else {
    result = at::xpu::create_out(index.sizes(), {}, self.options());
  }

  if (check_result) {
    at::assert_no_internal_overlap(result);
    at::assert_no_overlap(result, self);
    at::assert_no_partial_overlap(result, index);
  }

  auto is_index_empty = index.numel() == 0;
  if (!is_index_empty) {
    TORCH_CHECK(
        index.scalar_type() == at::ScalarType::Long,
        "gather",
        "(): Expected dtype int64 for index");
  }
  if (is_index_empty)
    return result;
  at::native::gather_shape_check(self, wrapped_dim, index);

  return result;
}

Tensor XPUNativeFunctions::gather(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    bool sparse_grad) {
  std::optional<Device> common_device = std::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, self, "xpu::gather", "self");
  c10::impl::check_and_update_common_device(
      common_device, index, "xpu::gather", "index");
  Tensor out;
  out = gather_meta(self, dim, index, sparse_grad, out);

  if (index.numel() == 0)
    return out;
  dim = at::maybe_wrap_dim(dim, self.dim());
  // TODO: enable gather_expanded_index_kernel
  gather_kernel(out, self, dim, index);
  return out;
}

Tensor& XPUNativeFunctions::gather_out(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    bool sparse_grad,
    Tensor& out) {
  std::optional<Device> common_device = std::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, out, "xpu::gather_out_out", "out");
  c10::impl::check_and_update_common_device(
      common_device, self, "xpu::gather_out_out", "self");
  c10::impl::check_and_update_common_device(
      common_device, index, "xpu::gather_out_out", "index");
  out = gather_meta(self, dim, index, sparse_grad, out);

  if (index.numel() == 0)
    return out;
  dim = at::maybe_wrap_dim(dim, self.dim());
  // TODO: enable gather_expanded_index_kernel
  gather_kernel(out, self, dim, index);
  return out;
}

Tensor XPUNativeFunctions::count_nonzero(const Tensor& self, IntArrayRef dims) {
  return (self != 0).sum(dims);
}

} // namespace at
