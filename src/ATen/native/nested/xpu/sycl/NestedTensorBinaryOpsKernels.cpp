#include <ATen/native/nested/NestedTensorBinaryOps.h>

#include <type_traits>

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>

#include <ATen/native/xpu/sycl/IndexUtils.h>
#include <ATen/native/xpu/sycl/KernelUtils.h>
#include <ATen/native/xpu/sycl/Loops.h>
#include <ATen/native/xpu/sycl/MemoryAccessUtils.h>
#include <ATen/xpu/XPUContext.h>

#include <comm/XPUMathCompat.h>

#include <ATen/native/nested/NestedTensorUtils.h>

#define GROUP_DIM 256

namespace at::native::xpu {

// only for nested [B, *, D], dense [B, 1, D]
template <typename T, typename func_t>
struct OpDenseEsuhmFunctor {
  void operator()(sycl::nd_item<1> item) const {
    // each batch is handled by a group
    const int64_t batch_idx = item.get_group(0);
    const int64_t grain_size = item.get_local_range(0);
    const int64_t tid = item.get_local_id(0);
    const int64_t range = offsets[batch_idx + 1] - offsets[batch_idx];
    // each thread handles (embedding_dim // grain_size + (embedding_dim %
    // grain_size <= tid)) elems of the dense embedding
    for (int64_t idx = tid; idx < embedding_dim; idx += grain_size) {
      const T dense_elem = dense[batch_idx * embedding_dim + idx];
      for (int64_t nested_idx = idx; nested_idx < range;
           nested_idx += embedding_dim) {
        output[offsets[batch_idx] + nested_idx] =
            f(input[offsets[batch_idx] + nested_idx], dense_elem);
      }
    }
  }

  OpDenseEsuhmFunctor(
      const T* input,
      const T* dense,
      T* output,
      int64_t embedding_dim,
      const int64_t* offsets,
      const func_t f)
      : input(input),
        dense(dense),
        output(output),
        embedding_dim(embedding_dim),
        offsets(offsets),
        f(f) {}

 private:
  const T* input;
  const T* dense;
  T* output;
  int64_t embedding_dim;
  const int64_t* offsets;
  const func_t f;
};

template <typename T, typename func_t>
void nested_op_dense_kernelLauncher(
    const T* input, // [sum(*) x embedding_dim]
    const T* dense, // [batch_size x embedding_dim]
    T* output, // [sum(*) x embedding_dim]
    int64_t batch_size,
    int64_t embedding_dim,
    const int64_t* input_offsets, // [batch_size]
    func_t f) {
  OpDenseEsuhmFunctor kfn(
      input, dense, output, embedding_dim, input_offsets, f);
  sycl_kernel_submit(
      GROUP_DIM * batch_size, GROUP_DIM, at::xpu::getCurrentSYCLQueue(), kfn);
}

template <typename scalar_t, typename func_t>
void _nested_op_dense_esuhm_kernel(
    Tensor& result,
    const Tensor& self,
    const Tensor& other,
    func_t f) {
  auto self_ptr = get_nested_tensor_impl(self);
  auto result_ptr = get_nested_tensor_impl(result);

  const auto self_buffer = self_ptr->get_buffer();
  const auto offsets = self_ptr->get_storage_offsets();
  const auto batch_size = other.size(0);
  const auto embedding_size = other.size(2);

  auto result_buffer = result_ptr->get_buffer();
  auto result_offsets = at::cat({offsets, at::tensor(self_ptr->numel())});
  result_offsets = result_offsets.to(kXPU);

  const scalar_t* self_data_ptr = self_buffer.const_data_ptr<scalar_t>();
  const scalar_t* other_data_ptr = other.const_data_ptr<scalar_t>();
  scalar_t* result_data_ptr = result_buffer.data_ptr<scalar_t>();
  int64_t* result_offsets_ptr = result_offsets.template data_ptr<int64_t>();

  nested_op_dense_kernelLauncher(
      self_data_ptr,
      other_data_ptr,
      result_data_ptr,
      batch_size,
      embedding_size,
      result_offsets_ptr,
      f);
}

template <typename scalar_t>
struct AddFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    return a + b;
  }
};

template <typename scalar_t>
struct MulFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    return a * b;
  }
};

void _nested_op_dense_esuhm_xpu(
    Tensor& result,
    const Tensor& self,
    const Tensor& other,
    const NESTED_DENSE_OP& op) {
  AT_DISPATCH_ALL_TYPES_AND2(
      ScalarType::Half,
      ScalarType::BFloat16,
      self.scalar_type(),
      "_nested_op_dense_esuhm",
      [&]() {
        switch (op) {
          case NESTED_DENSE_OP::ADD:
            _nested_op_dense_esuhm_kernel<scalar_t>(
                result, self, other, AddFunctor<scalar_t>{});
            break;
          case NESTED_DENSE_OP::MUL:
            _nested_op_dense_esuhm_kernel<scalar_t>(
                result, self, other, MulFunctor<scalar_t>{});
            break;
        }
      });
}

} // namespace at::native::xpu
