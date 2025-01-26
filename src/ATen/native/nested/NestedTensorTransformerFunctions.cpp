#include <ATen/ATen.h>
#include <ATen/native/nested/NestedTensorTransformerFunctions.h>

namespace at::native {

Tensor NestedTensor_softmax_dropout_xpu(
    const Tensor& self,
    const Tensor& query) {
  std::optional<Tensor> attn_mask;

  attn_mask = NestedTensor_to_mask(query, 2, self.size(2));
  attn_mask = attn_mask->to(query.device(), /*non-blocking=*/true);
  return _masked_softmax(
      self,
      *attn_mask,
      self.dim() - 1,
      /*mask type */ 1); // NestedTensor_to_mask produces a BxT mask
}

} // namespace at::native