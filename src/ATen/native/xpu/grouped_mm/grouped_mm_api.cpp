#include <ATen/native/xpu/grouped_mm/grouped_mm_api.h>

#include <c10/util/Exception.h>

#ifdef USE_SYCLTLA
#include <ATen/native/xpu/grouped_mm/sycltla/grouped_mm_sycltla.h>
#endif

namespace sycltla {

void grouped_mm_moe_forward(
    at::Tensor mat_a, // bf16
    at::Tensor mat_b, // bf16
    std::optional<at::Tensor> offs,
    std::optional<at::Tensor> bias, // BF16
    at::Tensor &out) {
#ifdef USE_SYCLTLA
  grouped_mm_moe_forward_sycltla(mat_a, mat_b, offs, bias, out);
#else
  TORCH_CHECK(
      false,
      "grouped_mm_moe_forward: torch-xpu-ops was built without USE_SYCLTLA");
#endif
}

} // namespace sycltla
