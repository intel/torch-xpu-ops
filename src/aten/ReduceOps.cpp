#include <ATen/ScalarOps.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/Fill.h>
#include <ATen/native/TensorIterator.h>
#include <torch/library.h>

#include <aten/sycl/Scan.h>

namespace at {
namespace native {
namespace xpu {
Tensor& cumsum_out(const Tensor& self, Dimname dim, c10::optional<ScalarType> dtype) {
  auto iter = TensorIteratorConfig()
                  .set_check_mem_overlap(
                      false) // Fill is idempotent, so overlap is okay
                  .check_all_same_dtype(false)
                  .add_output(self)
                  .resize_outputs(false)
                  .build();
  native::xpu::fill_kernel(iter, value);
  return self;
}
} // xpu
} // native
} // at