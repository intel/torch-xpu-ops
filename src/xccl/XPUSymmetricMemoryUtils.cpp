#include <xccl/XPUSymmetricMemoryUtils.hpp>

#include <c10/util/error.h>

namespace c10d::symmetric_memory {

std::string getSymmMemBackendXPU() {
  // TORCH_SYMMMEM environment variable can be used to indicate the preferred
  // backend.
  static auto val = c10::utils::get_env("TORCH_SYMMMEM");
  if (val.has_value()) {
    TORCH_CHECK(
        val.value() == "XPU",
        "TORCH_SYMMMEM environment variable must be 'XPU'.");
    return val.value();
  }
  return "XPU";
}

} // namespace c10d::symmetric_memory
