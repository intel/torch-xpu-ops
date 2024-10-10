#include <windows.h>
#include <sstream>
#include <stdexcept>

// The implementation helps walk around cyclic dependence, when we separate
// kernels into multiple dll/so to avoid a large bin (>2GB).
// The file is built into libtorch_xpu.so. libtorch_xpu.so won't depend on
// libtorch_xpu_ops_aten.so but dlopen the library to walk around cyclic
// dependence during linkage. To break cycle like,
// libtorch_xpu.so -> (dlopen) libtorch_xpu_ops_aten.so -> (link)
// libtorch_xpu_ops_kernels.so
//                                                      -> (link)
//                                                      libtorch_xpu_ops_unary_binary_kernels.so
// libtorch_xpu_ops_kernels.so -> (link) libtorch_xpu.so
// libtorch_xpu_ops_unary_binary_kernels.so -> (link) libtorch_xpu.so
namespace {

class LoadTorchXPUOps {
 public:
  LoadTorchXPUOps() {
    if (NULL == LoadLibrary(PATH_TO_TORCH_XPU_OPS_ATEN_LIB)) {
      std::ostringstream error;
      error
          << "PyTorch XPU operators library is not loaded (ERROR: "
          << GetLastError()
          << "). Please check if PyTorch is installed correctly."
          << " Or please file an issue on https://github.com/intel/torch-xpu-ops/issues.";
      throw std::runtime_error(error.str());
    }
  }
};

static LoadTorchXPUOps init;

} // namespace
