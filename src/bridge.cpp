#include <windows.h>
#include <iostream>

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
    if (!LoadLibrary(PATH_TO_TORCH_XPU_OPS_ATEN_LIB)) {
      std::cout << "PyTorch XPU operators library is not loaded ... "
          << "Please check if PyTorch is installed correctly. "
          << "Or file an issue on https://github.com/intel/torch-xpu-ops/issues."
          << std::endl;
    }
  }
};

static LoadTorchXPUOps init;

} // namespace
