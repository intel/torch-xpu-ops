#include <dlfcn.h>
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
    void* fd = dlopen(PATH_TO_TORCH_XPU_OPS_ATEN_LIB, RTLD_NOW);
    if (!fd) {
      std::cout
          << "Could not load ATen XPU backend: " << dlerror()
          << ". PyTorch operators could not work on XPU device ... Please check if libraries of PyTorch XPU backend are installed correctly or file an issue on https://github.com/intel/torch-xpu-ops/issues"
          << std::endl;
    }
  }
};

static LoadTorchXPUOps init;

} // namespace
