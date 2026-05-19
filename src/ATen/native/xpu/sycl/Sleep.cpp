#include <c10/xpu/XPUStream.h>
#include <comm/SYCLContext.h>

namespace at::xpu {
namespace {

struct SpinKernelFunctor {
  void operator()() const {
#if 0
    namespace syclex = sycl::ext::oneapi::experimental;
    uint64_t start_clock = syclex::clock<syclex::clock_scope::device>();
    uint64_t clock_offset = 0;
    while (clock_offset < cycles_) {
      clock_offset = syclex::clock<syclex::clock_scope::device>() - start_clock;
    }
#else
    // Fallback: spin on a volatile counter when the device clock extension is
    // unavailable. The iteration count is only a rough stand-in for real
    // cycles.
    volatile uint64_t remaining = cycles_;
    while (remaining > 0) {
      remaining = remaining - 1;
    }
#endif
  }

  SpinKernelFunctor(uint64_t cycles) : cycles_(cycles) {}

 private:
  uint64_t cycles_;
};

} // namespace

TORCH_XPU_API void sleep(uint64_t cycles) {
  SpinKernelFunctor kfn(cycles);
  sycl_kernel_submit(sycl_single_task, getCurrentSYCLQueue(), kfn);
}

} // namespace at::xpu
