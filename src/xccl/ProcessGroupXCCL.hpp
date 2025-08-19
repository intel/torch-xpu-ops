#ifdef BUILD_XCCL_V1
#include <torch/csrc/distributed/c10d/ProcessGroupXCCL_v1.hpp>
#else
#include <torch/csrc/distributed/c10d/ProcessGroupXCCL_v2.hpp>
#endif
