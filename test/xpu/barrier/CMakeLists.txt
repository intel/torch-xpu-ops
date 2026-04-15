set(TEST_XPU_BARRIER_ROOT ${TORCH_XPU_OPS_ROOT}/test/xpu/barrier)

set(TEST_XPU_BARRIER_SYCL_SRCS
  ${TEST_XPU_BARRIER_ROOT}/barrier_impl_xpu.cpp)
set(TEST_XPU_BARRIER_CXX_SRCS
  ${TEST_XPU_BARRIER_ROOT}/barrier_perf_ut.cpp)

sycl_add_executable(
  test_xpu_barrier_perf_ut
  SYCL_SOURCES ${TEST_XPU_BARRIER_SYCL_SRCS}
  CXX_SOURCES ${TEST_XPU_BARRIER_CXX_SRCS})

find_library(ZE_LOADER_LIBRARY NAMES ze_loader)
if(ZE_LOADER_LIBRARY)
  target_link_libraries(test_xpu_barrier_perf_ut ${ZE_LOADER_LIBRARY})
endif()

if(INSTALL_TEST)
  install(TARGETS test_xpu_barrier_perf_ut DESTINATION bin)
endif()
