set(TEST_XPU_CSRC_ROOT ${TORCH_XPU_OPS_ROOT}/test/xpu/csrc)
set(TEST_XPU_CSRC_SRCS ${TEST_XPU_CSRC_ROOT}/LocalPermuteCopy.cpp)

sycl_add_library(
  local_permute_copy_test
  STATIC
  SYCL_SOURCES ${TEST_XPU_CSRC_SRCS})

target_include_directories(local_permute_copy_test
  PRIVATE
    ${TORCH_XPU_OPS_INCLUDE_DIRS})

if(INSTALL_TEST)
  install(TARGETS local_permute_copy_test DESTINATION lib)
endif()
