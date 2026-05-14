set(TEST_XPU_CSRC_ROOT ${TORCH_XPU_OPS_ROOT}/test/xpu/csrc)
add_subdirectory(${TEST_XPU_CSRC_ROOT} ${CMAKE_BINARY_DIR}/test_xpu_csrc)
