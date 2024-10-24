# Build on Linux

set(TORCH_XPU_OPS_LIBRARIES)
set(SYCL_LINK_LIBRARIES_KEYWORD PRIVATE)

add_library(
  torch_xpu_ops
  STATIC
  ${ATen_XPU_CPP_SRCS}
  ${ATen_XPU_MKL_SRCS}
  ${ATen_XPU_NATIVE_CPP_SRCS}
  ${ATen_XPU_GEN_SRCS})

if(BUILD_SEPARATE_OPS)
  foreach(sycl_src ${ATen_XPU_SYCL_SRCS})
    get_filename_component(name ${sycl_src} NAME_WLE REALPATH)
    set(sycl_lib torch-xpu-ops-sycl-${name})
    sycl_add_library(
      ${sycl_lib}
      SHARED
      SYCL_SOURCES ${sycl_src})
    target_link_libraries(torch_xpu_ops PUBLIC ${sycl_lib})
    list(APPEND TORCH_XPU_OPS_LIBRARIES ${sycl_lib})

    # Decouple with PyTorch cmake definition.
    install(TARGETS ${sycl_lib} DESTINATION "${TORCH_INSTALL_LIB_DIR}")
  endforeach()
else()
  # Split SYCL kernels into 2 libraries as categories 1) Unary+Binary 2) Others.
  set(ATen_XPU_SYCL_UNARY_BINARY_SRCS)
  set(ATen_XPU_SYCL_NON_UNARY_BINARY_SRCS)

  foreach(sycl_src ${ATen_XPU_SYCL_SRCS})
    string(REGEX MATCH "Binary" IS_BINARY ${sycl_src})
    string(REGEX MATCH "Unary" IS_UNARY ${sycl_src})
    if(IS_BINARY STREQUAL "" AND IS_UNARY STREQUAL "")
      list(APPEND ATen_XPU_SYCL_NON_UNARY_BINARY_SRCS ${sycl_src})
    else()
      list(APPEND ATen_XPU_SYCL_UNARY_BINARY_SRCS ${sycl_src})
    endif()
  endforeach()

  set(sycl_unary_binary_lib torch_xpu_ops_sycl_unary_binary_kernels)
  sycl_add_library(
    ${sycl_unary_binary_lib}
    SHARED
    SYCL_SOURCES ${ATen_XPU_SYCL_UNARY_BINARY_SRCS})
  target_link_libraries(torch_xpu_ops PUBLIC ${sycl_unary_binary_lib})
  list(APPEND TORCH_XPU_OPS_LIBRARIES ${sycl_unary_binary_lib})

  # Decouple with PyTorch cmake definition.
  install(TARGETS ${sycl_unary_binary_lib} DESTINATION "${TORCH_INSTALL_LIB_DIR}")

  set(sycl_lib torch_xpu_ops_sycl_kernels)
  sycl_add_library(
    ${sycl_lib}
    SHARED
    SYCL_SOURCES ${ATen_XPU_SYCL_NON_UNARY_BINARY_SRCS})
  target_link_libraries(torch_xpu_ops PUBLIC ${sycl_lib})
  list(APPEND TORCH_XPU_OPS_LIBRARIES ${sycl_lib})

  # Decouple with PyTorch cmake definition.
  install(TARGETS ${sycl_lib} DESTINATION "${TORCH_INSTALL_LIB_DIR}")
endif()
set(SYCL_LINK_LIBRARIES_KEYWORD)

list(APPEND TORCH_XPU_OPS_LIBRARIES torch_xpu_ops)

foreach(lib ${TORCH_XPU_OPS_LIBRARIES})
  # Align with PyTorch compile options PYTORCH_SRC_DIR/cmake/public/utils.cmake
  torch_compile_options(${lib})
  target_compile_options_if_supported(${lib} "-Wno-deprecated-copy")
  target_compile_options(${lib} PRIVATE ${TORCH_XPU_OPS_FLAGS})

  target_include_directories(${lib} PUBLIC ${TORCH_XPU_OPS_INCLUDE_DIRS})
  target_include_directories(${lib} PUBLIC ${ATen_XPU_INCLUDE_DIRS})
  target_include_directories(${lib} PUBLIC ${SYCL_INCLUDE_DIR})

  target_link_libraries(${lib} PUBLIC ${SYCL_LIBRARY})
endforeach()
