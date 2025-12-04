# Build on Linux

set(TORCH_XPU_OPS_LIBRARIES)
set(SYCL_LINK_LIBRARIES_KEYWORD PRIVATE)

macro(setup_common_libraries)
  add_library(
    torch_xpu_ops
    STATIC
    ${ATen_XPU_CPP_SRCS}
    ${ATen_XPU_MKL_SRCS}
    ${ATen_XPU_NATIVE_CPP_SRCS}
    ${ATen_XPU_GEN_SRCS}
    ${ATen_XPU_XCCL_SRCS})

  if(USE_C10D_XCCL)
    target_compile_definitions(torch_xpu_ops PRIVATE USE_C10D_XCCL)
    target_link_libraries(torch_xpu_ops PUBLIC torch::xccl)
    target_link_libraries(torch_xpu_ops PUBLIC fmt::fmt-header-only)
    if(USE_ISHMEM AND PYTORCH_FOUND_ISHMEM)
      target_link_libraries(torch_xpu_ops PUBLIC torch::ishmem)
    endif()
  endif()

  if(USE_SYCLTLA)
    target_compile_definitions(torch_xpu_ops PRIVATE USE_SYCLTLA)
  endif()
  list(APPEND TORCH_XPU_OPS_LIBRARIES torch_xpu_ops)
endmacro()

if(BUILD_SEPARATE_OPS)
  setup_common_libraries()
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
  sycl_add_library(
    torch_xpu_ops
    STATIC
    CXX_SOURCES  ${ATen_XPU_CPP_SRCS} ${ATen_XPU_MKL_SRCS} ${ATen_XPU_NATIVE_CPP_SRCS} ${ATen_XPU_GEN_SRCS} ${ATen_XPU_XCCL_SRCS}
    SYCL_SOURCES ${ATen_XPU_SYCL_SRCS})
  if(USE_C10D_XCCL)
    target_compile_definitions(torch_xpu_ops PRIVATE USE_C10D_XCCL)
    target_link_libraries(torch_xpu_ops  PUBLIC torch::xccl)
    target_link_libraries(torch_xpu_ops PUBLIC fmt::fmt-header-only)
    if(USE_ISHMEM AND PYTORCH_FOUND_ISHMEM)
      target_link_libraries(torch_xpu_ops PUBLIC torch::ishmem)
    endif()
  endif()

  if(USE_SYCLTLA)
    target_compile_definitions(torch_xpu_ops PRIVATE USE_SYCLTLA)
  endif()

  install(TARGETS torch_xpu_ops DESTINATION "${TORCH_INSTALL_LIB_DIR}")
  list(APPEND TORCH_XPU_OPS_LIBRARIES torch_xpu_ops)
endif()

if(USE_SYCLTLA)
  set(REPLACE_FLAGS_FOR_SYCLTLA TRUE)
  set_build_flags()
  replace_cmake_build_flags()

  foreach(sycl_src ${ATen_XPU_SYCLTLA_SRCS})
    get_filename_component(name ${sycl_src} NAME_WLE REALPATH)
    set(sycl_lib torch-xpu-ops-sycltla-${name})
    sycl_add_library(
      ${sycl_lib}
      SHARED
      SYCL_SOURCES ${sycl_src})
    target_link_libraries(torch_xpu_ops PUBLIC ${sycl_lib})
    list(APPEND TORCH_XPU_OPS_LIBRARIES ${sycl_lib})

    # Decouple with PyTorch cmake definition.
    install(TARGETS ${sycl_lib} DESTINATION "${TORCH_INSTALL_LIB_DIR}")

    # Set Compile options for sycltla kernels
    target_compile_definitions(${sycl_lib} PRIVATE ${SYCLTLA_COMPILE_DEFINITIONS})
    target_include_directories(${sycl_lib} PRIVATE ${SYCLTLA_INCLUDE_DIRS})
  endforeach()

  set(REPLACE_FLAGS_FOR_SYCLTLA FALSE)
  set_build_flags()
  restore_cmake_build_flags()
endif()

set(SYCL_LINK_LIBRARIES_KEYWORD)

foreach(lib ${TORCH_XPU_OPS_LIBRARIES})
  # Align with PyTorch compile options PYTORCH_SRC_DIR/cmake/public/utils.cmake
  torch_compile_options(${lib})
  target_compile_options_if_supported(${lib} "-Wno-deprecated-copy")
  target_compile_options(${lib} PRIVATE ${TORCH_XPU_OPS_FLAGS})

  target_include_directories(${lib} PUBLIC ${TORCH_XPU_OPS_INCLUDE_DIRS})
  target_include_directories(${lib} PUBLIC ${ATen_XPU_INCLUDE_DIRS})
  target_include_directories(${lib} PUBLIC ${SYCL_INCLUDE_DIR})

  target_link_libraries(${lib} PUBLIC ${SYCL_LIBRARY})
  target_link_libraries(${lib} PRIVATE ATEN_XPU_OPS_FILES_GEN_LIB)
endforeach()

if(USE_ONEMKL_XPU)
  target_compile_options(torch_xpu_ops PRIVATE "-DUSE_ONEMKL_XPU")
  target_include_directories(torch_xpu_ops PUBLIC ${TORCH_XPU_OPS_ONEMKL_INCLUDE_DIR})
  target_link_libraries(torch_xpu_ops PUBLIC ${TORCH_XPU_OPS_ONEMKL_LIBRARIES})
endif()
