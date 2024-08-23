# Define archive static library target - torch_xpu_ops
include(${TORCH_XPU_OPS_ROOT}/cmake/Codegen.cmake)

set(ATen_XPU_CPP_SRCS)
set(ATen_XPU_NATIVE_CPP_SRCS)
set(ATen_XPU_SYCL_SRCS)

set(ATen_XPU_INCLUDE_DIRS ${TORCH_XPU_OPS_ROOT}/src CACHE STRING "ATen XPU Include directory")

add_subdirectory(ATen)

set(TORCH_XPU_OPS_LIBRARIES)
set(SYCL_LINK_LIBRARIES_KEYWORD PRIVATE)

# Walk around cyclic dependence
# libtorch_xpu.so links to libtorch_xpu_ops.a
# libtorch_xpu_ops.a dlopens libtorch_xpu_ops_aten.so (Break cycle)
# libtorch_xpu_ops_aten.so links to libtorch_xpu_ops_sycl_unary_binary_kernels.so and libtorch_xpu_ops_sycl_kernels.so
# libtorch_xpu_ops_sycl_unary_binary_kernels.so and libtorch_xpu_ops_sycl_kernels.so links to libtorch_xpu.so
add_library(
  torch_xpu_ops
  STATIC
  ${ATen_XPU_CPP_SRCS}
  "bridge.cpp")

target_compile_options(torch_xpu_ops PRIVATE "-D PATH_TO_TORCH_XPU_OPS_ATEN_LIB=\"libtorch_xpu_ops_aten.so\"")

add_library(
  torch_xpu_ops_aten
  SHARED
  ${ATen_XPU_NATIVE_CPP_SRCS}
  ${ATen_XPU_GEN_SRCS})
install(TARGETS torch_xpu_ops_aten DESTINATION "${TORCH_INSTALL_LIB_DIR}")
target_link_libraries(torch_xpu_ops_aten PUBLIC torch_xpu)

if(BUILD_SEPARATE_OPS)
  foreach(sycl_src ${ATen_XPU_SYCL_SRCS})
    get_filename_component(name ${sycl_src} NAME_WLE REALPATH)
    set(sycl_lib torch-xpu-ops-sycl-${name})
    sycl_add_library(
      ${sycl_lib}
      SHARED
      SYCL_SOURCES ${sycl_src})
    target_link_libraries(torch_xpu_ops_aten PUBLIC ${sycl_lib})
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
    string(REGEX MATCH "Pow" IS_POW ${sycl_src})
    string(REGEX MATCH "Copy" IS_COPY ${sycl_src})
    if(IS_BINARY STREQUAL "" AND IS_UNARY STREQUAL "" AND IS_POW STREQUAL "" AND IS_COPY STREQUAL "")
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
  target_compile_definitions(${sycl_unary_binary_lib} PRIVATE TORCH_XPU_BUILD_MAIN_LIB)
  target_link_libraries(torch_xpu_ops_aten PUBLIC ${sycl_unary_binary_lib})
  target_link_libraries(${sycl_unary_binary_lib} PUBLIC torch_xpu)
  list(APPEND TORCH_XPU_OPS_LIBRARIES ${sycl_unary_binary_lib})

  # Decouple with PyTorch cmake definition.
  install(TARGETS ${sycl_unary_binary_lib} DESTINATION "${TORCH_INSTALL_LIB_DIR}")

  set(sycl_lib torch_xpu_ops_sycl_kernels)
  sycl_add_library(
    ${sycl_lib}
    SHARED
    SYCL_SOURCES ${ATen_XPU_SYCL_NON_UNARY_BINARY_SRCS})
  target_compile_definitions(${sycl_lib} PRIVATE TORCH_XPU_BUILD_MAIN_LIB)
  target_link_libraries(torch_xpu_ops_aten PUBLIC ${sycl_lib})
  target_link_libraries(${sycl_lib} PUBLIC torch_xpu)
  list(APPEND TORCH_XPU_OPS_LIBRARIES ${sycl_lib})

  # Decouple with PyTorch cmake definition.
  install(TARGETS ${sycl_lib} DESTINATION "${TORCH_INSTALL_LIB_DIR}")
endif()
set(SYCL_LINK_LIBRARIES_KEYWORD)

list(APPEND TORCH_XPU_OPS_LIBRARIES torch_xpu_ops)
list(APPEND TORCH_XPU_OPS_LIBRARIES torch_xpu_ops_aten)

foreach(lib ${TORCH_XPU_OPS_LIBRARIES})
  # Align with PyTorch compile options PYTORCH_SRC_DIR/cmake/public/utils.cmake
  torch_compile_options(${lib})
  target_compile_options_if_supported(${lib} "-Wno-deprecated-copy")
  target_compile_options(${lib} PRIVATE ${TORCH_XPU_OPS_FLAGS})

  target_include_directories(${lib} PUBLIC ${TORCH_XPU_OPS_INCLUDE_DIRS})
  target_include_directories(${lib} PUBLIC ${ATen_XPU_INCLUDE_DIRS})
  target_include_directories(${lib} PUBLIC ${SYCL_INCLUDE_DIR})

  target_link_libraries(${lib} PUBLIC ${SYCL_LIBRARY})
  target_link_libraries(${lib} PUBLIC c10_xpu)
endforeach()

include(${TORCH_XPU_OPS_ROOT}/cmake/ClangFormat.cmake)
if(CLANG_FORMAT)
  file(GLOB_RECURSE ALL_CSRCS ${TORCH_XPU_OPS_ROOT}/**.[ch] ${TORCH_XPU_OPS_ROOT}/**.[ch]pp)
  add_custom_target(CL_FORMAT_CSRCS COMMAND ${CLANG_FORMAT_EXEC} -i -style=file ${ALL_CSRCS})
  add_dependencies(torch_xpu_ops CL_FORMAT_CSRCS)
endif()
