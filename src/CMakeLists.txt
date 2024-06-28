# Define archive static library target - torch_xpu_ops
include(${TORCH_XPU_OPS_ROOT}/cmake/Codegen.cmake)

set(ATen_XPU_CPP_SRCS)
set(ATen_XPU_SYCL_SRCS)

set(ATen_XPU_INCLUDE_DIRS ${TORCH_XPU_OPS_ROOT}/src)

add_subdirectory(ATen)

set(TORCH_XPU_OPS_LIBRARIES)
set(SYCL_LINK_LIBRARIES_KEYWORD PRIVATE)
if(BUILD_SEPARATE_OPS)
  add_library(
    torch_xpu_ops
    STATIC
    ${ATen_XPU_CPP_SRCS}
    ${ATen_XPU_GEN_SRCS})
  if(USE_PER_OPERATOR_HEADERS)
    target_compile_definitions(torch_xpu_ops PUBLIC -DAT_PER_OPERATOR_HEADERS)
  endif()

  foreach(sycl_src ${ATen_XPU_SYCL_SRCS})
    get_filename_component(name ${sycl_src} NAME_WLE REALPATH)
    set(sycl_lib torch-xpu-ops-sycl-${name})
    sycl_add_library(
      ${sycl_lib}
      SHARED
      SYCL_SOURCES ${sycl_src})
    target_link_libraries(torch_xpu_ops PUBLIC ${sycl_lib})
    if(USE_PER_OPERATOR_HEADERS)
      target_compile_definitions(torch_xpu_ops PUBLIC -DAT_PER_OPERATOR_HEADERS)
    endif()
    list(APPEND TORCH_XPU_OPS_LIBRARIES ${sycl_lib})

    # Decouple with PyTorch cmake definition.
    install(TARGETS ${sycl_lib} DESTINATION "${TORCH_INSTALL_LIB_DIR}")
  endforeach()
else()
  sycl_add_library(
    torch_xpu_ops
    STATIC
    SYCL_SOURCES ${ATen_XPU_SYCL_SRCS}
    CXX_SOURCES ${ATen_XPU_CPP_SRCS} ${ATen_XPU_GEN_SRCS})
  if(USE_PER_OPERATOR_HEADERS)
    target_compile_definitions(torch_xpu_ops PUBLIC -DAT_PER_OPERATOR_HEADERS)
  endif()
endif()
set(SYCL_LINK_LIBRARIES_KEYWORD)

list(APPEND TORCH_XPU_OPS_LIBRARIES torch_xpu_ops)

foreach(lib ${TORCH_XPU_OPS_LIBRARIES})
  # Align with PyTorch compile options PYTORCH_SRC_DIR/cmake/public/utils.cmake
  torch_compile_options(${lib})
  target_compile_options_if_supported(${lib} "-Wno-deprecated-copy")
  target_compile_options(${lib} PRIVATE ${TORCH_XPU_OPS_FLAGS})

  if(USE_PER_OPERATOR_HEADERS)
    target_compile_definitions(${lib}  PUBLIC -DAT_PER_OPERATOR_HEADERS)
  endif()

  target_include_directories(${lib} PUBLIC ${TORCH_XPU_OPS_INCLUDE_DIRS})
  target_include_directories(${lib} PUBLIC ${ATen_XPU_INCLUDE_DIRS})
  target_include_directories(${lib} PUBLIC ${SYCL_INCLUDE_DIR})

  target_link_libraries(${lib} PUBLIC ${SYCL_LIBRARIES})
endforeach()

include(${TORCH_XPU_OPS_ROOT}/cmake/ClangFormat.cmake)
if(CLANG_FORMAT)
  file(GLOB_RECURSE ALL_CSRCS ${TORCH_XPU_OPS_ROOT}/**.[ch] ${TORCH_XPU_OPS_ROOT}/**.[ch]pp)
  add_custom_target(CL_FORMAT_CSRCS COMMAND ${CLANG_FORMAT_EXEC} -i -style=file ${ALL_CSRCS})
  add_dependencies(torch_xpu_ops CL_FORMAT_CSRCS)
endif()
