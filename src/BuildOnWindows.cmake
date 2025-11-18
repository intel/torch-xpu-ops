# Build on Windows

set(TORCH_XPU_OPS_LIBRARIES)
set(SYCL_LINK_LIBRARIES_KEYWORD PRIVATE)

macro(setup_common_libraries)
  add_library(
    torch_xpu_ops
    STATIC
    ${ATen_XPU_CPP_SRCS})
  set(PATH_TO_TORCH_XPU_OPS_ATEN_LIB \"torch_xpu_ops_aten.dll\")
  target_compile_options(torch_xpu_ops PRIVATE -DPATH_TO_TORCH_XPU_OPS_ATEN_LIB=${PATH_TO_TORCH_XPU_OPS_ATEN_LIB})

  add_library(
    torch_xpu_ops_aten
    SHARED
    ${ATen_XPU_MKL_SRCS}
    ${ATen_XPU_NATIVE_CPP_SRCS}
    ${ATen_XPU_GEN_SRCS})
  install(TARGETS torch_xpu_ops_aten DESTINATION "${TORCH_INSTALL_LIB_DIR}")
  target_compile_definitions(torch_xpu_ops_aten PRIVATE TORCH_XPU_BUILD_MAIN_LIB)
  target_link_libraries(torch_xpu_ops_aten PUBLIC torch_xpu)
  target_link_libraries(torch_xpu_ops_aten PUBLIC torch_cpu)
  target_link_libraries(torch_xpu_ops_aten PUBLIC c10)
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
    target_link_libraries(torch_xpu_ops_aten PUBLIC ${sycl_lib})
    list(APPEND TORCH_XPU_OPS_LIBRARIES ${sycl_lib})

    # Decouple with PyTorch cmake definition.
    install(TARGETS ${sycl_lib} DESTINATION "${TORCH_INSTALL_LIB_DIR}")
  endforeach()
  list(APPEND TORCH_XPU_OPS_LIBRARIES torch_xpu_ops)
  list(APPEND TORCH_XPU_OPS_LIBRARIES torch_xpu_ops_aten)
else()
  # On Windows, it is not possible to combine all obj files into one library
  # because the obj files of kernels compiled on Windows are much larger than
  # those on Linux. If they are combined into one, the library size will exceed
  # 4GB, which conflicts with the size limit of a single library on Windows.
  # We will combine the libraries on Windows into one after the compiler is fixed.
  add_library(
    torch_xpu_ops
    STATIC
    ${ATen_XPU_CPP_SRCS}
    ${ATen_XPU_MKL_SRCS}
    ${ATen_XPU_NATIVE_CPP_SRCS}
    ${ATen_XPU_GEN_SRCS})
  target_compile_definitions(torch_xpu_ops PRIVATE TORCH_XPU_BUILD_MAIN_LIB)
 # Split SYCL kernels into 2 libraries as categories 1) Common (Unary+Binary+Reduce+Pow+Copy+Activation+Foreach) 2) Others.
  set(ATen_XPU_SYCL_COMMON_SRCS)
  set(ATen_XPU_SYCL_OTHERS_SRCS)
  foreach(sycl_src ${ATen_XPU_SYCL_SRCS})
    string(REGEX MATCH "Binary" IS_BINARY ${sycl_src})
    string(REGEX MATCH "Unary" IS_UNARY ${sycl_src})
    string(REGEX MATCH "Pow" IS_POW ${sycl_src})
    string(REGEX MATCH "Copy" IS_COPY ${sycl_src})
    string(REGEX MATCH "Reduce" IS_REDUCE ${sycl_src})
    string(REGEX MATCH "Activation" IS_ACTIVATION ${sycl_src})
    string(REGEX MATCH "Foreach" IS_FOREACH ${sycl_src})
    string(REGEX MATCH "Polynomial" IS_POLY ${sycl_src})
    string(REGEX MATCH "Norm" IS_NORM ${sycl_src})
    string(REGEX MATCH "Loss" IS_LOSS ${sycl_src})
    string(REGEX MATCH "Resize" IS_RESIZE ${sycl_src})
    string(REGEX MATCH "Distribution" IS_DISTRIBUTION ${sycl_src})

    if(NOT IS_FOREACH STREQUAL "")
      list(APPEND ATen_XPU_SYCL_COMMON_SRCS ${sycl_src})
    elseif(NOT IS_REDUCE STREQUAL "")
      list(APPEND ATen_XPU_SYCL_COMMON_SRCS ${sycl_src})
    elseif(NOT IS_UNARY STREQUAL "" OR NOT IS_BINARY STREQUAL "")
      list(APPEND ATen_XPU_SYCL_COMMON_SRCS ${sycl_src})
    elseif(NOT IS_COPY STREQUAL "" OR NOT IS_POW STREQUAL "")
      list(APPEND ATen_XPU_SYCL_COMMON_SRCS ${sycl_src})
    elseif(NOT IS_ACTIVATION STREQUAL "")
      list(APPEND ATen_XPU_SYCL_COMMON_SRCS ${sycl_src})
    elseif(NOT IS_NORM STREQUAL "")
      list(APPEND ATen_XPU_SYCL_COMMON_SRCS ${sycl_src})
    elseif(NOT IS_LOSS STREQUAL "")
      list(APPEND ATen_XPU_SYCL_COMMON_SRCS ${sycl_src})
    elseif(NOT IS_RESIZE STREQUAL "")
      list(APPEND ATen_XPU_SYCL_COMMON_SRCS ${sycl_src})
    elseif(NOT IS_DISTRIBUTION STREQUAL "")
      list(APPEND ATen_XPU_SYCL_COMMON_SRCS ${sycl_src})
    elseif(NOT IS_POLY STREQUAL "")
      list(APPEND ATen_XPU_SYCL_COMMON_SRCS ${sycl_src})
    else()
      list(APPEND ATen_XPU_SYCL_OTHERS_SRCS ${sycl_src})
    endif()
  endforeach()
  # Common kernel lib
  set(sycl_common_lib torch_xpu_ops_sycl_common_kernels)
  sycl_add_library(
    ${sycl_common_lib}
    STATIC
    SYCL_SOURCES ${ATen_XPU_SYCL_COMMON_SRCS})
  target_compile_definitions(${sycl_common_lib} PRIVATE TORCH_XPU_BUILD_MAIN_LIB)
  list(APPEND TORCH_XPU_OPS_LIBRARIES ${sycl_common_lib})

  # Other kernel lib
  set(sycl_lib torch_xpu_ops_sycl_kernels)
  sycl_add_library(
    ${sycl_lib}
    STATIC
    SYCL_SOURCES ${ATen_XPU_SYCL_OTHERS_SRCS})
  target_compile_definitions(${sycl_lib} PRIVATE TORCH_XPU_BUILD_MAIN_LIB)
  list(APPEND TORCH_XPU_OPS_LIBRARIES ${sycl_lib})

  target_link_libraries(torch_xpu_ops
      PUBLIC
      ${sycl_common_lib}
      ${sycl_lib}
  )
  target_link_options(torch_xpu_ops PUBLIC
      "-WHOLEARCHIVE:$<TARGET_FILE:${sycl_common_lib}>"
      "-WHOLEARCHIVE:$<TARGET_FILE:${sycl_lib}>"
  )
  list(APPEND TORCH_XPU_OPS_LIBRARIES torch_xpu_ops)
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
  target_link_libraries(${lib} PUBLIC c10_xpu)
  target_link_libraries(${lib} PUBLIC torch_cpu)
  target_link_libraries(${lib} PRIVATE ATEN_XPU_OPS_FILES_GEN_LIB)
endforeach()

if(USE_ONEMKL_XPU)
  target_compile_options(torch_xpu_ops PRIVATE "-DUSE_ONEMKL_XPU")
  target_include_directories(torch_xpu_ops PUBLIC ${TORCH_XPU_OPS_ONEMKL_INCLUDE_DIR})
  target_link_libraries(torch_xpu_ops PUBLIC ${TORCH_XPU_OPS_ONEMKL_LIBRARIES})
endif()
