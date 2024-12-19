# Build on Windows

set(TORCH_XPU_OPS_LIBRARIES)
set(SYCL_LINK_LIBRARIES_KEYWORD PRIVATE)

add_library(
  torch_xpu_ops
  STATIC
  ${ATen_XPU_CPP_SRCS}
  ${ATen_XPU_MKL_SRCS})
set(PATH_TO_TORCH_XPU_OPS_ATEN_LIB \"torch_xpu_ops_aten.dll\")
target_compile_options(torch_xpu_ops PRIVATE -DPATH_TO_TORCH_XPU_OPS_ATEN_LIB=${PATH_TO_TORCH_XPU_OPS_ATEN_LIB})

add_library(
  torch_xpu_ops_aten
  SHARED
  ${ATen_XPU_NATIVE_CPP_SRCS}
  ${ATen_XPU_GEN_SRCS})
install(TARGETS torch_xpu_ops_aten DESTINATION "${TORCH_INSTALL_LIB_DIR}")
target_compile_definitions(torch_xpu_ops_aten PRIVATE TORCH_XPU_BUILD_MAIN_LIB)
target_link_libraries(torch_xpu_ops_aten PUBLIC torch_xpu)
target_link_libraries(torch_xpu_ops_aten PUBLIC torch_cpu)
target_link_libraries(torch_xpu_ops_aten PUBLIC c10)

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
  set(ATen_XPU_SYCL_BINARY_SRCS)
  set(ATen_XPU_SYCL_UNARY_SRCS)
  set(ATen_XPU_SYCL_REDUCE_SRCS)
  set(ATen_XPU_SYCL_ACTIVATION_SRCS)
  set(ATen_XPU_SYCL_FOREACH_SRCS)
  set(ATen_XPU_SYCL_TENSOR_SRCS)
  set(ATen_XPU_SYCL_NORM_LOSS_SRCS)
  set(ATen_XPU_SYCL_POLY_SRCS)
  set(ATen_XPU_SYCL_DISTRIBUTION_SRCS)
  set(ATen_XPU_SYCL_OTHERS_SRCS)
  foreach(sycl_src ${ATen_XPU_SYCL_SRCS})
    string(REGEX MATCH "Binary" IS_BINARY ${sycl_src})
    string(REGEX MATCH "Unary" IS_UNARY ${sycl_src})
    # Resolve cyclic dependences between
    # torch_xpu_ops_sycl_unary_binary_kernels.dll and
    # torch_xpu_ops_sycl_kernels.dll. Move definition and invoke of kernels
    # into a same kernel library. Here we move elementwise kernel pow and copy
    # into torch_xpu_ops_sycl_unary_binary_kernels.dll.
    string(REGEX MATCH "Pow" IS_POW ${sycl_src})
    string(REGEX MATCH "Copy" IS_COPY ${sycl_src})
    string(REGEX MATCH "Activation" IS_ACTIVATION ${sycl_src})
    string(REGEX MATCH "Foreach" IS_FOREACH ${sycl_src})
    string(REGEX MATCH "Reduce" IS_REDUCE ${sycl_src})
	string(REGEX MATCH "Tensor" IS_TENSOR ${sycl_src})
	string(REGEX MATCH "Norm" IS_NORM ${sycl_src})
	string(REGEX MATCH "Loss" IS_LOSS ${sycl_src})
	string(REGEX MATCH "Polynomial" IS_POLY ${sycl_src})
	#Move resize kernel to Norm and Loss lib, to resolve symbol.
	string(REGEX MATCH "Resize" IS_RESIZE ${sycl_src})
	string(REGEX MATCH "Distribution" IS_DISTRIBUTION ${sycl_src})

    if(NOT IS_FOREACH STREQUAL "")
      list(APPEND ATen_XPU_SYCL_FOREACH_SRCS ${sycl_src})
    elseif(NOT IS_BINARY STREQUAL "")
      list(APPEND ATen_XPU_SYCL_BINARY_SRCS ${sycl_src})
    elseif(NOT IS_UNARY STREQUAL "" OR NOT IS_COPY STREQUAL "" OR NOT IS_POW STREQUAL "")
      list(APPEND ATen_XPU_SYCL_UNARY_SRCS ${sycl_src})
    elseif(NOT IS_REDUCE STREQUAL "")
      list(APPEND ATen_XPU_SYCL_REDUCE_SRCS ${sycl_src})
    elseif(NOT IS_ACTIVATION STREQUAL "")
      list(APPEND ATen_XPU_SYCL_ACTIVATION_SRCS ${sycl_src})
	elseif(NOT IS_TENSOR STREQUAL "")
	  list(APPEND ATen_XPU_SYCL_TENSOR_SRCS ${sycl_src})
	elseif(NOT IS_DISTRIBUTION STREQUAL "")
	  list(APPEND ATen_XPU_SYCL_DISTRIBUTION_SRCS ${sycl_src})
	elseif(NOT IS_NORM STREQUAL "" OR NOT IS_LOSS STREQUAL "" OR NOT IS_RESIZE STREQUAL "")
	  list(APPEND ATen_XPU_SYCL_NORM_LOSS_SRCS ${sycl_src})
	elseif(NOT IS_POLY STREQUAL "")
	  list(APPEND ATen_XPU_SYCL_POLY_SRCS ${sycl_src})
    else()
      list(APPEND ATen_XPU_SYCL_OTHERS_SRCS ${sycl_src})
    endif()
  endforeach()
  # Binary kernel lib
  set(sycl_binary_lib torch_xpu_ops_sycl_binary_kernels)
  sycl_add_library(
    ${sycl_binary_lib}
    SHARED
    SYCL_SOURCES ${ATen_XPU_SYCL_BINARY_SRCS})
  target_compile_definitions(${sycl_binary_lib} PRIVATE TORCH_XPU_BUILD_MAIN_LIB)
  target_link_libraries(torch_xpu_ops_aten PUBLIC ${sycl_binary_lib})
  target_link_libraries(${sycl_binary_lib} PUBLIC torch_xpu)
  list(APPEND TORCH_XPU_OPS_LIBRARIES ${sycl_binary_lib})

  # Decouple with PyTorch cmake definition.
  install(TARGETS ${sycl_binary_lib} DESTINATION "${TORCH_INSTALL_LIB_DIR}")

  # Unary kernel lib
  set(sycl_unary_lib torch_xpu_ops_sycl_unary_kernels)
  sycl_add_library(
    ${sycl_unary_lib}
    SHARED
    SYCL_SOURCES ${ATen_XPU_SYCL_UNARY_SRCS})
  target_compile_definitions(${sycl_unary_lib} PRIVATE TORCH_XPU_BUILD_MAIN_LIB)
  target_link_libraries(torch_xpu_ops_aten PUBLIC ${sycl_unary_lib})
  target_link_libraries(${sycl_unary_lib} PUBLIC torch_xpu)
  list(APPEND TORCH_XPU_OPS_LIBRARIES ${sycl_unary_lib})

  # Decouple with PyTorch cmake definition.
  install(TARGETS ${sycl_unary_lib} DESTINATION "${TORCH_INSTALL_LIB_DIR}")

  # Reduce kernel lib
  set(sycl_reduce_lib torch_xpu_ops_sycl_reduce_kernels)
  sycl_add_library(
    ${sycl_reduce_lib}
    SHARED
    SYCL_SOURCES ${ATen_XPU_SYCL_REDUCE_SRCS})
  target_compile_definitions(${sycl_reduce_lib} PRIVATE TORCH_XPU_BUILD_MAIN_LIB)
  target_link_libraries(torch_xpu_ops_aten PUBLIC ${sycl_reduce_lib})
  target_link_libraries(${sycl_reduce_lib} PUBLIC torch_xpu)
  list(APPEND TORCH_XPU_OPS_LIBRARIES ${sycl_reduce_lib})

  # Decouple with PyTorch cmake definition.
  install(TARGETS ${sycl_reduce_lib} DESTINATION "${TORCH_INSTALL_LIB_DIR}")

  # Activation kernel lib
  set(sycl_activation_lib torch_xpu_ops_sycl_activation_kernels)
  sycl_add_library(
    ${sycl_activation_lib}
    SHARED
    SYCL_SOURCES ${ATen_XPU_SYCL_ACTIVATION_SRCS})
  target_compile_definitions(${sycl_activation_lib} PRIVATE TORCH_XPU_BUILD_MAIN_LIB)
  target_link_libraries(torch_xpu_ops_aten PUBLIC ${sycl_activation_lib})
  target_link_libraries(${sycl_activation_lib} PUBLIC torch_xpu)
  list(APPEND TORCH_XPU_OPS_LIBRARIES ${sycl_activation_lib})

  # Decouple with PyTorch cmake definition.
  install(TARGETS ${sycl_activation_lib} DESTINATION "${TORCH_INSTALL_LIB_DIR}")

  # Foreach kernel lib
  set(sycl_foreach_lib torch_xpu_ops_sycl_foreach_kernels)
  sycl_add_library(
    ${sycl_foreach_lib}
    SHARED
    SYCL_SOURCES ${ATen_XPU_SYCL_FOREACH_SRCS})
  target_compile_definitions(${sycl_foreach_lib} PRIVATE TORCH_XPU_BUILD_MAIN_LIB)
  target_link_libraries(torch_xpu_ops_aten PUBLIC ${sycl_foreach_lib})
  target_link_libraries(${sycl_foreach_lib} PUBLIC torch_xpu)
  list(APPEND TORCH_XPU_OPS_LIBRARIES ${sycl_foreach_lib})

  # Decouple with PyTorch cmake definition.
  install(TARGETS ${sycl_foreach_lib} DESTINATION "${TORCH_INSTALL_LIB_DIR}")
  
  # Tensor kernel lib
  set(sycl_tensor_lib torch_xpu_ops_sycl_tensor_kernels)
  sycl_add_library(
    ${sycl_tensor_lib}
    SHARED
    SYCL_SOURCES ${ATen_XPU_SYCL_TENSOR_SRCS})
  target_compile_definitions(${sycl_tensor_lib} PRIVATE TORCH_XPU_BUILD_MAIN_LIB)
  target_link_libraries(torch_xpu_ops_aten PUBLIC ${sycl_tensor_lib})
  target_link_libraries(${sycl_tensor_lib} PUBLIC torch_xpu)
  list(APPEND TORCH_XPU_OPS_LIBRARIES ${sycl_tensor_lib})

  # Decouple with PyTorch cmake definition.
  install(TARGETS ${sycl_tensor_lib} DESTINATION "${TORCH_INSTALL_LIB_DIR}")
  
  # Norm and Loss kernel lib
  set(sycl_norm_loss_lib torch_xpu_ops_sycl_norm_loss_kernels)
  sycl_add_library(
    ${sycl_norm_loss_lib}
    SHARED
    SYCL_SOURCES ${ATen_XPU_SYCL_NORM_LOSS_SRCS})
  target_compile_definitions(${sycl_norm_loss_lib} PRIVATE TORCH_XPU_BUILD_MAIN_LIB)
  target_link_libraries(torch_xpu_ops_aten PUBLIC ${sycl_norm_loss_lib})
  target_link_libraries(${sycl_norm_loss_lib} PUBLIC torch_xpu)
  list(APPEND TORCH_XPU_OPS_LIBRARIES ${sycl_norm_loss_lib})

  # Decouple with PyTorch cmake definition.
  install(TARGETS ${sycl_norm_loss_lib} DESTINATION "${TORCH_INSTALL_LIB_DIR}")
  
  # Polynomial kernel lib
  set(sycl_poly_lib torch_xpu_ops_sycl_poly_kernels)
  sycl_add_library(
    ${sycl_poly_lib}
    SHARED
    SYCL_SOURCES ${ATen_XPU_SYCL_POLY_SRCS})
  target_compile_definitions(${sycl_poly_lib} PRIVATE TORCH_XPU_BUILD_MAIN_LIB)
  target_link_libraries(torch_xpu_ops_aten PUBLIC ${sycl_poly_lib})
  target_link_libraries(${sycl_poly_lib} PUBLIC torch_xpu)
  list(APPEND TORCH_XPU_OPS_LIBRARIES ${sycl_poly_lib})

  # Decouple with PyTorch cmake definition.
  install(TARGETS ${sycl_poly_lib} DESTINATION "${TORCH_INSTALL_LIB_DIR}")
  
  # Distribution kernel lib
  set(sycl_dist_lib torch_xpu_ops_sycl_dist_kernels)
  sycl_add_library(
    ${sycl_dist_lib}
    SHARED
    SYCL_SOURCES ${ATen_XPU_SYCL_DISTRIBUTION_SRCS})
  target_compile_definitions(${sycl_dist_lib} PRIVATE TORCH_XPU_BUILD_MAIN_LIB)
  target_link_libraries(torch_xpu_ops_aten PUBLIC ${sycl_dist_lib})
  target_link_libraries(${sycl_dist_lib} PUBLIC torch_xpu)
  list(APPEND TORCH_XPU_OPS_LIBRARIES ${sycl_dist_lib})

  # Decouple with PyTorch cmake definition.
  install(TARGETS ${sycl_dist_lib} DESTINATION "${TORCH_INSTALL_LIB_DIR}")
  
  # Other kernel lib
  set(sycl_lib torch_xpu_ops_sycl_kernels)
  sycl_add_library(
    ${sycl_lib}
    SHARED
    SYCL_SOURCES ${ATen_XPU_SYCL_OTHERS_SRCS})
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
  target_link_libraries(${lib} PUBLIC torch_cpu)
endforeach()
