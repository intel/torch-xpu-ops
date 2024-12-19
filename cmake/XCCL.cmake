if(NOT __XCCL_INCLUDED)
  set(__XCCL_INCLUDED TRUE)

  # XCCL_ROOT, XCCL_LIBRARY_DIR, XCCL_INCLUDE_DIR are handled by FindXCCL.cmake.
  find_package(XCCL REQUIRED)
  if(NOT XCCL_FOUND)
    message("${XCCL_NOT_FOUND_MESSAGE}")
    return()
  endif()
  if(XCCL_FOUND)
    add_library(torch::xccl INTERFACE IMPORTED)
    set_property(
      TARGET torch::xccl PROPERTY INTERFACE_INCLUDE_DIRECTORIES
      ${XCCL_INCLUDE_DIR})
    set_property(
      TARGET torch::xccl PROPERTY INTERFACE_LINK_LIBRARIES
      ${XCCL_LIBRARY})
    set(USE_C10D_XCCL ON)
    set(USE_C10D_XCCL ${USE_C10D_XCCL} PARENT_SCOPE)
  endif()
endif()
