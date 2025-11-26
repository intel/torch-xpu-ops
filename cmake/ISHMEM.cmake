if(NOT __ISHMEM_INCLUDED)
  set(__ISHMEM_INCLUDED TRUE)

  # ISHMEM_ROOT, ISHMEM_LIBRARY_DIR, ISHMEM_INCLUDE_DIR are handled by FindISHMEM.cmake.
  find_package(ISHMEM REQUIRED)
  if(NOT ISHMEM_FOUND)
    set(PYTORCH_FOUND_ISHMEM FALSE)
    message(WARNING "${ISHMEM_NOT_FOUND_MESSAGE}")
    return()
  endif()

  set(PYTORCH_FOUND_ISHMEM TRUE)
  add_library(torch::ishmem INTERFACE IMPORTED)
  set_property(
    TARGET torch::ishmem PROPERTY INTERFACE_INCLUDE_DIRECTORIES
    ${ISHMEM_INCLUDE_DIR})
  set_property(
    TARGET torch::ishmem PROPERTY INTERFACE_LINK_LIBRARIES
    ${ISHMEM_LIBRARY})
  
  message(STATUS "Found Intel SHMEM: ${ISHMEM_ROOT}")
  message(STATUS "  ISHMEM include dir: ${ISHMEM_INCLUDE_DIR}")
  message(STATUS "  ISHMEM library: ${ISHMEM_LIBRARY}")
endif()
