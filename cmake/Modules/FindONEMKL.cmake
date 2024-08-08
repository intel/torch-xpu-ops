set(ONEMKL_FOUND FALSE)

set(ONEMKL_LIBRARIES)

# MKL configuration will not be imported into the build system.
# CMPLR_ROOT and MKLROOT are the same in the Pytorch development bundle.
if(DEFINED ENV{CMPLR_ROOT})
  set(CMPLR_ROOT $ENV{CMPLR_ROOT})
endif()

if(NOT CMPLR_ROOT)
  message(
    WARNING
      "Cannot find ENV{CMPLR_ROOT}, please setup SYCL compiler Tool kit enviroment before building!"
  )
  return()
endif()

get_filename_component(ONEMKL_ROOT "${CMPLR_ROOT}/../../mkl/latest" REALPATH)

find_file(
  ONEMKL_INCLUDE_DIR
  NAMES include
  HINTS ${ONEMKL_ROOT}
  NO_DEFAULT_PATH)

find_file(
  ONEMKL_LIB_DIR
  NAMES lib
  HINTS ${ONEMKL_ROOT}
  NO_DEFAULT_PATH)

if((NOT ONEMKL_INCLUDE_DIR) OR (NOT ONEMKL_LIB_DIR))
  message(WARNING "oneMKL sdk is incomplete!!")
  return()
endif()

set(CMAKE_INCLUDE_PATH ${CMAKE_INCLUDE_PATH} "${ONEMKL_INCLUDE_DIR}")
set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} "${ONEMKL_LIB_DIR}")

set(MKL_LIB_NAMES "mkl_intel_lp64" "mkl_gnu_thread" "mkl_core" "mkl_sycl_dft")

foreach(LIB_NAME IN LISTS MKL_LIB_NAMES)
  find_library(
    ${LIB_NAME}_library
    NAMES ${LIB_NAME}
    HINTS ${ONEMKL_LIB_DIR}
    NO_CMAKE_PATH NO_CMAKE_ENVIRONMENT_PATH)
  list(APPEND ONEMKL_LIBRARIES ${${LIB_NAME}_library})
endforeach()

set(ONEMKL_FOUND TRUE)
