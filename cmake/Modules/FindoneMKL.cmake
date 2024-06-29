set(ONEMKL_FOUND FALSE)

set(ONEMKL_LIBRARIES)

set(ONEMKL_DEFAULT_DIR "/opt/intel/oneapi/mkl/latest")
if(DEFINED ENV{MKLROOT})
  set(ONEMKL_ROOT $ENV{MKLROOT})
elseif(EXISTS "${ONEMKL_DEFAULT_DIR}")
  set(ONEMKL_ROOT "${ONEMKL_DEFAULT_DIR}")
endif()

if(NOT ONEMKL_ROOT)
  message(WARNING "Cannot find oneMKL in ENV{MKLROOT} or default path, please setup oneMKL before building!!")
  return()
endif()

find_file(
  ONEMKL_INCLUDE_DIR
  NAMES include
  HINTS ${ONEMKL_ROOT}
  NO_DEFAULT_PATH
)

find_file(
  ONEMKL_LIB_DIR
  NAMES lib
  HINTS ${ONEMKL_ROOT}
  NO_DEFAULT_PATH
)

if((NOT ONEMKL_INCLUDE_DIR) OR (NOT ONEMKL_LIB_DIR))
  message(WARNING "oneMKL sdk is incomplete!!")
  return()
endif()

set(CMAKE_INCLUDE_PATH ${CMAKE_INCLUDE_PATH}
    "${ONEMKL_INCLUDE_DIR}")
set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH}
    "${ONEMKL_LIB_DIR}")

set(MKL_LIB_NAMES "mkl_intel_lp64" "mkl_gnu_thread" "mkl_core" "mkl_sycl_dft")

foreach(LIB_NAME IN LISTS MKL_LIB_NAMES)
  find_library(
    ${LIB_NAME}_library
    NAMES ${LIB_NAME}
    HINTS ${ONEMKL_LIB_DIR}
    NO_CMAKE_PATH
    NO_CMAKE_ENVIRONMENT_PATH
  )
  list(APPEND ONEMKL_LIBRARIES ${${LIB_NAME}_library})
endforeach()

set(ONEMKL_FOUND TRUE)
