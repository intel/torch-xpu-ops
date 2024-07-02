set(ONEMKL_LIBRARIES)

if(DEFINED ENV{MKLROOT})
  SET(INTEL_MKL_DIR $ENV{MKLROOT})
else()
  SET(INTEL_MKL_DIR "/opt/intel/oneapi/mkl/latest")
endif()

set(CMAKE_INCLUDE_PATH ${CMAKE_INCLUDE_PATH}
    "${INTEL_MKL_DIR}/include")
set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH}
    "${INTEL_MKL_DIR}/lib")

set(MKL_LIB_NAMES "mkl_intel_lp64" "mkl_gnu_thread" "mkl_core" "mkl_sycl_dft")

foreach(LIB_NAME IN LISTS MKL_LIB_NAMES)
  find_library(
    ${LIB_NAME}_library
    NAMES ${LIB_NAME}
    HINTS ${INTEL_MKL_DIR}
  )
  list(APPEND ONEMKL_LIBRARIES ${${LIB_NAME}_library})
endforeach()

