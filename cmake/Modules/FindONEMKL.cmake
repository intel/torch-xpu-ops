# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

set(ONEMKL_FOUND FALSE)

set(ONEMKL_LIBRARIES)

# ENV{MKLROOT} is authoritative when set; otherwise fall back to the oneAPI
# bundle layout next to the SYCL compiler.
if(DEFINED ENV{MKLROOT})
  set(ONEMKL_ROOT $ENV{MKLROOT})
elseif(SYCL_FOUND)
  # Not cmake_path(NORMAL_PATH): it only recognizes '/' as a separator, and on
  # Windows SYCL_ROOT arrives backslashed from setvars.bat, which would make
  # the whole thing one component for ../.. to swallow.
  get_filename_component(ONEMKL_ROOT "${SYCL_ROOT}/../../mkl/latest" REALPATH)
endif()

if(NOT DEFINED ONEMKL_ROOT)
  message(
    WARNING
      "Cannot find either ENV{MKLROOT} or SYCL_ROOT, please setup oneAPI environment before building!!"
  )
  return()
endif()

if(NOT EXISTS "${ONEMKL_ROOT}")
  message(
    WARNING
      "${ONEMKL_ROOT} not found, please setup oneAPI environment before building!!"
  )
  return()
endif()

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

if((ONEMKL_INCLUDE_DIR STREQUAL "ONEMKL_INCLUDE_DIR-NOTFOUND")
   OR(ONEMKL_LIB_DIR STREQUAL "ONEMKL_LIB_DIR-NOTFOUND"))
  message(WARNING "oneMKL SDK is incomplete!!")
  return()
endif()

set(MKL_LIB_NAMES "mkl_sycl_blas" "mkl_sycl_dft" "mkl_sycl_lapack"
                  "mkl_intel_lp64" "mkl_core")

if(WIN32)
  list(APPEND MKL_LIB_NAMES "mkl_intel_thread")
  list(TRANSFORM MKL_LIB_NAMES APPEND "_dll.lib")
else()
  list(APPEND MKL_LIB_NAMES "mkl_gnu_thread")
  list(TRANSFORM MKL_LIB_NAMES PREPEND "lib")
  list(TRANSFORM MKL_LIB_NAMES APPEND ".so")
endif()

foreach(LIB_NAME IN LISTS MKL_LIB_NAMES)
  find_library(
    ${LIB_NAME}_library
    NAMES ${LIB_NAME}
    HINTS ${ONEMKL_LIB_DIR}
    NO_CMAKE_PATH NO_CMAKE_ENVIRONMENT_PATH)
  if(NOT ${LIB_NAME}_library)
    message(WARNING "${LIB_NAME} not found in ${ONEMKL_LIB_DIR}!!")
    return()
  endif()
  list(APPEND ONEMKL_LIBRARIES ${${LIB_NAME}_library})
endforeach()

set(ONEMKL_FOUND TRUE)
