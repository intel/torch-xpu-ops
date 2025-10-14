macro(replace_cmake_build_flags)
  set(CMAKE_C_FLAG_BK "${CMAKE_C_FLAGS}")
  set(CMAKE_CXX_FLAGS_BK "${CMAKE_CXX_FLAGS}")
  string(REPLACE "-Werror=format" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
  string(REPLACE "-Werror=format" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endmacro()

macro(restore_cmake_build_flags)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAG_BK}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_BK}")
endmacro()

if(NOT __CUTLASS_INCLUDED)
  set(__CUTLASS_INCLUDED TRUE)
  include(FetchContent)
  FetchContent_Declare(
      repo-cutlass-sycl
      GIT_REPOSITORY https://github.com/intel/sycl-tla.git
      GIT_TAG        main
      GIT_SHALLOW    OFF
  )
  FetchContent_GetProperties(repo-cutlass-sycl)
  if(NOT repo-cutlass-sycl_POPULATED)
    FetchContent_Populate(repo-cutlass-sycl)
  endif()
  set(CUTLASS_SYCL_INCLUDE_DIRS ${repo-cutlass-sycl_SOURCE_DIR}/include
                                ${repo-cutlass-sycl_SOURCE_DIR}/tools/util/include
                                ${repo-cutlass-sycl_SOURCE_DIR}/examples/common)
  set(CUTLASS_SYCL_COMPILE_DEFINITIONS CUTLASS_ENABLE_SYCL SYCL_INTEL_TARGET)
endif()
