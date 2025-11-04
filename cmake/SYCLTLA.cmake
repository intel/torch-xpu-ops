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

if(NOT __SYCLTLA_INCLUDED)
  set(__SYCLTLA_INCLUDED TRUE)
  include(FetchContent)
  FetchContent_Declare(
      repo-sycl-tla
      GIT_REPOSITORY https://github.com/intel/sycl-tla.git
      GIT_TAG        v0.6
      GIT_SHALLOW    OFF
  )
  FetchContent_GetProperties(repo-sycl-tla)
  if(NOT repo-sycl-tla_POPULATED)
    FetchContent_Populate(repo-sycl-tla)
  endif()
  set(SYCLTLA_INCLUDE_DIRS ${repo-sycl-tla_SOURCE_DIR}/include
                           ${repo-sycl-tla_SOURCE_DIR}/applications
                           ${repo-sycl-tla_SOURCE_DIR}/tools/util/include)
  set(SYCLTLA_COMPILE_DEFINITIONS CUTLASS_ENABLE_SYCL SYCL_INTEL_TARGET)
endif()
