if(NOT __CUTLASS_INCLUDED)
  set(__CUTLASS_INCLUDED TRUE)
  include(FetchContent)
  FetchContent_Declare(
      repo-cutlass-sycl
      GIT_REPOSITORY https://github.com/intel/cutlass-sycl #https://github.com/rolandschulz/cutlass-fork.git
      GIT_TAG        sycl-develop #gcc-support 
      GIT_SHALLOW    OFF
  )
  FetchContent_GetProperties(repo-cutlass-sycl)
  if(NOT repo-cutlass-sycl_POPULATED)
    FetchContent_Populate(repo-cutlass-sycl)
  endif()
  set(CUTLASS_SYCL_INCLUDE_DIRS ${repo-cutlass-sycl_SOURCE_DIR}/include
                                ${repo-cutlass-sycl_SOURCE_DIR}/tools/util/include)
  set(CUTLASS_SYCL_COMPILE_DEFINITIONS CUTLASS_ENABLE_SYCL SYCL_INTEL_TARGET)
endif()