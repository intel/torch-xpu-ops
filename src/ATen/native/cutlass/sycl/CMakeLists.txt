cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

set(PROJECT_NAME "cutlass_kernels")
set(PROJECT_VERSION "0.0.0")

# Avoid SYCL compiler error
if(NOT WIN32)
  string(APPEND CMAKE_CXX_FLAGS " -Wno-error")
endif()

cmake_policy(SET CMP0048 NEW)
project(${PROJECT_NAME} VERSION "${PROJECT_VERSION}" LANGUAGES C CXX)

# Fetch cutlass-sycl repo
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
include_directories(${repo-cutlass-sycl_SOURCE_DIR}/include
                    ${repo-cutlass-sycl_SOURCE_DIR}/tools/util/include)
add_compile_definitions(CUTLASS_ENABLE_SYCL)
add_compile_definitions(SYCL_INTEL_TARGET)

# Build cutlass_kernels library
file(GLOB srcs "*.cpp")
add_library(
    cutlass_kernels
    SHARED
    ${srcs})

set(SYCL_FLAGS -fsycl)
set(SYCL_TARGETS_OPTION -fsycl-targets=intel_gpu_pvc,intel_gpu_bmg_g21)
set(SYCL_KERNEL_OPTIONS ${SYCL_KERNEL_OPTIONS} ${SYCL_FLAGS} ${SYCL_TARGETS_OPTION})
set(SYCL_KERNEL_OPTIONS ${SYCL_KERNEL_OPTIONS} --offload-new-driver)
set(SYCL_KERNEL_OPTIONS ${SYCL_KERNEL_OPTIONS} -Xspirv-translator)
set(SYCL_KERNEL_OPTIONS ${SYCL_KERNEL_OPTIONS} -spirv-ext=+SPV_INTEL_split_barrier)

if(CMAKE_BUILD_TYPE MATCHES Debug)
  set(SYCL_KERNEL_OPTIONS ${SYCL_KERNEL_OPTIONS} -g -O0 -Rno-debug-disables-optimization)
elseif(CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
  set(SYCL_KERNEL_OPTIONS ${SYCL_KERNEL_OPTIONS} -gline-tables-only -O2)
endif()

target_compile_options(cutlass_kernels PRIVATE ${SYCL_KERNEL_OPTIONS})
target_link_options(cutlass_kernels PRIVATE ${SYCL_KERNEL_OPTIONS})
