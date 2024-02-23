# Setup building flags for SYCL device and host codes.

# Support GCC only at the moment.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # # -- Host flags (SYCL_CXX_FLAGS)
  list(APPEND SYCL_HOST_FLAGS -fPIC)
  list(APPEND SYCL_HOST_FLAGS -std=c++17)
  # SYCL headers warnings
  list(APPEND SYCL_HOST_FLAGS -Wno-deprecated-declarations)
  list(APPEND SYCL_HOST_FLAGS -Wno-attributes)

  if(CMAKE_BUILD_TYPE MATCHES Debug)
    list(APPEND SYCL_HOST_FLAGS -g)
    list(APPEND SYCL_HOST_FLAGS -O0)
  endif(CMAKE_BUILD_TYPE MATCHES Debug)

  # -- Kernel flags (SYCL_KERNEL_OPTIONS)
  # The fast-math will be enabled by default in SYCL compiler.
  # Refer to [https://clang.llvm.org/docs/UsersManual.html#cmdoption-fno-fast-math]
  # 1. We enable below flags here to be warn about NaN and Infinity,
  # which will be hidden by fast-math by default.
  # 2. The associative-math in fast-math allows floating point
  # operations to be reassociated, which will lead to non-deterministic
  # results compared with CUDA backend.
  # 3. The approx-func allows certain math function calls (such as log, sqrt, pow, etc)
  # to be replaced with an approximately equivalent set of instructions or
  # alternative math function calls, which have great errors.
  set(SYCL_KERNEL_OPTIONS ${SYCL_KERNEL_OPTIONS} -fno-sycl-unnamed-lambda)
  set(SYCL_KERNEL_OPTIONS ${SYCL_KERNEL_OPTIONS} -sycl-std=2020)
  set(SYCL_KERNEL_OPTIONS ${SYCL_KERNEL_OPTIONS} -fhonor-nans)
  set(SYCL_KERNEL_OPTIONS ${SYCL_KERNEL_OPTIONS} -fhonor-infinities)
  set(SYCL_KERNEL_OPTIONS ${SYCL_KERNEL_OPTIONS} -fno-associative-math)
  set(SYCL_KERNEL_OPTIONS ${SYCL_KERNEL_OPTIONS} -fno-approx-func)
  # TODO: Align with PyTorch and switch to ABI=0 eventually, after
  # resolving incompatible implementation in SYCL runtime.
  set(SYCL_KERNEL_OPTIONS ${SYCL_KERNEL_OPTIONS} -D_GLIBCXX_USE_CXX11_ABI=1)
  set(SYCL_FLAGS ${SYCL_FLAGS} ${SYCL_KERNEL_OPTIONS})
else()
  message("Not compiling with XPU. Only support GCC compiler as CXX compiler.")
  return()
endif()
