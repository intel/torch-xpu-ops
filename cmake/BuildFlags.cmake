# Setup building flags for SYCL device and host codes.

# Support GCC only at the moment.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # # -- Host flags (SYCL_CXX_FLAGS)
  list(APPEND SYCL_HOST_FLAGS -fPIC)
  list(APPEND SYCL_HOST_FLAGS -std=c++17)
  # SYCL headers warnings
  list(APPEND SYCL_HOST_FLAGS -Wno-deprecated-declarations)
  list(APPEND SYCL_HOST_FLAGS -Wno-deprecated)
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
  #
  # PSEUDO of separate compilation with DPCPP compiler.
  # 1. Kernel source compilation:
  # icpx -fsycl -fsycl-target=${SYCL_TARGETS_OPTION} ${SYCL_FLAGS} -fsycl-host-compiler=gcc -fsycl-host-compiler-options='${CMAKE_HOST_FLAGS}' kernel.cpp -o kernel.o
  # 2. Device code linkage:
  # icpx -fsycl -fsycl-target=${SYCL_TARGETS_OPTION} -fsycl-link ${SYCL_DEVICE_LINK_FLAGS} -Xs '${SYCL_OFFLINE_COMPILER_FLAGS}' kernel.o -o device-code.o
  # 3. Host only source compilation:
  # gcc ${CMAKE_HOST_FLAGS} host.cpp -o host.o
  # 4. Linkage:
  # gcc -shared host.o kernel.o device-code.o -o libxxx.so
  set(SYCL_TARGETS_OPTION -fsycl-targets=spir64_gen,spir64)
  set(SYCL_KERNEL_OPTIONS ${SYCL_KERNEL_OPTIONS} ${SYCL_TARGETS_OPTION})
  set(SYCL_KERNEL_OPTIONS ${SYCL_KERNEL_OPTIONS} -fno-sycl-unnamed-lambda)
  set(SYCL_KERNEL_OPTIONS ${SYCL_KERNEL_OPTIONS} -sycl-std=2020)
  set(SYCL_KERNEL_OPTIONS ${SYCL_KERNEL_OPTIONS} -fhonor-nans)
  set(SYCL_KERNEL_OPTIONS ${SYCL_KERNEL_OPTIONS} -fhonor-infinities)
  set(SYCL_KERNEL_OPTIONS ${SYCL_KERNEL_OPTIONS} -fno-associative-math)
  set(SYCL_KERNEL_OPTIONS ${SYCL_KERNEL_OPTIONS} -fno-approx-func)
  set(SYCL_KERNEL_OPTIONS ${SYCL_KERNEL_OPTIONS} -Wno-absolute-value)
  # TODO: Align with PyTorch and switch to ABI=0 eventually, after
  # resolving incompatible implementation in SYCL runtime.
  set(SYCL_KERNEL_OPTIONS ${SYCL_KERNEL_OPTIONS} -D_GLIBCXX_USE_CXX11_ABI=1)
  set(SYCL_FLAGS ${SYCL_FLAGS} ${SYCL_KERNEL_OPTIONS})

  set(TORCH_XPU_OPS_FLAGS ${SYCL_HOST_FLAGS})

  # -- SYCL device object linkage flags
  include(ProcessorCount)
  ProcessorCount(proc_cnt)
  if ((DEFINED ENV{MAX_JOBS}) AND ("$ENV{MAX_JOBS}" LESS_EQUAL ${proc_cnt}))
    set(SYCL_MAX_PARALLEL_LINK_JOBS $ENV{MAX_JOBS})
  else()
    set(SYCL_MAX_PARALLEL_LINK_JOBS ${proc_cnt})
  endif()
  set(SYCL_DEVICE_LINK_FLAGS ${SYCL_DEVICE_LINK_FLAGS} -fsycl-max-parallel-link-jobs=${SYCL_MAX_PARALLEL_LINK_JOBS})
  set(SYCL_DEVICE_LINK_FLAGS ${SYCL_DEVICE_LINK_FLAGS} ${SYCL_TARGETS_OPTION})

  set(SYCL_OFFLINE_COMPILER_CG_OPTIONS ${SYCL_OFFLINE_COMPILER_CG_OPTIONS} "-options \"-cl-intel-enable-auto-large-GRF-mode\"")
  set(SYCL_OFFLINE_COMPILER_CG_OPTIONS ${SYCL_OFFLINE_COMPILER_CG_OPTIONS} "-options \"-cl-poison-unsupported-fp64-kernels\"")
  # Support PVC AOT only currently.
  set(SYCL_OFFLINE_COMPILER_AOT_OPTIONS "-device pvc")
  set(SYCL_OFFLINE_COMPILER_FLAGS "${SYCL_OFFLINE_COMPILER_AOT_OPTIONS} ${SYCL_OFFLINE_COMPILER_CG_OPTIONS}")
else()
  message("Not compiling with XPU. Only support GCC compiler as CXX compiler.")
  return()
endif()
