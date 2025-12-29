# Setup building flags for SYCL device and host codes.

function(CHECK_SYCL_FLAG FLAG VARIABLE_NAME)
  set(TEMP_DIR "${CMAKE_BINARY_DIR}/temp")
  file(MAKE_DIRECTORY ${TEMP_DIR})
  set(TEST_SRC_FILE "${TEMP_DIR}/check_options.cpp")
  set(TEST_EXE_FILE "${TEMP_DIR}/check_options.out")
  file(WRITE ${TEST_SRC_FILE} "#include <iostream>\nint main() { std::cout << \"Checking compiler options ...\" << std::endl; return 0; }\n")
  execute_process(
      COMMAND ${SYCL_COMPILER} -fsycl ${TEST_SRC_FILE} -o ${TEST_EXE_FILE} ${FLAG}
      WORKING_DIRECTORY ${TEMP_DIR}
      OUTPUT_VARIABLE output
      ERROR_VARIABLE output
      RESULT_VARIABLE result
      TIMEOUT 60
  )
  if(result EQUAL 0)
      set(${VARIABLE_NAME} TRUE PARENT_SCOPE)
  else()
      set(${VARIABLE_NAME} FALSE PARENT_SCOPE)
  endif()
  file(REMOVE_RECURSE ${TEMP_DIR})
endfunction()

macro(set_build_flags)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(SYCL_HOST_FLAGS)
    set(SYCL_KERNEL_OPTIONS)
    set(SYCL_COMPILE_FLAGS ${SYCL_FLAGS})
    set(SYCL_DEVICE_LINK_FLAGS ${SYCL_LINK_FLAGS})
    set(SYCL_OFFLINE_COMPILER_AOT_OPTIONS)
    set(SYCL_OFFLINE_COMPILER_CG_OPTIONS)
    set(SYCL_OFFLINE_COMPILER_FLAGS)

    if(REPLACE_FLAGS_FOR_SYCLTLA)
      set(CPP_STD c++20)
    else()
      set(CPP_STD c++17)
    endif()
    # # -- Host flags (SYCL_CXX_FLAGS)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
      list(APPEND SYCL_HOST_FLAGS /std:${CPP_STD})
      list(APPEND SYCL_HOST_FLAGS /MD)
      list(APPEND SYCL_HOST_FLAGS /EHsc) # exception handling
      # SYCL headers warnings
      list(APPEND SYCL_HOST_FLAGS /wd4996) # allow usage of deprecated functions
      list(APPEND SYCL_HOST_FLAGS /wd4018) # allow signed and unsigned comparison
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      list(APPEND SYCL_HOST_FLAGS -fPIC)
      list(APPEND SYCL_HOST_FLAGS -std=${CPP_STD})
      list(APPEND SYCL_HOST_FLAGS -Wunused-variable)
      list(APPEND SYCL_HOST_FLAGS -Wno-interference-size)
      # Some versions of DPC++ compiler pass paths to SYCL headers as user include paths (`-I`) rather
      # than system paths (`-isystem`). This makes host compiler to report warnings encountered in the
      # SYCL headers, such as deprecated warnings, even if warned API is not actually used in the program.
      # We expect that this issue will be addressed in the later version of DPC++ compiler. To workaround
      # the issue we wrap paths to SYCL headers in `-isystem`.
      if(SYCL_COMPILER_VERSION VERSION_LESS 20250300)
        foreach(FLAGS IN LISTS SYCL_INCLUDE_DIR)
          list(APPEND SYCL_HOST_FLAGS "-isystem ${FLAGS}")
        endforeach()
      endif()
      # Excluding warnings which flood the compilation output
      # TODO: fix warnings in the source code and then reenable them in compilation
      list(APPEND SYCL_HOST_FLAGS -Wno-sign-compare)
    endif()

    if(CMAKE_BUILD_TYPE MATCHES Debug)
      list(APPEND SYCL_HOST_FLAGS -g -fno-omit-frame-pointer -O0)
    elseif(CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
      list(APPEND SYCL_HOST_FLAGS -g -O2)
    endif()
    if(USE_PER_OPERATOR_HEADERS)
      list(APPEND SYCL_HOST_FLAGS -DAT_PER_OPERATOR_HEADERS)
    endif()
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
    # icpx -fsycl -fsycl-target=${SYCL_TARGETS_OPTION} ${SYCL_KERNEL_OPTIONS} -fsycl-host-compiler=gcc -fsycl-host-compiler-options='${CMAKE_HOST_FLAGS}' kernel.cpp -o kernel.o
    # 2. Device code linkage:
    # icpx -fsycl -fsycl-target=${SYCL_TARGETS_OPTION} -fsycl-link ${SYCL_DEVICE_LINK_FLAGS} -Xs '${SYCL_OFFLINE_COMPILER_FLAGS}' kernel.o -o device-code.o
    # 3. Host only source compilation:
    # gcc ${CMAKE_HOST_FLAGS} host.cpp -o host.o
    # 4. Linkage:
    # gcc -shared host.o kernel.o device-code.o -o libxxx.so
    set(SYCL_KERNEL_OPTIONS ${SYCL_KERNEL_OPTIONS} -fno-sycl-unnamed-lambda)
    set(SYCL_KERNEL_OPTIONS ${SYCL_KERNEL_OPTIONS} -sycl-std=2020)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
      set(SYCL_KERNEL_OPTIONS ${SYCL_KERNEL_OPTIONS} /fp:strict)
      # Suppress warnings about dllexport.
      set(SYCL_KERNEL_OPTIONS ${SYCL_KERNEL_OPTIONS} -Wno-ignored-attributes)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      set(SYCL_KERNEL_OPTIONS ${SYCL_KERNEL_OPTIONS} -fhonor-nans)
      set(SYCL_KERNEL_OPTIONS ${SYCL_KERNEL_OPTIONS} -fhonor-infinities)
      set(SYCL_KERNEL_OPTIONS ${SYCL_KERNEL_OPTIONS} -fno-associative-math)
      set(SYCL_KERNEL_OPTIONS ${SYCL_KERNEL_OPTIONS} -fno-approx-func)
      set(SYCL_KERNEL_OPTIONS ${SYCL_KERNEL_OPTIONS} -Wno-absolute-value)
      set(SYCL_KERNEL_OPTIONS ${SYCL_KERNEL_OPTIONS} -no-ftz)
    endif()

    if(CMAKE_BUILD_TYPE MATCHES Debug)
      set(SYCL_KERNEL_OPTIONS ${SYCL_KERNEL_OPTIONS} -g -O0 -Rno-debug-disables-optimization)
    elseif(CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
      set(SYCL_KERNEL_OPTIONS ${SYCL_KERNEL_OPTIONS} -gline-tables-only -O2)
    endif()

    CHECK_SYCL_FLAG("-fsycl-fp64-conv-emu" SUPPORTS_FP64_CONV_EMU)
    if(NOT SUPPORTS_FP64_CONV_EMU)
      message(WARNING "The compiler does not support the '-fsycl-fp64-conv-emu' flag, \
      will disable it. On some platforms that don't support FP64, \
      running operations with the FP64 datatype will raise a Runtime error: Required aspect fp64 is not supported on the device \
      or a Native API failed error.")
    endif()

    set(TORCH_XPU_OPS_FLAGS ${SYCL_HOST_FLAGS})

    # -- SYCL device object linkage flags
    include(ProcessorCount)
    ProcessorCount(proc_cnt)
    if((DEFINED ENV{MAX_JOBS}) AND ("$ENV{MAX_JOBS}" LESS_EQUAL ${proc_cnt}))
      set(SYCL_MAX_PARALLEL_LINK_JOBS $ENV{MAX_JOBS})
    else()
      set(SYCL_MAX_PARALLEL_LINK_JOBS ${proc_cnt})
    endif()
    set(SYCL_DEVICE_LINK_FLAGS ${SYCL_DEVICE_LINK_FLAGS} -fsycl-max-parallel-link-jobs=${SYCL_MAX_PARALLEL_LINK_JOBS})
    set(SYCL_DEVICE_LINK_FLAGS ${SYCL_DEVICE_LINK_FLAGS} --offload-compress)

    set(SYCL_OFFLINE_COMPILER_CG_OPTIONS "${SYCL_OFFLINE_COMPILER_CG_OPTIONS} -options -cl-poison-unsupported-fp64-kernels")
    set(SYCL_OFFLINE_COMPILER_CG_OPTIONS "${SYCL_OFFLINE_COMPILER_CG_OPTIONS} -options -cl-intel-enable-auto-large-GRF-mode")
    set(SYCL_OFFLINE_COMPILER_CG_OPTIONS "${SYCL_OFFLINE_COMPILER_CG_OPTIONS} -options -cl-fp32-correctly-rounded-divide-sqrt")
    set(SYCL_OFFLINE_COMPILER_CG_OPTIONS "${SYCL_OFFLINE_COMPILER_CG_OPTIONS} -options -cl-intel-greater-than-4GB-buffer-required")

    if(REPLACE_FLAGS_FOR_SYCLTLA)
      set(SYCL_TARGETS_OPTION -fsycl-targets=spir64_gen)
      set(SYCL_KERNEL_OPTIONS ${SYCL_KERNEL_OPTIONS} ${SYCL_TARGETS_OPTION})
      set(SYCL_DEVICE_LINK_FLAGS ${SYCL_DEVICE_LINK_FLAGS} ${SYCL_TARGETS_OPTION})
      set(SYCL_DEVICE_LINK_FLAGS ${SYCL_DEVICE_LINK_FLAGS} "-Xspirv-translator;-spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate")
      set(SYCL_OFFLINE_COMPILER_AOT_OPTIONS "-device pvc,bmg")
    else()
      if(WIN32)
        set(AOT_TARGETS "mtl,mtl-h,bmg,dg2,arl-h,lnl-m,ptl")
      else()
        set(AOT_TARGETS "pvc,bmg,dg2,arl-h,mtl-h,lnl-m,ptl-h,ptl-u")
      endif()
      if(TORCH_XPU_ARCH_LIST)
        set(AOT_TARGETS "${TORCH_XPU_ARCH_LIST}")
      endif()
      if(AOT_TARGETS STREQUAL "none")
        set(TORCH_XPU_ARCH_LIST "" PARENT_SCOPE)
      else()
        if(SUPPORTS_FP64_CONV_EMU)
          string(FIND "${AOT_TARGETS}" "dg2" _dg2_index)
          string(FIND "${AOT_TARGETS}" "ats-m" _atsm_index)
          if(_dg2_index GREATER_EQUAL 0 OR _atsm_index GREATER_EQUAL 0)
            set(SYCL_KERNEL_OPTIONS ${SYCL_KERNEL_OPTIONS} -fsycl-fp64-conv-emu)
            set(SYCL_DEVICE_LINK_FLAGS ${SYCL_DEVICE_LINK_FLAGS} -fsycl-fp64-conv-emu)
          endif()
        endif()
        set(SYCL_TARGETS_OPTION -fsycl-targets=spir64_gen,spir64)
        set(SYCL_KERNEL_OPTIONS ${SYCL_KERNEL_OPTIONS} ${SYCL_TARGETS_OPTION})
        set(SYCL_DEVICE_LINK_FLAGS ${SYCL_DEVICE_LINK_FLAGS} ${SYCL_TARGETS_OPTION})
        set(SYCL_OFFLINE_COMPILER_AOT_OPTIONS "-device ${AOT_TARGETS}")
        set(TORCH_XPU_ARCH_LIST ${AOT_TARGETS} PARENT_SCOPE)
      endif()
      message(STATUS "Compile Intel GPU AOT Targets for ${AOT_TARGETS}")
    endif()

    set(SYCL_COMPILE_FLAGS ${SYCL_COMPILE_FLAGS} ${SYCL_KERNEL_OPTIONS})

    set(SYCL_OFFLINE_COMPILER_FLAGS "${SYCL_OFFLINE_COMPILER_AOT_OPTIONS}${SYCL_OFFLINE_COMPILER_CG_OPTIONS}")
  else()
    message("Not compiling with XPU. Currently only support GCC compiler on Linux and MSVC compiler on Windows as CXX compiler.")
    return()
  endif()
endmacro()
