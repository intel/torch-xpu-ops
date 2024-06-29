set(mkl_root_hint $ENV{MKLROOT})

set(CMAKE_INCLUDE_PATH ${CMAKE_INCLUDE_PATH}
    "${mkl_root_hint}/include")
set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH}
    "${mkl_root_hint}/lib")

set(ONEMKL_LIBRARIES)

list(APPEND ONEMKL_LIBRARIES "${mkl_root_hint}/lib/libmkl_intel_lp64.so")
list(APPEND ONEMKL_LIBRARIES "${mkl_root_hint}/lib/libmkl_gnu_thread.so")
list(APPEND ONEMKL_LIBRARIES "${mkl_root_hint}/lib/libmkl_core.so")
list(APPEND ONEMKL_LIBRARIES "${mkl_root_hint}/lib/libmkl_sycl_dft.so")

