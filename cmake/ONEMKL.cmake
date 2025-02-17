option(USE_ONEMKL "Build with ONEMKL XPU support" OFF)

if(DEFINED ENV{USE_ONEMKL})
  set(USE_ONEMKL $ENV{USE_ONEMKL})
endif()

message(STATUS "USE_ONEMKL is set to ${USE_ONEMKL}")

if(NOT USE_ONEMKL)
  return()
endif()

find_package(ONEMKL)
if(NOT ONEMKL_FOUND)
  message(FATAL_ERROR "Can NOT find ONEMKL cmake helpers module!")
endif()

set(TORCH_XPU_OPS_ONEMKL_INCLUDE_DIR ${ONEMKL_INCLUDE_DIR})

set(TORCH_XPU_OPS_ONEMKL_LIBRARIES ${ONEMKL_LIBRARIES})

list(INSERT TORCH_XPU_OPS_ONEMKL_LIBRARIES 1 "-Wl,--start-group")
list(APPEND TORCH_XPU_OPS_ONEMKL_LIBRARIES "-Wl,--end-group")
