# This will define the following variables:
# ISHMEM_FOUND               : True if the system has the ISHMEM library.
# ISHMEM_INCLUDE_DIR         : Include directories needed to use ISHMEM.
# ISHMEM_LIBRARY_DIR         : The path to the ISHMEM library.
# ISHMEM_LIBRARY             : ISHMEM library fullname.

include(${CMAKE_ROOT}/Modules/FindPackageHandleStandardArgs.cmake)

if(NOT CMAKE_SYSTEM_NAME MATCHES "Linux")
  set(ISHMEM_FOUND False)
  set(ISHMEM_NOT_FOUND_MESSAGE "Intel SHMEM library is only supported on Linux!")
  return()
endif()

set(ISHMEM_ROOT $ENV{ISHMEM_ROOT})

if(NOT ISHMEM_ROOT)
  set(ISHMEM_FOUND False)
  set(ISHMEM_NOT_FOUND_MESSAGE "ISHMEM_ROOT environment variable not set. Please set it to your ISHMEM installation directory.")
  return()
endif()

# Find include path from binary.
find_path(
  ISHMEM_INCLUDE_DIR
  NAMES ishmem.h
  HINTS ${ISHMEM_ROOT}/include
  NO_DEFAULT_PATH
)

# Find library directory from binary.
find_path(
  ISHMEM_LIBRARY_DIR
  NAMES libishmem.a
  HINTS ${ISHMEM_ROOT}/lib
  NO_DEFAULT_PATH
)

# Find ISHMEM library fullname (static library).
find_library(
  ISHMEM_LIBRARY
  NAMES ishmem
  HINTS ${ISHMEM_LIBRARY_DIR}
  NO_DEFAULT_PATH
)

if((NOT ISHMEM_INCLUDE_DIR) OR (NOT ISHMEM_LIBRARY_DIR) OR (NOT ISHMEM_LIBRARY))
  set(ISHMEM_FOUND False)
  set(ISHMEM_NOT_FOUND_MESSAGE "Intel SHMEM library not found! Please set ISHMEM_ROOT environment variable.")
  return()
endif()

SET(CMAKE_INCLUDE_PATH ${CMAKE_INCLUDE_PATH}
  "${ISHMEM_INCLUDE_DIR}")
SET(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH}
  "${ISHMEM_LIBRARY_DIR}")

find_package_handle_standard_args(
  ISHMEM
  FOUND_VAR ISHMEM_FOUND
  REQUIRED_VARS ISHMEM_INCLUDE_DIR ISHMEM_LIBRARY_DIR ISHMEM_LIBRARY
  REASON_FAILURE_MESSAGE "${ISHMEM_NOT_FOUND_MESSAGE}"
)

mark_as_advanced(ISHMEM_INCLUDE_DIR ISHMEM_LIBRARY_DIR ISHMEM_LIBRARY)
