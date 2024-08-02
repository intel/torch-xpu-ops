find_package(ONEMKL)
if(NOT ONEMKL_FOUND)
  message(FATAL_ERROR "Can NOT find ONEMKL cmake helpers module!")
endif()

list(INSERT ONEMKL_LIBRARIES 0 "-Wl,--no-as-needed")
list(APPEND ONEMKL_LIBRARIES "-Wl,--as-needed")
