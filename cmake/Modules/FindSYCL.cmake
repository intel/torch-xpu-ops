# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

#.rst:
# FindSYCL
# --------
#
# .. note::

# The following variables affect the behavior of the macros in the script needed
# to be defined before calling ``SYCL_ADD_EXECUTABLE`` or ``SYCL_ADD_LIBRARY``::
#
#  SYCL_COMPILER
#  -- SYCL compiler's executable.
#
#  SYCL_COMPILE_FLAGS
#  -- SYCL compiler's compilation command line arguments.
#
#  SYCL_HOST_FLAGS
#  -- SYCL compiler's 3rd party host compiler (e.g. gcc) arguments .
#
#  SYCL_DEVICE_LINK_FLAGS
#  -- Arguments used when linking device object.
#
#  SYCL_OFFLINE_COMPILER_FLAGS
#  -- Arguments used by offline compiler at AOT compilation.
#
#  SYCL_INCLUDE_DIR
#  -- Include directory for SYCL compiler/runtime headers.
#
#  SYCL_LIBRARY_DIR
#  -- Include directory for SYCL compiler/runtime libraries.

# Helpers::
# Introduce SYCL compiler to build .cpp containing SYCL kernel.
#
#  SYCL_ADD_EXECUTABLE
#
#  SYCL_ADD_LIBRARY

if(NOT CMAKE_SYCL_COMPILER_LAUNCHER AND DEFINED ENV{CMAKE_SYCL_COMPILER_LAUNCHER})
  set(CMAKE_SYCL_COMPILER_LAUNCHER "$ENV{CMAKE_SYCL_COMPILER_LAUNCHER}"
    CACHE STRING "Compiler launcher for SYCL.")
endif()

macro(SYCL_FIND_HELPER_FILE _name _extension)
  set(_full_name "${_name}.${_extension}")
  set(SYCL_${_name} "${CMAKE_CURRENT_LIST_DIR}/FindSYCL/${_full_name}")
  if(NOT EXISTS "${SYCL_${_name}}")
    set(error_message "${_full_name} not found in ${CMAKE_CURRENT_LIST_DIR}/FindSYCL")
    message(FATAL_ERROR "${error_message}")
  endif()
  # Set this variable as internal, so the user isn't bugged with it.
  set(SYCL_${_name} ${SYCL_${_name}} CACHE INTERNAL "Location of ${_full_name}" FORCE)
endmacro()

# SYCL_HOST_COMPILER
set(SYCL_HOST_COMPILER "${CMAKE_CXX_COMPILER}"
  CACHE FILEPATH "Host side compiler used by SYCL")

# SYCL_EXECUTABLE
set(SYCL_EXECUTABLE ${SYCL_COMPILER} CACHE FILEPATH "SYCL compiler")

# Parse HOST_COMPILATION mode.
option(SYCL_HOST_COMPILATION_CXX "Generated file extension" ON)

# SYCL_VERBOSE_BUILD
option(SYCL_VERBOSE_BUILD "Print out the commands run while compiling the SYCL source file.  With the Makefile generator this defaults to VERBOSE variable specified on the command line, but can be forced on with this option." OFF)

macro(SYCL_INCLUDE_EXTERNAL_DEPENDENCIES dependency_file)
  list(APPEND SYCL_EXTERNAL_DEPEND ${dependency_file})
endmacro()

macro(SYCL_INCLUDE_DEPENDENCIES dependency_file)
  set(SYCL_DEPEND)
  set(SYCL_DEPEND_REGENERATE FALSE)

  # Make the output depend on the dependency file itself, which should cause the
  # rule to re-run.
  if(NOT EXISTS ${dependency_file})
    file(WRITE ${dependency_file} "#FindSYCL.cmake generated file.  Do not edit.\n")
  endif()

  # Always include this file to force CMake to run again next
  # invocation and rebuild the dependencies.
  include(${dependency_file})

  if(SYCL_DEPEND)
    foreach(f ${SYCL_DEPEND})
      if(NOT EXISTS ${f})
        set(SYCL_DEPEND_REGENERATE TRUE)
      endif()
    endforeach()
  else()
    set(SYCL_DEPEND_REGENERATE TRUE)
  endif()

  if(SYCL_DEPEND_REGENERATE)
    set(SYCL_DEPEND ${dependency_file})
    file(WRITE ${dependency_file} "#FindSYCL.cmake generated file.  Do not edit.\n")
  endif()
endmacro()

sycl_find_helper_file(make2cmake cmake)
sycl_find_helper_file(run_sycl cmake)

# Per-config subpaths under multi-config generators; flat for single-config.
if(CMAKE_CONFIGURATION_TYPES)
  set(SYCL_config_subdir "/$<CONFIG>")
  set(SYCL_config_suffix ".$<CONFIG>")
else()
  set(SYCL_config_subdir "")
  set(SYCL_config_suffix "")
endif()

function(SYCL_GET_SOURCES_AND_OPTIONS _sycl_sources _cxx_sources _cmake_options)
  cmake_parse_arguments(PARSE_ARGV 3 PARSED_SYCL
    "STATIC;SHARED;MODULE;EXCLUDE_FROM_ALL"
    ""
    "SYCL_SOURCES;CXX_SOURCES")
  if("OPTIONS" IN_LIST ARGN)
    message(FATAL_ERROR "sycl_add_executable/library doesn't support OPTIONS keyword.")
  endif()
  set(${_sycl_sources} ${PARSED_SYCL_SYCL_SOURCES} PARENT_SCOPE)
  set(${_cxx_sources} ${PARSED_SYCL_CXX_SOURCES} PARENT_SCOPE)
  set(_opts "")
  foreach(_kw IN ITEMS STATIC SHARED MODULE EXCLUDE_FROM_ALL)
    if(PARSED_SYCL_${_kw})
      list(APPEND _opts ${_kw})
    endif()
  endforeach()
  set(${_cmake_options} ${_opts} PARENT_SCOPE)
endfunction()

function(SYCL_BUILD_SHARED_LIBRARY shared_flag)
  set(cmake_args ${ARGN})
  # If SHARED, MODULE, or STATIC aren't already in the list of arguments, then
  # add SHARED or STATIC based on the value of BUILD_SHARED_LIBS.
  list(FIND cmake_args SHARED _sycl_found_SHARED)
  list(FIND cmake_args MODULE _sycl_found_MODULE)
  list(FIND cmake_args STATIC _sycl_found_STATIC)
  if( _sycl_found_SHARED GREATER -1 OR
      _sycl_found_MODULE GREATER -1 OR
      _sycl_found_STATIC GREATER -1)
    set(_sycl_build_shared_libs)
  else()
    if(BUILD_SHARED_LIBS)
      set(_sycl_build_shared_libs SHARED)
    else()
      set(_sycl_build_shared_libs STATIC)
    endif()
  endif()
  set(${shared_flag} ${_sycl_build_shared_libs} PARENT_SCOPE)
endfunction()

macro(SYCL_WRAP_SRCS sycl_target generated_files)
  # Optional arguments
  set(generated_extension ${CMAKE_${SYCL_C_OR_CXX}_OUTPUT_EXTENSION})

  set(SYCL_include_dirs "${SYCL_INCLUDE_DIR}")
  list(APPEND SYCL_include_dirs "$<TARGET_PROPERTY:${sycl_target},INCLUDE_DIRECTORIES>")

  set(SYCL_compile_definitions "$<TARGET_PROPERTY:${sycl_target},COMPILE_DEFINITIONS>")

  # Extra definitions for the SYCL device compiler only, not host C++ code.
  set(SYCL_compile_definitions "${SYCL_compile_definitions};${SYCL_DEVICE_COMPILE_DEFINITIONS}")

  SYCL_GET_SOURCES_AND_OPTIONS(
    _sycl_sources
    _cxx_sources
    _cmake_options
    ${ARGN})

  set(_SYCL_build_shared_libs FALSE)
  list(FIND _cmake_options SHARED _SYCL_found_SHARED)
  list(FIND _cmake_options MODULE _SYCL_found_MODULE)
  if(_SYCL_found_SHARED GREATER -1 OR _SYCL_found_MODULE GREATER -1)
    set(_SYCL_build_shared_libs TRUE)
  endif()
  # STATIC
  list(FIND _cmake_options STATIC _SYCL_found_STATIC)
  if(_SYCL_found_STATIC GREATER -1)
    set(_SYCL_build_shared_libs FALSE)
  endif()

  if(_SYCL_build_shared_libs)
    # If we are setting up code for a shared library, then we need to add extra flags for
    # compiling objects for shared libraries.
    set(SYCL_HOST_SHARED_FLAGS ${CMAKE_SHARED_LIBRARY_${SYCL_C_OR_CXX}_FLAGS})
  else()
    set(SYCL_HOST_SHARED_FLAGS)
  endif()

  set(_sycl_c_or_cxx_flags ${CMAKE_${SYCL_C_OR_CXX}_FLAGS})
  set(_sycl_host_flags "set(CMAKE_HOST_FLAGS ${_sycl_c_or_cxx_flags} ${SYCL_HOST_SHARED_FLAGS} ${SYCL_HOST_FLAGS})")
  set(SYCL_host_flags ${_sycl_host_flags})

  # Reset the output variable
  set(_SYCL_wrap_generated_files "")
  foreach(file ${_sycl_sources})
    get_source_file_property(_is_header ${file} HEADER_FILE_ONLY)
    # SYCL kernels are in .cpp file
    if((${file} MATCHES "\\.cpp$") AND NOT _is_header)

      set(SYCL_compile_intermediate_directory "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${sycl_target}.dir")

      # CMake only names objects for sources it compiles, so uniquify same-named
      # sources by hashing their source dir (relative, to survive a moved tree).
      cmake_path(GET file FILENAME basename)
      cmake_path(ABSOLUTE_PATH file BASE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" NORMALIZE OUTPUT_VARIABLE _sycl_abs_file)
      cmake_path(GET _sycl_abs_file PARENT_PATH _sycl_src_dir)
      cmake_path(RELATIVE_PATH _sycl_src_dir BASE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
      string(SHA256 _sycl_dir_hash "${_sycl_src_dir}")
      string(SUBSTRING "${_sycl_dir_hash}" 0 8 _sycl_dir_hash)

      set(generated_file_path "${SYCL_compile_intermediate_directory}${SYCL_config_subdir}")
      set(generated_file_basename "${sycl_target}_gen_${_sycl_dir_hash}_${basename}${generated_extension}")
      set(generated_file "${generated_file_path}/${generated_file_basename}")
      set(SYCL_generated_dependency_file "${SYCL_compile_intermediate_directory}/${generated_file_basename}${SYCL_config_suffix}.SYCL-depend") # generate by compiler options -M -MF
      set(cmake_dependency_file "${SYCL_compile_intermediate_directory}/${generated_file_basename}${SYCL_config_suffix}.depend") # parse and convert SYCL_generated_dependency_file(compiler format) to cmake format
      set(custom_target_script_pregen "${SYCL_compile_intermediate_directory}/${generated_file_basename}.cmake.pre-gen")
      set(custom_target_script "${SYCL_compile_intermediate_directory}/${generated_file_basename}${SYCL_config_suffix}.cmake")

      set_source_files_properties("${generated_file}"
        PROPERTIES
        EXTERNAL_OBJECT true # This is an object file not to be compiled, but only be linked.
        )

      # Don't add CMAKE_CURRENT_SOURCE_DIR if the path is already an absolute path.
      cmake_path(GET file PARENT_PATH file_path)
      if(IS_ABSOLUTE "${file_path}")
        set(source_file "${file}")
      else()
        set(source_file "${CMAKE_CURRENT_SOURCE_DIR}/${file}")
      endif()

      list(APPEND ${sycl_target}_INTERMEDIATE_LINK_OBJECTS "${generated_file}")

      # Configure-time bookkeeping can't expand $<CONFIG>, so enumerate the
      # per-config depend files literally; the command depends on their union.
      if(CMAKE_CONFIGURATION_TYPES)
        set(_sycl_depend_files "")
        foreach(_config ${CMAKE_CONFIGURATION_TYPES})
          list(APPEND _sycl_depend_files "${SYCL_compile_intermediate_directory}/${generated_file_basename}.${_config}.depend")
        endforeach()
      else()
        set(_sycl_depend_files "${cmake_dependency_file}")
      endif()
      set(SYCL_ACCUMULATED_DEPEND)
      foreach(_depend_file ${_sycl_depend_files})
        SYCL_INCLUDE_DEPENDENCIES("${_depend_file}")
        list(APPEND SYCL_ACCUMULATED_DEPEND ${SYCL_DEPEND})
      endforeach()

      set(SYCL_build_type "Device")

      # Configure the build script
      configure_file("${SYCL_run_sycl}" "${custom_target_script_pregen}" @ONLY)
      file(GENERATE
        OUTPUT "${custom_target_script}"
        INPUT "${custom_target_script_pregen}"
        )

      set(main_dep MAIN_DEPENDENCY ${source_file})

      if(SYCL_VERBOSE_BUILD)
        set(verbose_output ON)
      elseif(CMAKE_GENERATOR MATCHES "Makefiles")
        set(verbose_output "$(VERBOSE)")
      # This condition lets us also turn on verbose output when someone
      # specifies CMAKE_VERBOSE_MAKEFILE, even if the generator isn't
      # the Makefiles generator (this is important for us, Ninja users.)
      elseif(CMAKE_VERBOSE_MAKEFILE)
        set(verbose_output ON)
      else()
        set(verbose_output OFF)
      endif()

      set(SYCL_build_comment_string "Building SYCL (${SYCL_build_type}) object ${generated_file_basename}")

      # Build the generated file and dependency file ##########################
      add_custom_command(
        OUTPUT ${generated_file}
        # These output files depend on the source_file and the contents of cmake_dependency_file
        ${main_dep}
        DEPENDS ${SYCL_ACCUMULATED_DEPEND}
        DEPENDS ${SYCL_EXTERNAL_DEPEND}
        DEPENDS ${custom_target_script}
        # Make sure the output directory exists before trying to write to it.
        COMMAND ${CMAKE_COMMAND} -E make_directory "${generated_file_path}"
        COMMAND ${CMAKE_COMMAND} ARGS
          -D verbose:BOOL=${verbose_output}
          -D "generated_file:STRING=${generated_file}"
          -P "${custom_target_script}"
        WORKING_DIRECTORY "${SYCL_compile_intermediate_directory}"
        COMMENT "${SYCL_build_comment_string}"
        )

      # Make sure the build system knows the file is generated.
      set_source_files_properties(${generated_file} PROPERTIES GENERATED TRUE)

      list(APPEND _SYCL_wrap_generated_files ${generated_file})

      # Add the other files that we want cmake to clean on a cleanup ##########
      list(APPEND SYCL_ADDITIONAL_CLEAN_FILES "${cmake_dependency_file}")
      list(REMOVE_DUPLICATES SYCL_ADDITIONAL_CLEAN_FILES)
      set(SYCL_ADDITIONAL_CLEAN_FILES ${SYCL_ADDITIONAL_CLEAN_FILES} CACHE INTERNAL "List of intermediate files that are part of the SYCL dependency scanning.")
    endif()
  endforeach()

  # Set the return parameter
  set(${generated_files} ${_SYCL_wrap_generated_files})
endmacro()

function(_sycl_get_important_host_flags important_flags flag_string)
  string(REGEX MATCHALL "-fPIC" flags "${flag_string}")
  list(APPEND ${important_flags} ${flags})
  set(${important_flags} ${${important_flags}} PARENT_SCOPE)
endfunction()

###############################################################################
# Custom Intermediate Link

# Compute the filename to be used by SYCL_LINK_DEVICE_OBJECTS
function(SYCL_COMPUTE_DEVICE_OBJECT_FILE_NAME output_file_var sycl_target)
  set(generated_extension ${CMAKE_${SYCL_C_OR_CXX}_OUTPUT_EXTENSION})
  set(output_file "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${sycl_target}.dir${SYCL_config_subdir}/${sycl_target}_sycl_device_obj${generated_extension}")
  set(${output_file_var} "${output_file}" PARENT_SCOPE)
endfunction()

macro(SYCL_LINK_DEVICE_OBJECTS output_file sycl_target)
  set(object_files)
  list(APPEND object_files ${ARGN})

  if(object_files)

    set_source_files_properties("${output_file}"
      PROPERTIES
      EXTERNAL_OBJECT TRUE # This is an object file not to be compiled, but only
                           # be linked.
      GENERATED TRUE       # This file is generated during the build
      )

    set(SYCL_device_link_flags)
    set(important_host_flags)
    _sycl_get_important_host_flags(important_host_flags "${SYCL_HOST_FLAGS}")
    set(SYCL_device_link_flags
        ${important_host_flags}
        ${SYCL_DEVICE_LINK_FLAGS})

    # output_file is a macro arg, not a variable, so cmake_path() can't take it.
    file(RELATIVE_PATH output_file_relative_path "${CMAKE_BINARY_DIR}" "${output_file}")

    if(SYCL_VERBOSE_BUILD)
      set(verbose_output ON)
    elseif(CMAKE_GENERATOR MATCHES "Makefiles")
      set(verbose_output "$(VERBOSE)")
    # This condition lets us also turn on verbose output when someone
    # specifies CMAKE_VERBOSE_MAKEFILE, even if the generator isn't
    # the Makefiles generator (this is important for us, Ninja users.)
    elseif(CMAKE_VERBOSE_MAKEFILE)
      set(verbose_output ON)
    else()
      set(verbose_output OFF)
    endif()

    # Build the generated file and dependency file ##########################
    add_custom_command(
      OUTPUT ${output_file}
      DEPENDS ${object_files}
      COMMAND ${CMAKE_COMMAND} -E make_directory "$<PATH:REMOVE_FILENAME,${output_file}>"
      COMMAND ${CMAKE_SYCL_COMPILER_LAUNCHER} ${SYCL_EXECUTABLE}
      ${SYCL_device_link_flags}
      -fsycl-link ${object_files}
      -Xs ${SYCL_OFFLINE_COMPILER_FLAGS}
      -o ${output_file}
      COMMENT "Building SYCL device link file ${output_file_relative_path}"
      )
  endif()
endmacro()

###############################################################################
# ADD LIBRARY
macro(SYCL_ADD_LIBRARY sycl_target)

  if(SYCL_HOST_COMPILATION_CXX)
    set(SYCL_C_OR_CXX CXX)
  else()
    set(SYCL_C_OR_CXX C)
  endif()

  # Separate the sources from the options
  SYCL_GET_SOURCES_AND_OPTIONS(
    _sycl_sources
    _cxx_sources
    _cmake_options
    ${ARGN})

  SYCL_BUILD_SHARED_LIBRARY(_sycl_shared_flag ${ARGN})

  if(_sycl_sources)
    # Compile sycl sources
    SYCL_WRAP_SRCS(
      ${sycl_target}
      ${sycl_target}_sycl_objects
      ${_sycl_shared_flag}
      ${ARGN})

    # Compute the file name of the intermedate link file used for separable
    # compilation.
    SYCL_COMPUTE_DEVICE_OBJECT_FILE_NAME(device_object ${sycl_target})

    # Add a custom device linkage command to produce a host relocatable object
    # containing device object module.
    SYCL_LINK_DEVICE_OBJECTS(
      ${device_object}
      ${sycl_target}
      ${${sycl_target}_sycl_objects})

    add_library(
      ${sycl_target}
      ${_cmake_options}
      ${_cxx_sources}
      ${${sycl_target}_sycl_objects}
      ${device_object})
  else()
    add_library(
      ${sycl_target}
      ${_cmake_options}
      ${_cxx_sources})
  endif()

  target_link_libraries(
    ${sycl_target}
    ${SYCL_LINK_LIBRARIES_KEYWORD}
    ${SYCL_LIBRARY})

  set_target_properties(${sycl_target}
    PROPERTIES
    LINKER_LANGUAGE ${SYCL_C_OR_CXX})

endmacro()

###############################################################################
# ADD EXECUTABLE
macro(SYCL_ADD_EXECUTABLE sycl_target)

  if(SYCL_HOST_COMPILATION_CXX)
    set(SYCL_C_OR_CXX CXX)
  else()
    set(SYCL_C_OR_CXX C)
  endif()

  # Separate the sources from the options
  SYCL_GET_SOURCES_AND_OPTIONS(
    _sycl_sources
    _cxx_sources
    _cmake_options
    ${ARGN})

  if(_sycl_sources)
    # Compile sycl sources
    SYCL_WRAP_SRCS(
      ${sycl_target}
      ${sycl_target}_sycl_objects
      ${ARGN})

    # Compute the file name of the intermedate link file used for separable
    # compilation.
    SYCL_COMPUTE_DEVICE_OBJECT_FILE_NAME(device_object ${sycl_target})

    # Add a custom device linkage command to produce a host relocatable object
    # containing device object module.
    SYCL_LINK_DEVICE_OBJECTS(
      ${device_object}
      ${sycl_target}
      ${${sycl_target}_sycl_objects})

    add_executable(
      ${sycl_target}
      ${_cmake_options}
      ${_cxx_sources}
      ${${sycl_target}_sycl_objects}
      ${device_object})
  else()
    add_executable(
      ${sycl_target}
      ${_cmake_options}
      ${_cxx_sources})
  endif()

  target_link_libraries(
    ${sycl_target}
    ${SYCL_LINK_LIBRARIES_KEYWORD}
    ${SYCL_LIBRARY})

  set_target_properties(${sycl_target}
    PROPERTIES
    LINKER_LANGUAGE ${SYCL_C_OR_CXX})

endmacro()
