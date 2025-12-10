# Copyright 2020-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

## Include to trigger clang-format
if(BUILD_NO_CLANGFORMAT)
  return()
endif()

if(CLANGFORMAT_enabled)
  return()
endif()
set(CLANGFORMAT_enabled true)

set(CFMT_STYLE ${PROJECT_SOURCE_DIR}/.clang-format)
if(NOT EXISTS ${CFMT_STYLE})
  message(WARNING "Cannot find style file ${CFMT_STYLE}!")
  return()
endif()

find_program(CLANG_FORMAT "clang-format-12")
if(NOT CLANG_FORMAT)
  message("Please install clang-format-12 before contributing to torch-xpu-ops!")
else()
  set(CLANG_FORMAT_EXEC clang-format-12)
endif()
