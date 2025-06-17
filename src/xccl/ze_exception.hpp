#pragma once

#include <level_zero/ze_api.h>
#include <exception>
#include <unordered_map>
#include <iostream>

// Mapping from status to human readable string
class zeException : std::exception {
  const char * zeResultToString(ze_result_t status) const {
    static const std::unordered_map<ze_result_t, const char *> zeResultToStringMap{
      {ZE_RESULT_SUCCESS, "[Core] success"},
      {ZE_RESULT_NOT_READY, "[Core] synchronization primitive not signaled"},
      {ZE_RESULT_ERROR_DEVICE_LOST, "[Core] device hung, reset, was removed, or driver update occurred"},
      {ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, "[Core] insufficient host memory to satisfy call"},
      {ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY, "[Core] insufficient device memory to satisfy call"},
      {ZE_RESULT_ERROR_MODULE_BUILD_FAILURE, "[Core] error occurred when building module, see build log for details"},
      {ZE_RESULT_ERROR_UNINITIALIZED, "[Validation] driver is not initialized"},
      {ZE_RESULT_ERROR_INVALID_NULL_POINTER, "[Validation] pointer argument may not be nullptr"},
      {ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE, "[Validation] object pointed to by handle still in-use by device"},
      {ZE_RESULT_ERROR_INVALID_ENUMERATION, "[Validation] enumerator argument is not valid"},
      {ZE_RESULT_ERROR_INVALID_SIZE, "[Validation] size argument is invalid"},
      {ZE_RESULT_ERROR_UNSUPPORTED_SIZE, "[Validation] size argument is not supported by the device"},
      {ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT, "[Validation] alignment argument is not supported by the device"},
      {ZE_RESULT_ERROR_INVALID_NULL_HANDLE, "[Validation] handle argument is not valid"},
      {ZE_RESULT_ERROR_UNSUPPORTED_FEATURE, "[Validation] generic error code for unsupported features"},
      {ZE_RESULT_ERROR_INVALID_NATIVE_BINARY, "[Validation] native binary is not supported by the device"},
    };
    auto it = zeResultToStringMap.find(status);
    if (it != zeResultToStringMap.end())
      return it->second;
    else
      return "Unknown Reason";
  }

public:
  zeException(ze_result_t ret) : result_(ret) {}

  ze_result_t result_;

  const char* what() const noexcept override {
    return zeResultToString(result_);
  }
};

#define zeCheck(x)             \
  if (x != ZE_RESULT_SUCCESS)  {    \
    auto e = zeException(x);  \
    std::cout<<"Throw "<<e.what()<<std::endl; \
    throw e;      \
  }
