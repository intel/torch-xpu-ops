#pragma once

#include <dlfcn.h>
#include <exception>
#include <iostream>
#include <unordered_map>

#define zeVirtualMemMap zeVirtualMemMap_original
#define zeVirtualMemReserve zeVirtualMemReserve_original
#define zeVirtualMemSetAccessAttribute zeVirtualMemSetAccessAttribute_original

#include <level_zero/ze_api.h>

#undef zeVirtualMemMap
#undef zeVirtualMemReserve
#undef zeVirtualMemSetAccessAttribute

typedef ze_result_t (*zeInit_t)(ze_init_flags_t flags);
typedef ze_result_t (*zeMemGetAddressRange_t)(
    ze_context_handle_t hContext,
    const void* ptr,
    void** pBase,
    size_t* pSize);
typedef ze_result_t (*zeMemGetIpcHandle_t)(
    ze_context_handle_t hContext,
    const void* ptr,
    ze_ipc_mem_handle_t* pIpcHandle);
typedef ze_result_t (*zeMemOpenIpcHandle_t)(
    ze_context_handle_t hContext,
    ze_device_handle_t hDevice,
    ze_ipc_mem_handle_t handle,
    ze_ipc_memory_flags_t flags,
    void** pptr);
typedef ze_result_t (
    *zeMemCloseIpcHandle_t)(ze_context_handle_t hContext, const void* ptr);
typedef ze_result_t (*zeVirtualMemMap_t)(
    ze_context_handle_t hContext,
    const void* ptr,
    size_t size,
    ze_physical_mem_handle_t hPhysicalMemory,
    size_t offset,
    ze_memory_access_attribute_t access);
typedef ze_result_t (*zeVirtualMemReserve_t)(
    ze_context_handle_t hContext,
    const void* pStart,
    size_t size,
    void** pptr);
typedef ze_result_t (*zeVirtualMemSetAccessAttribute_t)(
    ze_context_handle_t hContext,
    const void* ptr,
    size_t size,
    ze_memory_access_attribute_t access);

bool load_level_zero_library();
void unload_level_zero_library();

#define zeCheck_dynamic(x)                                          \
  do {                                                              \
    if (!load_level_zero_library()) {                               \
      throw std::runtime_error("Level Zero library not available"); \
    }                                                               \
    ze_result_t result = (x);                                       \
    if (result != ZE_RESULT_SUCCESS) {                              \
      auto e = zeException(result);                                 \
      std::cout << "Throw " << e.what() << std::endl;               \
      throw e;                                                      \
    }                                                               \
  } while (0)

#define zeInit_dynamic(flags) zeInit_ptr(flags)
#define zeMemGetAddressRange_dynamic(ctx, ptr, base, size) \
  zeMemGetAddressRange_ptr(ctx, ptr, base, size)
#define zeMemGetIpcHandle_dynamic(ctx, ptr, handle) \
  zeMemGetIpcHandle_ptr(ctx, ptr, handle)
#define zeMemOpenIpcHandle_dynamic(ctx, dev, handle, flags, ptr) \
  zeMemOpenIpcHandle_ptr(ctx, dev, handle, flags, ptr)
#define zeMemCloseIpcHandle_dynamic(ctx, ptr) zeMemCloseIpcHandle_ptr(ctx, ptr)
#define zeVirtualMemMap_dynamic(ctx, ptr, size, phys_mem, offset, access) \
  zeVirtualMemMap_ptr(ctx, ptr, size, phys_mem, offset, access)
#define zeVirtualMemReserve_dynamic(ctx, start, size, ptr) \
  zeVirtualMemReserve_ptr(ctx, start, size, ptr)
#define zeVirtualMemSetAccessAttribute_dynamic(ctx, ptr, size, access) \
  zeVirtualMemSetAccessAttribute_ptr(ctx, ptr, size, access)

// Exception handling class
class zeException : std::exception {
  const char* zeResultToString(ze_result_t status) const {
    static const std::unordered_map<ze_result_t, const char*> zeResultToStringMap{
        {ZE_RESULT_SUCCESS, "[Core] success"},
        {ZE_RESULT_NOT_READY, "[Core] synchronization primitive not signaled"},
        {ZE_RESULT_ERROR_UNINITIALIZED,
         "[Validation] driver is not initialized"},
        {ZE_RESULT_ERROR_INVALID_NULL_POINTER,
         "[Validation] pointer argument may not be nullptr"},
        {ZE_RESULT_ERROR_INVALID_NULL_HANDLE,
         "[Validation] handle argument is not valid"},
        {ZE_RESULT_ERROR_INVALID_ENUMERATION,
         "[Validation] enumerator argument is not valid"},
        {ZE_RESULT_ERROR_INVALID_SIZE, "[Validation] size argument is invalid"},
        {ZE_RESULT_ERROR_UNSUPPORTED_SIZE,
         "[Validation] size argument is not supported by the device"},
        {ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT,
         "[Validation] alignment argument is not supported by the device"},
        {ZE_RESULT_ERROR_UNSUPPORTED_FEATURE,
         "[Validation] generic error code for unsupported features"},
        {ZE_RESULT_ERROR_INVALID_NATIVE_BINARY,
         "[Validation] native binary is not supported by the device"},
        {ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY,
         "[Core] insufficient host memory to satisfy call"},
        {ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY,
         "[Core] insufficient device memory to satisfy call"},
        {ZE_RESULT_ERROR_DEVICE_LOST,
         "[Core] device hung, reset, was removed, or driver update occurred"},
        {ZE_RESULT_ERROR_MODULE_BUILD_FAILURE,
         "[Core] error occurred when building module, see build log for details"},
        {ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE,
         "[Validation] object pointed to by handle still in-use by device"},
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

#define zeCheck(x)                                  \
  if (x != ZE_RESULT_SUCCESS) {                     \
    auto e = zeException(x);                        \
    std::cout << "Throw " << e.what() << std::endl; \
    throw e;                                        \
  }

static zeInit_t zeInit_ptr = nullptr;
static zeMemGetAddressRange_t zeMemGetAddressRange_ptr = nullptr;
static zeMemGetIpcHandle_t zeMemGetIpcHandle_ptr = nullptr;
static zeMemOpenIpcHandle_t zeMemOpenIpcHandle_ptr = nullptr;
static zeMemCloseIpcHandle_t zeMemCloseIpcHandle_ptr = nullptr;
static zeVirtualMemMap_t zeVirtualMemMap_ptr = nullptr;
static zeVirtualMemReserve_t zeVirtualMemReserve_ptr = nullptr;
static zeVirtualMemSetAccessAttribute_t zeVirtualMemSetAccessAttribute_ptr =
    nullptr;

static void* ze_handle = nullptr;

inline bool load_level_zero_library() {
  if (ze_handle != nullptr) {
    return true;
  }
  const char* lib_names[] = {"/usr/lib/x86_64-linux-gnu/libze_loader.so"};

  for (const char* lib_name : lib_names) {
    ze_handle = dlopen(lib_name, RTLD_LAZY);
    if (ze_handle != nullptr) {
      break;
    }
  }

  if (ze_handle == nullptr) {
    std::cerr << "Failed to load Level Zero library: " << dlerror()
              << std::endl;
    return false;
  }

  zeInit_ptr = (zeInit_t)dlsym(ze_handle, "zeInit");
  zeMemGetAddressRange_ptr =
      (zeMemGetAddressRange_t)dlsym(ze_handle, "zeMemGetAddressRange");
  zeMemGetIpcHandle_ptr =
      (zeMemGetIpcHandle_t)dlsym(ze_handle, "zeMemGetIpcHandle");
  zeMemOpenIpcHandle_ptr =
      (zeMemOpenIpcHandle_t)dlsym(ze_handle, "zeMemOpenIpcHandle");
  zeMemCloseIpcHandle_ptr =
      (zeMemCloseIpcHandle_t)dlsym(ze_handle, "zeMemCloseIpcHandle");
  zeVirtualMemMap_ptr = (zeVirtualMemMap_t)dlsym(ze_handle, "zeVirtualMemMap");
  zeVirtualMemReserve_ptr =
      (zeVirtualMemReserve_t)dlsym(ze_handle, "zeVirtualMemReserve");
  zeVirtualMemSetAccessAttribute_ptr = (zeVirtualMemSetAccessAttribute_t)dlsym(
      ze_handle, "zeVirtualMemSetAccessAttribute");

  if (!zeInit_ptr || !zeMemGetAddressRange_ptr || !zeMemGetIpcHandle_ptr ||
      !zeMemOpenIpcHandle_ptr || !zeMemCloseIpcHandle_ptr ||
      !zeVirtualMemMap_ptr || !zeVirtualMemReserve_ptr ||
      !zeVirtualMemSetAccessAttribute_ptr) {
    std::cerr << "Failed to load Level Zero API functions" << std::endl;
    dlclose(ze_handle);
    ze_handle = nullptr;
    return false;
  }

  return true;
}

inline void unload_level_zero_library() {
  if (ze_handle != nullptr) {
    dlclose(ze_handle);
    ze_handle = nullptr;
    zeInit_ptr = nullptr;
    zeMemGetAddressRange_ptr = nullptr;
    zeMemGetIpcHandle_ptr = nullptr;
    zeMemOpenIpcHandle_ptr = nullptr;
    zeMemCloseIpcHandle_ptr = nullptr;
    zeVirtualMemMap_ptr = nullptr;
    zeVirtualMemReserve_ptr = nullptr;
    zeVirtualMemSetAccessAttribute_ptr = nullptr;
  }
}

extern "C" {

__attribute__((weak)) ze_result_t zeVirtualMemMap(
    ze_context_handle_t hContext,
    const void* ptr,
    size_t size,
    ze_physical_mem_handle_t hPhysicalMemory,
    size_t offset,
    ze_memory_access_attribute_t access) {
  if (!load_level_zero_library() || !zeVirtualMemMap_ptr) {
    return ZE_RESULT_ERROR_UNINITIALIZED;
  }
  return zeVirtualMemMap_ptr(
      hContext, ptr, size, hPhysicalMemory, offset, access);
}

__attribute__((weak)) ze_result_t zeVirtualMemReserve(
    ze_context_handle_t hContext,
    const void* pStart,
    size_t size,
    void** pptr) {
  if (!load_level_zero_library() || !zeVirtualMemReserve_ptr) {
    return ZE_RESULT_ERROR_UNINITIALIZED;
  }
  return zeVirtualMemReserve_ptr(hContext, pStart, size, pptr);
}

__attribute__((weak)) ze_result_t zeVirtualMemSetAccessAttribute(
    ze_context_handle_t hContext,
    const void* ptr,
    size_t size,
    ze_memory_access_attribute_t access) {
  if (!load_level_zero_library() || !zeVirtualMemSetAccessAttribute_ptr) {
    return ZE_RESULT_ERROR_UNINITIALIZED;
  }
  return zeVirtualMemSetAccessAttribute_ptr(hContext, ptr, size, access);
}
}
