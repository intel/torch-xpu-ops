#include "sycl/sycl.hpp"
#include <iostream>
#include <unistd.h>


#define MAX_MEMORY_RATIO 0.9

int main() {
  for (auto const &p : sycl::platform::get_platforms()) {
    std::cout << "Found Platform:" << std::endl;
    std::cout << "name: " << p.get_info<sycl::info::platform::name>()
              << std::endl;
    std::cout << "vendor: " << p.get_info<sycl::info::platform::vendor>()
              << std::endl;
    std::cout << "version: " << p.get_info<sycl::info::platform::version>()
              << std::endl;
    std::cout << "profile: " << p.get_info<sycl::info::platform::profile>()
              << std::endl;

    std::cout << std::endl;

    if (p.get_backend() != sycl::backend::ext_oneapi_level_zero) {
        continue;
    }
    for (const auto &d : p.get_devices()) {
      if (!d.is_gpu()) {
        continue;
      }
      auto mem_size = d.get_info<sycl::info::device::global_mem_size>();
      sycl::queue q(d);
      mem_size = mem_size * MAX_MEMORY_RATIO;
      auto ptr = sycl::aligned_alloc_device(512, mem_size, d, q.get_context());
      if (ptr == nullptr) {
         std::cout << "[Error] Alloc GPU device " << mem_size << " Bytes failed." << std::endl;
      } else {
         std::cout << "[Info] Alloc GPU device " << mem_size << " Bytes pass." << std::endl;
      }
      sycl::free(ptr, q.get_context());
      q.wait();

      // test and free
      auto size_test = mem_size / 4;
      for (int i = 1; i < 100; i++) {
          std::cout << "[Info] Start to do allocate and free in round: " << i << std::endl;
          auto ptr1 = sycl::aligned_alloc_device(512, size_test, d, q.get_context());
          auto ptr2 = sycl::aligned_alloc_device(512, size_test, d, q.get_context());
          auto ptr3 = sycl::aligned_alloc_device(512, size_test, d, q.get_context());
          auto ptr4 = sycl::aligned_alloc_device(512, size_test, d, q.get_context());
          if (ptr1 == nullptr || ptr2 == nullptr || ptr3 == nullptr || ptr4 == nullptr) {
              std::cout << "[Error] Allocation fail - allocated ptr: " << ptr4 << " : "<< ptr3 << " : " << ptr2 << " : " << ptr1 << std::endl;
          } else {
              std::cout << "[Info] Allocation pass " << std::endl;
              // add a sycl kernel tests
              q.memcpy(ptr1, ptr2, size_test);
              q.memcpy(ptr3, ptr4, size_test);
          }
          q.wait();
          sycl::free(ptr1, q.get_context());
          sycl::free(ptr2, q.get_context());
          sycl::free(ptr3, q.get_context());
          sycl::free(ptr4, q.get_context());
          std::cout << "[Info] Finish to do allocate and free in round: " << i << std::endl;
      }
    }

    sleep(1);
    std::cout << std::endl;
    std::cout << std::endl;

  }
  return 0;
}
