
#include "sycl/sycl.hpp"
#include <iostream>
#include <unistd.h>

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

    for (const auto &d : p.get_devices()) {
      if (!d.is_gpu()) {
        continue;
      }

      std::cout << "GPU Device info:" << std::endl;
      std::cout << "name: " << d.get_info<sycl::info::device::name>()
                << std::endl;
      std::cout << "global_mem_size: "
                << d.get_info<sycl::info::device::global_mem_size>()
                << std::endl;
       auto max_sub_devices = d.get_info<sycl::info::device::partition_max_sub_devices>();
       if (max_sub_devices > 0) {
         constexpr auto partition_by_affinity = sycl::info::partition_property::partition_by_affinity_domain;
         constexpr auto next_partitionable = sycl::info::partition_affinity_domain::next_partitionable;
         auto sub_devices = d.create_sub_devices<partition_by_affinity>(next_partitionable);

         for (auto sd : sub_devices) {
               std::cout << "global_mem_size per tile: "
                 << sd.get_info<sycl::info::device::global_mem_size>()
                 << std::endl;
           auto mem_size = sd.get_info<sycl::info::device::global_mem_size>();
           std::cout << "Alloc " << mem_size << " Bytes failed." << std::endl;

//           sycl::queue q(sd);
//           mem_size -= 1800 * 1024 * 1024;
//           auto ptr = sycl::aligned_alloc_device(64, mem_size, q);
//           if (ptr == nullptr) {
//                 std::cout << "Alloc " << mem_size << " Bytes failed." << std::endl;
//           }
         }
       }
    }

    sleep(1);
    std::cout << std::endl;
    std::cout << std::endl;

  }
  return 0;
}
