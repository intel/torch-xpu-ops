/**
 * Test case for P2P atomic support query between SYCL devices.
 *
 * This test demonstrates the usage of sycl_ext_oneapi_peer_access extension
 * to query P2P atomic support between devices.
 *
 * Key Technical Points:
 * - ext_oneapi_can_access_peer with peer_access::atomics_supported query
 * - P2P atomics require memory_scope::system scope
 * - Only CUDA, HIP and Level Zero backends support P2P access
 *
 * Build: icpx -fsycl -o test_p2p_atomic test_p2p_atomic.cpp
 * Run: ./test_p2p_atomic
 */

#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>

std::vector<sycl::device> get_gpu_devices() {
    std::vector<sycl::device> devices;
    auto platform_list = sycl::platform::get_platforms();
    for (const auto& platform : platform_list) {
        // Filter Level Zero backend for Intel GPUs
        bool is_level_zero = platform.get_backend() == sycl::backend::ext_oneapi_level_zero;
        if (!is_level_zero) continue;

        auto device_list = platform.get_devices();
        constexpr auto partition_by_affinity =
            sycl::info::partition_property::partition_by_affinity_domain;
        constexpr auto next_partitionable =
            sycl::info::partition_affinity_domain::next_partitionable;

        for (auto& device : device_list) {
            if (device.is_gpu()) {
                auto max_sub_devices = device.get_info<sycl::info::device::partition_max_sub_devices>();
                if (max_sub_devices == 0) {
                    devices.push_back(device);
                } else {
                    auto sub_devices = device.create_sub_devices<partition_by_affinity>(next_partitionable);
                    devices.insert(devices.end(), sub_devices.begin(), sub_devices.end());
                }
            }
        }
    }
    return devices;
}

void print_device_info(const sycl::device& dev, int index) {
    std::cout << "  Device " << index << ": "
              << dev.get_info<sycl::info::device::name>() << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "=== P2P Atomic Support Test ===" << std::endl;

    std::vector<sycl::device> devices = get_gpu_devices();

    std::cout << "\nFound " << devices.size() << " GPU device(s)/tile(s):" << std::endl;
    for (size_t i = 0; i < devices.size(); ++i) {
        print_device_info(devices[i], i);
    }

    if (devices.size() < 2) {
        std::cout << "\nWARNING: Need at least 2 GPU devices/tiles to test P2P capabilities."
                  << std::endl;
        std::cout << "P2P atomic test skipped." << std::endl;
        return 0;
    }

    std::cout << "\n=== P2P Access Support Matrix ===" << std::endl;

    // Test P2P access support between all device pairs
    std::cout << "\nP2P Access Support (access_supported):" << std::endl;
    for (size_t i = 0; i < devices.size(); ++i) {
        for (size_t j = 0; j < devices.size(); ++j) {
            if (i == j) continue;

            bool access_supported = devices[i].ext_oneapi_can_access_peer(
                devices[j], sycl::ext::oneapi::peer_access::access_supported);

            std::cout << "  Device " << i << " -> Device " << j << ": "
                      << (access_supported ? "SUPPORTED" : "NOT SUPPORTED") << std::endl;
        }
    }

    // Test P2P atomic support between all device pairs
    std::cout << "\nP2P Atomic Support (atomics_supported):" << std::endl;
    int atomic_support_count = 0;
    for (size_t i = 0; i < devices.size(); ++i) {
        for (size_t j = 0; j < devices.size(); ++j) {
            if (i == j) continue;

            bool atomics_supported = devices[i].ext_oneapi_can_access_peer(
                devices[j], sycl::ext::oneapi::peer_access::atomics_supported);

            std::cout << "  Device " << i << " -> Device " << j << ": "
                      << (atomics_supported ? "SUPPORTED" : "NOT SUPPORTED") << std::endl;

            if (atomics_supported) {
                atomic_support_count++;
            }
        }
    }

    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "Total device pairs checked: " << devices.size() * (devices.size() - 1) << std::endl;
    std::cout << "P2P atomic support pairs: " << atomic_support_count << std::endl;

    if (atomic_support_count > 0) {
        std::cout << "\nNOTE: When P2P atomics are supported, atomic operations on peer device's"
                  << std::endl;
        std::cout << "      memory must use memory_scope::system scope." << std::endl;
    }

    return 0;
}

