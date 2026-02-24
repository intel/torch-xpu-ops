#include <sycl/sycl.hpp>
#include <thread>
#include <chrono>
#include <getopt.h>
#define BUFFER_SIZE 4096
std::vector<unsigned char*> dev_bufs;

std::vector<sycl::device> get_sycl_devices(){
    std::vector<sycl::device> devices;
    auto platform_list = sycl::platform::get_platforms();
    for (const auto& platform : platform_list) {
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
                if (max_sub_devices == 0) devices.push_back(device);
                else {
                    auto sub_devices = device.create_sub_devices<partition_by_affinity>(next_partitionable);
                    devices.insert(devices.end(), sub_devices.begin(), sub_devices.end());
                }
            }
        }
    }
    return devices;
}

std::vector<sycl::queue> get_sycl_queues(std::vector<sycl::device>& devices) {
    std::vector<sycl::queue> queues;
    sycl::property_list propList{sycl::property::queue::in_order()};
    sycl::context ctx(devices);
    for (auto& device : devices)
        queues.emplace_back(ctx, device, propList);
    return queues;
}

void verify_kernel_parallel(std::vector<sycl::queue> queues) {
    std::vector<std::thread> threads;
    for (size_t id = 0; id < queues.size(); ++id) {
        threads.emplace_back([](sycl::queue queue, unsigned char* dev_ptr, int id){
            auto begin = std::chrono::steady_clock::now();
            for (int i = 0; i < 1001; ++i) {
                queue.submit([=](sycl::handler& cgh){
                    cgh.parallel_for(sycl::nd_range<1>(1024, 1024),
                    [dev_ptr] (sycl::nd_item<1> item) {
                        size_t id = item.get_global_linear_id();
                        for (int i = id; i < BUFFER_SIZE; i+=1024){
                            dev_ptr[id] += dev_ptr[i]; 
                        }
                    });
                });

                if (i==0) queue.wait(); // skip first one because of long preparation time
                if (i==1) begin = std::chrono::steady_clock::now();
            }
            queue.wait();
            auto end = std::chrono::steady_clock::now();
            auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
            std::cout<< "Device: " << id << " uses " << dur << " us for 1000 kernels." << std::endl;
        }, queues[id], dev_bufs[id], id);
    }

    for (size_t id = 0; id < queues.size(); ++id) {
        threads[id].join();
    }
}

int main(int argc, char *argv[]){
    std::vector<sycl::device> devices = get_sycl_devices();
    std::vector<sycl::queue> queues = get_sycl_queues(devices);
    for (auto& queue : queues) {
        dev_bufs.push_back(sycl::malloc_device<unsigned char>(BUFFER_SIZE, queue));
    }  
    verify_kernel_parallel(queues);
}

