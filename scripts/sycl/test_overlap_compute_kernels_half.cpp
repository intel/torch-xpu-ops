#include <CL/sycl.hpp>
#include <vector>
#include <iostream>
#include <typeinfo>
#include <ext/intel/esimd.hpp>
using namespace std;
using namespace sycl;

#define SIMD 128

int main(int argc, char *argv[]) {
    const size_t count = 64 * 512;
    const size_t count2 = 50* 1024 * 1024;

    sycl::property_list propList{cl::sycl::property::queue::in_order()};
    sycl::queue queue1(sycl::gpu_selector{}, propList);
    sycl::queue queue2(sycl::gpu_selector{}, propList);

    /* create buffers */
    auto compute_buf1 = sycl::malloc_device<float>(count2, queue1);
    auto compute_buf2 = sycl::malloc_device<float>(count, queue1);

    const int wg_size = queue1.get_device().get_info<sycl::info::device::max_work_group_size>();
    const int total_wg_size = wg_size/2;
    const int num_wg = 64;
    int32_t max = 64 * 512;
    int32_t max2 = 50 * 1024 * 1024;

    const int num_wg2 = (count2 + wg_size - 1) / wg_size;

    std::cout << " work groups size max: " << wg_size << std::endl;

    for (int i = 0; i < 10; i ++) {
       for (int j = 0; j < 2; j ++) {
          queue2.submit([&](sycl::handler& cgh) {
                cgh.parallel_for<class queue2_compute_kernel>(sycl::nd_range<1>(num_wg * total_wg_size, total_wg_size),
                    [=](sycl::nd_item<1> item) {
                    int id = item.get_global_id();
                    if (id >= max) return;
                    compute_buf2[id] += id * 0.00001;
                    for (int j = 0; j < 5000; j++) {
                      compute_buf2[id] += j * 0.0001;
                    }
            });
           });

          queue1.submit([&](sycl::handler &cgh) {
            cgh.parallel_for<class queue1_compute_kernel>(sycl::nd_range<1>(num_wg2 * wg_size, wg_size),
                    [=](sycl::nd_item<1> item) {
                    int id = item.get_global_id();
                    if (id >= max2) return;
                    compute_buf1[id] += id * 0.00001;
                    for (int j = 0; j < 50; j++) {
                      compute_buf1[id] += j * 0.0001;
                    }
            });
          });
          queue2.ext_oneapi_submit_barrier({queue1.ext_oneapi_submit_barrier()});
       }
       printf("finished iteration %d", i);
       queue1.wait();
       queue2.wait();
    }

    printf("start to execute q2 sync kernels");
     for (int i = 0; i < 10; i ++) {
       for (int j = 0; j < 2; j ++) {
          queue1.submit([&](sycl::handler &cgh) {
            cgh.parallel_for<class queue1_compute_kernel2>(sycl::nd_range<1>(num_wg * total_wg_size, total_wg_size),
                    [=](sycl::nd_item<1> item) {
                    int id = item.get_global_id();
                    if (id >= max) return;
                    compute_buf1[id] += id * 0.00001;
                    for (int j = 0; j < 5000; j++) {
                      compute_buf1[id] += j * 0.0001;
                    }
            });
          });
       }
       queue1.wait();
    }

    printf("finished all");
    return 0;
}
