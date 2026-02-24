#include <CL/sycl.hpp>
#include <vector>
#include <iostream>
#include <typeinfo>
#include <ext/intel/esimd.hpp>
using namespace std;
using namespace sycl;

int main(int argc, char *argv[]) {
    const size_t count = 50 * 1024 * 1024;

    sycl::property_list propList1{cl::sycl::property::queue::in_order(), cl::sycl::ext::oneapi::property::queue::priority_low{}};
    sycl::property_list propList2{cl::sycl::property::queue::in_order(), cl::sycl::ext::oneapi::property::queue::priority_normal{}};
    sycl::queue queue1(sycl::gpu_selector{}, propList1);
    sycl::queue queue2(sycl::gpu_selector{}, propList2);

    /* create buffers */
    auto compute_buf1 = sycl::malloc_device<float>(count, queue1);
    auto compute_buf2 = sycl::malloc_device<float>(count, queue1);
    auto compute_buf3 = sycl::malloc_device<float>(count, queue1);

    int32_t max = 50 * 1024 * 1024;
    const int wg_size = queue1.get_device().get_info<sycl::info::device::max_work_group_size>();
    const int num_wg = (count + wg_size - 1) / wg_size;

    for (int i = 0; i < 10; i ++) {
       for (int j = 0; j < 2; j ++) {
          queue1.submit([&](sycl::handler &cgh) {
            cgh.parallel_for<class queue1_compute_kernel>(sycl::nd_range<1>(num_wg * wg_size, wg_size),
                    [=](sycl::nd_item<1> item) {
                    int id = item.get_global_id();
                    if (id >= max) return;
                    compute_buf1[id] += id * 0.00001;
                    for (int j = 0; j < 100; j++) {
                      compute_buf1[id] += j * 0.0001;
                    }
            });
          });

          queue1.submit([&](sycl::handler &cgh) {
            cgh.parallel_for<class queue1_compute_kernel2>(sycl::nd_range<1>(num_wg * wg_size, wg_size),
                    [=](sycl::nd_item<1> item) {
                    int id = item.get_global_id();
                    if (id >= max) return;
                    compute_buf1[id] += id * 0.00001;
                    for (int j = 0; j < 100; j++) {
                      compute_buf2[id] += j * 0.0001;
                    }
            });
          });


          queue1.submit([&](sycl::handler &cgh) {
            cgh.parallel_for<class queue1_compute_kernel3>(sycl::nd_range<1>(num_wg * wg_size, wg_size),
                    [=](sycl::nd_item<1> item) {
                    int id = item.get_global_id();
                    if (id >= max) return;
                    compute_buf1[id] += id * 0.00001;
                    for (int j = 0; j < 100; j++) {
                      compute_buf2[id] += j * 0.0001;
                    }
            });
          });

          queue1.submit([&](sycl::handler &cgh) {
            cgh.parallel_for<class queue1_compute_kernel4>(sycl::nd_range<1>(num_wg * wg_size, wg_size),
                    [=](sycl::nd_item<1> item) {
                    int id = item.get_global_id();
                    if (id >= max) return;
                    compute_buf1[id] += id * 0.00001;
                    for (int j = 0; j < 100; j++) {
                      compute_buf2[id] += j * 0.0001;
                    }
            });
          });

          queue1.submit([&](sycl::handler &cgh) {
            cgh.parallel_for<class queue1_compute_kernel5>(sycl::nd_range<1>(num_wg * wg_size, wg_size),
                    [=](sycl::nd_item<1> item) {
                    int id = item.get_global_id();
                    if (id >= max) return;
                    compute_buf1[id] += id * 0.00001;
                    for (int j = 0; j < 100; j++) {
                      compute_buf2[id] += j * 0.0001;
                    }
            });
          });

          queue2.submit([&](sycl::handler& cgh) {
                cgh.parallel_for<class queue2_compute_kernel>(sycl::nd_range<1>(num_wg * wg_size, wg_size),
                    [=](sycl::nd_item<1> item) {
                    int id = item.get_global_id();
                    if (id >= max) return;
                    compute_buf2[id] += id * 0.00001;
                    for (int j = 0; j < 100; j++) {
                      compute_buf3[id] += j * 0.0001;
                    }
            });
           });

          queue1.submit([&](sycl::handler &cgh) {
            cgh.parallel_for<class queue1_compute_kernel6>(sycl::nd_range<1>(num_wg * wg_size, wg_size),
                    [=](sycl::nd_item<1> item) {
                    int id = item.get_global_id();
                    if (id >= max) return;
                    compute_buf1[id] += id * 0.00001;
                    for (int j = 0; j < 100; j++) {
                      compute_buf2[id] += j * 0.0001;
                    }
            });
          });

          queue1.submit([&](sycl::handler &cgh) {
            cgh.parallel_for<class queue1_compute_kernel7>(sycl::nd_range<1>(num_wg * wg_size, wg_size),
                    [=](sycl::nd_item<1> item) {
                    int id = item.get_global_id();
                    if (id >= max) return;
                    compute_buf1[id] += id * 0.00001;
                    for (int j = 0; j < 100; j++) {
                      compute_buf2[id] += j * 0.0001;
                    }
            });
          });

          queue1.submit([&](sycl::handler &cgh) {
            cgh.parallel_for<class queue1_compute_kernel8>(sycl::nd_range<1>(num_wg * wg_size, wg_size),
                    [=](sycl::nd_item<1> item) {
                    int id = item.get_global_id();
                    if (id >= max) return;
                    compute_buf1[id] += id * 0.00001;
                    for (int j = 0; j < 100; j++) {
                      compute_buf2[id] += j * 0.0001;
                    }
            });
          });

          queue1.submit([&](sycl::handler &cgh) {
            cgh.parallel_for<class queue1_compute_kernel9>(sycl::nd_range<1>(num_wg * wg_size, wg_size),
                    [=](sycl::nd_item<1> item) {
                    int id = item.get_global_id();
                    if (id >= max) return;
                    compute_buf1[id] += id * 0.00001;
                    for (int j = 0; j < 100; j++) {
                      compute_buf2[id] += j * 0.0001;
                    }
            });
          });


          queue1.submit([&](sycl::handler &cgh) {
            cgh.parallel_for<class queue1_compute_kernel10>(sycl::nd_range<1>(num_wg * wg_size, wg_size),
                    [=](sycl::nd_item<1> item) {
                    int id = item.get_global_id();
                    if (id >= max) return;
                    compute_buf1[id] += id * 0.00001;
                    for (int j = 0; j < 100; j++) {
                      compute_buf2[id] += j * 0.0001;
                    }
            });
          });

          queue1.ext_oneapi_submit_barrier({queue2.ext_oneapi_submit_barrier()});
       }
       printf("finished iteration %d", i);
       queue1.wait();
       queue2.wait();
    }

    printf("finished all");
    return 0;
}
