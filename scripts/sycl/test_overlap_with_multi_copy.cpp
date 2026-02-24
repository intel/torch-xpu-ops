#include <CL/sycl.hpp>
#include <vector>
#include <iostream>
#include <typeinfo>
using namespace std;
using namespace sycl;

int main(int argc, char *argv[]) {
    const size_t count = 50 * 1024 * 1024;
    const size_t count_copy = 1024 * 1024 * 1024 * 2;

    sycl::property_list propList{cl::sycl::property::queue::in_order()};
    sycl::queue queue1(sycl::gpu_selector{}, propList);
    sycl::queue queue2(sycl::gpu_selector{}, propList);

    /* create buffers */
    auto compute_buf1 = sycl::malloc_device<float>(count, queue1);
    auto compute_buf2 = sycl::malloc_device<float>(count_copy, queue1);
    auto recv_bufs1 = sycl::malloc_device<float>(count, queue1);
    auto recv_bufs2 = sycl::malloc_device<float>(count_copy, queue1);

    auto num = count;
    int32_t max = 50 * 1024 * 1024;
    const int wg_size = queue1.get_device().get_info<sycl::info::device::max_work_group_size>();
    const int num_wg = (num + wg_size - 1) / wg_size;

    for (int i = 0; i < 10; i ++) {
        for (int j = 0; j < 1; j ++) {
          auto event1 = queue1.submit([&](sycl::handler &cgh) {
            cgh.parallel_for<class compute_kernel1>(sycl::nd_range<1>(num_wg * wg_size, wg_size),
                    [=](sycl::nd_item<1> item) {
                    int id = item.get_global_id();
                    if (id >= max) return;
                    compute_buf1[id] += id * 0.00001;
                    for (int j = 0; j < 50; j++) {
                      compute_buf1[id] += j * 0.0001;
                    }
            });
          });

          auto event2 = queue1.submit([&](sycl::handler &cgh) {
            cgh.parallel_for<class compute_kernel2>(sycl::nd_range<1>(num_wg * wg_size, wg_size),
                    [=](sycl::nd_item<1> item) {
                    int id = item.get_global_id();
                    if (id >= max) return;
                    compute_buf1[id] += id * 0.00001;
                    for (int j = 0; j < 50; j++) {
                      compute_buf1[id] += j * 0.0001;
                    }
            });
          });

          auto event3 = queue1.submit([&](sycl::handler &cgh) {
            cgh.parallel_for<class compute_kernel3>(sycl::nd_range<1>(num_wg * wg_size, wg_size),
                    [=](sycl::nd_item<1> item) {
                    int id = item.get_global_id();
                    if (id >= max) return;
                    compute_buf1[id] += id * 0.00001;
                    for (int j = 0; j < 50; j++) {
                      compute_buf1[id] += j * 0.0001;
                    }
            });
          });

          auto event4 = queue1.submit([&](sycl::handler &cgh) {
            cgh.parallel_for<class compute_kernel4>(sycl::nd_range<1>(num_wg * wg_size, wg_size),
                    [=](sycl::nd_item<1> item) {
                    int id = item.get_global_id();
                    if (id >= max) return;
                    compute_buf1[id] += id * 0.00001;
                    for (int j = 0; j < 50; j++) {
                      compute_buf1[id] += j * 0.0001;
                    }
            });
          });

          queue2.memcpy(compute_buf2, recv_bufs2, count_copy); // allgather

          auto event5 = queue1.submit([&](sycl::handler &cgh) {
            cgh.parallel_for<class compute_kernel5>(sycl::nd_range<1>(num_wg * wg_size, wg_size),
                    [=](sycl::nd_item<1> item) {
                    int id = item.get_global_id();
                    if (id >= max) return;
                    compute_buf1[id] += id * 0.00001;
                    for (int j = 0; j < 50; j++) {
                      compute_buf1[id] += j * 0.0001;
                    }
            });
          });

          auto event6 = queue1.submit([&](sycl::handler &cgh) {
            cgh.parallel_for<class compute_kernel6>(sycl::nd_range<1>(num_wg * wg_size, wg_size),
                    [=](sycl::nd_item<1> item) {
                    int id = item.get_global_id();
                    if (id >= max) return;
                    compute_buf1[id] += id * 0.00001;
                    for (int j = 0; j < 50; j++) {
                      compute_buf1[id] += j * 0.0001;
                    }
            });
          });


          auto event7 = queue1.submit([&](sycl::handler &cgh) {
            cgh.parallel_for<class compute_kernel7>(sycl::nd_range<1>(num_wg * wg_size, wg_size),
                    [=](sycl::nd_item<1> item) {
                    int id = item.get_global_id();
                    if (id >= max) return;
                    compute_buf1[id] += id * 0.00001;
                    for (int j = 0; j < 50; j++) {
                      compute_buf1[id] += j * 0.0001;
                    }
            });
          });

          auto event8 = queue1.submit([&](sycl::handler &cgh) {
            cgh.parallel_for<class compute_kernel8>(sycl::nd_range<1>(num_wg * wg_size, wg_size),
                    [=](sycl::nd_item<1> item) {
                    int id = item.get_global_id();
                    if (id >= max) return;
                    compute_buf1[id] += id * 0.00001;
                    for (int j = 0; j < 50; j++) {
                      compute_buf1[id] += j * 0.0001;
                    }
            });
          });

          auto event9 = queue1.submit([&](sycl::handler &cgh) {
            cgh.parallel_for<class compute_kernel9>(sycl::nd_range<1>(num_wg * wg_size, wg_size),
                    [=](sycl::nd_item<1> item) {
                    int id = item.get_global_id();
                    if (id >= max) return;
                    compute_buf1[id] += id * 0.00001;
                    for (int j = 0; j < 50; j++) {
                      compute_buf1[id] += j * 0.0001;
                    }
            });
          });

          auto event10 = queue1.submit([&](sycl::handler &cgh) {
            cgh.parallel_for<class compute_kernel10>(sycl::nd_range<1>(num_wg * wg_size, wg_size),
                    [=](sycl::nd_item<1> item) {
                    int id = item.get_global_id();
                    if (id >= max) return;
                    compute_buf1[id] += id * 0.00001;
                    for (int j = 0; j < 50; j++) {
                      compute_buf1[id] += j * 0.0001;
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
