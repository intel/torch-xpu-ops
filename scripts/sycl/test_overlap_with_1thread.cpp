#include <CL/sycl.hpp>
#include <vector>
#include <iostream>
#include <typeinfo>
using namespace std;
using namespace sycl;

int main(int argc, char *argv[]) {
    const size_t count = 50 * 1024 * 1024;

    sycl::property_list propList{cl::sycl::property::queue::in_order()};
    sycl::queue queue1(sycl::gpu_selector{}, propList);
    sycl::queue queue2(sycl::gpu_selector{}, propList);

    /* create buffers */
    auto compute_buf1 = sycl::malloc_device<float>(count, queue1);
    auto compute_buf2 = sycl::malloc_device<float>(count, queue1);
    auto recv_bufs1 = sycl::malloc_device<float>(count, queue1);
    auto recv_bufs2 = sycl::malloc_device<float>(count, queue1);

    int32_t max = 50 * 1024 * 1024;
    const int wg_size = queue1.get_device().get_info<sycl::info::device::max_work_group_size>();
    const int num_wg = (count + wg_size - 1) / wg_size;
    const int total_threads_needed = 1;
    const int total_wg_size = 1;

    for (int i = 0; i < 10; i ++) {
       for (int j = 0; j < 2; j ++) {
           queue2.submit([&](sycl::handler& cgh) {
                cgh.parallel_for<class queue2_sync_kernel>(
                    sycl::nd_range<1>({total_threads_needed}, total_wg_size), [=](sycl::item<1> item) {
                        for (int i =0; i < 1024*256; i++) {  // default 1024 * 1024
                            compute_buf2[i] = 0.0001 * count;
                        }
                });
           });
           
          queue1.submit([&](sycl::handler &cgh) {
            cgh.parallel_for<class queue1_compute_kernel>(sycl::nd_range<1>(num_wg * wg_size, wg_size),
                    [=](sycl::nd_item<1> item) {
                    int id = item.get_global_id();
                    if (id >= max) return;
                    compute_buf1[id] += id * 0.00001;
                    for (int j = 0; j < 50; j++) {
                      compute_buf1[id] += j * 0.0001;
                    }
            });
          });
          queue2.ext_oneapi_submit_barrier({queue1.ext_oneapi_submit_barrier()});
       }
       printf("finished iteration %d", i);
       queue2.wait();
       queue1.wait();
    }

    printf("start to execute q2 sync kernels");
    for (int i = 0; i < 10; i ++) {
       queue2.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<class queue2_sync_kernel2>(
                sycl::nd_range<1>({total_threads_needed}, total_wg_size), [=](sycl::item<1> item) {
                    for (int i =0; i < 1024*256; i++) {
                        compute_buf2[i] = 0.0001 * count;
                    }
            });
       });
       queue2.wait();
    }

    printf("start to execute q1 compute kernels");
    for (int i = 0; i < 10; i ++) {
       queue1.submit([&](sycl::handler &cgh) {
        cgh.parallel_for<class queue1_compute_kernel2>(sycl::nd_range<1>(num_wg * wg_size, wg_size),
                [=](sycl::nd_item<1> item) {
                int id = item.get_global_id();
                if (id >= max) return;
                compute_buf1[id] += id * 0.00001;
                for (int j = 0; j < 50; j++) {
                  compute_buf1[id] += j * 0.0001;
                }
        });
      });
      queue1.wait();
    }

    printf("finished all");
    return 0;
}
