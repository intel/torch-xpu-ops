#include <iostream>
#include <CL/sycl.hpp>
#include <CL/sycl.hpp>

void run1(sycl::queue& q1, sycl::queue& q2, float* dst, float* src, float* tmp1, float* tmp2, float* tmp3, int count)
{
    q1.submit([&](sycl::handler &h) {
        h.parallel_for(count, [=](cl::sycl::item<1> item) {
            int idx = item.get_id(0);
            tmp1[idx] = src[idx] * 2;
        });
    });

    q1.submit([&](sycl::handler &h) {
        h.parallel_for(count, [=](cl::sycl::item<1> item) {
            int idx = item.get_id(0);
            tmp2[idx] = tmp1[idx] + 3;
        });
    });

    auto e1 = q1.ext_oneapi_submit_barrier();
    auto e2 = q2.ext_oneapi_submit_barrier({e1});

    auto e3 = q2.submit([&](sycl::handler &h) {
        h.parallel_for(count, [=](cl::sycl::item<1> item) {
            int idx = item.get_id(0);
            tmp3[idx] = tmp1[idx] + 5;
        });
    });

    q1.ext_oneapi_submit_barrier({e3});
    q1.submit([&](sycl::handler &h) {
        h.parallel_for(count, [=](cl::sycl::item<1> item) {
            int idx = item.get_id(0);
            dst[idx] = tmp2[idx] + tmp3[idx];
        });
    });
}

int test1()
{
    sycl::queue q{sycl::gpu_selector_v, {sycl::property::queue::in_order(),
                                         sycl::property::queue::enable_profiling()}};
    int count = 1024 * 1024;
    float *inp = sycl::malloc_device<float>(count, q);
    float *outp = sycl::malloc_device<float>(count, q);
    float *tmp1 = sycl::malloc_device<float>(count, q);
    float *tmp2 = sycl::malloc_device<float>(count, q);
    float *tmp3 = sycl::malloc_device<float>(count, q);

    float *inp_h = new float[count];
    float *outp_h = new float[count];
    for (size_t i = 0; i < count; ++i) {
      inp_h[i] = i;
      outp_h[i] = 0;
    }

    sycl::queue q1{sycl::gpu_selector_v, {sycl::property::queue::in_order(),
                                         sycl::property::queue::enable_profiling()}};
    sycl::queue q2{sycl::gpu_selector_v, {sycl::property::queue::in_order(),
                                         sycl::property::queue::enable_profiling()}};

    // this cleanup is just for test purpose
    q.memset(inp, 0, count * sizeof(float)).wait();
    q.memset(outp, 0, count * sizeof(float)).wait();
    q.memset(tmp1, 0, count * sizeof(float)).wait();
    q.memset(tmp2, 0, count * sizeof(float)).wait();
    q.memset(tmp3, 0, count * sizeof(float)).wait();

    // record graph
    sycl::ext::oneapi::experimental::command_graph g {
            q.get_context(), q.get_device()};
    g.begin_recording({q1, q2});
    run1(q1, q2, outp, inp, tmp1, tmp2, tmp3, count);
    g.end_recording();
    auto execGraph = g.finalize();
    
    // this cleanup is just for test purpose
    q.memset(inp, 0, count * sizeof(float)).wait();
    q.memset(outp, 0, count * sizeof(float)).wait();
    q.memset(tmp1, 0, count * sizeof(float)).wait();
    q.memset(tmp2, 0, count * sizeof(float)).wait();
    q.memset(tmp3, 0, count * sizeof(float)).wait();

    // run graph
    q.memcpy(inp, inp_h, count * sizeof(float));
    q.ext_oneapi_graph(execGraph);
    q.memcpy(outp_h, outp, count * sizeof(float)).wait();
    for (int i = 0; i < count; ++i) {
      //if (i <= 20)
      //  std::cout << inp_h[i] << " " << outp_h[i] << std::endl;
      if (inp_h[i] * 4 + 8 != outp_h[i]) {
        std::cout << "in test_mqueues: not expected at index " << i << ", input " << inp_h[i] 
                  << ", expect " << inp_h[i] * 4 + 8 << " but got " << outp_h[i] << std::endl;
        return -1;
      }
    }

    std::cout << "test_mqueues finished." << std::endl;
    return 0;
}


int main()
{
  test1();
}
