#include <CL/sycl.hpp>
#include <vector>
#include <iostream>
#include <typeinfo>
#include <ext/intel/esimd.hpp>
using namespace std;
using namespace sycl;

int main(int argc, char *argv[]) {
    sycl::property_list propList{cl::sycl::property::queue::in_order()};
    sycl::queue queue1(sycl::gpu_selector{}, propList);

    const int wg_size = queue1.get_device().get_info<sycl::info::device::max_work_group_size>();
    const int t1 = queue1.get_device().get_info<sycl::info::device::max_compute_units>();
    const int t2 = queue1.get_device().get_info<sycl::info::device::max_sub_group_size>();
    const int t3 = queue1.get_device().get_info<sycl::info::device::max_work_item_per_compute_unit>();

    std::cout << " work groups size max: " << wg_size << " " << t1 << " : " << t2 << " : " << t3 << std::endl;

    return 0;
}