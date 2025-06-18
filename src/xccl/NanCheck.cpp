#ifdef USE_C10D_XCCL

#include <ATen/Dispatch.h>
#include <ATen/xpu/XPUContext.h>
#include <c10/xpu/XPUGuard.h>
#include <torch/torch.h>
#include <algorithm>
#include <xccl/NanCheck.hpp>
#include <ATen/native/xpu/sycl/MemoryAccessUtils.h>
#include <stdint.h>

namespace c10d {

using BytePack = memory::aligned_vector<uint64_t, 2>;

template <typename T, int EltPerPack>
struct CheckBytePack {
  static void check(BytePack* tmp) {
    T* data = (T*)tmp;
    #pragma unroll 8
    for (int i = 0; i < EltPerPack; i++) {
      if (isnan(data[i])) assert(0);
    }
  }
};

template <typename T>
struct CheckBytePack<T, /*EltPerPack*/2> {
  static void check(BytePack* tmp) {
    T* data = (T*)tmp;
    if (isnan(data[0]) || isnan(data[1])) assert(0);
  }
};

template <typename T>
struct CheckBytePack<T, /*EltPerPack*/4> {
  static void check(BytePack* tmp) {
    T* data = (T*)tmp;
    if (isnan(data[0]) || isnan(data[1]) || isnan(data[2]) || isnan(data[3])) assert(0);
  }
};


template <typename T>
struct CheckBytePack<T, /*EltPerPack*/8> {
  static void check(BytePack* tmp) {
    T* data = (T*)tmp;
    if (isnan(data[0]) || isnan(data[1]) || isnan(data[2]) || isnan(data[3]) ||
        isnan(data[4]) || isnan(data[5]) || isnan(data[6]) || isnan(data[7])) {
          assert(0);
    }
  }
};

template<typename T>
struct HasNanFP8x8 {
  static bool check(uint64_t fp8x8) = delete;
  /*
  {
    // `static_assert` in template definition requires c++23 onwards.
    // But the error message still applies if you find yourself here.
    static_assert(
      false,
      "You should never call this template definition because it is empty. You "
      "can follow the example of Float8_e4m3fn below to implement the check for "
      "your new datatype."
    );
  }
  */
};

template<>
struct HasNanFP8x8<c10::Float8_e4m3fn> {
  static bool check(uint64_t fp8x8) {
    auto t = fp8x8 & 0x7F7F7F7F7F7F7F7FULL;
    auto incremented = t + 0x0101010101010101ULL;
    auto overflow = incremented & 0x8080808080808080ULL;
    return overflow != 0;
  }
};

template<>
struct HasNanFP8x8<c10::Float8_e5m2> {
  static bool check(uint64_t fp8x8) {
    auto t = fp8x8 & 0x7F7F7F7F7F7F7F7FULL;
    auto incremented = t + 0x0303030303030303ULL;
    auto overflow = incremented & 0x8080808080808080ULL;
    return overflow != 0;
  }
};

template<typename T>
struct CheckBytePack<T, /*EltPerPack*/16> {
  static void check(BytePack* tmp) {
    if (HasNanFP8x8<T>::check(tmp->val[0]) || HasNanFP8x8<T>::check(tmp->val[1]))
        assert(0);
  }
};


#define UNROLL 8

template <typename T>
void checkChunk(BytePack* ptr, int nWorkers) {
  BytePack tmp[UNROLL];
  // int nWorkers = num_group_ * max_group_size_;
  #pragma unroll 8
  for (int j = 0; j < UNROLL; j++) {
    tmp[j] = ptr[nWorkers * j];
  }
  // Then check each BytePack in the tmp buffer
  #pragma unroll 8
  for (int j = 0; j < UNROLL; j++) {
    CheckBytePack<T, sizeof(BytePack)/sizeof(T)>::check(tmp + j);
  }
  // Note: we separate the check from the load for efficient loading
}

// Align address of `ptr` up, to the alignment of `T`
#define ALIGN_UP(ptr, T) (((uintptr_t)ptr + sizeof(T) - 1) / sizeof(T) * sizeof(T))


template <typename T>
struct checkForNaN {
  void operator()(sycl::nd_item<1> item) const {
    constexpr int EltPerPack = sizeof(BytePack) / sizeof(T);

    size_t offset = item.get_global_id()[2];

    // Align input address up to BytePack in case it is not
    T* ptrAlign = (T*)ALIGN_UP(data, BytePack);
    // Pre-process the data before alignment
    size_t preProcElts = min(ptrAlign - data, size);
    // Read memory by T (slow). One iter is enough bc the number of threads would
    // be bigger than `preProcElts`
    if (offset < preProcElts) {
      if (isnan(data[offset])) assert(0);
    }
    // We have processes this amount of data
    size -= preProcElts;

    // Start BytePack processing
    BytePack* ptr = (BytePack*)ptrAlign;
    // Size of input data in unit of BytePack
    size_t sizeInBP = size * sizeof(T) / sizeof(BytePack);
    // Number of BytePacks processed in one fast-path iteration
    size_t loopSize = num_group_ * max_group_size_ * UNROLL;

    // Fast path
    // The condition below makes sure there is enough data to process (`loopSize`)
    for (; offset + loopSize <= sizeInBP; offset += loopSize) {
      checkChunk<T>(ptr + offset, num_group_ * max_group_size_);
    }

    // The rest data goes on slow path
    // We just do regular load and check
    for (; offset < sizeInBP; offset += num_group_ * max_group_size_) {
      BytePack tmp = ptr[offset];
      CheckBytePack<T, EltPerPack>::check(&tmp);
    }

    // We can still have a tail smaller than 1 BytePack
    // TODO: merge this tail check with head check to make them concurrent
    if (item.get_local_id(1) < size % EltPerPack) {
      T* tailPtr = (T*)(ptr + sizeInBP);
      if (isnan(tailPtr[item.get_local_id(1)])) assert(0);
    }
  }
  checkForNaN(T* data, size_t size, int64_t num_group, int64_t max_group_size)
      : data(data), size(size), num_group_(num_group), max_group_size_(max_group_size) {}
  private:
    T* data;
    size_t size;
    int64_t num_group_;
    int64_t max_group_size_;
}

template <typename T>
void checkfornan_impl_xpu(
    const at::Tensor& tensor,
    at::xpu::XPUStream& stream) {
  // skip check for non float types
  if (!torch::is_floating_point(tensor)) {
    return;
  }

  int64_t maxNumThreadsPerBlock = syclMaxWorkGroupSize<checkForNaN<float>>();

  const size_t numThreadsPerBlock =
    std::min<size_t>(maxNumThreadsPerBlock, tensor.numel());

  if (!(numThreadsPerBlock > 0)) {
    return;
  }

  int64_t numBlocks = (tensor.numel() + maxNumThreadsPerBlock - 1) / maxNumThreadsPerBlock;
  auto global_range{numBlocks * maxNumThreadsPerBlock};
  auto local_range{maxNumThreadsPerBlock};

  using Kernel = checkForNaN<T>;
  auto knf = Kernel(
      tensor.data_ptr<T>(),
      tensor.numel(),
      numBlocks,
      maxNumThreadsPerBlock);

  sycl_kernel_submit(global_range, local_range, stream.queue(), knf);
}



// CHECK if a Tensor contains NAN in any of its element
void checkForNan(const at::Tensor& tensor, at::xpu::XPUStream& stream) {

  AT_DISPATCH_FLOATING_TYPES_AND4(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Float8_e4m3fn,
      at::ScalarType::Float8_e5m2,
      tensor.scalar_type(),
      "checkForNaN_XPU",
      [&] () {
        checkfornan_impl_xpu<scalar_t>(
            tensor,
            stream);
      });
}

} // namespace c10d

#endif // USE_C10D_XCCL


