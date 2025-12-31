/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#pragma clang diagnostic push
#pragma GCC diagnostic push
// Avoid SYCL compiler return-type error
#pragma clang diagnostic ignored "-Wreturn-type"
#pragma GCC diagnostic ignored "-Wreturn-type"

#ifdef _MSC_VER
  #pragma warning(push)
  #pragma warning(disable : 4715)
#endif

#include <ATen/AccumulateType.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/xpu/sycl/Atomics.h>
#include <ATen/native/xpu/sycl/DistributionTemplates.h>
#include <ATen/native/xpu/sycl/MultinomialKernel.h>
#include <ATen/native/xpu/sycl/SYCLGroupAlgorithm.h>
#include <ATen/xpu/EmptyTensor.h>
#include <ATen/xpu/XPUGeneratorImpl.h>
#include <comm/Runtime.h>
#include <comm/SYCLContext.h>
#include <comm/SYCLHelpers.h>

#include <ATen/native/xpu/sycl/MultinomialKernel.h>

namespace at::native::xpu {

template <typename scalar_t, typename item_t>
inline void renormRowsL1(
    item_t& item,
    scalar_t* dist,
    int64_t rows,
    int64_t cols,
    unsigned char* my_smem) {
  auto thread_idx = item.get_local_id(0);
  auto thread_range = item.get_local_range(0);
  auto group_idx = item.get_group(0);
  auto group_range = item.get_group_range(0);

  scalar_t* smem = reinterpret_cast<scalar_t*>(my_smem);
  scalar_t zero = static_cast<scalar_t>(0);
  scalar_t val;
  for (int64_t row = group_idx; row < rows; row += group_range) {
    scalar_t sum = static_cast<scalar_t>(0);
    for (int64_t col = thread_idx; col < cols; col += thread_range) {
      val = dist[row * cols + col];
      sum = sum + val;
    }

    sum = GroupReduceSumSGSizeEqualstoNumSG(item, sum, smem);
    if (thread_idx == 0) {
      smem[0] = sum;
    }
    sycl::group_barrier(item.get_group());

    sum = smem[0];
    if (sum > zero) {
      for (int64_t col = thread_idx; col < cols; col += thread_range) {
        dist[row * cols + col] = dist[row * cols + col] / sum;
      }
    }
  }
}

template <typename scalar_t>
struct RenormRowsKernelFunctor : public __SYCL_KER_CONFIG_CONVENTION__ {
  void operator()(sycl::nd_item<1> item) const {
    renormRowsL1<scalar_t>(
        item,
        t_ptr,
        rows,
        cols,
        (unsigned char*)(smem.template get_multi_ptr<
                                 sycl::access::decorated::no>()
                             .get()));
  }
  void sycl_ker_config_convention(sycl::handler& cgh) {
    smem = sycl_local_acc_t<scalar_t>(group_size_ / 8, cgh);
    // We use the smallest subgroup size to ensure enough space
  }
  RenormRowsKernelFunctor(
      int64_t rows_,
      int64_t cols_,
      scalar_t* t_ptr_,
      int group_size)
      : rows(rows_), cols(cols_), t_ptr(t_ptr_), group_size_(group_size) {}

 private:
  int64_t rows;
  int64_t cols;
  scalar_t* t_ptr;
  int group_size_;
  sycl_local_acc_t<scalar_t> smem;
};

inline void renormRows(Tensor& t) {
  TORCH_CHECK(t.dim() == 2);
  int64_t rows = t.size(0);
  int64_t cols = t.size(1);
  int subgroup_size = syclMaxSubGroupSize();
  int group_size = std::min(
      int(syclMaxWorkItemsPerSubSlice()), subgroup_size* subgroup_size);
  int num_groups = (rows + group_size - 1) / group_size;
  int hw_max_groups = syclMaxWorkItemsPerTile() / group_size;
  num_groups = num_groups > hw_max_groups ? hw_max_groups : num_groups;

  auto& sycl_queue = at::xpu::getCurrentSYCLQueue();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      t.scalar_type(),
      "renormRows_xpu",
      [&] {
        auto t_ptr = t.data_ptr<scalar_t>();
        auto kfn =
            RenormRowsKernelFunctor<scalar_t>(rows, cols, t_ptr, group_size);
        sycl_kernel_submit(
            num_groups * group_size, group_size, sycl_queue, kfn);
      });
}

template <typename scalar_t>
inline int binarySearchForMultinomial(
    scalar_t* cumdist,
    scalar_t* dist,
    int size,
    scalar_t val) {
  int start = 0;
  int end = size;
  // cumdist[size - 1] = 0 => all zero prob dist

  while (end - start > 0) {
    int mid = start + (end - start) / 2;

    scalar_t midVal = cumdist[mid];
    if (midVal < val) {
      start = mid + 1;
    } else {
      end = mid;
    }
  }

  if (start == size) {
    // No probability mass or precision problems; just return the
    // first non-zero element by setting start to size-1 here,
    // the code below will move it to the last non-zero probability
    // this actually can happen when the random number is 1
    // (github pytorch issue #4858).
    start = size - 1;
  }

  while (start >= 1 && dist[start] == 0)
    start--;

  return start;
}

template <typename scalar_t, typename item_t>
inline void sampleMultinomialWithReplacement(
    item_t& item,
    PhiloxXpuState philox_args,
    int totalSamples,
    int64_t* dest,
    int64_t distributions,
    int categories,
    scalar_t* normDistPrefixSum,
    scalar_t* normDist) {
  auto thread_idx = item.get_local_id(1);
  auto thread_range = item.get_local_range(1);
  auto group_idx_x = item.get_group(1);
  auto group_idx_y = item.get_group(0);
  auto group_range_x = item.get_group_range(1);
  auto group_range_y = item.get_group_range(0);

  // At the moment, each subgroup computes one sample value in the binary
  // search due to divergence. It seems possible to compute multiple
  // values and limit divergence though later on.

  auto seeds = at::xpu::philox::unpack(philox_args);

  // global index formula for 2D grid of 1D group
  int idx = group_idx_y * group_range_x * thread_range +
      group_idx_x * thread_range + thread_idx;

  randStatePhilox4_32_10_t state;
  rand_init(std::get<0>(seeds), idx, std::get<1>(seeds), &state);

  // The block determines the distribution for which we generate a point
  for (int64_t curDist = group_idx_y; curDist < distributions;
       curDist += group_range_y) {
    for (int sample = group_idx_x * thread_range + thread_idx;
         sample < totalSamples;
         sample += thread_range * group_range_x) {
      // we are losing 3 out of 4 generated numbers but it's ok
      // this kernel is not very efficient anyway
      auto rand = rand_uniform4(&state);
      scalar_t r = static_cast<scalar_t>(rand.x);

      // Find the bucket that a uniform sample lies in
      int choice = binarySearchForMultinomial<scalar_t>(
          normDistPrefixSum + curDist * categories,
          normDist + curDist * categories,
          categories,
          r);

      dest[curDist * totalSamples + sample] = choice;
    }
  }
}

template <typename scalar_t>
struct MultinomialWithReplacementKernelImplFunctor {
  void operator()(sycl::nd_item<2> item) const {
    sampleMultinomialWithReplacement(
        item,
        rng_engine_inputs,
        n_sample,
        result_ptr,
        numDist,
        numCategories,
        prefixSum_ptr,
        normDist_ptr);
  }
  MultinomialWithReplacementKernelImplFunctor(
      PhiloxXpuState rng_engine_inputs_,
      const int64_t n_sample_,
      int64_t* result_ptr_,
      int64_t numDist_,
      int numCategories_,
      scalar_t* prefixSum_ptr_,
      scalar_t* normDist_ptr_)
      : rng_engine_inputs(rng_engine_inputs_),
        n_sample(n_sample_),
        result_ptr(result_ptr_),
        numDist(numDist_),
        numCategories(numCategories_),
        prefixSum_ptr(prefixSum_ptr_),
        normDist_ptr(normDist_ptr_) {}

 private:
  PhiloxXpuState rng_engine_inputs;
  const int64_t n_sample;
  int64_t* result_ptr;
  int64_t numDist;
  int numCategories;
  scalar_t* prefixSum_ptr;
  scalar_t* normDist_ptr;
};
template <typename scalar_t, typename accscalar_t>
struct SampleMultinomialOnceFunctor : public __SYCL_KER_CONFIG_CONVENTION__ {
  void operator()(sycl::nd_item<1> item) const {
    accscalar_t* smem = reinterpret_cast<accscalar_t*>(
        smem_.template get_multi_ptr<sycl::access::decorated::no>().get());

    accscalar_t accZero = static_cast<accscalar_t>(0);
    scalar_t zero = static_cast<scalar_t>(0);
    int local_id = item.get_local_id(0);
    int local_range = item.get_local_range(0);

    for (int64_t curDist = item.get_group(0); curDist < distributions_;
         curDist += item.get_group_range(0)) {
      // First pass, find the total sum of the distribution
      accscalar_t sum = accZero;
      scalar_t val;
      for (int cat = item.get_local_id(0); cat < categories_;
           cat += item.get_local_range(0)) {
        val = dist_[curDist * stride_dist_ + cat * stride_categories_];
        SYCL_KERNEL_ASSERT(!at::_isnan(val));
        SYCL_KERNEL_ASSERT(!_isinf(val));
        SYCL_KERNEL_ASSERT(!(val < zero));
        sum = sum + static_cast<accscalar_t>(val);
      }

      sum = GroupReduceSumSGSizeEqualstoNumSG(item, sum, smem);

      // Broadcast sum and sample value
      if (item.get_local_id(0) == 0) {
        // Make sure the sum of our distribution didn't overflow
        SYCL_KERNEL_ASSERT(!_isinf(val));
        SYCL_KERNEL_ASSERT(sum > accZero);

        foundPos_[0] = 0;
        smem[0] = sum;
        smem[1] = sampled_[curDist];
      }
      sycl::group_barrier(item.get_group());

      sum = smem[0];
      scalar_t sample = static_cast<scalar_t>(smem[1]);
      sycl::group_barrier(item.get_group());

      if (sum == accZero) {
        // Choose the first element
        if (local_id == 0) {
          dest_[curDist] = 0;
        }

        continue;
      }

      int chunks = (categories_ + (int)local_range - 1) / local_range;
      accscalar_t prevHighProb = accZero;
      found_[0] = false;

      for (int chunk = 0; chunk < chunks && !found_[0]; ++chunk) {
        // All threads in bounds load a value
        int cat = chunk * local_range + local_id;

        accscalar_t dist_val = cat < categories_
            ? static_cast<accscalar_t>(
                  dist_[curDist * stride_dist_ + cat * stride_categories_]) /
                sum
            : accZero;

        smem[local_id] = dist_val;
        sycl::group_barrier(item.get_group());

        // Perform an inclusive prefix sum of the shared memory contents
        for (int offset = 1; offset < local_range; offset *= 2) {
          accscalar_t val = accZero;

          if (local_id >= offset) {
            val = smem[local_id - offset] + smem[local_id];
          }

          sycl::group_barrier(item.get_group());
          if (local_id >= offset) {
            smem[local_id] = val;
          }
          sycl::group_barrier(item.get_group());
        }

        // Each thread will check to see if the sample falls in its
        // bucket
        scalar_t curBucket =
            static_cast<scalar_t>(smem[local_id] + prevHighProb);
        scalar_t prevBucket = static_cast<scalar_t>(
            local_id == 0 ? prevHighProb : smem[local_id - 1] + prevHighProb);
        bool inBucket = (cat < categories_) &&
            (!(sample >= curBucket) && (sample >= prevBucket) &&
             (dist_val > zero));

        if (inBucket) {
          // We're done; we have the sample
          // Torch indices are 1-based

          atomicMax(
              sycl_local_ptr<int>(
                  foundPos_
                      .template get_multi_ptr<sycl::access::decorated::no>()
                      .get()),
              cat);

          found_[0] = true;
        }

        // Store the previous scan's high value for future use
        prevHighProb = prevHighProb + smem[local_range - 1];

        sycl::group_barrier(item.get_group());
      }

      if (local_id == 0) {
        if (found_[0]) {
          dest_[curDist] = foundPos_[0];
        } else {
          // This should address a rare bug where we don't select a valid index.
          // This likely occurs when due to floating point arithmetic rounding
          // errors, our cumulative sum does not add up to 1, but and our
          // uniform sample is greater than this value. In this case we likely
          // have unitialized memory in dest[curDist]. So basically we will loop
          // through the distribution and pick the largest index where the
          // distribution is non-zero. This is obviously terribly inefficient,
          // but due to the rarity in which this occurs, this should not be an
          // issue.
          for (int cat = categories_ - 1; cat >= 0; --cat) {
            if (dist_[curDist * stride_dist_ + cat * stride_categories_] >
                zero) {
              dest_[curDist] = cat;
              break;
            }
          }
        }
      }
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    smem_ = sycl_local_acc_t<scalar_t>(group_size_, cgh);
    found_ = sycl_local_acc_t<bool>(1, cgh);
    foundPos_ = sycl_local_acc_t<int>(1, cgh);
  }

  SampleMultinomialOnceFunctor(
      int64_t* dest,
      int64_t distributions,
      int categories,
      const scalar_t* sampled,
      const scalar_t* dist,
      int stride_dist, // dist->stride(0)
      int stride_categories, // dist->stride(1)
      int group_size)
      : dest_(dest),
        distributions_(distributions),
        categories_(categories),
        sampled_(sampled),
        dist_(dist),
        stride_dist_(stride_dist),
        stride_categories_(stride_categories),
        group_size_(group_size) {}

 private:
  int64_t* dest_;
  int64_t distributions_;
  int categories_;
  const scalar_t* sampled_;
  const scalar_t* dist_;
  int stride_dist_;
  int stride_categories_;
  int group_size_;
  sycl_local_acc_t<scalar_t> smem_;
  sycl_local_acc_t<bool> found_;
  sycl_local_acc_t<int> foundPos_;
};

void multinomial_kernel(
    Tensor& result,
    const Tensor& self,
    const int64_t n_sample,
    std::optional<Generator> generator) {
  auto& sycl_queue = at::xpu::getCurrentSYCLQueue();
  auto gen = get_generator_or_default<at::XPUGeneratorImpl>(
      generator, at::xpu::detail::getDefaultXPUGenerator());

  int inputSize = self.dim();
  int64_t numDist = inputSize == 1 ? 1 : self.size(0);
  int numCategories = inputSize == 1 ? self.size(0) : self.size(1);

  // Restructure data for 2d
  auto self_v = inputSize == 1 ? self.view({numDist, numCategories}) : self;

  result.resize_({numDist, n_sample});

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      self_v.scalar_type(),
      "multinomial_kernel_xpu",
      [&] {
        using accscalar_t = acc_type_device<scalar_t, kXPU>;
        using KernelClass = SampleMultinomialOnceFunctor<scalar_t, accscalar_t>;
        int maxThreads = syclMaxWorkGroupSize<KernelClass>();
        int maxShared = syclLocalMemSize();

        int SubGroupSize = syclMinSubGroupSize();
        int requiredSubGroups = at::ceil_div(numCategories, SubGroupSize);
        int requiredThreads =
            std::min(maxThreads, requiredSubGroups * SubGroupSize);
        int requiredShared = requiredThreads * sizeof(accscalar_t);
        if (n_sample == 1 && maxShared >= requiredShared) {
          // Break per source build. Build every kernel source file into a so.
          // libtorch_xpu.so which is the so of operator level depends the so
          // of kernels. Do not depends on libtorch_xpu.so inversively. Using
          // at::empty instead of at::detail::empty_xpu.
          Tensor sampled = at::empty({numDist, n_sample}, self_v.options());
          at::native::uniform_(sampled, 0.0, 1.0, generator);
          int group_size = requiredThreads;
          int group_range = numDist;
          auto kfn = KernelClass(
              result.mutable_data_ptr<int64_t>(),
              numDist,
              numCategories,
              sampled.const_data_ptr<scalar_t>(),
              self_v.const_data_ptr<scalar_t>(),
              self_v.stride(0),
              self_v.stride(1),
              group_size);

          sycl_kernel_submit(
              group_range * group_size, group_size, sycl_queue, kfn);
        } else {
          Tensor origDist = native::empty_like(
              self_v,
              std::nullopt /* dtype */,
              std::nullopt /* layout */,
              std::nullopt /* device */,
              std::nullopt /* pin_memory */,
              LEGACY_CONTIGUOUS_MEMORY_FORMAT);
          origDist.copy_(self_v);

          Tensor normDist = native::empty_like(
              self_v,
              std::nullopt /* dtype */,
              std::nullopt /* layout */,
              std::nullopt /* device */,
              std::nullopt /* pin_memory */,
              LEGACY_CONTIGUOUS_MEMORY_FORMAT);

          Tensor prefixSum = native::empty_like(
              self_v,
              std::nullopt /* dtype */,
              std::nullopt /* layout */,
              std::nullopt /* device */,
              std::nullopt /* pin_memory */,
              LEGACY_CONTIGUOUS_MEMORY_FORMAT);

          // Renorm along rows
          normDist.copy_(origDist);
          renormRows(normDist);

          // Prefix sum along rows
          at::cumsum_out(prefixSum, normDist, 1);
          int group_size = syclMaxWorkItemsPerSubSlice();
          int group_range_y = numDist;
          int group_range_x = (n_sample - 1) / group_size + 1;

          PhiloxXpuState rng_engine_inputs;
          {
            // See Note [Acquire lock when using random generators]
            std::lock_guard<std::mutex> lock(gen->mutex_);
            auto offset = ((numDist - 1) / group_range_y + 1) * 4;
            rng_engine_inputs = gen->philox_xpu_state(offset);
          }

          auto result_ptr = result.data_ptr<int64_t>();
          auto prefixSum_ptr = prefixSum.data_ptr<scalar_t>();
          auto normDist_ptr = normDist.data_ptr<scalar_t>();
          auto kfn = MultinomialWithReplacementKernelImplFunctor<scalar_t>(
              rng_engine_inputs,
              n_sample,
              result_ptr,
              numDist,
              numCategories,
              prefixSum_ptr,
              normDist_ptr);

          sycl_kernel_submit(
              sycl::range<2>(group_range_y, group_range_x * group_size),
              sycl::range<2>(1, group_size),
              sycl_queue,
              kfn);
        }
      });

  if (inputSize == 1) {
    result.resize_({n_sample});
  }
}

} // namespace at::native::xpu
#pragma GCC diagnostic pop
#pragma clang diagnostic pop

#ifdef _MSC_VER
  #pragma warning(pop)
#endif

