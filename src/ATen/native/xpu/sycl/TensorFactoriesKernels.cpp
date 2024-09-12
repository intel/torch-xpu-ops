#include <ATen/ATen.h>
#include <ATen/native/TensorFactories.h>
#include <ATen/native/xpu/sycl/ScanUtils.h>
#include <ATen/xpu/EmptyTensor.h>
#include <c10/core/TensorOptions.h>
#include <comm/DeviceProperties.h>
#include <comm/SYCLHelpers.h>

namespace at::native::xpu {

// To find the max integer that does not exceed the root of an int64_t variable,
// we could use a loop to test one bit at a time, which takes up to 31
// iterations. This would give the accurate result, but is relatively slow and
// is an overkill for most cases where double's precision suffice.
//
// If we directly use sqrt to calculate the root, the convertion from int64_t
// to double would lose 11 bits precision.
//
// The following solution uses sqrt directly for most cases, and would only
// special handle it if there is indeed precision loss.
inline int64_t resolve_root_int(
    int64_t b,
    int64_t cX4,
    int64_t x,
    int32_t sign) {
  int64_t bXb_cX4 = b * b - cX4;
  // potential precision loss could occur here when casting int64_t (63 bits
  // precision) to double (52 bits precision)
  double sr = std::sqrt((double)bXb_cX4);
  //
  // TODO: PyTorch uses ::__double2ll_rd. No corresponding API in DPCPP.
  // uses std::llround or std::ceil or std::float will cause error:
  // terminate called after throwing an instance of
  // 'sycl::compile_program_error'.
  //
  int64_t res = static_cast<int64_t>((-b + sign * sr) / 2);

  // have to cast double to int64_t, otherwise it would only compare up to the
  // precision of a double variable, ignoring the precision loss
  if (bXb_cX4 != (int64_t)(sr * sr)) {
    // TODO:PyTorch uses ::__double2ll_rd && ::__double2ll_ru. No corresponding
    // API in DPCPP.
  }

  return res;
}

inline void get_coordinate_in_triu_trapezoid(
    int64_t f,
    int64_t x,
    int64_t& row,
    int64_t& col) {
  f <<= 1; // all statements use 2f, so only calculate it once here.
  auto b = -1 - f;
  auto cX4 = x << 3; // 4 * c = 4 * (2x) = 8x;
  row = resolve_root_int(b, cX4, x, -1);
  col = x - ((f - row + 1) * row >> 1) + row;
}

inline void get_coordinate_in_tril_trapezoid(
    int64_t f,
    int64_t x,
    int64_t& row,
    int64_t& col) {
  f <<= 1; // all statements use 2f, so only calculate it once here.
  auto b = f - 1;
  auto cX4 = -(x << 3); // 4 * c = 4 * (-2x) = -8x;
  row = resolve_root_int(b, cX4, x, 1);
  col = x - ((f + row - 1) * row >> 1);
}

template <typename scalar_t>
struct TriuIndicesKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    auto tensor_ptr = data_;
    int64_t r, c;
    for (int64_t linearIndex = item.get_global_id(0);
         linearIndex < totalElements_;
         linearIndex += item.get_global_range()[0]) {
      if (linearIndex < rectangle_size_) {
        // the coordinate is within the top rectangle
        r = linearIndex / col_;
        c = linearIndex % col_;
      } else {
        // the coordinate falls in the bottom trapezoid
        get_coordinate_in_triu_trapezoid(
            m_first_row_, linearIndex - rectangle_size_, r, c);
        r += rectangle_size_ / col_;
      }
      c += col_offset_;
      tensor_ptr[linearIndex] = r;
      tensor_ptr[linearIndex + triu_size_] = c;
    }
  }
  TriuIndicesKernelFunctor(
      scalar_t* data,
      int64_t col_offset,
      int64_t m_first_row,
      int64_t col,
      int64_t rectangle_size,
      int64_t triu_size,
      int64_t totalElements)
      : data_(data),
        col_offset_(col_offset),
        m_first_row_(m_first_row),
        col_(col),
        rectangle_size_(rectangle_size),
        triu_size_(triu_size),
        totalElements_(totalElements) {}

 private:
  scalar_t* data_;
  int64_t col_offset_;
  int64_t m_first_row_;
  int64_t col_;
  int64_t rectangle_size_;
  int64_t triu_size_;
  int64_t totalElements_;
};

template <typename scalar_t>
void triu_indices_kernel_template(
    scalar_t* tensor,
    int64_t col_offset,
    int64_t m_first_row,
    int64_t col,
    int64_t rectangle_size,
    int64_t triu_size) {
  using Kernel = TriuIndicesKernelFunctor<scalar_t>;
  int64_t group_size = syclMaxWorkGroupSize<Kernel>();
  auto totalElements = triu_size;
  auto num_groups = CeilDiv(totalElements, group_size);
  auto total_items = num_groups * group_size;

  auto data = tensor;

  Kernel kfn(
      data,
      col_offset,
      m_first_row,
      col,
      rectangle_size,
      triu_size,
      totalElements);

  // kick off kernel
  sycl_kernel_submit({total_items}, {group_size}, getCurrentSYCLQueue(), kfn);
}

template <typename scalar_t>
struct TrilIndicesKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    auto tensor_ptr = data_;
    int64_t r, c;
    for (int64_t linearIndex = item.get_global_id(0);
         linearIndex < totalElements_;
         linearIndex += item.get_global_range()[0]) {
      if (linearIndex < trapezoid_size_) {
        // the coordinate is within the top trapezoid
        get_coordinate_in_tril_trapezoid(m_first_row_, linearIndex, r, c);
      } else {
        // the coordinate falls in the bottom rectangle
        auto surplus = linearIndex - trapezoid_size_;
        // add the height of trapezoid: m_last_row (col) - m_first_row + 1
        r = surplus / col_ + col_ - m_first_row_ + 1;
        c = surplus % col_;
      }
      r += row_offset_;
      tensor_ptr[linearIndex] = r;
      tensor_ptr[linearIndex + tril_size_] = c;
    }
  }
  TrilIndicesKernelFunctor(
      scalar_t* data,
      int64_t row_offset,
      int64_t m_first_row,
      int64_t col,
      int64_t trapezoid_size,
      int64_t tril_size,
      int64_t totalElements)
      : data_(data),
        row_offset_(row_offset),
        m_first_row_(m_first_row),
        col_(col),
        trapezoid_size_(trapezoid_size),
        tril_size_(tril_size),
        totalElements_(totalElements) {}

 private:
  scalar_t* data_;
  int64_t row_offset_;
  int64_t m_first_row_;
  int64_t col_;
  int64_t trapezoid_size_;
  int64_t tril_size_;
  int64_t totalElements_;
};

template <typename scalar_t>
void tril_indices_kernel_template(
    scalar_t* tensor,
    int64_t row_offset,
    int64_t m_first_row,
    int64_t col,
    int64_t trapezoid_size,
    int64_t tril_size) {
  using Kernel = TrilIndicesKernelFunctor<scalar_t>;
  int64_t group_size = syclMaxWorkGroupSize<Kernel>();
  auto totalElements = tril_size;
  auto num_groups = CeilDiv(totalElements, group_size);
  auto total_items = num_groups * group_size;

  auto data = tensor;

  Kernel kfn(
      data,
      row_offset,
      m_first_row,
      col,
      trapezoid_size,
      tril_size,
      totalElements);

  // kick off kernel
  sycl_kernel_submit({total_items}, {group_size}, getCurrentSYCLQueue(), kfn);
}

Tensor tril_indices_kernel(
    int64_t row,
    int64_t col,
    int64_t offset,
    const TensorOptions& options) {
  check_args(row, col, options.layout());

  auto tril_size = get_tril_size(row, col, offset);
  auto tensor = at::detail::empty_xpu({2, tril_size}, options);

  if (tril_size > 0) {
    auto m_first_row = (offset > 0) ? std::min<int64_t>(col, 1 + offset)
                                    : // upper bounded by col
        (row + offset > 0); // either 0 or 1
    auto trapezoid_row_offset = std::max<int64_t>(0, -offset);
    auto rectangle_row_offset = trapezoid_row_offset + col - m_first_row + 1;

    int64_t rectangle_size = 0;
    if (rectangle_row_offset < row) {
      rectangle_size = (row - rectangle_row_offset) * col;
    }

    AT_DISPATCH_INDEX_TYPES(tensor.scalar_type(), "tril_indices_xpu", [&] {
      tril_indices_kernel_template<index_t>(
          tensor.data_ptr<index_t>(),
          trapezoid_row_offset,
          m_first_row,
          col,
          tril_size - rectangle_size,
          tril_size);
    });
  }

  return tensor;
}

Tensor triu_indices_kernel(
    int64_t row,
    int64_t col,
    int64_t offset,
    const TensorOptions& options) {
  check_args(row, col, options.layout());

  auto triu_size = row * col - get_tril_size(row, col, offset - 1);
  auto tensor = at::detail::empty_xpu({2, triu_size}, options);

  if (triu_size > 0) {
    // # of triu elements in the first row
    auto m_first_row = (offset > 0) ? std::max<int64_t>(col - offset, 0)
                                    : // upper bounded by col
        col;

    // size of the top rectangle
    int64_t rectangle_size = 0;
    if (offset < 0) {
      rectangle_size = std::min<int64_t>(row, -offset) * col;
    }

    AT_DISPATCH_INDEX_TYPES(tensor.scalar_type(), "triu_indices_xpu", [&] {
      triu_indices_kernel_template<index_t>(
          tensor.data_ptr<index_t>(),
          std::max<int64_t>(0, offset),
          m_first_row,
          col,
          rectangle_size,
          triu_size);
    });
  }

  return tensor;
}

} // namespace at::native::xpu
