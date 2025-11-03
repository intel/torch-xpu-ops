#include <ATen/Operators.h>
#include <c10/xpu/XPUStream.h>
#include <oneapi/mkl/lapack.hpp>
#include <torch/all.h>
#include <torch/library.h>

namespace at::native::xpu {

void linalg_qr_kernel(
    const at::Tensor& A,
    std::string_view mode,
    const at::Tensor& Q,
    const at::Tensor& R) {
  std::cout << "Call mode is " << " " << mode << std::endl;

  TORCH_CHECK(A.device().is_xpu(), "A must be an XPU tensor");
  at::Tensor a_contig = A.contiguous();
  at::Tensor result_r = at::clone(a_contig);
  //  at::Tensor result_r = at::empty_like(a_contig);
  at::Tensor result_c = at::empty_like(a_contig);
  at::Tensor result = at::empty_like(a_contig);

  auto dimensions = A.sizes();

  std::cout << "dim " << dimensions << std::endl;

  std::cout << result_r << std::endl;
  result_r = result_r.transpose(-2, -1).contiguous();
  std::cout << result_r << std::endl;
  int numel = a_contig.numel();
  int range = a_contig.dim();
  int64_t n = a_contig.sizes().at(range - 2);
  int64_t m = a_contig.sizes().at(range - 1);
  int64_t mn = int64_t(m * n);
  int64_t b = numel / mn;

  int out_q_columns = m > n ? n : m;
  if (n > m && mode == "complete") {
    out_q_columns = n;
  }

  std::cout << "dim2 " << n << " " << m << " " << b << " " << out_q_columns
            << std::endl;
  // at::Tensor result_q = result_r.clone();
  // dimensions[1]=out_q_columns;
  std::vector v(dimensions.begin(), dimensions.end());
  v[range - 1] = v[range - 2];
  v[range - 2] = out_q_columns;
  auto ndimensions = at::IntArrayRef(v);
  at::Tensor result_q = at::empty(ndimensions);

  sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue();
  int64_t mn1 =
      oneapi::mkl::lapack::geqrf_scratchpad_size<float>(queue, n, m, n);
  int64_t mn2 =
      oneapi::mkl::lapack::orgqr_scratchpad_size<float>(queue, n, m, m, n);
  mn2 = mn1 > mn2 ? mn1 : mn2;
  float* sbuffer = sycl::malloc_device<float>(mn2, queue);
  float* tau_buf = sycl::malloc_device<float>(out_q_columns, queue);
  float* r_buf = result_r.data_ptr<float>();
  float* q_buf = result_q.data_ptr<float>();

  std::cout << "entering " << n << " " << m << " " << mode << " "
            << (mode == "complete") << std::endl;

  for (int batch_item = 0; batch_item < b; batch_item++) {
    oneapi::mkl::lapack::geqrf(queue, n, m, r_buf, n, tau_buf, sbuffer, mn2);

    if (mode != "r") {
      // copy relevant part of R matrix to Q matrix
      int copy_columns = out_q_columns > m ? m : out_q_columns;
      queue.memcpy(q_buf, r_buf, n * copy_columns * sizeof(float)).wait();

      oneapi::mkl::lapack::orgqr(
          queue,
          n,
          out_q_columns,
          out_q_columns,
          q_buf,
          n,
          tau_buf,
          sbuffer,
          mn2);
      std::cout << "done" << std::endl;

      sycl::free(sbuffer, queue);
      std::cout << "done2" << std::endl;
    }

    r_buf += mn;
    q_buf += n * out_q_columns;

  } // batch

  if (mode == "r") {
    result_q = at::empty({0, 0});
  }

  if ((mode == "reduced" || mode == "r") && n > m) {
    result_r =
        result_r
            .index(
                {"...", at::indexing::Slice(0, n), at::indexing::Slice(0, m)})
            .contiguous();
  }

  // result_q.transpose(0,1);
  // return std::make_tuple(
  // result_q.transpose(-2, -1), result_r.transpose(-2, -1).triu_());
}

} // namespace at::native::xpu
// }
