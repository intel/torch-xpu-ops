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

  //TORCH_CHECK(A.device().is_xpu(), "a must be an XPU tensor");
  //TORCH_CHECK(A.dtype() == at::kFloat, "a must be float");

  at::Tensor a_contig = A.contiguous();
  at::Tensor result_r = at::clone(a_contig);

  auto options = at::TensorOptions().dtype(at::kFloat).device(kXPU);
  auto dimensions = A.sizes();

  result_r = result_r.transpose(-2, -1).contiguous();

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

  std::vector v(dimensions.begin(), dimensions.end());
  if (mode != "r") {
    v[range - 1] = v[range - 2];
    v[range - 2] = out_q_columns;
  } else {
    v = std::vector<long>({0, 0});
  }
  auto q_dimensions = at::IntArrayRef(v);

  at::Tensor result_q = at::empty(q_dimensions, options);



  sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue();

  int64_t bufsize1 =
      oneapi::mkl::lapack::geqrf_scratchpad_size<float>(queue, n, m, n);
  int64_t bufsize2 =
      oneapi::mkl::lapack::orgqr_scratchpad_size<float>(queue, n, m, m, n);

  int64_t bufsize = bufsize2 > bufsize1 ? bufsize2 : bufsize1;
  int64_t tau_len = m > n ? n : m;
  float* sbuffer = sycl::malloc_device<float>(bufsize, queue);
  float* tau_buf = sycl::malloc_device<float>(tau_len, queue);
  float* r_buf = result_r.data_ptr<float>();

  float* q_buf = NULL;
  if (mode != "r") {
    q_buf = result_q.data_ptr<float>();
  }

  for (int batch_item = 0; batch_item < b; batch_item++) {
    oneapi::mkl::lapack::geqrf(queue, n, m, r_buf, n, tau_buf, sbuffer, bufsize)
        .wait();

    if (mode != "r") {
      // copy relevant part of R matrix to Q matrix
      int copy_columns = out_q_columns > m ? m : out_q_columns;
      queue.memcpy(q_buf, r_buf, n * copy_columns * sizeof(float)).wait();

      oneapi::mkl::lapack::orgqr(
          queue,
          n,
          out_q_columns,
          tau_len,
          q_buf,
          n,
          tau_buf,
          sbuffer,
          bufsize)
          .wait();

      q_buf += n * out_q_columns;
    }

    r_buf += mn;

  } // batch

  sycl::free(sbuffer, queue);
  sycl::free(tau_buf, queue);

  if ((mode == "reduced" || mode == "r") && n > m) {
    result_r =
        result_r
            .index(
                {"...", at::indexing::Slice(0, n), at::indexing::Slice(0, m)})
            .contiguous();
  }

  Q.set_(result_q.transpose(-2, -1));
  R.set_(result_r.transpose(-2, -1).triu_());
  queue.wait();
}

} // namespace at::native::xpu
