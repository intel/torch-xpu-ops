#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <aten/sycl/NMSKernel.h>
#include <comm/XPUGuard.h>
#include <torch/library.h>

namespace at::native::xpu {

Tensor nms(const Tensor& dets, const Tensor& scores, double iou_threshold_) {
  float iou_threshold = (float)iou_threshold_;
  TORCH_CHECK(dets.is_xpu(), "dets must be a XPU tensor");
  TORCH_CHECK(scores.is_xpu(), "scores must be a XPU tensor");

  TORCH_CHECK(
      dets.dim() == 2, "boxes should be a 2d tensor, got ", dets.dim(), "D");
  TORCH_CHECK(
      dets.size(1) == 4,
      "boxes should have 4 elements in dimension 1, got ",
      dets.size(1));
  TORCH_CHECK(
      scores.dim() == 1,
      "scores should be a 1d tensor, got ",
      scores.dim(),
      "D");
  TORCH_CHECK(
      dets.size(0) == scores.size(0),
      "boxes and scores should have same number of elements in ",
      "dimension 0, got ",
      dets.size(0),
      " and ",
      scores.size(0))

  c10::DeviceGuard device_guard(dets.device());

  if (dets.numel() == 0) {
    return at::empty({0}, dets.options().dtype(at::kLong));
  }

  constexpr int nms_items_per_group = sizeof(unsigned long long) * 8;

  auto order_t = std::get<1>(
      scores.sort(/*stable=*/true, /*dim=*/0, /* descending=*/true));
  auto dets_sorted = dets.index_select(0, order_t).contiguous();

  int dets_num = dets.size(0);
  int col_blocks = (dets_num + nms_items_per_group - 1) / nms_items_per_group;

  auto mask = nms_kernel(dets_sorted, iou_threshold);

  at::Tensor mask_cpu = mask.to(at::kCPU);
  unsigned long long* mask_host = (unsigned long long*)mask_cpu.data_ptr();

  std::vector<unsigned long long> remv(col_blocks);
  memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks);

  at::Tensor keep =
      at::empty({dets_num}, dets.options().dtype(at::kLong).device(at::kCPU));
  int64_t* keep_out = (int64_t*)keep.data_ptr();

  int num_to_keep = 0;
  for (int i = 0; i < dets_num; i++) {
    int nblock = i / nms_items_per_group;
    int inblock = i % nms_items_per_group;

    if (!(remv[nblock] & (1ULL << inblock))) {
      keep_out[num_to_keep++] = i;
      unsigned long long* p = mask_host + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv[j] |= p[j];
      }
    }
  }

  return order_t.index(
      {keep.narrow(/*dim=*/0, /*start=*/0, /*length=*/num_to_keep)
           .to(order_t.device(), keep.scalar_type())});
}

TORCH_LIBRARY_IMPL(torchvision, XPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("torchvision::nms"), TORCH_FN(nms));
}

} // namespace at::native::xpu
