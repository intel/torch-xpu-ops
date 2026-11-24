# `src/ATen/native/xpu` review guide

- **SYCL kernels live under `sycl/`.** Actual SYCL device kernel code must not be added directly in this directory; it belongs in `src/ATen/native/xpu/sycl/`.
- **Default verdict:** Flag as a **Request Changes** finding if a PR violates the rule above.
