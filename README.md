# Torch XPU Operators*

Torch XPU Operators* implements PyTorch ATen operators for Intel GPU devices, aiming to agilely support PyTorch ATen operations and buffer these operations for Intel GPU upstreaming . For more details, refer to [SYCL kernels for ATen Operators RFC](https://github.com/pytorch/pytorch/issues/114835) for more details.

## Overview

 <p align="center">
     <img src="docs/torch_xpu_ops.jpg" width="100%">
 </p>

 * SYCL Implementation for XPU Operators: The Operators in this staging branch will finally be upstreamed to PyTorch for Intel GPU.

## Requirements

For the hardware and software prerequiste, please refer to [PyTorch Prerequisites for Intel GPUs](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpus.html) to check your hardware and install the following packages.

* Intel GPU Driver: Install Intel GPU drivers along with compute and media runtimes and development packages.
* Intel® Deep Learning Essentials: Install a subset of Intel® oneAPI components needed for building and running PyTorch.

## Build

Need to built this project as a submodule of PyTorch, after install Intel GPU Driver and Intel Deep Learning Essentials.

```bash
# Setup PyTorch source project. torch-xpu-ops is included by default.
python setup.py install
```

## Security
See Intel's [Security Center](https://www.intel.com/content/www/us/en/security-center/default.html) for information on how to report a potential security issue or vulnerability.

See also: [Security Policy](SECURITY.md)

## License
[Apache License 2.0](LICENSE)
