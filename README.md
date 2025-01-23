# Torch XPU Operators*
===========================

Torch XPU Operators* implements PyTorch ATen operators for Intel GPU devices, aiming to agilely support PyTroch ATen operations and buffer these operations for Intel GPU upstreaming . For more details, refer to [SYCL kernels for ATen Operators RFC](https://github.com/pytorch/pytorch/issues/114835) for more details.

## 1. Overview

 <p align="center">
     <img src="docs/torch_xpu_ops.jpg" width="100%">
 </p>

 * SYCL Implementation for XPU Operators: The Operators in this staging branch will finally be upstreamed to PyTorch for Intel GPU.

## 2. Requirements

#### Hardware Requirements

Verified Hardware Platforms:

* Intel® Data Center GPU Max Series, Driver Version: [803](https://dgpu-docs.intel.com/releases/LTS_803.29_20240131.html)

#### Software Requirements

* Ubuntu 22.04, SUSE Linux Enterprise Server(SLES) 15 SP4
  * Intel® Data Center GPU Max Series
* Intel® oneAPI Base Toolkit 2024.0

#### Install Intel GPU Drivers

|OS|Intel GPU|Install Intel GPU Driver|
|-|-|-|
|Ubuntu 22.04, SLES 15 SP4|Intel® Data Center GPU Max Series|  Refer to the [Installation Guides](https://dgpu-docs.intel.com/installation-guides/index.html#intel-data-center-gpu-max-series) for latest driver installation. If install the verified Intel® Data Center GPU Max Series/Intel® Data Center GPU Flex Series [803](https://dgpu-docs.intel.com/releases/LTS_803.29_20240131.html), please append the specific version after components, such as `sudo apt-get install intel-opencl-icd==23.43.27642.38-803~22.04`|

#### Install oneAPI Base Toolkit Packages

Need to install components of Intel® oneAPI Base Toolkit:

* Intel® oneAPI DPC++ Compiler
* Intel® oneAPI Math Kernel Library (oneMKL)
* Intel® oneAPI Threading Building Blocks (TBB), dependency of DPC++ Compiler.

```bash
wget https://registrationcenter-download.intel.com/akdlm//IRC_NAS/20f4e6a1-6b0b-4752-b8c1-e5eacba10e01/l_BaseKit_p_2024.0.0.49564.sh
# 2 components are necessary: DPC++/C++ Compiler and oneMKL
sudo sh l_BaseKit_p_2024.0.0.49564.sh

# Source OneAPI environment
source /opt/intel/oneapi/compiler/2024.0/env/vars.sh
```


## 3. Build

Need to built this project as a submodule of PyTorch, after install Intel GPU Driver and oneAPI Base Toolkit Packages.

```bash
# Setup PyTorch source project. torch-xpu-ops is included by default.
python setup.py install
```

## 4. Security
See Intel's [Security Center](https://www.intel.com/content/www/us/en/security-center/default.html) for information on how to report a potential security issue or vulnerability.

See also: [Security Policy](SECURITY.md)

## 5. License
[Apache License 2.0](LICENSE)