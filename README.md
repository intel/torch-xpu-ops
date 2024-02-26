torch-xpu-ops*
===========================

torch-xpu-ops is an `xpu` implementation of PyTorch ATen operators.

## Build
* Standalone - Require pre-installation of PyTorch
```bash
mkdir build
cd build && cmake -DTORCH_XPU_OPS_BUILD_MODE=standalone -DBUILD_TEST=1 -DPYTORCH_INSTALL_DIR=YOUR_PYTORCH_INSTALLATION_DIR ..
make -j x
```
* Submodule - Build as a submodule of PyTorch
```bash
# Setup PyTorch source project. torch-xpu-ops is included by default.
python setup.py install
```
