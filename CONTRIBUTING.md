# Contributing

## License

Torch XPU Operators project is licensed under the terms in [LICENSE]](./LICENSE). By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms.

## Contributing to Torch XPU Operators project

Thank you for your interest in contributing to this project. Before you begin writing code, it is important that you share your intention to contribute with the team, based on the type of contribution:

1. You want to propose a new feature and implement it.
    - Post about your intended feature in a [GitHub issue](https://github.com/intel/torch-xpu-ops/issues), and we shall discuss the design and implementation. Once we agree that the plan looks good, go ahead and implement it.
2. You want to implement a feature or bug-fix for an outstanding issue.
    - Search for your issue in the [GitHub issue list](https://github.com/intel/torch-xpu-ops/issues).
    - Pick an issue and comment that you'd like to work on the feature or bug-fix.
    - If you need more context on a particular issue, ask and we shall provide.

Once you implement and test your feature or bug-fix, submit a Pull Request to https://github.com/intel/torch-xpu-ops.


## Developing Torch XPU Operators project on XPU


To develop on your machine, here are some tips:

### Build
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
