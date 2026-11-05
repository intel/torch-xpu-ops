# torch-xpu-ops Development Override

PyTorch pins torch-xpu-ops to a specific commit via `third_party/xpu.txt`.
CMake reads this file on every build and **unconditionally** runs `git fetch` +
`git checkout <pin>` on `third_party/torch-xpu-ops` — overwriting any local changes.

To protect local changes, update the pin to your HEAD **before** rebuilding so
CMake's checkout becomes a no-op:

```bash
# 1. Clone your fork into the path CMake expects (if not already there)
cd <pytorch_root>/third_party
git clone <your-fork-url> torch-xpu-ops
cd torch-xpu-ops
git checkout <your-pr-branch>

# 2. Update the pin so CMake checks out your HEAD (no-op)
git rev-parse HEAD > <pytorch_root>/third_party/xpu.txt
```

Do not commit `xpu.txt` — this is a local-only override.

After verification:
1. Submit PR to `intel/torch-xpu-ops`
2. After merge, update the pin in `pytorch/pytorch`
