#!/usr/bin/env python3
"""Check torch XPU AOT effectiveness.

AOT: AOT kernels are loaded as native binaries; any JIT compile goes through the
driver and leaves an entry in the NEO cache. So: empty NEO cache after the run =>
AOT PASS. Elapsed time is shown for reference only.

Usage: python check_aot.py [N]
       N    = reference seconds (default 60)
Exit:  0 PASS, 1 FAIL
"""

import os
import platform
import sys
import tempfile
import time
from pathlib import Path

try:
    ELAPSED_REF = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
except ValueError:
    print(f"Usage: python {Path(sys.argv[0]).name} [N]")
    sys.exit(2)

# 1. Env and fresh cache dir (before torch import so the driver picks it up).
# NEO reads NEO_CACHE_* the same way on Linux and Windows; mkdtemp uses /tmp or %TEMP%.
NEO_DIR = Path(tempfile.mkdtemp(prefix="check_aot_")) / "neo"
NEO_DIR.mkdir()
os.environ.update({
    "NEO_CACHE_PERSISTENT": "1",
    "NEO_CACHE_DIR": str(NEO_DIR),
    # A warm SYCL program cache would bypass the driver and hide JIT.
    "SYCL_CACHE_PERSISTENT": "0",
})
os.environ.pop("NEO_CACHE_MAX_SIZE", None)

import torch  # noqa: E402


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3, padding=1)
        self.fc = torch.nn.Linear(4 * 8 * 8, 10)

    def forward(self, x):
        y = self.conv(x)
        y = (y + x.mean()) - 0.5
        y = y * y / (y.abs() + 1.0)
        return self.fc(torch.relu(y).flatten(1))


def check_aot():
    print("-" * 60 + "\nAOT Check\n" + "-" * 60)
    print(f"NEO cache : {NEO_DIR}\n")

    # 2. Prepare model and inputs on CPU.
    model, inputs = Model().eval(), torch.randn(2, 3, 8, 8)
    dtypes = {"fp32": torch.float32, "fp16": torch.float16}
    if torch.xpu.is_bf16_supported():
        dtypes["bf16"] = torch.bfloat16

    # 3. Run and time each dtype.
    total = 0.0
    try:
        for name, dtype in dtypes.items():
            start = time.perf_counter()
            with torch.no_grad():
                model.to("xpu", dtype)(inputs.to("xpu", dtype))
            torch.xpu.synchronize()
            elapsed = time.perf_counter() - start
            total += elapsed
            print(f"Run {name:<5} : {elapsed:8.2f} s")
    except Exception as e:
        msg = str(e).strip().splitlines()
        print(f"AOT       : ERROR - {type(e).__name__}: {msg[0] if msg else ''}")
        return 2

    # 4. Check NEO cache and conclude.
    files = [p for p in NEO_DIR.rglob("*") if p.is_file() and p.name != "config.file"]
    hint = "JIT likely" if total > ELAPSED_REF else "OK"
    print(f"\nElapsed   : {total:8.2f} s (ref {ELAPSED_REF:g} s, {hint}; reference only)")
    print(f"JIT files : {len(files)}")
    for f in files[:5]:
        print(f"            {f.relative_to(NEO_DIR).as_posix()}")
    passed = not files
    print(f"AOT       : {'PASS - AOT effective' if passed else 'FAIL - JIT occurred'}")
    return 0 if passed else 1


def main():
    print("=" * 60 + "\nXPU AOT Check\n" + "=" * 60)
    print(f"PyTorch   : {torch.__version__}")
    print(f"OS        : {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"AOT archs : {', '.join(torch.xpu.get_arch_list()) or 'none'}")
    if not torch.xpu.is_available():
        code = 1
        print("AOT       : FAIL - XPU not available")
    else:
        for k in range(torch.xpu.device_count()):
            print(f"Device {k}  : {torch.xpu.get_device_name(k)}")
        # AOT is a plain verdict: anything other than an empty NEO cache fails.
        code = min(check_aot(), 1)
    print("=" * 60)
    print(f"RESULT    : {('PASS', 'FAIL')[code]}")
    print("=" * 60)
    return code


if __name__ == "__main__":
    sys.exit(main())
