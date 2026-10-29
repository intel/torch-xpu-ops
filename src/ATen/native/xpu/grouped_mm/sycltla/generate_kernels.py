# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Generate per-(kernel, dtype) explicit instantiation files for the XPU
SYCL-TLA grouped GEMM so template compilation is spread over translation
units (same scheme as flash_attn/sycltla/generate_kernels.py).

Usage:
    python generate_kernels.py [-o OUTPUT_DIR]
"""

import argparse
from pathlib import Path

DTYPES = {"bf16": "DType::BF16", "fp16": "DType::FP16"}

PRELUDE = """\
/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

// Explicit instantiations split across files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py".

#include <ATen/native/xpu/grouped_mm/sycltla/grouped_mm_kernel_impl.h>

namespace sycltla {
namespace grouped_gemm {

"""

EPILOGUE = """\

} // namespace grouped_gemm
} // namespace sycltla
"""


def m_grouped(dtype_enum: str) -> str:
    return "\n".join(
        f"template void run_m_grouped<{dtype_enum}, '{layout}'>("
        "sycl::queue&, const Params&);"
        for layout in ("R", "C")
    )


def k_grouped(dtype_enum: str) -> str:
    return f"template void run_k_grouped<{dtype_enum}>(sycl::queue&, const Params&);"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_dir", default=None)
    args = parser.parse_args()
    out = Path(args.output_dir) if args.output_dir else Path(__file__).parent
    for name, enum in DTYPES.items():
        (out / f"grouped_mm_m_{name}.cpp").write_text(PRELUDE + m_grouped(enum) + EPILOGUE)
        (out / f"grouped_mm_k_{name}.cpp").write_text(PRELUDE + k_grouped(enum) + EPILOGUE)


if __name__ == "__main__":
    main()
