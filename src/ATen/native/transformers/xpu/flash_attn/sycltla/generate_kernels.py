# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Generate per-headdim kernel instantiation files for XPU flash attention.

Each file provides explicit template specializations for all dtype x causal
combinations at a given head dimension. This splits heavy template
instantiation across compilation units to speed up parallel builds.

Usage:
    python generate_kernels.py [-o OUTPUT_DIR]

If no output directory is specified, files are written to the script's directory.
"""

import argparse
from pathlib import Path
from typing import Optional

HEAD_DIMENSIONS = [32, 64, 96, 128, 192, 256]

DTYPE_MAP = {
    "fp16": "cute::half_t",
    "bf16": "cute::bfloat16_t",
}

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

// Splitting the different head dimensions to different files to speed up
// compilation. This file is auto-generated. See "generate_kernels.py"
"""

FWD_TEMPLATE = """\
{prelude}
#include <sycltla/mha_fwd_launch.h>

namespace cute {{

{specializations}

}} // namespace cute
"""

BWD_TEMPLATE = """\
{prelude}
#include <sycltla/mha_bwd_launch.h>

namespace cute {{

{specializations}

}} // namespace cute
"""

FWD_SPECIALIZATION = """\
template <>
void run_mha_fwd_<{dtype}, {hdim}, {causal}>(
    sycl::queue& queue,
    FLASH_FWD_params& params) {{
  run_mha_fwd_hdim{hdim}<{dtype}, {causal}>(queue, params);
}}"""

BWD_SPECIALIZATION = """\
template <>
void run_mha_bwd_<{dtype}, {hdim}, {causal}>(
    sycl::queue& queue,
    FLASH_BWD_params& params) {{
  run_mha_bwd_hdim{hdim}<{dtype}, {causal}>(queue, params);
}}"""


def generate_specializations(template: str, hdim: int) -> str:
    specs = []
    for dtype_cpp in DTYPE_MAP.values():
        for causal in ["false", "true"]:
            specs.append(template.format(dtype=dtype_cpp, hdim=hdim, causal=causal))
    return "\n\n".join(specs)


def generate_file(direction: str, hdim: int) -> tuple[str, str]:
    if direction == "fwd":
        body = FWD_TEMPLATE.format(
            prelude=PRELUDE,
            specializations=generate_specializations(FWD_SPECIALIZATION, hdim),
        )
    else:
        body = BWD_TEMPLATE.format(
            prelude=PRELUDE,
            specializations=generate_specializations(BWD_SPECIALIZATION, hdim),
        )
    filename = f"mha_{direction}_hdim{hdim}.cpp"
    return filename, body


def main(output_dir: Optional[str]) -> None:
    if output_dir is None:
        out = Path(__file__).parent
    else:
        out = Path(output_dir)

    for direction in ["fwd", "bwd"]:
        for hdim in HEAD_DIMENSIONS:
            filename, content = generate_file(direction, hdim)
            (out / filename).write_text(content)
            print(f"  {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate per-headdim kernel instantiation files for XPU flash attention",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        required=False,
        help="Output directory (defaults to script directory)",
    )
    args = parser.parse_args()
    main(args.output_dir)
