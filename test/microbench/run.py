# Copyright 2020-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

import json
import sys

from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.resolve()))

import core.cli as cli
import core.runner as runner


def resolve_dtype_in_config(config):
    """Convert dtype string like 'torch.float32' → torch.float32 in-place."""
    if "dtype" in config and isinstance(config["dtype"], str):
        name = config["dtype"].replace("torch.", "")
        mapping = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        if name in mapping:
            config["dtype"] = mapping[name]
        else:
            raise ValueError(
                f"Unknown dtype: {config['dtype']}. Use torch.float32 etc."
            )
    return config


def main():
    args = cli.parse_args()
    op_mod = cli.load_op_module(args.op)

    if args.case:
        # 🔁 Single-case mode
        try:
            config = json.loads(args.case)
            config = resolve_dtype_in_config(config)
        except Exception as e:
            print(f"❌ Config error: {e}")
            sys.exit(1)
        print(f"[Single-case] OP: {args.op}\nconfig: {config}")
        runner.run_case(op_mod.run_op, config, args)
    else:
        # 🧪 Full mode
        print(f"[Full] OP: {args.op}")
        for cfg in op_mod.get_default_cases():
            cfg = resolve_dtype_in_config(cfg.copy())
            print(f"config: {cfg}")
            runner.run_case(op_mod.run_op, cfg, args)


if __name__ == "__main__":
    main()
