# core/cli.py
import argparse
import importlib
import json
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Unified OP Benchmark")
    parser.add_argument("--op", type=str, required=True, help="OP name (e.g., adaptive_avg_pool, cdist)")
    parser.add_argument("--device", default="xpu", choices=["cpu", "cuda", "xpu"])
    parser.add_argument("--num-iter", type=int, default=20)

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--profile-only", action="store_true")
    group.add_argument("--e2e-only", action="store_true")

    # ✅ Universal single-case trigger
    parser.add_argument(
        "--case",
        type=str,
        help='Single case JSON config, e.g.: \'{"shape":[8,512,32,32],"out":[7,7],"dtype":"bf16"}\''
    )
    return parser.parse_args()

def load_op_module(op_name):
    try:
        return importlib.import_module(f"ops.{op_name}")
    except ImportError as e:
        raise ValueError(f"OP '{op_name}' not found. Check ops/{op_name}.py exists.\nError: {e}")
