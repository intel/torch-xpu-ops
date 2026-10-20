#!/usr/bin/env python3
"""Split the upstream test file list in ``ut.txt`` into buckets.

- distributed: test/distributed/*
- inductor:    test/inductor/*
- default:     everything except test/distributed/*
"""

import argparse
import os
import re
import sys

DISTRIBUTED_PREFIX = "test/distributed/"
INDUCTOR_PREFIX = "test/inductor/"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
LIST_FILE = os.path.join(REPO_ROOT, "ut.txt")


def read_files(path):
    files = []
    seen = set()
    with open(path) as fh:
        for line in fh:
            entry = re.sub(r"\s+", "", line)
            if not entry or entry.startswith("#") or not entry.endswith(".py"):
                continue
            if entry not in seen:
                seen.add(entry)
                files.append(entry)
    return files


def split_files(files):
    distributed = [f for f in files if f.startswith(DISTRIBUTED_PREFIX)]
    inductor = [f for f in files if f.startswith(INDUCTOR_PREFIX)]
    default = [f for f in files if not f.startswith(DISTRIBUTED_PREFIX)]
    return distributed, inductor, default


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category",
                        choices=["distributed", "inductor", "default", "all"],
                        default="all", help="which bucket of files to print")
    parser.add_argument("--distributed-output",
                        help="write distributed file list to this path")
    parser.add_argument("--inductor-output",
                        help="write inductor file list to this path")
    parser.add_argument("--default-output",
                        help="write default file list to this path")
    parser.add_argument("--list-file", default=LIST_FILE,
                        help="plain list file of upstream test files")
    args = parser.parse_args()

    files = read_files(args.list_file)
    distributed, inductor, default = split_files(files)

    if args.distributed_output:
        with open(args.distributed_output, "w") as fh:
            fh.write("\n".join(distributed) + "\n" if distributed else "")
    if args.inductor_output:
        with open(args.inductor_output, "w") as fh:
            fh.write("\n".join(inductor) + "\n" if inductor else "")
    if args.default_output:
        with open(args.default_output, "w") as fh:
            fh.write("\n".join(default) + "\n" if default else "")

    print(f"Test files: distributed: {len(distributed)}, "
          f"inductor: {len(inductor)}, "
          f"default: {len(default)} (from {args.list_file})",
          file=sys.stderr)

    if args.category == "distributed":
        selected = distributed
    elif args.category == "inductor":
        selected = inductor
    elif args.category == "default":
        selected = default
    else:
        selected = files
    print("\n".join(selected))
    return 0


if __name__ == "__main__":
    sys.exit(main())
