#!/bin/bash
SRC_ROOT="$(realpath "$(dirname "$0")"/../..)"
# python fixheaders.py --force --root "$SRC_ROOT" --dry-run
python fixheaders.py --force --root "$SRC_ROOT"