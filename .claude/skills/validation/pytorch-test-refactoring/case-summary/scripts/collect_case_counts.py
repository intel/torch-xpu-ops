#!/usr/bin/env python3
"""Run `pytest --collect-only` per test file and count cases per test class.

Reads the file list from the merged xlsx, runs collection from a neutral CWD
(so the installed torch resolves, not the source checkout), parses collected
node IDs, strips per-device class suffixes (CPU/CUDA/XPU/Meta/HPU/MPS/...) to
recover the generic class name stored in the xlsx, and aggregates a case count
per (file_relpath, generic_class).

Output: TSV with columns file, class, case_count. Files that error or time out
during collection are logged to a separate errors TSV (still emitted so the
downstream merge leaves their case_count blank).
"""

import argparse
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import openpyxl

DEVICE_SUFFIX_RE = re.compile(r"(CPU|CUDA|XPU|Meta|HPU|MPS|MTIA|Lazy|PrivateUse1)$")
NODEID_RE = re.compile(r"^(?P<file>[^:]+\.py)::(?P<class>[A-Za-z_]\w*)::")


def generic_class(name: str) -> str:
    prev = None
    while prev != name:
        prev = name
        name = DEVICE_SUFFIX_RE.sub("", name)
    return name


def load_files(xlsx: Path) -> list[str]:
    wb = openpyxl.load_workbook(xlsx, read_only=True)
    ws = wb.active
    files = []
    seen = set()
    for row in ws.iter_rows(min_row=2, max_col=1, values_only=True):
        f = row[0]
        if f and f not in seen:
            seen.add(f)
            files.append(str(f))
    wb.close()
    return files


def collect_file(
    test_root: Path, rel: str, timeout: int
) -> tuple[dict[str, int] | None, str]:
    abspath = test_root / rel
    if not abspath.exists():
        return None, "missing"
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", str(abspath), "--collect-only", "-q",
             "-p", "no:cacheprovider"],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd="/tmp",
        )
    except subprocess.TimeoutExpired:
        return None, "timeout"
    counts: dict[str, int] = defaultdict(int)
    for line in proc.stdout.splitlines():
        m = NODEID_RE.match(line.strip())
        if not m:
            continue
        counts[generic_class(m.group("class"))] += 1
    if not counts:
        # Distinguish a clean zero from a collection failure.
        status = "ok-empty" if proc.returncode == 0 else f"error(rc={proc.returncode})"
        return (dict(counts) if proc.returncode == 0 else None), status
    return dict(counts), "ok"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xlsx", default="agent_space_xpu/test_class_classification_merged.xlsx")
    ap.add_argument("--test-root", default=str(Path.home() / "daisyden/upstream/test"))
    ap.add_argument("--out", default="agent_space_xpu/case_counts.tsv")
    ap.add_argument("--errors", default="agent_space_xpu/case_counts_errors.tsv")
    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args()

    test_root = Path(args.test_root).resolve()
    files = load_files(Path(args.xlsx))
    print(f"{len(files)} unique files to collect from {test_root}", flush=True)

    out_rows: list[tuple[str, str, int]] = []
    err_rows: list[tuple[str, str]] = []
    for i, rel in enumerate(files, 1):
        counts, status = collect_file(test_root, rel, args.timeout)
        if counts is None:
            err_rows.append((rel, status))
        else:
            for cls, n in counts.items():
                out_rows.append((rel, cls, n))
        if i % 50 == 0 or i == len(files):
            print(f"  {i}/{len(files)} files, {len(out_rows)} class rows, "
                  f"{len(err_rows)} errors", flush=True)

    with open(args.out, "w") as f:
        f.write("file\tclass\tcase_count\n")
        for rel, cls, n in out_rows:
            f.write(f"{rel}\t{cls}\t{n}\n")
    with open(args.errors, "w") as f:
        f.write("file\tstatus\n")
        for rel, st in err_rows:
            f.write(f"{rel}\t{st}\n")

    print(f"\nWrote {len(out_rows)} rows to {args.out}")
    print(f"Wrote {len(err_rows)} error files to {args.errors}")


if __name__ == "__main__":
    main()
