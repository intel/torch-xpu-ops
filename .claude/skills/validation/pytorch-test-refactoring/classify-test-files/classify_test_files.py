#!/usr/bin/env python3
"""Classify PyTorch test files as device_unrelated, device_agnostic, or device_specific.

Classification is hierarchical:
    1. device_specific (any match wins)
    2. device_agnostic (any match, unless already device_specific)
    3. device_unrelated (default)

Output:
    agent_space/classified_test_files/
        device_unrelated/   -> symlinks
        device_agnostic/    -> symlinks
        device_specific/    -> symlinks
        _report.json        -> summary + per-file classification
"""

import json
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TEST_DIR = REPO_ROOT / "test"
OUTPUT_DIR = REPO_ROOT / "agent_space" / "classified_test_files"

DEVICE_SPECIFIC_FILENAMES = {
    "test_cuda.py",
    "test_mps.py",
    "test_xpu.py",
    "test_rocm.py",
    "test_mtia.py",
    "test_tpu.py",
    "test_lazy.py",
}

DEVICE_SPECIFIC_IMPORTS = re.compile(
    r"(?:from\s+torch\.testing\._internal\.common_(?:cuda|mps|xpu|rocm|mtia)\s+import.*\b(TEST_CUDA|TEST_MPS|TEST_XPU|TEST_ROCM|TEST_MTIA)\b"
    r"|import\s+torch\.testing\._internal\.common_(?:cuda|mps|xpu|rocm|mtia)\b)"
)

DEVICE_SPECIFIC_APIS = re.compile(r"\btorch\.(cuda|mps|xpu)\.")

DEVICE_AGNOSTIC_IMPORTS = re.compile(
    r"from\s+torch\.testing\._internal\.common_device_type\s+import.*\binstantiate_device_type_tests\b"
    r"|from\s+torch\.testing\._internal\.common_methods_invocations\s+import"
    r"|from\s+torch\.testing\._internal\.common_(?:dtype|device_type)\s+import"
)

DEVICE_AGNOSTIC_CALL = re.compile(r"\binstantiate_device_type_tests\s*\(")

DEVICE_AGNOSTIC_DECORATORS = re.compile(
    r"@(?:ops|dtypes|dtypesIfCPU|dtypesIfCUDA|dtypesIfMPS|dtypesIfXPU)\s*\("
)

DEVICE_AGNOSTIC_ACCELERATOR = re.compile(
    r"\b(TEST_PRIVATEUSE1|TEST_ACCELERATOR)\b"
    r"|torch\.accelerator\."
)


def read_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def is_device_specific(filepath: Path, content: str) -> bool:
    if filepath.name in DEVICE_SPECIFIC_FILENAMES:
        return True

    if DEVICE_SPECIFIC_IMPORTS.search(content):
        return True

    if DEVICE_SPECIFIC_APIS.search(content):
        return True

    return False


def is_device_agnostic(content: str) -> bool:
    if DEVICE_AGNOSTIC_IMPORTS.search(content):
        return True

    if DEVICE_AGNOSTIC_CALL.search(content):
        return True

    if DEVICE_AGNOSTIC_DECORATORS.search(content):
        return True

    if DEVICE_AGNOSTIC_ACCELERATOR.search(content):
        return True

    return False


def classify_file(filepath: Path) -> str:
    content = read_file(filepath)

    if is_device_specific(filepath, content):
        return "device_specific"

    if is_device_agnostic(content):
        return "device_agnostic"

    return "device_unrelated"


def collect_test_files(test_dir: Path) -> list[Path]:
    seen = set()
    files = []
    for p in sorted(test_dir.glob("*.py")):
        if p.is_file() and p.name != "__init__.py" and p.name not in seen:
            seen.add(p.name)
            files.append(p)
    return files


def build_symlink_tree(classified: dict[str, list[Path]]):
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for category in ("device_unrelated", "device_agnostic", "device_specific"):
        cat_dir = OUTPUT_DIR / category
        cat_dir.mkdir(exist_ok=True)
        for f in classified.get(category, []):
            symlink = cat_dir / f.name
            symlink.unlink(missing_ok=True)
            symlink.symlink_to(os.path.relpath(f, cat_dir))


def write_report(classified: dict[str, list[Path]], total: int):
    report = {
        "summary": {
            "total": total,
            "device_unrelated": len(classified.get("device_unrelated", [])),
            "device_agnostic": len(classified.get("device_agnostic", [])),
            "device_specific": len(classified.get("device_specific", [])),
        },
        "files": {
            category: [str(f.relative_to(REPO_ROOT)) for f in files]
            for category, files in classified.items()
        },
    }
    report_path = OUTPUT_DIR / "_report.json"
    report_path.write_text(json.dumps(report, indent=2))


def main():
    test_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else TEST_DIR
    if not test_dir.is_dir():
        print(f"Not a directory: {test_dir}", file=sys.stderr)
        sys.exit(1)

    files = collect_test_files(test_dir)
    classified = defaultdict(list)

    for f in files:
        category = classify_file(f)
        classified[category].append(f)
        print(f"  [{category}] {f.relative_to(REPO_ROOT)}")

    build_symlink_tree(classified)
    write_report(classified, len(files))

    print(f"\nClassified {len(files)} files:")
    for cat in ("device_unrelated", "device_agnostic", "device_specific"):
        count = len(classified.get(cat, []))
        print(f"  {cat}: {count}")
    print(f"\nSymlinks written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
