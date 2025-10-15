import argparse
import os
import re
import shutil
from pathlib import Path

VERBOSE = False

parser = argparse.ArgumentParser(description="Utils for append ops headers")
parser.add_argument(
    "--src-header-dir", type=str, help="torch-xpu-ops build header file path"
)
parser.add_argument("--dst-header-dir", type=str, help="torch build header file path")
parser.add_argument(
    "--dry-run", action="store_true", help="run without writing any files"
)
args = parser.parse_args()


def append_xpu_function_header(src, dst):
    r"""
    Cleans trailing empty lines from the destination file, then appends #include
    lines from the source file that match `#include <ATen/ops/...` to the destination.
    Only modifies file if changes are actually needed.
    """
    if args.dry_run:
        return

    try:
        with open(src, encoding="utf-8") as fr:
            src_text = fr.read()
    except OSError as e:
        if VERBOSE:
            print(f"Warning: Could not read source file {src}: {e}")
        return

    pattern = r"^#include <ATen/ops/.*>\s*\r?\n"
    matches = re.findall(pattern, src_text, re.MULTILINE)
    if not matches:
        return

    try:
        with open(dst, encoding="utf-8") as f:
            dst_lines = f.readlines()
            dst_text = "".join(dst_lines)
    except OSError as e:
        if VERBOSE:
            print(f"Warning: Could not read destination file {dst}: {e}")
        return

    missing_headers = [match for match in matches if match not in dst_text]
    if not missing_headers:
        return

    new_dst_lines = dst_lines.copy()

    while new_dst_lines and not new_dst_lines[-1].strip():
        new_dst_lines.pop()
    new_dst_lines.extend(missing_headers)

    new_content = "".join(new_dst_lines)
    old_content = "".join(dst_lines)

    if new_content != old_content:
        try:
            with open(dst, "w", encoding="utf-8") as f:
                f.writelines(new_dst_lines)
        except OSError as e:
            if VERBOSE:
                print(f"Error: Could not write to {dst}: {e}")


def parse_ops_headers(src):
    r"""
    Parse ops headers from file, extracting header filenames from ATen/ops pattern.
    """
    with open(src, encoding="utf-8") as fr:
        src_text = fr.read()
    pattern = r".*/ATen/+ops/(.*.h)"
    ops_headers = re.findall(pattern, src_text, re.MULTILINE)
    return ops_headers


def classify_ops_headers(src_dir, dst_dir):
    r"""
    Classify ops headers into common headers and XPU-specific ops headers.
    """
    src_ops_headers = parse_ops_headers(os.path.join(src_dir, "ops_generated_headers.cmake"))
    dst_ops_headers = parse_ops_headers(os.path.join(dst_dir, "ops_generated_headers.cmake"))

    # Convert to sets for efficient set operations
    src_set = set(src_ops_headers)
    dst_set = set(dst_ops_headers)
    common_headers = sorted(src_set & dst_set)
    xpu_ops_headers = sorted(src_set - dst_set)
    return common_headers, xpu_ops_headers


def generate_xpu_ops_headers_cmake(src_dir, dst_dir, xpu_ops_headers):
    r"""
    Generate XPU ops headers xpu_ops_generated_headers.cmake only if content changes
    """
    output_file = os.path.join(src_dir, "xpu_ops_generated_headers.cmake")

    # Generate new content
    new_content = "set(xpu_ops_generated_headers\n"
    for header in xpu_ops_headers:
        new_content += f'    "{Path(os.path.join(dst_dir, header)).as_posix()}"\n'
    new_content += ")\n"

    # Check if file exists and has same content
    should_write = True
    if os.path.exists(output_file):
        try:
            with open(output_file, encoding="utf-8") as f:
                existing_content = f.read()
            should_write = existing_content != new_content
        except OSError:
            # If we can't read the file, write it anyway
            should_write = True

    if should_write:
        with open(output_file, "w", encoding="utf-8") as fw:
            fw.write(new_content)


def append_xpu_ops_headers(src_dir, dst_dir, common_headers, xpu_ops_headers):
    r"""
    For XPU-specific ops headers, copy them to destination build and append XPU declarations to common headers.
    Copies and appends are done only if leading to file changes to prevent unnecessary recompilations.
    """
    if args.dry_run:
        return

    for f in xpu_ops_headers:
        # TODO: Fix the incorrect op info registered in native_functions.yaml
        # assert "xpu" in f, f"Error: The function signature or namespace in '{f}' is incorrect. Expected 'xpu' to be present."
        src = os.path.join(src_dir, f)
        dst = os.path.join(dst_dir, f)
        # Only copy if src and dst differ or dst does not exist
        should_copy = True
        if os.path.exists(dst):
            try:
                with open(src, "rb") as fsrc, open(dst, "rb") as fdst:
                    should_copy = fsrc.read() != fdst.read()
            except OSError:
                should_copy = True
        if should_copy:
            shutil.copy(src, dst)

    for f in common_headers:
        src = os.path.join(src_dir, f)
        dst = os.path.join(dst_dir, f)
        with open(src, encoding="utf-8") as fr:
            src_text = fr.read()

        pattern = r"^TORCH_API.*xpu.*?;\s*\r?\n"
        xpu_declarations = re.findall(pattern, src_text, re.MULTILINE)
        pattern = r"struct TORCH_XPU_API.*xpu.*?{.*?};\s*\r?\n"
        xpu_declarations.extend(re.findall(pattern, src_text, re.DOTALL))

        if not xpu_declarations:
            continue

        with open(dst, "r+", encoding="utf-8") as f:
            dst_lines = f.readlines()
            dst_text = "".join(dst_lines)
            old_content = "".join(dst_lines)
            missing_declarations = []
            insertion_index = None
            for index, line in enumerate(dst_lines):
                if re.match(r"^(TORCH_API.*;|struct TORCH_API.*)", line):
                    insertion_index = index
                    # Check if any XPU declarations are missing
                    for xpu_declaration in xpu_declarations:
                        if xpu_declaration not in dst_text:
                            missing_declarations.append(xpu_declaration)
                    # Insert XPU declarations before the first TORCH_API declaration
                    if missing_declarations:
                        dst_lines[index:index] = missing_declarations
                    break
            assert (insertion_index is not None), f"Error: No TORCH_API declaration found in {dst}."

            if old_content != "".join(dst_lines):
                f.seek(0)
                f.writelines(dst_lines)
                f.truncate()


def main():
    src_xpu_function_header = os.path.join(args.src_header_dir, "XPUFunctions_inl.h")
    dst_xpu_function_header = os.path.join(args.dst_header_dir, "XPUFunctions_inl.h")
    append_xpu_function_header(src_xpu_function_header, dst_xpu_function_header)

    src_xpu_ops_header_dir = os.path.join(args.src_header_dir, "ops")
    dst_xpu_ops_header_dir = os.path.join(args.dst_header_dir, "ops")
    common_headers, xpu_ops_headers = classify_ops_headers(
        args.src_header_dir, args.dst_header_dir
    )
    generate_xpu_ops_headers_cmake(args.src_header_dir, dst_xpu_ops_header_dir, xpu_ops_headers)
    append_xpu_ops_headers(
        src_xpu_ops_header_dir, dst_xpu_ops_header_dir, common_headers, xpu_ops_headers
    )


if __name__ == "__main__":
    main()
