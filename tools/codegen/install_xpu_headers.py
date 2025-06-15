import argparse
import os
import re
import shutil
from pathlib import Path


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
    Append XPU function header XPUFunctions_inl.h from source to destination build.
    """
    if args.dry_run:
        return

    with open(dst) as fr:
        lines = fr.readlines()
    while lines and lines[-1].strip() == "":
        lines.pop()
    with open(dst, "w") as fw:
        fw.writelines(lines)

    with open(src) as fr, open(dst, "a") as fa:
        src_lines = fr.readlines()
        for line in src_lines:
            if re.match(r"^#include <ATen/ops/.*", line):
                fa.write(line)


def parse_ops_headers(src):
    r"""
    Parse ops headers from file.
    """
    ops_headers = []
    with open(src) as fr:
        src_text = fr.read()
        ops_headers.extend(re.findall(r".*/ATen/+ops/(.*.h)", src_text))
    return ops_headers


def classify_ops_headers(src_dir, dst_dir):
    r"""
    Classify ops headers into common headers and XPU-specific ops headers.
    """
    src_ops_headers = parse_ops_headers(os.path.join(src_dir, "ops_generated_headers.cmake"))
    dst_ops_headers = parse_ops_headers(os.path.join(dst_dir, "ops_generated_headers.cmake"))
    common_headers = [f for f in src_ops_headers if f in dst_ops_headers]
    xpu_ops_headers = [f for f in src_ops_headers if f not in common_headers]
    return common_headers, xpu_ops_headers


def generate_xpu_ops_headers_cmake(src_dir, dst_dir, xpu_ops_headers):
    r"""
    Generate XPU ops headers xpu_ops_generated_headers.cmake
    """
    with open(os.path.join(src_dir, "xpu_ops_generated_headers.cmake"), "w") as fw:
        fw.write("set(xpu_ops_generated_headers\n")
        for header in xpu_ops_headers:
            fw.write(f'    "{Path(os.path.join(dst_dir, header)).as_posix()}"\n')
        fw.write(")\n")


def append_xpu_ops_headers(src_dir, dst_dir, common_headers, xpu_ops_headers):
    r"""
    For XPU-specific ops headers, copy them to destination build and append XPU declarations to common headers.
    """
    if args.dry_run:
        return

    for f in xpu_ops_headers:
        # TODO: fix the incorrect op info registered in native_functions.yaml
        # assert "xpu" in f, f"Error: The function signature or namespace in '{f}' is incorrect. Expected 'xpu' to be present."
        src = os.path.join(src_dir, f)
        dst = os.path.join(dst_dir, f)
        shutil.copy(src, dst)

    for f in common_headers:
        src = os.path.join(src_dir, f)
        dst = os.path.join(dst_dir, f)
        xpu_declarations = []
        with open(src) as fr:
            src_text = fr.read()
            xpu_declarations.extend(
                re.findall(r"^TORCH_API.*xpu.*?;\n", src_text, re.MULTILINE)
            )
            xpu_declarations.extend(
                re.findall(r"struct TORCH_XPU_API.*xpu.*?{.*?};\n", src_text, re.DOTALL)
            )

        if not xpu_declarations:
            continue

        with open(dst, "r+") as f:
            dst_lines = f.readlines()
            dst_text = "".join(dst_lines)
            for index, line in enumerate(dst_lines):
                if re.match(r"^(TORCH_API.*;|struct TORCH_API.*)", line):
                    for xpu_declaration in xpu_declarations:
                        if not re.search(re.escape(xpu_declaration), dst_text):
                            dst_lines.insert(index, xpu_declaration)
                    break

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
