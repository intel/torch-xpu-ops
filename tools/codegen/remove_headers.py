import argparse
import re

parser = argparse.ArgumentParser(description="Utils for remove unused headers")
parser.add_argument("--register_xpu_path", type=str, help="file location of RegisterXPU.cpp")
args = parser.parse_args()

def rm_as_strided_native():
    with open(args.register_xpu_path, 'r') as fr:
        lines = fr.readlines()

        with open(args.register_xpu_path, 'w') as fw:
            for ln in lines:
                if "#include <ATen/ops/as_strided_native.h>" not in ln:
                    fw.write(ln)

def replace_op_headers():
    with open(args.register_xpu_path, 'r') as fr:
        lines = fr.readlines()
        patt = r'#include <ATen/ops'
        rep = r'#include <xpu/ATen/ops'
        with open(args.register_xpu_path, 'w') as fw:
            for ln in lines:
                if 'empty.h' in ln:
                    continue
                replaced = re.sub(patt, rep, ln)
                fw.write(replaced)

if __name__ == "__main__":
    # rm_as_strided_native()
    replace_op_headers()