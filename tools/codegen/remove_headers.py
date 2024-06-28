import argparse

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


if __name__ == "__main__":
    rm_as_strided_native()