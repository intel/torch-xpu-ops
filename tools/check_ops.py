import re
import os
import glob
from addict import Dict
from pathlib import Path


onednn_keys = [
    'addmm',
    'addmm_',
    'addmm.out',
    '_addmm_activation',
    '_addmm_activation.out',
    'mm.out',
    'mm',
    'baddbmm.out',
    'baddbmm_',
    'baddbmm',
    'addbmm.out',
    'addbmm_',
    'addbmm',
    'bmm.out',
    'bmm',
    'addmv',
    'addmv_',
    'addmv.out',
    'tensordot.out',
    'convolution_overrideable',
    'convolution_backward_overrideable',
]
onednn_keys = set(onednn_keys)


def parse_keys(folder, backend, filename=None, startswith='m.impl("', pattern=r'm\.impl\("([^"]+)"', check=True):
    if filename is None:
        files = glob.glob(f'{folder}/**/Register{backend}_0.cpp', recursive=True)
        register_xxx_file = files[0]
    else:
        register_xxx_file = os.path.join(folder, filename)
    print(register_xxx_file)
    with open(register_xxx_file) as f:
        lines = f.readlines()
    if startswith is not None:
        lines = [l.strip() for l in lines if l.strip().startswith(startswith)]
    else:
        lines = [l.strip() for l in lines]
    # pattern = r'm\.impl\("([^"]+)"'
    keys = []
    for line in lines:
        match = re.search(pattern, line)
        if match is not None:
            keys.append(match.group(1))
    if check:
        assert len(lines) == len(keys)
    keys = set(keys)
    return keys


def find_pytorch_dir():
    path = Path(__file__).resolve()
    while path != path.root:
        if path.name == "pytorch":
            return str(path)
        path = path.parent
    return ''


if __name__ == '__main__':
    root_folder = find_pytorch_dir().strip()

    kcuda = Dict()
    kxpu = Dict()

    kcuda.basic_keys = parse_keys(root_folder + '/build', 'CUDA')
    kcuda.basic_keys = {item for item in kcuda.basic_keys if 'cudnn' not in item}
    kcuda.sparse_keys = parse_keys(root_folder + '/build', 'SparseCUDA')
    kcuda.sparse_csr_keys = parse_keys(root_folder + '/build', 'SparseCsrCUDA')
    kcuda.nested_tensor_keys = parse_keys(root_folder + '/build', 'NestedTensorCUDA')

    xpu_keys = parse_keys(root_folder + '/build/xpu', 'XPU')
    kxpu.basic_keys = (xpu_keys | onednn_keys) & kcuda.basic_keys
    kxpu.sparse_keys = parse_keys(root_folder + '/build/xpu', 'SparseXPU')
    kxpu.sparse_csr_keys = parse_keys(root_folder + '/build/xpu', 'SparseCsrXPU')
    kxpu.nested_tensor_keys = parse_keys(root_folder + '/build/xpu', 'NestedTensorXPU')

    print('============ CUDA ============')
    for key in kcuda.keys():
        print(f"{key}: {len(kcuda[key])}")
    print('============ XPU ============')
    for key in kxpu.keys():
        print(f"{key}: {len(kxpu[key])}")

    kcuda_kxpu = Dict()
    for cudakeys in kcuda.keys():
        kcuda_kxpu[cudakeys] = kcuda[cudakeys] - kxpu[cudakeys]
    print(' ')

    for key in kcuda_kxpu.keys():
        print(f"============ {key} ============")
        values = sorted(kcuda_kxpu[key])
        for v in values:
            print(v)
