# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

import argparse
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


def parse_keys_from_codegen(folder, backend, startswith='m.impl("', pattern=r'm\.impl\("([^"]+)"'):
    """Parse registered op keys from codegen Register*.cpp files."""
    files = glob.glob(f'{folder}/**/Register{backend}_0.cpp', recursive=True)
    if not files:
        raise FileNotFoundError(f'Register{backend}_0.cpp not found under {folder}')
    register_file = files[0]
    print(f'  Parsing: {register_file}')
    with open(register_file) as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines if l.strip().startswith(startswith)]
    keys = set()
    for line in lines:
        match = re.search(pattern, line)
        if match is not None:
            keys.add(match.group(1))
    return keys


def parse_keys_from_yaml(yaml_path, cuda_key='CUDA', xpu_key='XPU'):
    """Parse dispatch keys from native_functions.yaml directly.

    Returns a dict mapping dispatch_category -> set of op names for both
    CUDA-family and XPU-family backends.
    """
    import yaml

    with open(yaml_path) as f:
        funcs = yaml.safe_load(f)

    cuda_backends = {'CUDA', 'SparseCUDA', 'SparseCsrCUDA', 'NestedTensorCUDA'}
    xpu_backends = {'XPU', 'SparseXPU', 'SparseCsrXPU', 'NestedTensorXPU'}

    category_map = {
        'CUDA': 'basic_keys',
        'SparseCUDA': 'sparse_keys',
        'SparseCsrCUDA': 'sparse_csr_keys',
        'NestedTensorCUDA': 'nested_tensor_keys',
        'XPU': 'basic_keys',
        'SparseXPU': 'sparse_keys',
        'SparseCsrXPU': 'sparse_csr_keys',
        'NestedTensorXPU': 'nested_tensor_keys',
    }

    kcuda = Dict()
    kxpu = Dict()
    for cat in ['basic_keys', 'sparse_keys', 'sparse_csr_keys', 'nested_tensor_keys']:
        kcuda[cat] = set()
        kxpu[cat] = set()

    for entry in funcs:
        if not isinstance(entry, dict) or 'func' not in entry:
            continue
        func_sig = entry['func']
        op_name = func_sig.split('(')[0].strip()
        dispatch = entry.get('dispatch')
        if dispatch is None:
            continue
        for dispatch_key, _ in dispatch.items():
            # dispatch_key can be comma-separated like "CPU, CUDA, MPS, XPU"
            backends = [b.strip() for b in dispatch_key.split(',')]
            for backend in backends:
                if backend in cuda_backends:
                    cat = category_map[backend]
                    kcuda[cat].add(op_name)
                if backend in xpu_backends:
                    cat = category_map[backend]
                    kxpu[cat].add(op_name)

    return kcuda, kxpu


def find_pytorch_dir():
    path = Path(__file__).resolve()
    while path != path.root:
        if path.name == "pytorch":
            return str(path)
        path = path.parent
    return ''


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Check XPU op coverage compared to CUDA')
    parser.add_argument(
        '--mode', choices=['codegen', 'yaml'], default='codegen',
        help='codegen: parse from build Register*.cpp files (requires build); '
             'yaml: parse from native_functions.yaml directly')
    parser.add_argument(
        '--pytorch-dir', default=None,
        help='Path to pytorch root directory (auto-detected if not specified)')
    args = parser.parse_args()

    root_folder = args.pytorch_dir or find_pytorch_dir()
    if not root_folder:
        raise RuntimeError('Cannot find pytorch root directory')

    kcuda = Dict()
    kxpu = Dict()

    if args.mode == 'codegen':
        build_dir = os.path.join(root_folder, 'build')
        if not os.path.isdir(build_dir):
            raise RuntimeError(f'Build directory not found: {build_dir}. '
                               'Run build first or use --mode yaml.')

        print('Parsing CUDA ops from codegen...')
        kcuda.basic_keys = parse_keys_from_codegen(build_dir, 'CUDA')
        kcuda.basic_keys = {item for item in kcuda.basic_keys if 'cudnn' not in item}
        kcuda.sparse_keys = parse_keys_from_codegen(build_dir, 'SparseCUDA')
        kcuda.sparse_csr_keys = parse_keys_from_codegen(build_dir, 'SparseCsrCUDA')
        kcuda.nested_tensor_keys = parse_keys_from_codegen(build_dir, 'NestedTensorCUDA')

        print('Parsing XPU ops from codegen...')
        xpu_keys = parse_keys_from_codegen(build_dir, 'XPU')
        kxpu.basic_keys = (xpu_keys | onednn_keys) & kcuda.basic_keys
        kxpu.sparse_keys = parse_keys_from_codegen(build_dir, 'SparseXPU')
        kxpu.sparse_csr_keys = parse_keys_from_codegen(build_dir, 'SparseCsrXPU')
        kxpu.nested_tensor_keys = parse_keys_from_codegen(build_dir, 'NestedTensorXPU')

    elif args.mode == 'yaml':
        yaml_path = os.path.join(
            root_folder, 'aten', 'src', 'ATen', 'native', 'native_functions.yaml')
        if not os.path.isfile(yaml_path):
            raise RuntimeError(f'native_functions.yaml not found: {yaml_path}')
        print(f'Parsing from: {yaml_path}')
        kcuda, kxpu = parse_keys_from_yaml(yaml_path)
        kcuda.basic_keys = {item for item in kcuda.basic_keys if 'cudnn' not in item}
        # Add onednn ops that XPU supports via oneDNN
        kxpu.basic_keys = (kxpu.basic_keys | onednn_keys) & kcuda.basic_keys

    print('\n============ CUDA ============')
    for key in kcuda.keys():
        print(f"{key}: {len(kcuda[key])}")
    print('============ XPU ============')
    for key in kxpu.keys():
        print(f"{key}: {len(kxpu[key])}")

    kcuda_kxpu = Dict()
    for cudakeys in kcuda.keys():
        kcuda_kxpu[cudakeys] = kcuda[cudakeys] - kxpu[cudakeys]

    total_missing = sum(len(v) for v in kcuda_kxpu.values())
    print(f'\nTotal missing XPU ops: {total_missing}')

    for key in kcuda_kxpu.keys():
        print(f"\n============ {key} (missing {len(kcuda_kxpu[key])}) ============")
        values = sorted(kcuda_kxpu[key])
        for v in values:
            print(v)