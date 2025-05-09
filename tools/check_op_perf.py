import os
import subprocess
from pathlib import Path


def find_pytorch_dir():
    path = Path(__file__).resolve()
    while path != path.root:
        if path.name == "pytorch":
            return str(path)
        path = path.parent
    return ''


OP_LIST = {
    'layer_norm.py': ['aten::native_layer_norm', 'aten::native_layer_norm_backward'],
    'group_norm.py': ['aten::native_group_norm', 'aten::native_group_norm_backward'],
    'batch_norm_1d.py': ['aten::native_batch_norm', 'aten::native_batch_norm_backward'],
    'batch_norm_2d.py': ['aten::native_batch_norm', 'aten::native_batch_norm_backward'],
    'batch_norm_3d.py': ['aten::native_batch_norm', 'aten::native_batch_norm_backward'],
}


def find_op_time(text, ops):
    res = []

    def transform_to_us(time):
        if time.endswith('us'):
            return float(time[:-2])
        elif time.endswith('ms'):
            return float(time[:-2]) * 1000.0
        elif time.endswith('s'):
            return float(time[:-1]) * 1000000.0
        else:
            raise Exception("time format not support")
    flag = "None"
    for line in text.split('\n'):
        line = line.strip()
        if line.startswith('shape:'):
            flag = line
        for op in ops:
            if op in line:
                items = []
                for item in line.strip().split('  '):
                    if len(item) > 1:
                        items.append(item.strip())
                op_name = items[0]
                op_time = transform_to_us(items[-2])
                res.append([op_name, flag, str(op_time)])
    res_ = ["@@".join(item) for item in res]
    res_ = list(set(res_))
    res = [item.split("@@") for item in res_]
    res = sorted(res, key=lambda x: x[1])
    res = sorted(res, key=lambda x: x[0])
    return res


if __name__ == '__main__':
    root_folder = find_pytorch_dir().strip()
    perf_suit = os.path.join(root_folder, 'third_party/torch-xpu-ops/test/microbench/')
    import csv
    csv_data = [
        ["Operator", "Tag", "Latency(us)"],
    ]
    for item, ops in OP_LIST.items():
        print(item)
        f = os.path.join(perf_suit, item)
        result = subprocess.run(
            ["python", f],
            capture_output=True,
            text=True
        )
        output = result.stdout
        res = find_op_time(output, ops)
        csv_data += res
        for item in res:
            print(item)
    with open("check_op_perf.csv", mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerows(csv_data)
