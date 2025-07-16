"""
Microbenchmark Summary Tool - Parses performance logs and generates CSV/Excel reports
# Usage
# Summary forward op time, forward_op_summary.csv is forward summary file
python microbench_summary.py path/to/profile's log forward_op_summary.csv
# Summary backward op time, backward_op_summary.csv is backward summary file, True means summary backward, default is false.
python microbench_summary.py path/to/profile's log backward_op_summary.csv --backward
"""

import re
import pandas as pd
import glob
import os
import argparse
import bisect
from pathlib import Path
from typing import Dict, List

def main():
    parser = argparse.ArgumentParser(
        description="Parse performance logs and generate summary reports",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("log_dir", help="Directory containing log files")
    parser.add_argument("output_file", help="Output CSV file path")
    parser.add_argument("--backward", action="store_true",
                       help="Process backward operations instead of forward")
    args = parser.parse_args()

    try:
        df = parse_logs(args.log_dir, args.backward)
        if df.empty:
            print("Warning: No valid data found in log files!")
            return

        save_reports(df, args.output_file)
        print(f"Successfully generated reports: {args.output_file} and {args.output_file.replace('.csv', '.xlsx')}")
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

def parse_logs(log_dir: str, get_backward: bool = False) -> pd.DataFrame:
    data = []
    base_columns = [
        "case_name", "datatype", "op_name", "shape", "channels_last", "dim",
        "output_size", "P", "reduce", "kernel_size", "stride", "replacement",
        "num_samples", "scale_factor", "mode", "padding_mode", "align_corners",
        "shifts", "affine", "backward", "time(us)"
    ]

    for log_file in glob.glob(os.path.join(log_dir, "*.log")):
        try:
            with open(log_file) as f:
                content = f.read()

            case_name = Path(log_file).stem
            base_op_name = case_name.split('.')[-1]
            op_name, time_pattern = get_op_pattern(base_op_name, get_backward)

            # First find all shape lines and their positions
            shape_matches = list(re.finditer(r"(shape\s*[:=].*?)(?=\n\S|$)", content))
            shape_lines = [match.group(0) for match in shape_matches]
            shape_positions = [match.start() for match in shape_matches]
            # Parse all E2E forward time in the log
            E2E_forward_times = []
            E2E_total_times = []
            e2e_forward_time_matches = re.finditer(r"E2E forward time:\s*(\d+\.?\d*)", content)
            for match in e2e_forward_time_matches:
                time_val = float(match.group(1)) * 1_000_000
                time_pos = match.start()
                # Find the preceding shape for this E2E time
                preceding_shape_idx = bisect.bisect_right(shape_positions, time_pos) - 1
                if preceding_shape_idx >= 0:
                    E2E_forward_times.append((preceding_shape_idx, time_val))

            e2e_total_time_matches = re.finditer(r"E2E total time:\s*(\d+\.?\d*)", content)
            for match in e2e_total_time_matches:
                time_val = float(match.group(1)) * 1_000_000
                time_pos = match.start()
                # Find the preceding shape for this E2E time
                preceding_shape_idx = bisect.bisect_right(shape_positions, time_pos) - 1
                if preceding_shape_idx >= 0:
                    E2E_total_times.append((preceding_shape_idx, time_val))

            # Determine if we need E2E time column
            has_e2e_forward = len(E2E_forward_times) > 0
            has_e2e_total = len(E2E_total_times) > 0
            columns = base_columns.copy()
            if has_e2e_forward:
                columns.append("E2E forward time(us)")
            if has_e2e_total:
                columns.append("E2E total time(us)")

            if get_backward and base_op_name == "l1_loss":
                process_l1_loss(content, case_name, data, columns)
                continue

            # Find all time matches and their positions
            time_matches = []
            for match in re.finditer(fr"{time_pattern}.*?(?:\s+\S+){{8}}\s+(\d+\.?\d*)([a-zA-Z]*)", content):
                time = match.group(1)
                unit = match.group(2)
                time_pos = match.start()
                # Find the preceding shape for this time
                preceding_shape_idx = bisect.bisect_right(shape_positions, time_pos) - 1
                if preceding_shape_idx >= 0:
                    time_matches.append((time, unit, preceding_shape_idx))

             # Create mappings from shape index to E2E times
            shape_to_e2e_forward = {}
            for shape_idx, time in E2E_forward_times:
                shape_to_e2e_forward[shape_idx] = time

            shape_to_e2e_total = {}
            for shape_idx, time in E2E_total_times:
                shape_to_e2e_total[shape_idx] = time

            # time_matches = extract_times(content, time_pattern, get_backward)
            # shape_lines = re.findall(r"(shape\s*[:=].*?)(?=\n\S|$)", content)
            # Process matched times, skipping zeros but maintaining shape relationships
            processed_shapes = set()
            for time, unit, shape_idx in time_matches:
                time_us = convert_to_us(float(time), unit)
                if time_us == 0:
                    continue

                shape_text = shape_lines[shape_idx]
                if shape_idx in processed_shapes:
                    continue
                processed_shapes.add(shape_idx)
                params = extract_params(shape_text)

                if get_backward and params.get("backward", "False") == "False":
                    continue

                record = create_record(params, case_name, op_name, str(get_backward), time_us)

                # Add E2E times if available for this specific shape
                if has_e2e_forward:
                    record["E2E forward time(us)"] = shape_to_e2e_forward.get(shape_idx, "")
                if has_e2e_total:
                    record["E2E total time(us)"] = shape_to_e2e_total.get(shape_idx, "")

                data.append([record.get(col, "") for col in columns])

        except Exception as e:
            print(f"Warning: Error processing {log_file} - {str(e)}")
            continue

    return pd.DataFrame(data, columns=columns) if data else pd.DataFrame()

def get_op_pattern(base_op_name: str, get_backward: bool) -> tuple:
    op_name_map = {
        'forward': {
            'batch_norm': ('aten::batch_norm', 'aten::batch_norm'),
            'unique': ('unique2', 'unique2'),
            'fractional_max_pool2d': ('fractional_max_pool2d', r'\bfractional_max_pool2d\b'),
            'fractional_max_pool3d': ('fractional_max_pool3d', r'\bfractional_max_pool3d\b'),
            'adaptive_max_pool2d': ('adaptive_max_pool2d', r'\badaptive_max_pool2d\b'),
            'max_pool3d': ('max_pool3d_with_indices', 'max_pool3d_with_indices '),
            'max_pool2d': ('max_pool2d_with_indices', 'max_pool2d_with_indices '),
            'exponential': ('exponential_', r'\bexponential_\b'),
            'geometric': ('geometric_', r'\bgeometric_\b'),
            'uniform': ('uniform_', r'\buniform_\b'),
            'random': ('random_', r'\brandom_\b'),
            'log_normal': ('log_normal_', r'\blog_normal_\b'),
            'normal': ('normal_', r'\bnormal_\b'),
            'bernoulli': ('bernoulli_', r'\bbernoulli_\b'),
            'cauchy': ('cauchy_', r'\bcauchy_\b'),
            'embedding_bag': ('_embedding_bag', r'\b_embedding_bag\b'),
            'nonzero': ('nonzero', r'\bnonzero\b'),
            'index_fill': ('index_fill_', r'\bindex_fill_\b'),
            'index_put': ('index_put_', r'\bindex_put_\b'),
            'put': ('put_', r'\bput_\b'),
            'masked_fill': ('masked_fill_', r'\bmasked_fill_\b'),
            'scatter_add': ('scatter_add_', r'\bscatter_add_\b'),
            'scatter': ('scatter_', r'\bscatter_\b'),
            'dropout': ('dropout', r'\bdropout\b'),
            'layer_norm': ('layer_norm', r'\blayer_norm\b'),
            'ctc_loss': ('_ctc_loss', r'\b_ctc_loss\b'),
            'adaptive_avg_pool2d': ('adaptive_avg_pool2d', r'\badaptive_avg_pool2d\b'),
            'softmax': ('aten::softmax', 'aten::softmax'),
            'group_norm': ('aten::group_norm', 'aten::group_norm'),
        },
        'backward': {
            'batch_norm': ('batch_norm_backward', 'batch_norm_backward'),
            'fractional_max_pool2d': ('fractional_max_pool2d_backward', r'\bfractional_max_pool2d_backward\b'),
            'fractional_max_pool3d': ('fractional_max_pool3d_backward', r'\bfractional_max_pool3d_backward\b'),
            'adaptive_max_pool2d': ('adaptive_max_pool2d_backward', r'\badaptive_max_pool2d_backward\b'),
            'max_unpool2d': ('MaxUnpool2DBackward0', 'MaxUnpool2DBackward0 '),
            'max_unpool3d': ('MaxUnpool3DBackward0', 'MaxUnpool3DBackward0 '),
            'max_pool3d': ('max_pool3d_with_indices_backward', 'max_pool3d_with_indices_backward '),
            'max_pool2d': ('max_pool2d_with_indices_backward', 'max_pool2d_with_indices_backward '),
            'col2im': ('Col2ImBackward0', 'Col2ImBackward0 '),
            'im2col': ('Im2ColBackward0', 'Im2ColBackward0 '),
            'flip': ('FlipBackward0', 'FlipBackward0 '),
            'matmul': ('MmBackward0', 'MmBackward0 '),
            'roll': ('RollBackward0', 'RollBackward0 '),
            'softmax': ('softmax_backward_data', 'softmax_backward_data '),
            'remainder': ('RemainderBackward0', 'RemainderBackward0 '),
            'smooth_l1_loss': ('smooth_l1_loss_backward', 'smooth_l1_loss_backward'),
            'l1_loss': ('l1_loss', 'l1_loss'),
        }
    }

    mode = 'backward' if get_backward else 'forward'

    for op_pattern in op_name_map[mode]:
        if op_pattern in base_op_name:
            return op_name_map[mode][op_pattern]

    if get_backward:
        return (f"{base_op_name}_backward", f"{base_op_name}_backward ")
    else:
        return (base_op_name, f"{base_op_name} ")

def process_l1_loss(content: str, case_name: str, data: List, columns: List):
    shape_matches = list(re.finditer(r"(shape\s*[:=].*?)(?=\n\S|$)", content))
    shape_lines = [match.group(0) for match in shape_matches]
    shape_positions = [match.start() for match in shape_matches]

    # Parse E2E times if present in columns
    has_e2e_forward = "E2E forward time(us)" in columns
    has_e2e_total = "E2E total time(us)" in columns

    # Create mappings from shape index to E2E times
    shape_to_e2e_forward = {}
    shape_to_e2e_total = {}

    if has_e2e_forward:
        e2e_forward_time_matches = re.finditer(r"E2E forward time:\s*(\d+\.?\d*)", content)
        for match in e2e_forward_time_matches:
            time_val = float(match.group(1)) * 1_000_000
            time_pos = match.start()
            preceding_shape_idx = bisect.bisect_right(shape_positions, time_pos) - 1
            if preceding_shape_idx >= 0:
                shape_to_e2e_forward[preceding_shape_idx] = time_val

    if has_e2e_total:
        e2e_total_time_matches = re.finditer(r"E2E total time:\s*(\d+\.?\d*)", content)
        for match in e2e_total_time_matches:
            time_val = float(match.group(1)) * 1_000_000
            time_pos = match.start()
            preceding_shape_idx = bisect.bisect_right(shape_positions, time_pos) - 1
            if preceding_shape_idx >= 0:
                shape_to_e2e_total[preceding_shape_idx] = time_val

    filtered_content = [line for line in content.split('\n') if "autograd::engine" not in line]
    filtered_content = '\n'.join(filtered_content)
    abs_times = re.findall(r"AbsBackward0(?:\s+\S+){8}\s+(\d+\.?\d*)([a-zA-Z]*)", filtered_content)
    mean_times = re.findall(r"MeanBackward0(?:\s+\S+){8}\s+(\d+\.?\d*)([a-zA-Z]*)", filtered_content)
    shape_lines = re.findall(r"(shape\s*[:=].*?)(?=\n\S|$)", content)

    for i, (time, unit) in enumerate(abs_times[:6]):
        if i >= len(shape_lines):
            break
        time_us = convert_to_us(float(time), unit)
        params = extract_params(shape_lines[i])
        record = create_record(params, case_name, "AbsBackward0", "True", time_us)

        # Add E2E times if available
        if has_e2e_forward:
            record["E2E forward time(us)"] = shape_to_e2e_forward.get(i, "")
        if has_e2e_total:
            record["E2E total time(us)"] = shape_to_e2e_total.get(i, "")

        data.append([record.get(col, "") for col in columns])

    for i, (time, unit) in enumerate(mean_times):
        if (i + 6) >= len(shape_lines):
            break
        time_us = convert_to_us(float(time), unit)
        params = extract_params(shape_lines[i + 6])
        record = create_record(params, case_name, "MeanBackward0", "True", time_us)

        # Add E2E times if available
        if has_e2e_forward:
            record["E2E forward time(us)"] = shape_to_e2e_forward.get(i + 6, "")
        if has_e2e_total:
            record["E2E total time(us)"] = shape_to_e2e_total.get(i + 6, "")

        data.append([record.get(col, "") for col in columns])

def extract_times(content: str, pattern: str, get_backward: bool) -> List:
    lines = content.split('\n')
    results = []
    for line in lines:
        if get_backward and any(x in pattern for x in ["Col2ImBackward0", "Im2ColBackward0",
                                                     "FlipBackward0", "MmBackward0",
                                                     "RollBackward0", "MaxUnpool2DBackward0", "MaxUnpool3DBackward0"]):
            if "autograd::engine" in line:
                continue

        match = re.search(fr"{pattern}.*?(?:\s+\S+){{8}}\s+(\d+\.?\d*)([a-zA-Z]*)", line)
        if match:
            results.append((match.group(1), match.group(2)))

    return results

def create_record(params: Dict, case_name: str, op_name: str,
                 backward: str, time_us: float) -> Dict:
    return {
        "P": params.get("p", ""),
        **params,
        "case_name": case_name,
        "op_name": op_name,
        "backward": backward,
        "time(us)": time_us
    }

def convert_to_us(value: float, unit: str) -> float:
    unit = unit.lower()
    if unit == "ms":
        return value * 1000
    elif unit == "s":
        return value * 1_000_000
    return value

def extract_params(text: str) -> Dict:
    params = {}
    pairs = re.split(r'[;]', text.strip())

    for pair in pairs:
        if not any(delim in pair for delim in [':', '=']):
            continue

        delim = ':' if ':' in pair else '='
        key, value = pair.split(delim, 1)
        key = key.strip().lower()
        value = value.strip()

        if key in ['p', 'P']:
            key = 'p'
        elif key in ['dims', 'dim']:
            key = 'dim'
        elif key in ['shape']:
            key = 'shape'

        params[key] = value

    return params

def save_reports(df: pd.DataFrame, csv_path: str):
    os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
    df.to_csv(csv_path, index=False, sep=';')
    excel_path = csv_path.replace('.csv', '.xlsx')
    df.to_excel(excel_path, index=False)


if __name__ == "__main__":
    main()
