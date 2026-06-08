#!/usr/bin/env python3
"""
compare_projection_vs_actual.py

Compare roofline projection (from Calculate_Flops) vs actual GPU time (from
trace.json or unitrace) for per-op and aggregate analysis.

Supports two "actual" sources:
  - torch profiler trace.json (default, works for both XPU and CUDA)
  - unitrace via map_kernels_to_ops.py (XPU only, more accurate — no profiler overhead)

Hardware specs are loaded from config/hardware_specs.yaml (single source of truth).

Usage:
    # Using profiler trace as actual source
    python compare_projection_vs_actual.py calc_flops.txt trace.json --platform B580

    # Using unitrace as actual source (requires both trace.json and unitrace.json)
    python compare_projection_vs_actual.py calc_flops.txt trace.json --platform B580 \
        --unitrace python.1234.json
"""

import json
import os
import argparse
from collections import defaultdict

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# ---------------------------------------------------------------------------
# Hardware specs
# ---------------------------------------------------------------------------

# calc_flops output column indices for platform-specific cache-adjusted memory.
# These are tied to context_func.py output format:
#   col 0: op_name
#   col 1: cum_flops
#   col 2: cum_memory (raw)
#   col 3: cum_gemm_conv_flops
#   col 4: cum_gemm_conv_memory
#   col 5: mem_B580 (cache-adjusted)
#   col 6: mem_4080 (cache-adjusted)
#   col 7: mem_G31  (cache-adjusted)
#   ...
_MEM_COL_MAP = {"B580": 5, "4080": 6, "G31": 7}

# CLI platform name -> config key in hardware_specs.yaml
_PLATFORM_CONFIG_MAP = {"G31": "b70", "B580": "b580", "4080": "4080s"}


def _find_config_path():
    """Search for config/hardware_specs.yaml relative to this script."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "..", "..", "config", "hardware_specs.yaml"),
        os.path.join(script_dir, "config", "hardware_specs.yaml"),
    ]
    for p in candidates:
        rp = os.path.realpath(p)
        if os.path.exists(rp):
            return rp
    return None


def load_platform_specs(platform, config_path=None):
    """Load specs for *platform* from hardware_specs.yaml.

    Parameters
    ----------
    platform : str
        CLI platform key: "G31", "B580", or "4080".
    config_path : str or None
        Explicit path to hardware_specs.yaml.  Auto-detected if None.

    Returns
    -------
    dict with keys: peak_tflops (raw), bandwidth (raw), cache_threshold (bytes),
        mem_col, label, roofline_ratio
    """
    if config_path is None:
        config_path = _find_config_path()

    config_key = _PLATFORM_CONFIG_MAP[platform]

    if config_path and HAS_YAML:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        hw = cfg["platforms"][config_key]
        peak = hw["peak_tflops_fp16"] * 1e12
        bw = hw["bandwidth_gbs"] * 1e9
        label = hw.get("label", config_key)
    else:
        # Fallback: hardcoded values (keep in sync with config)
        _FALLBACK = {
            "G31":  {"peak": 154e12,    "bw": 532e9,   "label": "B70 (G31)"},
            "B580": {"peak": 93e12,     "bw": 410e9,   "label": "B580"},
            "4080": {"peak": 100.96e12, "bw": 716.8e9, "label": "RTX 4080 SUPER"},
        }
        fb = _FALLBACK[platform]
        peak, bw, label = fb["peak"], fb["bw"], fb["label"]
        if not HAS_YAML:
            print("WARNING: PyYAML not installed, using fallback hardware specs")
        else:
            print(f"WARNING: config/hardware_specs.yaml not found, using fallback specs")

    return {
        "peak_tflops": peak,
        "bandwidth": bw,
        "roofline_ratio": peak / bw,
        "mem_col": _MEM_COL_MAP[platform],
        "label": label,
    }


# ---------------------------------------------------------------------------
# Op name normalization
# ---------------------------------------------------------------------------

# Vector engine ops: these run on the vector engine, not the matrix engine.
# Since our peak TFLOPS is for the matrix engine, counting their FLOPs would
# misclassify them as compute-bound. They are memory-bound in practice.
# FLOPs are zeroed so roofline projection uses memory BW only.
VECTOR_ENGINE_OPS = {
    "aten::_softmax",
    "aten::native_layer_norm",
    "aten::native_batch_norm",
    "aten::max_pool2d_with_indices",
}


def normalize_op_name(name):
    """Normalize op names to enable matching between calc_flops and trace.

    calc_flops uses variant names like aten::add.Tensor, aten::mul.Scalar
    trace uses base names like aten::add, aten::mul

    SDPA naming varies by platform and nesting level:
      Forward nesting (CUDA trace):
        aten::scaled_dot_product_attention          (outermost, gpu_time=0)
          -> aten::_scaled_dot_product_flash_attention  (middle, gpu_time=0)
            -> aten::_flash_attention_forward            (innermost, has gpu_time)
      Forward (XPU):
        aten::_scaled_dot_product_fused_attention_overrideable (has gpu_time)
      Backward nesting (CUDA trace):
        aten::_scaled_dot_product_flash_attention_backward  (outer, gpu_time=0)
          -> aten::_flash_attention_backward                (inner, has gpu_time)
      Backward (XPU):
        aten::_scaled_dot_product_fused_attention_overrideable_backward

    We normalize all variants to common names: sdpa_forward / sdpa_backward.
    """
    # Map convolution variants to common name.
    conv_names = {
        "aten::convolution", "aten::_convolution",
        "aten::cudnn_convolution", "aten::mkldnn_convolution",
        "aten::xpu_convolution", "aten::conv2d",
        "aten::convolution_overrideable",
    }
    if name in conv_names:
        return "aten::convolution"

    # Map convolution backward variants to common name.
    conv_bwd_names = {
        "aten::convolution_backward",
        "aten::convolution_backward_overrideable",
        "aten::cudnn_convolution_backward",
        "aten::mkldnn_convolution_backward",
    }
    if name in conv_bwd_names:
        return "aten::convolution_backward"

    # View / metadata ops that never launch GPU kernels.
    # context_func.py may incorrectly attribute memory to these, but they
    # have zero actual GPU time.  Treat them as no-ops for comparison.
    view_ops = {
        "aten::unbind", "aten::unbind.int",
        "aten::contiguous",   # view-like, dispatches to clone only if needed
        "aten::reshape",      # view-like, dispatches to clone only if needed
        "aten::t",            # transpose metadata, no GPU kernel
    }
    if name in view_ops:
        return "__view_noop__"

    # Map high-level wrappers to the op name DispatchLog uses.
    # Also maps kernel-level ops to their dispatch-level parent
    # (e.g. clamp_min is the GPU kernel implementation of relu).
    #
    # IMPORTANT: Every intermediate wrapper in PyTorch's dispatch chain must
    # normalize to the same canonical name as the leaf op.  This is how
    # parse_trace_ops detects impl-detail nesting (parent normalized ==
    # child normalized → child is impl-detail).
    wrapper_to_dispatch = {
        "aten::linear": "aten::addmm",
        "aten::layer_norm": "aten::native_layer_norm",
        # copy_ no longer mapped here — parent-attribution in parse_trace_ops
        # correctly attributes copy_'s GPU time to its parent (clone, select_backward, etc.)
        "aten::batch_norm": "aten::native_batch_norm",
        "aten::_batch_norm_impl_index": "aten::native_batch_norm",
        "aten::softmax": "aten::_softmax",
        "aten::log_softmax": "aten::_log_softmax",
        "aten::matmul": "aten::bmm",
        "aten::adaptive_avg_pool2d": "aten::mean",
        "aten::max_pool2d": "aten::max_pool2d_with_indices",
        # relu/relu_ use clamp_min/clamp_min_ as kernel implementation.
        # Trace attributes GPU time to clamp_min, but calcflops captures relu.
        "aten::clamp_min": "aten::relu",
        "aten::clamp_min_": "aten::relu_",
        # nll_loss wrappers: nll_loss_nd → nll_loss → nll_loss_forward
        "aten::nll_loss_nd": "aten::nll_loss_forward",
        "aten::nll_loss": "aten::nll_loss_forward",
    }
    if name in wrapper_to_dispatch:
        return wrapper_to_dispatch[name]

    # Map SDPA forward variants to common name
    sdpa_forward_names = {
        "aten::scaled_dot_product_attention",
        "aten::_scaled_dot_product_flash_attention",
        "aten::_scaled_dot_product_efficient_attention",
        "aten::_scaled_dot_product_fused_attention_overrideable",
        "aten::_flash_attention_forward",
        "aten::_efficient_attention_forward",
    }
    if name in sdpa_forward_names:
        return "aten::sdpa_forward"

    # Map SDPA backward variants to common name
    sdpa_backward_names = {
        "aten::_scaled_dot_product_flash_attention_backward",
        "aten::_scaled_dot_product_fused_attention_overrideable_backward",
        "aten::_flash_attention_backward",
        "aten::_efficient_attention_backward",
    }
    if name in sdpa_backward_names:
        return "aten::sdpa_backward"

    # Strip overload suffixes (e.g. aten::add.Tensor -> aten::add)
    if "." in name.split("::")[-1]:
        name = name.rsplit(".", 1)[0]

    # Merge in-place variants with their out-of-place counterparts.
    # TorchDispatchMode (calcflops) may functionalize add_ → add, but trace
    # keeps the original in-place name. Merging ensures correct R_op comparison.
    inplace_to_outofplace = {
        "aten::add_": "aten::add",
        "aten::mul_": "aten::mul",
        "aten::sub_": "aten::sub",
        "aten::div_": "aten::div",
        "aten::masked_fill_": "aten::masked_fill",
    }
    if name in inplace_to_outofplace:
        return inplace_to_outofplace[name]

    return name


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_calc_flops(path, spec, iteration=-1):
    """Parse Calculate_Flops output and return per-op data for one iteration.

    Parameters
    ----------
    path : str
        Path to calcflops output file.
    spec : dict
        Platform spec from load_platform_specs().
    iteration : int
        Which benchmark iteration to use (-1 = last).

    Returns
    -------
    list[dict] with keys: name, name_raw, flops, memory, memory_platform,
        args, bound, proj_time_s
    """
    peak = spec["peak_tflops"]
    bw = spec["bandwidth"]
    rr = spec["roofline_ratio"]
    mem_col = spec["mem_col"]

    all_ops = []
    current_iter = -1
    has_iter_markers = False

    with open(path) as f:
        lines = f.readlines()

    for line in lines:
        if line.strip().startswith("===== ITERATION"):
            has_iter_markers = True
            break

    if not has_iter_markers:
        current_iter = 0

    for line in lines:
        line = line.strip()
        if line.startswith("===== ITERATION"):
            current_iter += 1
            continue
        if current_iter != 0:
            continue
        if not line.startswith("aten::"):
            continue

        parts = line.split("|")
        if len(parts) < 24:
            continue

        name_raw = parts[0]
        cum_flops = int(float(parts[1]))
        cum_mem = int(float(parts[2]))
        cum_mem_platform = int(float(parts[mem_col]))
        args_str = parts[23].replace("args:", "") if parts[23].startswith("args:") else ""

        all_ops.append({
            "name_raw": name_raw,
            "name": normalize_op_name(name_raw),
            "cum_flops": cum_flops,
            "cum_mem": cum_mem,
            "cum_mem_platform": cum_mem_platform,
            "args": args_str,
        })

    # Detect benchmark iterations by cumulative value resets
    if len(all_ops) > 0:
        iter_boundaries = [0]
        for i in range(1, len(all_ops)):
            if all_ops[i]["cum_mem_platform"] < all_ops[i - 1]["cum_mem_platform"] * 0.5:
                iter_boundaries.append(i)

        num_bench_iters = len(iter_boundaries)
        iter_boundaries.append(len(all_ops))

        if iteration == -1:
            iteration = num_bench_iters - 1

        if iteration >= num_bench_iters:
            print(f"  WARNING: requested iteration {iteration} but only {num_bench_iters} found. Using last.")
            iteration = num_bench_iters - 1

        start_idx = iter_boundaries[iteration]
        end_idx = iter_boundaries[iteration + 1]
        ops = all_ops[start_idx:end_idx]
        print(f"  Detected {num_bench_iters} benchmark iterations, {end_idx - start_idx} ops/iter")
        print(f"  Using benchmark iteration {iteration}: ops [{start_idx}:{end_idx}] ({len(ops)} ops)")
    else:
        ops = all_ops

    # Compute per-op deltas
    result = []
    prev_flops = 0
    prev_mem = 0
    prev_mem_platform = 0

    for op in ops:
        delta_flops = op["cum_flops"] - prev_flops
        delta_mem = op["cum_mem"] - prev_mem
        delta_mem_platform = op["cum_mem_platform"] - prev_mem_platform

        # Vector engine ops: zero FLOPs so roofline uses memory BW only.
        # Old data may have non-zero FLOPs from earlier context_func.py versions.
        if op["name"] in VECTOR_ENGINE_OPS:
            delta_flops = 0

        if delta_flops == 0 and delta_mem_platform == 0:
            bound = "skip"
            proj_time_s = 0
        elif delta_mem_platform == 0:
            bound = "compute"
            proj_time_s = delta_flops / peak
        elif delta_flops == 0:
            bound = "memory"
            proj_time_s = delta_mem_platform / bw
        else:
            intensity = delta_flops / delta_mem_platform
            if intensity > rr:
                bound = "compute"
                proj_time_s = delta_flops / peak
            else:
                bound = "memory"
                proj_time_s = delta_mem_platform / bw

        result.append({
            "name": op["name"],
            "name_raw": op["name_raw"],
            "flops": delta_flops,
            "memory": delta_mem,
            "memory_platform": delta_mem_platform,
            "args": op["args"],
            "bound": bound,
            "proj_time_s": proj_time_s,
        })

        prev_flops = op["cum_flops"]
        prev_mem = op["cum_mem"]
        prev_mem_platform = op["cum_mem_platform"]

    return result


def parse_trace_ops(path):
    """Parse trace.json and return dispatch-level aten:: ops with GPU times.

    Handles the copy_ attribution problem: copy_ is the kernel implementation
    of higher-level ops (clone, select_backward, constant_pad_nd, etc.).
    Its GPU time is attributed to its immediate aten:: parent, and copy_ itself
    is not emitted as a separate entry.

    Similarly handles convolution_backward nesting: the outer (composite) op
    gets the inner (dispatch-level) op's GPU time.

    Returns list of dicts: name, name_raw, gpu_dur_us, ext_id, input_dims, input_strides
    """
    with open(path) as f:
        data = json.load(f)
    events = data["traceEvents"]

    cpu_ops = []
    device_events = []

    for e in events:
        if not isinstance(e, dict) or e.get("ph") != "X":
            continue
        cat = e.get("cat", "")
        if cat == "cpu_op":
            cpu_ops.append(e)
        elif cat in ("kernel", "gpu_memcpy"):
            device_events.append(e)

    # Map External id → total GPU kernel duration
    ext_map = defaultdict(float)
    for d in device_events:
        ext_id = d.get("args", {}).get("External id")
        if ext_id is not None:
            ext_map[ext_id] += d.get("dur", 0)

    # Filter to aten:: ops and sort by timestamp
    aten_ops = []
    for op in cpu_ops:
        name_raw = op.get("name", "")
        if not name_raw.startswith("aten::"):
            continue
        aten_ops.append(op)
    aten_ops.sort(key=lambda e: e.get("ts", 0))

    # Build immediate parent for each op using a stack.
    n = len(aten_ops)
    parent_idx = [-1] * n
    stack = []  # stack of (end_time, index)

    for i, op in enumerate(aten_ops):
        ts = op.get("ts", 0)
        dur = op.get("dur", 0)
        end = ts + dur

        # Pop ops that have ended before this op starts
        while stack and stack[-1][0] <= ts:
            stack.pop()

        if stack:
            parent_idx[i] = stack[-1][1]

        stack.append((end, i))

    # Identify "implementation-detail" ops whose GPU time should be attributed
    # to their parent.  These ops are kernel implementations that don't appear
    # as independent dispatch-level ops in DispatchLog (calcflops).
    #
    # Detection rules:
    # 1. copy_ is always an impl-detail of its parent.
    # 2. If a child's NORMALIZED name matches its parent's NORMALIZED name,
    #    the child is an impl-detail.  This catches PyTorch's multi-level
    #    dispatch chains where different raw names map to the same canonical op:
    #      conv2d → convolution → _convolution → cudnn_convolution  (all → aten::convolution)
    #      linear → addmm  (both → aten::addmm)
    #      batch_norm → _batch_norm_impl_index → native_batch_norm  (all → aten::native_batch_norm)
    #      scaled_dot_product_attention → _scaled_dot_product_flash_attention → _flash_attention_forward  (all → sdpa_forward)
    # 3. Certain "internal" ops (fill_, zero_, empty, etc.) are impl-details
    #    when nested inside another aten:: op.  These are C++ internal ops that
    #    DispatchLog doesn't see (they happen inside func(*args) in __torch_dispatch__).
    #    Example: nll_loss_backward internally calls zero_/fill_ to zero grad_input,
    #    whose 701us GPU time must be attributed to nll_loss_backward, not counted
    #    as a separate op.
    IMPL_DETAIL_OPS = {"aten::copy_"}

    # Ops that are impl-details ONLY when nested inside another aten:: op.
    # When standalone (no parent), they may be real user-level operations.
    NESTED_IMPL_DETAIL_OPS = {
        "aten::zero_", "aten::fill_",           # memory fill (e.g. inside nll_loss_backward)
        "aten::zeros",                           # alloc + zero-fill
        "aten::empty", "aten::empty_like",       # allocation only (0 GPU time)
        "aten::empty_strided",                   # allocation only
        "aten::resize_",                         # metadata change (0 GPU time)
        "aten::new_zeros", "aten::new_empty",    # factory ops inside backward
    }

    # Pre-compute normalized names for efficient parent-child comparison
    normalized_names = [normalize_op_name(aten_ops[i].get("name", "")) for i in range(n)]

    is_impl_detail = [False] * n
    for i in range(n):
        name = aten_ops[i].get("name", "")
        if name in IMPL_DETAIL_OPS:
            is_impl_detail[i] = True
        elif parent_idx[i] != -1:
            # Child with same normalized name as parent → impl-detail
            if normalized_names[i] == normalized_names[parent_idx[i]]:
                is_impl_detail[i] = True
            # Internal ops nested inside another aten:: op → impl-detail
            elif name in NESTED_IMPL_DETAIL_OPS:
                is_impl_detail[i] = True

    # Compute GPU time per op (own ext_id's kernels)
    own_gpu = []
    for op in aten_ops:
        ext_id = op.get("args", {}).get("External id")
        own_gpu.append(ext_map.get(ext_id, 0) if ext_id else 0)

    # Attribute impl-detail GPU time to their immediate parent.
    # Walk bottom-up: if an impl-detail's parent is also an impl-detail,
    # propagate further up.
    attributed_gpu = list(own_gpu)
    for i in range(n):
        if is_impl_detail[i] and parent_idx[i] != -1:
            # Find the closest non-impl-detail ancestor
            target = parent_idx[i]
            while is_impl_detail[target] and parent_idx[target] != -1:
                target = parent_idx[target]
            attributed_gpu[target] += own_gpu[i]

    # Emit only non-impl-detail ops
    result = []
    for i in range(n):
        if is_impl_detail[i]:
            continue
        op = aten_ops[i]
        name_raw = op.get("name", "")
        ext_id = op.get("args", {}).get("External id")
        input_dims = op.get("args", {}).get("Input Dims", "")
        input_strides = op.get("args", {}).get("Input Strides", "")
        normalized = normalize_op_name(name_raw)
        result.append({
            "name": normalized,
            "name_raw": name_raw,
            "gpu_dur_us": attributed_gpu[i],
            "ext_id": ext_id,
            "input_dims": str(input_dims),
            "input_strides": str(input_strides),
        })

    return result


def parse_unitrace_ops(trace_path, unitrace_path):
    """Parse unitrace via map_kernels_to_ops and return top-level ops with actual GPU times.

    Uses map_kernels_to_ops to map unitrace kernel durations to aten:: ops, then
    applies impl-detail parent-attribution (copy_ → parent) consistent with
    parse_trace_ops, and normalizes op names for comparison with calc_flops.

    Returns list of dicts: name, name_raw, gpu_dur_us, input_dims, input_strides
    """
    from map_kernels_to_ops import map_kernels_to_ops

    _mapped, toplevel_ops, _mismatches = map_kernels_to_ops(trace_path, unitrace_path)

    # --- Parent-attribution for IMPL_DETAIL_OPS (same logic as parse_trace_ops) ---
    # map_kernels_to_ops returns ALL aten:: ops; copy_ entries get kernel time
    # attributed to them.  We need to redirect that time to the parent op.
    IMPL_DETAIL_OPS = {"aten::copy_"}

    n = len(toplevel_ops)

    # Build parent index using timestamp-based nesting
    parent_idx = [-1] * n
    stack = []  # (end_time, index)
    for i, op in enumerate(toplevel_ops):
        ts = op["ts"]
        end = op["end"]
        while stack and stack[-1][0] <= ts:
            stack.pop()
        if stack:
            parent_idx[i] = stack[-1][1]
        stack.append((end, i))

    # Mark impl-detail ops (copy_, and normalized-name nesting)
    # Same logic as parse_trace_ops: use normalized names to detect
    # multi-level dispatch chains.
    NESTED_IMPL_DETAIL_OPS = {
        "aten::zero_", "aten::fill_",
        "aten::zeros",
        "aten::empty", "aten::empty_like", "aten::empty_strided",
        "aten::resize_", "aten::new_zeros", "aten::new_empty",
    }
    normalized_names = [normalize_op_name(toplevel_ops[i]["name"]) for i in range(n)]

    is_impl_detail = [False] * n
    for i in range(n):
        name = toplevel_ops[i]["name"]
        if name in IMPL_DETAIL_OPS:
            is_impl_detail[i] = True
        elif parent_idx[i] != -1:
            if normalized_names[i] == normalized_names[parent_idx[i]]:
                is_impl_detail[i] = True
            elif name in NESTED_IMPL_DETAIL_OPS:
                is_impl_detail[i] = True

    # Attribute impl-detail unitrace time to nearest non-impl-detail ancestor
    attributed_dur = [op["unitrace_dur_us"] for op in toplevel_ops]
    attributed_cnt = [op["unitrace_kernel_count"] for op in toplevel_ops]
    for i in range(n):
        if is_impl_detail[i] and parent_idx[i] != -1:
            target = parent_idx[i]
            while is_impl_detail[target] and parent_idx[target] != -1:
                target = parent_idx[target]
            attributed_dur[target] += toplevel_ops[i]["unitrace_dur_us"]
            attributed_cnt[target] += toplevel_ops[i]["unitrace_kernel_count"]

    # Emit only non-impl-detail ops
    result = []
    for i in range(n):
        if is_impl_detail[i]:
            continue
        op = toplevel_ops[i]
        normalized = normalize_op_name(op["name"])
        result.append({
            "name": normalized,
            "name_raw": op["name"],
            "gpu_dur_us": attributed_dur[i],
            "input_dims": op.get("input_dims", ""),
            "input_strides": op.get("input_strides", ""),
        })

    return result


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def aggregate_comparison(calc_ops, actual_ops):
    """Aggregate by op name and compare projected vs actual.

    Returns dict: op_name -> {proj_time_ms, actual_time_ms, ratio, diff_ms,
        dominant_shape, dominant_stride, actual_tflops, actual_bw_gbs, ...}
    """
    calc_agg = defaultdict(lambda: {
        "proj_time_s": 0, "flops": 0, "memory": 0, "memory_platform": 0,
        "count": 0, "compute_count": 0, "memory_count": 0, "skip_count": 0
    })
    for op in calc_ops:
        name = op["name"]
        calc_agg[name]["proj_time_s"] += op["proj_time_s"]
        calc_agg[name]["flops"] += op["flops"]
        calc_agg[name]["memory"] += op["memory"]
        calc_agg[name]["memory_platform"] += op["memory_platform"]
        calc_agg[name]["count"] += 1
        if op["bound"] == "compute":
            calc_agg[name]["compute_count"] += 1
        elif op["bound"] == "memory":
            calc_agg[name]["memory_count"] += 1
        else:
            calc_agg[name]["skip_count"] += 1

    actual_agg = defaultdict(lambda: {"gpu_time_us": 0, "count": 0})
    # Track per-shape actual time for dominant shape detection
    shape_time = defaultdict(lambda: defaultdict(float))  # op_name -> {shape_key: total_us}
    shape_stride_map = {}  # shape_key -> (dims_str, strides_str)
    for op in actual_ops:
        name = op["name"]
        actual_agg[name]["gpu_time_us"] += op["gpu_dur_us"]
        actual_agg[name]["count"] += 1
        # Track shape contribution to actual time
        dims = op.get("input_dims", "")
        strides = op.get("input_strides", "")
        shape_key = f"{dims}|{strides}"
        shape_time[name][shape_key] += op["gpu_dur_us"]
        shape_stride_map[shape_key] = (dims, strides)

    all_names = set(calc_agg.keys()) | set(actual_agg.keys())
    # Filter out sentinel view/noop ops and impl-detail ops (safety net —
    # parse_trace_ops and parse_unitrace_ops should already exclude these,
    # but calcflops can occasionally capture copy_ via TorchDispatchMode).
    AGGREGATE_EXCLUDE = {"__view_noop__", "aten::copy_"}
    all_names -= AGGREGATE_EXCLUDE
    result = {}
    for name in all_names:
        c = calc_agg.get(name, {
            "proj_time_s": 0, "flops": 0, "memory": 0, "memory_platform": 0,
            "count": 0, "compute_count": 0, "memory_count": 0, "skip_count": 0
        })
        t = actual_agg.get(name, {"gpu_time_us": 0, "count": 0})
        proj_ms = c["proj_time_s"] * 1000
        actual_ms = t["gpu_time_us"] / 1000

        # Find dominant shape (highest actual time contribution)
        dom_dims, dom_strides = "", ""
        all_shapes = set()
        if name in shape_time and shape_time[name]:
            best_key = max(shape_time[name], key=shape_time[name].get)
            dom_dims, dom_strides = shape_stride_map.get(best_key, ("", ""))
            for sk in shape_time[name]:
                dims_val, _ = shape_stride_map.get(sk, ("", ""))
                if dims_val:
                    all_shapes.add(dims_val)

        # Compute actual TFLOPS and actual BW (GB/s)
        actual_time_s = t["gpu_time_us"] / 1e6
        actual_tflops = None
        actual_bw_gbs = None
        if actual_time_s > 0:
            if c["flops"] > 0:
                actual_tflops = c["flops"] / actual_time_s / 1e12
            if c["memory_platform"] > 0:
                actual_bw_gbs = c["memory_platform"] / actual_time_s / 1e9

        result[name] = {
            "proj_time_ms": proj_ms,
            "actual_time_ms": actual_ms,
            "ratio": proj_ms / actual_ms if actual_ms > 0 else float("inf"),
            "diff_ms": proj_ms - actual_ms,
            "flops": c["flops"],
            "memory": c["memory"],
            "memory_platform": c["memory_platform"],
            "count_calc": c["count"],
            "count_actual": t["count"],
            "compute_count": c["compute_count"],
            "memory_count": c["memory_count"],
            "skip_count": c["skip_count"],
            "in_calc": name in calc_agg,
            "in_actual": name in actual_agg,
            "dominant_shape": dom_dims,
            "dominant_stride": dom_strides,
            "all_shapes": all_shapes,
            "actual_tflops": actual_tflops,
            "actual_bw_gbs": actual_bw_gbs,
        }

    return result


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_aggregate_table(agg, spec, actual_source, sort_by="diff_ms", top_n=40):
    """Print aggregate comparison table."""
    if sort_by == "diff_ms":
        items = sorted(agg.items(), key=lambda x: abs(x[1]["diff_ms"]), reverse=True)
    elif sort_by == "proj":
        items = sorted(agg.items(), key=lambda x: x[1]["proj_time_ms"], reverse=True)
    elif sort_by == "actual":
        items = sorted(agg.items(), key=lambda x: x[1]["actual_time_ms"], reverse=True)
    else:
        items = sorted(agg.items(), key=lambda x: x[1]["proj_time_ms"], reverse=True)

    total_proj = sum(v["proj_time_ms"] for v in agg.values())
    total_actual = sum(v["actual_time_ms"] for v in agg.values())

    print(f"\n{'=' * 155}")
    print(f"AGGREGATE COMPARISON: Roofline Projection vs Actual GPU Time (per op type)")
    rr = spec["roofline_ratio"]
    print(f"  {spec['label']}: {spec['peak_tflops']/1e12:.2f} TFLOPS, "
          f"{spec['bandwidth']/1e9:.1f} GB/s, roofline ratio = {rr:.1f} OPs/byte")
    print(f"  Actual source: {actual_source}")
    print(f"{'=' * 155}")
    print(f"{'OP NAME':<35} {'Proj(ms)':>10} {'Actual(ms)':>10} {'Ratio':>7} "
          f"{'Diff(ms)':>10} {'#Calc':>6} {'#Actual':>7} {'Bound':>12} "
          f"{'FLOPs(G)':>10} {'Mem(MB)':>12}")
    print(f"{'-' * 155}")

    for name, v in items[:top_n]:
        bound_str = f"C:{v['compute_count']} M:{v['memory_count']}"
        if v["skip_count"] > 0:
            bound_str += f" S:{v['skip_count']}"

        marker = ""
        if not v["in_actual"]:
            marker = " [NO ACTUAL]"
        elif not v["in_calc"]:
            marker = " [NO CALC]"

        print(f"{name:<35} {v['proj_time_ms']:>10.3f} {v['actual_time_ms']:>10.3f} "
              f"{v['ratio']:>7.2f} {v['diff_ms']:>+10.3f} {v['count_calc']:>6} "
              f"{v['count_actual']:>7} {bound_str:>12} {v['flops']/1e9:>10.1f} "
              f"{v['memory_platform']/1e6:>12.1f}{marker}")

    print(f"{'=' * 155}")
    overall_ratio = total_proj / total_actual if total_actual > 0 else 0
    print(f"{'TOTAL':<35} {total_proj:>10.3f} {total_actual:>10.3f} "
          f"{overall_ratio:>7.2f} {total_proj - total_actual:>+10.3f}")
    print()

    # Print ops only in one source
    calc_only = [(n, v) for n, v in agg.items()
                 if not v["in_actual"] and v["proj_time_ms"] > 0.001]
    actual_only = [(n, v) for n, v in agg.items()
                   if not v["in_calc"] and v["actual_time_ms"] > 0.001]

    if calc_only:
        print(f"\nOps in Calculate_Flops but NOT in actual (proj > 0.001ms):")
        for n, v in sorted(calc_only, key=lambda x: x[1]["proj_time_ms"], reverse=True):
            print(f"  {n:<40} proj={v['proj_time_ms']:.3f}ms  count={v['count_calc']}")

    if actual_only:
        print(f"\nOps in actual but NOT in Calculate_Flops (actual > 0.001ms):")
        for n, v in sorted(actual_only, key=lambda x: x[1]["actual_time_ms"], reverse=True):
            print(f"  {n:<40} actual={v['actual_time_ms']:.3f}ms  count={v['count_actual']}")


def print_summary(agg, actual_source):
    """Print overall summary."""
    total_proj = sum(v["proj_time_ms"] for v in agg.values())
    total_actual = sum(v["actual_time_ms"] for v in agg.values())
    proj_compute = sum(v["proj_time_ms"] for v in agg.values()
                       if v["compute_count"] > 0 and v["memory_count"] == 0)
    proj_memory = sum(v["proj_time_ms"] for v in agg.values()
                      if v["memory_count"] > 0 and v["compute_count"] == 0)
    proj_mixed = sum(v["proj_time_ms"] for v in agg.values()
                     if v["compute_count"] > 0 and v["memory_count"] > 0)

    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    print(f"  Actual source: {actual_source}")
    print(f"  Total projected (T1): {total_proj:.3f} ms")
    print(f"  Total actual:         {total_actual:.3f} ms")
    if total_actual > 0:
        print(f"  Ratio (proj/actual):  {total_proj / total_actual:.3f}")
    print(f"  Difference:           {total_proj - total_actual:+.3f} ms")
    print(f"  Projection from compute-bound ops: {proj_compute:.3f} ms")
    print(f"  Projection from memory-bound ops:  {proj_memory:.3f} ms")
    print(f"  Projection from mixed-bound ops:   {proj_mixed:.3f} ms")

    if "unitrace" not in actual_source.lower():
        print(f"\n  NOTE: 'actual' from profiler trace may include overhead.")
        print(f"  For XPU, prefer --unitrace for accurate per-op timing.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare roofline projection vs actual GPU time")
    parser.add_argument("calc_flops_file", help="Path to Calculate_Flops output")
    parser.add_argument("trace_file", help="Path to trace.json")
    parser.add_argument("--platform", choices=list(_PLATFORM_CONFIG_MAP.keys()),
                        default="B580",
                        help="Target platform (default: B580)")
    parser.add_argument("--unitrace", type=str, default=None,
                        help="Path to unitrace JSON — use unitrace as actual source "
                             "(more accurate than profiler trace for XPU)")
    parser.add_argument("--iteration", type=int, default=-1,
                        help="Which calc_flops benchmark iteration (-1=last, default: -1)")
    parser.add_argument("--top", type=int, default=40, help="Show top N ops")
    parser.add_argument("--sort-by", choices=["diff_ms", "proj", "actual"],
                        default="diff_ms")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to hardware_specs.yaml (auto-detected if omitted)")
    args = parser.parse_args()

    # Load platform specs
    spec = load_platform_specs(args.platform, config_path=args.config)

    print(f"Platform: {spec['label']}")
    print(f"  Peak: {spec['peak_tflops']/1e12:.2f} TFLOPS, BW: {spec['bandwidth']/1e9:.1f} GB/s, "
          f"Roofline ratio: {spec['roofline_ratio']:.1f} OPs/byte")

    # Parse calc_flops
    print(f"\nLoading Calculate_Flops data from {args.calc_flops_file} "
          f"(iteration {args.iteration})...")
    calc_ops = parse_calc_flops(args.calc_flops_file, spec, args.iteration)
    print(f"  Found {len(calc_ops)} ops")

    total_proj = sum(op["proj_time_s"] for op in calc_ops) * 1000
    compute_ops = sum(1 for op in calc_ops if op["bound"] == "compute")
    memory_ops = sum(1 for op in calc_ops if op["bound"] == "memory")
    skip_ops = sum(1 for op in calc_ops if op["bound"] == "skip")
    print(f"  Compute-bound: {compute_ops}, Memory-bound: {memory_ops}, Skipped: {skip_ops}")
    print(f"  Total projected time (T1): {total_proj:.3f} ms")

    # Parse actual GPU times
    if args.unitrace:
        actual_source = f"unitrace ({os.path.basename(args.unitrace)})"
        print(f"\nLoading actual from unitrace: {args.unitrace}")
        print(f"  (using {args.trace_file} for kernel-to-op mapping)")
        actual_ops = parse_unitrace_ops(args.trace_file, args.unitrace)
        total_actual = sum(op["gpu_dur_us"] for op in actual_ops) / 1000
        print(f"  Found {len(actual_ops)} top-level ops, "
              f"total unitrace GPU time: {total_actual:.3f} ms")
    else:
        actual_source = f"profiler trace ({os.path.basename(args.trace_file)})"
        print(f"\nLoading actual from profiler trace: {args.trace_file}...")
        actual_ops = parse_trace_ops(args.trace_file)
        print(f"  Found {len(actual_ops)} top-level aten:: ops")

        total_actual = sum(op["gpu_dur_us"] for op in actual_ops) / 1000
        print(f"  Total actual GPU time: {total_actual:.3f} ms")

    # Compare
    agg = aggregate_comparison(calc_ops, actual_ops)
    print_aggregate_table(agg, spec, actual_source,
                          sort_by=args.sort_by, top_n=args.top)
    print_summary(agg, actual_source)


if __name__ == "__main__":
    main()
