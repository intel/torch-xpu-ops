# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Standalone torch._grouped_mm benchmark with DeepSeek-V3 16B MoE shapes.

Reproduces, per EP rank, the six grouped GEMMs one torchtitan MoE layer issues
per training step (torchtitan GroupedLinear calls
``torch._grouped_mm(x, weight.transpose(-2, -1), offs)``):

  op        mat_a                    mat_b                        out
  w13_fwd   x  [M, 2048]             w13^T [E, 2048, 2816] (K-contig) [M, 2816]
  w2_fwd    h  [M, 1408]             w2^T  [E, 1408, 2048] (K-contig) [M, 2048]
  w13_dX    dg [M, 2816]             w13   [E, 2816, 2048] (N-contig) [M, 2048]
  w2_dX     dy [M, 2048]             w2    [E, 2048, 1408] (N-contig) [M, 1408]
  w13_dW    dg^T [2816, M] (M-contig) x  [M, 2048]                    [E, 2816, 2048]
  w2_dW     dy^T [2048, M] (M-contig) h  [M, 1408]                    [E, 2048, 1408]

The layouts are exactly what autograd produces (see --show-calls).

Row counts. With expert parallelism every rank holds E_local = 64 / EP experts
and receives tokens from all EP ranks via variable all_to_all splits:
  * per-expert rows m_e range from 0 to EP * T (T = tokens per rank per
    microbatch = 4 * 2048 = 8192 in the recipe), because a token sends at most
    one copy to a given expert -> 0 .. 65536 for EP = 8;
  * the per-rank total M = sum(m_e) averages T * top_k = 49152 and changes on
    every call when EP > 1 (it is exactly T * top_k when EP = 1).

Presets (--preset):
  recipe   EP=8, 8192 tokens per rank: the torchtitan DSv3 16B test recipe
           (8 local experts, ~6k rows per expert)
  tp4ep4   EP=4, 512 tokens per rank: the measured layout from a torchtitan run
           with PPN=8 TP=4 EP=4 SEQ=2048 TOKENS=2048 (16 local experts, ~190
           rows per expert). Also enables --profiled.

--profiled adds cases built from the grouped_mm calls recorded in that run's
Kineto profile (steps 25-27, all 8 ranks; per-rank total M from 187 to 6361,
median 3053): one static case per quantile and one "step" case that replays
the 26 per-layer row counts of rank 0, step 25. The profile records total M
but not the per-expert split, so each total is split across the local
experts with the uniform router.

Modes:
  static      one routing reused every call (fixed shapes, best case)
  vary_split  new per-expert split every call, total M held fixed
  vary_total  new routing every call, so total M changes too (what training does)

Run:  python grouped_mm.dsv3_16b.py [--preset recipe|tp4ep4] [--ep N] [--tokens T] [--profiled]
                                   [--iters 20] [--check] [--print-dispatch] [--json out.json]

--print-dispatch shows (1) the dispatcher's kernel table for aten::_grouped_mm
(which backend key serves XPU, or that it falls through to the
CompositeExplicitAutograd per-group loop) and (2) for each op which XPU
implementation actually runs, using the backend's TORCH_XPU_GROUPED_MM_LOG
logging (SYCL-TLA plan: kernel, layout, tile, work-groups; or fallback and
the reason). TORCH_SHOW_DISPATCH_TRACE=1 only works in debug builds or builds
compiled with -DHAS_TORCH_SHOW_DISPATCH_TRACE; release wheels ignore it.
Env knobs of the XPU backend (if built):
  TORCH_XPU_GROUPED_MM_BACKEND=auto|sycltla|onednn|fallback   force a backend
  TORCH_XPU_GROUPED_MM_TILE=1|2|3|4 (256x128|256x256|128x128|128x256), TORCH_XPU_GROUPED_MM_NUM_WORKGROUPS=N
                                 SYCL-TLA tuning
  TORCH_XPU_GROUPED_MM_LOG=1|2   print the backend decision per distinct call / every call
For kernel tuning without PyTorch see sycltla_grouped_mm_tune/.
"""

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

# DeepSeek-V3 16B (torchtitan/models/deepseek_v3/flavors.py::_16b)
DIM = 2048
MOE_HIDDEN = 1408
NUM_EXPERTS = 64
TOP_K = 6
SEQ_LEN = 2048
SEQS_PER_MICROBATCH = 4  # num_tokens_per_microbatch_per_dp_rank = 4 * seq_len

OPS = ["w13_fwd", "w2_fwd", "w13_dX", "w2_dX", "w13_dW", "w2_dW"]

PRESETS = {
    "recipe": {"ep": 8, "tokens": SEQS_PER_MICROBATCH * SEQ_LEN},
    # TOKENS=2048 per DP group, sequence parallel over TP=4 -> 512 tokens per rank
    "tp4ep4": {"ep": 4, "tokens": 512},
}

# grouped_mm row counts measured with Kineto in torchtitan-githubsgi, DSv3 16B,
# PPN=8 TP=4 EP=4 SEQ=2048 TOKENS=2048, steps 25-27, all 8 ranks (2026-10-23,
# torch216_sycltla_ggemm_p312). M = rows routed to one rank, over its 16 experts.
PROFILED_TP4EP4 = {
    "ep": 4,
    "quantiles": {
        "min": 187,
        "p10": 2296,
        "p25": 2794,
        "p50": 3053,
        "p75": 3355,
        "p90": 3846,
        "max": 6361,
    },
    # per-layer M of the 26 MoE layers, rank 0, training step 25
    "step": [
        5288,
        3626,
        3526,
        3015,
        3559,
        3075,
        3806,
        2667,
        3617,
        3742,
        3035,
        2912,
        2966,
        3563,
        2964,
        2865,
        3310,
        2539,
        3088,
        3005,
        2856,
        3432,
        2991,
        2945,
        3099,
        2965,
    ],
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="recipe",
        help="EP / tokens layout (see the module docstring)",
    )
    p.add_argument(
        "--profiled",
        action="store_true",
        help="add the cases measured in the torchtitan TP4/EP4 profile",
    )
    p.add_argument(
        "--ep",
        type=int,
        default=None,
        help="expert-parallel degree (default: from --preset)",
    )
    p.add_argument(
        "--tokens",
        type=int,
        default=None,
        help="tokens per rank per microbatch, T (default: from --preset)",
    )
    p.add_argument("--routings", default="balanced,uniform,skewed,hot,sparse")
    p.add_argument("--modes", default="static,vary_split,vary_total")
    p.add_argument(
        "--sweep",
        default="0,8,64,256,1024,2048,4096,6144,12288,24576,49152",
        help="equal rows per local expert for the M sweep ('' to skip)",
    )
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="xpu")
    p.add_argument("--check", action="store_true", help="verify vs per-expert torch.mm")
    p.add_argument(
        "--show-calls",
        action="store_true",
        help="print the _grouped_mm calls autograd issues for one MoE block",
    )
    p.add_argument(
        "--print-dispatch",
        action="store_true",
        help="print the dispatcher table for aten::_grouped_mm and, per op and "
        "routing, which XPU implementation runs (SYCL-TLA plan or fallback "
        "+ reason) via TORCH_XPU_GROUPED_MM_LOG",
    )
    p.add_argument("--json", default="", help="write results to this file")
    return p.parse_args()


# --------------------------------------------------------------------------
# Routing generators: return int64 per-local-expert row counts (rank 0's slice)
# --------------------------------------------------------------------------


def _topk_counts(logits_GE, e_local):
    ids = logits_GE.topk(TOP_K, dim=-1).indices  # [G, TOP_K], distinct per token
    counts = torch.bincount(ids.flatten(), minlength=NUM_EXPERTS)
    return counts[:e_local].clone()


def route(kind, g_tokens, e_local, gen):
    if kind == "balanced":
        return torch.full(
            (e_local,), g_tokens * TOP_K // NUM_EXPERTS, dtype=torch.int64
        )
    if kind == "uniform":  # router logits i.i.d. -> multinomial-like loads
        return _topk_counts(torch.rand(g_tokens, NUM_EXPERTS, generator=gen), e_local)
    if kind == "skewed":  # Zipf-like expert popularity, random order
        rank = torch.randperm(NUM_EXPERTS, generator=gen).float() + 1.0
        bias = -1.2 * torch.log(rank)
        noise = -torch.log(-torch.log(torch.rand(g_tokens, NUM_EXPERTS, generator=gen)))
        return _topk_counts(bias + noise, e_local)
    if kind == "hot":  # every token picks local expert 0 (its maximum, EP * T rows)
        logits = torch.rand(g_tokens, NUM_EXPERTS, generator=gen)
        logits[:, 0] = 2.0
        return _topk_counts(logits, e_local)
    if kind == "sparse":  # half the local experts receive nothing
        logits = torch.rand(g_tokens, NUM_EXPERTS, generator=gen)
        dead = torch.randperm(e_local, generator=gen)[: e_local // 2]
        logits[:, dead] = -1.0
        return _topk_counts(logits, e_local)
    raise ValueError(f"unknown routing {kind}")


def split_like_router(total, g_tokens, e_local, gen):
    """Split a measured per-rank total M across the local experts with the
    proportions of a uniform-router draw (the profile has totals only)."""
    if total == 0:
        return torch.zeros(e_local, dtype=torch.int64)
    w = route("uniform", g_tokens, e_local, gen).double()
    w = (
        w / w.sum()
        if w.sum() > 0
        else torch.full((e_local,), 1.0 / e_local, dtype=torch.double)
    )
    raw = w * total
    c = raw.floor().to(torch.int64)
    rem = total - int(c.sum())
    if rem:
        c[torch.argsort(raw - c.double(), descending=True)[:rem]] += 1
    return c


def resplit(total, e_local, gen):
    """Random split of a fixed total into e_local non-negative counts."""
    cuts = torch.sort(torch.randint(0, total + 1, (e_local - 1,), generator=gen)).values
    edges = torch.cat([torch.tensor([0]), cuts, torch.tensor([total])])
    return (edges[1:] - edges[:-1]).to(torch.int64)


# --------------------------------------------------------------------------
# Problem setup
# --------------------------------------------------------------------------


class Problem:
    def __init__(self, e_local, m_max, device, dtype=torch.bfloat16):
        g = torch.Generator(device="cpu").manual_seed(1234)

        def rnd(*shape):
            return (torch.randn(*shape, generator=g) * 0.5).to(dtype).to(device)

        self.w13 = rnd(e_local, 2 * MOE_HIDDEN, DIM)  # weight_EOI
        self.w2 = rnd(e_local, DIM, MOE_HIDDEN)
        self.x = rnd(m_max, DIM)
        self.h = rnd(m_max, MOE_HIDDEN)
        self.dg = rnd(m_max, 2 * MOE_HIDDEN)
        self.dy = rnd(m_max, DIM)
        self.device = device

    def op(self, name, m, offs):
        x, h, dg, dy = self.x[:m], self.h[:m], self.dg[:m], self.dy[:m]
        if name == "w13_fwd":
            return torch._grouped_mm(x, self.w13.transpose(-2, -1), offs=offs)
        if name == "w2_fwd":
            return torch._grouped_mm(h, self.w2.transpose(-2, -1), offs=offs)
        if name == "w13_dX":
            return torch._grouped_mm(dg, self.w13, offs=offs)
        if name == "w2_dX":
            return torch._grouped_mm(dy, self.w2, offs=offs)
        if name == "w13_dW":
            return torch._grouped_mm(dg.transpose(-2, -1), x, offs=offs)
        if name == "w2_dW":
            return torch._grouped_mm(dy.transpose(-2, -1), h, offs=offs)
        raise ValueError(name)

    @staticmethod
    def flops(name, m):
        k_n = {
            "w13_fwd": DIM * 2 * MOE_HIDDEN,
            "w2_fwd": MOE_HIDDEN * DIM,
            "w13_dX": 2 * MOE_HIDDEN * DIM,
            "w2_dX": DIM * MOE_HIDDEN,
            "w13_dW": 2 * MOE_HIDDEN * DIM,
            "w2_dW": DIM * MOE_HIDDEN,
        }[name]
        return 2.0 * m * k_n

    def reference(self, name, counts):
        """Per-expert torch.mm reference (fp32 accumulate)."""
        starts = [0]
        for c in counts[:-1]:
            starts.append(starts[-1] + int(c))
        outs = []
        for e, (s, c) in enumerate(zip(starts, counts.tolist())):
            sl = slice(s, s + c)
            if name == "w13_fwd":
                outs.append(self.x[sl].float() @ self.w13[e].float().t())
            elif name == "w2_fwd":
                outs.append(self.h[sl].float() @ self.w2[e].float().t())
            elif name == "w13_dX":
                outs.append(self.dg[sl].float() @ self.w13[e].float())
            elif name == "w2_dX":
                outs.append(self.dy[sl].float() @ self.w2[e].float())
            elif name == "w13_dW":
                outs.append((self.dg[sl].float().t() @ self.x[sl].float()).unsqueeze(0))
            elif name == "w2_dW":
                outs.append((self.dy[sl].float().t() @ self.h[sl].float()).unsqueeze(0))
        return torch.cat(outs, 0)


def offs_of(counts, device):
    return torch.cumsum(counts, 0).to(torch.int32).to(device)


def sync(device):
    getattr(torch, device).synchronize()


# --------------------------------------------------------------------------
# Measurements
# --------------------------------------------------------------------------


def time_calls(prob, name, schedule, warmup, device):
    """schedule: list of (m, offs). Returns (mean steady ms, cold first-call ms);
    (nan, nan) if the backend raises for any shape in the schedule."""
    try:
        return _time_calls(prob, name, schedule, warmup, device)
    except RuntimeError as e:
        print(f"     [{name}] backend error: {str(e).splitlines()[0][:120]}")
        return float("nan"), float("nan")


def _time_calls(prob, name, schedule, warmup, device):
    m0, o0 = schedule[0]
    sync(device)
    t = time.perf_counter()
    prob.op(name, m0, o0)
    sync(device)
    cold_ms = (time.perf_counter() - t) * 1e3
    for i in range(warmup):
        m, o = schedule[(i + 1) % len(schedule)]
        prob.op(name, m, o)
    sync(device)
    t = time.perf_counter()
    for m, o in schedule:
        prob.op(name, m, o)
    sync(device)
    steady_ms = (time.perf_counter() - t) * 1e3 / len(schedule)
    return steady_ms, cold_ms


def build_schedule(kind, mode, iters, g_tokens, e_local, gen, device):
    base = route(kind, g_tokens, e_local, gen)
    counts = []
    for i in range(iters):
        if mode == "static":
            c = base
        elif mode == "vary_split":
            c = (
                base
                if kind == "balanced" and i == 0
                else resplit(int(base.sum()), e_local, gen)
            )
        elif mode == "vary_total":
            c = (
                route(kind, g_tokens, e_local, gen)
                if kind != "balanced"
                else torch.full(
                    (e_local,), int(base[0]) + 8 * (i - iters // 2), dtype=torch.int64
                ).clamp(min=0)
            )
        else:
            raise ValueError(mode)
        counts.append(c)
    return [(int(c.sum()), offs_of(c, device)) for c in counts], counts


def check_ops(prob, counts, device):
    m, offs = int(counts.sum()), offs_of(counts, device)
    worst = {}
    for name in OPS:
        try:
            got = prob.op(name, m, offs).float()
        except RuntimeError as e:  # record backend failures, keep going
            worst[name] = (float("nan"), 0.0, False, str(e).splitlines()[0][:120])
            continue
        ref = prob.reference(name, counts)
        if name.endswith("_dW"):
            got = got[: ref.shape[0]]
        err = (got - ref).abs().max().item() if ref.numel() else 0.0
        scale = ref.abs().max().item() if ref.numel() else 1.0
        worst[name] = (err, scale, err <= 0.02 * scale + 0.1)
    return worst


def show_calls(prob, counts, device):
    from torch.utils._python_dispatch import TorchDispatchMode

    class Spy(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func is torch.ops.aten._grouped_mm.default:
                a, b = args[0], args[1]
                print(
                    f"  _grouped_mm a{tuple(a.shape)} stride{a.stride()}  "
                    f"b{tuple(b.shape)} stride{b.stride()}"
                )
            return func(*args, **(kwargs or {}))

    m, offs = int(counts.sum()), offs_of(counts, device)
    x = prob.x[:m].clone().requires_grad_(True)
    w13 = prob.w13.clone().requires_grad_(True)
    w2 = prob.w2.clone().requires_grad_(True)
    with Spy():
        g = torch._grouped_mm(x, w13.transpose(-2, -1), offs=offs)
        a, b = g.chunk(2, dim=-1)
        y = torch._grouped_mm(F.silu(a) * b, w2.transpose(-2, -1), offs=offs)
        y.backward(torch.ones_like(y))


def print_dispatch(prob, device, g_tokens, e_local, gen):
    """Dispatcher view + XPU implementation choice for each op."""
    import sys

    print("== dispatcher: kernels registered for aten::_grouped_mm")
    try:
        for line in torch._C._dispatch_dump("aten::_grouped_mm").splitlines():
            if line.strip():
                print("   " + line)
    except Exception as e:
        print("   (unavailable:", e, ")")
    print("== dispatcher: computed table (relevant keys)")
    try:
        keys = ("XPU", "AutogradXPU", "CompositeExplicitAutograd", "CPU", "CUDA")
        for line in torch._C._dispatch_dump_table("aten::_grouped_mm").splitlines():
            if line.split(":")[0].strip() in keys:
                print("   " + line)
    except Exception as e:
        print("   (unavailable:", e, ")")

    cases = {
        "uniform (DSv3 typical)": route("uniform", g_tokens, e_local, gen),
        "small (64 rows/expert)": torch.full((e_local,), 64, dtype=torch.int64),
        "all_empty": torch.zeros(e_local, dtype=torch.int64),
    }
    print(
        "== XPU implementation per call (TORCH_XPU_GROUPED_MM_LOG=2; lines come from "
        "the backend on stderr, none if this torch build has no XPU grouped_mm logger)"
    )
    prev = os.environ.get("TORCH_XPU_GROUPED_MM_LOG")
    os.environ["TORCH_XPU_GROUPED_MM_LOG"] = "2"
    try:
        for cname, counts in cases.items():
            m, offs = int(counts.sum()), offs_of(counts, device)
            for name in OPS:
                print(f"  -- {cname}: {name}  (total M={m})", flush=True)
                try:
                    prob.op(name, m, offs)
                    sync(device)
                except RuntimeError as e:
                    print(
                        f"     backend error: {str(e).splitlines()[0][:120]}",
                        flush=True,
                    )
                sys.stderr.flush()
    finally:
        if prev is None:
            os.environ.pop("TORCH_XPU_GROUPED_MM_LOG", None)
        else:
            os.environ["TORCH_XPU_GROUPED_MM_LOG"] = prev


def backend_info(device):
    info = {"torch": torch.__version__}
    try:
        info["xpu_kernel"] = torch._C._dispatch_has_kernel_for_dispatch_key(
            "aten::_grouped_mm", device.upper()
        )
    except Exception:
        info["xpu_kernel"] = "unknown"
    lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    info["sycltla_lib"] = any("sycltla" in f for f in os.listdir(lib))
    for k in (
        "TORCH_XPU_GROUPED_MM_BACKEND",
        "TORCH_XPU_GROUPED_MM_TILE",
        "TORCH_XPU_GROUPED_MM_NUM_WORKGROUPS",
    ):
        if os.environ.get(k):
            info[k] = os.environ[k]
    if device == "xpu":
        info["device_name"] = torch.xpu.get_device_name(0)
    return info


def main():
    a = parse_args()
    dev = a.device
    preset = PRESETS[a.preset]
    a.ep = a.ep if a.ep is not None else preset["ep"]
    a.tokens = a.tokens if a.tokens is not None else preset["tokens"]
    profiled = a.profiled or a.preset == "tp4ep4"
    if profiled and a.ep != PROFILED_TP4EP4["ep"]:
        print(
            f"== note: profiled cases were measured with EP={PROFILED_TP4EP4['ep']}, "
            f"running them with EP={a.ep}"
        )
    assert NUM_EXPERTS % a.ep == 0
    e_local = NUM_EXPERTS // a.ep
    g_tokens = a.ep * a.tokens
    gen = torch.Generator(device="cpu").manual_seed(a.seed)
    sweep = [int(s) for s in a.sweep.split(",") if s.strip()]

    # Upper bound on rows we may need: hot routing or the largest sweep point.
    # Routed cases never exceed all routed copies, but the sweep's synthetic
    # equal-rows points can; size for them too, since offs past the end of a
    # tensor makes the kernels read and write out of bounds.
    m_max = min(g_tokens * min(TOP_K, e_local), g_tokens * TOP_K)
    m_max = max([m_max] + [s * e_local for s in sweep])
    if profiled:
        m_max = max(
            m_max, PROFILED_TP4EP4["quantiles"]["max"], *PROFILED_TP4EP4["step"]
        )
    prob = Problem(e_local, m_max, dev)

    info = backend_info(dev)
    info.update(
        preset=a.preset,
        profiled=profiled,
        ep=a.ep,
        e_local=e_local,
        tokens_per_rank=a.tokens,
        global_tokens=g_tokens,
        per_expert_max=g_tokens,
        total_rows_mean=a.tokens * TOP_K,
    )
    print("== backend:", json.dumps(info))
    results = {"info": info, "routing": [], "sweep": [], "profiled": [], "check": {}}

    if a.print_dispatch:
        print_dispatch(prob, dev, g_tokens, e_local, gen)

    if a.show_calls:
        print("== _grouped_mm calls autograd issues for one MoE expert block:")
        show_calls(prob, route("uniform", g_tokens, e_local, gen), dev)

    if a.check:
        cases = {
            k: route(k, g_tokens, e_local, gen) for k in ("uniform", "sparse", "hot")
        }
        cases["all_empty"] = torch.zeros(
            e_local, dtype=torch.int64
        )  # rank got no tokens
        for kind, counts in cases.items():
            worst = check_ops(prob, counts, dev)
            results["check"][kind] = {k: list(v) for k, v in worst.items()}
            bad = [k for k, v in worst.items() if not v[2]]
            print(
                f"== check[{kind}]: {'OK' if not bad else 'FAIL ' + ','.join(bad)}  "
                + " ".join(f"{k}={v[0]:.3g}/{v[1]:.3g}" for k, v in worst.items())
            )
            for k, v in worst.items():
                if len(v) > 3:
                    print(f"     {k}: {v[3]}")

    hdr = (
        f"{'routing':9s} {'mode':10s} {'rows/expert min..max':>22s} {'total M':>15s}  "
        + "  ".join(f"{op:>14s}" for op in OPS)
        + "   all6_ms  cold_max_ms"
    )
    routings = [r for r in a.routings.split(",") if r]
    if routings and a.modes:
        print("\n== per-op steady time ms (TFLOPS); cold = first call on a new shape")
        print(hdr)
    for kind in routings:
        for mode in [m for m in a.modes.split(",") if m]:
            sched, counts = build_schedule(
                kind, mode, a.iters, g_tokens, e_local, gen, dev
            )
            cmin = min(int(c.min()) for c in counts)
            cmax = max(int(c.max()) for c in counts)
            tmin = min(m for m, _ in sched)
            tmax = max(m for m, _ in sched)
            cells, total_ms, cold_max = [], 0.0, 0.0
            row = {
                "routing": kind,
                "mode": mode,
                "rows_min": cmin,
                "rows_max": cmax,
                "total_min": tmin,
                "total_max": tmax,
                "ops": {},
            }
            for name in OPS:
                ms, cold = time_calls(prob, name, sched, a.warmup, dev)
                flop = sum(prob.flops(name, m) for m, _ in sched) / len(sched)
                tf = flop / (ms * 1e-3) / 1e12 if ms > 0 else float("nan")
                cells.append(f"{ms:7.3f} ({tf:5.1f})")
                total_ms += ms
                cold_max = cold if cold != cold else max(cold_max, cold)
                row["ops"][name] = {"ms": ms, "tflops": tf, "cold_ms": cold}
            row["all6_ms"], row["cold_max_ms"] = total_ms, cold_max
            results["routing"].append(row)
            print(
                f"{kind:9s} {mode:10s} {f'{cmin}..{cmax}':>22s} {f'{tmin}..{tmax}':>15s}  "
                + "  ".join(f"{c:>14s}" for c in cells)
                + f"  {total_ms:8.3f}  {cold_max:10.1f}"
            )

    if sweep:
        print("\n== M sweep: equal rows per local expert, static shapes")
        print(
            f"{'rows/expert':>11s} {'total M':>8s}  "
            + "  ".join(f"{op:>14s}" for op in OPS)
        )
        for m_e in sweep:
            counts = torch.full((e_local,), m_e, dtype=torch.int64)
            sched = [(int(counts.sum()), offs_of(counts, dev))] * a.iters
            cells, row = (
                [],
                {"rows_per_expert": m_e, "total": int(counts.sum()), "ops": {}},
            )
            for name in OPS:
                ms, cold = time_calls(prob, name, sched, a.warmup, dev)
                tf = (
                    prob.flops(name, sched[0][0]) / (ms * 1e-3) / 1e12
                    if ms > 0
                    else float("nan")
                )
                cells.append(f"{ms:7.3f} ({tf:5.1f})")
                row["ops"][name] = {"ms": ms, "tflops": tf, "cold_ms": cold}
            results["sweep"].append(row)
            print(
                f"{m_e:11d} {int(counts.sum()):8d}  "
                + "  ".join(f"{c:>14s}" for c in cells)
            )

    if profiled:
        print(
            "\n== profiled shapes (torchtitan DSv3 16B TP4/EP4 Kineto, steps 25-27); "
            "per-op ms (TFLOPS)"
        )
        print(
            f"{'case':10s} {'total M':>13s} {'rows/expert':>13s}  "
            + "  ".join(f"{op:>14s}" for op in OPS)
            + "   all6_ms"
        )
        # own generator so the splits do not depend on which other sections ran
        pgen = torch.Generator(device="cpu").manual_seed(a.seed + 1)
        pcases = [(k, [v]) for k, v in PROFILED_TP4EP4["quantiles"].items()]
        pcases.append(("step25", PROFILED_TP4EP4["step"]))
        for label, totals in pcases:
            counts = [split_like_router(t, g_tokens, e_local, pgen) for t in totals]
            reps = max(1, a.iters // len(totals)) if len(totals) > 1 else a.iters
            sched = [(int(c.sum()), offs_of(c, dev)) for c in counts] * reps
            tmin, tmax = min(totals), max(totals)
            cmin = min(int(c.min()) for c in counts)
            cmax = max(int(c.max()) for c in counts)
            cells, total_ms = [], 0.0
            row = {
                "case": label,
                "totals": totals,
                "rows_min": cmin,
                "rows_max": cmax,
                "ops": {},
            }
            for name in OPS:
                ms, cold = time_calls(prob, name, sched, a.warmup, dev)
                flop = sum(prob.flops(name, m) for m, _ in sched) / len(sched)
                tf = flop / (ms * 1e-3) / 1e12 if ms > 0 else float("nan")
                cells.append(f"{ms:7.3f} ({tf:5.1f})")
                total_ms += ms
                row["ops"][name] = {"ms": ms, "tflops": tf, "cold_ms": cold}
            row["all6_ms"] = total_ms
            results["profiled"].append(row)
            tot = f"{tmin}" if tmin == tmax else f"{tmin}..{tmax}"
            print(
                f"{label:10s} {tot:>13s} {f'{cmin}..{cmax}':>13s}  "
                + "  ".join(f"{c:>14s}" for c in cells)
                + f"  {total_ms:8.3f}"
            )
        print(
            "(step25 replays the 26 per-layer totals in order; times are per call, averaged)"
        )

    if a.json:
        with open(a.json, "w") as f:
            json.dump(results, f, indent=1)
        print("\n== wrote", a.json)


if __name__ == "__main__":
    main()
