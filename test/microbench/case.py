# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Case model, input helpers and case discovery.

Nothing here imports torch at module load, so listing and selecting cases
works without a wheel: case modules reach torch through lazy proxies.
"""

import hashlib
import importlib
import itertools
import json
import math
import pkgutil
import sys
from types import SimpleNamespace

from . import CONFIG_DIR, load_yaml


class LazyModule:
    def __init__(self, name):
        self._name = name
        self._mod = None

    def __getattr__(self, attr):
        if attr.startswith("__") and attr.endswith("__"):
            raise AttributeError(attr)
        if self._mod is None:
            self._mod = importlib.import_module(self._name)
        return getattr(self._mod, attr)


torch = LazyModule("torch")
F = LazyModule("torch.nn.functional")
nn = LazyModule("torch.nn")
torchvision = LazyModule("torchvision")

# ---------------------------------------------------------------------------
# Case model
# ---------------------------------------------------------------------------
# A case module defines one ``OpBench`` subclass per benchmarked op and pass
# set; every (params, pass) pair becomes a ``Case`` with a stable ``case_id``.
#
# * ``setup`` builds inputs, outside the timed region, once per input set.
# * ``fwd`` calls the op and nothing else: it is the timed region of a fwd
#   case.
# * ``bwd`` cases build the graph and ``grad_out`` in setup; the timed region
#   is ``torch.autograd.grad(out, inputs, grad_out, retain_graph=True)``, or an
#   overridden ``bwd`` that calls an aten ``*_backward`` op directly.
# * ``reset`` restores the inputs of in-place ops before every iteration.

FLOAT_DTYPES = ("bf16", "fp16", "fp32")
DTYPE_NAMES = {
    "bf16": "bfloat16",
    "fp16": "float16",
    "fp32": "float32",
    "fp64": "float64",
    "c32": "complex32",
    "c64": "complex64",
    "c128": "complex128",
    "int64": "int64",
    "int32": "int32",
    "int16": "int16",
    "int8": "int8",
    "uint8": "uint8",
    "bool": "bool",
}
SHORT_DTYPE = {v: k for k, v in DTYPE_NAMES.items()}
NOISE_CLASSES = ("normal", "atomic", "random", "host_bound")


def to_dtype(name):
    return getattr(torch, DTYPE_NAMES[name]) if isinstance(name, str) else name


def dtype_name(dtype):
    full = str(dtype).replace("torch.", "")
    return SHORT_DTYPE.get(full, full)


class State(SimpleNamespace):
    """Per-input-set state returned by ``setup``."""


def grid(**axes):
    """Cartesian product of parameter axes, in declaration order."""
    return [dict(zip(axes, values)) for values in itertools.product(*axes.values())]


def with_dtypes(cases, dtypes=FLOAT_DTYPES):
    """One param dict per (case, dtype); cases that fix ``dtype`` are kept."""
    return [
        dict(c, dtype=dt)
        for c in cases
        for dt in ([c["dtype"]] if "dtype" in c else dtypes)
    ]


def _normalize(value):
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _normalize(v) for k, v in sorted(value.items())}
    # json keeps 2.0 and 2 apart, so float and int scalars hash differently.
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"unsupported param value {value!r} ({type(value).__name__})")


def canonical_params(params):
    return json.dumps(_normalize(params), sort_keys=True, separators=(",", ":"))


# Per-case logs of a run live in ``logs/<op>/<pass>_<case_id>/`` and every
# file starts with ``<op>_<pass>_``; ``logs/index.csv`` maps each case_id (a
# hash of op, pass and params) to its parameters.


def case_log_dir(op, pass_, case_id):
    """Log directory of a case, relative to ``logs/``."""
    return f"{op}/{pass_}_{case_id}"


def case_log_name(op, pass_, *parts):
    """``<op>_<pass>_<parts...>``: the name of a file in ``case_log_dir``."""
    return "_".join((op, pass_, *parts))


def make_case_id(op, pass_, params):
    text = f"{op}|{pass_}|{canonical_params(params)}"
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def _is_shape_list(v):
    return (
        isinstance(v, (tuple, list))
        and len(v) > 4
        and all(x is None or isinstance(x, (tuple, list)) for x in v)
    )


def format_params(params):
    """Human readable case label; long tensor lists are summarized."""
    parts = []
    for k, v in params.items():
        if k == "shapes" and _is_shape_list(v):
            numel = sum(math.prod(sh) for sh in v) / 1e6
            v = f"[{len(v)} tensors, {numel:.1f}M elements, first {tuple(v[0])}]"
        elif _is_shape_list(v):
            v = f"[{sum(x is not None for x in v)} of {len(v)} non-contiguous]"
        parts.append(f"{k}={v}")
    return ", ".join(parts)


class OpBench:
    """Base class of a benchmarked op. See the section comment above."""

    op = None
    family = None
    passes = ("fwd",)
    # aten ops (no ``aten::`` prefix) each pass must run on the device; 'a|b'
    # accepts either. The validator fails a case that runs none of them.
    expected = {}
    noise_class = "normal"
    inplace = False
    # no XPU kernel of its own; ``expected`` lists the XPU ops it decomposes into
    composite = False
    # False keeps the op out of PR gating (nightly report only), e.g. oneDNN ops
    gating = True
    params = ()
    pr = ()

    def setup(self, p, device):
        raise NotImplementedError

    def fwd(self, s):
        raise NotImplementedError

    def reset(self, s):
        """Restore the inputs of an in-place op."""

    def passes_for(self, p):
        return self.passes

    def case_inplace(self, p):
        return self.inplace

    def expected_ops(self, p, pass_):
        return tuple(self.expected.get(pass_, ()))

    def expect_inputs(self, p, pass_):
        """Tensor arguments the first expected op must receive.

        A list of ``{shape, dtype[, stride]}`` dicts (``None`` entries are not
        checked) or ``None`` to skip the check. Default: the first tensor has
        ``p["shape"]``, ``p["in_dtype"]`` (else ``p["dtype"]``) and, if given,
        ``p["in_stride"]``.
        """
        if pass_ != "fwd" or "dtype" not in p:
            return None
        if not isinstance(p.get("shape"), (tuple, list)):
            return None
        spec = {"shape": tuple(p["shape"]), "dtype": p.get("in_dtype", p["dtype"])}
        if "in_stride" in p:
            spec["stride"] = tuple(p["in_stride"])
        return [spec]

    def grad_inputs(self, s):
        """Tensors a ``bwd`` case differentiates with respect to."""
        return s.inputs if hasattr(s, "inputs") else [s.x]

    def setup_bwd(self, s):
        inputs = self.grad_inputs(s)
        for t in inputs:
            t.requires_grad_(True)
        out = self.fwd(s)
        s._bwd_out = out[0] if isinstance(out, (tuple, list)) else out
        s._bwd_inputs = list(inputs)
        s._bwd_grad = torch.randn_like(s._bwd_out)
        return s

    def bwd(self, s):
        return torch.autograd.grad(
            s._bwd_out, s._bwd_inputs, s._bwd_grad, retain_graph=True
        )


class Case:
    """One (op, params, pass) benchmark unit.

    ``case_code_hash`` hashes the source of the case module: a change to
    ``setup``/``reset``/``run`` keeps the ``case_id`` but invalidates its
    reference records.
    """

    def __init__(self, bench, params, pass_, module=None, code_hash=""):
        self.bench = bench
        self.op = bench.op
        self.family = bench.family or bench.op
        self.params = params
        self.pass_ = pass_
        self.module = module
        self.case_code_hash = code_hash
        self.case_id = make_case_id(self.op, pass_, params)
        key = canonical_params(params)
        is_pr = bench.gating and any(canonical_params(q) == key for q in bench.pr)
        self.tier = "pr" if is_pr else "nightly"
        self.expected_aten_ops = bench.expected_ops(params, pass_)
        self.inplace = bench.case_inplace(params)
        self.label = format_params(params)

    def repro(self):
        params = canonical_params(self.params)
        return f"python -m microbench run --op {self.op} --pass {self.pass_} --case '{params}'"

    def manifest(self):
        """JSON-serializable description, consumed by torch-free tooling."""
        b = self.bench
        return {
            "case_id": self.case_id,
            "op": self.op,
            "family": self.family,
            "pass": self.pass_,
            "params": json.loads(canonical_params(self.params)),
            "label": self.label,
            "tier": self.tier,
            "expected_aten_ops": list(self.expected_aten_ops),
            "inplace": self.inplace,
            "noise_class": b.noise_class,
            "composite": b.composite,
            "gating": b.gating,
            "module": self.module,
            "case_code_hash": self.case_code_hash,
            "repro": self.repro(),
        }

    def build(self, device):
        s = self.bench.setup(self.params, device)
        return self.bench.setup_bwd(s) if self.pass_ == "bwd" else s

    def run(self, s):
        return self.bench.bwd(s) if self.pass_ == "bwd" else self.bench.fwd(s)

    def reset(self, s):
        if self.inplace:
            self.bench.reset(s)

    def __repr__(self):
        return f"Case({self.op}/{self.pass_} {self.label} id={self.case_id})"


# ---------------------------------------------------------------------------
# aten op names
# ---------------------------------------------------------------------------


def base_name(op):
    """'aten::add_.Tensor' -> 'add_' (namespace and overload stripped)."""
    return op.split("::", 1)[-1].split(".", 1)[0]


def functional_name(op):
    """Fold in-place and out variants onto the functional op: relu_ -> relu."""
    name = base_name(op)
    if name.startswith("__") and name.endswith("__"):
        return name
    return name.removesuffix("_out").removesuffix("_")


def alternatives(expected):
    """Functional names of an ``expected_aten_ops`` entry ('a|b' = either)."""
    return [functional_name(a.strip()) for a in expected.split("|") if a.strip()]


def expected_names(entries):
    return {name for e in entries for name in alternatives(e)}


# ---------------------------------------------------------------------------
# Input construction for setup()
# ---------------------------------------------------------------------------
# The protocol seeds the global RNG before every ``setup``, so inputs are
# deterministic per (case, input set). Values that drive performance
# (indices, masks, repeats, targets) are drawn on the CPU and copied over:
# the CPU generator is not part of torch-xpu-ops, so a PR that changes an XPU
# RNG kernel cannot change the A/B inputs.


def _strided(make, shape, stride):
    """``make(size)`` reshaped to ``stride`` (non-contiguous / broadcast)."""
    if stride is None:
        return make(tuple(shape))
    numel = 0 if 0 in shape else 1 + sum((d - 1) * s for d, s in zip(shape, stride))
    return make((numel,)).as_strided(tuple(shape), tuple(stride))


def randn(shape, dtype, device, stride=None, scale=None, offset=None):
    dt = to_dtype(dtype)

    def make(size):
        if dt == torch.complex32:  # no RNG kernel for chalf
            t = torch.randn(size, dtype=torch.complex64, device=device).to(dt)
        else:
            t = torch.randn(size, dtype=dt, device=device)
        if scale is not None:
            t.mul_(scale)
        if offset is not None:
            t.add_(offset)
        return t

    return _strided(make, shape, stride)


def rand(shape, dtype, device, low=0.0, high=1.0, stride=None):
    def make(size):
        return torch.empty(size, dtype=to_dtype(dtype), device=device).uniform_(
            low, high
        )

    return _strided(make, shape, stride)


def randint(low, high, shape, device, dtype="int64", stride=None):
    """Integers in [low, high), drawn on the CPU."""

    def make(size):
        return torch.randint(low, high, size, dtype=to_dtype(dtype)).to(device)

    return _strided(make, shape, stride)


def mask(shape, p_true, device, stride=None):
    """Bool tensor with a fixed True ratio, drawn on the CPU."""
    return _strided(lambda size: (torch.rand(size) < p_true).to(device), shape, stride)


def nonfinite(shape, dtype, device, frac=0.01, stride=None):
    """randn with ``frac`` NaN, +inf and -inf each, at fixed positions."""

    def make(size):
        t = torch.randn(size)
        flat = t.view(-1)
        k = max(1, int(flat.numel() * frac))
        idx = torch.randperm(flat.numel())
        flat[idx[:k]] = float("nan")
        flat[idx[k : 2 * k]] = float("inf")
        flat[idx[2 * k : 3 * k]] = float("-inf")
        return t.to(to_dtype(dtype)).to(device)

    return _strided(make, shape, stride)


def tensor(shape, dtype, device, stride=None, values=None):
    """Input of any dtype: normal floats/complex, 50% True bools, ints in [0, 100).

    ``values`` overrides: ``("uniform", lo, hi)``, ``("int", lo, hi)``,
    ``("mask", p_true)`` or ``("nonfinite", frac)``.
    """
    kind = values[0] if values else None
    if kind == "uniform":
        return rand(shape, dtype, device, values[1], values[2], stride=stride)
    if kind == "int":
        return randint(values[1], values[2], shape, device, dtype=dtype, stride=stride)
    if kind == "mask":
        return mask(shape, values[1], device, stride=stride)
    if kind == "nonfinite":
        return nonfinite(shape, dtype, device, values[1], stride=stride)
    if dtype == "bool":
        return mask(shape, 0.5, device, stride=stride)
    if dtype.startswith(("int", "uint")):
        return randint(0, 100, shape, device, dtype=dtype, stride=stride)
    return randn(shape, dtype, device, stride=stride)


def full(shape, value, dtype, device):
    return torch.full(tuple(shape), value, dtype=to_dtype(dtype), device=device)


def cpu_to(t, device, dtype=None):
    """Copy a CPU-built tensor to the device (optionally casting)."""
    return (t if dtype is None else t.to(to_dtype(dtype))).to(device)


def nd_input(shape, dtype, device, channels_last=False):
    """randn input, optionally in channels_last(_3d) memory format."""
    x = randn(shape, dtype, device)
    if channels_last:
        fmt = torch.channels_last if len(shape) == 4 else torch.channels_last_3d
        x = x.contiguous(memory_format=fmt)
    return x


# ---------------------------------------------------------------------------
# Table-driven cases
# ---------------------------------------------------------------------------


class TableBench(OpBench):
    """OpBench whose ``setup()`` follows the keys of a param dict.

    * ``shape``, ``in_stride``, ``in_dtype``: main input ``s.x``;
    * ``shapes``, ``strides``: tensor-list input ``s.xs``; ``lists`` names more
      list arguments with the same shapes (``s.tensor1``, ...);
    * ``<arg>_shape``, ``<arg>_stride``, ``<arg>_dtype``: tensor ``s.<arg>``;
    * any other key: scalar ``s.<key>``; ``defaults`` covers keys that only
      some cases set. ``scalars`` of a list op becomes one value per tensor.

    ``values`` maps an input name to a value rule of ``tensor()`` (or to a
    function of the params returning one). ``inplace = True``, or a case with
    ``inplace=True``, restores ``s.x``/``s.xs`` before every iteration unless
    the op is ``idempotent``. ``factory`` ops get ``s.dtype`` and ``s.device``.
    Backward classes (``passes = ("bwd",)``) call the aten backward op
    directly in ``bwd()``.
    """

    lists = ()
    values = {}
    defaults = {}
    factory = False
    idempotent = False
    KEYS = {"shape", "in_stride", "in_dtype", "dtype", "shapes", "strides", "values"}

    def case_inplace(self, p):
        return not self.idempotent and bool(self.inplace or p.get("inplace"))

    def _tensor(self, name, shape, dtype, device, stride, p):
        rule = self.values.get(name)
        rule = rule(p) if callable(rule) else rule
        return tensor(shape, dtype, device, stride, rule)

    def setup(self, p, device):
        dt = p["dtype"]
        s = State(**self.defaults)
        if self.factory:
            s.dtype, s.device = to_dtype(dt), device
        if "shape" in p:
            in_dt = p.get("in_dtype", dt)
            s.x = self._tensor("x", p["shape"], in_dt, device, p.get("in_stride"), p)
        if "shapes" in p:
            strides = p.get("strides") or [None] * len(p["shapes"])
            for name in ("xs", *self.lists):
                setattr(
                    s,
                    name,
                    [
                        self._tensor(name, sh, dt, device, st, p)
                        for sh, st in zip(p["shapes"], strides)
                    ],
                )
        for k, v in p.items():
            if k.endswith("_shape"):
                arg = k.removesuffix("_shape")
                arg_dt = p.get(f"{arg}_dtype", dt)
                stride = p.get(f"{arg}_stride")
                setattr(s, arg, self._tensor(arg, v, arg_dt, device, stride, p))
            elif k not in self.KEYS and not k.endswith(("_stride", "_dtype")):
                setattr(s, k, v)
        if "scalars" in p and "shapes" in p:
            s.scalars = [p["scalars"]] * len(p["shapes"])
        s.inplace = bool(p.get("inplace"))
        if self.case_inplace(p):
            s.pristine = [t.clone() for t in s.xs] if "shapes" in p else s.x.clone()
        return s

    def reset(self, s):
        if isinstance(s.pristine, list):
            torch._foreach_copy_(s.xs, s.pristine)
        else:
            s.x.copy_(s.pristine)

    def setup_bwd(self, s):
        return s


def table_bench(op, cases, pr=(), dtypes=FLOAT_DTYPES, **attrs):
    """A one-op ``TableBench`` class; ``attrs`` sets ``fwd``/``bwd`` and other
    class attributes.

    ``params`` are ``cases`` in every one of ``dtypes`` (unless a case fixes one);
    ``pr`` indexes ``cases`` and gates bf16 (or the fixed dtype). ``expected``
    defaults to ``op`` for fwd and ``<op>_backward`` for bwd.
    """
    passes = attrs.get("passes", ("fwd",))
    ns = {
        "op": op,
        "expected": {p: (op if p == "fwd" else f"{op}_backward",) for p in passes},
        "params": with_dtypes(cases, dtypes),
        "pr": [dict(cases[i], dtype=cases[i].get("dtype", "bf16")) for i in pr],
        **attrs,
    }
    return type(op, (TableBench,), ns)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class RegistryError(Exception):
    pass


def iter_benches():
    """(module name, bench) for every ``OpBench`` with an ``op`` in ``cases/``."""
    pkg = importlib.import_module(f"{__package__}.cases")
    for info in sorted(pkgutil.iter_modules(pkg.__path__), key=lambda i: i.name):
        mod = importlib.import_module(f"{pkg.__name__}.{info.name}")
        for obj in vars(mod).values():
            if isinstance(obj, type) and issubclass(obj, OpBench) and obj.op:
                yield info.name, obj()


def _check_bench(where, bench):
    errors = []
    if bench.noise_class not in NOISE_CLASSES:
        errors.append(f"{where}: bad noise_class {bench.noise_class!r}")
    keys = {canonical_params(p) for p in bench.params}
    if not keys:
        errors.append(f"{where}: no params")
    if len(keys) != len(bench.params):
        errors.append(f"{where}: duplicate params")
    errors += [
        f"{where}: pr entry not in params: {q}"
        for q in bench.pr
        if canonical_params(q) not in keys
    ]
    for p in bench.params:
        for pass_ in bench.passes_for(p):
            if not bench.expected_ops(p, pass_):
                errors.append(f"{where}/{pass_}: no expected_aten_ops")
    return errors


def module_code_hash(mod_name):
    """sha1 (12 hex) of the source of ``cases/<mod_name>.py``."""
    path = sys.modules[f"{__package__}.cases.{mod_name}"].__file__
    with open(path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()[:12]


def always_ab_cases(path=None):
    data = load_yaml(path or CONFIG_DIR / "gating.yaml", {}) or {}
    entries = (data.get("pr_vs_reference") or {}).get("always_ab") or []
    return {e["case_id"] for e in entries}


def load_cases(strict=True):
    """``{case_id: Case}`` for every registered case."""
    cases, ops, errors, hashes = {}, {}, [], {}
    for mod_name, bench in iter_benches():
        if ops.setdefault(bench.op, mod_name) != mod_name:
            errors.append(f"op {bench.op!r} defined in {ops[bench.op]} and {mod_name}")
        errors += _check_bench(f"{mod_name}:{bench.op}", bench)
        if mod_name not in hashes:
            hashes[mod_name] = module_code_hash(mod_name)
        for p in bench.params:
            for pass_ in bench.passes_for(p):
                case = Case(
                    bench, dict(p), pass_, module=mod_name, code_hash=hashes[mod_name]
                )
                if case.case_id in cases:
                    errors.append(f"case_id collision: {case} vs {cases[case.case_id]}")
                cases[case.case_id] = case
    if errors and strict:
        raise RegistryError("\n".join(errors))
    return cases


def manifest(cases=None):
    """Case manifests for selection, with ``screen_mode`` from
    ``config/gating.yaml`` (``always_ab`` cases are never screened against
    reference data). Needs pyyaml, unlike ``load_cases`` (worker venvs)."""
    cases = load_cases() if cases is None else cases
    always_ab = always_ab_cases()
    return [
        dict(
            c.manifest(),
            screen_mode="always_ab" if c.case_id in always_ab else "reference",
        )
        for c in cases.values()
    ]
