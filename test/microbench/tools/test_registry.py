# Owner(s): ["module: intel"]
# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Config consistency check (run in the microbench-select CI job).

Cases must load without torch, keep stable case_ids and declare expected aten
ops; op_source_map.yaml (with the core set) must match the registered cases and
the source tree. Run: ``cd test && python -m pytest microbench/tools/test_registry.py``
"""

import fnmatch
import os
import sys
import warnings

import pytest
import yaml

# allow `python test/microbench/tools/test_registry.py` from anywhere
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from microbench import CONFIG_DIR, REPO_ROOT
from microbench.case import canonical_params, expected_names, load_cases, make_case_id
from microbench.run import lpt_shards, round_order
from microbench.selection import git_changes, protocol_note


@pytest.fixture(scope="module")
def cases():
    return load_cases()


def load(name):
    with open(os.path.join(CONFIG_DIR, name)) as f:
        return yaml.safe_load(f)


def test_registry_loads_without_torch_import(cases):
    import subprocess
    import sys

    assert len(cases) > 2000
    # listing cases must not import torch (ubuntu jobs) nor yaml (workers and
    # the validator run in the wheel's venv, which has no pyyaml)
    code = (
        "import sys; from microbench.case import load_cases; load_cases(); "
        "assert 'torch' not in sys.modules and 'yaml' not in sys.modules"
    )
    subprocess.run(
        [sys.executable, "-c", code], check=True, cwd=os.path.join(REPO_ROOT, "test")
    )


def test_case_id_is_stable():
    assert (
        make_case_id(
            "softmax", "fwd", {"shape": (8192, 8192), "dtype": "bf16", "dim": 1}
        )
        == "851d7b5871b5"
    )
    assert canonical_params({"b": (1, 2), "a": 2.0}) == '{"a":2.0,"b":[1,2]}'


def test_every_case_declares_expected_ops(cases):
    assert all(c.expected_aten_ops for c in cases.values())


def test_plan_ops_have_cases(cases):
    # forward and backward of one op share a module and op name, so check the
    # aten ops the cases execute
    ops = {c.op for c in cases.values()} | expected_names(
        e for c in cases.values() for e in c.expected_aten_ops
    )
    wanted = """abs addcmul bitwise_and bitwise_not bitwise_or clamp cos div eq exp floor ge gt isin isinf le log log2 lt
    maximum minimum mul nan_to_num ne neg pow reciprocal rsqrt rsub sgn sin sqrt sub where gelu gelu_backward hardsigmoid
    hardsigmoid_backward hardswish hardswish_backward hardtanh hardtanh_backward leaky_relu leaky_relu_backward relu sigmoid
    sigmoid_backward silu silu_backward all any argmax mean std _log_softmax _log_softmax_backward_data _safe_softmax
    nll_loss2d_forward nll_loss2d_backward cat repeat_interleave unfold_backward fill arange linspace _foreach_add
    _foreach_addcdiv _foreach_addcmul _foreach_div _foreach_lerp _foreach_mul _foreach_sqrt
    _scaled_dot_product_flash_attention _scaled_dot_product_flash_attention_backward nms roi_align _fft_c2c""".split()
    assert not set(wanted) - ops


def test_oneDNN_matmul_never_gates(cases):
    assert all(
        c.tier == "nightly" and not c.bench.gating
        for c in cases.values()
        if c.op == "matmul"
    )


def test_core_set(cases):
    srcmap = load("op_source_map.yaml")
    ops = {c.op for c in cases.values()}
    n = 0
    assert set(srcmap["core_set"]) == set(srcmap["common_groups"])
    for name, g in srcmap["core_set"].items():
        assert set(g["ops"]) <= ops, name
        for e in g["cases"]:
            c = cases[e["case_id"]]
            assert (
                c.tier == "pr"
                and c.bench.noise_class == "normal"
                and c.params.get("dtype") == "bf16"
            )
            n += 1
        assert len(g["cases"]) <= 2 and (g["cases"] or g.get("note"))
    assert 10 <= n <= 25


def test_op_source_map(cases):
    srcmap = load("op_source_map.yaml")
    ops = {c.op for c in cases.values()}
    assert set(srcmap["ops"]) <= ops
    assert ops <= set(srcmap["ops"])
    for op, globs in srcmap["ops"].items():
        for g in globs:
            hits = [p for p in _all_files() if fnmatch.fnmatch(p, g)]
            assert hits, f"{op}: {g} matches nothing"


def _all_files():
    out = []
    for root, _, files in os.walk(os.path.join(REPO_ROOT, "src")):
        out += [os.path.relpath(os.path.join(root, f), REPO_ROOT) for f in files]
    return out


def test_coverage_exclude_has_reasons():
    excl = load("coverage_exclude.yaml")
    assert excl["excluded"] and excl["deferred"]
    for section in ("excluded", "deferred"):
        assert all(isinstance(v, str) and v for v in excl[section].values())


def test_screening_config(cases):
    scr = load("gating.yaml")
    for e in scr["pr_vs_reference"]["always_ab"] or []:
        assert e["case_id"] in cases and e.get("reason"), e
    assert all(len(c.case_code_hash) == 12 for c in cases.values())


def test_protocol_version_bumped_with_measurement_changes():
    """Report only: a warning, never a failure. The select job
    sets MICROBENCH_DIFF_BASE to the merge-base."""
    base = os.environ.get("MICROBENCH_DIFF_BASE")
    if not base:
        pytest.skip("MICROBENCH_DIFF_BASE not set")
    try:
        changed, _, diff, _ = git_changes(str(REPO_ROOT), base, "HEAD")
    except Exception as e:  # noqa: BLE001 - report only
        pytest.skip(f"git diff unavailable: {e}")
    note = protocol_note(changed, diff)
    if note:
        warnings.warn(note, stacklevel=1)


def test_lpt_and_round_order():
    shards, load_ = lpt_shards(
        ["a", "b", "c", "d"], {"a": 10, "b": 9, "c": 2, "d": 1}, 2
    )
    assert sorted(load_) == [11, 11]
    assert [round_order(r) for r in range(3)] == [["A", "B"], ["B", "A"], ["A", "B"]]
    assert round_order(0, first="B") == ["B", "A"]
    sides = ("P", "Pp", "N")
    assert [round_order(r, "P", sides) for r in range(2)] == [
        ["P", "Pp", "N"],
        ["N", "Pp", "P"],
    ]


def run_tests(path):
    sys.exit(pytest.main([path, "-q"]))


if __name__ == "__main__":
    run_tests(__file__)
