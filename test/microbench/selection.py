# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Case selection: changed files -> cases, and new XPU op detection.

Shared by the PR ``microbench-select`` job and nightly/on-demand runs.
"""

import ast
import fnmatch
import json
import os
import re
import subprocess
import sys
from pathlib import PurePosixPath

from . import (
    CONFIG_DIR,
    DEFAULT_CASE_SECONDS,
    load_history,
    load_yaml,
    WORKER_START_SECONDS,
)
from .case import expected_names, functional_name

# ---------------------------------------------------------------------------
# New XPU op detection (plan 5.3.1, informational only)
# ---------------------------------------------------------------------------
# Lines added under src/: ``REGISTER_XPU_DISPATCH(foo_stub, ...)``,
# ``m.impl("name", ...)`` / ``TORCH_SELECTIVE_NAME("ns::name")``; and new
# kernel files under src/ATen/native/xpu/.

RE_DISPATCH = re.compile(r"REGISTER_XPU_DISPATCH\s*\(\s*(\w+?)(?:_stub)?\s*,")
RE_IMPL = re.compile(
    r'm\.impl\s*\(\s*(?:TORCH_SELECTIVE_NAME\s*\(\s*)?"(?:(\w+)::)?([\w.]+)"'
)
RE_SELECTIVE = re.compile(r'TORCH_SELECTIVE_NAME\s*\(\s*"(\w+)::([\w.]+)"')
KERNEL_DIR = "src/ATen/native/xpu/"


def parse_diff(diff_text):
    """Yield (file, added_line) for every added line of a unified diff."""
    cur = None
    for line in diff_text.splitlines():
        if line.startswith("+++ "):
            path = line[4:].strip()
            cur = None if path == "/dev/null" else path.removeprefix("b/")
        elif line.startswith("+") and cur:
            yield cur, line[1:]


def detect(diff_text, added_files=()):
    """New XPU ops: list of {op, where, kind}."""
    found = {}
    for path, line in parse_diff(diff_text):
        if not path.startswith("src/"):
            continue
        m = RE_DISPATCH.search(line)
        if m:
            found.setdefault(
                m.group(1),
                {"op": m.group(1), "where": path, "kind": "REGISTER_XPU_DISPATCH"},
            )
        m = RE_IMPL.search(line) or RE_SELECTIVE.search(line)
        if m:
            op = m.group(2).split(".", 1)[0]
            found.setdefault(op, {"op": op, "where": path, "kind": "m.impl"})
    for path in added_files:
        if path.startswith(KERNEL_DIR) and path.endswith((".cpp", ".h")):
            stem = re.sub(r"Kernels?$", "", PurePosixPath(path).stem)
            op = re.sub(r"(?<!^)(?=[A-Z])", "_", stem).lower()
            found.setdefault(op, {"op": op, "where": path, "kind": "new kernel file"})
    return sorted(found.values(), key=lambda d: d["op"])


def annotate(new_ops, covered_ops, excluded, deferred):
    """Add a human readable status to each detected op."""
    for n in new_ops:
        f = functional_name(n["op"])
        if f in covered_ops:
            n["status"] = "has microbench case"
        elif f in excluded:
            n["status"] = f"excluded ({excluded[f]})"
        elif f in deferred:
            n["status"] = f"deferred ({deferred[f]})"
        else:
            n["status"] = "not covered, not registered in coverage_exclude.yaml"
    return new_ops


# ---------------------------------------------------------------------------
# Case selection (plan 5.3.2, 6)
# ---------------------------------------------------------------------------
# | change                                  | mode | cases                                   |
# |-----------------------------------------|------|-----------------------------------------|
# | file mapped in op_source_map.yaml       | A/B  | PR-tier cases of the mapped ops         |
# | file of one common group                | A/B  | core set + all PR cases of that group   |
# | global build file / several groups      | A/B  | core set                                |
# | unmapped kernel file or new XPU op      | A/B  | core set (+ other mapped cases)         |
# | no_bench file                           | -    | nothing                                 |
# | only test/microbench/cases/**           | A/A  | the changed cases (validated first)     |
# | only microbench framework / CI          | A/A  | core set                                |
# | anything else (docs, UT, ...)           | -    | nothing                                 |
# | maintainer-added ``disable_microbench`` | -    | nothing                                 |

MB_CASES = "test/microbench/cases/"
MB_FRAMEWORK = (
    "test/microbench/*",
    ".github/workflows/_linux_microbench.yml",
    ".github/workflows/microbench_*.yml",
)
MB_IGNORED = ("test/microbench/README.md", "test/microbench/tools/test_registry.py")
SRC_PATTERNS = ("src/*", "cmake/*", "CMakeLists.txt")


def match_any(path, globs):
    return any(fnmatch.fnmatch(path, g) for g in globs or ())


class Config:
    def __init__(self, config_dir=None):
        d = config_dir or CONFIG_DIR
        self.srcmap = load_yaml(os.path.join(d, "op_source_map.yaml"), {})
        self.groups = (
            load_yaml(os.path.join(d, "core_set.yaml"), {}).get("groups") or {}
        )
        excl = load_yaml(os.path.join(d, "coverage_exclude.yaml"), {})
        self.excluded = excl.get("excluded") or {}
        self.deferred = excl.get("deferred") or {}
        self.costs = load_history(os.path.join(d, "history.json"))["case_cost"]
        self.budget = load_yaml(os.path.join(d, "gating.yaml"), {}).get("budget", {})

    def core_cases(self):
        return [
            e["case_id"] for g in self.groups.values() for e in g.get("cases") or []
        ]


def estimate_minutes(
    case_ids, costs, cards, screen_rounds, confirm_rounds, confirm_frac=0.2
):
    if not case_ids:
        return 0.0
    per_card = sum(costs.get(c, DEFAULT_CASE_SECONDS) for c in case_ids) / max(1, cards)
    screen = (per_card + WORKER_START_SECONDS) * screen_rounds * 2
    confirm = (per_card * confirm_frac + WORKER_START_SECONDS) * confirm_rounds * 2
    return (screen + confirm) / 60.0


def classify_files(changed):
    kinds = {"cases": [], "framework": [], "src": [], "other": []}
    for f in changed:
        if f.startswith(MB_CASES):
            kinds["cases"].append(f)
        elif match_any(f, MB_IGNORED):
            kinds["other"].append(f)
        elif match_any(f, MB_FRAMEWORK):
            kinds["framework"].append(f)
        elif match_any(f, SRC_PATTERNS):
            kinds["src"].append(f)
        else:
            kinds["other"].append(f)
    return kinds


def diff_hunks(diff_text):
    """Yield (file, first_line, n_lines) of the new side of every hunk."""
    path = None
    for line in diff_text.splitlines():
        if line.startswith("+++ "):
            path = line[4:].strip().removeprefix("b/")
        elif line.startswith("@@") and path:
            start, _, n = line.split("+", 1)[1].split()[0].partition(",")
            yield path, int(start), int(n or 1)


def touched_ops(source, lines):
    """Ops of a case module whose definitions overlap ``lines``; None = all.

    Only ``class X(...): op = "..."`` and ``X = factory("op", ...)`` count as
    op definitions (plus subclasses of a touched class); any other touched
    statement (imports, helpers, shared tables, base classes) affects the
    whole module.
    """
    defs, touched = {}, set()
    for node in ast.parse(source).body:
        name, op, parents = None, None, set()
        if isinstance(node, ast.ClassDef):
            name, parents = (
                node.name,
                {b.id for b in node.bases if isinstance(b, ast.Name)},
            )
            for stmt in node.body:
                targets = getattr(stmt, "targets", [])
                if [getattr(t, "id", None) for t in targets] == ["op"]:
                    op = getattr(stmt.value, "value", None)
        elif (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and node.value.args
        ):
            first = node.value.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                name, op = getattr(node.targets[0], "id", None), first.value
        if name:
            defs[name] = (op, parents)
        if lines & set(range(node.lineno, node.end_lineno + 1)):
            if not (name and op):
                return None
            touched.add(name)
    grown = True
    while grown:
        new = {n for n, (_, parents) in defs.items() if parents & touched} - touched
        touched |= new
        grown = bool(new)
    return {defs[n][0] for n in touched}


def changed_cases(manifest, case_files, diff_text, read):
    """Cases defined or changed by a diff of case modules (whole modules
    when the diff does not say which lines changed)."""
    lines = {}
    for path, start, n in diff_hunks(diff_text):
        if path in case_files:
            lines.setdefault(path, set()).update(range(start, start + max(n, 1)))
    ops = {}
    for path in case_files:
        mod = PurePosixPath(path).stem
        ops[mod] = touched_ops(read(path), lines[path]) if path in lines else None
    return [
        m["case_id"]
        for m in manifest
        if m.get("module") in ops
        and (ops[m["module"]] is None or m["op"] in ops[m["module"]])
    ]


def empty_selection(**kw):
    sel = {
        "need_run": False,
        "need_base": False,
        "mode": "none",
        "reasons": [],
        "cases": [],
        "validate": [],
        "ops": [],
        "file_map": {},
        "unmapped": [],
        "new_ops": [],
        "truncated": [],
        "core_set_used": False,
    }
    sel.update(kw)
    return sel


def _map_src(sel, files, srcmap, pr_cases, add):
    """A/B: map changed source files to cases. Returns (want_core, groups_hit)."""
    want_core, groups_hit = False, set()
    for f in files:
        if match_any(f, srcmap.get("no_bench")):
            sel["file_map"][f] = ["no-bench"]
        elif match_any(f, srcmap.get("global")):
            sel["file_map"][f] = ["global -> core set"]
            want_core = True
            groups_hit.add(None)
        elif hit := [
            g
            for g, globs in (srcmap.get("common_groups") or {}).items()
            if match_any(f, globs)
        ]:
            sel["file_map"][f] = [f"common:{g}" for g in hit]
            groups_hit.update(hit)
            want_core = True
        elif ops := [
            op for op, globs in (srcmap.get("ops") or {}).items() if match_any(f, globs)
        ]:
            sel["file_map"][f] = ops
            for op in ops:
                if not pr_cases(op):
                    sel["reasons"].append(
                        f"Op '{op}' (from `{f}`) has no PR-tier case yet."
                    )
                add(pr_cases(op), 1)
        else:
            sel["file_map"][f] = []
            sel["unmapped"].append(f)
            want_core = True
    return want_core, groups_hit


def select(
    changed,
    manifest,
    cfg,
    diff_text="",
    added_files=(),
    skip=None,
    read=None,
    cards=4,
    screen_rounds=3,
    confirm_rounds=6,
    budget_min=None,
):
    """Selection dict for a change (see the table above)."""
    by_op = {}
    for m in manifest:
        by_op.setdefault(m["op"], []).append(m)

    def pr_cases(op):
        return [m["case_id"] for m in by_op.get(op, []) if m["tier"] == "pr"]

    covered = set(by_op) | {
        n for m in manifest for n in expected_names(m["expected_aten_ops"])
    }
    budget_min = budget_min or cfg.budget.get("test_minutes", 30)
    detected = annotate(
        detect(diff_text, added_files), covered, cfg.excluded, cfg.deferred
    )
    sel = empty_selection(new_ops=detected)

    if skip and skip.get("valid"):
        sel["reasons"].append(
            f"Microbench skipped by label 'disable_microbench' (added by @{skip.get('by')})"
        )
        return sel
    if skip and skip.get("ignored"):
        sel["reasons"].append(f"Label 'disable_microbench' ignored: {skip['ignored']}")

    kinds = classify_files(changed)
    prio = {}  # case_id -> priority (lower survives truncation)

    def add(ids, p):
        for c in ids:
            prio[c] = min(prio.get(c, p), p)

    sel["validate"] = changed_cases(
        manifest, kinds["cases"], diff_text if read else "", read
    )
    if kinds["src"]:
        sel["mode"] = "ab"
        want_core, groups_hit = _map_src(sel, kinds["src"], cfg.srcmap, pr_cases, add)
        if sel["unmapped"]:
            sel["reasons"].append(
                "Changed file(s) not mapped in op_source_map.yaml; running the default core set: "
                + ", ".join(f"`{f}`" for f in sel["unmapped"])
            )
        for n in detected:
            want_core = True
            if n["status"] != "has microbench case":
                sel["reasons"].append(
                    f"New XPU op '{n['op']}' has no microbench case yet; running the default core set."
                )
        if len(groups_hit) == 1 and None not in groups_hit:
            (g,) = groups_hit
            sel["reasons"].append(
                f"Common files of group '{g}' changed: running its PR-tier cases plus the core set."
            )
            for op in (cfg.groups.get(g) or {}).get("ops") or []:
                add(pr_cases(op), 2)
        elif groups_hit:
            sel["reasons"].append(
                "Global build files or several common groups changed: running the full core set."
            )
        if want_core:
            sel["core_set_used"] = True
            add(cfg.core_cases(), 0)
        if not prio:
            sel["mode"] = "none"
            sel["reasons"].append(
                "No microbench cases selected (changed sources are no-bench)."
            )
    elif kinds["cases"] or kinds["framework"]:
        sel["mode"] = "aa"
        if kinds["cases"]:
            add(sel["validate"], 1)
            sel["reasons"].append(
                "Only microbench cases changed: validating them and running A/A with the PR wheel (report only)."
            )
        if kinds["framework"]:
            add(cfg.core_cases(), 0)
            sel["core_set_used"] = True
            sel["reasons"].append(
                "Microbench framework/CI changed: running the core set as A/A with the PR wheel (report only)."
            )
    else:
        sel["reasons"].append(
            "No microbench cases selected (no kernel, build or microbench change)."
        )
        return sel

    def minutes(ids):
        return estimate_minutes(ids, cfg.costs, cards, screen_rounds, confirm_rounds)

    kept = []
    for cid in sorted(prio, key=lambda c: (prio[c], c)):
        if kept and minutes(kept + [cid]) > budget_min:
            sel["truncated"].append(cid)
        else:
            kept.append(cid)
    if sel["truncated"]:
        sel["reasons"].append(
            f"{len(sel['truncated'])} case(s) truncated to stay within the {budget_min:.0f} minute budget."
        )
    ops = {m["case_id"]: m["op"] for m in manifest}
    sel.update(
        cases=kept,
        estimated_minutes=round(minutes(kept), 1),
        ops=sorted({ops[c] for c in kept if c in ops}),
        need_run=bool(kept or sel["validate"]),
        need_base=sel["mode"] == "ab" and bool(kept),
    )
    return sel


def select_by_spec(spec, manifest, cfg):
    """Non-diff selection for nightly / ondemand: core, all, pr, or op list."""
    if spec == "core":
        return cfg.core_cases()
    if spec == "all":
        return [m["case_id"] for m in manifest]
    if spec == "pr":
        return [m["case_id"] for m in manifest if m["tier"] == "pr"]
    ops = {o.strip() for o in spec.split(",")}
    return [m["case_id"] for m in manifest if m["op"] in ops]


def git_changes(repo, base, head):
    def git(*a):
        return subprocess.run(
            ["git", "-C", repo, *a], capture_output=True, text=True, check=True
        ).stdout

    changed, added = [], []
    for line in git("diff", "--name-status", "--no-renames", base, head).splitlines():
        status, _, path = line.partition("\t")
        if path:
            changed.append(path)
            if status.startswith("A"):
                added.append(path)
    diff = git("diff", "-U0", base, head, "--", "src", MB_CASES)
    return changed, added, diff, lambda path: git("show", f"{head}:{path}")


def summary_md(sel):
    lines = [
        "## Microbench case selection",
        "",
        f"- mode: `{sel['mode']}`, need_run: `{sel['need_run']}`, need_base: `{sel['need_base']}`",
        f"- cases: {len(sel['cases'])} (validate: {len(sel['validate'])}, truncated: "
        f"{len(sel['truncated'])}), estimated {sel.get('estimated_minutes', 0)} min",
    ]
    lines += [f"- {r}" for r in sel["reasons"]]
    if sel["file_map"]:
        lines += ["", "| Changed file | Ops |", "|---|---|"]
        lines += [
            f"| `{f}` | {', '.join(ops) or '(unmapped)'} |"
            for f, ops in sorted(sel["file_map"].items())
        ]
    if sel["new_ops"]:
        lines += [
            "",
            "### New XPU ops detected (informational)",
            "",
            "| Op | Where | Status |",
            "|---|---|---|",
        ]
        lines += [
            f"| `{n['op']}` | `{n['where']}` | {n['status']} |" for n in sel["new_ops"]
        ]
    return "\n".join(lines) + "\n"


def run_select(args):
    from .case import manifest as build_manifest

    cfg = Config(args.config_dir)
    man = build_manifest()
    if args.spec:
        ids = select_by_spec(args.spec, man, cfg)
        sel = empty_selection(
            need_run=bool(ids),
            need_base=args.need_base,
            mode="ab" if args.need_base else "aa",
            reasons=[f"Cases selected by spec '{args.spec}'."],
            cases=ids,
            ops=sorted({m["op"] for m in man if m["case_id"] in set(ids)}),
        )
    else:
        if args.changed_files:
            changed = [
                line.strip() for line in open(args.changed_files) if line.strip()
            ]
            added, diff, read = [], open(args.diff).read() if args.diff else "", None
        else:
            changed, added, diff, read = git_changes(args.repo, args.base, args.head)
        skip = None
        if args.skip_json:
            skip = json.load(open(args.skip_json))
        elif args.pr and args.gh_repo:
            from . import github

            skip = github.fetch_skip(args.gh_repo, args.pr)
        sel = select(
            changed,
            man,
            cfg,
            diff,
            added,
            skip,
            read,
            cards=args.cards,
            budget_min=args.budget_min,
        )
        if args.disable_reason:
            sel.update(
                need_run=False, need_base=False, mode="none", cases=[], validate=[]
            )
            sel["reasons"].insert(0, args.disable_reason)
    sel.update(base=args.base, head=args.head)
    with open(args.out, "w") as f:
        json.dump(sel, f, indent=2)
    md = summary_md(sel)
    print(md)
    if args.summary:
        with open(args.summary, "a") as f:
            f.write(md)
    if args.github_output:
        with open(args.github_output, "a") as f:
            for key in ("need_run", "need_base"):
                f.write(f"{key}={str(sel[key]).lower()}\n")
            f.write(f"mode={sel['mode']}\nn_cases={len(sel['cases'])}\n")
    return 0


def add_select_parser(sub):
    p = sub.add_parser("select", help="select cases for a change (PR / ondemand)")
    p.add_argument("--repo", default=".")
    p.add_argument("--base", default=None, help="base commit (merge-base)")
    p.add_argument("--head", default="HEAD")
    p.add_argument(
        "--changed-files", default=None, help="file with changed paths (instead of git)"
    )
    p.add_argument(
        "--diff", default=None, help="unified diff file (with --changed-files)"
    )
    p.add_argument("--spec", default=None, help="core | all | pr | op1,op2 (no diff)")
    p.add_argument(
        "--need-base", action="store_true", help="with --spec: A/B instead of A/A"
    )
    p.add_argument("--pr", default=None)
    p.add_argument("--gh-repo", default=None, help="owner/name for label lookups")
    p.add_argument(
        "--skip-json", default=None, help="precomputed disable_microbench status JSON"
    )
    p.add_argument(
        "--disable-reason", default=None, help="force 'no run' with this reason"
    )
    p.add_argument("--cards", type=int, default=4)
    p.add_argument("--budget-min", type=float, default=None)
    p.add_argument("--config-dir", default=None)
    p.add_argument("--out", default="selection.json")
    p.add_argument("--summary", default=None)
    p.add_argument("--github-output", default=os.environ.get("GITHUB_OUTPUT"))
    p.set_defaults(func=lambda a: sys.exit(run_select(a)))
