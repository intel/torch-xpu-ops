# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Case selection of nightly and on-demand runs; candidate commits of a
nightly regression."""

import fnmatch
import json
import os
import re
import subprocess
import sys

from . import CONFIG_DIR, judge, load_history, load_yaml, reference


def match_any(path, globs):
    return any(fnmatch.fnmatch(path, g) for g in globs or ())


class Config:
    def __init__(self, config_dir=None):
        d = config_dir or CONFIG_DIR
        self.srcmap = load_yaml(os.path.join(d, "op_source_map.yaml"), {})
        self.groups = self.srcmap.get("core_set") or {}
        hist = load_history(os.path.join(d, "history.json"))
        self.costs = hist["case_cost"]
        self.screening = reference.load_screening(os.path.join(d, "gating.yaml"))
        self.gating = judge.load_gating(os.path.join(d, "gating.yaml"))

    def core_cases(self):
        return [
            e["case_id"] for g in self.groups.values() for e in g.get("cases") or []
        ]


def related_globs(op, srcmap):
    """Files whose change can move the cases of ``op``: the op's mapped files,
    every common group and the global build files."""
    globs = list((srcmap.get("ops") or {}).get(op) or []) + list(
        srcmap.get("global") or []
    )
    for g in (srcmap.get("common_groups") or {}).values():
        globs += g
    return globs


def candidate_commits(repo, prev_sha, cur_sha, op, srcmap):
    """xpu-ops commits in prev..cur touching files related to ``op``,
    oldest first: [{"sha", "subject", "files"}]."""
    globs = related_globs(op, srcmap)
    out = subprocess.run(
        [
            "git",
            "-C",
            repo,
            "log",
            "--reverse",
            "--format=%H%x09%s",
            "--name-only",
            f"{prev_sha}..{cur_sha}",
        ],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        return None
    commits, cur = [], None
    for line in out.stdout.splitlines():
        if "\t" in line and re.fullmatch(r"[0-9a-f]{40}", line.split("\t", 1)[0]):
            sha, subject = line.split("\t", 1)
            cur = {"sha": sha, "subject": subject, "files": []}
            commits.append(cur)
        elif line.strip() and cur is not None:
            cur["files"].append(line.strip())
    return [
        dict(c, files=[f for f in c["files"] if match_any(f, globs)])
        for c in commits
        if any(match_any(f, globs) for f in c["files"])
    ]


def select_by_spec(spec, manifest, cfg):
    """core, all, pr (the PR-tier cases), or a comma separated op list."""
    if spec == "core":
        return cfg.core_cases()
    if spec == "all":
        return [m["case_id"] for m in manifest]
    if spec == "pr":
        return [m["case_id"] for m in manifest if m["tier"] == "pr"]
    ops = {o.strip() for o in spec.split(",")}
    return [m["case_id"] for m in manifest if m["op"] in ops]


# flow -> path of the run: nightly (single-sided measurement, detection,
# recheck, attribution), vs-base (A/B against a base build), aa (A/A of one
# wheel, report only)
FLOWS = {"nightly": "nightly", "ab": "vs-base", "aa": "aa"}
REASONS = {
    "ab": "Direct A/B requested (on-demand microbench_base).",
    "aa": "A/A requested: the B wheel on both sides (report only).",
}


def summary_md(sel):
    lines = [
        "## Microbench case selection",
        "",
        f"- path: `{sel['path']}`, need_run: `{sel['need_run']}`, cases: {len(sel['cases'])}",
    ]
    lines += [f"- {r}" for r in sel["reasons"]]
    return "\n".join(lines) + "\n"


def run_select(args):
    from .case import manifest as build_manifest

    cfg = Config(args.config_dir)
    man = build_manifest()
    ids = select_by_spec(args.spec, man, cfg)
    by_id = {m["case_id"]: m for m in man}
    path = FLOWS[args.flow]
    sel = {
        "path": path,
        "need_run": bool(ids),
        "need_base": path == "vs-base" and bool(ids),
        "reasons": [f"Cases selected by spec '{args.spec}'."]
        + ([REASONS[args.flow]] if args.flow in REASONS else []),
        "cases": ids,
        "ops": sorted({by_id[c]["op"] for c in ids}),
        "base": args.base,
        "head": args.head,
        "pytorch_sha": args.pytorch_sha,
    }
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
            f.write(f"path={sel['path']}\nn_cases={len(sel['cases'])}\n")
    return 0


def add_select_parser(sub):
    p = sub.add_parser("select", help="select cases for a nightly / on-demand run")
    p.add_argument("--spec", default="core", help="core | all | pr | op1,op2")
    p.add_argument(
        "--flow",
        choices=sorted(FLOWS),
        default="ab",
        help="nightly: single-sided nightly measurement; ab: A/B against --base; "
        "aa: A/A with the B wheel",
    )
    p.add_argument("--base", default=None, help="xpu-ops commit of the base wheel (ab)")
    p.add_argument("--head", default="HEAD", help="xpu-ops commit of the B / N wheel")
    p.add_argument("--pytorch-sha", default=None, help="recorded in selection.json")
    p.add_argument("--config-dir", default=None)
    p.add_argument("--out", default="selection.json")
    p.add_argument("--summary", default=None)
    p.add_argument("--github-output", default=os.environ.get("GITHUB_OUTPUT"))
    p.set_defaults(func=lambda a: sys.exit(run_select(a)))
