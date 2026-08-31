#!/usr/bin/env python3
# Copyright 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0

"""Export the fix pipeline's verified fixes as downloadable patch series.

Usage:
    AGENT_SPACE=<scratch dir> OUT=<patch out dir> \
        [GITHUB_WORKSPACE=<repo root>] python bot_export_fix_patch.py

`issue-handler` commits each fix onto its own branch and writes one
`fix_result*.json` per unit (`fix_result.json` single bug,
`fix_result-<slug>.json` per batch sub-item) with a strict schema:
verdict / target_repo / fix_repo_dir / branch / base_sha / changed_files.

This is the schema gate AND the exporter: a PASSED unit with a bad schema, or
one that yields no patch, fails the job -- a lost fix is never a silent green.
The calling workflow runs it before the reclaim wipes AGENT_SPACE, and before
redaction so the patches are scrubbed too.
"""

import glob
import json
import os
import subprocess
import sys


def git(repo, *args):
    """Run a git command in `repo` and return the CompletedProcess."""
    return subprocess.run(["git", "-C", repo, *args],
                          capture_output=True, text=True)


def slug_for(fix_result_path):
    """Derive a unit's slug from its fix_result file name.

    `fix_result.json` -> "single"; `fix_result-<slug>.json` -> "<slug>".
    """
    name = os.path.basename(fix_result_path)
    return name[len("fix_result"):].lstrip("-").removesuffix(".json") or "single"


def fix_branches(root_dir):
    """List every agent/fix-issue-* branch in any git repo under `root_dir`.

    Cross-checks the agent's own claim against a fact the workflow can observe:
    such a branch means a fix was committed. If one exists but no
    fix_result*.json was written, the agent produced a fix and then lost the
    hand-off contract -- the caller fails loudly instead of the silent green
    that "nothing to export" + exit 0 would otherwise be (the single most
    likely way to lose a verified fix).

    Scan from the workspace root, not just AGENT_SPACE: for a torch-xpu-ops
    fix, fix_repo_dir is the main checkout at the workspace root, while
    AGENT_SPACE (a subdirectory of it) holds only the pytorch build tree.
    Walking the root covers both. Match `.git` as a file too, not just a
    directory: a submodule populated by `git submodule update` has a `.git`
    FILE, and the fix can land in one (third_party/torch-xpu-ops is a submodule
    of the pytorch tree). Directory-only walk with .git pruned: measured 0.4s
    over a 12GB tree with 70 repos.
    """
    found = []
    for root, dirs, files in os.walk(root_dir):
        if ".git" in dirs or ".git" in files:
            r = git(root, "for-each-ref", "--format=%(refname:short)",
                    "refs/heads/agent/fix-issue-*")
            if r.returncode == 0:
                found += [f"{root}:{b}" for b in r.stdout.split() if b]
            dirs[:] = [d for d in dirs if d != ".git"]
    return found


def export(agent_space, out, root_dir):
    """Export every fix_result unit under `agent_space` into `out`.

    Returns `(made, salvaged, errors)`: the number of verified patch series
    written, the number of unverified ones salvaged, and a list of schema /
    lost-fix problems that must fail the job. Never calls sys.exit, so the
    behaviour is testable; `main` maps the result onto exit codes.
    """
    errors = []
    made = 0
    salvaged = 0

    results = sorted(glob.glob(os.path.join(agent_space, "fix_result*.json")))
    if not results:
        return made, salvaged, errors

    for fr in results:
        try:
            d = json.load(open(fr))
        except Exception as e:
            errors.append(f"{fr}: unparseable ({e})")
            continue

        verdict = str(d.get("verdict", "")).upper()
        # A non-PASSED unit carries no verified fix, but it may still point at
        # a real commit -- PENDING_VERIFY means the fix was committed and
        # verification never finished. That branch is local and dies with the
        # runner, so exporting only PASSED would lose the work while the job
        # stayed green and the uploaded fix_result.json described a fix nobody
        # can retrieve. Salvage it as a clearly-separated unverified patch
        # (under the same fixpatch root, so the redact step still scrubs it).
        # Best effort: a non-PASSED unit with an unusable contract is reported,
        # not fatal -- only PASSED units are schema-gated.
        if verdict != "PASSED":
            repo = d.get("fix_repo_dir") or ""
            branch = d.get("branch") or ""
            base = d.get("base_sha") or ""
            if (repo and branch and base and os.path.isdir(repo)
                    and git(repo, "rev-parse", "--verify", branch).returncode == 0
                    and git(repo, "rev-parse", "--verify", base).returncode == 0):
                dest = os.path.join(out, "unverified", slug_for(fr))
                os.makedirs(dest, exist_ok=True)
                git(repo, "format-patch", f"--base={base}", f"{base}..{branch}",
                    "-o", dest)
                if any(f.endswith(".patch") for f in os.listdir(dest)):
                    salvaged += 1
                    print(f"{fr}: verdict={verdict or 'MISSING'} -> salvaged "
                          f"UNVERIFIED patch from {branch} ({base[:12]}..)")
                    continue
            print(f"{fr}: verdict={verdict or 'MISSING'} -> no patch expected")
            continue

        # Schema gate: a PASSED unit MUST carry a usable, valid contract.
        repo = d.get("fix_repo_dir") or ""
        branch = d.get("branch") or ""
        base = d.get("base_sha") or ""
        changed = d.get("changed_files") or []
        missing = [k for k, v in
                   (("fix_repo_dir", repo), ("branch", branch),
                    ("base_sha", base), ("changed_files", changed)) if not v]
        if missing:
            errors.append(f"{fr}: PASSED but missing {', '.join(missing)}")
            continue
        if not os.path.isdir(repo):
            errors.append(f"{fr}: fix_repo_dir does not exist: {repo}")
            continue
        if git(repo, "rev-parse", "--verify", branch).returncode != 0:
            errors.append(f"{fr}: branch not found in {repo}: {branch}")
            continue
        if git(repo, "rev-parse", "--verify", base).returncode != 0:
            errors.append(f"{fr}: base_sha not found in {repo}: {base}")
            continue

        dest = os.path.join(out, slug_for(fr))
        os.makedirs(dest, exist_ok=True)
        git(repo, "format-patch", f"--base={base}", f"{base}..{branch}", "-o", dest)
        if any(f.endswith(".patch") for f in os.listdir(dest)):
            made += 1
            print(f"{fr}: exported {branch} ({base[:12]}..)")
        else:
            errors.append(f"{fr}: PASSED but format-patch produced nothing "
                          f"({base}..{branch})")

    return made, salvaged, errors


def main():
    agent_space = os.environ["AGENT_SPACE"]
    out = os.environ["OUT"]
    root_dir = os.environ.get("GITHUB_WORKSPACE") or agent_space

    if not sorted(glob.glob(os.path.join(agent_space, "fix_result*.json"))):
        branches = fix_branches(root_dir)
        if branches:
            print("ERROR: fix branch(es) exist but no fix_result*.json was "
                  "written -- the fix was committed then lost:\n  "
                  + "\n  ".join(branches), file=sys.stderr)
            sys.exit(1)
        print("no fix_result*.json and no fix branch; nothing to export")
        sys.exit(0)

    made, salvaged, errors = export(agent_space, out, root_dir)

    if errors:
        print("ERROR: patch export problems:\n  " + "\n  ".join(errors),
              file=sys.stderr)
        sys.exit(1)
    # made==0 with fix_result(s) present is fine: those files record a
    # non-PASSED outcome (FAILED / NEEDS_HUMAN / PENDING_VERIFY) that the agent
    # deliberately wrote, so the branch is explained. The lost-fix trap (branch
    # exists, nothing explains it) is caught by the no-results guard above,
    # before any file is read.
    print(f"exported {made} patch series")
    if salvaged:
        print(f"WARNING: also salvaged {salvaged} UNVERIFIED patch series "
              f"under unverified/ -- a fix was committed but never passed "
              f"verification. Review before applying.")


if __name__ == "__main__":
    main()
