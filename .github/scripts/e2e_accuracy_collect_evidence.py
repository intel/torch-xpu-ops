#!/usr/bin/env python3
"""Evidence for the E2E accuracy auto-issue pipeline. Runs on CPU, never writes to GitHub.

collect --run-id N --out-dir D
    in:  run N's E2E accuracy artifacts, open `test: e2e` issues, and the accuracy values of the
         HISTORY_RUNS scheduled build runs before N (first_seen, last_good)
    out: D/evidence.json  gate-red rows grouped by failure signature, plus the confirm plan
         D/report.json    counts, suites the gate could not judge, open issues that cannot mute
         GITHUB_OUTPUT    need_confirm, confirm_models, has_failures (a group other than new_pass/machine)
    exit 1 if a suite could not be judged (the files are still written)
replay --run-id N [N ...] --out-dir D
    out: D/<N>/evidence.json and D/report.json, without history
merge --evidence F --confirm-dir D
    in:  D/**/evidence.json from `e2e_accuracy_triage.py confirm`, one per SKU
    out: F, updated in place with confirm.<sku> and groups[].confirm_result

Rows are judged by check_expected.py's own functions, so the result always matches the gate.
Schema: .claude/skills/e2e-accuracy-issue-authoring/references/evidence-schema.md
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from functools import cache
from itertools import takewhile
from pathlib import Path

REF_DIR = Path(__file__).resolve().parents[1] / "ci_expected_accuracy"
sys.path.insert(0, str(REF_DIR))
from check_expected import (  # noqa: E402
    SKIP_LABELS, SKIPPED_MODELS, categorize_model, find_model_row, get_test_result, is_tracked, load_data)
from get_issue import fast_extract_table_rows  # noqa: E402

REPO = os.environ.get("GITHUB_REPOSITORY") or "intel/torch-xpu-ops"
SUITES = ("huggingface", "timm_models", "torchbench")
SKUS = {"1100": "pvc-1100", "1550": "pvc-1550"}
GATE_RED = ("real_failed", "timeout", "new_pass")
NEVER_FILED = ("new_pass", "machine")  # categories the authoring skill does not file, so never confirmed
HISTORY_RUNS = 8  # earlier scheduled build runs read for first_seen/last_good: over a week, so a weekly too
AUTO_MARKER = "<!-- e2e-accuracy-auto-issue"
# Candidate category from the result value: first match wins, the authoring skill has the last word.
CATEGORIES = [
    ("oom", r"oom|out of memory|outofmemory|memory>"),
    ("machine", r"ur_result_error"),
    ("timeout", r"timeout"),
    ("new_pass", r"pass"),
    ("accuracy", r"fail_accuracy"),
    ("eager_nondeterministic", r"eager_two_runs_differ"),
    ("eager_fail", r"eager_(1st|2nd)_run_fail"),
    ("env", r"eager_fail_to_run|model_fail_to_load"),
    ("crash", r""),
]
KEY_LINE = re.compile(r"^[\w.]+(Error|Exception)\b|Accuracy failed|out of memory|Segmentation fault|core dumped", re.I)
LOG_PREFIX = re.compile(r"^\[\d{4}-\d\d-\d\d [\d:]+\] ?")  # run_benchmarks' timestamp
GLOG_PREFIX = re.compile(r"^[EWI]\d{4} [\d:.]+\s+\d+ \S+\] ")


def gh(*args: str) -> str:
    return subprocess.run(["gh", *args], check=True, capture_output=True, text=True).stdout


def download(run_id: int, artifact: str, dest: Path) -> bool:
    shutil.rmtree(dest, ignore_errors=True)
    cmd = ["gh", "run", "download", str(run_id), "--repo", REPO, "-n", artifact, "-D", str(dest)]
    return subprocess.run(cmd, capture_output=True).returncode == 0


def category(value: str) -> str:
    return next(name for name, pattern in CATEGORIES if re.search(pattern, value, re.I))


def command(suite: str, dtype: str, mode: str, model: str) -> str:
    """The benchmark command run_benchmarks runs for one accuracy task, without --timeout/--output."""
    amp = {"amp_bf16": "--amp --amp-dtype bfloat16", "amp_fp16": "--amp --amp-dtype float16"}
    return (f"python benchmarks/dynamo/{suite}.py --accuracy -d xpu -n10 --backend=inductor --cold-start-latency "
            f"--disable-cudagraphs {amp.get(dtype, '--' + dtype)} --{mode} --only {model}")


@cache
def reference(driver: str, suite: str, mode: str):
    return load_data(REF_DIR / driver / f"inductor_{suite}_{mode}.csv")


def read_results(root: Path) -> list[dict]:
    """One dict per accuracy row in a run_benchmarks output tree (an artifact or a local rerun)."""
    run_log = root / "run_e2e.log"
    devices = dict(re.findall(r"card (\d+): (.+)", run_log.read_text(errors="replace"))) if run_log.exists() else {}
    rows = []
    for csv_file in sorted(root.rglob("inductor-*-accuracy.csv")):
        for _, row in load_data(csv_file).iterrows():
            suite, dtype, mode, model, value = (str(row[c]) for c in ("suite", "dtype", "mode", "name", "accuracy"))
            logs = sorted(csv_file.parent.glob(f"logs-{model.replace('/', '_')}-worker*-card*.log"))
            lines = [LOG_PREFIX.sub("", x) for x in logs[-1].read_text(errors="replace").splitlines()] if logs else []
            start = next((i for i, x in enumerate(lines) if x.startswith("Traceback")), max(0, len(lines) - 40))
            excerpt = lines[start:] if len(lines) - start <= 60 else lines[start:start + 20] + ["[...]"] + lines[-40:]
            key_lines = [GLOG_PREFIX.sub("", x).strip() for x in lines if KEY_LINE.search(GLOG_PREFIX.sub("", x))]
            card = re.search(r"-card(\d+)\.log$", logs[-1].name).group(1) if logs else ""
            device = devices.get(card, "unknown")
            rows.append({
                "key": f"{suite},{dtype},{mode},{model}", "suite": suite, "dtype": dtype, "mode": mode,
                "model": model, "value": value, "batch_size": str(row.get("batch_size", "")),
                "elapsed": str(row.get("elapsed", "")), "card": card, "device": device,
                "sku": next((sku for k, sku in SKUS.items() if k in device), "unknown"),
                "message": (key_lines[-1] if key_lines else value).replace("|", "/")[:200],
                "log": str(logs[-1].relative_to(root)) if logs else "", "log_excerpt": excerpt,
                "cmd": command(suite, dtype, mode, model),
            })
    return rows


def open_issues() -> list[dict]:
    """Open `test: e2e` issues, their tables parsed exactly as get_issue.py parses them for the gate."""
    jq = '.[] | select(.pull_request == null) | {number, title, body: (.body // ""), labels: [.labels[].name]} | tojson'
    out = gh("api", "--paginate", f"repos/{REPO}/issues?state=open&labels=test:%20e2e&per_page=100", "-q", jq)
    return [{"issue_number": i["number"], "issue_title": i["title"], "labels": i["labels"],
             "table_rows": fast_extract_table_rows(i["body"]), "auto": AUTO_MARKER in i["body"]}
            for i in map(json.loads, out.splitlines())]


def issue_health(issues: list[dict]) -> list[dict]:
    """Open issues that look meant to mute a model but cannot, and why. Report only."""
    report = []
    for i in issues:
        rows = [r for r in i["table_rows"] if len(r) >= 6 and r[0].strip("`") in SUITES]
        reasons = [reason for bad, reason in [
            ("Accuracy" not in i["labels"], "no Accuracy label"),
            (not rows, "no `suite | dtype | mode | model | value | message` row"),
            (any("`" in c for r in rows for c in r[:5]), "backticks in a cell"),
        ] if bad]
        if reasons and ("Accuracy" in i["labels"] or rows):
            report.append({"issue": i["issue_number"], "title": i["issue_title"], "reasons": reasons})
    return report


def read_env(root: Path) -> dict:
    """Versions linux-testenv saved into the artifact's env/ directory."""
    def read(name: str) -> str:
        return (root / "env" / name).read_text(errors="replace") if (root / "env" / name).exists() else ""
    wanted = r"(torch|torchvision|torchaudio|torchao|pytorch-triton-xpu|transformers|timm|numpy|accelerate)\s"
    return {"versions": dict(x.split("=", 1) for x in read("versions.txt").split() if "=" in x),
            "packages": dict(x.split()[:2] for x in read("pip_list.txt").splitlines() if re.match(wanted, x)),
            "collect_env": read("collect_env.log")}


def accuracy_artifacts(run_id: int) -> dict[str, str]:
    """{suite: name} of run_id's unexpired E2E accuracy data artifacts."""
    names = gh("api", "--paginate", f"repos/{REPO}/actions/runs/{run_id}/artifacts?per_page=100",
               "-q", ".artifacts[] | select(.expired | not) | .name").split()
    found = {}
    for suite in SUITES:
        name = next((a for a in sorted(names, reverse=True)
                     if re.search(rf"-XPU-E2E-Data-.*-{suite}(-accuracy)?-{run_id}-", a)), None)
        if name:
            found[suite] = name
    return found


def judge_run(run_id: int, work: Path, driver: str, issues: list[dict]) -> tuple[dict, list[dict]]:
    """Download run_id's accuracy artifacts and return (run facts, rows the gate turns red)."""
    artifacts = accuracy_artifacts(run_id)
    jobs = dict(x.split("\t") for x in gh("api", "--paginate", f"repos/{REPO}/actions/runs/{run_id}/jobs?per_page=100",
                                          "-q", '.jobs[] | [.name, .runner_name // ""] | @tsv').splitlines())
    known = [i for i in issues if "Accuracy" in i["labels"]]  # what get_issue.py hands the gate
    skipped = [i for i in known if SKIP_LABELS & set(i["labels"])]
    run = {"run_id": run_id, "driver": driver, "env": {}, "runners": {}, "integrity": []}
    red, by_model = [], defaultdict(dict)
    for suite in SUITES:
        job = next((name for name in jobs if f"({suite}, accuracy)" in name), None)
        if job is None:
            continue  # the suite was not scheduled
        run["runners"][suite] = jobs[job]
        artifact = artifacts.get(suite)
        root = work / str(run_id) / suite
        if artifact is None or not download(run_id, artifact, root):
            run["integrity"].append(f"{suite}: no accuracy artifact")
            continue
        run["env"] = run["env"] or read_env(root)
        log = (root / "run_e2e.log").read_text(errors="replace") if (root / "run_e2e.log").exists() else ""
        if log.count("Wall time:") < max(1, len(re.findall(r"^  Task file ", log, re.M))):  # as e2e_summary.sh
            run["integrity"].append(f"{suite}: run_e2e exited before its final summary")
        rows = read_results(root)  # only models that ran tonight, never the whole reference
        if not rows:
            run["integrity"].append(f"{suite}: no accuracy CSV")
        for r in rows:
            by_model[(suite, r["model"])][f"{r['mode']} {r['dtype']}"] = r["value"]
            ref_row = find_model_row(reference(driver, suite, r["mode"]), r["model"])
            r["reference"] = str(ref_row[r["dtype"]]) if ref_row is not None else "N/A"
            if is_tracked(skipped, suite, r["dtype"], r["mode"], r["model"]):
                continue
            known_value = get_test_result(known, suite, r["dtype"], r["mode"], r["model"])
            r["gate"], _ = categorize_model(r["model"], r["value"], r["reference"], known_value, SKIPPED_MODELS)
            if r["gate"] in GATE_RED:
                r["log"] = f"{suite}/{r['log']}"
                red.append(r)
    for r in red:
        r["other_results"] = by_model[(r["suite"], r["model"])]
        r["issues"] = [i["issue_number"] for i in issues if any(t[:4] == r["key"].split(",") for t in i["table_rows"])]
    return run, red


def group_rows(run_id: int, rows: list[dict], hist: list[tuple[int, dict]]) -> tuple[list[dict], list[dict]]:
    """Group rows by signature and decide which groups the GPU confirm stage reruns."""
    by_sig = defaultdict(list)
    for r in rows:
        # Only the runs that ran this row: a nightly does not run p2, so a p2 row is compared with the last weekly.
        seen = [(rid, values[r["key"]]) for rid, values in hist if r["key"] in values]
        same = list(takewhile(lambda s: s[1] == r["value"], seen))
        r["first_seen"] = same[-1][0] if same else run_id
        r["last_good"] = next((rid for rid, value in seen if "pass" in value), None)
        text = re.sub(r"0x[0-9a-f]+|\d+(\.\d+)?", "N", r["message"].lower())  # drop addresses and numbers
        by_sig[r["value"] + "|" + re.sub(r"(/[\w.-]+)+", "PATH", text)].append(r)
    budget = int(os.environ.get("E2E_ACCURACY_CONFIRM_MAX_PER_RUN", 10))
    groups, plan = [], []
    for n, (sig, members) in enumerate(sorted(by_sig.items(), key=lambda kv: -len(kv[1])), 1):
        cat, skus = category(members[0]["value"]), sorted({r["sku"] for r in members})
        if cat in NEVER_FILED:
            skip = f"{cat}: nothing to confirm"
        elif set(SKUS.values()) <= set(skus):
            skip = "failed on both SKUs tonight with the same signature"
        elif len(plan) >= budget and any(r["first_seen"] != run_id for r in members):
            skip = "same value in the last run that ran it; filed without cross-SKU confirmation"
        elif len(plan) >= budget:
            skip = "deferred: over E2E_ACCURACY_CONFIRM_MAX_PER_RUN, confirmed on a later night"
        else:
            skip = ""
        first = members[0]  # plus one row from another suite or dtype, if any
        reps = [first["key"]] + [r["key"] for r in members
                                 if (r["suite"], r["dtype"]) != (first["suite"], first["dtype"])][:1]
        groups.append({"id": f"g{n}", "signature": sig, "category": cat, "skus": skus, "reps": reps,
                       "confirm_skip": skip, "rows": members})
        if not skip:
            plan.append({"group": f"g{n}", "reps": reps})
    return groups, plan


def build_evidence(run_id: int, args, issues: list[dict], hist: list[tuple[int, dict]]) -> dict:
    run, rows = judge_run(run_id, args.work_dir, args.driver, issues)
    groups, plan = group_rows(run_id, rows, hist)
    digest = hashlib.sha256(json.dumps(sorted((r["key"], r["value"]) for r in rows)).encode()).hexdigest()[:16]
    run["history"] = [rid for rid, _ in hist]
    return {"run": run, "digest": digest, "groups": groups, "confirm_plan": plan, "confirm": {}, "issues": issues}


def history(run_id: int, work: Path) -> list[tuple[int, dict]]:
    """(run id, {key: value}) of the HISTORY_RUNS scheduled build-from-source runs before run_id, newest first.

    Read from each run's E2E data, not from an earlier evidence.json: a night the gate passed has none.
    """
    jq = '.workflow_runs[] | select(.display_title | endswith("Build-from-source")) | .id'
    ids = gh("api", f"repos/{REPO}/actions/workflows/nightly_ondemand.yml/runs?event=schedule&per_page=50",
             "-q", jq).split()
    hist = []
    for prev in [int(i) for i in ids if int(i) < run_id][:HISTORY_RUNS]:
        values = {}
        for suite, artifact in accuracy_artifacts(prev).items():
            root = work / str(prev) / suite
            if download(prev, artifact, root):
                for csv_file in root.rglob("inductor-*-accuracy.csv"):
                    for _, row in load_data(csv_file).iterrows():
                        key = ",".join(str(row[c]) for c in ("suite", "dtype", "mode", "name"))
                        values[key] = str(row["accuracy"])
        hist.append((prev, values))
    return hist


def conclude(row: dict, confirm: dict) -> dict:
    """Tonight's value vs reruns on the failing SKU (A) and on the other one (B); see evidence-schema.md."""
    night, a = row["value"], row["sku"]
    if a not in SKUS.values():
        return {"conclusion": "unknown_sku"}
    b = next(s for s in SKUS.values() if s != a)
    a_runs, b_runs = ([r["value"] for r in confirm.get(s, {}).get(row["key"], {}).get("runs", [])] for s in (a, b))
    failed = [v for v in a_runs + b_runs if "pass" not in v]
    flaky = lambda runs: any("pass" in v for v in runs) and any("pass" not in v for v in runs)  # noqa: E731
    if not a_runs and not b_runs:
        conclusion = "unconfirmed"
    elif not failed:
        conclusion = "not_reproduced"
    elif flaky([night, *a_runs]) or flaky(b_runs):
        conclusion = "random"
    elif len({night, *a_runs}) > 1 or len(set(b_runs)) > 1:
        conclusion = "value_varies"  # fails every time, but not always with the same value
    elif not b_runs:
        conclusion = "single_sku"
    elif b_runs[0] == night:
        conclusion = "sku_independent"
    else:
        conclusion = "sku_specific" if "pass" in b_runs[0] else "sku_different_value"
    public = confirm.get(a, {}).get(row["key"], {}).get("public_wheel", {})
    seen_failures = {v for v in [night, *a_runs] if "pass" not in v}
    return {"conclusion": conclusion, "runs": {a: [night, *a_runs], b: b_runs}, "public_wheel": public.get("value"),
            "public_wheel_reproduces": public.get("value") in seen_failures,
            "public_versions": public.get("versions", {})}


def merge(evidence: dict, confirm_dir: Path) -> dict:
    for path in confirm_dir.rglob("evidence.json"):
        evidence["confirm"].update(json.loads(path.read_text()).get("confirm", {}))
    for g in evidence["groups"]:
        if not g["confirm_skip"]:
            rows = {r["key"]: r for r in g["rows"]}
            reps = {k: conclude(rows[k], evidence["confirm"]) for k in g["reps"]}
            verdicts = {v["conclusion"] for v in reps.values()}
            g["confirm_result"] = {"conclusion": verdicts.pop() if len(verdicts) == 1 else "mixed", "reps": reps}
    return evidence


def report(evidence: dict, health: list[dict]) -> dict:
    rows = [r for g in evidence["groups"] for r in g["rows"]]
    return {"run_id": evidence["run"]["run_id"],
            "gate": {v: sum(r["gate"] == v for r in rows) for v in GATE_RED},
            "values": {v: sum(r["value"] == v for r in rows) for v in sorted({r["value"] for r in rows})},
            "groups": len(evidence["groups"]), "confirm_plan": len(evidence["confirm_plan"]),
            "integrity": evidence["run"]["integrity"], "issue_health": health}


def write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("collect", "replay"):
        p = sub.add_parser(name)
        p.add_argument("--run-id", type=int, required=True, nargs="+" if name == "replay" else None)
        p.add_argument("--out-dir", type=Path, required=True)
        p.add_argument("--test-type", default="", help="recorded in evidence.json, e.g. build-nightly")
        p.add_argument("--driver", default="rolling", help="reference CSV directory, as check_expected.py")
        p.add_argument("--work-dir", type=Path, default=Path("e2e_work"), help="artifact downloads")
    p = sub.add_parser("merge")
    p.add_argument("--evidence", type=Path, required=True)
    p.add_argument("--confirm-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.cmd == "merge":
        write(args.evidence, merge(json.loads(args.evidence.read_text()), args.confirm_dir))
        return 0
    issues = open_issues()
    if args.cmd == "replay":
        reports = []
        for run_id in args.run_id:
            evidence = build_evidence(run_id, args, issues, [])
            write(args.out_dir / str(run_id) / "evidence.json", evidence)
            reports.append(report(evidence, []))
        write(args.out_dir / "report.json", reports)
        return 0

    evidence = build_evidence(args.run_id, args, issues, history(args.run_id, args.work_dir))
    evidence["run"]["test_type"] = args.test_type
    write(args.out_dir / "evidence.json", evidence)
    write(args.out_dir / "report.json", report(evidence, issue_health(issues)))
    models = sorted({k.split(",")[3] for p in evidence["confirm_plan"] for k in p["reps"]})
    failures = any(g["category"] not in NEVER_FILED for g in evidence["groups"])
    with open(os.environ.get("GITHUB_OUTPUT", os.devnull), "a") as f:
        f.write(f"need_confirm={str(bool(models)).lower()}\nconfirm_models={','.join(models)}\n"
                f"has_failures={str(failures).lower()}\n")
    for problem in evidence["run"]["integrity"]:
        print(f"::error::Accuracy not judged: {problem}")
    return 1 if evidence["run"]["integrity"] else 0


if __name__ == "__main__":
    sys.exit(main())
