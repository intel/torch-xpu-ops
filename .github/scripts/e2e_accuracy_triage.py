#!/usr/bin/env python3
"""GPU side of the E2E accuracy auto-issue pipeline. Runs models, never writes to GitHub.

Run from the workspace that holds the `pytorch` checkout prepared by linux-testenv.

confirm --evidence IN --sku S --out OUT
    in:  IN = evidence.json from `e2e_accuracy_collect_evidence.py collect`; reruns its confirm_plan
    out: OUT = IN plus confirm.<S> = {key: {runs: [...], public_wheel: {...}}}
         (each run: value, card, device, sku, elapsed, tail; the public nightly wheel runs last, own-SKU rows only)
deep --created IN --sku S --out-dir D
    in:  IN = created.json from `e2e_accuracy_create_issues.py`; its deep_queue items for S
    out: D/<issue>-<model>/triage.json = {item, experiments}, with every experiment's log next to it
verify --triage F
    in:  F with `result.repro_script` written by the e2e-accuracy-triage skill
    out: F plus verify = {passed, reason}, also printed

Timeouts are fixed here on purpose. Schema: .claude/skills/e2e-accuracy-triage/references/triage-schema.md
"""

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

from e2e_accuracy_collect_evidence import command, read_results

ROOT = Path(__file__).resolve().parents[2]
PYTORCH = Path(os.environ.get("GITHUB_WORKSPACE", ROOT)) / "pytorch"
NIGHTLY = "https://download.pytorch.org/whl/nightly/xpu"
RERUNS = int(os.environ.get("E2E_ACCURACY_CONFIRM_RERUNS", 2))
MAX_DEEP = int(os.environ.get("E2E_ACCURACY_TRIAGE_MAX_PER_RUN", 3))
CONFIRM_TIMEOUT = 3600
ITEM_TIMEOUT = int(os.environ.get("E2E_ACCURACY_TRIAGE_TIMEOUT_MIN", 180)) * 60
MINIFY_TIMEOUT = 2 * 3600
VERIFY_TIMEOUT = 300
NOTE = re.compile(r"memory: eager|Compilation time|RMSE|Accuracy failed|Error|Repro script|minif", re.I)


def stage() -> None:
    """What linux-e2etest does before run_e2e: the runner and xpu-ops' benchmark yamls into pytorch/."""
    subprocess.run(["rsync", "-a", "--delete", f"{ROOT}/.github/scripts/run_benchmarks/", f"{PYTORCH}/run_e2e/"],
                   check=True)
    subprocess.run(["rsync", "-a", f"{ROOT}/.ci/benchmarks/", f"{PYTORCH}/benchmarks/dynamo/"], check=True)
    shutil.rmtree(PYTORCH / "torch", ignore_errors=True)


def run_e2e(keys: list[str], out: Path) -> dict[str, dict]:
    """Run `suite,dtype,mode,model` keys through run_e2e as the nightly does; logs go to `out`."""
    tasks = out.with_suffix(".txt")
    # TXT task file: `suite dtype mode model result`; a non-numeric result selects the accuracy scenario
    tasks.write_text("".join(f"{k.replace(',', ' ')} fail\n" for k in keys))
    shutil.rmtree(PYTORCH / "inductor_log", ignore_errors=True)
    (PYTORCH / "inductor_log").mkdir()
    with open(PYTORCH / "inductor_log" / "run_e2e.log", "w") as log:
        # run_e2e runs in pytorch/, so a relative task file would not be found there.
        subprocess.run([sys.executable, "-m", "run_e2e", "--task-file", str(tasks.resolve()), "--task-timeout",
                        str(CONFIRM_TIMEOUT)], cwd=PYTORCH, stdout=log, stderr=subprocess.STDOUT)
    shutil.rmtree(out, ignore_errors=True)
    shutil.move(PYTORCH / "inductor_log", out)
    rows = read_results(out)
    if not rows:  # even a crashed model gets a row, so this is run_e2e itself failing
        sys.exit(f"run_e2e produced no result for {keys}; see {out}/run_e2e.log")
    fields = ("value", "card", "device", "sku", "elapsed")
    return {r["key"]: {f: r[f] for f in fields} | {"tail": r["log_excerpt"][-20:]} for r in rows}


def confirm(evidence: dict, sku: str, work: Path) -> dict:
    rows = {r["key"]: r for g in evidence["groups"] for r in g["rows"]}
    keys = [k for p in evidence["confirm_plan"] for k in p["reps"]]
    found = {k: {"runs": []} for k in keys}
    for i in range(RERUNS):
        for key, result in run_e2e(keys, work / f"confirm-{i}").items():
            if result["sku"] != sku:
                sys.exit(f"{key} reran on {result['device']} ({result['sku']}), not on the {sku} this job asked for")
            found[key]["runs"].append(result)
    # Last, since it replaces the CI wheel. Decides whether the issue's Reproducer can use the public wheel.
    own = [k for k in keys if rows[k]["sku"] == sku]
    if own:
        subprocess.run(["uv", "pip", "uninstall", "torch", "torchvision", "torchaudio"], check=True)
        subprocess.run(["uv", "pip", "install", "--pre", "torch", "torchvision", "torchaudio", "--index-url", NIGHTLY],
                       check=True)
        freeze = subprocess.run(["uv", "pip", "freeze"], capture_output=True, text=True).stdout.splitlines()
        wanted = r"(torch|torchvision|torchaudio|pytorch-triton-xpu)=="
        versions = dict(x.split("==") for x in freeze if re.match(wanted, x))
        for key, result in run_e2e(own, work / "public").items():
            found[key]["public_wheel"] = result | {"versions": versions}
    return evidence | {"confirm": {sku: found}}


def experiment_plan(item: dict, out: Path) -> list[tuple[str, str, dict]]:
    """(name, command, env) for the item's category; every one is read against `rerun`."""
    suite, dtype, mode, model = item["key"].split(",")
    base = command(suite, dtype, mode, model)
    backend = lambda name: base.replace("--backend=inductor", f"--backend={name}")  # noqa: E731
    perf = base.replace("--accuracy", "--performance").replace("-n10", "-n2")
    repro = lambda after, level: {"TORCHDYNAMO_REPRO_AFTER": after, "TORCHDYNAMO_REPRO_LEVEL": str(level)}  # noqa: E731
    batch = int(item["batch_size"]) if str(item.get("batch_size", "")).isdigit() else 0
    plans = {
        "accuracy": [("fp32", command(suite, "float32", mode, model), {}), ("aot_eager", backend("aot_eager"), {}),
                     ("output_code", base, {"TORCH_LOGS": "output_code"}),
                     ("minify_dynamo", f"{base} --minify --output-directory {out}", {}),
                     ("minify_aot", base, repro("aot", 4))],
        "crash": [("aot_eager", backend("aot_eager"), {}), ("eager", backend("eager"), {}),
                  ("minify_dynamo", base, repro("dynamo", 3)), ("minify_aot", base, repro("aot", 3))],
        "eager_nondeterministic": [("deterministic", f"{base} --deterministic", {}),
                                   ("fp32", command(suite, "float32", mode, model), {})],
        "eager_fail": [("cpu_eager", backend("eager").replace("-d xpu", "-d cpu"), {}),
                       ("xpu_eager", backend("eager"), {})],
        "oom": [("memory", f"{perf} --print-memory", {})]
               + ([("half_batch", f"{base} --batch-size {batch // 2}", {})] if batch > 1 else []),
        "timeout": [("trace", base, {"TORCH_TRACE": str(out / "trace")}),
                    ("compile_time", f"{perf} --print-compilation-time", {})],
    }
    return [("rerun", base, {})] + plans.get(item["category"], [])


def run_experiment(name: str, cmd: str, env: dict, timeout: int, out: Path) -> dict:
    log = out / f"{name}.log"
    start = time.monotonic()
    with open(log, "w") as f:
        try:
            rc = subprocess.run(shlex.split(cmd), cwd=PYTORCH, env={**os.environ, **env}, stdout=f,
                                stderr=subprocess.STDOUT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    lines = log.read_text(errors="replace").splitlines()
    for produced in ("repro.py", "torch_compile_debug", "minifier"):  # minifier output lands in the cwd
        if (PYTORCH / produced).exists():
            shutil.move(PYTORCH / produced, out / f"{name}-{produced}")
    last = lines[-1].split() if lines else []
    return {"name": name, "cmd": cmd, "env": env, "rc": rc, "elapsed_s": round(time.monotonic() - start),
            "result": "timeout" if rc == "timeout" else (last[-1] if last else ""),  # the benchmark prints it last
            "log": log.name, "notes": [x for x in lines if NOTE.search(x)][-20:], "tail": lines[-30:]}


def deep(item: dict, out: Path) -> dict:
    deadline = time.monotonic() + ITEM_TIMEOUT
    experiments = []
    for name, cmd, env in experiment_plan(item, out):
        left = int(deadline - time.monotonic())
        if name == "minify_aot" and list(out.glob("*repro.py")):
            continue  # the after-dynamo minifier already produced a subgraph
        if left <= 0:
            experiments.append({"name": name, "cmd": cmd, "env": env, "result": "not run: item out of time"})
        else:
            timeout = min(left, MINIFY_TIMEOUT) if name.startswith("minify") else left
            experiments.append(run_experiment(name, cmd, env, timeout, out))
    return {"item": item, "experiments": experiments}


def verify(triage: dict, base: Path) -> dict:
    """The script fails twice the same way on XPU, passes on CPU, each run within 5 minutes."""
    item, result = triage["item"], triage.get("result") or {}
    script = (base / result.get("repro_script", "")).resolve()
    sig = result.get("signature", "")

    def run(device: str) -> tuple:
        try:
            p = subprocess.run([sys.executable, str(script), "--device", device], capture_output=True, text=True,
                               timeout=VERIFY_TIMEOUT, cwd=script.parent)
            return p.returncode, p.stdout + p.stderr
        except subprocess.TimeoutExpired:
            return "timeout", ""

    if not result.get("repro_script") or not script.is_file():
        return {"passed": False, "reason": "no reproducer script"}
    if item["category"] in ("crash", "eager_fail") and (not sig or sig not in "\n".join(
            [item.get("message", ""), *item.get("log_excerpt", [])])):
        return {"passed": False, "reason": "signature missing or not found in the original failure"}
    (rc1, out1), (rc2, out2), (ref_rc, _) = run("xpu"), run("xpu"), run("cpu")
    reason = next((reason for bad, reason in [
        ("timeout" in (rc1, rc2, ref_rc), f"a run exceeded {VERIFY_TIMEOUT}s"),
        (rc1 == 0 or rc2 == 0, "does not fail on XPU"),
        (rc1 != rc2 or (sig in out1) != (sig in out2), "two XPU runs disagree"),
        (sig not in out1, "XPU failure does not carry the original signature"),
        (ref_rc != 0, "reference run fails"),
    ] if bad), "")
    return {"passed": not reason, "reason": reason}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("confirm")
    p.add_argument("--evidence", type=Path, required=True)
    p.add_argument("--sku", required=True, help="runner label of this job: pvc-1100 or pvc-1550")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--work-dir", type=Path, default=Path("e2e_triage"), help="run_e2e logs")
    p = sub.add_parser("deep")
    p.add_argument("--created", type=Path, required=True)
    p.add_argument("--sku", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p = sub.add_parser("verify")
    p.add_argument("--triage", type=Path, required=True)
    args = parser.parse_args()

    if args.cmd == "verify":
        triage = json.loads(args.triage.read_text())
        triage["verify"] = verify(triage, args.triage.parent)
        args.triage.write_text(json.dumps(triage, indent=1))
        print(json.dumps(triage["verify"]))
        return
    stage()
    if args.cmd == "confirm":
        args.work_dir.mkdir(parents=True, exist_ok=True)
        evidence = confirm(json.loads(args.evidence.read_text()), args.sku, args.work_dir)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(evidence, indent=1))
        return
    queue = [i for i in json.loads(args.created.read_text())["deep_queue"] if i["sku"] == args.sku][:MAX_DEEP]
    for item in queue:
        out = (args.out_dir / f"{item['issue'] or 'draft'}-{item['key'].split(',')[3]}").resolve()
        out.mkdir(parents=True, exist_ok=True)
        (out / "triage.json").write_text(json.dumps(deep(item, out), indent=1))


if __name__ == "__main__":
    main()
