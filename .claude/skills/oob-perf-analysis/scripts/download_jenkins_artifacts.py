#!/usr/bin/env python3
"""
Download Jenkins sub-job artifacts for a multi-pass profiling session.

Usage (YAML session config — preferred):
    python scripts/download_jenkins_artifacts.py \
        --session config/sessions/b70_vs_4080s_fp16_eager.yaml \
        [--parallel 5] [--dry-run]

Usage (CLI arguments — legacy):
    python scripts/download_jenkins_artifacts.py \
        --unitrace 738 \
        --xpu-profiler 737 \
        --cuda-profiler 735 \
        --xpu-t2 740 \
        --cuda-t2 739 \
        --t1 743 \
        --output raw_logs/jenkins/my_session \
        [--server https://your-jenkins-server.example.com] \
        [--user your-username --token <token>] \
        [--parallel 5] \
        [--dry-run]

Each argument is the trigger job build number. The script will:
1. Query each trigger job's summary.log
2. Parse model info and sub-job links
3. Download all sub-job artifacts organized by model and pass type
"""

import argparse
import csv
import io
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


DEFAULT_SERVER = os.environ.get("JENKINS_SERVER", "https://your-jenkins-server.example.com")
TRIGGER_JOB = "newOOB_launch_benchmark_trigger"
SUB_JOB = "newOOB_launch_benchmark"


def load_session_config(yaml_path):
    """Load session configuration from YAML file.

    YAML format is simply:
        pass_type: <jenkins_trigger_job_url>

    Example:
        t1: https://your-jenkins-server.example.com/job/newOOB_launch_benchmark_trigger/743/
        unitrace: https://your-jenkins-server.example.com/job/newOOB_launch_benchmark_trigger/738/

    Returns dict with keys: server, auth, passes, session_name
    """
    if not HAS_YAML:
        print("ERROR: PyYAML required for --session. Install: pip install pyyaml",
              file=sys.stderr)
        sys.exit(1)

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    # Parse each entry: pass_type -> URL
    # Extract server and build number from URLs
    passes = {}
    server = None
    auth = None

    for key, value in cfg.items():
        if not isinstance(value, str) or not value.startswith("http"):
            continue
        # Parse URL: https://<server>/job/<trigger_job>/<build>/
        match = re.match(
            r"(https?://[^/]+)/job/([^/]+)/(\d+)/?$", value.strip()
        )
        if not match:
            print(f"WARNING: Cannot parse URL for '{key}': {value}", file=sys.stderr)
            continue
        if server is None:
            server = match.group(1)
        passes[key] = int(match.group(3))

    if not passes:
        print("ERROR: No valid Jenkins URLs found in YAML", file=sys.stderr)
        sys.exit(1)

    # Derive session name from YAML filename
    session_name = os.path.splitext(os.path.basename(yaml_path))[0]

    return {
        "server": server or DEFAULT_SERVER,
        "auth": auth,
        "passes": passes,
        "session_name": session_name,
    }

# Artifact patterns per pass type
PASS_ARTIFACTS = {
    "unitrace": {
        "patterns": ["python.*.json", "rcpi1-ins0.log"],
        "description": "XPU kernel-level GPU timing (unitrace chrome trace)",
    },
    "xpu_profiler": {
        "patterns": [
            "timeline/trace.json",
            "timeline/profile.pt",
            "timeline/profile_detail.pt",
            "profile_parser.log",
            "operator.log",
            "rcpi1-ins0.log",
        ],
        "description": "XPU PyTorch profiler trace",
    },
    "cuda_profiler": {
        "patterns": [
            "timeline/trace.json",
            "timeline/profile.pt",
            "timeline/profile_detail.pt",
            "profile_parser.log",
            "operator.log",
            "rcpi1-ins0.log",
        ],
        "description": "CUDA PyTorch profiler trace",
    },
    "xpu_t2": {
        "patterns": ["rcpi1-ins0.log"],
        "description": "XPU actual wall time (no profiler overhead)",
    },
    "cuda_t2": {
        "patterns": ["rcpi1-ins0.log"],
        "description": "CUDA actual wall time (no profiler overhead)",
    },
    "t1": {
        "patterns": ["rcpi1-ins0.log"],  # calcflops.txt TBD
        "description": "Theoretical minimum time (roofline)",
    },
}


def curl_fetch(url, auth=None, timeout=60):
    """Fetch URL content using curl (handles Intel internal SSL/proxy issues)."""
    cmd = ["curl", "-sk", "--noproxy", "*", "--max-time", str(timeout)]
    if auth:
        cmd.extend(["-u", f"{auth[0]}:{auth[1]}"])
    cmd.append(url)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"curl failed for {url}: {result.stderr}")
    return result.stdout


def curl_download(url, output_path, auth=None, timeout=120):
    """Download a file using curl."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cmd = ["curl", "-sk", "--noproxy", "*", "--max-time", str(timeout),
           "-o", output_path]
    if auth:
        cmd.extend(["-u", f"{auth[0]}:{auth[1]}"])
    cmd.append(url)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return False, f"curl failed: {result.stderr}"
    # Check if file was actually created and non-empty
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        return False, "Empty or missing file"
    return True, None


def query_trigger_job(server, trigger_build, auth=None):
    """Query trigger job API and return parsed JSON."""
    url = f"{server}/job/{TRIGGER_JOB}/{trigger_build}/api/json"
    content = curl_fetch(url, auth=auth)
    return json.loads(content)


def download_summary_log(server, trigger_build, auth=None):
    """Download and parse summary.log from a trigger job."""
    url = f"{server}/job/{TRIGGER_JOB}/{trigger_build}/artifact/summary.log"
    content = curl_fetch(url, auth=auth)
    reader = csv.DictReader(io.StringIO(content))
    models = []
    for row in reader:
        link = row.get("link", "").strip()
        # Extract sub-job build number and artifact dir from link
        # Link format: https://.../job/newOOB_launch_benchmark/<N>/artifact/<dir>
        sub_build = None
        artifact_dir = None
        match = re.search(r"/job/newOOB_launch_benchmark/(\d+)/artifact/(.+)$", link)
        if match:
            sub_build = int(match.group(1))
            artifact_dir = match.group(2)
        models.append({
            "model_name": row.get("model_name", ""),
            "mode_name": row.get("mode_name", ""),
            "compile_mode": row.get("compile_mode", ""),
            "precision": row.get("precision", ""),
            "batch_size": int(row.get("batch_size", 0)),
            "throughput": float(row.get("throughput", 0)),
            "device": row.get("device", ""),
            "link": link,
            "sub_build": sub_build,
            "artifact_dir": artifact_dir,
        })
    return models


def query_sub_job_artifacts(server, sub_build, auth=None):
    """Query sub-job API to get exact artifact file list."""
    url = f"{server}/job/{SUB_JOB}/{sub_build}/api/json"
    content = curl_fetch(url, auth=auth)
    data = json.loads(content)
    return [art["relativePath"] for art in data.get("artifacts", [])]


def get_short_model_name(model_name):
    """Strip prefix to get short model name."""
    for prefix in ["torchbench_", "timm_", "HF_"]:
        if model_name.startswith(prefix):
            return model_name[len(prefix):]
    return model_name


def download_model_pass(server, model_info, pass_type, output_base, auth=None,
                        timeout=120):
    """Download all artifacts for one model + one pass type."""
    sub_build = model_info["sub_build"]
    artifact_dir = model_info["artifact_dir"]
    short_name = get_short_model_name(model_info["model_name"])
    out_dir = os.path.join(output_base, short_name, pass_type)
    os.makedirs(out_dir, exist_ok=True)

    # Query sub-job to get actual file list
    artifacts = query_sub_job_artifacts(server, sub_build, auth=auth)

    results = []
    for rel_path in artifacts:
        # Skip summary.log at top level
        if rel_path == "summary.log":
            continue
        # Determine the file relative to artifact_dir
        if artifact_dir and rel_path.startswith(artifact_dir):
            file_rel = rel_path[len(artifact_dir):].lstrip("/")
        else:
            file_rel = os.path.basename(rel_path)

        # Determine local output path
        # Flatten: timeline/trace.json -> trace.json, keep structure for clarity
        local_path = os.path.join(out_dir, file_rel)

        # Download
        url = f"{server}/job/{SUB_JOB}/{sub_build}/artifact/{rel_path}"
        success, err = curl_download(url, local_path, auth=auth, timeout=timeout)
        results.append({
            "file": file_rel,
            "success": success,
            "error": err,
            "local_path": local_path,
        })

    return short_name, pass_type, results


def main():
    parser = argparse.ArgumentParser(
        description="Download Jenkins sub-job artifacts for multi-pass profiling"
    )
    # YAML session config (preferred)
    parser.add_argument("--session", type=str,
                        help="Path to session YAML config file (preferred input method)")
    # Legacy CLI arguments
    parser.add_argument("--unitrace", type=int, help="Unitrace trigger job build number")
    parser.add_argument("--xpu-profiler", type=int, help="XPU profiler trigger job build number")
    parser.add_argument("--cuda-profiler", type=int, help="CUDA profiler trigger job build number")
    parser.add_argument("--xpu-t2", type=int, help="XPU T2 trigger job build number")
    parser.add_argument("--cuda-t2", type=int, help="CUDA T2 trigger job build number")
    parser.add_argument("--t1", type=int, help="T1 trigger job build number")
    parser.add_argument("--output", type=str,
                        help="Output directory for downloaded artifacts")
    parser.add_argument("--server", type=str, default=DEFAULT_SERVER,
                        help="Jenkins server URL")
    parser.add_argument("--user", type=str, help="Jenkins username (if auth required)")
    parser.add_argument("--token", type=str, help="Jenkins API token (if auth required)")
    parser.add_argument("--parallel", type=int, default=5,
                        help="Max parallel downloads")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only show what would be downloaded")
    args = parser.parse_args()

    # Load configuration from YAML or CLI args
    if args.session:
        # YAML-based configuration
        cfg = load_session_config(args.session)
        server = cfg["server"]
        auth = cfg["auth"]
        passes = cfg["passes"]
        output_base = args.output or f"raw_logs/{cfg['session_name']}"
        print(f"Session: {cfg['session_name']}")
        print(f"Config:  {args.session}")
        print(f"Server:  {server}")
        print(f"Output:  {output_base}")
        print(f"Passes:  {list(passes.keys())}")
        print()
    else:
        # Legacy CLI-based configuration
        server = args.server
        auth = None
        if args.user and args.token:
            auth = (args.user, args.token)

        passes = {}
        if args.unitrace:
            passes["unitrace"] = args.unitrace
        if args.xpu_profiler:
            passes["xpu_profiler"] = args.xpu_profiler
        if args.cuda_profiler:
            passes["cuda_profiler"] = args.cuda_profiler
        if args.xpu_t2:
            passes["xpu_t2"] = args.xpu_t2
        if args.cuda_t2:
            passes["cuda_t2"] = args.cuda_t2
        if args.t1:
            passes["t1"] = args.t1

        if not passes:
            print("ERROR: Provide --session <yaml> or at least one pass (--t1, --unitrace, etc.)",
                  file=sys.stderr)
            sys.exit(1)

        if not args.output:
            print("ERROR: --output is required when not using --session", file=sys.stderr)
            sys.exit(1)
        output_base = args.output
    os.makedirs(output_base, exist_ok=True)

    # Step 1: Query trigger jobs and parse summary.logs
    print("=" * 60)
    print("Step 1: Querying trigger jobs...")
    print("=" * 60)

    all_pass_models = {}  # pass_type -> [model_info, ...]
    trigger_info = {}

    for pass_type, build_num in passes.items():
        print(f"\n  [{pass_type}] Querying trigger job #{build_num}...")
        try:
            job_data = query_trigger_job(server, build_num, auth=auth)
            result = job_data.get("result")
            if result is None:
                print("    WARNING: Job still RUNNING. Skipping.")
                continue
            if result != "SUCCESS":
                print(f"    WARNING: Job result = {result}. Attempting anyway.")

            # Extract parameters
            params = {}
            for action in job_data.get("actions", []):
                if action.get("_class") == "hudson.model.ParametersAction":
                    for p in action.get("parameters", []):
                        params[p["name"]] = p["value"]

            trigger_info[pass_type] = {
                "build": build_num,
                "result": result,
                "device": params.get("OOB_DEVICE"),
                "node": params.get("NODE_LABEL"),
                "mode": params.get("MODE_PRECISION_COMPILE_LIST"),
            }

            # Download summary.log
            models = download_summary_log(server, build_num, auth=auth)
            all_pass_models[pass_type] = models
            print(f"    Result: {result}")
            print(f"    Device: {params.get('OOB_DEVICE')} ({params.get('NODE_LABEL')})")
            print(f"    Models: {len(models)}")
        except Exception as e:
            print(f"    ERROR: {e}")
            continue

    # Step 2: Build download plan
    print("\n" + "=" * 60)
    print("Step 2: Building download plan...")
    print("=" * 60)

    download_tasks = []  # (pass_type, model_info)
    for pass_type, models in all_pass_models.items():
        for model in models:
            if model["sub_build"] is None:
                print(f"  SKIP: {model['model_name']} ({pass_type}) - no sub-job link")
                continue
            download_tasks.append((pass_type, model))

    print(f"\n  Total download tasks: {len(download_tasks)}")
    print(f"  (= {len(all_pass_models)} passes x ~{len(download_tasks) // max(len(all_pass_models), 1)} models)")

    if args.dry_run:
        print("\n  [DRY RUN] Would download:")
        for pass_type, model in download_tasks:
            short_name = get_short_model_name(model["model_name"])
            print(f"    {short_name}/{pass_type}/ <- sub-job #{model['sub_build']}")
        sys.exit(0)

    # Step 3: Download in parallel
    print("\n" + "=" * 60)
    print(f"Step 3: Downloading artifacts (parallel={args.parallel})...")
    print("=" * 60)

    success_count = 0
    fail_count = 0

    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = {}
        for pass_type, model in download_tasks:
            future = executor.submit(
                download_model_pass,
                server, model, pass_type, output_base, auth=auth
            )
            futures[future] = (pass_type, model["model_name"])

        for future in as_completed(futures):
            pass_type, model_name = futures[future]
            short_name = get_short_model_name(model_name)
            try:
                _, _, results = future.result()
                ok = sum(1 for r in results if r["success"])
                fail = sum(1 for r in results if not r["success"])
                success_count += ok
                fail_count += fail
                status = "OK" if fail == 0 else f"PARTIAL ({fail} failed)"
                print(f"  {short_name}/{pass_type}: {ok} files {status}")
                for r in results:
                    if not r["success"]:
                        print(f"    FAILED: {r['file']} - {r['error']}")
            except Exception as e:
                fail_count += 1
                print(f"  {short_name}/{pass_type}: ERROR - {e}")

    # Step 4: Generate metadata
    print("\n" + "=" * 60)
    print("Step 4: Generating metadata...")
    print("=" * 60)

    # Collect all unique models with their info
    all_models = {}
    for pass_type, models in all_pass_models.items():
        for model in models:
            name = model["model_name"]
            if name not in all_models:
                all_models[name] = {
                    "model_name": name,
                    "short_name": get_short_model_name(name),
                    "batch_size": model["batch_size"],
                    "mode": model["mode_name"],
                    "compile_mode": model["compile_mode"],
                    "precision": model["precision"],
                    "sub_jobs": {},
                    "throughput": {},
                }
            all_models[name]["sub_jobs"][pass_type] = model["sub_build"]
            all_models[name]["throughput"][pass_type] = model["throughput"]

    metadata = {
        "trigger_jobs": trigger_info,
        "output_dir": output_base,
        "models": list(all_models.values()),
    }

    metadata_path = os.path.join(output_base, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved: {metadata_path}")

    # Step 5: Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Output directory: {output_base}")
    print(f"  Files downloaded: {success_count}")
    print(f"  Files failed:     {fail_count}")
    print(f"  Models:           {len(all_models)}")
    print(f"  Passes:           {list(all_pass_models.keys())}")
    print()

    # Print T2 comparison if both xpu_t2 and cuda_t2 available
    if "xpu_t2" in all_pass_models and "cuda_t2" in all_pass_models:
        print("  T2 Comparison (XPU vs CUDA):")
        print(f"  {'Model':<45} {'BS':>5} {'XPU(ms)':>10} {'CUDA(ms)':>10} {'Ratio':>7}")
        print("  " + "-" * 80)
        for name, info in sorted(all_models.items()):
            xpu_t2 = info["throughput"].get("xpu_t2", 0)
            cuda_t2 = info["throughput"].get("cuda_t2", 0)
            ratio = xpu_t2 / cuda_t2 if cuda_t2 > 0 else 0
            marker = " *" if ratio < 1.0 else ""
            print(f"  {info['short_name']:<45} {info['batch_size']:>5} "
                  f"{xpu_t2:>10.2f} {cuda_t2:>10.2f} {ratio:>6.2f}x{marker}")
        print("\n  (* = XPU faster than CUDA)")


if __name__ == "__main__":
    main()
