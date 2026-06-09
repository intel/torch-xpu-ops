#!/usr/bin/env python3
"""
Launch a full T1/T2/R profiling session on Jenkins from a model list.

Triggers 6 Jenkins builds (t1, unitrace, xpu_profiler, cuda_profiler, xpu_t2, cuda_t2),
waits for all to complete, downloads artifacts, and generates reports.

Usage:
    # From a YAML session file (just model list):
    python scripts/launch_session.py --session my_session.yaml

    # Directly from CLI:
    python scripts/launch_session.py \
        --models "torchbench_resnet50 torchbench_vgg16 HF_AlbertForMaskedLM" \
        --name my_session

    # Dry run (show what would be triggered):
    python scripts/launch_session.py --session my_session.yaml --dry-run

    # Skip launch, only download + report (if jobs already ran):
    python scripts/launch_session.py --session config/sessions/my_session.yaml --download-only
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_SERVER = os.environ.get("JENKINS_SERVER", "https://your-jenkins-server.example.com")
TRIGGER_JOB = "newOOB_launch_benchmark_trigger"

# Required environment variables:
#   JENKINS_SERVER    - Jenkins server URL (default: https://your-jenkins-server.example.com)
#   HF_TOKEN          - Hugging Face Hub access token
#   JENKINS_API_TOKEN - Jenkins API token for triggering builds
#   OOB_MODEL_REPO    - Benchmark model repository URL (optional)
#   HTTP_PROXY        - HTTP proxy server (optional)
_HF_TOKEN = os.environ.get("HF_TOKEN", "")
_JENKINS_TOKEN = os.environ.get("JENKINS_API_TOKEN", "")

# Fixed parameters (same across all 6 passes)
COMMON_PARAMS = {
    "OOB_FRAMEWORK": "pytorch",
    "OOB_MODEL_REPO": os.environ.get(
        "OOB_MODEL_REPO",
        "https://github.com/your-org/benchmark-models.git"
    ),
    "OOB_MODEL_BRANCH": "develop-gpu",
    "OOB_CHANNELS_LAST": "1",
    "BATCH_SIZE_LIST": "default",
    "OOB_NUMA_NODES_USE": "1",
    "CORES_PER_INSTANCE_LIST": "1",
    "OOB_NUM_WARMUP": "10",
    "OOB_NUM_ITER": "100",
    "OOB_PROFILE": "0",
    "OOB_DNNL_VERBOSE": "0",
    "OOB_SLEEP_TIMEOUT": "0,240,20",
    "OOB_REFERENCE_LOG": "",
    "OOB_MLPC_DASHBOARD": "",
    "JENKINS_API_TOKEN": _JENKINS_TOKEN,
    "AUTO_TRIGGER": "false",
    "NODE_PROXY": os.environ.get("HTTP_PROXY", "http://your-proxy.example.com:8080"),
    "train_MODEL_LIST": "",
}

# Per-pass parameter overrides
# Each pass differs only in: NODE_LABEL, OOB_DEVICE, OOB_CONDA_ENV, OOB_ADDITION_ENV, OOB_ADDITION_PARAMS
PASS_TEMPLATES = {
    "t1": {
        "NODE_LABEL": "OOB-RTX4080",
        "OOB_DEVICE": "cuda",
        "OOB_ADDITION_ENV": f"HUGGING_FACE_HUB_TOKEN={_HF_TOKEN},Calculate_Flops=1",
        "OOB_ADDITION_PARAMS": "",
    },
    "unitrace": {
        "NODE_LABEL": "OOB-B70",
        "OOB_DEVICE": "xpu",
        "OOB_ADDITION_ENV": f"HUGGING_FACE_HUB_TOKEN={_HF_TOKEN},OOB_USE_UNITRACE=1",
        "OOB_ADDITION_PARAMS": "",
    },
    "xpu_profiler": {
        "NODE_LABEL": "OOB-B70",
        "OOB_DEVICE": "xpu",
        "OOB_ADDITION_ENV": f"HUGGING_FACE_HUB_TOKEN={_HF_TOKEN}",
        "OOB_ADDITION_PARAMS": "--profile_test",
    },
    "cuda_profiler": {
        "NODE_LABEL": "OOB-RTX4080",
        "OOB_DEVICE": "cuda",
        "OOB_ADDITION_ENV": f"HUGGING_FACE_HUB_TOKEN={_HF_TOKEN}",
        "OOB_ADDITION_PARAMS": "--profile_test",
    },
    "xpu_t2": {
        "NODE_LABEL": "OOB-B70",
        "OOB_DEVICE": "xpu",
        "OOB_ADDITION_ENV": f"HUGGING_FACE_HUB_TOKEN={_HF_TOKEN}",
        "OOB_ADDITION_PARAMS": "",
    },
    "cuda_t2": {
        "NODE_LABEL": "OOB-RTX4080",
        "OOB_DEVICE": "cuda",
        "OOB_ADDITION_ENV": f"HUGGING_FACE_HUB_TOKEN={_HF_TOKEN}",
        "OOB_ADDITION_PARAMS": "",
    },
}

# Conda env depends on device
CONDA_ENVS = {
    "xpu": "OOB_260512",
    "cuda": "OOB_260513",
}


# ---------------------------------------------------------------------------
# Jenkins API helpers
# ---------------------------------------------------------------------------

def curl_fetch(url, auth=None, timeout=30):
    """Fetch URL via curl."""
    cmd = ["curl", "-sk", "--noproxy", "*", "--max-time", str(timeout)]
    if auth:
        cmd.extend(["-u", f"{auth[0]}:{auth[1]}"])
    cmd.append(url)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"curl failed: {result.stderr}")
    return result.stdout


def get_crumb(server):
    """Get Jenkins CSRF crumb."""
    url = f"{server}/crumbIssuer/api/json"
    content = curl_fetch(url)
    data = json.loads(content)
    return data["crumbRequestField"], data["crumb"]


def trigger_build(server, params, crumb_header, crumb_value, auth=None):
    """Trigger a parameterized build. Returns the queue item URL or None."""
    url = f"{server}/job/{TRIGGER_JOB}/buildWithParameters"
    cmd = ["curl", "-sk", "--noproxy", "*", "-X", "POST",
           "-H", f"{crumb_header}: {crumb_value}",
           "-w", "\n%{http_code}",
           "--max-time", "30"]
    if auth:
        cmd.extend(["-u", f"{auth[0]}:{auth[1]}"])
    # Add parameters
    for key, value in params.items():
        cmd.extend(["-d", f"{key}={value}"])
    cmd.append(url)

    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout.strip()
    lines = output.split("\n")
    http_code = lines[-1] if lines else "0"

    if http_code == "201":
        # Extract queue URL from Location header
        # Try -D approach
        return True
    elif http_code == "303":
        return True
    else:
        print(f"    ERROR: HTTP {http_code}")
        if result.stderr:
            print(f"    {result.stderr[:200]}")
        return False


def trigger_build_get_queue(server, params, crumb_header, crumb_value, auth=None):
    """Trigger build and return queue item location."""
    url = f"{server}/job/{TRIGGER_JOB}/buildWithParameters"
    cmd = ["curl", "-sk", "--noproxy", "*", "-X", "POST",
           "-H", f"{crumb_header}: {crumb_value}",
           "-D", "-",  # Dump headers to stdout
           "--max-time", "30"]
    if auth:
        cmd.extend(["-u", f"{auth[0]}:{auth[1]}"])
    for key, value in params.items():
        cmd.extend(["-d", f"{key}={value}"])
    cmd.append(url)

    result = subprocess.run(cmd, capture_output=True, text=True)
    # Parse Location header from response headers
    queue_url = None
    for line in result.stdout.split("\n"):
        if line.lower().startswith("location:"):
            queue_url = line.split(":", 1)[1].strip()
            break

    http_code = None
    for line in result.stdout.split("\n"):
        if line.startswith("HTTP/"):
            parts = line.split()
            if len(parts) >= 2:
                http_code = parts[1]

    if http_code in ("201", "303"):
        return queue_url
    else:
        print(f"    ERROR: HTTP {http_code}")
        return None


def wait_for_queue_item(server, queue_url, auth=None, timeout=120):
    """Wait for a queue item to get a build number."""
    if not queue_url:
        return None
    api_url = queue_url.rstrip("/") + "/api/json"
    start = time.time()
    while time.time() - start < timeout:
        try:
            content = curl_fetch(api_url, auth=auth)
            data = json.loads(content)
            executable = data.get("executable")
            if executable:
                return executable.get("number")
            if data.get("cancelled"):
                return None
        except Exception:
            pass
        time.sleep(5)
    return None


def wait_for_build(server, build_number, auth=None, poll_interval=30, timeout=7200):
    """Wait for a build to complete. Returns result string or None."""
    url = f"{server}/job/{TRIGGER_JOB}/{build_number}/api/json"
    start = time.time()
    while time.time() - start < timeout:
        try:
            content = curl_fetch(url, auth=auth)
            data = json.loads(content)
            result = data.get("result")
            if result is not None:
                return result
        except Exception:
            pass
        time.sleep(poll_interval)
    return None


def get_next_build_number(server, auth=None):
    """Get the next build number that will be assigned."""
    url = f"{server}/job/{TRIGGER_JOB}/api/json"
    content = curl_fetch(url, auth=auth)
    data = json.loads(content)
    return data.get("nextBuildNumber")


# ---------------------------------------------------------------------------
# Session config
# ---------------------------------------------------------------------------

def load_session_yaml(yaml_path):
    """Load session YAML.

    Supports two formats:
    1. Simple model list:
        models:
          - torchbench_resnet50
          - HF_AlbertForMaskedLM

    2. With existing job URLs (for download-only mode):
        t1: https://jenkins.../job/.../743/
        unitrace: https://jenkins.../job/.../738/
    """
    if not HAS_YAML:
        print("ERROR: PyYAML required. Install: pip install pyyaml", file=sys.stderr)
        sys.exit(1)

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    result = {
        "models": [],
        "existing_jobs": {},
        "name": os.path.splitext(os.path.basename(yaml_path))[0],
    }

    # Check for model list
    if "models" in cfg:
        result["models"] = cfg["models"]

    # Check for mode/precision overrides
    if "mode" in cfg:
        result["mode"] = cfg["mode"]
    if "precision" in cfg:
        result["precision"] = cfg["precision"]
    if "conda_env_xpu" in cfg:
        result["conda_env_xpu"] = cfg["conda_env_xpu"]
    if "conda_env_cuda" in cfg:
        result["conda_env_cuda"] = cfg["conda_env_cuda"]

    # Check for existing job URLs (pass_type: url)
    for key, value in cfg.items():
        if isinstance(value, str) and value.startswith("http"):
            match = re.match(r"(https?://[^/]+)/job/([^/]+)/(\d+)/?$", value.strip())
            if match:
                result["existing_jobs"][key] = {
                    "server": match.group(1),
                    "build": int(match.group(3)),
                }

    return result


def build_params(pass_type, model_list, mode="realtime-float16-eager",
                 conda_env_xpu=None, conda_env_cuda=None):
    """Build full parameter dict for a given pass type."""
    template = PASS_TEMPLATES[pass_type]
    device = template["OOB_DEVICE"]

    # Determine conda env
    if device == "xpu":
        conda_env = conda_env_xpu or CONDA_ENVS["xpu"]
    else:
        conda_env = conda_env_cuda or CONDA_ENVS["cuda"]

    params = dict(COMMON_PARAMS)
    params.update(template)
    params["OOB_CONDA_ENV"] = conda_env
    params["MODE_PRECISION_COMPILE_LIST"] = mode
    params["realtime_MODEL_LIST"] = " ".join(model_list)

    return params


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Launch T1/T2/R profiling session from model list")
    parser.add_argument("--session", type=str,
                        help="Path to session YAML config")
    parser.add_argument("--models", type=str,
                        help="Space-separated model list (alternative to YAML)")
    parser.add_argument("--name", type=str, default="unnamed_session",
                        help="Session name (used for output directory)")
    parser.add_argument("--mode", type=str, default="realtime-float16-eager",
                        help="MODE_PRECISION_COMPILE_LIST value")
    parser.add_argument("--server", type=str, default=DEFAULT_SERVER,
                        help="Jenkins server URL")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show parameters without triggering")
    parser.add_argument("--download-only", action="store_true",
                        help="Skip launch, only download from existing job URLs in YAML")
    parser.add_argument("--no-wait", action="store_true",
                        help="Trigger builds and exit without waiting")
    parser.add_argument("--no-download", action="store_true",
                        help="Launch and wait, but skip download/report")
    parser.add_argument("--poll-interval", type=int, default=60,
                        help="Seconds between status checks (default: 60)")
    parser.add_argument("--timeout", type=int, default=7200,
                        help="Max seconds to wait per build (default: 7200)")
    args = parser.parse_args()

    # Check required environment variables
    missing_env = []
    if not _HF_TOKEN:
        missing_env.append("HF_TOKEN")
    if not _JENKINS_TOKEN:
        missing_env.append("JENKINS_API_TOKEN")
    if missing_env and not args.download_only:
        print("ERROR: Missing required environment variables:", file=sys.stderr)
        for var in missing_env:
            print(f"  export {var}=\"<your-token>\"", file=sys.stderr)
        print("\nSet them before running this script.", file=sys.stderr)
        sys.exit(1)

    # Load config
    if args.session:
        cfg = load_session_yaml(args.session)
        model_list = cfg["models"]
        session_name = cfg["name"]
        mode = cfg.get("mode", args.mode)
        existing_jobs = cfg.get("existing_jobs", {})
        conda_env_xpu = cfg.get("conda_env_xpu")
        conda_env_cuda = cfg.get("conda_env_cuda")
    elif args.models:
        model_list = args.models.split()
        session_name = args.name
        mode = args.mode
        existing_jobs = {}
        conda_env_xpu = None
        conda_env_cuda = None
    else:
        parser.error("Provide --session or --models")
        return

    server = args.server
    output_dir = f"raw_logs/{session_name}"

    print("=" * 60)
    print(f"OOB Profiling Session: {session_name}")
    print("=" * 60)
    print(f"  Server:  {server}")
    print(f"  Models:  {len(model_list)}")
    for m in model_list:
        print(f"    - {m}")
    print(f"  Mode:    {mode}")
    print(f"  Output:  {output_dir}")
    print(f"  Passes:  {list(PASS_TEMPLATES.keys())}")
    print()

    # -----------------------------------------------------------------------
    # Download-only mode: use existing job URLs
    # -----------------------------------------------------------------------
    if args.download_only:
        if not existing_jobs:
            print("ERROR: --download-only requires job URLs in YAML", file=sys.stderr)
            sys.exit(1)
        print("Download-only mode: using existing job URLs")
        _run_download(existing_jobs, server, output_dir, session_name)
        return

    # -----------------------------------------------------------------------
    # Build parameters for all 6 passes
    # -----------------------------------------------------------------------
    all_params = {}
    for pass_type in PASS_TEMPLATES:
        all_params[pass_type] = build_params(
            pass_type, model_list, mode=mode,
            conda_env_xpu=conda_env_xpu, conda_env_cuda=conda_env_cuda,
        )

    # -----------------------------------------------------------------------
    # Dry run: show params and exit
    # -----------------------------------------------------------------------
    if args.dry_run:
        print("DRY RUN — would trigger 6 builds with these parameters:\n")
        for pass_type, params in all_params.items():
            print(f"--- {pass_type} ---")
            # Only show params that differ from common
            diffs = {k: v for k, v in params.items()
                     if k not in COMMON_PARAMS or COMMON_PARAMS.get(k) != v}
            for k, v in sorted(diffs.items()):
                val_display = v if len(v) < 80 else v[:77] + "..."
                print(f"  {k}: {val_display}")
            print()
        print(f"URL: {server}/job/{TRIGGER_JOB}/buildWithParameters")
        return

    # -----------------------------------------------------------------------
    # Launch all 6 builds
    # -----------------------------------------------------------------------
    print("Step 1: Getting CRUMB...")
    crumb_header, crumb_value = get_crumb(server)
    print(f"  OK: {crumb_header}")

    print("\nStep 2: Triggering 6 builds...")
    next_build = get_next_build_number(server)
    print(f"  Next build number: {next_build}")

    triggered = {}  # pass_type -> build_number
    for pass_type, params in all_params.items():
        print(f"\n  [{pass_type}] Triggering...")
        queue_url = trigger_build_get_queue(
            server, params, crumb_header, crumb_value)
        if queue_url:
            print(f"    Queue: {queue_url}")
            # Wait for queue item to get assigned a build number
            build_num = wait_for_queue_item(server, queue_url, timeout=120)
            if build_num:
                triggered[pass_type] = build_num
                print(f"    Build #{build_num} started")
            else:
                print(f"    WARNING: Could not get build number from queue")
                # Estimate based on next_build
                triggered[pass_type] = next_build
                next_build += 1
                print(f"    Estimated: #{triggered[pass_type]}")
        else:
            print(f"    FAILED to trigger")

    if not triggered:
        print("\nERROR: No builds triggered", file=sys.stderr)
        sys.exit(1)

    # Save triggered build info to YAML for resume
    session_yaml_path = os.path.join(output_dir, "session.yaml")
    os.makedirs(output_dir, exist_ok=True)
    _save_session_yaml(session_yaml_path, model_list, triggered, server, mode)
    print(f"\n  Session saved: {session_yaml_path}")
    print(f"  (Can resume with: --session {session_yaml_path} --download-only)")

    # Print summary
    print(f"\n  Triggered {len(triggered)}/6 builds:")
    for pass_type, build_num in triggered.items():
        url = f"{server}/job/{TRIGGER_JOB}/{build_num}/"
        print(f"    {pass_type}: #{build_num}  {url}")

    if args.no_wait:
        print("\n  --no-wait: Exiting. Re-run with --download-only when builds finish.")
        return

    # -----------------------------------------------------------------------
    # Wait for all builds to complete
    # -----------------------------------------------------------------------
    print(f"\nStep 3: Waiting for builds (poll every {args.poll_interval}s, timeout {args.timeout}s)...")

    results = {}
    for pass_type, build_num in triggered.items():
        print(f"  [{pass_type}] Waiting for #{build_num}...", end="", flush=True)
        result = wait_for_build(
            server, build_num, poll_interval=args.poll_interval, timeout=args.timeout)
        results[pass_type] = result
        print(f" {result}")

    # Check results
    failures = [p for p, r in results.items() if r != "SUCCESS"]
    if failures:
        print(f"\n  WARNING: {len(failures)} builds failed: {failures}")
        print("  Continuing with download for successful builds...")

    if args.no_download:
        print("\n  --no-download: Skipping download and report generation.")
        return

    # -----------------------------------------------------------------------
    # Download artifacts
    # -----------------------------------------------------------------------
    print("\nStep 4: Downloading artifacts...")
    job_info = {p: {"server": server, "build": b} for p, b in triggered.items()
                if results.get(p) == "SUCCESS"}
    _run_download(job_info, server, output_dir, session_name)


def _save_session_yaml(path, model_list, triggered, server, mode):
    """Save session state as YAML (with both model list and job URLs)."""
    lines = [
        f"# Auto-generated session config",
        f"# Triggered: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
        f"models:",
    ]
    for m in model_list:
        lines.append(f"  - {m}")
    lines.append("")
    lines.append(f"mode: {mode}")
    lines.append("")
    lines.append("# Jenkins job URLs (for --download-only)")
    for pass_type, build_num in triggered.items():
        url = f"{server}/job/{TRIGGER_JOB}/{build_num}/"
        lines.append(f"{pass_type}: {url}")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _run_download(job_info, server, output_dir, session_name):
    """Run the download step using download_jenkins_artifacts.py."""
    # Build CLI args for the download script
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    download_script = os.path.join(scripts_dir, "download_jenkins_artifacts.py")

    cmd = [sys.executable, download_script, "--output", output_dir]
    for pass_type, info in job_info.items():
        build = info["build"]
        flag = f"--{pass_type.replace('_', '-')}"
        cmd.extend([flag, str(build)])
    # Use server from first job
    first_server = next(iter(job_info.values()))["server"]
    cmd.extend(["--server", first_server])

    print(f"  Running: {' '.join(cmd[:6])} ...")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("  WARNING: Download had errors")

    # Generate reports
    print("\nStep 5: Generating reports...")
    report_cmd = [
        sys.executable,
        os.path.join(scripts_dir, "generate_all_eager_reports.py"),
        "--session", output_dir,
        "--output", f"reports/{session_name}/per_model/eager",
    ]
    print(f"  Running: {' '.join(report_cmd[:4])} ...")
    subprocess.run(report_cmd)

    print(f"\n  Done! Reports in: reports/{session_name}/")


if __name__ == "__main__":
    main()
