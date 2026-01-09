# Copyright 2020-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

import json
import subprocess
import sys
from pathlib import Path

def main(jsonl_path, repeats=3, log_dir="logs_by_op"):
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    print(f"📝 Logs will be saved per-op in: {log_dir.resolve()}")

    with open(jsonl_path, encoding="utf-8") as f:
        total_cases = sum(1 for line in f if line.strip())

    case_idx = 0
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            case_idx += 1

            try:
                obj = json.loads(line)
                op = obj["op"]
                case_dict = obj["case"]
                case_str = json.dumps(case_dict, separators=(",", ":"))

                log_path = log_dir / f"{op}.log"

                print(f"[{case_idx}/{total_cases}] op={op} → {log_path.name}")

                for run_idx in range(1, repeats + 1):
                    cmd = [sys.executable, "run.py", "--op", op, "--case", case_str]

                    with open(log_path, "a", encoding="utf-8") as log_file:
                        log_file.write("\n" + "=" * 80 + "\n")
                        log_file.write(f"# CASE {case_idx} | RUN {run_idx}/{repeats}\n")
                        log_file.write(f"# Command: {' '.join(cmd)}\n")
                        log_file.write(f"# Config: {case_str}\n")
                        log_file.write("=" * 80 + "\n\n")

                        proc = subprocess.Popen(
                            cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            bufsize=1,
                            universal_newlines=True,
                        )

                        for out_line in proc.stdout:
                            log_file.write(out_line)
                            log_file.flush()

                        proc.wait()
                        log_file.write(
                            f"\n# [RUN {run_idx}] Exit code: {proc.returncode}\n"
                        )
                        log_file.write("-" * 80 + "\n")

                print(f"    ✅ {repeats} runs appended to {log_path.name}")

            except Exception as e:
                error_msg = f"[CASE {case_idx}] Parse/launch error: {e}"
                print(f"❌ {error_msg}", file=sys.stderr)
                try:
                    op = obj.get("op", "UNKNOWN")
                except Exception:
                    op = "UNKNOWN"
                with open(log_dir / f"{op}_ERRORS.log", 'a') as ef:
                    ef.write(error_msg + "\n")

    print(
        f"\n🎉 Finished {case_idx} cases ({repeats} runs). Logs in: {log_dir.resolve()}"
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python run_cases.py <jsonl_file> [repeats=3] [log_dir=./logs_by_op]"
        )
        sys.exit(1)

    jsonl_path = sys.argv[1]
    repeats = int(sys.argv[2]) if len(sys.argv) >= 3 else 3
    log_dir = sys.argv[3] if len(sys.argv) >= 4 else "logs_by_op"

    main(jsonl_path, repeats=repeats, log_dir=log_dir)
