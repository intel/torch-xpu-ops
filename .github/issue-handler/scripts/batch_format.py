#!/usr/bin/env python3
"""Batch format issues on intel-sandbox/torch-xpu-ops-exp.

Usage:
  python scripts/batch_format.py 191 160 32 ...
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

# ISSUE_REPO defaults to intel-sandbox/torch-xpu-ops-exp via agent_config.yml
# Unset REVIEW_GH_TOKEN so _token_for_repo falls through to GH_TOKEN
os.environ.pop("REVIEW_GH_TOKEN", None)

from issue_handler.format_agent import run
from issue_handler.utils.config import ISSUE_REPO
from issue_handler.utils.body_templates import get_status
from issue_handler.utils import git as gh


def _log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{ts}] {msg}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch format issues")
    parser.add_argument("issues", type=int, nargs="+", help="Issue numbers")
    args = parser.parse_args()

    _log(f"Target repo: {ISSUE_REPO}")
    _log(f"Processing {len(args.issues)} issues: {args.issues}")

    results: dict[int, str] = {}
    for num in args.issues:
        _log(f"  #{num}: checking...")
        try:
            detail = gh.get_issue_detail(ISSUE_REPO, num)
            body = detail.get("body", "") or ""
            stage = get_status(body)

            if stage is not None:
                _log(f"  #{num}: already formatted (stage={stage}), skipping")
                results[num] = f"skipped:{stage}"
                continue

            _log(f"  #{num}: formatting...")
            run(num)
            _log(f"  #{num}: done ✓")
            results[num] = "ok"

        except Exception as e:
            _log(f"  #{num}: ERROR — {e}")
            results[num] = f"error:{e}"

        # Brief pause between issues to avoid rate limits
        time.sleep(5)

    _log(f"\n{'='*60}")
    _log(f"Results:")
    ok = sum(1 for v in results.values() if v == "ok")
    skipped = sum(1 for v in results.values() if v.startswith("skipped"))
    errors = sum(1 for v in results.values() if v.startswith("error"))
    _log(f"  ✓ {ok} formatted | ⏭ {skipped} skipped | ✗ {errors} errors")
    for num, result in results.items():
        _log(f"  #{num}: {result}")


if __name__ == "__main__":
    main()
