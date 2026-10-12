#!/usr/bin/env python3
"""Fetch the "Done" upstream test file list from a tracking issue.

The tracking issue (default: intel/torch-xpu-ops#5205) keeps an
auto-generated list of upstream test files between the
``<!-- auto-file-lists:begin -->`` / ``<!-- auto-file-lists:end -->`` markers.
This script fetches the issue body, extracts the files under the
"Done test files" section and splits them into distributed, inductor
and default buckets.
"""

import argparse
import json
import os
import re
import sys
import urllib.request

API_URL = "https://api.github.com/repos/{owner}/{repo}/issues/{number}"

# Test files are bucketed by their path prefix; anything that does not
# match a specific prefix falls into the default bucket.
DISTRIBUTED_PREFIX = "test/distributed/"
INDUCTOR_PREFIX = "test/inductor/"


def fetch_issue_body(owner, repo, number):
    url = API_URL.format(owner=owner, repo=repo, number=number)
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/vnd.github+json")
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.load(resp)
    return data.get("body") or ""


def extract_done_section(body):
    """Return the markdown for the "Done test files" <details> block."""
    return _extract_section(body, "Done test files")


def extract_not_applicable_section(body):
    """Return the markdown for the "Not Applicable test files" block."""
    return _extract_section(body, "Not Applicable test files")


def _extract_section(body, heading):
    begin = body.find("auto-file-lists:begin")
    if begin != -1:
        body = body[begin:]
    start = body.find(heading)
    if start == -1:
        return ""
    end = body.find("</details>", start)
    if end == -1:
        end = len(body)
    return body[start:end]


def parse_files(section):
    files = []
    seen = set()
    for match in re.findall(r"`([^`]+?\.py)`", section):
        path = re.sub(r"\s+", "", match)
        if path and path not in seen:
            seen.add(path)
            files.append(path)
    return files


def split_files(files):
    distributed = [f for f in files if f.startswith(DISTRIBUTED_PREFIX)]
    inductor = [f for f in files if f.startswith(INDUCTOR_PREFIX)]
    default = [f for f in files
               if not f.startswith((DISTRIBUTED_PREFIX, INDUCTOR_PREFIX))]
    return distributed, inductor, default


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="intel/torch-xpu-ops",
                        help="owner/repo of the tracking issue")
    parser.add_argument("--issue", type=int, default=5205,
                        help="tracking issue number")
    parser.add_argument("--category",
                        choices=["distributed", "inductor", "default", "all"],
                        default="all", help="which bucket of files to print")
    parser.add_argument("--distributed-output",
                        help="write distributed file list to this path")
    parser.add_argument("--inductor-output",
                        help="write inductor file list to this path")
    parser.add_argument("--default-output",
                        help="write default file list to this path")
    args = parser.parse_args()

    owner, _, repo = args.repo.partition("/")
    if not repo:
        parser.error("--repo must be in 'owner/repo' format")

    body = fetch_issue_body(owner, repo, args.issue)
    done_files = parse_files(extract_done_section(body))
    if not done_files:
        print("No Done test files found in issue body", file=sys.stderr)
        return 1

    # Keep only files that are in Done and not in Not Applicable.
    not_applicable = set(parse_files(extract_not_applicable_section(body)))
    files = [f for f in done_files if f not in not_applicable]

    distributed, inductor, default = split_files(files)

    if args.distributed_output:
        with open(args.distributed_output, "w") as fh:
            fh.write("\n".join(distributed) + "\n" if distributed else "")
    if args.inductor_output:
        with open(args.inductor_output, "w") as fh:
            fh.write("\n".join(inductor) + "\n" if inductor else "")
    if args.default_output:
        with open(args.default_output, "w") as fh:
            fh.write("\n".join(default) + "\n" if default else "")

    print(f"Done test files: {len(files)} "
          f"(distributed: {len(distributed)}, inductor: {len(inductor)}, "
          f"default: {len(default)}; "
          f"excluded {len(done_files) - len(files)} Not Applicable)",
          file=sys.stderr)

    if args.category == "distributed":
        selected = distributed
    elif args.category == "inductor":
        selected = inductor
    elif args.category == "default":
        selected = default
    else:
        selected = files
    print("\n".join(selected))
    return 0


if __name__ == "__main__":
    sys.exit(main())
