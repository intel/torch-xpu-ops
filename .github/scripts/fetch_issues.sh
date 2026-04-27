#!/usr/bin/env bash

set -o pipefail

UT_SKIP_ISSUE=1624

fetch_static() {
  gh --repo intel/torch-xpu-ops issue view "$UT_SKIP_ISSUE" --json body -q .body \
    | sed -E '/^(#|$)/d'
}

fetch_open() {
  gh api --paginate "repos/${GITHUB_REPOSITORY}/issues?labels=skipped&state=open" \
    --jq '.[] | select(.pull_request == null) | "Issue #\(.number): \(.title)\n\(.body)\n"'
}

if [[ "$1" == "static" ]]; then
  fetch_function=fetch_static
elif [[ "$1" == "open" ]]; then
  fetch_function=fetch_open
else
  printf 'Error: invalid mode "%s". Expected "static" or "open".\n' "$1" >&2
  exit 1
fi

if ! output=$("$fetch_function"); then
  printf 'Error: gh call failed.\n' >&2
  exit 1
fi

printf '%s\n' "$output"
