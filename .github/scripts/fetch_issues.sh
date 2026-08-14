#!/usr/bin/env bash

set -xe
set -o pipefail

UT_SKIP_ISSUE=1624

fetch_static() {
  gh --repo intel/torch-xpu-ops issue view "$UT_SKIP_ISSUE" --json body -q .body \
    | sed -E '/^(#|$)/d'
}

fetch_open() {
  gh api --method GET --paginate search/issues \
    -f q="repo:${GITHUB_REPOSITORY} is:issue state:open (label:skipped OR label:skipped_bmg)" \
    --jq '.items[] | "Issue #\(.number): \(.title)\n\(.body)\n"'
}

if [[ "$1" == "static" ]]; then
  fetch_function=fetch_static
elif [[ "$1" == "open" ]]; then
  fetch_function=fetch_open
else
  printf 'Error: invalid mode "%s". Expected "static" or "open".\n' "$1" >&2
  exit 1
fi

"$fetch_function"
