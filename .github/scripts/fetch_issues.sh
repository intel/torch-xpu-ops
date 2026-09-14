#!/usr/bin/env bash

set -xe
set -o pipefail

UT_SKIP_ISSUE=1624

fetch_static() {
  gh --repo intel/torch-xpu-ops issue view "$UT_SKIP_ISSUE" --json body -q .body \
    | sed -E '/^(#|$)/d'
}

fetch_open() {
  local category="${1:-}"
  local runner="${2:-}"
  local label_scope="${3:-all}"
  local labels=()

  # extra-only skips the base 'skipped' label so callers can snapshot runner/category
  # specific labels separately from the shared list.
  if [[ "${label_scope}" != "extra-only" ]]; then
    labels+=("skipped")
  fi
  # skipped_bmg is a BMG-only known failure; honor it only on BMG runners.
  if [[ "${runner}" == *"bmg"* ]]; then
    labels+=("skipped_bmg")
  fi
  if [[ "${category}" == "dpclang" ]]; then
    labels+=("skipped_dpclang")
  fi

  # Use the REST list endpoint (strongly consistent) instead of search/issues so
  # issues opened after build start are picked up by the live fetch.
  for label in "${labels[@]}"; do
    gh api --paginate "repos/${GITHUB_REPOSITORY}/issues?state=open&labels=${label}" \
      --jq '.[] | select(.pull_request == null) | "Issue #\(.number): \(.title)\n\(.body)\n"'
  done
}

if [[ "$1" == "static" ]]; then
  fetch_function=fetch_static
elif [[ "$1" == "open" ]]; then
  fetch_function=fetch_open
else
  printf 'Error: invalid mode "%s". Expected "static" or "open".\n' "$1" >&2
  exit 1
fi

"$fetch_function" "${2:-}" "${3:-}" "${4:-}"
