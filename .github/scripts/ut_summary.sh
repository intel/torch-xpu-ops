#!/usr/bin/env bash
# Combined UT summary for one workflow run.
#
# Walks every Inductor-XPU-UT-Data artifact of the run, compares each category
# against the known-failure list and writes one new-failure report plus one
# markdown panel for all UT jobs together.
#
# Usage: ut_summary.sh <artifacts_dir> <snapshot_dir> <work_dir> [pytorch]
#   artifacts_dir  where `gh run download` put the UT data artifacts
#   snapshot_dir   known-failure issues captured at build start (may be absent)
#   work_dir       output: <ut_job>/ per job, all_new_failures.{txt,csv}, panel.md
# Exits non-zero when any category has new failures or an abnormal case count.

set -uo pipefail

ARTIFACTS_DIR="${1:?artifacts dir required}"
SNAPSHOT_DIR="${2:-}"
WORK_DIR="${3:?work dir required}"
PYTORCH="${4:-nightly_wheel}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "${WORK_DIR}"
# ut_result_check.sh runs from the per-job directory, so every path handed to it
# or to its environment has to survive the `cd`.
WORK_DIR="$(cd "${WORK_DIR}" && pwd)"
ARTIFACTS_DIR="$(cd "${ARTIFACTS_DIR}" && pwd)"
if [[ -d "${SNAPSHOT_DIR}" ]]; then
    SNAPSHOT_DIR="$(cd "${SNAPSHOT_DIR}" && pwd)"
fi

NEW_FAILURE_LIST="${WORK_DIR}/all_new_failures.txt"
NEW_FAILURE_CSV="${WORK_DIR}/all_new_failures.csv"
PANEL="${WORK_DIR}/panel.md"
VERDICTS="${WORK_DIR}/verdicts.md"
KNOWN_STATIC="${WORK_DIR}/known_issue_static.log"

: > "${NEW_FAILURE_LIST}"
: > "${NEW_FAILURE_CSV}"
: > "${PANEL}"
: > "${VERDICTS}"
: > "${KNOWN_STATIC}"

# The static list is runner independent, so it is fetched once for the whole run.
cat "${SNAPSHOT_DIR}/known_issue_static.log" >> "${KNOWN_STATIC}" 2>/dev/null || true
bash "${SCRIPT_DIR}/fetch_issues.sh" static >> "${KNOWN_STATIC}" || true

# The open-issue list depends on the runner and category labels, so it is built
# once per (category, runner) pair and reused by every UT job that matches.
open_issues_for() {
    local category="$1" runner="$2" key cache
    key="$(printf '%s_%s' "${category:-none}" "${runner:-none}" | tr -c 'A-Za-z0-9_' '_')"
    cache="${WORK_DIR}/issues_${key}.log"
    if [[ -f "${cache}" ]]; then
        echo "${cache}"
        return 0
    fi
    : > "${cache}"
    # Two sources: the build-start snapshot covers issues closed mid-run, the
    # live fetch covers issues opened after the build started.
    cat "${SNAPSHOT_DIR}/issues_snapshot.log" >> "${cache}" 2>/dev/null || true
    if [[ "${runner}" == *bmg* ]]; then
        cat "${SNAPSHOT_DIR}/issues_snapshot_bmg.log" >> "${cache}" 2>/dev/null || true
    fi
    if [[ "${category}" == "dpclang" ]]; then
        cat "${SNAPSHOT_DIR}/issues_snapshot_dpclang.log" >> "${cache}" 2>/dev/null || true
    fi
    bash "${SCRIPT_DIR}/fetch_issues.sh" open "${category}" "${runner}" >> "${cache}" || true
    echo "${cache}"
}

categories_of() {
    case "$1" in
        basic) echo "op_regression op_extended op_regression_dev1" ;;
        *) echo "$1" ;;
    esac
}

# Artifact content is test output, so it is read as data only - never sourced.
meta_value() {
    awk -F= -v k="$2" '$1 == k {sub(/^[^=]*=/, ""); print; exit}' "$1"
}

panel_row() {
    printf '| %s | %s | %s | %s |\n' "$1" "$2" "$3" "$4" >> "${VERDICTS}"
}

{
    echo "### Known-failure comparison"
    echo ""
    echo "| UT job | Category | New failures | Result |"
    echo "| --- | --- | --- | --- |"
} >> "${VERDICTS}"

{
    echo "### UT Results Summary"
    echo ""
} >> "${PANEL}"

exit_code=0
job_count=0
shopt -s nullglob
for artifact in "${ARTIFACTS_DIR}"/*/; do
    meta="$(find "${artifact}" -type f -name ut_meta.env -print -quit)"
    if [[ -z "${meta}" ]]; then
        echo "::warning::$(basename "${artifact}") carries no ut_meta.env, skipped"
        continue
    fi
    ut_job="$(meta_value "${meta}" UT_NAME)"
    ut_runner="$(meta_value "${meta}" UT_RUNNER)"
    ut_category="$(meta_value "${meta}" UT_CATEGORY)"
    if [[ -z "${ut_job}" ]]; then
        echo "::warning::$(basename "${artifact}") names no UT job, skipped"
        continue
    fi
    job_count=$((job_count + 1))

    # ut_result_check.sh reads every log from its working directory, and the
    # test job spreads them over ut_log/ and ut_log/<ut_job>/.
    job_dir="${WORK_DIR}/${ut_job}"
    mkdir -p "${job_dir}"
    find "${artifact}" -type f \( -name '*.log' -o -name '*.csv' -o -name '*.xml' \) \
        -exec cp -n {} "${job_dir}/" \; 2>/dev/null || true

    issues_log="$(open_issues_for "${ut_category}" "${ut_runner}")"
    cp "${issues_log}" "${job_dir}/issues.log"
    job_failure_list="${job_dir}/new_failure_list.txt"
    : > "${job_failure_list}"
    export UT_RUN_HEALTH_FILE="${job_dir}/run_health.jsonl"
    : > "${UT_RUN_HEALTH_FILE}"

    echo "::group::${ut_job} (runner: ${ut_runner:-unknown})"
    # Turns the junit reports into the per-category case logs every check below
    # reads, and into the run's result table.
    report="${job_dir}/check_ut.md"
    if compgen -G "${job_dir}/*.xml" > /dev/null; then
        (
            cd "${job_dir}" || exit 1
            python3 "${SCRIPT_DIR}/check-ut.py" -n "${ut_job}" -i ./*.xml
        ) > "${report}" 2>&1 || true
        # Only the counts table: the failure listing it prints first is the whole
        # of ut_failure_list.csv, which is in the artifact.
        {
            echo "#### ${ut_job}"
            echo ""
            awk '/^### Results Summary/{p=1; next} p' "${report}"
            echo ""
        } >> "${PANEL}"
    else
        echo "::warning::${ut_job} produced no junit report"
    fi

    for category in $(categories_of "${ut_job}"); do
        cp "${KNOWN_STATIC}" "${job_dir}/Known_issue.log"
        # Known failures are listed per category under a `Cases:` heading, so
        # only the block belonging to this category is appended.
        awk -v r="${category}" 'BEGIN{ print_row = 0 }{
            if ( ! ( $0 ~ /[a-zA-Z0-9]/ ) ) { print_row = 0 };
            if ( print_row == 1 && $1 ~ r ) { print $0 };
            if ( $0 ~ /Cases:/ ) { print_row = 1 };
        }' "${job_dir}/issues.log" >> "${job_dir}/Known_issue.log"
        sed -i 's/[[:space:]]*$//g' "${job_dir}/Known_issue.log"

        verdict="✅ passed"
        if ! (
            cd "${job_dir}" || exit 1
            TEST_PLATFORM=linux TEST_RUNNER="${ut_runner}" \
                bash "${SCRIPT_DIR}/ut_result_check.sh" "${category}" "${PYTORCH}"
        ); then
            verdict="❌ failed"
            exit_code=1
        fi

        new_failed=0
        filtered="${job_dir}/failures_${category}_filtered.log"
        if [[ -s "${filtered}" ]]; then
            new_failed="$(grep -c . "${filtered}")"
            cat "${filtered}" >> "${job_failure_list}"
        fi
        panel_row "${ut_job}" "${category}" "${new_failed}" "${verdict}"
    done
    echo "::endgroup::"

    # Keep the error message of every new failure: ut_failure_list.csv is the
    # only place it exists, and it is headerless (see check-ut.py print_md_row).
    ut_csv="${job_dir}/ut_failure_list.csv"
    if [[ -s "${job_failure_list}" ]]; then
        cat "${job_failure_list}" >> "${NEW_FAILURE_LIST}"
        if [[ -f "${ut_csv}" ]]; then
            job_csv="${job_dir}/new_ut_failure_list.csv"
            : > "${job_csv}"
            # Match on class name AND test name so a shorter name cannot match
            # a longer one it happens to be a substring of.
            while IFS=',' read -r _category class_name test_name; do
                [[ -z "${test_name}" ]] && continue
                grep -F "| ${class_name} | ${test_name} |" "${ut_csv}" >> "${job_csv}" || true
            done < "${job_failure_list}"
            cat "${job_csv}" >> "${NEW_FAILURE_CSV}"
            expected_count="$(grep -c . "${job_failure_list}")"
            actual_count="$(grep -c . "${job_csv}")"
            if [[ "${actual_count}" -ne "${expected_count}" ]]; then
                echo "::warning::${ut_job} new failure count mismatch: filtered=${expected_count}, csv=${actual_count}"
            fi
        fi
    fi
done

if [[ "${job_count}" -eq 0 ]]; then
    echo "::error::no UT data artifact carried a ut_meta.env; nothing to summarize"
    exit 1
fi

cat "${VERDICTS}" >> "${PANEL}"

exit "${exit_code}"
