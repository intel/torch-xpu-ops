<!--
Body of an issue filed from a nightly UT run. Rendered by
.github/scripts/ut_create_issues.py, which strips this comment and replaces
every {{TOKEN}}. Nothing here is written by a model: the skill supplies the
summary, and every other slot is filled from the evidence.

The headings below are what the 🐛 Dynamic skip form in ../dynamic-skip.yml
renders from its fields, in its order, so a machine-filed issue and a
hand-written one are the same document. Change a field there and change the
matching heading here. That form's `labels:` apply only to the web flow, so
the script reads them out of it and applies them itself, along with the ones
no form can know: skipped_bmg, regression, new_case_failure.

Two parts are load-bearing and must not change without updating their
consumers:

  * The `Cases:` block is parsed by fetch_issues.sh, by the awk filter in
    _linux_ut.yml and by mark_passed_issue in ut_result_check.sh, and every
    line in it is removed from the next run's failures by `grep -vFxf` in
    ut_result_check.sh. It must start with a line beginning `Cases:`, carry one
    `<category>,<class name>,<test name>` line per case with no blank lines in
    between, end with a blank line, and never be truncated - an incomplete
    block silently fails to skip the case, and a line that names no real case
    silently mutes a future failure. No other line in the body may contain the
    string `Cases:`, or the awk filter starts collecting again there.
  * The trailing ut-auto-issue marker is how a machine-filed issue is
    recognised on later nights.
-->
### 🐛 Describe the bug with skip template

<!-- cases:begin -->
Cases:
{{CASES}}

<!-- cases:end -->

### Summary

{{SUMMARY}}

### ErrorLog

{{ERROR_LOG}}

### Reproduce

```bash
{{REPRODUCE}}
```

### Pytorch Version

{{EVIDENCE}}

### Versions

<details><summary>Detail</summary>

```
{{COLLECT_ENV}}
```

</details>

{{MARKER}}
