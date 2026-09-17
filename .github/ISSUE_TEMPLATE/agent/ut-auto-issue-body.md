<!--
Body of an issue filed from a nightly UT run. Rendered by
.github/scripts/ut_create_issues.py, which strips this comment and replaces
every {{TOKEN}}. Nothing here is written by a model: the skill supplies the
summary, and every other slot is filled from the evidence.

The two `###` headings are what the 🐛 Dynamic skip form in ../dynamic-skip.yml
renders, so a machine-filed issue and a hand-written one read the same and are
parsed the same. That form's `labels:` apply only to the web flow, so the
labels of a machine-filed issue are applied by the script instead.

Two parts are load-bearing and must not change without updating their
consumers:

  * The `Cases:` block is parsed by fetch_issues.sh plus the awk filter in
    _linux_ut.yml, and every line in it is removed from the next run's failures
    by `grep -vFxf` in ut_result_check.sh. It must start with a line containing
    `Cases:`, carry one `<category>,<class name>,<test name>` line per case with
    no blank lines in between, and never be truncated - an incomplete block
    silently fails to skip the case, and a line that names no real case
    silently mutes a future failure. Every line is copied from the evidence and
    checked against it before the issue is created.
  * The trailing ut-auto-issue marker is how a machine-filed issue is
    recognised on later nights.
-->
### 🐛 Describe the bug with skip template

<!-- cases:begin -->
Cases:
{{CASES}}
<!-- cases:end -->

## Summary

{{SUMMARY}}

## ErrorLog

{{ERROR_LOG}}

## Reproduce

```bash
{{REPRODUCE}}
```

## Pytorch Version

{{EVIDENCE}}

### Versions

<details><summary>Detail</summary>

```
{{COLLECT_ENV}}
```

</details>

{{MARKER}}
