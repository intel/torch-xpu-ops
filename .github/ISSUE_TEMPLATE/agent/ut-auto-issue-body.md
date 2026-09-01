<!--
Body template for issues filed from a nightly UT run by the ut-issue-authoring
skill. Copy the structure exactly and replace the <angle bracket> slots.

Two parts are load-bearing and must not change without updating their consumers:

  * The `Cases:` block is parsed by fetch_issues.sh plus the awk filter in
    _linux_ut.yml, and every line in it is removed from the next run's failures
    by `grep -vFxf` in ut_result_check.sh. It must start with a line containing
    `Cases:`, carry one `<category>,<class name>,<test name>` line per case with
    no blank lines in between, and never be truncated or abbreviated - an
    incomplete block silently fails to skip the case, and a line that names no
    real case silently mutes a future failure. Every line is copied verbatim
    from the evidence; none is ever typed out or reformatted.
  * The trailing ut-auto-issue marker is how a machine-filed issue is
    recognised on later nights.

Everything else is for humans and is safe to edit by hand.
-->
### 🐛 Describe the bug with skip template

<!-- cases:begin -->
Cases:
<one verbatim case line per case, no blank lines, never abbreviated>
<!-- cases:end -->

## Summary

<one to three sentences on what is failing and why these cases are one bug>

## ErrorLog

### <the failing message, as it appears in the evidence>

```
<traceback lines copied from tracebacks.json, or "No traceback captured in the JUnit XML.">
```

## Reproduce

```bash
<the cd from reproduce.file_path>
<the command from reproduce.command_template, with the case substituted>
```

## Pytorch Version

<the version lines: detected in, latest good where it applies, current>

<the evidence block for this issue's classification, pasted from blocks.json>

### Versions

<details><summary>Detail</summary>

<collect_env for this issue's leg, from run.json>

</details>

<!-- ut-auto-issue:v1:run=<run id>:part=<n>/<total> -->
