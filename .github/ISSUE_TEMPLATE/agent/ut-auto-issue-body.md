<!--
Body template for issues filed by .github/scripts/ut_auto_issue.py.
Placeholders use Python str.format() syntax: {field_name}. Every slot is filled
with a fully pre-rendered string, so add no literal braces to this file.

The shape mirrors issue #5070. Two parts are load-bearing and must not change
without updating their consumers:

  * The `Cases:` block is parsed by fetch_issues.sh plus the awk filter in
    _linux_ut.yml. It must start with a line containing `Cases:`, carry one
    `<category>,<class name>,<test name>` line per case with no blank lines in
    between, and never be truncated - an incomplete block silently fails to
    skip the case. The cases:begin/end markers bound the region the bot is
    allowed to append to on a re-sighting.
  * The trailing ut-auto-issue marker carries the grouping signature and is how
    the bot recognises its own issues. Editing it will cause a duplicate issue.

Everything else is for humans and is safe to edit by hand.
-->
### 🐛 Describe the bug with skip template

<!-- cases:begin -->
Cases:
{cases}
<!-- cases:end -->

{error_log}## Pytorch Version

{version_block}
{evidence_block}### Versions

<details><summary>Detail</summary>

{collect_env}

</details>

<!-- {marker} -->
