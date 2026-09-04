# Platform-Specific Rules

An `os` or `hw` label is added **only when the issue is platform-specific** —
that is, only when the issue title, a label, or the description states that the
failure happens *only* on a specific OS or hardware. A platform named merely
incidentally (a `collect_env` dump, a log path, a hostname) is NOT
platform-specific and yields no OS/hardware label.

All labels, codes, and keywords live in `categories.os` and `categories.hw` of
`labels.json`. Always read the label spellings, codes,
and keywords from there — never hard-code an OS or hardware name, code, or label
in this file.

## Step 1 — Is the issue platform-specific?

Judge `platform_specific` from the title, labels, and description text alone
(never probe local hardware). A "platform" is any `code` in `categories.os` or
`categories.hw`. Set `platform_specific = true` if ANY of these holds:

1. The title has a bracketed platform tag naming an OS or hardware code.
2. An OS or hardware label from `categories.os` / `categories.hw` is already
   present on the issue.
3. One platform is named as the only affected (phrasing such as "only on",
   "... only", "fails only on").
4. Two platforms are contrasted (phrasing such as "passed on ..., failed on
   ...", "... passes, ... fails").

Otherwise `platform_specific = false`, and both `os` and `hw` are `""` —
stop here.

Case 4 is the strongest signal; when two platforms disagree, the FAILING one is
the platform to record.

## Step 2 — `os` (only when platform-specific)

An OS `code` from `categories.os`, or `""` when the OS is not the specific
dimension. Determine it only from the text that made the issue OS-specific
(the bracketed tag, the OS label, or the "only on ..." / contrast phrasing):

1. If that text has an `OS:` line (`collect_env` form), classify the text after
   `OS:`; else classify the platform-specifying phrase.
2. Match each entry's `keywords` in `categories.os` file order; first hit wins.

Entries are ordered so the more specific OS is checked first (`WSL` precedes both
`Windows` and `Linux`), so file order is authoritative.

## Step 3 — `hw` (only when platform-specific)

A hardware `code` from `categories.hw`, or `""` when hardware is not the specific
dimension. Stop at the first hit:

1. A hardware label from `categories.hw` present on the issue.
2. The title's platform tag.
3. The platform-specifying phrase in the description.

Check `categories.hw` entries in file order and take the first matching keyword;
record that entry's `code`. Short codes and bare numbers match as **whole words**
only — a code must not match inside a larger word. Multi-word keyword phrases may
match anywhere.
