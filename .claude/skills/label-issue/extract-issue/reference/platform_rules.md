# OS and Platform Rules

All codes and keywords live in `categories.os` and `categories.hw` of
`../../reference/proposed_labels.json`. Always read them from there - never
hard-code OS or hardware names here.

## `os`

An OS `code` from `categories.os`, or `""`.

1. Empty body -> `""`.
2. If the body has an `OS:` line (`collect_env` form, e.g.
   `OS: Ubuntu 22.04 LTS`), classify the text after `OS:`; else classify the
   whole body.
3. Match each entry's `keywords` in `categories.os` file order; first hit wins.

Entries are ordered so the more specific OS is checked first (a log for one OS
often mentions another's paths, not the reverse), so file order is authoritative.

## `platform`

A hardware `code` from `categories.hw`, or `""`. Stop at the first hit:

1. A `hw: <CODE>` label (e.g. `hw: BMG` -> `BMG`).
2. The title.
3. The body.

In title/body, check `categories.hw` entries in file order and take the first
matching keyword; record that entry's `code`.

Short codes and bare numbers (illustrative: `pvc`, `arc`, `1100`, ...) match as
**whole words** only - a code must not match inside a larger word (`arc` not
inside `architecture`, `1100` not inside `11000`). Multi-word keyword phrases may
match anywhere.

## `platform_specific`

`true` when the issue is reported as affecting specific hardware, judged from the
text alone. A "platform" is any `categories.hw` `code`. Set `true` if ANY holds
(examples below use placeholder codes for illustration only):

1. Title has a bracketed platform tag - e.g. `[<CODE>]`.
2. A `hw:` label is present.
3. One platform named as the only affected - e.g. `only on <CODE>`, `<CODE>
   only`.
4. Two platforms contrasted - e.g. `passed on <A>, failed on <B>`, `<A> passes,
   <B> fails`.

Otherwise `false`.

Case 4 is the strongest signal; when two platforms disagree, record the FAILING
one in `platform`. A code appearing incidentally (log path, hostname,
`collect_env` dump) does NOT set `platform_specific`.
