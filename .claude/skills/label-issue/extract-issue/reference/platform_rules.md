# OS and Platform Rules

## `os`

`Linux`, `Windows`, or `""`.

1. Empty body -> `""`.
2. If the body has an `OS:` line (the `collect_env` form,
   `OS: Ubuntu 22.04 LTS`), classify the text after `OS:`.
3. Otherwise classify the whole body.

Check Windows first, then Linux. First hit wins.

- **Windows**: `windows`, ` win `, `[win]`, `win32`, `msvc`
- **Linux**: `linux`, `ubuntu`, `wsl`, `debian`, `centos`, `rhel`, `fedora`

Windows goes first because a Windows log often also mentions Linux paths, but
not the reverse.

## `platform`

One of `PVC`, `BMG`, `ARC`, `ARL`, `LNL`, `MTL`, `CRI`, or `""`.

Look in this order and stop at the first hit:

1. A `hw: <CODE>` label, e.g. `hw: BMG` -> `BMG`.
2. The title.
3. The body.

Within the title or body, check the codes in table order and take the first
keyword that matches:

| Code | Keywords |
|---|---|
| PVC | `ponte vecchio`, `data center gpu max`, `gpu max 1550`, `gpu max 1100`, `max 1550`, `max 1100`, `pvc`, `1550`, `1100`, `0x0bd5`, `0x0bd6`, `0x0bd9`, `0x0bda`, `0x0bdb` |
| BMG | `battlemage`, `b580`, `b570`, `bmg`, `0xe20b`, `0xe20c`, `0xe223` |
| ARC | `alchemist`, `a770`, `a750`, `a380`, `arc`, `arc a`, `arc graphics` |
| ARL | `arrow lake`, `arl` |
| LNL | `lunar lake`, `lnl` |
| MTL | `meteor lake`, `mtl` |
| CRI | `crescent island`, `cri` |

Short codes and bare numbers (`pvc`, `arc`, `bmg`, `1100`, `b580`, ...) must
match as **whole words**. `arc` must not match inside `architecture` or
`search`; `1100` must not match inside `11000`. Multi-word phrases like
`arc graphics` can match anywhere.

## `platform_specific`

`true` when the issue is reported as affecting specific hardware. Judge this
from the issue text alone - do not probe the local machine.

Set `true` when ANY of these holds:

1. **The title carries a platform tag in brackets** - `[BMG]`, `[PVC]`,
   `[ARL]`.
2. **A `hw:` label is present** - `hw: BMG`, `hw: PVC`.
3. **The description names one platform as the only affected one** -
   `only on BMG`, `only on PVC`, `BMG only`, `happens only on ARL`.
4. **The description contrasts two platforms** - `passed on PVC, failed on
   BMG`, `works on ARC but fails on LNL`, `PVC passes, BMG fails`.

Otherwise `false`.

Case 4 is the strongest signal: an explicit pass/fail split between two
platforms proves the failure is hardware-dependent. Record the FAILING platform
in `platform` when the two disagree.

`false` is the right answer for an issue that never mentions hardware. A
platform code appearing incidentally - in a log path, a machine hostname, or a
`collect_env` dump - is not a claim that the issue is platform-specific, so it
does not set this field.
