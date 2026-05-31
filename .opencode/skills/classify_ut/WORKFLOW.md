# classify_ut — Workflow Chart

End-to-end flow for classifying blank-`Reason` rows in XPU UT status workbooks.

## High-level pipeline

```mermaid
flowchart TD
    Start([New classification session]) --> PrepGate{Run optional preparation?<br/>requested OR verdict needs fresh local run}

    PrepGate -- "No (default)" --> InitWB
    PrepGate -- Yes --> S_Neg1

    subgraph S_Neg1["Preparation (OPTIONAL) — classify_ut/preparation/SKILL.md"]
        EnvNeg1["Environment Setup: update_env_from_nightly.sh"]
        EnvNeg1 --> InstallWheel["pip install --pre torch + pytorch-triton-xpu<br/>(from /tmp, PyPI XPU nightly = source of truth)"]
        InstallWheel --> AlignPT["Align PYTORCH_SRC to torch.version.git_version<br/>(auto-stash if dirty, then detach)"]
        AlignPT --> AlignXPU["Align torch-xpu-ops to SHA in third_party/xpu.txt"]
        AlignXPU --> Provenance["Write provenance JSON<br/>(versions, SHAs, xpu_available)"]
        Provenance --> Step0["Local Pre-Screen (OPTIONAL):<br/>run_blank_local_prescreen.py <workbook.xlsx>"]
        Step0 --> LocalResult["Write local_result = STATUS;log_path<br/>{PASS,FAIL,ERROR,TIMEOUT,SKIP,SEGFAULT}"]
    end

    LocalResult --> InitWB

    subgraph S0_prep["Workbook init (always)"]
        InitWB["Open target sheet"]
        InitWB --> ReasonTBD["Init Reason TBD from ORIGINAL workbook<br/>(blank Reason -> True). NEVER modify again"]
        ReasonTBD --> DeriveXPU["Derive XPU metadata from CUDA when blank<br/>CUDA->XPU, _cuda->_xpu"]
    end

    DeriveXPU --> PassCheck{Pre-screen ran AND<br/>local_result == PASS?}
    PassCheck -- "PASS (TERMINAL)" --> LocalPassed["Reason = Local Passed<br/>DetailReason cites log path<br/>SKIP all subskills & further analysis"]
    LocalPassed --> Save
    PassCheck -- "non-PASS / no pre-screen = analyze" --> Route

    Route{Route by status_xpu} -- blank/empty --> BlankSkill["classify_ut/blank/SKILL.md<br/>case-existence analysis"]
    Route -- failed --> FailedSkill["classify_ut/failed/SKILL.md<br/>failure-message + local-run + issue search"]
    Route -- "skipped / xfail" --> SkippedSkill["classify_ut/skipped/SKILL.md<br/>skip-message + linked issue + source"]

    BlankSkill --> Policy
    FailedSkill --> Policy
    SkippedSkill --> Policy

    subgraph POL["User Policy Overrides (authoritative)"]
        Policy["P1 local-retest mandate · P2 NA must carry evidence (don't run)<br/>P3 JIT = Not applicable"]
    end

    Policy --> Classify

    subgraph CLS["Classification decision"]
        Classify["Assign canonical Reason + DetailReason"]
        Classify --> Conf["Tag Confidence: HIGH (issue URL / log / file:line)<br/>MEDIUM (best-fit, no link) / LOW -> Need human check"]
    end

    Conf --> Save

    subgraph OUT["Output"]
        Save["Mark cells blue (ADD8E6) · save evidence to<br/>/tmp/opencode/<workbook>_local_verify/"]
        Save --> SaveWB["Save as .agent.xlsx (never modify original)"]
        SaveWB --> Verify["Verify: 0 blank Reason rows · ZIP integrity · counts match"]
    end

    Verify --> Done([Done])
```

## Reason-label decision (per row, after routing)

```mermaid
flowchart TD
    Row([Blank-Reason row, not Local Passed]) --> CPU{CPU-only test?<br/>_cpu name / requires GPU}
    CPU -- Yes --> NA_CPU["Not applicable<br/>DetailReason = CPU Case"]
    CPU -- No --> Regress{last_status_xpu=passed<br/>but now skipped/blank?}

    Regress -- Yes --> CommunityChk["git log/show on test file"]
    CommunityChk --> CC["Community Change<br/>cite guilty commit / disabled-test URL"]
    Regress -- No --> BaseFn{Base function exists<br/>in PYTORCH_SRC?}

    BaseFn -- No --> CC
    BaseFn -- Yes --> InstDevice{Uses instantiate_device_type_tests?}

    InstDevice -- "No (plain CUDA class)" --> SourceJudge["Judge by CUDA-only API / owner-team"]
    InstDevice -- Yes --> AllowXPU{allow_xpu=True?}
    AllowXPU -- "No (default False)" --> ToEnable["To be enabled<br/>cite file:line, allow_xpu needed"]
    AllowXPU -- Yes --> RunSibling["Run XPU sibling locally (P1)"]

    RunSibling --> Outcome{Outcome}
    Outcome -- PASS --> LP["Local Passed"]
    Outcome -- "FAIL / SIGSEGV" --> Fail["Failures (xpu broken)<br/>issue URL or [Issue_TBD]"]
    Outcome -- "NotImplementedError / dispatch miss" --> FG["Feature gap"]
    Outcome -- "skipIfXpu + linked issue" --> SkipRule["Follow skipped subskill / Dynamic-Skip Rule"]

    SourceJudge --> NAChk{op/API in 'Operation/API'<br/>of 'Not applicable' sheet?}
    NAChk -- Yes --> NA["Not applicable<br/>cite Issue ID + Category"]
    NAChk -- No --> Precedent{>=5 peers & >=95% same Reason?}
    Precedent -- Yes --> WBPrec["Adopt peer Reason (Confidence MEDIUM)<br/>+ 1 independent source axis"]
    Precedent -- No --> Reroute["Re-route: To be enabled / Feature gap /<br/>Failures / Community Change / Need human check"]
```

## Skipped-label (Dynamic-Skip) sub-flow

```mermaid
flowchart TD
    Sk([Issue has 'skipped' label,<br/>no not_target/wontfix]) --> LV["Local verify REQUIRED<br/>port via port-pytorch-tests-xpu on daisyden/pytorch branch"]
    LV --> Mode{Failure mode after skip lifted}
    Mode -- "crash / accuracy / runtime error in existing XPU kernel" --> F["Failures (xpu broken)"]
    Mode -- "NotImplementedError / no XPU kernel / dispatch gap" --> Fg["Feature gap"]
    Mode -- "PASS (skip is stale)" --> Te["To be enabled"]
    F --> Cluster["Same skipped-issue -> same verdict for all siblings in scope"]
    Fg --> Cluster
    Te --> Cluster
```
