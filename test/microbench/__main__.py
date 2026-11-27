# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""``python -m microbench <command>`` (run from ``test/``)."""

import argparse
import json
import os
import statistics
import sys

from .case import canonical_params, load_cases


def _filter(args):
    ops = set(args.op.split(",")) if args.op else None
    ids = set(args.case_id.split(",")) if args.case_id else None
    key = canonical_params(json.loads(args.case)) if args.case else None
    return [
        c
        for c in load_cases().values()
        if (not ops or c.op in ops)
        and (not args.pass_ or c.pass_ == args.pass_)
        and (not args.tier or c.tier == args.tier)
        and (not key or canonical_params(c.params) == key)
        and (not ids or c.case_id in ids)
    ]


def run_list(args):
    cases = _filter(args)
    if args.json:
        json.dump([c.manifest() for c in cases], sys.stdout, indent=1)
        print()
        return 0
    for c in cases:
        print(
            c.case_id
            if args.ids
            else f"{c.case_id}  {c.tier:7s} {c.op}/{c.pass_}  {c.label}"
        )
    print(f"{len(cases)} case(s), {len({c.op for c in cases})} op(s)", file=sys.stderr)
    return 0


def run_local(args):
    """Measure cases in this process (local repro, no A/B)."""
    if args.device == "xpu":
        os.environ.setdefault("ZE_ENABLE_TRACING_LAYER", "1")
    from .measure import Device, measure, ProtocolConfig

    cases = _filter(args)
    if not cases:
        print("no matching case", file=sys.stderr)
        return 1
    dev = Device(args.device, ProtocolConfig.load(args.config, args.quick))
    rc = 0
    for c in cases:
        head = f"{c.case_id} {c.op}/{c.pass_} {c.label}"
        try:
            rec = measure(c, dev, keep_trace=args.keep_trace)
        except Exception as e:  # noqa: BLE001 - report and continue
            print(f"{head}: ERROR {type(e).__name__}: {e}")
            rc = 1
            continue
        dvals = [v for v in rec["device_us"] or [] if v is not None]
        dev_t = f"{statistics.median(dvals):.2f}" if dvals else "-"
        e2e = statistics.median(rec["e2e_us"])
        print(f"{head}: device {dev_t} us, e2e {e2e:.2f} us, iters {rec['iters']}")
        if args.out:
            with open(args.out, "a") as f:
                rec.update(case_id=c.case_id, op=c.op, pass_=c.pass_, label=c.label)
                f.write(json.dumps(rec) + "\n")
    return rc


def main(argv=None):
    from . import judge, measure, reference, run, selection
    from .tools import maintenance

    parser = argparse.ArgumentParser(prog="python -m microbench", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fn, hlp in (
        ("list", run_list, "list registered cases (no torch needed)"),
        ("run", run_local, "measure cases in this process (local repro)"),
    ):
        p = sub.add_parser(name, help=hlp)
        p.add_argument("--op", default=None, help="comma separated op names")
        p.add_argument("--pass", dest="pass_", choices=["fwd", "bwd"], default=None)
        p.add_argument("--tier", choices=["pr", "nightly"], default=None)
        p.add_argument("--case", default=None, help="params JSON as printed in reports")
        p.add_argument("--case-id", default=None, help="comma separated case ids")
        p.set_defaults(func=lambda a, fn=fn: sys.exit(fn(a)))
        if name == "list":
            p.add_argument("--json", action="store_true")
            p.add_argument("--ids", action="store_true")
        else:
            p.add_argument("--device", default="xpu")
            p.add_argument("--config", default=None, help="protocol config JSON")
            p.add_argument("--quick", action="store_true")
            p.add_argument(
                "--keep-trace", default=None, help="save the device-pass trace"
            )
            p.add_argument("--out", default=None, help="append JSONL results")
    measure.add_validate_parser(sub)
    measure.add_worker_parser(sub)
    run.add_schedule_parser(sub)
    run.add_env_parser(sub)
    judge.add_check_parser(sub)
    selection.add_select_parser(sub)
    run.add_ci_parser(sub)
    reference.add_reference_parser(sub)
    maintenance.add_parsers(sub)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
