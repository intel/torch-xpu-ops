import argparse
import sys

from junitparser import JUnitXml, Error, Failure, Skipped
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('junitxml', nargs='+')
parser.add_argument('--stats', action='store_true', help='Print results summary table')
parser.add_argument('--errors', action='store_true', help='Print errorred tests in detailed results table')
parser.add_argument('--failed', action='store_true', help='Print failed tests in detailed results table')
parser.add_argument('--passed', action='store_true', help='Print passed tests in detailed results table')
parser.add_argument('--skipped', action='store_true', help='Print skipped tests in detailed results table')
args = parser.parse_args()

if not any([args.stats, args.errors, args.failed, args.passed, args.skipped]):
    args.stats = args.passed = args.failed = args.skipped = True

xmls = [ JUnitXml.fromfile(f) for f in args.junitxml ]

need_group = len(args.junitxml) > 1
need_suite = any(len(x) > 1 for x in xmls)

def get_test_group(xml_name):
    return Path(xml_name).stem

def get_rn(xml_name, suite_name):
    rn = {}
    if need_group:
        rn |= { "Test group": get_test_group(args.junitxml[idx]) }
    if need_suite:
        rn |= { "Test suite": suite.name  }
    return rn

def print_md_row(row, print_header):
    if print_header:
        header = " | ".join([f"{key}" for key, _ in row.items()])
        print(f"| {header} |")
        header = " | ".join(["-"*len(key) for key, _ in row.items()])
        print(f"| {header} |")
    row = " | ".join([f"{value}" for _, value in row.items()])
    print(f"| {row} |")

if args.stats:
    print_header = True
    for idx, xml in enumerate(xmls):
        for suite in xml:
            passed = suite.tests - suite.errors - suite.failures - suite.skipped
            rn = get_rn(args.junitxml[idx], suite.name)
            rn |= {
                "Errors": suite.errors,
                "Failed": suite.failures,
                "Passed": passed,
                "Skipped": suite.skipped,
                "Tests": suite.tests,
                "Time(s)": suite.time
            }
            print_md_row(rn, print_header)
            print_header = False

if any([args.passed, args.failed, args.skipped]):
    if args.stats:
        print("")
    print_header = True
    for idx, xml in enumerate(xmls):
        for suite in xml:
            rn01 = get_rn(args.junitxml[idx], suite.name)
            for case in suite:
                output, result, message = (args.passed, "passed", "")
                if case.result:
                    message = f"{case.result[0].message.splitlines()[0]}"
                    if isinstance(case.result[0], Error):
                        output, result = (args.errors, "error")
                    elif isinstance(case.result[0], Skipped):
                        output, result = (args.skipped, "skipped")
                    elif isinstance(case.result[0], Failure):
                        output, result = (args.failed, "failed")
                        output = args.failed
                    else:
                        print("fatal: unknown result type", file=sys.stderr)
                        sys.exit(1)
                if output:
                    rn = rn01
                    rn |= {
                        # Be warned on pytest-pspec: if installed, pytest will
                        # use descriptions of classes and methods as test node
                        # ids. These descriptions might contain spaces and line
                        # breaks not suitable for simple Markdown tables.
                        "Class name": ' '.join(case.classname.split()),
                        "Test name": ' '.join(case.name.split()),
                        "Status": result,
                        "Time(s)": case.time,
                        "Message": message
                    }
                    print_md_row(rn, print_header)
                    print_header = False
