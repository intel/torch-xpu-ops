import argparse
import sys
import os
from junitparser import JUnitXml, Error, Failure, Skipped

parser = argparse.ArgumentParser()
parser.add_argument('junitxml', nargs='+')
args = parser.parse_args()

failures = []
suites = []

def get_classname(case):
    return ' '.join(case.classname.split())

def get_name(case):
    return ' '.join(case.name.split())

def get_result(case):
    result = "passed"
    if case.result:
        if isinstance(case.result[0], Error):
            result = "error"
        elif isinstance(case.result[0], Skipped):
            result = "skipped"
        elif isinstance(case.result[0], Failure):
            result = "failed"
    return result

def get_message(case):
    if not case.result:
        return ""
    full_text = case.result[0].text if hasattr(case.result[0], 'text') else case.result[0].message
    if not full_text:
        return ""
    error_types = [
        "RuntimeError",
        "ValueError",
        "TypeError",
        "AttributeError",
        "KeyError",
        "IndexError",
        "ImportError",
        "AssertionError",
        "Exception",
        "OSError",
        "Failed",
        "TimeoutError",
        "asyncio.TimeoutError",
        "FileNotFoundError",
        "PermissionError",
    ]

    error_messages = []
    capture_next_lines = False
    indent_level = 0

    for line in full_text.splitlines():
        stripped_line = line.strip()
        if not stripped_line:
            continue

        for error_type in error_types:
            if stripped_line.startswith(error_type + ": "):
                error_msg = stripped_line[len(error_type)+2:]
                error_messages.append(f"{error_type}: {error_msg}")
                capture_next_lines = True
                indent_level = 0
                break
            elif f"{error_type}:" in stripped_line and "Traceback" not in stripped_line:
                error_msg = stripped_line.split(f'{error_type}:')[-1].strip()
                error_messages.append(f"{error_type}: {error_msg}")
                capture_next_lines = True
                indent_level = 0
                break

    return " | ".join(error_messages) if error_messages else f"{case.result[0].message.splitlines()[0]}"

def print_md_row(row, print_header):
    if print_header:
        header = " | ".join([f"{key}" for key, _ in row.items()])
        print(f"| {header} |")
        header = " | ".join(["-"*len(key) for key, _ in row.items()])
        print(f"| {header} |")
    row = " | ".join([f"{value}" for _, value in row.items()])
    print(f"| {row} |")

def print_cases(cases):
    print_header = True
    for case in cases:
        classname = get_classname(case)
        name = get_name(case)
        result = get_result(case)
        message = get_message(case)
        row = {
            'Class name': classname,
            'Test name': name,
            'Status': result,
            'Message': message,
        }
        print_md_row(row, print_header)
        print_header = False

def print_suite(suite):
    print_header = True
    for suite in suites:
        ut = args.junitxml[0]
        del(args.junitxml[0])
        ut = os.path.basename(ut).split('.')[0]
        tests = suite.tests
        skipped = suite.skipped
        failures = suite.failures
        errors = suite.errors
        if ut == 'op_regression':
            category = 'op_regression'
        elif ut == 'op_regression_dev1':
            category = 'op_regression_dev1'
        elif ut == 'op_extended':
            category = 'op_extended'
        elif 'op_ut' in ut:
            category = 'op_ut'
        else:
            category = "unknown"
        row = {
            'Category': category,
            'UT': ut,
            'Test cases': tests,
            'Passed': tests-skipped-failures-errors,
            'Skipped': skipped,
            'Failures': failures,
            'Errors': errors,
        }
        print_md_row(row, print_header)
        print_header = False

xmls = [ JUnitXml.fromfile(f) for f in args.junitxml ]
for idx, xml in enumerate(xmls):
    for suite in xml:
        suites.append(suite)
        for case in suite:
            classname = get_classname(case)
            name = get_name(case)
            result = get_result(case)
            if result not in ["passed", "skipped"]:
                failures.append(case)

printed = False
def print_break(needed):
    if needed:
        print("")

if failures:
    print_break(printed)
    print("### Failures")
    print_cases(failures)
    printed = True

print("### Results Summary")
print_suite(suites)

sys.exit(0)
