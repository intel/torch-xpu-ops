import argparse
import sys
import os
import re
from junitparser import JUnitXml, Error, Failure, Skipped
from collections import defaultdict

parser = argparse.ArgumentParser(description='Test results analyzer')
parser.add_argument('-n', '--ut-name', type=str, default='', help='UT name')
parser.add_argument('-i', '--input-files', nargs='+', help='JUnit XML files or log files')
args = parser.parse_args()

failures = []
summaries = []
failures_by_category = defaultdict(list)
passed_cases = []
passed_by_category = defaultdict(list)
category_totals = defaultdict(lambda: {
    'Test cases': 0,
    'Passed': 0,
    'Skipped': 0,
    'Failures': 0,
    'Errors': 0
})

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
    "NotImplementedError",
]

def get_classname(case):
    return ' '.join(case.classname.split()) if hasattr(case, 'classname') else case.get('classname', '')

def get_name(case):
    if isinstance(case, dict):
        return case.get('name', '')
    return ' '.join(case.name.split())

def get_category_from_case(case):
    if isinstance(case, dict):
        return case.get('category', 'unknown')
    else:
        if hasattr(case, '_file_category'):
            return case._file_category
        return 'unknown'

def get_result(case):
    if isinstance(case, dict):
        return case.get('status', 'failed')

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
    if isinstance(case, dict):
        return case.get('error', '')

    if not case.result:
        return ""
    full_text = case.result[0].text if hasattr(case.result[0], 'text') else case.result[0].message
    if not full_text:
        return ""

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

    return " ; ".join(error_messages) if error_messages else f"{case.result[0].message.splitlines()[0]}"

def print_md_row(row, print_header=False, failure_list=None):
    if print_header:
        header = " | ".join([f"{key}" for key in row.keys()])
        print(f"| {header} |")
        header = " | ".join(["---"] * len(row))
        print(f"| {header} |")
    row_values = " | ".join([f"{value}" for value in row.values()])
    print(f"| {row_values} |")

    if failure_list is not None:
        failure_list.write(f"| {row_values} |\n")


def print_failures(failure_list=None):
    if not failures:
        return

    print("### Test Failures")
    print_header = True
    for case in failures:
        print_md_row({
            'Category': get_category_from_case(case),
            'Class name': get_classname(case),
            'Test name': get_name(case),
            'Status': get_result(case),
            'Message': get_message(case),
            'Source': case['source'] if isinstance(case, dict) else 'XML'
        }, print_header, failure_list=failure_list)
        print_header = False

def generate_failures_log():
    if not failures:
        return

    for case in failures:
        category = get_category_from_case(case)
        failures_by_category[category].append(case)

    for category, category_failures in failures_by_category.items():
        if not category_failures:
            continue

        log_filename = f"failures_{category}.log"
        with open(log_filename, "w", encoding='utf-8') as log_file:
            for case in category_failures:
                class_name = get_classname(case)
                test_name = get_name(case)
                log_file.write(f"{category},{class_name},{test_name}\n")

def parse_log_file(log_file):
    with open(log_file, encoding='utf-8') as f:
        content = f.read()

    ut_name = os.path.splitext(os.path.basename(log_file))[0]
    category = determine_category(ut_name)
    summary = {
        'Category': category,
        'UT': ut_name,
        'Test cases': 0,
        'Passed': 0,
        'Skipped': 0,
        'Failures': 0,
        'Errors': 0,
        'Source': 'Log'
    }

    # Extract test counts
    test_run_match = re.search(r"Ran (\d+) tests in [\d.]+s", content)
    if test_run_match:
        summary['Test cases'] = int(test_run_match.group(1))

    # Extract skipped case number
    skipped_match = re.search(r"skipped[ =](\d+)", content, re.IGNORECASE)
    if skipped_match:
        summary['Skipped'] = int(skipped_match.group(1))
    else:
        skipped_match = re.search(r"skipped (\d+) cases?", content, re.IGNORECASE)
        if skipped_match:
            summary['Skipped'] = int(skipped_match.group(1))

    # Extract failures
    failure_blocks = re.findall(r"(FAIL:.*?)(?:\n\n|\n=+\n|\Z)", content, re.DOTALL)
    exist_test_names = set()
    failures_number = 0

    for block in failure_blocks:
        case_match = re.match(r"FAIL: (\w+) \(__mp_main__\.(\w+)\)", block)
        if not case_match:
            continue

        test_name = case_match.group(1)
        if test_name in exist_test_names:
            continue
        exist_test_names.add(test_name)

        error_msg = []
        error_pattern = r"(" + "|".join(error_types) + r"):.*?(?=\n\S|\n\n|\n=+\n|\Z)"
        error_matches = re.finditer(error_pattern, block, re.DOTALL)
        if not error_matches and "Traceback" in block:
            error_msg.append("Unknown error (see traceback)")
        else:
            for match in error_matches:
                error_msg.append(match.group(0).strip())

        failure_case = {
            'classname': ut_name,
            'name': f"{case_match.group(2)}:{test_name}",
            'error': " ".join(error_msg),
            'status': 'failed',
            'source': 'Log',
            'category': category
        }
        failures.append(failure_case)
        failures_by_category[category].append(failure_case)
        failures_number += 1

    if failures_number > summary['Failures']:
        summary['Failures'] = failures_number
    summary['Passed'] = summary['Test cases'] - summary['Failures'] - summary['Skipped']

    # Update category totals
    category_totals[category]['Test cases'] += summary['Test cases']
    category_totals[category]['Passed'] += summary['Passed']
    category_totals[category]['Skipped'] += summary['Skipped']
    category_totals[category]['Failures'] += summary['Failures']
    category_totals[category]['Errors'] += summary['Errors']

    return summary

def determine_category(ut):
    if ut == 'op_regression':
        return 'op_regression'
    elif ut == 'op_regression_dev1':
        return 'op_regression_dev1'
    elif ut == 'op_extended':
        return 'op_extended'
    elif ut == 'op_transformers':
        return 'op_transformers'
    elif ut == 'test_xpu':
        return 'test_xpu'
    elif ut == 'xpu_profiling':
        return 'xpu_profiling'
    elif 'op_ut' in ut:
        return 'op_ut'
    else:
        return 'unknown'

def process_log_file(log_file):
    try:
        summary = parse_log_file(log_file)
        summaries.append(summary)
    except Exception as e:
        print(f"Error processing {log_file}: {e}", file=sys.stderr)

def process_xml_file(xml_file):
    try:
        xml = JUnitXml.fromfile(xml_file)
        parts = os.path.basename(xml_file).rsplit('.', 2)
        ut = parts[0]
        parts_category = os.path.basename(xml_file).split('.')[0]
        category = determine_category(parts_category)

        for suite in xml:
            suite_summary = {
                'Category': category,
                'UT': ut,
                'Test cases': suite.tests,
                'Passed': suite.tests - suite.skipped - suite.failures - suite.errors,
                'Skipped': suite.skipped,
                'Failures': suite.failures,
                'Errors': suite.errors,
                'Source': 'XML'
            }
            summaries.append(suite_summary)

            # Update category totals
            category_totals[category]['Test cases'] += suite_summary['Test cases']
            category_totals[category]['Passed'] += suite_summary['Passed']
            category_totals[category]['Skipped'] += suite_summary['Skipped']
            category_totals[category]['Failures'] += suite_summary['Failures']
            category_totals[category]['Errors'] += suite_summary['Errors']

            for case in suite:
                if get_result(case) not in ["passed", "skipped"]:
                    case._file_category = category
                    failures.append(case)
                elif get_result(case) == "passed":
                    case._file_category = category
                    passed_cases.append(case)
                    passed_by_category[category].append(case)
    except Exception as e:
        print(f"Error processing {xml_file}: {e}", file=sys.stderr)

def generate_passed_log():
    if not passed_cases:
        return

    for category, category_passed in passed_by_category.items():
        if not category_passed:
            continue

        log_filename = f"passed_{category}.log"
        with open(log_filename, "w", encoding='utf-8') as log_file:
            for case in category_passed:
                class_name = get_classname(case)
                test_name = get_name(case)
                status = get_result(case)
                log_file.write(f"{category},{class_name},{test_name}\n")

def generate_category_totals_log():
    """Generate log files with category totals"""
    for category, totals in category_totals.items():
        if totals['Test cases'] == 0:
            continue

        log_filename = f"category_{category}.log"
        with open(log_filename, "w", encoding='utf-8') as log_file:
            log_file.write(f"Category: {category}\n")
            log_file.write(f"Test cases: {totals['Test cases']}\n")
            log_file.write(f"Passed: {totals['Passed']}\n")
            log_file.write(f"Skipped: {totals['Skipped']}\n")
            log_file.write(f"Failures: {totals['Failures']}\n")
            log_file.write(f"Errors: {totals['Errors']}\n")

def print_summary():
    print("### Results Summary")
    print_header = True

    totals = {
        'Category': '**Total**',
        'UT': '',
        'Test cases': 0,
        'Passed': 0,
        'Skipped': 0,
        'Failures': 0,
        'Errors': 0,
        'Source': ''
    }

    for summary in summaries:
        print_md_row({
            'Category': summary['Category'],
            'UT': summary['UT'],
            'Test cases': summary['Test cases'],
            'Passed': summary['Passed'],
            'Skipped': summary['Skipped'],
            'Failures': summary['Failures'],
            'Errors': summary['Errors'],
            'Source': summary['Source']
        }, print_header)
        print_header = False

        totals['Test cases'] += summary['Test cases']
        totals['Passed'] += summary['Passed']
        totals['Skipped'] += summary['Skipped']
        totals['Failures'] += summary['Failures']
        totals['Errors'] += summary['Errors']

    print_md_row(totals)

def main():
    for input_file in args.input_files:
        if input_file.endswith('.log'):
            process_log_file(input_file)
        elif input_file.endswith('.xml'):
            process_xml_file(input_file)
        else:
            print(f"Skipping unknown file type: {input_file}", file=sys.stderr)
    if args.ut_name != "skipped_ut":
        with open("ut_failure_list.csv", "w") as failure_list:
            print_failures(failure_list=failure_list)

    generate_failures_log()
    generate_passed_log()
    generate_category_totals_log()
    print_summary()


if __name__ == "__main__":
    main()
