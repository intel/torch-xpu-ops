import re
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple
import json

def parse_pytest_log_file(log_file_path: str) -> Tuple[Dict[str, Dict[str, int]], Dict[str, int]]:
    """
    Parse a pytest log file and collect test results by test file.
    
    Args:
        log_file_path: Path to the pytest log file
        
    Returns:
        Tuple containing:
        - Dictionary with test file paths as keys and result counts as values
        - Dictionary with total counts across all files
    """
    # Initialize with all possible outcome types set to 0
    default_counts = {
        'passed': 0,
        'failed': 0,
        'skipped': 0,
        'xfailed': 0,
        'xpassed': 0,
        'deselected': 0,
        'error': 0,
        'total': 0
    }
    
    results = defaultdict(lambda: default_counts.copy())
    total_counts = default_counts.copy()
    current_session = "" 
    
    # Patterns to match different test outcomes with timestamps and durations
    patterns = {
        'session_start': re.compile(r'^=+ test session starts =+$'),
        'session_end': re.compile(r'^=+ (\d+) (passed|failed|skipped|xfailed|xpassed|deselected|error).*=+$'),
        #'session_end': re.compile(r'^=+ (?:P<number>\d+)\s+(passed|failed|skipped|deselected|xfailed|xpassed|error),\s+.* =+$'),
        'retry_passed': re.compile(
            r"""
            ^=+\s+
            (?:,\s+(?P<retry_passed>\d+)\s+passed)?
            .*
            =+$
            """
        ),
        'deselected': re.compile(
            r"""
            ^=+\s+
            (?:,\s+(?P<deselected>\d+)\s+deselected)?
            .*
            =+$
            """
        ),
        'test_result': re.compile(
            r'(?P<file>^.*.py::)'
            r'(?:\s*<-\s*[^\s]+)?'
            r'.* (?P<outcome>PASSED|FAILED|SKIPPED|XFAIL|XPASS|ERROR)\b'
            r'(?:\[\d+\.\d+s\])?\s*'  # Optional time
            r'(?:\[\s*\d+%\])?'  # Optional percentage
            #r'(\[\d+\.\d+s\]\s+)?(\[\s*\d+%\])?$'
        ),
        'test_session_starts': re.compile(
            r'=+ test session starts =+'
        )

    }
    
    outcome_mapping = {
        'PASSED': 'passed',
        'FAILED': 'failed',
        'SKIPPED': 'skipped',
        'XFAIL': 'xfailed',
        'XPASS': 'xpassed',
        'ERROR': 'error'
    }
    
    try:
        
        with open(log_file_path, 'r', encoding='utf-8') as f:
            current_file = ""
            for line in f:
                line = line.strip()
                line = " ".join(line.split(' ')[1:])
                #print(line)

                 # Check for session start
                if "Retrying single test..." in line:
                    current_session = "retry"

                # Check for session start
                if patterns['test_session_starts'].search(line):
                    if current_session != "retry":
                        current_session = "running"

                # Check for test results
                test_match = patterns['test_result'].search(line)
                if test_match:
                    if current_session == "running":
                        file = test_match.group('file').rsplit('::', 1)[0]
                        try:
                            outcome = outcome_mapping.get(test_match.group('outcome'), 'passed')
                            results[file][outcome] += 1
                            results[file]['total'] += 1
                            current_file = file
                            total_counts[outcome] += 1
                            total_counts['total'] += 1
                        except:
                            print("cannot extraced passed number from {}".format(line))
                    continue
                
                # Check for session summary (end)
                session_end_match = patterns['session_end'].search(line)
                if session_end_match:
                    if current_session == "running":
                        current_session = "completed"
                        deselected_match = patterns['deselected'].search(line)
                        if deselected_match:
                            try:
                                deselected = deselected_match.group('deselected')
                                # Update totals from session summary for verification
                                results[file]['deselected'] += deselected 
                                total_counts['deselected'] += deselected 
                            except:
                                print("cannot extract deselected number form log {}".format(line))
                    elif current_session == "retry":
                        current_session = "completed"

                        retry_passed_match = patterns['retry_passed'].search(line)
                        if retry_passed_match:
                            try:
                                retry_passed = int(retry_passed_match.group('passed'), 10)
                                assert(retry_passed==1)
                                results[file]['passed'] += retry_passed
                                total_counts['passed'] += retry_passed
                                results[file]['failed'] -= retry_passed
                                total_counts['failed'] -= retry_passed
                            except:
                                print("cannot extraced passed number from  retry log {}".format(line))
                    else:
                        assert(current_session != "running" and current_session != "retry")

    
    except UnicodeDecodeError:
        print("unexpected decoding error")
   
    return dict(results), total_counts

def print_results_summary(results: Dict[str, Dict[str, int]], total_counts: Dict[str, int], output_format: str = 'table'):
    """Print a formatted summary of the test results grouped by file."""
    # Ensure all expected keys exist with at least 0
    expected_keys = ['passed', 'failed', 'skipped', 'xfailed', 'xpassed', 'deselected', 'error', 'total']
    for key in expected_keys:
        total_counts[key] = total_counts.get(key, 0)
    
    if output_format == 'json':
        print(json.dumps({
            'by_file': results,
            'totals': total_counts
        }, indent=2))
    elif output_format == 'csv':
        print("TestFile,Total,Passed,Failed,Skipped,XFail,XPass,Error,Deselected")
        for file, counts in sorted(results.items()):
            print(f'"{file}",{counts.get("total", 0)},{counts.get("passed", 0)},'
                  f'{counts.get("failed", 0)},{counts.get("skipped", 0)},'
                  f'{counts.get("xfailed", 0)},{counts.get("xpassed", 0)},'
                  f'{counts.get("error", 0)},{counts.get("deselected", 0)}')
        print(f'"TOTALS",{total_counts.get("total", 0)},{total_counts.get("passed", 0)},'
              f'{total_counts.get("failed", 0)},{total_counts.get("skipped", 0)},'
              f'{total_counts.get("xfailed", 0)},{total_counts.get("xpassed", 0)},'
              f'{total_counts.get("error", 0)},{total_counts.get("deselected", 0)}')
    else:
        # Default table format
        print("\nTest Results Summary (Grouped by Test File):")
        print("-" * 120)
        header = f"{'Test File':<60} {'Total':>6} {'Passed':>6} {'Failed':>6} {'Skipped':>6} "
        header += f"{'XFail':>6} {'XPass':>6} {'Error':>6} {'Deselected':>6}"
        print(header)
        print("-" * 120)
        
        # Sort by filename
        for file in sorted(results.keys()):
            counts = results[file]
            row = f"{file:<60} {counts.get('total', 0):>6} {counts.get('passed', 0):>6} "
            row += f"{counts.get('failed', 0):>6} {counts.get('skipped', 0):>6} "
            row += f"{counts.get('xfailed', 0):>6} {counts.get('xpassed', 0):>6} "
            row += f"{counts.get('error', 0):>6} {counts.get('deselected', 0):>6}"
            print(row)
        
        print("-" * 120)
        total_row = f"{'TOTAL':<60} {total_counts.get('total', 0):>6} {total_counts.get('passed', 0):>6} "
        total_row += f"{total_counts.get('failed', 0):>6} {total_counts.get('skipped', 0):>6} "
        total_row += f"{total_counts.get('xfailed', 0):>6} {total_counts.get('xpassed', 0):>6} "
        total_row += f"{total_counts.get('error', 0):>6} {total_counts.get('deselected', 0):>6}"
        print(total_row)
        print("-" * 120)

def main():
    parser = argparse.ArgumentParser(description='Parse pytest log file and generate test results summary grouped by test file.')
    parser.add_argument('-f', '--format', choices=['table', 'json', 'csv'], 
                       default='table', help='Output format (table, json, csv)')
    parser.add_argument('-o', '--output', help='Output file path (default: print to stdout)')
    parser.add_argument('-s', '--sort', choices=['file', 'total', 'passed', 'failed'], 
                       default='file', help='Sort order for results')
    
    args = parser.parse_args()
    
    from pathlib import Path

    directory = Path('./')
    # Get all `.txt` files
    txt_files = list(directory.glob('*.txt'))

    for log_file in txt_files:
        try:
            if "_linux-jammy-xpu-2025.1-py3.9 _ test" not in str(log_file):
                continue

            print(f"Parsing pytest log file: {log_file}")
            results, total_counts = parse_pytest_log_file(log_file)
            
            # Apply sorting if requested
            if args.sort != 'file':
                reverse_sort = args.sort in ['total', 'failed']
                sorted_files = sorted(
                    results.keys(),
                    key=lambda x: results[x].get(args.sort, 0),
                    reverse=reverse_sort
                )
                sorted_results = {k: results[k] for k in sorted_files}
                results = sorted_results
            
            if args.output:
                import sys
                original_stdout = sys.stdout
                with open(args.output, 'a', encoding='utf-8') as f:
                    sys.stdout = f
                    print_results_summary(results, total_counts, args.format)
                    sys.stdout = original_stdout
                print(f"Results saved to {args.output}")
            else:
                print_results_summary(results, total_counts, args.format)
                
        except FileNotFoundError:
            print(f"Error: File not found - {log_file}")
        except Exception as e:
            print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
