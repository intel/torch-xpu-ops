import re
import csv
import argparse
from collections import defaultdict

def parse_log_sections(log_content):
    sections = []
    pattern = re.compile(r'^(datatype: torch\.float16 ; i: \d+)(.*?)(?=^datatype: |\Z)', re.MULTILINE | re.DOTALL)

    for match in pattern.finditer(log_content):
        header = match.group(1).strip()
        content = match.group(2).strip()
        if content:
            sections.append((header, content))

    return sections

def extract_non_aten_data(section_content):
    pattern = re.compile(
        r'^\s*([^\s].*?)\s+(\d+\.\d+%|\d+%)\s+(\d+\.\d+\w*s)\s+(\d+\.\d+%|\d+%)\s+(\d+\.\d+\w*s)\s+'
        r'(\d+\.\d+\w*s)\s+(\d+\.\d+\w*s)\s+(\d+\.\d+%|\d+%)\s+(\d+\.\d+\w*s)\s+'
        r'(\d+\.\d+\w*s)\s+(\d+)',
        re.MULTILINE
    )

    section_data = []
    for match in pattern.finditer(section_content):
        name = match.group(1).strip()
        if not name.startswith('aten::'):
            num_calls = int(match.group(11))
            section_data.append((name, num_calls))

    return section_data

def process_log_file(input_file):
    with open(input_file) as f:
        log_content = f.read()

    sections = parse_log_sections(log_content)
    all_data = defaultdict(dict)
    section_headers = []
    duplicate_names = defaultdict(list)

    print("\nFind the test log:")
    for i, (header, content) in enumerate(sections):
        print(f"[part {i+1}] {header}")
        section_headers.append(header)
        section_data = extract_non_aten_data(content)

        # Track duplicate names within the same section
        seen_in_section = defaultdict(int)
        for name, num_calls in section_data:
            seen_in_section[name] += 1
            if seen_in_section[name] > 1:
                duplicate_names[name].append((i, num_calls))

        for name, num_calls in section_data:
            all_data[name][i] = all_data[name].get(i, 0) + num_calls

    # Print duplicate names and their calls
    if duplicate_names:
        print("\nDuplicate names found:")
        for name, calls in duplicate_names.items():
            print(f"Name: {name}")
            for section_idx, num_calls in calls:
                print(f"  Section {section_idx+1}: {num_calls} calls")
    else:
        print("\nNo duplicate names found.")

    return all_data, section_headers

def write_to_csv(data, section_headers, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        headers = ['Operation Name'] + [f"Section {i+1} Calls" for i in range(len(section_headers))] + ['Total Calls']
        writer.writerow(headers)
        for name, calls_data in sorted(data.items(), key=lambda x: sum(x[1].values()), reverse=True):
            row = [name]
            total = 0
            for i in range(len(section_headers)):
                calls = calls_data.get(i, 0)
                row.append(str(calls))
                total += calls
            row.append(str(total))
            writer.writerow(row)

    print(f"\nGenerated result CSV file: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='obtain the calls of nono aten op')
    parser.add_argument('-i', '--input', required=True, help='input log path')
    parser.add_argument('-o', '--output', default='output.csv',
                       help='output summary file')

    args = parser.parse_args()

    try:
        print(f"\nProcessing the log file: {args.input}")
        csv_data, section_headers = process_log_file(args.input)

        if csv_data:
            write_to_csv(csv_data, section_headers, args.output)
            print("\nThe summary of none aten op:")
            for name, calls in csv_data.items():
                print(f"{name}: {calls}")
        else:
            print("Warning: No none aten op")
    except FileNotFoundError:
        print(f"Error: Input file {args.input} not found")
    except Exception as e:
        print(f"Error when processing the Input file: {str(e)}")

if __name__ == "__main__":
    main()
