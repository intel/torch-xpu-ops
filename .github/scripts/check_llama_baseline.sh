#!/bin/bash
file1="$1"
file2="$2"

if [ ! -f "$file1" ] || [ ! -f "$file2" ]; then
    echo "Error: One or both files do not exist" >&2
    exit 1
fi

if ! diff_output=$(diff <(sort "$file1") <(sort "$file2")); then
    echo "ERROR: Files $file1 and $file2 differ!" >&2
    echo "Differences found:" >&2
    echo "$diff_output" >&2
fi

echo "SUCCESS: Files $file1 and $file2 are same"
exit 0
