---
name: unitrace-parse-hw-metrics
description: Parse unitrace hardware-counter profiling output (CSV metrics). Use this skill when the user wants to parse, extract, or analyze unitrace metric logs, hardware counter CSV output, ComputeBasic metrics, VectorEngineStalls metrics, or any unitrace -g / --metrics output. Also trigger when the user mentions parsing profiling CSV where kernel names contain commas, or needs to robustly split unitrace metric rows. Handles the comma-in-kernel-name ambiguity by parsing columns from right-to-left.
---

# Unitrace Hardware-Counter Metrics Parsing

This skill provides a robust strategy for parsing the CSV-formatted hardware-counter output produced by `unitrace -g <MetricGroup>` or `unitrace --metrics <MetricGroup>`.

## The Problem

Unitrace hardware metric output is comma-separated. The first column is the kernel name, which may contain commas (e.g., kernel names with template parameters, dimension lists, or SYCL-generated names). Naive left-to-right comma splitting misaligns all columns when the kernel name contains commas.

Example problematic kernel name:
```
"my_kernel<float, 3>[SIMD16 {4096, 1, 1} {16, 1, 1}]"
```

The header uses `,` as delimiter and the data rows use `,` as well, so a naive `split(",")` on the data row produces more fields than the header expects.

## The Solution: Right-to-Left Parsing

Since **all columns except the first (Kernel)** are numeric or fixed-format values, we parse from the **last column backward**:

1. Count the number of columns `N` from the header line.
2. For each data row, split by `,` to get `M` fields (where `M >= N` when kernel name has commas).
3. The last `N - 1` fields (from the right) correspond to columns 2..N of the header.
4. Everything remaining on the left (fields 0..M-N) is joined back with `,` to reconstruct the kernel name.

This works because:
- Numeric metric values never contain commas.
- Only the kernel name (first column) can contain commas.
- The number of metric columns is fixed and known from the header.

## Output Format

Unitrace hardware metric output has this structure:

```
<application stdout/stderr>

=== Device #<id> Metrics ===

<header line: comma-separated column titles>
<data row 1>
<data row 2>
...
```

### Header Line

The header is a single line with column names separated by commas. Common columns include:

```
Kernel,GlobalInstanceId,SubDeviceId,GpuTime[ns],GpuCoreClocks[cycles],...
```

The first column is always `Kernel`. The remaining columns are metric names with units in brackets.

### Data Rows

Each row has the kernel name (possibly quoted with `"`) followed by numeric values:

```
"coal_read_s1[SIMD16 {4096; 1; 1} {16; 1; 1}]",1,0,7708,21852,2834,...
```

## Pre-check: Verify Output Format (MANDATORY)

Before collecting or parsing any unitrace hardware metrics, you MUST first run
the format validation test to confirm the current unitrace version produces
output compatible with the parser. If the format has changed (columns renamed,
added, or removed), the parser may silently produce incorrect results.

The test validates headers for **all 23 metric groups** enumerated by
`unitrace --metric-list`:

| Group | Columns | Collection Flag |
|-------|---------|-----------------|
| RenderBasic | 95 | `-g RenderBasic` |
| ComputeBasic | 86 | `-g ComputeBasic` |
| DepthProfile | 52 | `-g DepthProfile` |
| DeviceCacheProfile | 61 | `-g DeviceCacheProfile` |
| MemoryProfile | 86 | `-g MemoryProfile` |
| RenderPipeProfile | 71 | `-g RenderPipeProfile` |
| RTProfile | 62 | `-g RTProfile` |
| VectorEngineProfile | 85 | `-g VectorEngineProfile` |
| VectorEngineStalls | 57 | `-g VectorEngineStalls` |
| XvePipelineRasterizationProfile | 58 | `-g XvePipelineRasterizationProfile` |
| XvePipelineRaytracingProfile | 58 | `-g XvePipelineRaytracingProfile` |
| TestOa | 58 | `-g TestOa` |
| InternalSet1–9 | 49–95 | `-g InternalSet<N>` |
| RenderPipeCtrl | 27 | `-g RenderPipeCtrl` |
| EuStallSampling | 12 | `--stall-sampling` |

**Note**: EuStallSampling uses `--stall-sampling` (not `-g EuStallSampling`). Its output includes an `IP[Address]` column and reports per-instruction-pointer stall events rather than per-kernel aggregate metrics. The same right-to-left parser handles it correctly.

```bash
python tests/test_parse_hw_metrics.py
```

**If all tests PASS**: Proceed with metric collection and parsing.

**If any test FAILS**: The unitrace output format has changed. Do NOT use the
parser on new data until the issue is resolved:

1. Collect fresh reference logs for all groups using the test SYCL program:
   ```bash
   # Build the test program
   icpx -fsycl -O2 -o /tmp/metric_collect tests/metric_collect.cpp

   # Collect logs for every metric group (except EuStallSampling)
   for group in $(unitrace --metric-list 2>&1 | grep "^Group" | sed 's/Group [0-9]*: \([^ ]*\).*/\1/' | grep -v EuStallSampling); do
       unitrace -q -g "$group" /tmp/metric_collect \
           > tests/reference_input/${group}.log 2>&1
   done

   # Collect EuStallSampling separately (uses --stall-sampling flag)
   unitrace --stall-sampling /tmp/metric_collect \
       > tests/reference_input/EuStallSampling.log 2>&1
   ```

2. Regenerate the reference headers:
   ```bash
   python tests/test_parse_hw_metrics.py --regenerate
   ```

3. Review what changed (new/removed/renamed columns) and update
   `parse_hw_metrics.py` if the parsing logic needs adjustment.

4. Re-run the test to confirm:
   ```bash
   python tests/test_parse_hw_metrics.py
   ```

This step ensures you never get silently wrong parsed results from a format
change you didn't notice.

## Parsing Algorithm (Reference)

### Step 1: Locate the Metrics Section

Find the line matching `=== Device #<N> Metrics ===`. The header is the next non-empty line after it.

### Step 2: Parse the Header

Split the header by `,` and strip whitespace from each field to get column names. Record the total count `N`.

```python
header_fields = [h.strip() for h in header_line.split(",")]
num_columns = len(header_fields)
```

Note: Stripping is necessary because `--stall-sampling` (EuStallSampling) produces space-padded headers.

### Step 3: Parse Each Data Row (Right-to-Left)

For each non-empty data line after the header:

```python
fields = row.split(",")
# Metric values are the last (num_columns - 1) fields
metric_values = fields[-(num_columns - 1):]
# Kernel name is everything before that, rejoined
kernel_name = ",".join(fields[:len(fields) - (num_columns - 1)])
# Strip surrounding quotes from kernel name
kernel_name = kernel_name.strip().strip('"')
```

### Step 4: Build Structured Data

Combine header names with parsed values into a dictionary or DataFrame per row.

## Usage

Use the helper script provided with this skill:

```bash
python scripts/parse_hw_metrics.py <unitrace_log_file> [options]
```

Options:
- `--metric <name>`: Extract a specific metric column (e.g., `GPU_MEMORY_BYTE_READ[bytes]`)
- `--kernel <pattern>`: Filter rows by kernel name substring
- `--skip-first <N>`: Skip the first N instances (warmup), default 1
- `--format {csv,json,table}`: Output format, default `table`
- `--summary`: Show per-kernel statistics (median, mean, min, max)

### Examples

```bash
# Show all GPU_MEMORY_BYTE_READ values, skipping first warmup instance
python scripts/parse_hw_metrics.py coal_read_s1_ComputeBasic.log \
    --metric "GPU_MEMORY_BYTE_READ[bytes]" --skip-first 1

# Summarize XVE_STALL_SBID for a specific kernel
python scripts/parse_hw_metrics.py stalls.log \
    --metric "XVE_STALL_SBID[%]" --kernel "coal_read_s1" --summary

# Export all metrics to CSV
python scripts/parse_hw_metrics.py metrics.log --format csv > parsed.csv

# Export as JSON for further processing
python scripts/parse_hw_metrics.py metrics.log --format json > parsed.json
```

## Inline Parsing (No Script)

If you cannot use the helper script, here is the minimal parsing logic:

```python
import re

def parse_unitrace_metrics(log_text):
    """Parse unitrace hardware metric CSV from log text.

    Returns (header_names, rows) where each row is a dict.
    """
    lines = log_text.splitlines()

    # Find metrics section
    header_line = None
    data_start = None
    for i, line in enumerate(lines):
        if re.match(r"^=== Device #\d+ Metrics ===$", line.strip()):
            # Header is next non-empty line
            for j in range(i + 1, len(lines)):
                if lines[j].strip():
                    header_line = lines[j].strip()
                    data_start = j + 1
                    break
            break

    if header_line is None:
        raise ValueError("No metrics section found in log")

    header_fields = [h.strip() for h in header_line.split(",")]
    num_columns = len(header_fields)

    rows = []
    for line in lines[data_start:]:
        line = line.strip()
        if not line:
            continue

        fields = line.split(",")
        if len(fields) < num_columns:
            continue  # skip malformed lines

        # Right-to-left: last (num_columns - 1) fields are metrics
        metric_values = fields[-(num_columns - 1):]
        kernel_name = ",".join(fields[:len(fields) - (num_columns - 1)])
        kernel_name = kernel_name.strip().strip('"')

        row = {"Kernel": kernel_name}
        for col_name, value in zip(header_fields[1:], metric_values):
            row[col_name] = value.strip()
        rows.append(row)

    return header_fields, rows
```

## Edge Cases

- **Multiple devices**: The log may have multiple `=== Device #N Metrics ===` sections. Parse each separately.
- **Empty data rows**: Skip blank lines between data rows.
- **Quoted kernel names**: The kernel name is typically wrapped in `"..."`. Always strip these.
- **Warmup instances**: The first instance (GlobalInstanceId=1) often has different characteristics (cold caches, first dispatch overhead). Default to skipping it in analysis.
- **Truncated lines**: If a line has fewer fields than expected, skip it as malformed.

## Integration with Other Skills

- Use `unitrace-setup` to ensure unitrace is available before collecting metrics.
- Use `gpu-memory-bandwidth-bench` or `gpu-memory-coalescing` skills to generate workloads that produce these metric logs.
- Use `evidence-documentation` to document parsed results with full provenance.
