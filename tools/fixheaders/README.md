# License Header Management Tool

A comprehensive tool for managing license headers across the torch-xpu-ops repository. This tool automates the process of adding, verifying, and replacing license headers in source files.

## Features

- **Automatic header detection**: Identifies existing license headers in source files
- **Multi-format support**: Handles Python, C/C++, CUDA, YAML, and CMake files with appropriate comment styles
- **Configurable headers**: YAML-based configuration for default and custom headers per file
- **Multiple operations**: Check, add, replace, or force-update headers
- **Dry-run mode**: Preview changes before applying them
- **Glob pattern support**: Flexible file matching for custom header assignments

## Installation

No additional dependencies required beyond Python 3.6+ standard library.

## Usage

### Basic Commands

```bash
# Check all files for correct headers (no modifications)
python tools/fixheaders/fixheaders.py check .

# Add headers only to files missing them
python tools/fixheaders/fixheaders.py add .

# Replace incorrect headers with correct ones
python tools/fixheaders/fixheaders.py replace .

# Force replace all headers regardless of current state
python tools/fixheaders/fixheaders.py force .
```

### Options

| Option | Description |
|--------|-------------|
| `--config`, `-c` | Path to YAML configuration file (default: `default.yaml` in script directory) |
| `--dry-run`, `-n` | Show what would be done without making changes |
| `--verbose`, `-v` | Enable verbose output |

### Examples

```bash
# Dry-run check with verbose output
python tools/fixheaders/fixheaders.py check . --dry-run --verbose

# Use custom config for CUDA-derived files
python tools/fixheaders/fixheaders.py replace . --config tools/fixheaders/cuda.yaml

# Add headers to specific directory
python tools/fixheaders/fixheaders.py add src/ATen/native/xpu/
```

## Configuration

Configuration files use YAML format with the following structure:

```yaml
# Default header applied to all files unless specified otherwise
default_header: |
  Copyright (c) 2025 Intel Corporation
  SPDX-License-Identifier: Apache-2.0

# Files/patterns to exclude from processing entirely
exclude:
  - ".git/*"
  - "__pycache__/*"
  - "**/__init__.py"

# Custom headers for specific files
custom_headers:
  - header: |
      Copyright (c) 2025 Intel Corporation
      SPDX-License-Identifier: Apache-2.0

      Portions derived from Third Party Project
      Original Copyright Notice Here
    files:
      - "path/to/specific/file.cpp"
      - "src/some/pattern/*.h"
```

### Configuration Files

| File | Purpose |
|------|---------|
| `default.yaml` | Default Intel Apache-2.0 header and common exclusions |
| `*.yaml` | Headers for files derived from third-party sources |

## Supported File Types

| Extension | Comment Style |
|-----------|---------------|
| `.py`, `.pyi` | `# comment` |
| `.yml`, `.yaml` | `# comment` |
| `.cmake` | `# comment` |
| `.c`, `.h` | `/* block comment */` |
| `.cpp`, `.hpp`, `.cxx`, `.hxx`, `.cc` | `/* block comment */` |
| `.cu`, `.cuh` | `/* block comment */` |

## Pre-Release License Compliance Process

Before each release, follow this process to ensure all files have correct license headers:

### Step 1: Run Protex Scan

Launch a Protex (Black Duck) scan on the repository to identify all third-party code.

### Step 2: Identify Matches

Review the Protex scan results to identify:

- Files containing third-party code
- The original project/source of the code
- The applicable license for each match

### Step 3: Create/Update YAML Configuration

For each third-party project identified:

1. **Create a separate YAML file** for each project (if not already existing):
   ```
   tools/fixheaders/<project_name>.yaml
   ```

2. **Copy the exact copyright header** from the original project. The header must be reproduced exactly as it appears in the source:

   ```yaml
   # filepath: tools/fixheaders/<project_name>.yaml
   # Copyright (c) 2025 Intel Corporation
   # SPDX-License-Identifier: Apache-2.0

   custom_headers:
     - header: |
         Copyright (c) 2025 Intel Corporation
         SPDX-License-Identifier: Apache-2.0

         <PASTE EXACT ORIGINAL COPYRIGHT HEADER HERE>
         <PRESERVE ALL ORIGINAL TEXT, FORMATTING, AND NOTICES>
       files:
         - "path/to/derived/file1.cpp"
         - "path/to/derived/file2.h"
   ```

3. **Important**: Always include the Intel copyright first, followed by the original project's copyright/license text.

### Step 4: Apply Headers

Run the tool to apply correct headers:

```bash
# First, do a dry-run to verify changes
python tools/fixheaders/fixheaders.py replace . --config tools/fixheaders/<project_name>.yaml --dry-run

# If everything looks correct, apply the changes
python tools/fixheaders/fixheaders.py replace . --config tools/fixheaders/<project_name>.yaml
```

### Step 5: Verify

Run a final check to ensure all files have correct headers:

```bash
python tools/fixheaders/fixheaders.py check .
```

## Example: Adding PyTorch-Derived Code

For files derived from PyTorch:

```yaml
# filepath: tools/fixheaders/pytorch.yaml
custom_headers:
  - header: |
      Copyright (c) 2025 Intel Corporation
      SPDX-License-Identifier: Apache-2.0

      Portions of this file are derived from PyTorch
      Copyright (c) Meta Platforms, Inc. and affiliates.
      SPDX-License-Identifier: BSD-3-Clause
    files:
      - "test/xpu/test_modules_xpu.py"
      - "src/derived_from_pytorch/*.cpp"
```

## Troubleshooting

### File shows as needing replacement but content looks correct

The tool normalizes headers for comparison, ignoring whitespace differences. However, if you still see false positives:

1. Check for trailing whitespace in your YAML file
2. Ensure the header text in YAML exactly matches what should be in the file
3. Use `--verbose` flag to see detailed comparison information

### File type not recognized

Add the extension to `FILE_TYPE_MAP` in `fixheaders.py`:

```python
FILE_TYPE_MAP = {
    # ... existing entries ...
    ".new_ext": "cpp",  # Use appropriate comment style
}
```

### Excluding files from processing

Add patterns to the `exclude` section in your YAML config:

```yaml
exclude:
  - "vendor/*"           # Exclude vendor directory
  - "**/*.generated.py"  # Exclude generated files
  - "specific/file.cpp"  # Exclude specific file
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success (all files OK or changes applied) |
| 1 | Check mode: some files have incorrect/missing headers |

## Contributing

When adding support for new file types or features:

1. Update `COMMENT_STYLES` for new comment syntax
2. Update `FILE_TYPE_MAP` for new extensions
3. Add tests for the new functionality
4. Update this README with new information