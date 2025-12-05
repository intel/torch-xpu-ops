#!/usr/bin/env python3

# Copyright (c) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
License header management tool.
Adds, replaces, or verifies license headers in source files.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

import yaml


# Comment styles for different file types
COMMENT_STYLES = {
    "python": {"start": "#", "middle": "#", "end": "#", "block": False},
    "yaml": {"start": "#", "middle": "#", "end": "#", "block": False},
    "cmake": {"start": "#", "middle": "#", "end": "#", "block": False},
    "c": {"start": "/*", "middle": " *", "end": " */", "block": True},
    "cpp": {"start": "/*", "middle": " *", "end": " */", "block": True},
}

FILE_TYPE_MAP = {
    ".py": "python",
    ".pyi": "python",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".cc": "cpp",
    ".hpp": "cpp",
    ".hxx": "cpp",
    ".cu": "cpp",
    ".cuh": "cpp",
    ".cmake": "cmake",
}

# Patterns to detect existing license headers
# These patterns are more conservative - they only match comment blocks at the start
LICENSE_PATTERNS = {
    "block": re.compile(
        r"^(\s*/\*[\s\S]*?\*/\s*)",
        re.MULTILINE,
    ),
    "line_hash": re.compile(
        r"^((?:[ \t]*#(?!pragma|include|ifdef|ifndef|define|endif|else|elif|error|warning|undef|if )[^\n]*\n)+)",
        re.MULTILINE,
    ),
    "line_slash": re.compile(
        r"^((?:[ \t]*//[^\n]*\n)+)",
        re.MULTILINE,
    ),
}


def has_license_keywords(text: str) -> bool:
    """Check if text contains license-related keywords."""
    keywords = ["copyright", "license", "spdx"]
    text_lower = text.lower()
    return any(kw in text_lower for kw in keywords)


def normalize_header_text(text: str) -> str:
    """Extract and normalize the actual text content from a header, ignoring comment syntax."""
    # Remove block comment markers
    text = re.sub(r'/\*|\*/', '', text)
    # Remove line comment markers (both // and # and leading *)
    text = re.sub(r'^[ \t]*(?://|#|\*)+[ \t]?', '', text, flags=re.MULTILINE)
    # Normalize whitespace
    lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
    return '\n'.join(lines).lower()


def get_file_type(filepath: Path) -> Optional[str]:
    """Determine the file type based on extension or filename."""
    if filepath.name == "CMakeLists.txt":
        return "cmake"
    return FILE_TYPE_MAP.get(filepath.suffix.lower())


def format_header(header_text: str, file_type: str) -> str:
    """Format header text with appropriate comment style."""
    style = COMMENT_STYLES[file_type]
    lines = header_text.strip().split("\n")

    if style["block"]:
        formatted_lines = [style["start"]]
        for line in lines:
            if line.strip():
                formatted_lines.append(f"{style['middle']} {line}")
            else:
                formatted_lines.append(style["middle"])
        formatted_lines.append(style["end"])
        return "\n".join(formatted_lines) + "\n\n"
    else:
        formatted_lines = [f"{style['start']} {line}" if line.strip() else style["start"] for line in lines]
        return "\n".join(formatted_lines) + "\n\n"


def extract_existing_header(content: str, file_type: str) -> Optional[str]:
    """Extract the existing license header from content."""
    # Skip shebang for Python
    check_content = content
    if file_type == "python" and content.startswith("#!"):
        first_newline = content.find("\n")
        if first_newline != -1:
            check_content = content[first_newline + 1:].lstrip()

    # For C/C++ files, prefer block comments for license headers
    if file_type in ("c", "cpp"):
        match = LICENSE_PATTERNS["block"].match(check_content)
        if match and has_license_keywords(match.group(1)):
            return match.group(1)
        # Also check line_slash for // style headers
        match = LICENSE_PATTERNS["line_slash"].match(check_content)
        if match and has_license_keywords(match.group(1)):
            return match.group(1)
        return None

    # For hash-comment languages (Python, YAML, CMake)
    match = LICENSE_PATTERNS["line_hash"].match(check_content)
    if match and has_license_keywords(match.group(1)):
        return match.group(1)

    return None


def has_license_header(content: str, file_type: str) -> bool:
    """Check if content already has a license header."""
    return extract_existing_header(content, file_type) is not None


def has_correct_header(content: str, expected_header: str, file_type: str) -> bool:
    """Check if the file has the expected header (regardless of comment style)."""
    existing = extract_existing_header(content, file_type)
    if not existing:
        return False

    # Normalize both headers and compare the actual text content
    existing_normalized = normalize_header_text(existing)
    expected_normalized = normalize_header_text(expected_header)

    return existing_normalized == expected_normalized


def add_header(content: str, header_text: str, file_type: str) -> str:
    """Add header to content, preserving shebang if present."""
    formatted_header = format_header(header_text, file_type)

    # Preserve shebang for Python files
    if file_type == "python" and content.startswith("#!"):
        first_newline = content.find("\n")
        if first_newline != -1:
            shebang = content[: first_newline + 1]
            rest = content[first_newline + 1 :].lstrip()
            return shebang + "\n" + formatted_header + rest

    return formatted_header + content.lstrip()


def remove_existing_header(content: str, file_type: str) -> str:
    """Remove existing license header from content."""
    # Preserve shebang for Python files
    shebang = ""
    if file_type == "python" and content.startswith("#!"):
        first_newline = content.find("\n")
        if first_newline != -1:
            shebang = content[: first_newline + 1]
            content = content[first_newline + 1 :]

    # Only remove the header we actually detected
    header = extract_existing_header(content, file_type)
    if header:
        # Remove just that specific header
        content = content.replace(header, "", 1)

    return shebang + content.lstrip()


def process_file(
    filepath: Path,
    header_text: str,
    dry_run: bool = False,
    force: bool = False,
    verbose: bool = False,
) -> tuple[bool, str]:
    """
    Process a single file.
    Returns (was_modified, status_message).
    """
    file_type = get_file_type(filepath)
    if not file_type:
        return False, f"SKIP (unknown type): {filepath}"

    try:
        content = filepath.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        return False, f"ERROR (read): {filepath} - {e}"

    # Check if already has correct header
    if has_correct_header(content, header_text, file_type):
        if verbose:
            return False, f"OK (correct header): {filepath}"
        return False, ""

    # Check if has any license header
    if has_license_header(content, file_type):
        if force:
            # Remove existing and add new
            content = remove_existing_header(content, file_type)
            new_content = add_header(content, header_text, file_type)
            action = "REPLACE"
        else:
            return False, f"SKIP (has header, use --force): {filepath}"
    else:
        # No header, add one
        new_content = add_header(content, header_text, file_type)
        action = "ADD"

    if dry_run:
        return True, f"WOULD {action}: {filepath}"

    try:
        filepath.write_text(new_content, encoding="utf-8")
        return True, f"{action}: {filepath}"
    except OSError as e:
        return False, f"ERROR (write): {filepath} - {e}"


def load_single_config(config_path: Path) -> dict:
    """Load configuration from a single YAML file."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_configs(config_dir: Path, pattern: str, verbose: bool = False) -> dict:
    """
    Load and merge multiple config files matching a pattern.

    The first file with 'default_header' becomes the main config.
    All files can contribute 'exclude' and 'custom_headers' sections.
    """
    merged_config = {
        "default_header": None,
        "exclude": [],
        "custom_headers": [],
    }

    # Find all matching config files
    config_files = sorted(config_dir.glob(pattern))

    if not config_files:
        return merged_config

    if verbose:
        print(f"Loading config files matching '{pattern}' from {config_dir}:")

    for config_path in config_files:
        if verbose:
            print(f"  - {config_path.name}")

        try:
            config = load_single_config(config_path)
        except (OSError, yaml.YAMLError) as e:
            print(f"Warning: Failed to load {config_path}: {e}", file=sys.stderr)
            continue

        # Only the first file with default_header sets it
        if merged_config["default_header"] is None and config.get("default_header"):
            merged_config["default_header"] = config["default_header"]
            if verbose:
                print("    (provides default_header)")

        # Merge excludes
        if config.get("exclude"):
            merged_config["exclude"].extend(config["exclude"])
            if verbose:
                print(f"    (adds {len(config['exclude'])} exclude patterns)")

        # Merge custom headers
        if config.get("custom_headers"):
            merged_config["custom_headers"].extend(config["custom_headers"])
            if verbose:
                print(f"    (adds {len(config['custom_headers'])} custom header groups)")

    return merged_config


def collect_files(root: Path, extensions: set[str], exclude_patterns: list[str]) -> list[Path]:
    """Collect all files matching extensions, excluding patterns."""
    files = []
    exclude_set = set()

    # Resolve exclude patterns to absolute paths
    for pattern in exclude_patterns:
        # Handle absolute paths
        if pattern.startswith("/"):
            exclude_path = Path(pattern)
            if exclude_path.exists():
                exclude_set.add(exclude_path.resolve())
            continue

        if "*" in pattern:
            exclude_set.update(p.resolve() for p in root.glob(pattern))
        else:
            exclude_path = root / pattern
            if exclude_path.exists():
                exclude_set.add(exclude_path.resolve())

    # Collect files by extension
    for ext in extensions:
        for filepath in root.rglob(f"*{ext}"):
            if filepath.resolve() not in exclude_set:
                files.append(filepath)

    # Also collect CMakeLists.txt files
    for filepath in root.rglob("CMakeLists.txt"):
        if filepath.resolve() not in exclude_set:
            files.append(filepath)

    return sorted(files)


def main():
    parser = argparse.ArgumentParser(
        description="Manage license headers in source files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Process files using *.yaml configs
  %(prog)s --dry-run                # Show what would be changed
  %(prog)s --force                  # Replace existing headers
  %(prog)s --check                  # Check mode (exit 1 if changes needed)
  %(prog)s -c default.yaml          # Use single config file
  %(prog)s -p "headers_*.yaml"      # Use pattern to match multiple configs
  %(prog)s --config-dir ./configs   # Specify config directory

Config file structure:
  - Only ONE file should define 'default_header'
  - All files can define 'exclude' and 'custom_headers' sections
  - Sections from all matching files are merged
        """,
    )
    parser.add_argument(
        "-c", "--config",
        type=Path,
        default=None,
        help="Path to single config YAML file (mutually exclusive with -p)",
    )
    parser.add_argument(
        "-p", "--pattern",
        type=str,
        default="*.yaml",
        help="Glob pattern to match config files (default: *.yaml)",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="Directory containing config files (default: script directory)",
    )
    parser.add_argument(
        "-r", "--root",
        type=Path,
        default=Path.cwd(),
        help="Root directory to process (default: current directory)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without modifying files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing license headers",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check mode: exit with code 1 if any files need changes",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show all files, including those already correct",
    )

    args = parser.parse_args()

    # Determine config directory
    if args.config_dir:
        config_dir = args.config_dir.resolve()
    else:
        config_dir = Path(__file__).parent

    # Load configuration
    if args.config:
        # Single config file mode
        if not args.config.exists():
            print(f"Error: Config file not found: {args.config}", file=sys.stderr)
            sys.exit(1)
        config = load_single_config(args.config)
        # Wrap in merged format
        config = {
            "default_header": config.get("default_header"),
            "exclude": config.get("exclude", []),
            "custom_headers": config.get("custom_headers", []),
        }
        if args.verbose:
            print(f"Loaded config from: {args.config}")
    else:
        # Multi-config file mode
        config = load_configs(config_dir, args.pattern, verbose=args.verbose)

    root = args.root.resolve()

    # Validate default header
    default_header = config.get("default_header")
    if not default_header:
        print("Error: No default_header found in any config file", file=sys.stderr)
        sys.exit(1)

    # Get global excludes
    global_excludes = config.get("exclude", [])

    # Collect all supported extensions
    all_extensions = set(FILE_TYPE_MAP.keys())

    # Track files with custom headers
    custom_header_files: dict[Path, str] = {}

    # Process custom header groups
    for group in config.get("custom_headers", []):
        group_header = group.get("header", default_header)
        try:
            for file_pattern in group.get("files", []):
                # Handle absolute paths
                if file_pattern.startswith("/"):
                    filepath = Path(file_pattern)
                    if filepath.exists():
                        custom_header_files[filepath.resolve()] = group_header
                    continue

                if "*" in file_pattern:
                    for filepath in root.glob(file_pattern):
                        custom_header_files[filepath.resolve()] = group_header
                else:
                    filepath = root / file_pattern
                    if filepath.exists():
                        custom_header_files[filepath.resolve()] = group_header
        except Exception as e:
            print("Warning: Failed to process custom header group", file=sys.stderr)
            print(f"  Group header: \n{group_header}", file=sys.stderr)
            raise

    # Collect all processable files
    all_files = collect_files(root, all_extensions, global_excludes)

    modified_count = 0
    error_count = 0

    if args.check:
        args.dry_run = True

    for filepath in all_files:
        resolved = filepath.resolve()
        header = custom_header_files.get(resolved, default_header)

        modified, message = process_file(
            filepath,
            header,
            dry_run=args.dry_run,
            force=args.force,
            verbose=args.verbose,
        )

        if message:
            print(message)

        if modified:
            modified_count += 1
        if message.startswith("ERROR"):
            error_count += 1

    # Summary
    print(f"\n{'Would modify' if args.dry_run else 'Modified'}: {modified_count} files")
    if error_count:
        print(f"Errors: {error_count} files")

    if args.check and modified_count > 0:
        sys.exit(1)

    sys.exit(0 if error_count == 0 else 1)


if __name__ == "__main__":
    main()
