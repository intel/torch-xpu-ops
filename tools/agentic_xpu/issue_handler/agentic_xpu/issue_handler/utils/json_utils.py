# Copyright 2024-2026 Intel Corporation
# Co-authored with GitHub Copilot
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""JSON extraction utilities."""
import json


def extract_json(text: str) -> str:
    """Extract the first JSON object from text.

    Uses json.JSONDecoder.raw_decode which handles braces inside
    string values correctly, unlike naive brace counting.
    Falls back to truncation repair if the JSON is cut off mid-string.
    """
    decoder = json.JSONDecoder()
    # Find the first '{' and try raw_decode from there
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                obj, end = decoder.raw_decode(text, i)
                return json.dumps(obj)
            except json.JSONDecodeError:
                continue

    # Fallback: try to repair truncated JSON by closing open strings/braces
    for i, ch in enumerate(text):
        if ch == "{":
            fragment = text[i:]
            # Try progressively closing the JSON
            for suffix in ['"}', '"]}', '"}}', '"]}}']: 
                try:
                    obj = json.loads(fragment + suffix)
                    return json.dumps(obj)
                except json.JSONDecodeError:
                    continue
            # More aggressive: find last complete key-value, truncate there
            # Look for last '", "' pattern and close after it
            last_comma = fragment.rfind('", "')
            if last_comma > 0:
                truncated = fragment[:last_comma + 1] + "}"
                try:
                    obj = json.loads(truncated)
                    return json.dumps(obj)
                except json.JSONDecodeError:
                    pass
            break

    raise ValueError("No JSON object found in text")
