# Copyright 2020-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["oncall: profiler"]

import json
from pathlib import Path

from torch.testing._internal.common_utils import run_tests, TestCase


EXCLUDE_MEMCPY = True
OVERLAP_TOLERANCE = 1  # in microseconds


def load_trace(filepath):
    """Load trace JSON file"""
    with open(filepath) as f:
        return json.load(f)


def extract_kernels(trace_data):
    """Extract kernel events with timing info"""
    
    kernels = []

    # GPU-related categories to include
    gpu_categories = {"fwdbwd", "ac2g", "kernel", "gpu", "cuda", "hip"}

    for event in trace_data.get("traceEvents", []):
        # Filter events that have timestamp and duration
        if "ts" in event and "dur" in event:
            cat = event.get("cat", "").lower()

            # Skip gpu_memcpy if requested
            if EXCLUDE_MEMCPY and cat == 'gpu_memcpy':
                continue

            # Only include GPU-related events
            if any(gpu_cat in cat for gpu_cat in gpu_categories):
                kernels.append(
                    {
                        "name": event.get("name", "unknown"),
                        "cat": event.get("cat", "unknown"),
                        "ts": event["ts"],
                        "dur": event["dur"],
                        "end": event["ts"] + event["dur"],
                    }
                )

    return kernels


def check_overlaps(kernels):
    """Check if any kernels overlap in time"""
    overlaps = []

    # Sort by start time
    sorted_kernels = sorted(kernels, key=lambda k: k["ts"])

    for i in range(len(sorted_kernels)):
        for j in range(i + 1, len(sorted_kernels)):
            k1 = sorted_kernels[i]
            k2 = sorted_kernels[j]

            # If k2 starts after k1 ends, no more overlaps possible for k1
            if k2["ts"] >= k1["end"]:
                break

            # Check if they overlap
            if k1["ts"] < k2["end"] and k2["ts"] < k1["end"]:
                overlap_start = max(k1["ts"], k2["ts"])
                overlap_end = min(k1["end"], k2["end"])
                overlap_duration = overlap_end - overlap_start


                if overlap_duration > OVERLAP_TOLERANCE:
                    overlaps.append(
                        {
                            "kernel1": k1,
                            "kernel2": k2,
                            "overlap_start": overlap_start,
                            "overlap_end": overlap_end,
                            "overlap_duration": overlap_duration,
                        }
                    )

    if overlaps:
        print(f"Found {len(overlaps)} overlapping kernel pairs!")

        # Sort by overlap duration (longest first)
        sorted_overlaps = sorted(
            overlaps, key=lambda x: x["overlap_duration"], reverse=True
        )

        print("Top 10 longest overlaps:\n")
        for idx, overlap in enumerate(sorted_overlaps[:10], 1):
            k1 = overlap["kernel1"]
            k2 = overlap["kernel2"]
            print(f"Overlap #{idx}:")
            print(
                f"  Kernel 1: {k1['name']} ({k1['cat']}) - [{k1['ts']} -> {k1['end']}]"
            )
            print(
                f"  Kernel 2: {k2['name']} ({k2['cat']}) - [{k2['ts']} -> {k2['end']}]"
            )
            print(f"  Overlap duration: {overlap['overlap_duration']} units\n")

        if len(overlaps) > 10:
            print(f"... and {len(overlaps) - 10} more overlaps")

    return overlaps


class TestOverlappingKernels(TestCase):
    def test_overlapping_kernels(self):
        filepath = Path("rn50.log.json")

        assert (
            filepath.exists()
        ), f"File {filepath} does not exist, cannot find tracefile."

        trace_data = load_trace(filepath)
        kernels = extract_kernels(trace_data)
        overlaps = check_overlaps(kernels)

        assert len(overlaps) == 0, f"Found {len(overlaps)} overlapping kernel pairs!"


if __name__ == "__main__":
    run_tests()
