# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
import inspect
import subprocess
import sys

import torch
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


def _invalid_target_case(
    device, target_dtype, layout, phase, label, input_length, position
):
    import sys

    import torch

    if sys.platform != "win32":
        import resource

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    torch.manual_seed(0)
    log_probs = (
        torch.randn(input_length, 2, 4, device=device).log_softmax(2).requires_grad_()
    )
    targets = torch.tensor([[1, 2], [2, 1]], device=device, dtype=target_dtype)
    if layout == "concatenated":
        targets = targets.flatten()
    elif layout == "noncontiguous":
        targets = targets.t().contiguous().t()
    lengths = [input_length, input_length]

    def set_invalid_label():
        # Keep the bad class index within the neighboring batch's storage.
        batch = 1 if label < 0 else 0
        if targets.dim() == 1:
            targets.data[batch * 2 + position] = label
        else:
            targets.data[batch, position] = label

    if phase == "forward":
        set_invalid_label()
    if target_dtype == torch.int32 or phase == "direct_backward":
        print("CTC_FORWARD", flush=True)
        nll, log_alpha = torch._ctc_loss(log_probs, targets, lengths, [2, 2], 0, False)
        loss = nll.sum()
    else:
        print("CTC_FORWARD", flush=True)
        loss = torch.nn.functional.ctc_loss(
            log_probs, targets, lengths, [2, 2], reduction="sum"
        )
    torch.xpu.synchronize()

    if phase != "forward":
        set_invalid_label()
        print("CTC_BACKWARD", flush=True)
        if phase == "direct_backward":
            torch.ops.aten._ctc_loss_backward(
                torch.ones_like(nll),
                log_probs,
                targets,
                lengths,
                [2, 2],
                nll,
                log_alpha,
                0,
                False,
            )
        else:
            loss.backward()
        torch.xpu.synchronize()


class TestCTCLossTargetChecks(TestCase):
    def _check_invalid_target(
        self, device, dtype, layout, phase, label, length, position=0
    ):
        code = inspect.getsource(_invalid_target_case)
        code += (
            "\nimport torch\n"
            f"_invalid_target_case({device!r}, {dtype}, {layout!r}, "
            f"{phase!r}, {label}, {length}, {position})\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
        )
        stage = "CTC_FORWARD" if phase == "forward" else "CTC_BACKWARD"
        self.assertIn(stage, result.stdout, result.stderr)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("LossCTCKernels.cpp", result.stderr)
        self.assertRegex(result.stderr, r"Assertion .*>= 0.*< num_labels")

    @dtypes(torch.int32, torch.int64)
    @parametrize("layout", ["padded", "concatenated", "noncontiguous"])
    @parametrize("label", [-1, 4])
    @parametrize("position", [0, 1])
    def test_out_of_range_target(self, device, dtype, layout, label, position):
        self._check_invalid_target(device, dtype, layout, "forward", label, 8, position)

    @parametrize("length", [8, 256])
    @parametrize("label", [-1, 4])
    @parametrize("phase", ["backward", "direct_backward"])
    def test_out_of_range_target_backward(self, device, length, label, phase):
        dtype = torch.int64 if phase == "backward" else torch.int32
        layout = "padded" if phase == "backward" else "concatenated"
        self._check_invalid_target(device, dtype, layout, phase, label, length)

    @dtypes(torch.float32, torch.float64)
    @parametrize("length", [8, 256])
    @parametrize(
        "layout",
        ["padded", "concatenated", "noncontiguous", "stepped_concatenated"],
    )
    @parametrize("target_dtype", [torch.int32, torch.int64])
    def test_valid_targets(self, device, dtype, length, layout, target_dtype):
        log_probs = torch.randn(length, 2, 4, dtype=dtype).log_softmax(2)
        targets = torch.tensor([[1, -7, 99], [2, 1, 99]], dtype=target_dtype)
        actual_targets = None
        if layout == "concatenated":
            targets = torch.cat((targets[0, :1], targets[1, :2]))
        elif layout == "noncontiguous":
            targets = targets.t().contiguous().t()
        elif layout == "stepped_concatenated":
            storage = torch.tensor([1, -1, 2, -1, 1, -1], dtype=target_dtype)
            targets = storage[::2]
            actual_targets = storage.to(device)[::2]
            self.assertEqual(actual_targets.stride(), (2,))
        if actual_targets is None:
            actual_targets = targets.to(device)
        actual_input = log_probs.to(device).requires_grad_()
        expected_input = log_probs.requires_grad_()
        actual, _ = torch._ctc_loss(
            actual_input, actual_targets, [length, length], [1, 2], 0, False
        )
        expected, _ = torch._ctc_loss(
            expected_input,
            targets.contiguous(),
            [length, length],
            [1, 2],
            0,
            False,
        )
        actual.sum().backward()
        expected.sum().backward()
        self.assertEqual(actual.cpu(), expected)
        # Long float32 recurrences accumulate rounding differences across devices.
        tolerance = 1e-4 if dtype == torch.float32 else 1e-7
        self.assertEqual(
            actual_input.grad.cpu(),
            expected_input.grad,
            atol=tolerance,
            rtol=tolerance,
        )

    @dtypes(torch.int32, torch.int64)
    def test_empty_targets_ignore_padding(self, device, dtype):
        log_probs = torch.randn(8, 2, 4).log_softmax(2)
        targets = torch.tensor([[4, -1], [-1, 4]], dtype=dtype)
        actual_input = log_probs.to(device).requires_grad_()
        expected_input = log_probs.requires_grad_()
        actual, _ = torch._ctc_loss(
            actual_input, targets.to(device), [8, 8], [0, 0], 0, False
        )
        expected, _ = torch._ctc_loss(expected_input, targets, [8, 8], [0, 0], 0, False)
        actual.sum().backward()
        expected.sum().backward()
        self.assertEqual(actual.cpu(), expected)
        self.assertEqual(actual_input.grad.cpu(), expected_input.grad)


instantiate_device_type_tests(
    TestCTCLossTargetChecks, globals(), only_for="xpu", allow_xpu=True
)

if __name__ == "__main__":
    run_tests()
