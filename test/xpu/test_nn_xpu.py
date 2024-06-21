# Owner(s): ["module: intel"]


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing._internal.common_cuda import tf32_on_and_off

from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    run_tests,
)


def grid_sample_bfloat16_precision(self):
    def helper(shape_in, shape_out, align_corners):
        for mode in ("bilinear", "nearest", "bicubic"):
            if len(shape_in) != 4 and mode == "bicubic":
                continue
            data = torch.randn(shape_in, device="xpu", dtype=torch.bfloat16)
            grid = torch.rand(shape_out, device="xpu", dtype=torch.bfloat16) * 2.0 - 1.0

            out_half = F.grid_sample(
                data, grid, mode=mode, padding_mode="zeros", align_corners=align_corners
            )
            out_double = F.grid_sample(
                data.double(),
                grid.double(),
                mode=mode,
                padding_mode="zeros",
                align_corners=align_corners,
            )

            self.assertEqual(
                out_half,
                out_double.bfloat16(),
                msg=f"grid_sample with mode = {mode} doesn't match",
            )

    helper((32, 64, 16, 16), (32, 8, 8, 2), True)
    # helper((32, 64, 16, 16, 16), (32, 8, 8, 8, 3), True) # grid_sampler_3d is not supported in xpu

    helper((32, 64, 16, 16), (32, 8, 8, 2), False)
    # helper((32, 64, 16, 16, 16), (32, 8, 8, 8, 3), False) # grid_sampler_3d is not supported in xpu


def grid_sample_half_precision(self):
    def helper(shape_in, shape_out, align_corners):
        for mode in ("bilinear", "nearest", "bicubic"):
            if len(shape_in) != 4 and mode == "bicubic":
                continue
            data = torch.randn(shape_in, device="xpu", dtype=torch.half)
            grid = torch.rand(shape_out, device="xpu", dtype=torch.half) * 2.0 - 1.0

            out_half = F.grid_sample(
                data, grid, mode=mode, padding_mode="zeros", align_corners=align_corners
            )
            out_double = F.grid_sample(
                data.double(),
                grid.double(),
                mode=mode,
                padding_mode="zeros",
                align_corners=align_corners,
            )

            self.assertEqual(
                out_half,
                out_double.half(),
                msg=f"grid_sample with mode = {mode} doesn't match",
            )

    helper((32, 64, 16, 16), (32, 8, 8, 2), True)
    # helper((32, 64, 16, 16, 16), (32, 8, 8, 8, 3), True) # grid_sampler_3d is not supported in xpu

    helper((32, 64, 16, 16), (32, 8, 8, 2), False)
    # helper((32, 64, 16, 16, 16), (32, 8, 8, 8, 3), False) # grid_sampler_3d is not supported in xpu


@tf32_on_and_off(0.005)
def grid_sample_large(self, device=torch.device("xpu")):
    def issue_35202():
        input_tensor = torch.rand(
            1, 1, 480, 640, dtype=torch.float, device=device, requires_grad=True
        )
        coords = torch.tensor(
            [[-10059144, 67680944], [67680944, 67680944]],
            dtype=torch.float,
            device=device,
        )
        coords = coords.unsqueeze(0).unsqueeze(0).repeat(1, 1, 1, 1)
        result = torch.nn.functional.grid_sample(input_tensor, coords)
        self.assertEqual(
            result, torch.tensor([[[[0.0, 0.0]]]], dtype=torch.float, device=device)
        )
        result.backward(torch.ones_like(result))
        torch.xpu.synchronize()

    issue_35202()

    def issue_24823_1(dtype):
        image = torch.arange(27, 0, -1, dtype=dtype, device=device).view(1, 1, 3, 3, 3)
        image.requires_grad_()
        grid = torch.nn.functional.affine_grid(
            torch.tensor(
                [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]], dtype=dtype, device=device
            ),
            (1, 1, 3, 3, 3),
        )
        grid[:, 1, 1, 1, 0] = float("inf")
        result = torch.nn.functional.grid_sample(image, grid, padding_mode="zeros")
        tol_override = {"atol": 0.005, "rtol": 0} if dtype == torch.half else {}
        self.assertEqual(
            result,
            torch.tensor(
                [
                    [
                        [
                            [
                                [27.0, 26.0, 25.0],
                                [24.0, 23.0, 22.0],
                                [21.0, 20.0, 19.0],
                            ],
                            [[18.0, 17.0, 16.0], [15.0, 0.0, 13.0], [12.0, 11.0, 10.0]],
                            [[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]],
                        ]
                    ]
                ],
                device=device,
                dtype=dtype,
            ),
            **tol_override,
        )
        result.backward(torch.ones_like(result))
        expected_grad = torch.ones_like(image)
        expected_grad[0, 0, 1, 1, 1] = 0
        self.assertEqual(image.grad, expected_grad, atol=0.005, rtol=0)

    # grid_sampler_3d is not supported in xpu
    # issue_24823_1(torch.half)
    # issue_24823_1(torch.float)
    # issue_24823_1(torch.double)

    def issue_24823_2():
        param = torch.tensor(
            [[[-1.0e20, 0.0, 0.0], [0.0, -1.0e20, 0.0]]],
            dtype=torch.float,
            device=device,
        )
        img = torch.zeros(
            (1, 1, 4, 4), dtype=torch.float, device=device, requires_grad=True
        )
        grid = torch.nn.functional.affine_grid(param, img.size())
        result = torch.nn.functional.grid_sample(img, grid)
        self.assertEqual(
            result, torch.zeros(1, 1, 4, 4, device=device, dtype=torch.float)
        )
        result.backward(torch.ones_like(result))
        torch.xpu.synchronize()

    issue_24823_2()


try:
    from xpu_test_utils import XPUPatchForImport
except Exception as e:
    from .xpu_test_utils import XPUPatchForImport
with XPUPatchForImport(False):
    from test_nn import TestAddRelu, TestNN, TestNNDeviceType

    # Some cases named with "cuda" will pass, but they actully run on xpu in this UT. These cases are added by "add_test" in test_nn.py

    TestNNDeviceType.test_grid_sample_bfloat16_precision = (
        grid_sample_bfloat16_precision
    )
    TestNNDeviceType.test_grid_sample_half_precision = grid_sample_half_precision
    TestNNDeviceType.test_grid_sample_large = grid_sample_large


def _test_CTCLoss_lengthchecks_xpu(self):
    for target_lengths in [[30, 25, 20], [-1, -1, -1]]:
        for input_lengths in [[50, 50, 50], [-1, -1, -1]]:
            targets = torch.randint(1, 15, (3, 29), dtype=torch.long, device="xpu")
            log_probs = torch.randn(
                50, 3, 15, dtype=torch.float, device="xpu"
            ).log_softmax(2)
            with self.assertRaises(RuntimeError):
                nn.functional.ctc_loss(
                    log_probs, targets, input_lengths, target_lengths
                )


TestNN.test_CTCLoss_lengthchecks_cuda = _test_CTCLoss_lengthchecks_xpu


def _test_CTCLoss_long_targets(self):
    input_length = 4000
    vocab_size = 3
    batch_size = 4
    target_length = 1200

    log_probs = (
        torch.randn(input_length, batch_size, vocab_size, dtype=torch.double)
        .log_softmax(2)
        .requires_grad_()
    )
    targets = torch.randint(
        low=1, high=vocab_size - 1, size=(batch_size, target_length), dtype=torch.long
    )
    input_lengths = batch_size * [input_length]
    target_lengths = batch_size * [target_length]

    res_cpu = nn.functional.ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        reduction="sum",
        zero_infinity=True,
    )
    grad_out = torch.randn_like(res_cpu)
    (grad_cpu,) = torch.autograd.grad(res_cpu, log_probs, grad_out)

    res_xpu = nn.functional.ctc_loss(
        log_probs.xpu(),
        targets.xpu(),
        input_lengths,
        target_lengths,
        reduction="sum",
        zero_infinity=True,
    )
    (grad_xpu,) = torch.autograd.grad(res_xpu, log_probs, grad_out.xpu())
    self.assertEqual(res_cpu, res_xpu, atol=1e-4, rtol=0)
    self.assertEqual(grad_cpu, grad_xpu, atol=1e-4, rtol=0)


TestNN.test_CTCLoss_long_targets = _test_CTCLoss_long_targets


def _test_CTCLoss_critical_target_len(self):
    # cudnn has an unexpected problem with target length 256, see issue #53505

    N = 1
    S = 256
    C = 10
    T = 500
    target = torch.randint(low=1, high=C, size=(S,), dtype=torch.int)
    input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.int)
    target_lengths = torch.tensor(S, dtype=torch.int)
    inp = (
        torch.randn(T, N, C, dtype=torch.float, device="xpu")
        .log_softmax(2)
        .requires_grad_()
    )
    res_xpu = nn.functional.ctc_loss(
        inp, target, input_lengths, target_lengths, reduction="none"
    )
    res_cpu = nn.functional.ctc_loss(
        inp.cpu(), target, input_lengths, target_lengths, reduction="none"
    )
    self.assertEqual(res_cpu, res_xpu, atol=1e-3, rtol=0)


TestNN.test_CTCLoss_critical_target_len = _test_CTCLoss_critical_target_len


def _test_CTCLoss_zero_infinity(self):
    target_lengths = [60, 25, 20]
    input_lengths = [50, 50, 50]
    targets = torch.randint(
        1, 15, (sum(target_lengths),), dtype=torch.int, device="xpu"
    )
    log_probs = (
        torch.randn(50, 3, 15, dtype=torch.float, device="xpu")
        .log_softmax(2)
        .requires_grad_()
    )
    res = nn.functional.ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        reduction="sum",
        zero_infinity=True,
    )
    res2 = nn.functional.ctc_loss(
        log_probs,
        targets.xpu().long(),
        input_lengths,
        target_lengths,
        reduction="sum",
        zero_infinity=True,
    )
    res_cpu = nn.functional.ctc_loss(
        log_probs.cpu(),
        targets.cpu(),
        input_lengths,
        target_lengths,
        reduction="sum",
        zero_infinity=True,
    )

    self.assertEqual(res2, res, atol=1e-4, rtol=0)
    self.assertEqual(res_cpu, res.cpu(), atol=1e-4, rtol=0)
    (g1,) = torch.autograd.grad(res, log_probs)
    (g2,) = torch.autograd.grad(res2, log_probs)
    (g3,) = torch.autograd.grad(res_cpu, log_probs)
    self.assertEqual(g2, g3, atol=1e-4, rtol=0)
    self.assertEqual(g1, g2, atol=1e-4, rtol=0)
    self.assertTrue((g1 == g1).all().item())  # check that we don't have NaN


TestNN.test_CTCLoss_zero_infinity = _test_CTCLoss_zero_infinity


def _test_pack_sequence_batch_sizes_throw(self):
    with self.assertRaisesRegex(ValueError, r"batch_sizes should always be on CPU"):
        m = nn.LSTM(3, 4, bidirectional=True, num_layers=2).to("xpu")
        a = torch.rand(5, 3, device="xpu")
        b = torch.tensor([1, 1, 1, 1, 1], device="xpu")
        input = nn.utils.rnn.PackedSequence(a, b)


TestNN.test_pack_sequence_batch_sizes_throw = _test_pack_sequence_batch_sizes_throw


def _test_xpu_weight_format(self):
    rnns = [
        nn.LSTM(10, 20, batch_first=True),
        nn.LSTM(10, 20, batch_first=True, proj_size=10),
        nn.GRU(10, 20, batch_first=True),
        nn.RNN(10, 20, batch_first=True),
    ]
    first_warn = True
    for rnn in rnns:
        rnn.xpu()
        input = torch.randn(5, 4, 10, requires_grad=True, device="xpu")
        hx = torch.randn(1, 5, 20, requires_grad=True, device="xpu")
        all_vars = [input, hx] + list(rnn.parameters())
        if isinstance(rnn, nn.LSTM):
            # LSTM with projections has different hx size

            if rnn.proj_size > 0:
                hx = torch.randn(1, 5, 10, requires_grad=True, device="xpu")
                all_vars[1] = hx
            cx = torch.randn(1, 5, 20, requires_grad=True, device="xpu")
            all_vars[2:2] = [cx]
            hx = (hx, cx)
        output = rnn(input, hx)
        output[0].sum().backward()
        grads = [v.grad.data.clone() for v in all_vars]
        for v in all_vars:
            v.grad.data.zero_()
        # Weights will no longer view onto the same chunk of memory

        weight = all_vars[4]
        weight_data = weight.data.clone()
        with torch.no_grad():
            weight.set_(weight_data)
        for _ in range(2):
            with warnings.catch_warnings(record=True) as w:
                output_noncontig = rnn(input, hx)
            if first_warn:
                self.assertEqual(len(w), 1)
                self.assertIn(
                    "weights are not part of single contiguous chunk of memory",
                    w[0].message.args[0],
                )
                first_warn = False
                warnings.resetwarnings()
            output_noncontig[0].sum().backward()
            grads_noncontig = [v.grad.data.clone() for v in all_vars]
            for v in all_vars:
                v.grad.data.zero_()
            self.assertEqual(output, output_noncontig)
            self.assertEqual(grads_noncontig, grads)
        # Make sure these still share storage

        weight_data[:] = 4
        self.assertEqual(weight_data, all_vars[4].data)


TestNN.test_cudnn_weight_format = _test_xpu_weight_format


def _test_xpu_weight_tying(self):
    rnns = [
        nn.LSTM(10, 20, batch_first=True, bidirectional=True),
        nn.LSTM(10, 20, batch_first=True, bidirectional=True, proj_size=10),
        nn.GRU(10, 20, batch_first=True, bidirectional=True),
        nn.RNN(10, 20, batch_first=True, bidirectional=True),
    ]
    for rnn in rnns:
        rnn.bias_ih_l0_reverse = rnn.bias_ih_l0
        rnn.xpu()
        input = torch.randn(5, 4, 10, requires_grad=True, device="xpu")
        hx = torch.randn(2, 5, 20, requires_grad=True, device="xpu")
        all_vars = [input, hx] + list(rnn.parameters())
        opt = torch.optim.SGD(rnn.parameters(), lr=0.1)
        opt.zero_grad()
        if isinstance(rnn, nn.LSTM):
            # LSTM with projections has different hx size

            if rnn.proj_size > 0:
                hx = torch.randn(2, 5, 10, requires_grad=True, device="xpu")
                all_vars[1] = hx
            cx = torch.randn(2, 5, 20, requires_grad=True, device="xpu")
            all_vars[2:2] = [cx]
            hx = (hx, cx)
        with warnings.catch_warnings(record=True) as w:
            output = rnn(input, hx)
        output[0].sum().backward()

        opt.step()
        with warnings.catch_warnings(record=True) as w:
            output_xpu = rnn(input, hx)
        rnn.cpu()
        hx = (hx[0].cpu(), hx[1].cpu()) if isinstance(rnn, nn.LSTM) else hx.cpu()
        output_cpu = rnn(input.cpu(), hx)
        self.assertEqual(output_xpu, output_cpu)


TestNN.test_cudnn_weight_tying = _test_xpu_weight_tying


def _test_RNN_cpu_vs_xpu(self, dropout, dtype=torch.double):

    def forward_backward(
        xpu,
        rnn,
        input_val,
        grad_output,
        weights_val,
        hx_val,
        grad_hy,
        cx_val=None,
        grad_cy=None,
    ):
        is_lstm = isinstance(rnn, nn.LSTM)

        for x_layer, y_layer in zip(rnn.all_weights, weights_val):
            for x, y in zip(x_layer, y_layer):
                x.data.copy_(y.data)
        if isinstance(input_val, rnn_utils.PackedSequence):
            input = rnn_utils.PackedSequence(
                input_val.data.data.requires_grad_(True), input_val.batch_sizes
            )
            input_var = input.data
        else:
            input = input_val.clone().requires_grad_(True)
            input_var = input
        if is_lstm:
            if cx_val is None:
                hx = (
                    hx_val.clone().requires_grad_(True),
                    hx_val.add(1).requires_grad_(True),
                )
            else:
                hx = (
                    hx_val.clone().requires_grad_(True),
                    cx_val.add(1).requires_grad_(True),
                )
        else:
            hx = hx_val.clone().requires_grad_(True)
        if xpu:
            rnn.xpu()
            input_var.data = input_var.data.xpu()
            if is_lstm:
                hx[0].data = hx[0].data.xpu()
                hx[1].data = hx[1].data.xpu()
            else:
                hx.data = hx.data.xpu()
            grad_hy = grad_hy.xpu()
            if grad_cy is not None:
                grad_cy = grad_cy.xpu()
            grad_output = grad_output.xpu()
        output, hy = rnn(input, hx)

        if isinstance(output, rnn_utils.PackedSequence):
            output = output.data
        if is_lstm:
            if grad_cy is None:
                torch.autograd.backward(
                    [output, hy[0], hy[1]], [grad_output, grad_hy, grad_hy + 1]
                )
            else:
                torch.autograd.backward(
                    [output, hy[0], hy[1]], [grad_output, grad_hy, grad_cy + 1]
                )
        else:
            torch.autograd.backward([output, hy], [grad_output, grad_hy])
        return {
            "output": output.data,
            "hy": hy[0].data if is_lstm else hy.data,
            "weights": rnn.all_weights,
            "grad_input": input_var.grad.data,
            "grad_hx": hx[0].grad.data if is_lstm else hx.grad.data,
            "cy": hy[1].data if is_lstm else None,
            "grad_cx": hx[1].grad.data if is_lstm else None,
        }

    input_size = 10
    hidden_size = 6
    proj_size = 3
    num_layers = 2
    seq_length = 7
    batch = 6

    def make_noncontig(tensor):
        ndim = tensor.dim()
        return torch.stack([tensor.clone().zero_(), tensor], ndim).select(ndim, 1)

    def compare_cpu_xpu(outputs_cpu, outputs_xpu):
        self.assertEqual(list(outputs_cpu.keys()), list(outputs_xpu.keys()))
        for key in outputs_cpu.keys():
            if key != "weights":
                self.assertEqual(
                    outputs_cpu[key], outputs_xpu[key], atol=5e-5, rtol=0, msg=key
                )
        # check grad weights separately, as nested dict

        for cpu_layer_weight, xpu_layer_weight in zip(
            outputs_cpu["weights"], outputs_xpu["weights"]
        ):
            for cpu_weight, xpu_weight in zip(cpu_layer_weight, xpu_layer_weight):
                self.assertEqual(
                    cpu_weight.grad.data, xpu_weight.grad.data, atol=5e-5, rtol=0
                )

    for module in (nn.RNN, nn.LSTM, nn.GRU):
        for (
            bias,
            bidirectional,
            batch_first,
            contig,
            variable_len,
            lens_as_tensor,
        ) in product((True, False), repeat=6):

            num_directions = 2 if bidirectional else 1
            if batch_first:
                input_val = torch.randn(batch, seq_length, input_size, dtype=dtype)
                grad_output = torch.randn(
                    batch, seq_length, hidden_size * num_directions, dtype=dtype
                )
            else:
                input_val = torch.randn(seq_length, batch, input_size, dtype=dtype)
                grad_output = torch.randn(
                    seq_length, batch, hidden_size * num_directions, dtype=dtype
                )
            hx_val = torch.randn(
                num_layers * num_directions, batch, hidden_size, dtype=dtype
            )
            grad_hy = torch.randn(
                num_layers * num_directions, batch, hidden_size, dtype=dtype
            )

            if not contig:
                grad_output = make_noncontig(grad_output)
                grad_hy = make_noncontig(grad_hy)
                input_var = make_noncontig(input_val)
                hx_val = make_noncontig(hx_val)
            if variable_len:
                lengths = [7, 5, 5, 2, 1, 1]
                if lens_as_tensor:
                    lengths = torch.tensor(lengths, dtype=torch.long)
                input_val = rnn_utils.pack_padded_sequence(
                    input_val, lengths, batch_first=batch_first
                )
                grad_output = rnn_utils.pack_padded_sequence(
                    grad_output, lengths, batch_first=batch_first
                ).data
            rnn = module(
                input_size,
                hidden_size,
                num_layers,
                bias=bias,
                dropout=dropout,
                bidirectional=bidirectional,
                batch_first=batch_first,
            ).to(dtype)

            outputs_cpu = forward_backward(
                False, rnn, input_val, grad_output, rnn.all_weights, hx_val, grad_hy
            )

            rnn_xpu = module(
                input_size,
                hidden_size,
                num_layers,
                bias=bias,
                dropout=dropout,
                bidirectional=bidirectional,
                batch_first=batch_first,
            ).to(dtype)

            outputs_xpu = forward_backward(
                True, rnn_xpu, input_val, grad_output, rnn.all_weights, hx_val, grad_hy
            )

            compare_cpu_xpu(outputs_cpu, outputs_xpu)
    for nonlinearity in ("tanh", "relu"):
        hx_val = torch.randn(num_layers, batch, hidden_size, dtype=dtype)
        input_val = torch.randn(seq_length, batch, input_size, dtype=dtype)
        grad_output = torch.randn(
            seq_length, batch, hidden_size * num_directions, dtype=dtype
        )
        grad_hy = torch.randn(
            num_layers * num_directions, batch, hidden_size, dtype=dtype
        )

        rnn = nn.RNN(
            input_size, hidden_size, num_layers, bias=bias, nonlinearity=nonlinearity
        ).to(dtype)
        outputs_cpu = forward_backward(
            False, rnn, input_val, grad_output, rnn.all_weights, hx_val, grad_hy
        )

        rnn_xpu = nn.RNN(
            input_size, hidden_size, num_layers, bias=bias, nonlinearity=nonlinearity
        ).to(dtype)
        outputs_xpu = forward_backward(
            True, rnn_xpu, input_val, grad_output, rnn.all_weights, hx_val, grad_hy
        )

        compare_cpu_xpu(outputs_cpu, outputs_xpu)
    # checking LSTM with projections

    for (
        bias,
        bidirectional,
        batch_first,
        contig,
        variable_len,
        lens_as_tensor,
    ) in product((True, False), repeat=6):
        num_directions = 2 if bidirectional else 1
        if batch_first:
            input_val = torch.randn(batch, seq_length, input_size, dtype=dtype)
            grad_output = torch.randn(
                batch, seq_length, proj_size * num_directions, dtype=dtype
            )
        else:
            input_val = torch.randn(seq_length, batch, input_size, dtype=dtype)
            grad_output = torch.randn(
                seq_length, batch, proj_size * num_directions, dtype=dtype
            )
        hx_val = torch.randn(num_layers * num_directions, batch, proj_size, dtype=dtype)
        cx_val = torch.randn(
            num_layers * num_directions, batch, hidden_size, dtype=dtype
        )
        grad_hy = torch.randn(
            num_layers * num_directions, batch, proj_size, dtype=dtype
        )
        grad_cy = torch.randn(
            num_layers * num_directions, batch, hidden_size, dtype=dtype
        )

        if not contig:
            grad_output = make_noncontig(grad_output)
            grad_hy = make_noncontig(grad_hy)
            grad_cy = make_noncontig(grad_cy)
            input_var = make_noncontig(input_val)
            hx_val = make_noncontig(hx_val)
            cx_val = make_noncontig(cx_val)
        if variable_len:
            lengths = [7, 5, 5, 2, 1, 1]
            if lens_as_tensor:
                lengths = torch.tensor(lengths, dtype=torch.long)
            input_val = rnn_utils.pack_padded_sequence(
                input_val, lengths, batch_first=batch_first
            )
            grad_output = rnn_utils.pack_padded_sequence(
                grad_output, lengths, batch_first=batch_first
            ).data
        rnn = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            bias=bias,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=batch_first,
            proj_size=proj_size,
        ).to(dtype)

        outputs_cpu = forward_backward(
            False,
            rnn,
            input_val,
            grad_output,
            rnn.all_weights,
            hx_val,
            grad_hy,
            cx_val,
            grad_cy,
        )

        rnn_xpu = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            bias=bias,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=batch_first,
            proj_size=proj_size,
        ).to(dtype)

        outputs_xpu = forward_backward(
            True,
            rnn_xpu,
            input_val,
            grad_output,
            rnn.all_weights,
            hx_val,
            grad_hy,
            cx_val,
            grad_cy,
        )
        compare_cpu_xpu(outputs_cpu, outputs_xpu)


TestNN._test_RNN_cpu_vs_xpu = _test_RNN_cpu_vs_xpu


def _test_RNN_cpu_vs_xpu_no_dropout(self):
    dtype = torch.double
    self._test_RNN_cpu_vs_xpu(0, dtype)


TestNN.test_RNN_cpu_vs_cudnn_no_dropout = _test_RNN_cpu_vs_xpu_no_dropout


def _test_RNN_cpu_vs_xpu_with_dropout(self):
    # Because of dropout randomness, can only compare dropout=0 and dropout=1

    self._test_RNN_cpu_vs_xpu(1)


TestNN.test_RNN_cpu_vs_cudnn_with_dropout = _test_RNN_cpu_vs_xpu_with_dropout


def _test_RNN_xpu_weight_norm(self):
    input_size = 10
    hidden_size = 6
    num_layers = 2
    seq_length = 7
    batch = 6

    # runs on CPU to acquire expected output

    def check_weight_norm(m, name):
        input = torch.randn(seq_length, batch, input_size)
        expected_output = m(input)

        # adds weight normalization

        m = torch.nn.utils.weight_norm(m, name=name)

        # moves to CUDA

        m = m.xpu()
        input = input.xpu()

        # otherwise, subsequent warnings will be hidden, and further tests rely on them

        warnings.simplefilter("always")
        self.assertEqual(m(input), expected_output)

        # remove weight norm

        m = torch.nn.utils.remove_weight_norm(m, name=name)
        self.assertEqual(m(input), expected_output)

    check_weight_norm(nn.LSTM(input_size, hidden_size, num_layers), "weight_hh_l0")
    check_weight_norm(
        nn.LSTM(input_size, hidden_size, num_layers, proj_size=3), "weight_hr_l0"
    )


TestNN.test_RNN_cudnn_weight_norm = _test_RNN_xpu_weight_norm


def _test_partial_flat_weights(self):
    input_size = 10
    hidden_size = 6
    num_layers = 2

    m = nn.LSTM(input_size, hidden_size, num_layers)
    inp = torch.randn(3, 2, 10)
    out_expected = m(inp)
    # deletes an attribute of original LSTM

    weight_orig = m.weight_hh_l0
    del m.weight_hh_l0
    self.assertFalse(hasattr(m, "weight_hh_l0"))
    # verifies that moving to CUDA with only some attributes defined
    # does not throw an error

    m.xpu()
    # recompute the weight and make sure that module can be used

    m.weight_hh_l0 = weight_orig.xpu()
    inp = inp.xpu()
    # otherwise, subsequent warnings will be hidden, and further tests rely on them

    warnings.simplefilter("always")
    self.assertEqual(m(inp)[0].cpu(), out_expected[0])


TestNN.test_partial_flat_weights = _test_partial_flat_weights


@set_default_dtype(torch.double)
def _test_RNN_dropout(self):
    # checking the assumption that cuDNN sticks dropout in between
    # RNN layers

    for p in (0, 0.276, 0.731, 1):
        for train in (True, False):
            for xpu in (True, False):
                rnn = nn.RNN(10, 1000, 2, bias=False, dropout=p, nonlinearity="relu")
                if xpu:
                    rnn.xpu()
                if train:
                    rnn.train()
                else:
                    rnn.eval()
                rnn.weight_ih_l0.data.fill_(1)
                rnn.weight_hh_l0.data.fill_(1)
                rnn.weight_ih_l1.data.fill_(1)
                rnn.weight_hh_l1.data.fill_(1)
                input = torch.ones(1, 1, 10)
                hx = torch.zeros(2, 1, 1000)
                if xpu:
                    input = input.xpu()
                    hx = hx.xpu()
                output, hy = rnn(input, hx)
                self.assertEqual(output.data.min(), output.data.max())
                output_val = output.data[0][0][0]
                if p == 0 or not train:
                    self.assertEqual(output_val, 10000)
                elif p == 1:
                    self.assertEqual(output_val, 0)
                else:
                    self.assertGreater(output_val, 8000)
                    self.assertLess(output_val, 12000)
                    denorm_mod = (output_val * (1 - p)) % 10
                    self.assertLess(min(denorm_mod, 10 - denorm_mod), 1e-2)
                self.assertEqual(hy[0].data.min(), hy[0].data.max())
                self.assertEqual(hy[1].data.min(), hy[1].data.max())
                self.assertEqual(hy.data[0][0][0], 10)
                self.assertEqual(hy.data[1][0][0], output_val)


TestNN.test_RNN_dropout = _test_RNN_dropout


@set_default_dtype(torch.double)
def _test_error_RNN_seq_len_zero(self):
    # checking error message when RNN has seq_len = 0

    for module in (nn.RNN, nn.LSTM, nn.GRU):
        for bidirectional in [True, False]:
            input = torch.ones(0, 10, 5)
            rnn = module(5, 6, bidirectional=bidirectional)
            rnn.xpu()
            input = input.xpu()

            with self.assertRaisesRegex(
                RuntimeError, "Expected sequence length to be larger than 0 in RNN"
            ):
                rnn(input)


TestNN.test_error_RNN_seq_len_zero = _test_error_RNN_seq_len_zero


def _test_RNN_input_size_zero(self):
    for module in (nn.RNN, nn.LSTM, nn.GRU):
        input = torch.zeros((5, 0, 3))
        rnn = module(input_size=3, hidden_size=4)
        rnn.xpu()
        input = input.xpu()
        outs = rnn(input)
        self.assertEqual(outs[0].shape, torch.Size([5, 0, 4]))
        # Check that backward does not cause a hard error

        outs[0].sum().backward()


TestNN.test_RNN_input_size_zero = _test_RNN_input_size_zero


def _test_RNN_dropout_state(self):
    for p in (0, 0.1234):
        for train in (True, False):
            for xpu in (True, False):
                rnn = nn.RNN(100, 100, 2, bias=False, dropout=p, nonlinearity="relu")
                if xpu:
                    rnn.xpu()
                if train:
                    rnn.train()
                else:
                    rnn.eval()
                input = torch.rand(1, 1, 100)
                hx = torch.rand(2, 1, 100)
                if xpu:
                    input = input.xpu()
                    hx = hx.xpu()
                output1, hy1 = rnn(input, hx)
                output2, hy2 = rnn(input, hx)

                buf = io.BytesIO()
                rnn_pickle = torch.save(rnn, buf)
                buf.seek(0)
                rnn2 = torch.load(buf)
                rnn2.flatten_parameters()
                output3, hy3 = rnn2(input, hx)

                if p == 0 or not train:
                    self.assertEqual(output1, output2)
                    self.assertEqual(output1, output3)
                    self.assertEqual(hy1, hy2)
                    self.assertEqual(hy1, hy3)
                else:
                    self.assertNotEqual(output1, output2)
                    self.assertNotEqual(output1, output3)
                    self.assertNotEqual(hy1, hy2)
                    self.assertNotEqual(hy1, hy3)


TestNN.test_RNN_dropout_state = _test_RNN_dropout_state


@set_default_dtype(torch.double)
def _test_RNN_change_dropout(self):
    for train, xpu in product((True, False), repeat=2):
        rnn = nn.RNN(100, 100, 2, dropout=0, nonlinearity="relu")
        input = torch.rand(3, 2, 100)
        if xpu:
            input.data = input.data.xpu()
            rnn.xpu()
        if train:
            rnn.train()
        else:
            rnn.eval()
        prev_output = None
        for p in (0, 0.5, 0, 0.7, 0.2, 1, 0.2, 0):
            rnn.dropout = p
            output1, hy1 = rnn(input)
            output2, hy2 = rnn(input)

            if p == 0 or p == 1 or not train:
                self.assertEqual(output1, output2)
                self.assertEqual(hy1, hy2)
            else:
                self.assertNotEqual(output1, output2)
                self.assertNotEqual(hy1, hy2)
            if prev_output is not None:
                if not train:
                    self.assertEqual(output1.data, prev_output)
                    self.assertEqual(output2.data, prev_output)
                else:
                    self.assertNotEqual(output1.data, prev_output)
                    self.assertNotEqual(output2.data, prev_output)
            prev_output = output1.data


TestNN.test_RNN_change_dropout = _test_RNN_change_dropout


def _test_batchnorm_xpu_nhwc(self):
    def run_test(input, grad_output):
        c = input.size(1)
        mod = nn.BatchNorm2d(c).xpu().float()
        mod.weight.data.uniform_()
        mod.bias.data.uniform_()
        ref_input = input.detach().clone().contiguous().requires_grad_(True)
        ref_grad = grad.detach().clone().contiguous()
        ref_mod = nn.BatchNorm2d(c).xpu().float()
        ref_mod.load_state_dict(mod.state_dict())
        out = mod(input)
        out.backward(grad_output)
        ref_out = ref_mod(ref_input)
        ref_out.backward(ref_grad)
        self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))
        self.assertTrue(ref_out.is_contiguous())
        self.assertEqual(out, ref_out)
        self.assertEqual(mod.weight.grad, ref_mod.weight.grad)
        self.assertEqual(mod.bias.grad, ref_mod.bias.grad)
        self.assertEqual(input.grad, ref_input.grad)

    input = torch.randint(1, 10, (4, 8, 2, 2), dtype=torch.float32, device="xpu")
    input = (
        input.contiguous(memory_format=torch.channels_last).detach().requires_grad_()
    )

    grad = torch.randint(1, 10, (4, 8, 2, 2), dtype=torch.float32, device="xpu")
    grad = grad.contiguous(memory_format=torch.channels_last)
    run_test(input, grad)
    # see #42588, grad is channels_last contiguous, but grad.suggest_memory_format (rightly) return "contiguous"
    # not channels_last

    input = torch.randint(1, 10, (2, 8, 8, 1), dtype=torch.float32, device="xpu")
    input = (
        input.contiguous(memory_format=torch.channels_last).detach().requires_grad_()
    )
    grad = torch.randint(1, 10, (2, 8, 8, 1), dtype=torch.float32, device="xpu")
    grad = grad.permute(0, 2, 1, 3)
    run_test(input, grad)


TestNN.test_batchnorm_cudnn_nhwc = _test_batchnorm_xpu_nhwc


def _test_batchnorm_xpu_half(self):
    # THNN

    input = torch.randint(
        1, 10, (2, 3, 2, 2), dtype=torch.half, device="xpu", requires_grad=True
    )
    m = nn.BatchNorm2d(3).half().xpu()
    thnn_output = m(input)
    thnn_output.sum().backward()
    thnn_input_grad = input.grad.data.clone()
    self.assertEqualTypeString(thnn_output, input)

    input.grad = None
    m = m.float()
    xpu_output = m(input)
    xpu_output.sum().backward()
    xpu_input_grad = input.grad.data.clone()
    self.assertEqualTypeString(xpu_output, input)
    self.assertEqual(xpu_output, thnn_output)
    self.assertEqual(xpu_input_grad, thnn_input_grad, atol=1e-3, rtol=0)


TestNN.test_batchnorm_cudnn_half = _test_batchnorm_xpu_half


def _test_batchnorm_nonaffine_xpu_half_input(self):
    input = torch.randn(16, 3, 24, 24, dtype=torch.half, device="xpu")
    m = nn.BatchNorm2d(3, affine=False).xpu().float()  # keep running stats in FP32
    output = m(input)
    self.assertEqualTypeString(output, input)
    m.eval()
    output = m(input)
    self.assertEqualTypeString(output, input)


TestNN.test_batchnorm_nonaffine_cuda_half_input = (
    _test_batchnorm_nonaffine_xpu_half_input
)


def _test_batchnorm_nhwc_xpu(self):
    for dtype in (torch.half, torch.float):
        (N, C, H, W) = 2, 64, 50, 50
        model = torch.nn.BatchNorm2d(
            C, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        model = model.eval().xpu().to(dtype)
        inp1 = torch.randn(N, C, H, W, device=torch.device("xpu"), dtype=dtype)
        inp2 = inp1.contiguous(memory_format=torch.channels_last)
        out1 = model(inp1)
        out2 = model(inp2)
        self.assertTrue(torch.equal(out1, out2))


TestNN.test_batchnorm_nhwc_cuda = _test_batchnorm_nhwc_xpu


def _test_interpolate_illegal_memory_access(self):
    in_s = 45
    out_s = 14

    input = torch.ones((1, 1, in_s), device="xpu", requires_grad=True)
    # note we allocated grad_output to be larger so out of bound access
    # would be visible in grad_input

    grad = torch.ones((1, 1, out_s * 2), device="xpu", requires_grad=True)
    grad = grad[:, :, :out_s]

    input_ref = input.detach().cpu().requires_grad_()
    grad_ref = grad.cpu()

    out = F.interpolate(input, size=(out_s,), mode="nearest")
    out.backward(grad)

    out_ref = F.interpolate(input_ref, size=(out_s,), mode="nearest")
    out_ref.backward(grad_ref)

    self.assertEqual(out_ref, out)
    self.assertEqual(input_ref.grad, input.grad)


TestNN.test_interpolate_illegal_memory_access = _test_interpolate_illegal_memory_access


def _test_convert_sync_batchnorm(self):
    module = torch.nn.Sequential(
        torch.nn.BatchNorm1d(100), torch.nn.InstanceNorm1d(100)
    ).xpu()

    # necessary to have an anchor point for comparison, in case the
    # convert_sync_batchnorm updates in place

    comp_module = torch.nn.Sequential(
        torch.nn.BatchNorm1d(100), torch.nn.InstanceNorm1d(100)
    ).xpu()
    comp_module.load_state_dict(module.state_dict())

    sync_bn_module = torch.nn.SyncBatchNorm.convert_sync_batchnorm(module)
    children = list(sync_bn_module.children())
    self.assertEqual(children[0].__class__, torch.nn.SyncBatchNorm)
    self.assertEqual(children[1].__class__, torch.nn.InstanceNorm1d)

    for layer, converted_layer in zip(
        comp_module.children(), sync_bn_module.children()
    ):
        for key in layer.state_dict().keys():
            self.assertEqual(
                layer.state_dict()[key].device, converted_layer.state_dict()[key].device
            )
            self.assertEqual(layer.state_dict()[key], converted_layer.state_dict()[key])


TestNN.test_convert_sync_batchnorm = _test_convert_sync_batchnorm


def _test_sync_batchnorm_backward_elemt(self):
    device = "xpu"
    saved_input = torch.rand(2, 3, 2, 1, device=device)
    grad_output = torch.rand(2, 3, 2, 1, device=device)
    mean = torch.rand(3, device=device)
    invstd = torch.rand(3, device=device)
    weight = torch.rand(3, device=device)
    sum_dy = torch.rand(3, device=device)
    sum_dy_xmu = torch.rand(3, device=device)
    count_tensor = torch.tensor([5, 5, 5], dtype=torch.int32, device=device)

    gI_contiguous = torch.batch_norm_backward_elemt(
        grad_output, saved_input, mean, invstd, weight, sum_dy, sum_dy_xmu, count_tensor
    )

    # Test batch_norm_backward_elemt gives the same answer for all
    # combinations of contiguous as channels_last input

    for a, b in [
        (torch.channels_last, torch.contiguous_format),
        (torch.contiguous_format, torch.channels_last),
        (torch.channels_last, torch.channels_last),
    ]:
        gI_actual = torch.batch_norm_backward_elemt(
            grad_output.contiguous(memory_format=a),
            saved_input.contiguous(memory_format=b),
            mean,
            invstd,
            weight,
            sum_dy,
            sum_dy_xmu,
            count_tensor,
        )
        self.assertEqual(gI_actual, gI_contiguous)


TestNN.test_sync_batchnorm_backward_elemt = _test_sync_batchnorm_backward_elemt


def _test_sync_batchnorm_accuracy_xpu(self):
    # The target of this test is to test the functionality and accuracy of
    #   those single-GPU cuda kernels used in SyncBatchNorm
    # They are:
    #   fwd: torch.batch_norm_stats, torch.batch_norm_gather_stats_with_counts, torch.batch_norm_elemt
    #   bwd: torch.batch_norm_backward_reduce, torch.batch_norm_backward_elemt

    def _batch_norm_stats(data, memory_format, mean_axes):
        mean1, _ = torch.batch_norm_stats(data, 1e-5)
        mean2, _ = torch.batch_norm_stats(data.to(memory_format=memory_format), 1e-5)
        mean_ref = torch.mean(data, mean_axes, keepdim=False)

        self.assertEqual(mean_ref, mean1)
        self.assertEqual(mean_ref, mean2)

    _batch_norm_stats(
        torch.randn(1, 96, 112, 112, dtype=torch.float, device="xpu"),
        torch.channels_last,
        (0, 2, 3),
    )
    _batch_norm_stats(
        torch.randn(1, 96, 112, 112, 112, dtype=torch.float, device="xpu"),
        torch.channels_last_3d,
        (0, 2, 3, 4),
    )


TestNN.test_sync_batchnorm_accuracy_cuda = _test_sync_batchnorm_accuracy_xpu


def _test_CTCLoss_xpu(self, device):
    def _helper(zero_infinity):
        target_lengths = [30, 25, 20]
        input_lengths = [50, 50, 50]
        targets = torch.randint(1, 15, (sum(target_lengths),), dtype=torch.int)
        log_probs = (
            torch.randn(50, 3, 15, dtype=torch.float, device=device)
            .log_softmax(2)
            .requires_grad_()
        )

        log_probs_ref = log_probs.detach().clone().requires_grad_()

        res = torch.nn.functional.ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            zero_infinity=zero_infinity,
        )
        res.backward()

        expected = ctcloss_reference(
            log_probs, targets.xpu(), input_lengths, target_lengths
        ).float()

        res2 = torch.nn.functional.ctc_loss(
            log_probs_ref,
            targets.xpu().long(),
            input_lengths,
            target_lengths,
            zero_infinity=zero_infinity,
        )
        res2.backward()

        self.assertEqual(res, expected)
        self.assertEqual(res2, res)
        self.assertEqual(log_probs.grad, log_probs_ref.grad)

    _helper(zero_infinity=True)
    _helper(zero_infinity=False)


TestNNDeviceType.test_CTCLoss_cudnn = _test_CTCLoss_xpu


def _test_masked_softmax_devices_parity(self):
    # Test that softmax with mask type 0 (LxL attention mask), mask type 1 (BxL padding mask),
    # and mask type 2 (BxHxLxL generic mask) gives the same result on CPU and on CUDA.

    sizes = [(1, 1, 32), (3, 16, 310), (12, 4, 1024), (4, 2, 1200)]
    for B, num_heads, L in sizes:
        # mask_type == 0 => attention mask of shape LxL

        src_mask = torch.randint(0, 2, (L, L)).bool()
        # mask_type == 1 => padding mask of shape BxL

        src_key_padding_mask = torch.randint(0, 2, (B, L)).bool()
        # mask_type == 2 => generic mask of shape BxHxLxL

        generic_mask = torch.randint(0, 2, (B, num_heads, L, L)).bool()
        masks = [(src_mask, 0), (src_key_padding_mask, 1), (generic_mask, 2)]
        input = torch.randn((B, num_heads, L, L))
        for dim in [0, 3]:
            for mask, mask_type in masks:
                if (num_heads % 2) and (mask_type == 1):
                    # CUDA path doesn't support padding mask when the number of heads is odd

                    continue

                def softmax_on_device(mask, input, device):
                    # Compute softmax on a given device

                    input_device = input.to(device)
                    mask_device = mask.to(device)
                    softmax_res = torch._masked_softmax(
                        input_device, mask_device, dim, mask_type
                    )
                    if mask_type == 0:
                        mask_expanded = (
                            mask_device.reshape(1, 1, L, L)
                            .expand(B, num_heads, L, L)
                            .bool()
                        )
                    elif mask_type == 1:
                        mask_expanded = (
                            mask_device.reshape(B, 1, 1, L)
                            .expand(B, num_heads, L, L)
                            .bool()
                        )
                    else:
                        mask_expanded = mask_device
                    # In result, should only fill the entirely masked out rows since those are non-deterministic (*may* be 0)
                    # Fill rows with all True's with 0

                    mask_out = mask_expanded.all(dim, keepdim=True).expand(
                        mask_expanded.shape
                    )
                    softmax_res = softmax_res.masked_fill(mask_out, 0)
                    return softmax_res

                cpu_res = softmax_on_device(mask, input, "cpu")
                xpu_res = softmax_on_device(mask, input, "xpu")
                self.assertEqual(cpu_res, xpu_res, exact_dtype=True)


TestNNDeviceType.test_masked_softmax_devices_parity = (
    _test_masked_softmax_devices_parity
)


@dtypes(torch.float16, torch.float32)
def _test_cross_entropy_loss_2d_out_of_bounds_class_index(self, device, dtype):
    # Test for issue #117532
    # Run in a different process to prevent the device-side assert from affecting other tests

    stderr = TestCase.runWithPytorchAPIUsageStderr(
        f"""\



#!/usr/bin/env python3







import torch



import torch.nn.functional as F



from torch.testing._internal.common_utils import (run_tests, TestCase)







class TestThatContainsCUDAAssert(TestCase):



def test_cross_entropy_loss_2d_out_of_bounds_class_index(self):



    device = '{str(device)}'



    dtype = {str(dtype).strip("'")}



    ignore_index = 255



    b = 10



    n_classes = 3



    w = 768



    h = 1024



    pred = torch.randn(b, n_classes, w, h, dtype=dtype, device=device)



    labels = torch.zeros(b, w, h, dtype=torch.int64, device=device)



    labels[5, 200, 200] = ignore_index



    # Set invalid class index



    labels[5, 200, 200] = 254







    x = F.cross_entropy(



        pred, labels, reduction="none", ignore_index=ignore_index



    )



    torch.xpu.synchronize()











if __name__ == '__main__':



run_tests()



    """
    )
    self.assertIn("XPU error: device-side assert triggered", stderr)


TestNNDeviceType.test_cross_entropy_loss_2d_out_of_bounds_class_index = (
    _test_cross_entropy_loss_2d_out_of_bounds_class_index
)


def _test_ctc_loss_xpu(self, device):
    batch_size = 16
    input_length = 30
    num_labels = 101
    target_length = 15
    targets = torch.randint(
        1, num_labels, (batch_size * target_length,), device="xpu", dtype=torch.long
    )
    log_probs = torch.log_softmax(
        torch.randn(
            input_length, batch_size, num_labels, device="xpu", dtype=torch.float
        ),
        2,
    )
    log_probs.requires_grad_()

    input_lengths = batch_size * [input_length]
    target_lengths = batch_size * [target_length]
    grad_out = torch.randn(batch_size, device="xpu", dtype=torch.float)
    loss_native = torch.nn.functional.ctc_loss(
        log_probs, targets, input_lengths, target_lengths, reduction="none"
    )
    (grad_native,) = torch.autograd.grad(loss_native, log_probs, grad_out)
    loss_xpu = torch.nn.functional.ctc_loss(
        log_probs,
        targets.to("cpu", torch.int32),
        input_lengths,
        target_lengths,
        reduction="none",
    )
    self.assertTrue("xpu" in str(loss_xpu.grad_fn))
    (grad_xpu,) = torch.autograd.grad(loss_xpu, log_probs, grad_out)
    self.assertEqual(grad_xpu, grad_native, atol=1e-4, rtol=0)


TestNNDeviceType.test_ctc_loss_cudnn = _test_ctc_loss_xpu


def _test_grid_sample_large(self, device):
    def issue_35202():
        input_tensor = torch.rand(
            1, 1, 480, 640, dtype=torch.float, device=device, requires_grad=True
        )
        coords = torch.tensor(
            [[-10059144, 67680944], [67680944, 67680944]],
            dtype=torch.float,
            device=device,
        )
        coords = coords.unsqueeze(0).unsqueeze(0).repeat(1, 1, 1, 1)
        result = torch.nn.functional.grid_sample(input_tensor, coords)
        self.assertEqual(
            result, torch.tensor([[[[0.0, 0.0]]]], dtype=torch.float, device=device)
        )
        result.backward(torch.ones_like(result))
        torch.xpu.synchronize()

    issue_35202()

    def issue_24823_1(dtype):
        image = torch.arange(27, 0, -1, dtype=dtype, device=device).view(1, 1, 3, 3, 3)
        image.requires_grad_()
        grid = torch.nn.functional.affine_grid(
            torch.tensor(
                [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]], dtype=dtype, device=device
            ),
            (1, 1, 3, 3, 3),
        )
        grid[:, 1, 1, 1, 0] = float("inf")
        result = torch.nn.functional.grid_sample(image, grid, padding_mode="zeros")
        tol_override = {"atol": 0.005, "rtol": 0} if dtype == torch.half else {}
        self.assertEqual(
            result,
            torch.tensor(
                [
                    [
                        [
                            [
                                [27.0, 26.0, 25.0],
                                [24.0, 23.0, 22.0],
                                [21.0, 20.0, 19.0],
                            ],
                            [[18.0, 17.0, 16.0], [15.0, 0.0, 13.0], [12.0, 11.0, 10.0]],
                            [[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]],
                        ]
                    ]
                ],
                device=device,
                dtype=dtype,
            ),
            **tol_override,
        )
        result.backward(torch.ones_like(result))
        expected_grad = torch.ones_like(image)
        expected_grad[0, 0, 1, 1, 1] = 0
        self.assertEqual(image.grad, expected_grad, atol=0.005, rtol=0)

    issue_24823_1(torch.half)
    issue_24823_1(torch.float)
    issue_24823_1(torch.double)

    def issue_24823_2():
        param = torch.tensor(
            [[[-1.0e20, 0.0, 0.0], [0.0, -1.0e20, 0.0]]],
            dtype=torch.float,
            device=device,
        )
        img = torch.zeros(
            (1, 1, 4, 4), dtype=torch.float, device=device, requires_grad=True
        )
        grid = torch.nn.functional.affine_grid(param, img.size())
        result = torch.nn.functional.grid_sample(img, grid)
        self.assertEqual(
            result, torch.zeros(1, 1, 4, 4, device=device, dtype=torch.float)
        )
        result.backward(torch.ones_like(result))
        torch.xpu.synchronize()

    issue_24823_2()


TestNNDeviceType.test_grid_sample_large = _test_grid_sample_large


def _test_grid_sample_half_precision(self):
    def helper(shape_in, shape_out, align_corners):
        for mode in ("bilinear", "nearest", "bicubic"):
            if len(shape_in) != 4 and mode == "bicubic":
                continue
            data = torch.randn(shape_in, device="xpu", dtype=torch.half)
            grid = torch.rand(shape_out, device="xpu", dtype=torch.half) * 2.0 - 1.0

            out_half = F.grid_sample(
                data, grid, mode=mode, padding_mode="zeros", align_corners=align_corners
            )
            out_double = F.grid_sample(
                data.double(),
                grid.double(),
                mode=mode,
                padding_mode="zeros",
                align_corners=align_corners,
            )

            self.assertEqual(
                out_half,
                out_double.half(),
                msg=f"grid_sample with mode = {mode} doesn't match",
            )

    helper((32, 64, 16, 16), (32, 8, 8, 2), True)
    helper((32, 64, 16, 16, 16), (32, 8, 8, 8, 3), True)
    helper((32, 64, 16, 16), (32, 8, 8, 2), False)
    helper((32, 64, 16, 16, 16), (32, 8, 8, 8, 3), False)


TestNNDeviceType.test_grid_sample_half_precision = _test_grid_sample_half_precision


def _test_grid_sample_bfloat16_precision(self):
    def helper(shape_in, shape_out, align_corners):
        for mode in ("bilinear", "nearest", "bicubic"):
            if len(shape_in) != 4 and mode == "bicubic":
                continue
            data = torch.randn(shape_in, device="xpu", dtype=torch.bfloat16)
            grid = torch.rand(shape_out, device="xpu", dtype=torch.bfloat16) * 2.0 - 1.0

            out_half = F.grid_sample(
                data, grid, mode=mode, padding_mode="zeros", align_corners=align_corners
            )
            out_double = F.grid_sample(
                data.double(),
                grid.double(),
                mode=mode,
                padding_mode="zeros",
                align_corners=align_corners,
            )

            self.assertEqual(
                out_half,
                out_double.bfloat16(),
                msg=f"grid_sample with mode = {mode} doesn't match",
            )

    helper((32, 64, 16, 16), (32, 8, 8, 2), True)
    helper((32, 64, 16, 16, 16), (32, 8, 8, 8, 3), True)
    helper((32, 64, 16, 16), (32, 8, 8, 2), False)
    helper((32, 64, 16, 16, 16), (32, 8, 8, 8, 3), False)


TestNNDeviceType.test_grid_sample_bfloat16_precision = (
    _test_grid_sample_bfloat16_precision
)


def _test_layernorm_half_precision(self):
    width = 128
    input = torch.rand(1, 5, width, device="xpu", dtype=torch.half) * 0.1
    normalized_shape = (width,)
    weight = torch.ones(width, device="xpu", dtype=torch.half)
    bias = torch.zeros(width, device="xpu", dtype=torch.half)
    eps = 1e-5

    output_fp16 = torch.layer_norm(input, normalized_shape, weight, bias, eps)
    output_fp32 = torch.layer_norm(
        input.float(), normalized_shape, weight.float(), bias.float(), eps
    ).half()
    self.assertEqual(output_fp16, output_fp32, atol=0, rtol=0)


TestNNDeviceType.test_layernorm_half_precision = _test_layernorm_half_precision


def _test_layernorm_weight_bias(self):
    width = 128
    input = torch.rand(1, 5, width, device="xpu", dtype=torch.float32) * 0.1
    normalized_shape = (width,)
    data = torch.randn(width, device="xpu", dtype=torch.float32)
    weight = torch.ones(width, device="xpu", dtype=torch.float32)
    bias = torch.zeros(width, device="xpu", dtype=torch.float32)
    eps = 1e-5

    out_none_weight = torch.layer_norm(input, normalized_shape, None, data, eps)
    out_one_weight = torch.layer_norm(input, normalized_shape, weight, data, eps)
    self.assertEqual(out_none_weight, out_one_weight)

    out_none_bias = torch.layer_norm(input, normalized_shape, data, None, eps)
    out_zero_bias = torch.layer_norm(input, normalized_shape, data, bias, eps)
    self.assertEqual(out_none_bias, out_zero_bias)


TestNNDeviceType.test_layernorm_weight_bias = _test_layernorm_weight_bias


@dtypes(torch.double, torch.float, torch.half)
def _test_transformerencoderlayer(self, device, dtype):
    # this is a deterministic test for TransformerEncoderLayer

    d_model = 4
    nhead = 2
    dim_feedforward = 16
    dropout = 0.0
    bsz = 2

    atol = 1e-5
    rtol = 1e-7
    if "xpu" in device:
        atol = 1e-3
        rtol = 1e-2

    def _test(training, batch_first, atol, rtol):
        def perm_fn(x):
            return x.transpose(1, 0) if batch_first else x

        model = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            batch_first=batch_first,
            device=device,
            dtype=dtype,
        )

        if not training:
            assert dropout == 0
            model = model.eval()
        # set constant weights of the model

        for idx, p in enumerate(model.parameters()):
            x = p.data
            sz = x.view(-1).size(0)
            shape = x.shape
            x = torch.cos(torch.arange(0, sz).float().view(shape))
            p.data.copy_(x)
        # deterministic input

        encoder_input = torch.tensor(
            [[[20.0, 30.0, 40.0, 50.0]]], device=device, dtype=dtype
        )
        result = model(encoder_input)
        ref_output = torch.tensor(
            [[[2.258703, 0.127985, -0.697881, 0.170862]]], device=device, dtype=dtype
        )
        self.assertEqual(result.shape, ref_output.shape)
        torch.testing.assert_close(result, ref_output, atol=atol, rtol=rtol)
        # 0 values are NOT masked. This shouldn't mask anything.

        mask = torch.tensor([[0]], device=device) == 1
        # TODO: enable fast path for calls with a mask!

        result = model(encoder_input, src_key_padding_mask=mask)
        self.assertEqual(result.shape, ref_output.shape)
        torch.testing.assert_close(result, ref_output, atol=atol, rtol=rtol)
        # 1 values are masked. Since there is only 1 input embedding this
        # will result in nan.

        mask = torch.tensor([[1]], device=device) == 1
        result = model(encoder_input, src_key_padding_mask=mask)
        result = result.cpu().detach().numpy()
        self.assertTrue(np.isnan(result).all())

        # deterministic input

        encoder_input = perm_fn(
            torch.tensor(
                [[[1.0, 2.0, 3.0, 4.0]], [[5.0, 6.0, 7.0, 8.0]]],
                device=device,
                dtype=dtype,
            )
        )
        result = model(encoder_input)
        ref_output = perm_fn(
            torch.tensor(
                [
                    [[2.272644, 0.119035, -0.691669, 0.153486]],
                    [[2.272644, 0.119035, -0.691669, 0.153486]],
                ],
                device=device,
                dtype=dtype,
            )
        )
        self.assertEqual(result.shape, ref_output.shape)
        torch.testing.assert_close(result, ref_output, atol=atol, rtol=rtol)
        # all 0 which is no masking

        mask = torch.tensor([[0, 0]], device=device) == 1
        result = model(encoder_input, src_key_padding_mask=mask)
        self.assertEqual(result.shape, ref_output.shape)
        torch.testing.assert_close(result, ref_output, atol=atol, rtol=rtol)
        mask = torch.tensor([[1, 0]], device=device) == 1
        result = model(encoder_input, src_key_padding_mask=mask)
        ref_output = perm_fn(
            torch.tensor(
                [
                    [[2.301516, 0.092249, -0.679101, 0.103088]],
                    [[2.301516, 0.092249, -0.679101, 0.103088]],
                ],
                device=device,
                dtype=dtype,
            )
        )
        self.assertEqual(result.shape, ref_output.shape)
        torch.testing.assert_close(result, ref_output, atol=atol, rtol=rtol)

        # deterministic input

        encoder_input = perm_fn(
            torch.tensor(
                [
                    [
                        [0.7462, 0.6653, 0.5679, 0.4891],
                        [0.5387, 0.1655, 0.3565, 0.0471],
                    ],
                    [
                        [0.8335, 0.2799, 0.5031, 0.2947],
                        [0.1402, 0.0318, 0.7636, 0.1346],
                    ],
                    [
                        [0.6333, 0.9344, 0.1376, 0.9938],
                        [0.8924, 0.2872, 0.6692, 0.2944],
                    ],
                    [
                        [0.9897, 0.6915, 0.3154, 0.1733],
                        [0.8645, 0.3513, 0.3064, 0.0767],
                    ],
                    [
                        [0.8117, 0.2366, 0.4838, 0.7881],
                        [0.3718, 0.4945, 0.9511, 0.0864],
                    ],
                ],
                device=device,
                dtype=dtype,
            )
        )
        result = model(encoder_input)
        ref_output = perm_fn(
            torch.tensor(
                [
                    [
                        [2.428589, 0.020835, -0.602055, -0.085249],
                        [2.427987, 0.021213, -0.602496, -0.084103],
                    ],
                    [
                        [2.424689, 0.019155, -0.604793, -0.085672],
                        [2.413863, 0.022211, -0.612486, -0.072490],
                    ],
                    [
                        [2.433774, 0.021598, -0.598343, -0.087548],
                        [2.425104, 0.019748, -0.604515, -0.084839],
                    ],
                    [
                        [2.436185, 0.022682, -0.596625, -0.087261],
                        [2.433556, 0.021891, -0.598509, -0.086832],
                    ],
                    [
                        [2.416246, 0.017512, -0.610712, -0.082961],
                        [2.422901, 0.024187, -0.606178, -0.074929],
                    ],
                ],
                device=device,
                dtype=dtype,
            )
        )
        self.assertEqual(result.shape, ref_output.shape)
        torch.testing.assert_close(result, ref_output, atol=atol, rtol=rtol)

        # all 0

        mask = torch.zeros([2, 5], device=device) == 1
        result = model(encoder_input, src_key_padding_mask=mask)
        self.assertEqual(result.shape, ref_output.shape)
        torch.testing.assert_close(result, ref_output, atol=atol, rtol=rtol)
        mask[0, 1] = 1
        mask[1, 3] = 1
        mask[1, 4] = 1
        result = model(encoder_input, src_key_padding_mask=mask)
        ref_output = perm_fn(
            torch.tensor(
                [
                    [
                        [2.429026, 0.020793, -0.601741, -0.085642],
                        [2.428811, 0.021445, -0.601912, -0.084252],
                    ],
                    [
                        [2.425009, 0.019155, -0.604566, -0.085899],
                        [2.415408, 0.02249, -0.611415, -0.073],
                    ],
                    [
                        [2.434199, 0.021682, -0.598039, -0.087699],
                        [2.42598, 0.019941, -0.603896, -0.085091],
                    ],
                    [
                        [2.436457, 0.022736, -0.59643, -0.08736],
                        [2.434021, 0.022093, -0.598179, -0.08679],
                    ],
                    [
                        [2.416531, 0.017498, -0.610513, -0.083181],
                        [2.4242, 0.024653, -0.605266, -0.074959],
                    ],
                ],
                device=device,
                dtype=dtype,
            )
        )
        self.assertEqual(result.shape, ref_output.shape)
        torch.testing.assert_close(result, ref_output, atol=atol, rtol=rtol)

        # NestedTensor is only supported for the fast path
        # currently, which won't be used if training.

        if (
            batch_first
            and not training
            and ("xpu" in str(device) or "cpu" in str(device))
            and not TEST_WITH_CROSSREF
        ):
            encoder_input[0][-1] = torch.zeros_like(encoder_input[0][1])
            mask = torch.zeros(
                encoder_input.shape[:-1], device=device, dtype=torch.bool
            )
            mask[0][-1] = True

            nt = torch.nested.nested_tensor(
                [encoder_input[0][:-1], encoder_input[1]], device=device
            )
            result = model(nt)
            ref_output = torch.tensor(
                [
                    [
                        [2.4268184, 0.02042419, -0.603311, -0.08476824],
                        [2.423306, 0.01889652, -0.6057701, -0.08519465],
                        [2.431538, 0.02078694, -0.5999354, -0.08746159],
                        [2.4348664, 0.02212971, -0.5975677, -0.08733892],
                        [2.423133, 0.02097577, -0.60594773, -0.08113337],
                    ],
                    [
                        [2.4279876, 0.02121329, -0.60249615, -0.08410317],
                        [2.4138637, 0.02221113, -0.6124869, -0.07249016],
                        [2.4251041, 0.01974815, -0.6045152, -0.08483928],
                        [2.4335563, 0.0218913, -0.59850943, -0.08683228],
                        [2.4229012, 0.02418739, -0.6061784, -0.07492948],
                    ],
                ],
                device=device,
                dtype=dtype,
            )
            result = result.to_padded_tensor(0)
            ref_output[0][-1] = torch.zeros_like(
                ref_output[0][-1], device=device, dtype=dtype
            )
            result[0][-1] = torch.zeros_like(result[0][-1], device=device, dtype=dtype)
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            if "xpu" in device:
                if dtype == torch.float:
                    atol = 2e-4
                    rtol = 4e-3
                else:
                    atol = 7e-4
                    rtol = 2e-2
                torch.testing.assert_close(result, ref_output, atol=atol, rtol=rtol)
            else:
                torch.testing.assert_close(result, ref_output)

    for batch_first in (True, False):
        for training in (True, False):
            if training:
                cm = contextlib.nullcontext()
            else:
                # Fast path requires inference mode.

                cm = torch.no_grad()
            with cm:
                _test(batch_first=batch_first, training=training, atol=atol, rtol=rtol)


TestNNDeviceType.test_transformerencoderlayer = _test_transformerencoderlayer


@dtypes(torch.half, torch.float)
def _test_transformerencoderlayer_gelu(self, device, dtype):
    # this is a deterministic test for TransformerEncoderLayer with gelu activation

    d_model = 4
    nhead = 2
    dim_feedforward = 16
    dropout = 0.0
    bsz = 2

    atol = 0
    rtol = 1e-5
    if "xpu" in device:
        atol = 1e-3
        rtol = 1e-2

    def _test(activation, batch_first, training):
        def perm_fn(x):
            return x.transpose(1, 0) if batch_first else x

        model = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            batch_first=batch_first,
            device=device,
            dtype=dtype,
        )
        if not training:
            assert dropout == 0
            model = model.eval()
        # set constant weights of the model

        for idx, p in enumerate(model.parameters()):
            x = p.data
            sz = x.view(-1).size(0)
            shape = x.shape
            x = torch.cos(torch.arange(0, sz).float().view(shape))
            p.data.copy_(x)
        # deterministic input

        encoder_input = torch.tensor(
            [[[20.0, 30.0, 40.0, 50.0]]], device=device, dtype=dtype
        )
        result = model(encoder_input)
        ref_output = torch.tensor(
            [[[2.249815, 0.131006, -0.702199, 0.177868]]], device=device, dtype=dtype
        )
        torch.testing.assert_close(result, ref_output, rtol=rtol, atol=atol)

        # deterministic input

        encoder_input = perm_fn(
            torch.tensor(
                [[[1.0, 2.0, 3.0, 4.0]], [[5.0, 6.0, 7.0, 8.0]]],
                device=device,
                dtype=dtype,
            )
        )
        result = model(encoder_input)
        ref_output = perm_fn(
            torch.tensor(
                [
                    [[2.264103, 0.121417, -0.696012, 0.159724]],
                    [[2.264103, 0.121417, -0.696012, 0.159724]],
                ],
                device=device,
                dtype=dtype,
            )
        )
        torch.testing.assert_close(result, ref_output, rtol=rtol, atol=atol)

        # deterministic input

        encoder_input = perm_fn(
            torch.tensor(
                [
                    [
                        [0.7462, 0.6653, 0.5679, 0.4891],
                        [0.5387, 0.1655, 0.3565, 0.0471],
                    ],
                    [
                        [0.8335, 0.2799, 0.5031, 0.2947],
                        [0.1402, 0.0318, 0.7636, 0.1346],
                    ],
                    [
                        [0.6333, 0.9344, 0.1376, 0.9938],
                        [0.8924, 0.2872, 0.6692, 0.2944],
                    ],
                    [
                        [0.9897, 0.6915, 0.3154, 0.1733],
                        [0.8645, 0.3513, 0.3064, 0.0767],
                    ],
                    [
                        [0.8117, 0.2366, 0.4838, 0.7881],
                        [0.3718, 0.4945, 0.9511, 0.0864],
                    ],
                ],
                device=device,
                dtype=dtype,
            )
        )
        result = model(encoder_input)
        ref_output = perm_fn(
            torch.tensor(
                [
                    [
                        [2.42163188, 0.03227153, -0.60714219, -0.05908082],
                        [2.42151276, 0.03302179, -0.60722523, -0.05762651],
                    ],
                    [
                        [2.41926761, 0.02974034, -0.60879519, -0.0621269],
                        [2.41626395, 0.03539356, -0.61087842, -0.04978623],
                    ],
                    [
                        [2.42382808, 0.03218872, -0.6055963, -0.06073591],
                        [2.41983477, 0.03085259, -0.60840145, -0.06046414],
                    ],
                    [
                        [2.42500749, 0.03328855, -0.60476388, -0.0595334],
                        [2.4237977, 0.03290575, -0.60561789, -0.05940082],
                    ],
                    [
                        [2.41383916, 0.02686345, -0.61256377, -0.06380707],
                        [2.42000277, 0.03800944, -0.60824798, -0.04754947],
                    ],
                ],
                device=device,
                dtype=dtype,
            )
        )
        torch.testing.assert_close(result, ref_output, rtol=rtol, atol=atol)

    for activation, batch_first, training in product(
        ("gelu", F.gelu, nn.GELU()), (True, False), (True, False)
    ):
        # Fast path requires inference mode.

        if training:
            cm = contextlib.nullcontext()
        else:
            cm = torch.no_grad()
        with cm:
            _test(activation=activation, batch_first=batch_first, training=training)


TestNNDeviceType.test_transformerencoderlayer_gelu = _test_transformerencoderlayer_gelu

instantiate_device_type_tests(
    TestNNDeviceType, globals(), only_for="xpu", allow_xpu=True
)
instantiate_parametrized_tests(TestNN)
instantiate_parametrized_tests(TestAddRelu)


if __name__ == "__main__":
    run_tests()
