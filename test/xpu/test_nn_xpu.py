# Owner(s): ["module: intel"]

from unittest import SkipTest
import contextlib
import math
import random
import warnings
from itertools import product
import unittest

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import torch.nn.utils.rnn as rnn_utils
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_dtype import integral_types,get_all_math_dtypes
from torch.testing._internal.common_nn import ctcloss_reference
from torch.testing._internal.common_utils import (
    IS_PPC,
    subtest,
    parametrize as parametrize_test,
    instantiate_parametrized_tests,
    run_tests,
    set_default_dtype,
    TEST_WITH_CROSSREF,
    gradcheck,
    gradgradcheck,
    GRADCHECK_NONDET_TOL,
)
try:
    from xpu_test_utils import XPUPatchForImport
except Exception as e:
    from .xpu_test_utils import XPUPatchForImport
with XPUPatchForImport(False):
    from test_nn import TestFusionEval, TestConstantPadNd, TestAddRelu, TestNN, TestNNDeviceType, TestFunctionalPickle, TestFusionEval, TestUtils

# Some cases named with "cuda" will pass, but they actully run on xpu in this UT. These cases are added by "add_test" in test_nn.py
torch.Tensor.cuda=torch.Tensor.xpu
torch.nn.Module.cuda=torch.nn.Module.xpu
torch.cuda.set_rng_state=torch.xpu.set_rng_state
torch.cuda.get_rng_state=torch.xpu.get_rng_state

def _test_type(self):
    l = nn.Linear(10, 20)
    net = nn.Module()
    net.l = l
    net.l2 = l
    net.add_module('empty', None)
    net.register_buffer('indices', torch.LongTensor(1))
    net.float()
    self.assertIsInstance(l.weight.data, torch.FloatTensor)
    self.assertIsInstance(l.bias.data, torch.FloatTensor)
    self.assertIsInstance(net.indices, torch.LongTensor)
    net.double()
    self.assertIsInstance(l.weight.data, torch.DoubleTensor)
    self.assertIsInstance(l.bias.data, torch.DoubleTensor)
    self.assertIsInstance(net.indices, torch.LongTensor)
    net.to(torch.half)
    self.assertIsInstance(l.weight.data, torch.HalfTensor)
    self.assertIsInstance(l.bias.data, torch.HalfTensor)
    self.assertIsInstance(net.indices, torch.LongTensor)
    net.float().xpu()
    self.assertIsInstance(l.weight.data, torch.xpu.FloatTensor)
    self.assertIsInstance(l.bias.data, torch.xpu.FloatTensor)
    self.assertIsInstance(net.indices, torch.xpu.LongTensor)
    net.cpu()
    self.assertIsInstance(l.weight.data, torch.FloatTensor)
    self.assertIsInstance(l.bias.data, torch.FloatTensor)
    self.assertIsInstance(net.indices, torch.LongTensor)
    net.to("xpu", torch.double, True)
    self.assertIsInstance(l.weight.data, torch.xpu.DoubleTensor)
    self.assertIsInstance(l.bias.data, torch.xpu.DoubleTensor)
    self.assertIsInstance(net.indices, torch.xpu.LongTensor)
    net.to(torch.empty(1, device="xpu:0", dtype=torch.half))
    self.assertIsInstance(l.weight.data, torch.xpu.HalfTensor)
    self.assertIsInstance(l.bias.data, torch.xpu.HalfTensor)
    self.assertIsInstance(net.indices, torch.xpu.LongTensor)
    net.to(torch.device("cpu"), non_blocking=True)
    self.assertIsInstance(l.weight.data, torch.HalfTensor)
    self.assertIsInstance(l.bias.data, torch.HalfTensor)
    self.assertIsInstance(net.indices, torch.LongTensor)
    net.to(torch.float)
    self.assertIsInstance(l.weight.data, torch.FloatTensor)
    self.assertIsInstance(l.bias.data, torch.FloatTensor)
    net.to(torch.DoubleTensor(1))
    self.assertIsInstance(l.weight.data, torch.DoubleTensor)
    self.assertIsInstance(l.bias.data, torch.DoubleTensor)
    net.to(device='xpu', dtype=torch.float)
    self.assertIsInstance(l.weight.data, torch.xpu.FloatTensor)
    self.assertIsInstance(l.bias.data, torch.xpu.FloatTensor)
TestNN.test_type=_test_type

def _test_CTCLoss_lengthchecks_xpu(self):
    for target_lengths in [[30, 25, 20], [-1, -1, -1]]:
        for input_lengths in [[50, 50, 50], [-1, -1, -1]]:
            targets = torch.randint(1, 15, (3, 29), dtype=torch.long, device="xpu")
            log_probs = torch.randn(50, 3, 15, dtype=torch.float, device="xpu").log_softmax(2)
            with self.assertRaises(RuntimeError):
                nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths)
TestNN.test_CTCLoss_lengthchecks_cuda = _test_CTCLoss_lengthchecks_xpu

def _test_CTCLoss_critical_target_len(self):
    N = 1
    S = 256
    C = 10
    T = 500
    target = torch.randint(low=1, high=C, size=(S,), dtype=torch.int)
    input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.int)
    target_lengths = torch.tensor(S, dtype=torch.int)
    inp = torch.randn(T, N, C, dtype=torch.float, device='xpu').log_softmax(2).requires_grad_()
    res_gpu = torch.nn.functional.ctc_loss(inp, target, input_lengths, target_lengths, reduction='none')
    res_cpu = torch.nn.functional.ctc_loss(inp.cpu(), target, input_lengths, target_lengths, reduction='none')
    self.assertEqual(res_cpu, res_gpu, atol=1e-3, rtol=0)
TestNN.test_CTCLoss_critical_target_len = _test_CTCLoss_critical_target_len

def _test_CTCLoss_zero_lengths(self):
    devices = ["xpu"]
    N = 3
    S = 2
    C = 200
    T = 1
    target = torch.randint(low=1, high=C, size=(N, S), dtype=torch.int)
    input_lengths = torch.full(size=(N,), fill_value=0, dtype=torch.int)
    target_lengths = torch.full(size=(N,), fill_value=0, dtype=torch.int)
    for device in devices:
        inp = torch.randn(T, N, C, dtype=torch.float, device=device).log_softmax(2).requires_grad_()
        res = torch.nn.functional.ctc_loss(inp, target, input_lengths, target_lengths, reduction='none')
        self.assertTrue((res == 0).all().item())
        res.sum().backward()
        self.assertTrue((inp.grad == 0).all().item())
    target_lengths = torch.full(size=(N,), fill_value=1, dtype=torch.int)
    for device in devices:
        inp = torch.randn(T, N, C, dtype=torch.float, device=device).log_softmax(2).requires_grad_()
        res = torch.nn.functional.ctc_loss(inp, target, input_lengths, target_lengths, reduction='none')
        self.assertTrue((res == torch.inf).all().item())
        res.sum().backward()
        self.assertTrue((inp.grad == 0).all().item())
TestNN.test_CTCLoss_zero_lengths=_test_CTCLoss_zero_lengths

def _test_CTCLoss_zero_infinity(self):
    target_lengths = [60, 25, 20]
    input_lengths = [50, 50, 50]
    targets = torch.randint(1, 15, (sum(target_lengths),), dtype=torch.int, device="xpu")
    log_probs = torch.randn(50, 3, 15, dtype=torch.float, device="xpu").log_softmax(2).requires_grad_()
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

def _test_transformerdecoder(self):
    def get_a_test_layer(use_xpu, activation, batch_first=False):
        d_model = 4
        nhead = 2
        dim_feedforward = 16
        dropout = 0.0
        device = torch.device("xpu" if use_xpu else "cpu")

        layer = nn.TransformerDecoderLayer(
            d_model,
            nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=batch_first).to(device)

        with torch.no_grad():
            # set constant weights of the model
            for idx, p in enumerate(layer.parameters()):
                x = p.data
                sz = x.view(-1).size(0)
                shape = x.shape
                x = torch.cos(torch.arange(0, sz).float().view(shape))
                p.data.copy_(x)

        return layer

    # this is a deterministic test for TransformerDecoder
    for batch_first in (False, True):
        def perm_fn(x):
            return x.transpose(1, 0) if batch_first else x
        activation = F.relu
        use_xpu = torch.xpu.is_available()
        device = torch.device("xpu" if use_xpu else "cpu")

        decoder_layer = get_a_test_layer(use_xpu=use_xpu, activation=activation,
                                            batch_first=batch_first)

        model = nn.TransformerDecoder(decoder_layer, 1).to(device)

        # deterministic input
        decoder_input = torch.tensor([[[20., 30., 40., 50.]]]).to(device)
        memory_input = torch.tensor([[[60., 70., 80., 90.]]]).to(device)
        result = model(decoder_input, memory_input)
        ref_output = torch.tensor(
            [[[2.314351, 0.094805, -0.671322, 0.101977]]]).to(device)
        self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
        torch.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-3)

        # deterministic input
        decoder_input = perm_fn(torch.tensor([[[9., 10., 11., 12.]],
                                                [[11., 12., 13., 14.]]])).to(device)
        memory_input = perm_fn(torch.tensor([[[1., 2., 3., 4.]]])).to(device)
        result = model(decoder_input, memory_input)
        ref_output = perm_fn(torch.tensor([[[2.422245, 0.051716, -0.606338, -0.024756]],
                                            [[2.422245, 0.051716, -0.606338, -0.024756]]]
                                            )).to(device)
        self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
        torch.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-4)

        # deterministic input
        decoder_input = perm_fn(torch.tensor([[[1., 2., 3., 4.]],
                                                [[5., 6., 7., 8.]]])).to(device)
        memory_input = perm_fn(torch.tensor([[[9., 10., 11., 12.]],
                                                [[11., 12., 13., 14.]]])).to(device)
        result = model(decoder_input, memory_input)
        ref_output = perm_fn(torch.tensor([[[2.343536, 0.085561, -0.654954, 0.074991]],
                                            [[2.343536, 0.085561, -0.654954, 0.074991]]]
                                            )).to(device)
        self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
        torch.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-4)

        # deterministic input
        decoder_input = perm_fn(torch.tensor([[[0.4517, 0.6793, 0.5313, 0.0034],
                                                [0.2678, 0.3677, 0.4459, 0.7166]],
                                                [[0.8100, 0.3716, 0.4096, 0.1976],
                                                [0.6958, 0.8844, 0.6081, 0.8315]],
                                                [[0.0494, 0.9343, 0.5955, 0.3830],
                                                [0.5404, 0.3464, 0.9378, 0.6200]]]
                                                )).to(device)
        memory_input = perm_fn(torch.tensor([[[0.7462, 0.6653, 0.5679, 0.4891],
                                                [0.5387, 0.1655, 0.3565, 0.0471]],
                                                [[0.8335, 0.2799, 0.5031, 0.2947],
                                                [0.1402, 0.0318, 0.7636, 0.1346]],
                                                [[0.6333, 0.9344, 0.1376, 0.9938],
                                                [0.8924, 0.2872, 0.6692, 0.2944]],
                                                [[0.9897, 0.6915, 0.3154, 0.1733],
                                                [0.8645, 0.3513, 0.3064, 0.0767]],
                                                [[0.8117, 0.2366, 0.4838, 0.7881],
                                                [0.3718, 0.4945, 0.9511, 0.0864]]]
                                            )).to(device)
        result = model(decoder_input, memory_input)
        ref_output = perm_fn(torch.tensor([[[2.430065, 0.027862, -0.601136, -0.073096],
                                            [2.431935, 0.028907, -0.599809, -0.072488]],
                                            [[2.428457, 0.027053, -0.602275, -0.073462],
                                            [2.431970, 0.029387, -0.599789, -0.071621]],
                                            [[2.431934, 0.028196, -0.599802, -0.073809],
                                            [2.432306, 0.028858, -0.599542, -0.072846]]]
                                            )).to(device)
        self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
        torch.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-5)

        # key_padding_mask
        key_padding_mask = torch.zeros(2, 3).to(device) == 1
        result = model(decoder_input, memory_input,
                        tgt_key_padding_mask=key_padding_mask)
        ref_output = perm_fn(torch.tensor([[[2.430065, 0.027862, -0.601136, -0.073096],
                                            [2.431935, 0.028907, -0.599809, -0.072488]],
                                            [[2.428457, 0.027053, -0.602275, -0.073462],
                                            [2.431970, 0.029387, -0.599789, -0.071621]],
                                            [[2.431934, 0.028196, -0.599802, -0.073809],
                                            [2.432306, 0.028858, -0.599542, -0.072846]]]
                                            )).to(device)
        self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
        torch.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-5)

        # key_padding_mask
        key_padding_mask[0, 2] = 1
        key_padding_mask[1, 1] = 1
        key_padding_mask[1, 2] = 1
        result = model(decoder_input, memory_input,
                        tgt_key_padding_mask=key_padding_mask)
        ref_output = perm_fn(torch.tensor([[[2.430025, 0.027643, -0.601164, -0.073476],
                                            [2.4323, 0.029375, -0.599553, -0.071881]],
                                            [[2.428523, 0.026838, -0.602226, -0.07391],
                                            [2.432634, 0.029842, -0.599318, -0.071253]],
                                            [[2.432278, 0.028152, -0.599555, -0.074139],
                                            [2.432659, 0.029244, -0.599294, -0.072382]]]
                                            )).to(device)
        self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
        torch.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-5)

        # memory_key_padding_mask
        key_padding_mask = torch.zeros(2, 5).to(device) == 1
        result = model(decoder_input, memory_input,
                        memory_key_padding_mask=key_padding_mask)
        ref_output = perm_fn(torch.tensor([[[2.430065, 0.027862, -0.601136, -0.073096],
                                            [2.431935, 0.028907, -0.599809, -0.072488]],
                                            [[2.428457, 0.027053, -0.602275, -0.073462],
                                            [2.431970, 0.029387, -0.599789, -0.071621]],
                                            [[2.431934, 0.028196, -0.599802, -0.073809],
                                            [2.432306, 0.028858, -0.599542, -0.072846]]]
                                            )).to(device)
        self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
        torch.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-5)

        # memory_key_padding_mask
        key_padding_mask[0, 4] = 1
        key_padding_mask[1, 3] = 1
        key_padding_mask[1, 4] = 1
        result = model(decoder_input,
                        memory_input,
                        memory_key_padding_mask=key_padding_mask)
        ref_output = perm_fn(torch.tensor([[[2.429757, 0.027358, -0.601351, -0.073816],
                                            [2.432692, 0.028583, -0.599263, -0.073634]],
                                            [[2.428247, 0.02662, -0.602419, -0.074123],
                                            [2.432657, 0.029055, -0.599293, -0.072732]],
                                            [[2.431515, 0.027687, -0.600096, -0.074459],
                                            [2.433075, 0.028543, -0.598987, -0.073985]]]
                                            )).to(device)
        self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
        torch.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-5)

        # multiple layers no norm
        model = nn.TransformerDecoder(decoder_layer, 2).to(device)

        # deterministic input
        decoder_input = torch.tensor([[[20., 30., 40., 50.]]]).to(device)
        memory_input = torch.tensor([[[60., 70., 80., 90.]]]).to(device)
        result = model(decoder_input, memory_input)
        ref_output = torch.tensor(
            [[[2.31316, 0.0950293, -0.671995, 0.102802]]]).to(device)
        self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
        torch.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-3)

        # multiple layers no norm
        model = nn.TransformerDecoder(decoder_layer, 6).to(device)

        # deterministic input
        decoder_input = perm_fn(torch.tensor([[[0.4517, 0.6793, 0.5313, 0.0034],
                                                [0.2678, 0.3677, 0.4459, 0.7166]],
                                                [[0.8100, 0.3716, 0.4096, 0.1976],
                                                [0.6958, 0.8844, 0.6081, 0.8315]],
                                                [[0.0494, 0.9343, 0.5955, 0.3830],
                                                [0.5404, 0.3464, 0.9378, 0.6200]]]
                                                )).to(device)
        memory_input = perm_fn(torch.tensor([[[0.7462, 0.6653, 0.5679, 0.4891],
                                                [0.5387, 0.1655, 0.3565, 0.0471]],
                                                [[0.8335, 0.2799, 0.5031, 0.2947],
                                                [0.1402, 0.0318, 0.7636, 0.1346]],
                                                [[0.6333, 0.9344, 0.1376, 0.9938],
                                                [0.8924, 0.2872, 0.6692, 0.2944]],
                                                [[0.9897, 0.6915, 0.3154, 0.1733],
                                                [0.8645, 0.3513, 0.3064, 0.0767]],
                                                [[0.8117, 0.2366, 0.4838, 0.7881],
                                                [0.3718, 0.4945, 0.9511, 0.0864]]]
                                            )).to(device)
        result = model(decoder_input, memory_input)
        ref_output = perm_fn(torch.tensor([[[2.42794, 0.026164, -0.60263, -0.0747591],
                                            [2.43113, 0.0279516, -0.600376, -0.0736896]],
                                            [[2.42794, 0.026164, -0.60263, -0.0747591],
                                            [2.43113, 0.0279516, -0.600376, -0.0736896]],
                                            [[2.42794, 0.026164, -0.60263, -0.0747591],
                                            [2.43113, 0.0279516, -0.600376, -0.0736896]]]
                                            )).to(device)
        self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
        torch.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-5)

        # multiple layers with norm
        # d_model = 4
        norm = nn.LayerNorm(4)
        model = nn.TransformerDecoder(decoder_layer, 2, norm=norm).to(device)

        # deterministic input
        decoder_input = torch.tensor([[[20., 30., 40., 50.]]]).to(device)
        memory_input = torch.tensor([[[60., 70., 80., 90.]]]).to(device)
        result = model(decoder_input, memory_input)
        ref_output = torch.tensor(
            [[[1.66166, -0.326986, -1.01466, -0.320017]]]).to(device)
        self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
        torch.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-3)

        # multiple layers with norm
        model = nn.TransformerDecoder(decoder_layer, 6, norm=norm).to(device)

        # deterministic input
        decoder_input = perm_fn(torch.tensor([[[0.4517, 0.6793, 0.5313, 0.0034],
                                                [0.2678, 0.3677, 0.4459, 0.7166]],
                                                [[0.8100, 0.3716, 0.4096, 0.1976],
                                                [0.6958, 0.8844, 0.6081, 0.8315]],
                                                [[0.0494, 0.9343, 0.5955, 0.3830],
                                                [0.5404, 0.3464, 0.9378, 0.6200]]]
                                                )).to(device)
        memory_input = perm_fn(torch.tensor([[[0.7462, 0.6653, 0.5679, 0.4891],
                                                [0.5387, 0.1655, 0.3565, 0.0471]],
                                                [[0.8335, 0.2799, 0.5031, 0.2947],
                                                [0.1402, 0.0318, 0.7636, 0.1346]],
                                                [[0.6333, 0.9344, 0.1376, 0.9938],
                                                [0.8924, 0.2872, 0.6692, 0.2944]],
                                                [[0.9897, 0.6915, 0.3154, 0.1733],
                                                [0.8645, 0.3513, 0.3064, 0.0767]],
                                                [[0.8117, 0.2366, 0.4838, 0.7881],
                                                [0.3718, 0.4945, 0.9511, 0.0864]]]
                                            )).to(device)
        result = model(decoder_input, memory_input)
        ref_output = perm_fn(torch.tensor([[[1.69559, -0.357291, -0.894741, -0.443553],
                                            [1.69571, -0.357363, -0.894154, -0.444196]],
                                            [[1.69559, -0.357291, -0.894741, -0.443553],
                                            [1.69571, -0.357363, -0.894154, -0.444196]],
                                            [[1.69559, -0.357291, -0.894741, -0.443553],
                                            [1.69571, -0.357363, -0.894154, -0.444196]]]
                                            )).to(device)
        self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
        torch.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-5)

        # gelu activation test cases
        activation = "gelu"
        use_xpu = torch.xpu.is_available()
        device = torch.device("xpu" if use_xpu else "cpu")

        decoder_layer = get_a_test_layer(use_xpu=use_xpu, activation=activation,
                                            batch_first=batch_first)

        model = nn.TransformerDecoder(decoder_layer, 1).to(device)

        # deterministic input
        decoder_input = torch.tensor([[[20., 30., 40., 50.]]]).to(device)
        memory_input = torch.tensor([[[60., 70., 80., 90.]]]).to(device)
        result = model(decoder_input, memory_input)
        ref_output = torch.tensor([[[2.306435, 0.095946, -0.675796, 0.10687]]]).to(device)
        self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
        torch.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-3)

        # deterministic input
        decoder_input = perm_fn(torch.tensor([[[9., 10., 11., 12.]],
                                                [[11., 12., 13., 14.]]])).to(device)
        memory_input = perm_fn(torch.tensor([[[1., 2., 3., 4.]]])).to(device)
        result = model(decoder_input, memory_input)
        ref_output = perm_fn(torch.tensor([[[2.415448, 0.054389, -0.610932, -0.0156613]],
                                            [[2.415448, 0.054389, -0.610932, -0.0156613]]])).to(device)
        self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
        torch.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-4)

        # deterministic input
        decoder_input = perm_fn(torch.tensor([[[1., 2., 3., 4.]],
                                                [[5., 6., 7., 8.]]])).to(device)
        memory_input = perm_fn(torch.tensor([[[9., 10., 11., 12.]],
                                                [[11., 12., 13., 14.]]])).to(device)
        result = model(decoder_input, memory_input)
        ref_output = perm_fn(torch.tensor([[[2.338531, 0.087709, -0.65776, 0.080646]],
                                            [[2.338531, 0.087709, -0.65776, 0.080646]]])).to(device)
        self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
        torch.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-4)

        # deterministic input
        decoder_input = perm_fn(torch.tensor([[[0.4517, 0.6793, 0.5313, 0.0034],
                                                [0.2678, 0.3677, 0.4459, 0.7166]],
                                                [[0.8100, 0.3716, 0.4096, 0.1976],
                                                [0.6958, 0.8844, 0.6081, 0.8315]],
                                                [[0.0494, 0.9343, 0.5955, 0.3830],
                                                [0.5404, 0.3464, 0.9378, 0.6200]]]
                                                )).to(device)
        memory_input = perm_fn(torch.tensor([[[0.7462, 0.6653, 0.5679, 0.4891],
                                                [0.5387, 0.1655, 0.3565, 0.0471]],
                                                [[0.8335, 0.2799, 0.5031, 0.2947],
                                                [0.1402, 0.0318, 0.7636, 0.1346]],
                                                [[0.6333, 0.9344, 0.1376, 0.9938],
                                                [0.8924, 0.2872, 0.6692, 0.2944]],
                                                [[0.9897, 0.6915, 0.3154, 0.1733],
                                                [0.8645, 0.3513, 0.3064, 0.0767]],
                                                [[0.8117, 0.2366, 0.4838, 0.7881],
                                                [0.3718, 0.4945, 0.9511, 0.0864]]]
                                            )).to(device)
        result = model(decoder_input, memory_input)
        ref_output = perm_fn(torch.tensor([[[2.42049104, 0.03443088, -0.60793706, -0.05436271],
                                            [2.42210631, 0.03546578, -0.60679895, -0.05357488]],
                                            [[2.41907674, 0.0336104, -0.60892977, -0.05490462],
                                            [2.42216881, 0.03586554, -0.6067524, -0.05289126]],
                                            [[2.42205716, 0.03488046, -0.60683681, -0.05460596],
                                            [2.42240309, 0.0354595, -0.60659063, -0.05378816]]]
                                            )).to(device)
        self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
        torch.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-5)
TestNN.test_transformerdecoder=_test_transformerdecoder

def _test_xpu_weight_format(self):
    rnns = [
        nn.LSTM(10, 20, batch_first=True),
        nn.LSTM(10, 20, batch_first=True, proj_size=10),
        nn.GRU(10, 20, batch_first=True),
        nn.RNN(10, 20, batch_first=True)
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
                self.assertIn('weights are not part of single contiguous chunk of memory', w[0].message.args[0])
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
TestNN.test_cudnn_weight_format=_test_xpu_weight_format

def _test_xpu_weight_tying(self):
    rnns = [
        nn.LSTM(10, 20, batch_first=True, bidirectional=True),
        nn.LSTM(10, 20, batch_first=True, bidirectional=True, proj_size=10),
        nn.GRU(10, 20, batch_first=True, bidirectional=True),
        nn.RNN(10, 20, batch_first=True, bidirectional=True)
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
TestNN.test_cudnn_weight_tying=_test_xpu_weight_tying

def _test_RNN_input_size_zero(self):
        for module in (nn.RNN, nn.LSTM, nn.GRU):
            for device in ["xpu"]:
                input = torch.zeros((5, 0, 3))
                rnn = module(input_size=3, hidden_size=4)
                rnn.xpu()
                input = input.xpu()
                outs = rnn(input)
                self.assertEqual(outs[0].shape, torch.Size([5, 0, 4]))
                # Check that backward does not cause a hard error
                outs[0].sum().backward()
TestNN.test_RNN_input_size_zero=_test_RNN_input_size_zero

def _test_PReLU_backward_requires_grad_false(self):
    devices = ['xpu']
    for d in devices:
        m = nn.PReLU().to(d)
        x = torch.randn(2, 3, 4, 5, device=d, requires_grad=False)
        y = m(x)
        y.mean().backward()
        self.assertEqual(x.grad, None)
TestNN.test_PReLU_backward_requires_grad_false=_test_PReLU_backward_requires_grad_false

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
    input = input.contiguous(memory_format=torch.channels_last).detach().requires_grad_()
    

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

    input = torch.randint(1, 10, (2, 3, 2, 2), dtype=torch.half, device="xpu", requires_grad=True)
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
TestNN.test_batchnorm_nonaffine_cuda_half_input = _test_batchnorm_nonaffine_xpu_half_input


def _test_batchnorm_nhwc_xpu(self):
    for dtype in (torch.half, torch.float):
        (N, C, H, W) = 2, 64, 50, 50
        model = torch.nn.BatchNorm2d(C, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        model = model.eval().xpu().to(dtype)
        inp1 = torch.randn(N, C, H, W, device=torch.device("xpu"), dtype=dtype)
        inp2 = inp1.contiguous(memory_format=torch.channels_last)
        out1 = model(inp1)
        out2 = model(inp2)
        self.assertTrue(torch.equal(out1, out2))
TestNN.test_batchnorm_nhwc_cuda = _test_batchnorm_nhwc_xpu

def _test_to(self):
    m = nn.Linear(3, 5)
    self.assertIs(m, m.to('cpu'))
    self.assertIs(m, m.to('cpu', dtype=torch.float32))
    self.assertEqual(m.double(), m.to(torch.float64))
    self.assertRaises(RuntimeError, lambda: m.to('cpu', copy=True))

    for xpu in ['xpu', 'xpu:0']:
        m2 = m.xpu(device=xpu)
        self.assertIs(m2, m2.to(xpu))
        self.assertEqual(m, m2.to('cpu'))
        self.assertEqual(m2, m.to(xpu))
        self.assertIs(m2, m2.to(dtype=torch.float32))
        self.assertEqual(m2.double(), m2.to(dtype=torch.float64))
TestNN.test_to=_test_to

def _test_pdist(self):
    for device, trans in product(["xpu"], [False, True]):
        inp = torch.randn(4, 5, dtype=torch.double, device=device, requires_grad=True)
        if trans:
            inp = inp.transpose(0, 1)
        for p in [0, 1, 2, 0.5, 1.5, 2.5, float('inf')]:
            self.assertTrue(gradcheck(lambda x: F.pdist(x, p), (inp,)))
TestNN.test_pdist=_test_pdist

def _test_pdist_zeros(self):
    """Test that grad is still valid when dist is 0"""
    for device in ["xpu"]:
        inp = torch.randn(1, 3, dtype=torch.double, device=device, requires_grad=True).repeat([2, 1])
        for p in [0, 1, 2, 0.5, 1.5, 2.5, float('inf')]:
            self.assertTrue(gradcheck(lambda x: F.pdist(x, p), (inp,)))
TestNN.test_pdist_zeros=_test_pdist_zeros

def _test_pdist_empty_row(self):
    for device in ["xpu"]:
        inp = torch.randn(1, 3, dtype=torch.double, device=device, requires_grad=True)
        self.assertTrue(gradcheck(F.pdist, (inp,)))
TestNN.test_pdist_empty_row=_test_pdist_empty_row

def _test_pdist_empty_col(self):
    for device in ["xpu"]:
        inp = torch.randn(4, 0, dtype=torch.double, device=device, requires_grad=True)
        self.assertTrue(gradcheck(F.pdist, (inp,)))
TestNN.test_pdist_empty_col=_test_pdist_empty_col

@unittest.expectedFailure
def _test_pdist_xpu_gradgrad_unimplemented(self):
    inp = torch.randn(4, 5, device='xpu', requires_grad=True)
    gradgradcheck(F.pdist, (inp,))
TestNN.test_pdist_cuda_gradgrad_unimplemented=_test_pdist_xpu_gradgrad_unimplemented

# Merge into OpInfo?
# test for backward in https://github.com/pytorch/pytorch/issues/15511
def _test_pdist_large(self):
    for device in ["xpu"]:
        def func(x):
            return torch.pdist(x, p=2)

        # shape[0] should be able to be (roughly) arbitrarily large, but the kernel
        # is currently limited to smaller sizes (see issue above); this is just testing
        # a floor.
        shape = (1000, 1)
        x = torch.randn(shape, device=device).requires_grad_()
        output = torch.pdist(x, p=2)
        # just run a single backward, as gradcheck/gradgradcheck is expensive here
        output.sum().backward()
TestNN.test_pdist_large=_test_pdist_large

def _test_cosine_embedding_loss_with_diff_type(self):
    for device in ["xpu"]:
        input1 = torch.tensor([[2, 3, 4], [6, 2, 4]], dtype=torch.double, device=device)
        input2 = torch.tensor([[2, 3, 5], [3, 2, 1]], dtype=torch.double, device=device)
        target = torch.tensor([1, -1], dtype=torch.int, device=device)
        expected = torch.nn.functional.cosine_embedding_loss(input1, input2, target)
        for dt1 in get_all_math_dtypes(device):
            for dt2 in get_all_math_dtypes(device):
                for dt3 in get_all_math_dtypes(device):
                    # dt3 is used as dtype for target = [1, -1], so let's skip unsigned type
                    if dt3 == torch.uint8:
                        continue
                    if dt1.is_complex or dt2.is_complex or dt3.is_complex:
                        continue
                    input1 = input1.to(dt1)
                    input2 = input2.to(dt2)
                    target = target.to(dt3)
                    result = torch.nn.functional.cosine_embedding_loss(input1, input2, target)
                    self.assertEqual(result.item(), expected.item(), atol=0.001, rtol=0)
TestNN.test_cosine_embedding_loss_with_diff_type=_test_cosine_embedding_loss_with_diff_type

def _test_cosine_embedding_loss_error_on_diff_shapes(self):
    for device in ["xpu"]:
        input1 = torch.empty((0, 0), dtype=torch.double, device=device)
        input2 = torch.empty((0,), dtype=torch.double, device=device)
        target = torch.empty((0,), dtype=torch.int, device=device)
        with self.assertRaisesRegex(RuntimeError, ".*expects 2D.*"):
            torch.nn.functional.cosine_embedding_loss(input1, input2, target)
TestNN.test_cosine_embedding_loss_error_on_diff_shapes=_test_cosine_embedding_loss_error_on_diff_shapes

def _test_cosine_embedding_loss_error_on_nonexpandable_shapes(self):
    for device in ["xpu"]:
        input1 = torch.empty((1, 5), dtype=torch.double, device=device)
        input2 = torch.empty((1, 6), dtype=torch.double, device=device)
        target = torch.ones((1,), dtype=torch.int, device=device)
        with self.assertRaisesRegex(RuntimeError, ".*must match the size.*"):
            torch.nn.functional.cosine_embedding_loss(input1, input2, target)
TestNN.test_cosine_embedding_loss_error_on_nonexpandable_shapes=_test_cosine_embedding_loss_error_on_nonexpandable_shapes

def _test_kl_div_with_diff_type(self):
    for device in ["xpu"]:
        input = torch.tensor([[2, 3, 5], [3, 2, 1]], dtype=torch.double, device=device)
        target = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.double, device=device)
        expected = torch.nn.functional.kl_div(input, target)
        real_dtypes = (torch.float32, torch.float64, torch.float16)
        for input_dtype, target_dtype in product(real_dtypes, repeat=2):
            if (torch.device(device).type == 'cpu' and target_dtype == torch.float16):
                continue
            input = input.to(input_dtype)
            target = target.to(target_dtype)
            result = torch.nn.functional.kl_div(input, target)
            self.assertEqual(result.item(), expected.item(), atol=0.001, rtol=0)
TestNN.test_kl_div_with_diff_type=_test_kl_div_with_diff_type

def _test_kl_div_with_diff_type_log_target(self):
    for device in ["xpu"]:
        input = torch.tensor([[2, 3, 5], [3, 2, 1]], dtype=torch.double, device=device)
        target = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.double, device=device).log()
        expected = torch.nn.functional.kl_div(input, target, log_target=True)
        real_dtypes = (torch.float32, torch.float64, torch.float16)
        for input_dtype, target_dtype in product(real_dtypes, repeat=2):
            if (torch.device(device).type == 'cpu' and target_dtype == torch.float16):
                continue
            input = input.to(input_dtype)
            target = target.to(target_dtype)
            result = torch.nn.functional.kl_div(input, target, log_target=True)
            self.assertEqual(result.item(), expected.item(), atol=0.001, rtol=0)
TestNN.test_kl_div_with_diff_type_log_target=_test_kl_div_with_diff_type_log_target

def _test_kl_div_log_softmax_target(self):
    for device in ["xpu"]:
        a = torch.tensor([[1.0, 2, 3], [5.0, 5, 5]], device=device)
        b = torch.tensor([[1.0, 2, 3], [5.0, 5, 5]], device=device)
        self.assertEqual(
            F.kl_div(F.log_softmax(a, 1), F.log_softmax(b, 1), reduction='none', log_target=True),
            torch.zeros_like(a)
        )
TestNN.test_kl_div_log_softmax_target=_test_kl_div_log_softmax_target

def _test_smoothl1loss_intergral_target(self):
    def _input_grad(input, target, reduction):
        output = F.smooth_l1_loss(input, target, reduction=reduction, beta=0.5)
        output.sum().backward()
        return input.grad

    for device, dtype, reduction in product(["xpu"],
                                            integral_types(),
                                            ('none', 'sum', 'mean')):
        input = torch.randn(2, 2, device=device, requires_grad=True)
        target = torch.randint(0, 9, (2, 2), device=device, dtype=dtype)

        input_grad_with_float_target = _input_grad(input, target.float(), reduction)

        input_grad = _input_grad(input.detach().clone().requires_grad_(True),
                                    target,
                                    reduction)
        self.assertEqual(input_grad, input_grad_with_float_target)
TestNN.test_smoothl1loss_intergral_target=_test_smoothl1loss_intergral_target

@parametrize_test('device', ['xpu'])
@parametrize_test('nd', [2, 3])
def _test_affine_grid_backward_cl_cf_consistency(self, device, nd):
    # Test based on reported issue: https://github.com/pytorch/pytorch/issues/124154

    theta = torch.rand([6, nd, nd + 1], requires_grad=True, device=device)
    size = [6, 3, 4, 5] if nd == 2 else [6, 3, 4, 5, 5]
    grid = torch.nn.functional.affine_grid(theta, size, align_corners=False)

    grad_tensor = torch.rand(grid.shape, device=device)

    memory_format_cl = torch.channels_last if nd == 2 else torch.channels_last_3d
    grad_tensor_cl = grad_tensor.contiguous(memory_format=memory_format_cl)

    assert theta.grad is None
    grid.backward(grad_tensor_cl)
    theta_grad_cl = theta.grad.clone().contiguous()

    theta.grad.zero_()
    grid.backward(grad_tensor)
    theta_grad_cf = theta.grad

    self.assertEqual(theta_grad_cf, theta_grad_cl)
TestNN.test_affine_grid_backward_cl_cf_consistency=_test_affine_grid_backward_cl_cf_consistency

@set_default_dtype(torch.double)
def _test_grid_sample(self):
    # Backward pass of native C++ and CUDA kernels branch depending on whether input requires gradient,
    # so we test both cases.
    def test(N, C, H, W, mode, padding_mode, align_corners, input_requires_grad):
        def test_shape(N, C, IH, IW, H, W, mode, padding_mode, align_corners):
            for grid_dim_contig_order in [(0, 1, 2, 3), (0, 3, 1, 2), (3, 0, 1, 2), (0, 2, 1, 3)]:
                # grid_dim_contig_order specifies the dimension order that can
                # make grid to be contiguous.
                # i.e., grid.permute(grid_dim_contig_order) is contiguous.
                # e.g., with grid_dim_contig_order=[0, 3, 1, 2], grid should be
                #       initialized with contiguous tensor of shape [N, 2, H, W]
                #       and permuted to [N, H, W, 2] afterwards.
                grid_shape = [N, H, W, 2]
                grid_init_shape = [grid_shape[d] for d in grid_dim_contig_order]
                grid_fwd_permute = [None, None, None, None]
                for i, d in enumerate(grid_dim_contig_order):
                    grid_fwd_permute[d] = i

                def get_grid(device='cpu', data=None):
                    if data is not None:
                        assert list(data.shape) == grid_shape
                        data = data.permute(grid_dim_contig_order).to(device)
                    else:
                        data = torch.randn(grid_init_shape, device=device)
                    grid = data.permute(grid_fwd_permute)
                    assert grid.permute(grid_dim_contig_order).is_contiguous()
                    return grid

                input_cpu = torch.randn(C, N, IH, IW).transpose(0, 1).requires_grad_(input_requires_grad)
                grid_cpu = get_grid().requires_grad_()
                out_cpu = F.grid_sample(input_cpu, grid_cpu, mode=mode, padding_mode=padding_mode,
                                        align_corners=align_corners)
                self.assertTrue(out_cpu.size() == torch.Size([N, C, H, W]))

                gradients = torch.randn_like(out_cpu)
                out_cpu.backward(gradients)


                # Compare against unvectorized CPU fallback

                # NOTE [ grid_sample CPU fallback ]
                # grid_sample uses AVX for 2d images, but that requires 32-bit indexing for
                # 32-bit floats. So we also have a fallback that is used only for float tensors
                # requiring 64-bit indexing. That requires too much memory to run on CI, so we
                # also export the fallback and test it here to ensure feature parity with
                # the vectorized version.
                input_fallback = input_cpu.float().detach_().requires_grad_()
                grid_fallback = grid_cpu.float().detach_().requires_grad_()
                out_fallback = torch._grid_sampler_2d_cpu_fallback(
                    input_fallback, grid_fallback,
                    F.GRID_SAMPLE_INTERPOLATION_MODES[mode],
                    F.GRID_SAMPLE_PADDING_MODES[padding_mode],
                    align_corners)
                self.assertEqual(out_fallback, out_cpu.float(), atol=1e-5, rtol=5e-5)

                out_fallback.backward(gradients.float())
                if input_requires_grad:
                    self.assertEqual(input_fallback.grad, input_cpu.grad.float(), atol=1e-4, rtol=5e-5)
                self.assertEqual(grid_fallback.grad, grid_cpu.grad.float(), atol=1e-4, rtol=5e-5)

                input_xpu = input_cpu.detach().transpose(0, 1).xpu().transpose(0, 1).requires_grad_(input_requires_grad)
                grid_xpu = get_grid('xpu', grid_cpu.detach()).requires_grad_()
                out_xpu = F.grid_sample(input_xpu, grid_xpu, mode=mode, padding_mode=padding_mode,
                                            align_corners=align_corners)
                self.assertEqual(out_cpu, out_xpu)

                out_xpu.backward(gradients.xpu())
                if input_requires_grad:
                    self.assertEqual(input_cpu.grad, input_xpu.grad)
                self.assertEqual(grid_cpu.grad, grid_xpu.grad, atol=5e-5, rtol=0)

                # check that zero-dimensional input strides don't error out
                base_input = torch.randn(N, C, 1, IW)
                input_cpu = base_input.expand_as(input_xpu).requires_grad_(input_requires_grad)
                out_cpu = F.grid_sample(input_cpu, grid_cpu, mode=mode, padding_mode=padding_mode,
                                        align_corners=align_corners)

                input_xpu = base_input.xpu().expand_as(input_xpu).requires_grad_(input_requires_grad)
                out_xpu = F.grid_sample(input_xpu, grid_xpu, mode=mode, padding_mode=padding_mode,
                                            align_corners=align_corners)
                self.assertEqual(out_cpu, out_xpu)

        # test same size output
        test_shape(N, C, H, W, H, W, mode, padding_mode, align_corners)

        # test larger output
        N = random.randint(2, 8)
        C = random.randint(2, 8)
        IH = random.randint(2, 8)
        IW = random.randint(2, 8)
        H = random.randint(IH + 1, 12)
        W = random.randint(IW + 1, 12)
        test_shape(N, C, IH, IW, H, W, mode, padding_mode, align_corners)

        # test smaller output
        N = random.randint(2, 8)
        C = random.randint(2, 8)
        IH = random.randint(2, 8)
        IW = random.randint(2, 8)
        H = random.randint(2, IH)
        W = random.randint(2, IW)
        test_shape(N, C, IH, IW, H, W, mode, padding_mode, align_corners)

        # test 1x1 inpput
        N = random.randint(2, 8)
        C = random.randint(2, 8)
        IH = 1
        IW = 1
        H = random.randint(2, 5)
        W = random.randint(2, 5)
        test_shape(N, C, IH, IW, H, W, mode, padding_mode, align_corners)

        # testing empty grid
        N = random.randint(2, 8)
        C = random.randint(2, 8)
        IH = random.randint(2, 8)
        IW = random.randint(2, 8)
        W = random.randint(3, IW + 2)
        test_shape(N, C, IH, IW, 0, W, mode, padding_mode, align_corners)

        # testing empty channel
        N = random.randint(2, 8)
        IH = random.randint(2, 8)
        IW = random.randint(2, 8)
        H = random.randint(3, IH + 2)
        W = random.randint(3, IW + 2)
        test_shape(N, 0, IH, IW, H, W, mode, padding_mode, align_corners)

        # testing empty batch
        C = random.randint(2, 8)
        IH = random.randint(2, 8)
        IW = random.randint(2, 8)
        H = random.randint(3, IH + 2)
        W = random.randint(3, IW + 2)
        test_shape(0, C, IH, IW, H, W, mode, padding_mode, align_corners)

    for mode in ('bilinear', 'nearest', 'bicubic'):
        for padding_mode in ('zeros', 'border', 'reflection'):
            for align_corners in (True, False):
                # test known input on CPU
                input = torch.arange(1., 11).view(1, 1, 2, 5)
                grid = torch.tensor(
                    [[[-0.9, -4.1], [0, 0.2000], [1, -1], [-0.333, 1e-6], [0.5, 1.0]],
                        [[-1.0, -0.5], [0, 0.3333], [1, -1], [-0.200, 1e-6], [1.5, 0.5]]]).view(1, 2, 5, 2)
                if mode == 'bilinear':
                    if padding_mode == 'zeros':
                        if align_corners:
                            groundtruth = torch.tensor(
                                [[0.0000, 6.0000000000, 5.0000, 4.8340, 9.0000],
                                    [2.2500, 6.3332500450, 5.0000, 5.1000, 0.0000]]).view(1, 1, 2, 5)
                        else:
                            groundtruth = torch.tensor(
                                [[0.0000, 6.5000000000, 1.2500, 4.6675000191, 4.6250],
                                    [0.5000, 7.1665000916, 1.2500, 5.0000000000, 0.0000]]).view(1, 1, 2, 5)
                    elif padding_mode == 'border':
                        if align_corners:
                            groundtruth = torch.tensor(
                                [[1.2000, 6.0000000000, 5.0000, 4.8340, 9.0000],
                                    [2.2500, 6.3332500450, 5.0000, 5.1000, 8.7500]]).view(1, 1, 2, 5)
                        else:
                            groundtruth = torch.tensor(
                                [[1.0000, 6.5000000000, 5.0000, 4.6675000191, 9.2500],
                                    [1.0000, 7.1665000916, 5.0000, 5.0000000000, 10.0000]]).view(1, 1, 2, 5)
                    elif padding_mode == 'reflection':
                        if align_corners:
                            groundtruth = torch.tensor(
                                [[3.4500, 6.0000000000, 5.0000, 4.8340, 9.0000],
                                    [2.2500, 6.3332500450, 5.0000, 5.1000, 7.7500]]).view(1, 1, 2, 5)
                        else:
                            groundtruth = torch.tensor(
                                [[3.0000004768, 6.5000000000, 5.0000, 4.6675000191, 9.2500],
                                    [1.0000000000, 7.1665000916, 5.0000, 5.0000000000, 9.2500]]).view(1, 1, 2, 5)
                    else:
                        raise AssertionError(f"missing groundtruth test for padding mode '{padding_mode}'")
                elif mode == 'nearest':
                    if padding_mode == 'zeros':
                        if align_corners:
                            groundtruth = torch.tensor(
                                [[0., 8., 5., 7., 9.],
                                    [1., 8., 5., 8., 0.]]).view(1, 1, 2, 5)
                        else:
                            groundtruth = torch.tensor(
                                [[0., 8., 5., 7., 0.],
                                    [1., 8., 5., 8., 0.]]).view(1, 1, 2, 5)
                    elif padding_mode == 'border':
                        if align_corners:
                            groundtruth = torch.tensor(
                                [[1., 8., 5., 7., 9.],
                                    [1., 8., 5., 8., 10.]]).view(1, 1, 2, 5)
                        else:
                            groundtruth = torch.tensor(
                                [[1., 8., 5., 7., 9.],
                                    [1., 8., 5., 8., 10.]]).view(1, 1, 2, 5)
                    elif padding_mode == 'reflection':
                        if align_corners:
                            groundtruth = torch.tensor(
                                [[1., 8., 5., 7., 9.],
                                    [1., 8., 5., 8., 9.]]).view(1, 1, 2, 5)
                        else:
                            groundtruth = torch.tensor(
                                [[1., 8., 5., 7., 9.],
                                    [1., 8., 5., 8., 9.]]).view(1, 1, 2, 5)
                    else:
                        raise AssertionError(f"missing groundtruth test for padding mode '{padding_mode}'")
                elif mode == 'bicubic':
                    if padding_mode == 'zeros':
                        if align_corners:
                            groundtruth = torch.tensor(
                                [[-0.10424726, 7.1400003, 5.0000, 5.7842274, 9.0000],
                                    [2.4492188, 7.4814040, 5.0000, 6.0277520, 0.0000]]).view(1, 1, 2, 5)
                        else:
                            groundtruth = torch.tensor(
                                [[0.00000, 7.6287503, 1.0625, 5.5977230, 5.3270264],
                                    [0.40625, 8.0288770, 1.0625, 5.9375067, -0.3515625]]).view(1, 1, 2, 5)
                    elif padding_mode == 'border':
                        if align_corners:
                            groundtruth = torch.tensor(
                                [[1.1520010, 6.0599990, 5.0000, 4.870930, 9.0000000],
                                    [2.1328125, 6.4258375, 5.0000, 5.076003, 8.8671875]]).view(1, 1, 2, 5)
                        else:
                            groundtruth = torch.tensor(
                                [[0.894531, 6.6050020, 4.625, 4.7138715, 9.800781],
                                    [0.906250, 7.2822485, 4.625, 5.0000052, 10.00000]]).view(1, 1, 2, 5)
                    elif padding_mode == 'reflection':
                        if align_corners:
                            groundtruth = torch.tensor(
                                [[3.1822524, 6.239998, 5.0000, 4.8709273, 9.00000],
                                    [1.7812500, 6.703594, 5.0000, 5.0760007, 8.21875]]).view(1, 1, 2, 5)
                        else:
                            groundtruth = torch.tensor(
                                [[2.7993753, 6.6050020, 4.25, 4.7138715, 10.269531],
                                    [0.8125000, 7.2822485, 4.25, 5.0000052, 9.332031]]).view(1, 1, 2, 5)
                    else:
                        raise AssertionError(f"missing groundtruth test for padding mode '{padding_mode}'")

                else:
                    raise AssertionError(f"missing groundtruth test for interpolation mode '{mode}'")
                output = F.grid_sample(input, grid, mode=mode, padding_mode=padding_mode,
                                        align_corners=align_corners)
                self.assertEqual(output, groundtruth, atol=1e-5, rtol=0,
                                    msg=f"groundtruth comparison failed for mode={mode}, "
                                    f"padding_mode={padding_mode}")

                # See NOTE [ grid_sample CPU fallback ]
                output = torch._grid_sampler_2d_cpu_fallback(
                    input.float(), grid.float(),
                    F.GRID_SAMPLE_INTERPOLATION_MODES[mode],
                    F.GRID_SAMPLE_PADDING_MODES[padding_mode],
                    align_corners)
                self.assertEqual(output, groundtruth.float(), atol=1e-5, rtol=0)

                # explicit check for gradient edge cases
                input = torch.arange(0., 5).expand((1, 1, 5, 5))
                grid = torch.tensor(
                    [[[1.0, 1.0], [1.0, -1.0], [0.8, 0.8], [0.8, -0.8]],
                        [[-1.0, -1.0], [-1.0, 1.0], [-0.8, -0.8], [-0.8, 0.8]]]).view(1, 2, 4, 2).requires_grad_()
                if mode == 'bilinear':
                    if padding_mode == 'zeros':
                        if align_corners:
                            groundtruth = torch.tensor(
                                [[[[-8., -8.], [-8., 0.], [2., 0.], [2., 0.]],
                                    [[2., 0.], [2., 0.], [2., 0.], [2., 0.]]]]).view(1, 2, 4, 2)
                        else:
                            groundtruth = torch.tensor(
                                [[[[-5., -5.], [-5., 5.], [-10., -10.], [-10., 10.]],
                                    [[0., 0.], [0., 0.], [0., 0.], [0., 0.]]]]).view(1, 2, 4, 2)
                    elif padding_mode == 'border':
                        if align_corners:
                            groundtruth = torch.tensor(
                                [[[[-0., -0.], [-0., 0.], [2., 0.], [2., 0.]],
                                    [[0., 0.], [0., 0.], [2., 0.], [2., 0.]]]]).view(1, 2, 4, 2)
                        else:
                            groundtruth = torch.tensor(
                                [[[[-0., -0.], [-0., 0.], [-0., -0.], [-0., 0.]],
                                    [[0., 0.], [0., 0.], [0., 0.], [0., 0.]]]]).view(1, 2, 4, 2)
                    elif padding_mode == 'reflection':
                        if align_corners:
                            groundtruth = torch.tensor(
                                [[[[-0., -0.], [-0., 0.], [2., 0.], [2., 0.]],
                                    [[0., 0.], [0., 0.], [2., 0.], [2., 0.]]]]).view(1, 2, 4, 2)
                        else:
                            groundtruth = torch.tensor(
                                [[[[-0., -0.], [-0., 0.], [-0., -0.], [-0., 0.]],
                                    [[0., 0.], [0., 0.], [0., 0.], [0., 0.]]]]).view(1, 2, 4, 2)
                    else:
                        raise AssertionError(f"missing gradient groundtruth test for padding mode '{padding_mode}'")
                elif mode == 'nearest':
                    groundtruth = torch.tensor(
                        [[[[-0., -0.], [-0., 0.], [-0., -0.], [-0., 0.]],
                            [[0., 0.], [0., 0.], [0., 0.], [0., 0.]]]]).view(1, 2, 4, 2)
                elif mode == 'bicubic':
                    if padding_mode == 'zeros':
                        if align_corners:
                            groundtruth = torch.tensor(
                                [[[[-4.5, -6.], [-4.5, 6.], [2.725679, 0.740878], [2.725679, -0.740878]],
                                    [[1.5, 0.], [1.5, 0.], [1.927921, -0.05688], [1.927921, 0.05688]]]]).view(1, 2, 4, 2)
                        else:
                            groundtruth = torch.tensor(
                                [[[[-5.859375, -5.888672], [-5.859375, 5.888672], [-5.6250, -7.5000], [-5.6250, 7.5000]],
                                    [[-0.234375, -0.263672], [-0.234375, 0.263672], [1.8750, 0.], [1.8750, 0.]]]]
                            ).view(1, 2, 4, 2)
                    elif padding_mode == 'border':
                        if align_corners:
                            groundtruth = torch.tensor(
                                [[[[1.5, 0.], [1.5, 0.], [1.74, 0.], [1.74, 0.]],
                                    [[1.5, 0.], [1.5, 0.], [1.74, 0.], [1.74, 0.]]]]).view(1, 2, 4, 2)
                        else:
                            groundtruth = torch.tensor(
                                [[[[-0.46875, 0.], [-0.46875, 0.], [1.8750, 0.], [1.8750, 0.]],
                                    [[-0.46875, 0.], [-0.46875, 0.], [1.8750, 0.], [1.8750, 0.]]]]).view(1, 2, 4, 2)
                    elif padding_mode == 'reflection':
                        if align_corners:
                            groundtruth = torch.tensor(
                                [[[[0., 0.], [0., 0.], [1.92, 0.], [1.92, 0.]],
                                    [[0., 0.], [0., 0.], [1.92, 0.], [1.92, 0.]]]]).view(1, 2, 4, 2)
                        else:
                            groundtruth = torch.tensor(
                                [[[[0., 0.], [0., 0.], [1.875, 0.], [1.875, 0.]],
                                    [[0., 0.], [0., 0.], [1.875, 0.], [1.875, 0.]]]]).view(1, 2, 4, 2)
                    else:
                        raise AssertionError(f"missing gradient groundtruth test for padding mode '{padding_mode}'")
                else:
                    raise AssertionError(f"missing gradient groundtruth test for interpolation mode '{mode}'")
                for input_requires_grad in [False, True]:
                    input = input.requires_grad_(input_requires_grad)
                    F.grid_sample(input, grid, mode=mode, padding_mode=padding_mode,
                                    align_corners=align_corners).sum().backward()
                    self.assertEqual(grid.grad, groundtruth, atol=1e-5, rtol=0,
                                        msg=f"gradient groundtruth comparison failed for mode={mode}, "
                                        f"padding_mode={padding_mode}, input_requires_grad={input_requires_grad}")
                    grid.grad.zero_()

                # See NOTE [ grid_sample CPU fallback ]
                torch._grid_sampler_2d_cpu_fallback(
                    input.float(), grid.float(),
                    F.GRID_SAMPLE_INTERPOLATION_MODES[mode],
                    F.GRID_SAMPLE_PADDING_MODES[padding_mode],
                    align_corners).sum().backward()
                self.assertEqual(grid.grad, groundtruth, atol=1e-5, rtol=0)

                # do gradcheck
                N = random.randint(2, 8)
                C = random.randint(2, 6)
                H = random.randint(2, 8)
                W = random.randint(2, 8)
                input = torch.randn(N, C, H, W, requires_grad=True)
                grid = torch.randn(N, H, W, 2, requires_grad=True)

                for input_requires_grad in [False, True]:
                    input.requires_grad_(input_requires_grad)
                    self.assertTrue(gradcheck(
                        lambda inp, grd: F.grid_sample(inp, grd, mode=mode, padding_mode=padding_mode,
                                                        align_corners=align_corners),
                        (input, grid)))
                    test(N, C, H, W, mode, padding_mode, align_corners, input_requires_grad)

TestNN.test_grid_sample=_test_grid_sample

def _test_grid_sample_nearest_neighbor_rounding_mode_consistency(self):

    device_list = ['xpu']

    def normalize_indices(indices_unnormalized: torch.Tensor, dim_size: int, align_corners: bool):
        if align_corners:
            indices_normalized = 2 * indices_unnormalized / (dim_size - 1) - 1
        else:
            indices_normalized = (indices_unnormalized * 2 + 1) / dim_size - 1
        return indices_normalized

    test_dim_size = 10
    non_test_dim_size = 9
    step_size = 0.1

    batch_size = 1
    channel_size = 1

    mode = 'nearest'
    for device in device_list:
        for padding_mode in ('zeros', 'border', 'reflection'):
            for align_corners in (True, False):
                # Unnormalized inquiry indices
                inquiry_indices_unnormalized = torch.arange(
                    0,
                    test_dim_size - 1 + step_size, step_size,
                    dtype=torch.float32,
                    device=device
                )
                # Note that even though we are trying to create normalized indices
                # which results in x.0 and x.5 indices after unnormalization,
                # because of the numerical error,
                # the rounding direction might not always be expected as designed.
                # The best we could do is to ensure the rounding behaviors across
                # different implementations for different dimensions are
                # exactly the same.
                inquiry_indices = normalize_indices(
                    indices_unnormalized=inquiry_indices_unnormalized,
                    dim_size=test_dim_size,
                    align_corners=align_corners
                )
                num_inqueries = inquiry_indices.shape[0]
                inquiry_fixed_indices = torch.full((num_inqueries,), 0.5, dtype=torch.float32, device=device)
                array_data = torch.rand(test_dim_size, dtype=torch.float32, device=device)
                # 2D grid sample x-dim interpolation
                # The input_tensor_2d_x is of shape
                # [batch_size, channel_size, non_test_dim_size, test_dim_size]
                input_tensor_2d_x = array_data.reshape(1, test_dim_size).repeat(
                    batch_size,
                    channel_size,
                    non_test_dim_size,
                    1
                )
                # The grid_tensor_2d_x is of shape
                # [batch_size, 1, num_inqueries]
                grid_tensor_2d_x = torch.cat(
                    tensors=(
                        inquiry_indices.reshape(num_inqueries, 1),
                        inquiry_fixed_indices.reshape(num_inqueries, 1),
                    ),
                    dim=1
                ).repeat(batch_size, 1, 1, 1)
                # The output_tensor_2d_x is of shape
                # [batch_size, channel_size, 1, num_inqueries]
                output_tensor_2d_x = F.grid_sample(
                    input=input_tensor_2d_x,
                    grid=grid_tensor_2d_x,
                    mode=mode,
                    padding_mode=padding_mode,
                    align_corners=align_corners,
                )
                # 2D grid sample y-dim interpolation
                # The input_tensor_2d_y is of shape
                # [batch_size, channel_size, test_dim_size, non_test_dim_size]
                input_tensor_2d_y = torch.transpose(input_tensor_2d_x, 3, 2)
                # The grid_tensor_2d_y is of shape
                # [batch_size, 1, num_inqueries]
                grid_tensor_2d_y = torch.index_select(
                    grid_tensor_2d_x,
                    -1,
                    torch.tensor([1, 0], dtype=torch.int64, device=device)
                )
                # The output_tensor_2d_y is of shape
                # [batch_size, channel_size, 1, num_inqueries]
                output_tensor_2d_y = F.grid_sample(
                    input=input_tensor_2d_y,
                    grid=grid_tensor_2d_y,
                    mode=mode,
                    padding_mode=padding_mode,
                    align_corners=align_corners,
                )
                self.assertEqual(output_tensor_2d_x[0, 0, 0, :], output_tensor_2d_y[0, 0, 0, :], atol=0, rtol=0)
                # 3D grid sample x-dim interpolation
                # The input_tensor_3d_x is of shape
                # [batch_size, channel_size, non_test_dim_size, non_test_dim_size, test_dim_size]
                input_tensor_3d_x = array_data.reshape(1, test_dim_size).repeat(
                    batch_size, channel_size, non_test_dim_size, non_test_dim_size, 1)
                # The grid_tensor_3d_x is of shape
                # [batch_size, 1, 1, num_inqueries]
                grid_tensor_3d_x = torch.cat(
                    tensors=(
                        inquiry_indices.reshape(num_inqueries, 1),
                        inquiry_fixed_indices.reshape(num_inqueries, 1),
                        inquiry_fixed_indices.reshape(num_inqueries, 1),
                    ),
                    dim=1
                ).repeat(batch_size, 1, 1, 1, 1)
                # The output_tensor_3d_x is of shape
                # [batch_size, channel_size, 1, 1, num_inqueries]
                output_tensor_3d_x = F.grid_sample(
                    input=input_tensor_3d_x,
                    grid=grid_tensor_3d_x,
                    mode=mode,
                    padding_mode=padding_mode,
                    align_corners=align_corners,
                )
                self.assertEqual(output_tensor_2d_x[0, 0, 0, :], output_tensor_3d_x[0, 0, 0, 0, :], atol=0, rtol=0)
                # 3D grid sample y-dim interpolation
                # The input_tensor_3d_y is of shape
                # [batch_size, channel_size, non_test_dim_size, test_dim_size, non_test_dim_size]
                input_tensor_3d_y = torch.transpose(input_tensor_3d_x, 4, 3)
                # The grid_tensor_3d_y is of shape
                # [batch_size, 1, 1, num_inqueries]
                grid_tensor_3d_y = torch.index_select(
                    grid_tensor_3d_x,
                    -1,
                    torch.tensor([1, 0, 2], dtype=torch.int64, device=device)
                )
                # The output_tensor_3d_y is of shape
                # [batch_size, channel_size, 1, 1, num_inqueries]
                output_tensor_3d_y = F.grid_sample(
                    input=input_tensor_3d_y,
                    grid=grid_tensor_3d_y,
                    mode=mode,
                    padding_mode=padding_mode,
                    align_corners=align_corners,
                )
                self.assertEqual(output_tensor_2d_x[0, 0, 0, :], output_tensor_3d_y[0, 0, 0, 0, :], atol=0, rtol=0)
                # 3D grid sample z-dim interpolation
                # The input_tensor_3d_z is of shape
                # [batch_size, channel_size, non_test_dim_size, non_test_dim_size, test_dim_size]
                input_tensor_3d_z = torch.transpose(input_tensor_3d_x, 4, 2)
                # The grid_tensor_3d_z is of shape
                # [batch_size, 1, 1, num_inqueries]
                grid_tensor_3d_z = torch.index_select(
                    grid_tensor_3d_x,
                    -1,
                    torch.tensor([1, 2, 0], dtype=torch.int64, device=device)
                )
                # The output_tensor_3d_z is of shape
                # [batch_size, channel_size, 1, 1, num_inqueries]
                output_tensor_3d_z = F.grid_sample(
                    input=input_tensor_3d_z,
                    grid=grid_tensor_3d_z,
                    mode=mode,
                    padding_mode=padding_mode,
                    align_corners=align_corners,
                )
                self.assertEqual(output_tensor_2d_x[0, 0, 0, :], output_tensor_3d_z[0, 0, 0, 0, :], atol=0, rtol=0)
TestNN.test_grid_sample_nearest_neighbor_rounding_mode_consistency=_test_grid_sample_nearest_neighbor_rounding_mode_consistency

@set_default_dtype(torch.double)
def _test_upsampling_not_recompute_scale_factor(self):
    # test output against known input: result must match opencv
    in_t = torch.arange(8.).view(1, 2, 2, 2)
    expected_out_t = torch.tensor(
        [[[[-0.32725, -0.08843, 0.37933, 0.79744],
            [0.15039, 0.38921, 0.85697, 1.27508],
            [1.08591, 1.32473, 1.79249, 2.21060],
            [1.92213, 2.16095, 2.62871, 3.04682]],

            [[3.67275, 3.91157, 4.37933, 4.79744],
            [4.15039, 4.38921, 4.85697, 5.27508],
            [5.08591, 5.32473, 5.79249, 6.21060],
            [5.92213, 6.16095, 6.62871, 7.04682]]]])
    if IS_PPC:
        # Both OpenCV and PyTorch give a slightly different result on PPC
        expected_out_t = torch.tensor(
            [[[[-0.32725, -0.08843, 0.37933, 0.79744],
                [0.15039, 0.38921, 0.85697, 1.27508],
                [1.08591, 1.32473, 1.79249, 2.21060],
                [1.92212, 2.16094, 2.62870, 3.04681]],

                [[3.67275, 3.91157, 4.37933, 4.79743],
                [4.15039, 4.38921, 4.85697, 5.27508],
                [5.08591, 5.32473, 5.79249, 6.21059],
                [5.92212, 6.16094, 6.62870, 7.04680]]]])
    out_t = F.interpolate(in_t, scale_factor=2.3, mode='bicubic', align_corners=False, recompute_scale_factor=False)
    torch.set_printoptions(precision=5)
    self.assertEqual(out_t, expected_out_t, atol=1e-4, rtol=0)

    device_list = ['xpu']

    for align_corners in [True, False]:
        kwargs = dict(mode='bicubic', align_corners=align_corners)
        # test float scale factor up & downsampling
        for device in device_list:
            for scale_factor in [0.6, 1.6, 2.3]:
                in_t = torch.ones(2, 2, 2, 2).to(device)
                out_t = F.interpolate(in_t, scale_factor=scale_factor, **kwargs)
                out_size = int(math.floor(in_t.shape[-1] * scale_factor))
                self.assertEqual(torch.ones(2, 2, out_size, out_size), out_t.data, atol=1e-5, rtol=0)

                input = torch.randn(2, 2, 2, 2, requires_grad=True)
                gradcheck(lambda x: F.interpolate(x, out_size, **kwargs), [input])
TestNN.test_upsampling_not_recompute_scale_factor=_test_upsampling_not_recompute_scale_factor


@set_default_dtype(torch.double)
def _test_interpolate(self):
    def _test_interpolate_non_integer_size_warning(in_t, out_size, dim, **kwargs):
        test_sizes = [float(out_size),
                        torch.tensor(out_size, dtype=torch.float)]
        for size in test_sizes:
            self.assertRaisesRegex(TypeError,
                                    "(expected size to be one of int or).*",
                                    F.interpolate, in_t, size=(size,) * dim, **kwargs)

    def _test_interpolate_helper(in_t, scale_factor, layer):
        out_size = int(math.floor(in_t.shape[-1] * scale_factor))
        dim = len(in_t.shape) - 2
        out_shape = [1, 1] + [out_size] * dim
        with warnings.catch_warnings(record=True) as w:
            out_t = layer(in_t)
        self.assertEqual(torch.ones(out_shape), out_t)

        self.assertEqual(
            F.interpolate(in_t, (out_size,) * dim, **kwargs),
            F.interpolate(in_t, scale_factor=scale_factor, **kwargs))
        gradcheck(lambda x: F.interpolate(x, out_size, **kwargs), [in_t], nondet_tol=GRADCHECK_NONDET_TOL)
        gradgradcheck(lambda x: F.interpolate(x, out_size, **kwargs), [in_t], nondet_tol=GRADCHECK_NONDET_TOL)
        _test_interpolate_non_integer_size_warning(in_t, out_size, dim, **kwargs)

    def _make_input(dim, device):
        size = [1, 1]
        size += [2] * dim
        return torch.ones(size, requires_grad=True, device=device)

    device_list = ['xpu']

    for device in device_list:
        for scale_factor in [0.5, 1.5, 2]:
            for mode in ['nearest', 'area']:
                kwargs = dict(mode=mode)
                m = nn.Upsample(scale_factor=scale_factor, **kwargs).to(device)
                for input in [_make_input(1, device), _make_input(2, device), _make_input(3, device)]:
                    _test_interpolate_helper(input, scale_factor, m)

            for align_corners in [True, False]:
                kwargs = dict(mode='linear', align_corners=align_corners)
                m = nn.Upsample(scale_factor=scale_factor, **kwargs).to(device)
                _test_interpolate_helper(_make_input(1, device), scale_factor, m)

                kwargs = dict(mode='bilinear', align_corners=align_corners)
                m = nn.Upsample(scale_factor=scale_factor, **kwargs).to(device)
                _test_interpolate_helper(_make_input(2, device), scale_factor, m)

                kwargs = dict(mode='bicubic', align_corners=align_corners)

                def m(t):
                    return F.interpolate(t, scale_factor=scale_factor, **kwargs).to(device)
                _test_interpolate_helper(_make_input(2, device), scale_factor, m)

                kwargs = dict(mode='trilinear', align_corners=align_corners)
                m = nn.Upsample(scale_factor=scale_factor, **kwargs).to(device)
                _test_interpolate_helper(_make_input(3, device), scale_factor, m)
TestNN.test_interpolate=_test_interpolate

@parametrize_test('device', ['xpu'])
@parametrize_test('bias', [
    subtest(False, name='nobias'), subtest(True, name='bias')])
@parametrize_test('weight_layout', [
    subtest(torch.strided, name='weightStrided'),
    subtest(torch.sparse_coo, name='weightCOO'),
    subtest(torch.sparse_csr, name='weightCSR'),
    subtest(torch.sparse_csc, name='weightCSC'),
    # TODO: addmm: computation on CPU is not implemented for Strided + Strided @ SparseBsr
    # subtest(torch.sparse_bsr, name='weightBSR'),
    # subtest(torch.sparse_bsc, name='weightBSC'),
])
def _test_linear_autograd(self, device, bias, weight_layout):
    module = nn.Linear(4, 4, bias=bias, device=device)
    if weight_layout == torch.strided:
        pass
    elif weight_layout == torch.sparse_csr:
        module.weight = nn.Parameter(module.weight.to_sparse_csr())
    elif weight_layout == torch.sparse_csc:
        module.weight = nn.Parameter(module.weight.to_sparse_csc())
    elif weight_layout == torch.sparse_bsr:
        module.weight = nn.Parameter(module.weight.to_sparse_bsr((2, 2)))
    elif weight_layout == torch.sparse_bsc:
        module.weight = nn.Parameter(module.weight.to_sparse_bsc((2, 2)))
    elif weight_layout == torch.sparse_coo:
        module.weight = nn.Parameter(module.weight.to_sparse_coo())
    else:
        raise AssertionError

    inp = torch.randn(4, requires_grad=True, device=device)
    res = module(inp)
    if bias:
        expected = (torch.einsum("i,ji->j", inp, module.weight.to_dense())) + module.bias
    else:
        expected = (torch.einsum("i,ji->j", inp, module.weight.to_dense()))
    self.assertEqual(res, expected)

    grad_output = torch.randn(4, device=device)
    grads = torch.autograd.grad(res, [module.weight, inp], grad_output)
    grads_expected = torch.autograd.grad(expected, [module.weight, inp], grad_output)

    self.assertEqual(grads_expected[0].layout, weight_layout)

    for g, ge in zip(grads, grads_expected):
        self.assertEqual(g, ge)
TestNN.test_linear_autograd=_test_linear_autograd

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

def _test_layer_norm_grads_with_create_graph_flag(self):
    atol = 1e-5
    rtol = 1e-3

    x = torch.randn((4, 4, 16), requires_grad=True)
    layer_norm = nn.LayerNorm((16,), 1e-5, True)
    with torch.no_grad():
        layer_norm.weight = torch.nn.Parameter(0.1 * torch.ones_like(layer_norm.weight))

    grads1 = torch.autograd.grad(layer_norm(x).sum(), x, create_graph=False)[0]
    grads2 = torch.autograd.grad(layer_norm(x).sum(), x, create_graph=True)[0]

    self.assertEqual(grads1, grads2, rtol=rtol, atol=atol)

    x = x.to('xpu')
    layer_norm = layer_norm.to('xpu')

    grads1 = torch.autograd.grad(layer_norm(x).sum(), x, create_graph=False)[0]
    grads2 = torch.autograd.grad(layer_norm(x).sum(), x, create_graph=True)[0]

    self.assertEqual(grads1, grads2, rtol=rtol, atol=atol)
TestNN.test_layer_norm_grads_with_create_graph_flag=_test_layer_norm_grads_with_create_graph_flag

def _test_sync_batchnorm_backward_elemt(self):
    device = 'xpu'
    saved_input = torch.rand(2, 3, 2, 1, device=device)
    grad_output = torch.rand(2, 3, 2, 1, device=device)
    mean = torch.rand(3, device=device)
    invstd = torch.rand(3, device=device)
    weight = torch.rand(3, device=device)
    sum_dy = torch.rand(3, device=device)
    sum_dy_xmu = torch.rand(3, device=device)
    count_tensor = torch.tensor([5, 5, 5], dtype=torch.int32, device=device)

    gI_contiguous = torch.batch_norm_backward_elemt(
        grad_output,
        saved_input,
        mean,
        invstd,
        weight,
        sum_dy,
        sum_dy_xmu,
        count_tensor
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
            count_tensor
        )
        self.assertEqual(gI_actual, gI_contiguous)
TestNN.test_sync_batchnorm_backward_elemt=_test_sync_batchnorm_backward_elemt

def _test_sync_batchnorm_accuracy_xpu(self):
    # The target of this test is to test the functionality and accuracy of
    #   those single-GPU xpu kernels used in SyncBatchNorm
    # They are:
    #   fwd: torch.batch_norm_stats, torch.batch_norm_gather_stats_with_counts, torch.batch_norm_elemt
    #   bwd: torch.batch_norm_backward_reduce, torch.batch_norm_backward_elemt

    def _batch_norm_stats(data, memory_format, mean_axes):
        mean1, _ = torch.batch_norm_stats(data, 1e-5)
        mean2, _ = torch.batch_norm_stats(data.to(memory_format=memory_format), 1e-5)
        mean_ref = torch.mean(data, mean_axes, keepdim=False)

        self.assertEqual(mean_ref, mean1)
        self.assertEqual(mean_ref, mean2)

    _batch_norm_stats(torch.randn(1, 96, 112, 112, dtype=torch.float, device='xpu'), torch.channels_last, (0, 2, 3))
    _batch_norm_stats(torch.randn(1, 96, 112, 112, 112, dtype=torch.float, device='xpu'), torch.channels_last_3d, (0, 2, 3, 4))
TestNN.test_sync_batchnorm_accuracy_cuda=_test_sync_batchnorm_accuracy_xpu


def _test_groupnorm_nhwc(self):
    def helper(self, size, groups, memory_format, is_mixed, device, dtype):
        channels = size[1]
        input = torch.randn(size, dtype=dtype, device=device, requires_grad=True)
        input = input.contiguous(memory_format=memory_format)
        input.retain_grad()
        grad = torch.randn(size, dtype=dtype, device=device)
        grad = grad.contiguous(memory_format=memory_format)
        if dtype == torch.bfloat16 and is_mixed:
            gn = nn.GroupNorm(groups, channels).to(device).to(torch.float)
        else:
            gn = nn.GroupNorm(groups, channels).to(device).to(dtype)
        gn.weight.data.uniform_()
        gn.bias.data.uniform_()

        ref_input = input.detach().clone().contiguous(memory_format=torch.contiguous_format).requires_grad_(True)
        ref_grad = grad.detach().clone().contiguous(memory_format=torch.contiguous_format)
        if dtype == torch.bfloat16 and is_mixed:
            ref_gn = nn.GroupNorm(groups, channels).to(device).to(torch.float)
        else:
            ref_gn = nn.GroupNorm(groups, channels).to(device).to(dtype)
        ref_gn.load_state_dict(gn.state_dict())
        out = gn(input)
        out.backward(grad)
        ref_out = ref_gn(ref_input)
        ref_out.backward(ref_grad)

        self.assertTrue(out.is_contiguous(memory_format=memory_format))
        print(f'{memory_format}')
        self.assertTrue(ref_out.is_contiguous(memory_format=torch.contiguous_format))

        self.assertEqual(out, ref_out)
        # parameters in bfloat16/Half is not recommended
        atol = 5e-4
        rtol = 8e-3

        self.assertEqual(gn.weight.grad, ref_gn.weight.grad, atol=atol, rtol=rtol)
        self.assertEqual(gn.bias.grad, ref_gn.bias.grad, atol=atol, rtol=rtol)
        self.assertEqual(input.grad, ref_input.grad, atol=atol, rtol=rtol)

    for device in ['xpu']:
        for dtype in [torch.float, torch.double]:
            if device == 'xpu' and dtype not in [torch.float, torch.double]:
                continue
            for is_mixed in [True, False]:
                helper(self, (4, 8, 10, 10), 4, torch.channels_last, is_mixed, device, dtype)
                helper(self, (2, 30, 9, 9), 3, torch.channels_last, is_mixed, device, dtype)
                helper(self, (4, 8, 40, 40), 4, torch.channels_last, is_mixed, device, dtype)
                helper(self, (4, 40, 40, 40), 2, torch.channels_last, is_mixed, device, dtype)
                helper(self, (2, 30, 50, 50), 3, torch.channels_last, is_mixed, device, dtype)
                helper(self, (2, 60, 50, 50), 3, torch.channels_last, is_mixed, device, dtype)
TestNN.test_groupnorm_nhwc = None # TODO: Disable it temporarily as Pytorch has revert the PR: https://github.com/pytorch/pytorch/pull/126635

@parametrize_test("memory_format", [torch.contiguous_format, torch.channels_last])
@parametrize_test("mode", ["bilinear", "bicubic"])
@parametrize_test("antialias", [True, False])
@parametrize_test("align_corners", [True, False])
@parametrize_test("num_channels", [3, 5])
@parametrize_test("output_size", [32, 600])
@parametrize_test("check_as_unsqueezed_3d_tensor", [True, False])
@parametrize_test("non_contig", [False, "sliced", "restrided"])
@parametrize_test("batch_size", [1, 5])
def _test_upsamplingBiMode2d_consistency(
    self,
    device,
    memory_format,
    mode,
    antialias,
    align_corners,
    num_channels,
    output_size,
    check_as_unsqueezed_3d_tensor,
    non_contig,
    batch_size,
):
    # Check output value consistency between resized_input_uint8 and resized input_float
    if torch.device(device).type == "xpu":
        raise SkipTest("XPU implementation is not yet supporting uint8")

    torch.manual_seed(0)

    # - input range is set to [30, 220] for bicubic mode, because the bicubic kernel may create
    #   [intermediate] values outside of the [0, 255] range, which need
    #   to be clipped in uint8 path, but not in float path. This isn't
    #   an issue with bilinear kernel.
    input_range = (30, 220) if mode == "bicubic" else (0, 256)
    input_ui8 = torch.randint(*input_range, size=(batch_size, num_channels, 400, 400), dtype=torch.uint8, device=device)
    input_ui8 = input_ui8.contiguous(memory_format=memory_format)

    if non_contig == "sliced":
        input_ui8 = input_ui8[:, :, 10:-10, 10:-10]
    elif non_contig == "restrided":
        input_ui8 = input_ui8[:, :, ::2, ::2]

    if batch_size == 1 and check_as_unsqueezed_3d_tensor:
        input_ui8 = input_ui8[0, ...]
        input_ui8 = input_ui8[None, ...]

    input_f32 = input_ui8.float()

    output_f32 = F.interpolate(
        input_f32, size=(output_size, output_size), mode=mode, align_corners=align_corners, antialias=antialias
    ).round().clip(0, 255)
    output_ui8 = F.interpolate(
        input_ui8, size=(output_size, output_size), mode=mode, align_corners=align_corners, antialias=antialias
    )

    if non_contig is False:
        self.assertTrue(input_ui8.is_contiguous(memory_format=memory_format))

    # FIXME if-clause shows the current behaviour which is definitely unexpected.
    # Ideally we want to fix it such that both the ui8 and f32 outputs are also channels_last
    # See for more details: https://github.com/pytorch/pytorch/pull/100373
    if batch_size == 1 and check_as_unsqueezed_3d_tensor and memory_format == torch.channels_last:
        self.assertTrue(output_ui8.is_contiguous())
        self.assertTrue(output_f32.is_contiguous())
    else:
        self.assertTrue(output_ui8.is_contiguous(memory_format=memory_format))
        self.assertTrue(output_f32.is_contiguous(memory_format=memory_format))

    if mode == "bilinear":
        torch.testing.assert_close(output_f32, output_ui8.float(), rtol=0, atol=1)
    else:
        diff = (output_f32 - output_ui8.float()).abs()
        self.assertLess(diff.max(), 15)

        threshold = 2
        percent = 3
        self.assertLess((diff > threshold).float().mean(), percent / 100)

        threshold = 5
        percent = 1
        self.assertLess((diff > threshold).float().mean(), percent / 100)

        self.assertLess(diff.mean(), 0.4)
TestNNDeviceType.test_upsamplingBiMode2d_consistency=_test_upsamplingBiMode2d_consistency

@parametrize_test("memory_format", [torch.contiguous_format, torch.channels_last])
@parametrize_test("align_corners", [True, False])
@parametrize_test("input_size, output_size", [(399, 437), (403, 377)])
def _test_upsamplingBiLinear2d_consistency_interp_size_bug(self, device, memory_format, align_corners, input_size, output_size):
    # Non-regression test for https://github.com/pytorch/pytorch/pull/101403

    if torch.device(device).type == "xpu":
        raise SkipTest("XPU implementation is not yet supporting uint8")

    mode = "bilinear"
    input_ui8 = torch.randint(0, 256, size=(1, 3, input_size, input_size), dtype=torch.uint8, device=device)
    input_ui8 = input_ui8.contiguous(memory_format=memory_format)
    input_f32 = input_ui8.float()

    output_f32 = F.interpolate(
        input_f32, size=(output_size, output_size), mode=mode, align_corners=align_corners, antialias=False
    ).round().to(torch.uint8)
    output_ui8 = F.interpolate(
        input_ui8, size=(output_size, output_size), mode=mode, align_corners=align_corners, antialias=False
    )
    torch.testing.assert_close(output_f32, output_ui8, atol=1, rtol=0)
TestNNDeviceType.test_upsamplingBiLinear2d_consistency_interp_size_bug=_test_upsamplingBiLinear2d_consistency_interp_size_bug

def _test_device_mask(self, device):
    def is_xpu(packed):
        return packed.data.device.type=="xpu"
    for enforce_sorted in [True, False]:
        padded, lengths = self._padded_sequence('cpu', torch.float)
        packed = rnn_utils.pack_padded_sequence(
            padded, lengths, enforce_sorted=enforce_sorted)
        self.assertFalse(is_xpu(packed))
        packed = packed.to(device)
        self.assertTrue(is_xpu(packed))
        unpacked, _ = rnn_utils.pad_packed_sequence(packed)
        self.assertTrue(is_xpu(unpacked))
        self.assertEqual(unpacked.dtype, torch.float)
TestNNDeviceType.test_device_mask=_test_device_mask

def _test_overwrite_module_params_on_conversion_cpu_device(self, device):
    # Test that under the current default settings
    # (`torch.__future__.get_overwrite_module_params_on_conversion() == False`),
    # a view to a module's parameters is not pointing to the same storage as
    # its base variable after converting the module to a different device.
    m = nn.Linear(20, 10)
    mw = m.weight[:]
    m.to(device)
    with torch.no_grad():
        # Without using `torch.no_grad()`, this will leak CUDA memory.
        # (Issue is filed at https://github.com/pytorch/pytorch/issues/21875)
        mw[0][0] = 5
        self.assertTrue(mw[0][0].device.type == "cpu")
        self.assertTrue(mw._base[0][0].device.type == "xpu")

    try:
        torch.__future__.set_overwrite_module_params_on_conversion(True)

        # Test that if `torch.__future__.get_overwrite_module_params_on_conversion() == True`,
        # a view to a module's parameters is still pointing to the same storage as
        # its base variable after converting the module to a different device.
        m = nn.Linear(20, 10)
        mw = m.weight[:]
        m.to(device)
        with torch.no_grad():
            mw[0][0] = 5
        self.assertTrue(mw[0][0] == mw._base[0][0])

        # Test that if `torch.__future__.get_overwrite_module_params_on_conversion() == True`,
        # `cpu_module.to("cuda")` doesn't preserve previous references to
        # `cpu_module`'s parameters or gradients.
        m = nn.Linear(20, 10)
        m.weight.grad = torch.randn(10, 20)
        weight_ref = m.weight
        weight_grad_ref = m.weight.grad
        m.to(device)
        self.assertNotEqual(weight_ref.device, m.weight.device)
        self.assertNotEqual(weight_grad_ref.device, m.weight.grad.device)
    finally:
        torch.__future__.set_overwrite_module_params_on_conversion(False)
TestNNDeviceType.test_overwrite_module_params_on_conversion_cpu_device=_test_overwrite_module_params_on_conversion_cpu_device

def _test_ctc_loss_xpu(self, device):
    batch_size = 16
    input_length = 30
    num_labels = 101
    target_length = 15
    targets = torch.randint(1, num_labels, (batch_size * target_length,),
                            device='xpu', dtype=torch.long)
    log_probs = torch.log_softmax(torch.randn(input_length, batch_size, num_labels, device='xpu', dtype=torch.float), 2)
    log_probs.requires_grad_()

    input_lengths = batch_size * [input_length]
    target_lengths = batch_size * [target_length]
    grad_out = torch.randn(batch_size, device='xpu', dtype=torch.float)
    with torch.backends.cudnn.flags(enabled=False):
        loss_native = torch.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths, reduction='none')
        grad_native, = torch.autograd.grad(loss_native, log_probs, grad_out)
    loss_cudnn = torch.nn.functional.ctc_loss(log_probs, targets.to('cpu', torch.int32),
                                                input_lengths, target_lengths, reduction='none')
    self.assertTrue("xpu" in str(loss_cudnn.grad_fn))
    grad_cudnn, = torch.autograd.grad(loss_cudnn, log_probs, grad_out)
    self.assertEqual(grad_cudnn, grad_native, atol=1e-4, rtol=0)
TestNNDeviceType.test_ctc_loss_cudnn=_test_ctc_loss_xpu

def _test_ctc_loss_xpu_tensor(self, device):
    batch_size = 16
    input_length = 30
    num_labels = 101
    target_length = 15
    targets = torch.randint(1, num_labels, (batch_size * target_length,),
                            device='xpu', dtype=torch.long)
    log_probs = torch.log_softmax(torch.randn(input_length, batch_size, num_labels, device='xpu', dtype=torch.float), 2)
    log_probs.requires_grad_()

    input_lengths = batch_size * [input_length]
    input_lengths = torch.linspace(start=15, end=input_length, steps=batch_size, dtype=torch.long, device='xpu')
    target_lengths = torch.tensor(batch_size * [target_length], dtype=torch.long, device='xpu')
    grad_out = torch.randn(batch_size, device='xpu', dtype=torch.float)
    with torch.backends.cudnn.flags(enabled=False):
        loss_native = torch.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths, reduction='none')
        grad_native, = torch.autograd.grad(loss_native, log_probs, grad_out)
    loss_cudnn = torch.nn.functional.ctc_loss(log_probs,
                                                targets.to('xpu', torch.int32),
                                                input_lengths.to('xpu', torch.int32),
                                                target_lengths.to('xpu', torch.int32),
                                                reduction='none')
    self.assertTrue("xpu" in str(loss_cudnn.grad_fn))
    grad_cudnn, = torch.autograd.grad(loss_cudnn, log_probs, grad_out)
    self.assertEqual(grad_cudnn, grad_native, atol=1e-4, rtol=0)
TestNNDeviceType.test_ctc_loss_cudnn_tensor=_test_ctc_loss_xpu_tensor

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
TestNNDeviceType.test_masked_softmax_devices_parity = _test_masked_softmax_devices_parity

def _test_layernorm_half_precision(self):
    width = 128
    input = torch.rand(1, 5, width, device="xpu", dtype=torch.half) * 0.1
    normalized_shape = (width,)
    weight = torch.ones(width, device="xpu", dtype=torch.half)
    bias = torch.zeros(width, device="xpu", dtype=torch.half)
    eps = 1e-5

    output_fp16 = torch.layer_norm(input, normalized_shape, weight, bias, eps)
    output_fp32 = torch.layer_norm(input.float(), normalized_shape, weight.float(), bias.float(), eps).half()
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

        model = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                            batch_first=batch_first, device=device, dtype=dtype)

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
        encoder_input = torch.tensor([[[20., 30., 40., 50.]]], device=device, dtype=dtype)
        result = model(encoder_input)
        ref_output = torch.tensor([[[2.258703, 0.127985, -0.697881, 0.170862]]], device=device, dtype=dtype)
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
        encoder_input = perm_fn(torch.tensor([[[1., 2., 3., 4.]],
                                                [[5., 6., 7., 8.]]], device=device, dtype=dtype))
        result = model(encoder_input)
        ref_output = perm_fn(torch.tensor([[[2.272644, 0.119035, -0.691669, 0.153486]],
                                            [[2.272644, 0.119035, -0.691669, 0.153486]]], device=device, dtype=dtype))
        self.assertEqual(result.shape, ref_output.shape)
        torch.testing.assert_close(result, ref_output, atol=atol, rtol=rtol)
        # all 0 which is no masking
        mask = torch.tensor([[0, 0]], device=device) == 1
        result = model(encoder_input, src_key_padding_mask=mask)
        self.assertEqual(result.shape, ref_output.shape)
        torch.testing.assert_close(result, ref_output, atol=atol, rtol=rtol)
        mask = torch.tensor([[1, 0]], device=device) == 1
        result = model(encoder_input, src_key_padding_mask=mask)
        ref_output = perm_fn(torch.tensor([[[2.301516, 0.092249, -0.679101, 0.103088]],
                                            [[2.301516, 0.092249, -0.679101, 0.103088]]], device=device, dtype=dtype))
        self.assertEqual(result.shape, ref_output.shape)
        torch.testing.assert_close(result, ref_output, atol=atol, rtol=rtol)

        # deterministic input
        encoder_input = perm_fn(torch.tensor([[[0.7462, 0.6653, 0.5679, 0.4891],
                                                [0.5387, 0.1655, 0.3565, 0.0471]],
                                                [[0.8335, 0.2799, 0.5031, 0.2947],
                                                [0.1402, 0.0318, 0.7636, 0.1346]],
                                                [[0.6333, 0.9344, 0.1376, 0.9938],
                                                [0.8924, 0.2872, 0.6692, 0.2944]],
                                                [[0.9897, 0.6915, 0.3154, 0.1733],
                                                [0.8645, 0.3513, 0.3064, 0.0767]],
                                                [[0.8117, 0.2366, 0.4838, 0.7881],
                                                [0.3718, 0.4945, 0.9511, 0.0864]]], device=device, dtype=dtype))
        result = model(encoder_input)
        ref_output = perm_fn(torch.tensor([[[2.428589, 0.020835, -0.602055, -0.085249],
                                            [2.427987, 0.021213, -0.602496, -0.084103]],
                                            [[2.424689, 0.019155, -0.604793, -0.085672],
                                            [2.413863, 0.022211, -0.612486, -0.072490]],
                                            [[2.433774, 0.021598, -0.598343, -0.087548],
                                            [2.425104, 0.019748, -0.604515, -0.084839]],
                                            [[2.436185, 0.022682, -0.596625, -0.087261],
                                            [2.433556, 0.021891, -0.598509, -0.086832]],
                                            [[2.416246, 0.017512, -0.610712, -0.082961],
                                            [2.422901, 0.024187, -0.606178, -0.074929]]], device=device, dtype=dtype))
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
        ref_output = perm_fn(torch.tensor([[[2.429026, 0.020793, -0.601741, -0.085642],
                                            [2.428811, 0.021445, -0.601912, -0.084252]],
                                            [[2.425009, 0.019155, -0.604566, -0.085899],
                                            [2.415408, 0.02249 , -0.611415, -0.073]],
                                            [[2.434199, 0.021682, -0.598039, -0.087699],
                                            [2.42598, 0.019941, -0.603896, -0.085091]],
                                            [[2.436457, 0.022736, -0.59643 , -0.08736],
                                            [2.434021, 0.022093, -0.598179, -0.08679]],
                                            [[2.416531, 0.017498, -0.610513, -0.083181],
                                            [2.4242, 0.024653, -0.605266, -0.074959]]], device=device, dtype=dtype))
        self.assertEqual(result.shape, ref_output.shape)
        torch.testing.assert_close(result, ref_output, atol=atol, rtol=rtol)

        # NestedTensor is only supported for the fast path
        # currently, which won't be used if training.
        if (batch_first and not training and
                ('xpu' in str(device) or 'cpu' in str(device)) and not TEST_WITH_CROSSREF):
            encoder_input[0][-1] = torch.zeros_like(encoder_input[0][1])
            mask = torch.zeros(encoder_input.shape[:-1], device=device, dtype=torch.bool)
            mask[0][-1] = True

            nt = torch.nested.nested_tensor([encoder_input[0][:-1], encoder_input[1]], device=device)
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
                device=device, dtype=dtype
            )
            result = result.to_padded_tensor(0)
            ref_output[0][-1] = torch.zeros_like(
                ref_output[0][-1], device=device, dtype=dtype
            )
            result[0][-1] = torch.zeros_like(
                result[0][-1], device=device, dtype=dtype
            )
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            if 'xpu' in device:
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
TestNNDeviceType.test_transformerencoderlayer=_test_transformerencoderlayer

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

        model = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                            activation, batch_first=batch_first, device=device, dtype=dtype)
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
        encoder_input = torch.tensor([[[20., 30., 40., 50.]]], device=device, dtype=dtype)
        result = model(encoder_input)
        ref_output = torch.tensor([[[2.249815, 0.131006, -0.702199, 0.177868]]], device=device, dtype=dtype)
        torch.testing.assert_close(result, ref_output, rtol=rtol, atol=atol)

        # deterministic input
        encoder_input = perm_fn(torch.tensor([[[1., 2., 3., 4.]],
                                                [[5., 6., 7., 8.]]], device=device, dtype=dtype))
        result = model(encoder_input)
        ref_output = perm_fn(torch.tensor([[[2.264103, 0.121417, -0.696012, 0.159724]],
                                            [[2.264103, 0.121417, -0.696012, 0.159724]]], device=device, dtype=dtype))
        torch.testing.assert_close(result, ref_output, rtol=rtol, atol=atol)

        # deterministic input
        encoder_input = perm_fn(torch.tensor([[[0.7462, 0.6653, 0.5679, 0.4891],
                                                [0.5387, 0.1655, 0.3565, 0.0471]],
                                                [[0.8335, 0.2799, 0.5031, 0.2947],
                                                [0.1402, 0.0318, 0.7636, 0.1346]],
                                                [[0.6333, 0.9344, 0.1376, 0.9938],
                                                [0.8924, 0.2872, 0.6692, 0.2944]],
                                                [[0.9897, 0.6915, 0.3154, 0.1733],
                                                [0.8645, 0.3513, 0.3064, 0.0767]],
                                                [[0.8117, 0.2366, 0.4838, 0.7881],
                                                [0.3718, 0.4945, 0.9511, 0.0864]]], device=device, dtype=dtype))
        result = model(encoder_input)
        ref_output = perm_fn(torch.tensor([[[2.42163188, 0.03227153, -0.60714219, -0.05908082],
                                            [2.42151276, 0.03302179, -0.60722523, -0.05762651]],
                                            [[2.41926761, 0.02974034, -0.60879519, -0.0621269],
                                            [2.41626395, 0.03539356, -0.61087842, -0.04978623]],
                                            [[2.42382808, 0.03218872, -0.6055963, -0.06073591],
                                            [2.41983477, 0.03085259, -0.60840145, -0.06046414]],
                                            [[2.42500749, 0.03328855, -0.60476388, -0.0595334],
                                            [2.4237977, 0.03290575, -0.60561789, -0.05940082]],
                                            [[2.41383916, 0.02686345, -0.61256377, -0.06380707],
                                            [2.42000277, 0.03800944, -0.60824798, -0.04754947]]], device=device, dtype=dtype))
        torch.testing.assert_close(result, ref_output, rtol=rtol, atol=atol)
    for activation, batch_first, training in product(('gelu', F.gelu, nn.GELU()), (True, False), (True, False)):
        # Fast path requires inference mode.
        if training:
            cm = contextlib.nullcontext()
        else:
            cm = torch.no_grad()
        with cm:
            _test(activation=activation, batch_first=batch_first, training=training)
TestNNDeviceType.test_transformerencoderlayer_gelu = _test_transformerencoderlayer_gelu

def _test_grid_sample_large(self, device):
    def issue_35202():
        input_tensor = torch.rand(1, 1, 480, 640, dtype=torch.float, device=device, requires_grad=True)
        coords = torch.tensor([[-10059144, 67680944], [67680944, 67680944]], dtype=torch.float, device=device)
        coords = coords.unsqueeze(0).unsqueeze(0).repeat(1, 1, 1, 1)
        result = torch.nn.functional.grid_sample(input_tensor, coords)
        self.assertEqual(result, torch.tensor([[[[0., 0.]]]], dtype=torch.float, device=device))
        result.backward(torch.ones_like(result))
        torch.xpu.synchronize()
    issue_35202()

    def issue_24823_1(dtype):
        image = torch.arange(27, 0, -1, dtype=dtype, device=device).view(1, 1, 3, 3, 3)
        image.requires_grad_()
        grid = torch.nn.functional.affine_grid(
            torch.tensor([[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]], dtype=dtype, device=device),
            (1, 1, 3, 3, 3))
        grid[:, 1, 1, 1, 0] = float('inf')
        result = torch.nn.functional.grid_sample(image, grid, padding_mode='zeros')
        tol_override = {'atol': 0.005, 'rtol': 0} if dtype == torch.half else {}
        self.assertEqual(result, torch.tensor([[[[[27., 26., 25.], [24., 23., 22.], [21., 20., 19.]],
                                                    [[18., 17., 16.], [15., 0., 13.], [12., 11., 10.]],
                                                    [[9., 8., 7.], [6., 5., 4.], [3., 2., 1.]]]]],
                                                device=device, dtype=dtype), **tol_override)
        result.backward(torch.ones_like(result))
        expected_grad = torch.ones_like(image)
        expected_grad[0, 0, 1, 1, 1] = 0
        self.assertEqual(image.grad, expected_grad, atol=0.005, rtol=0)
    # grid_sampler_3d is not supported in xpu
    # issue_24823_1(torch.half)
    # issue_24823_1(torch.float)
    # issue_24823_1(torch.double)

    def issue_24823_2():
        param = torch.tensor([[[-1.0e+20, 0.0, 0.0], [0.0, -1.0e+20, 0.0]]], dtype=torch.float, device=device)
        img = torch.zeros((1, 1, 4, 4), dtype=torch.float, device=device, requires_grad=True)
        grid = torch.nn.functional.affine_grid(param, img.size())
        result = torch.nn.functional.grid_sample(img, grid)
        self.assertEqual(result, torch.zeros(1, 1, 4, 4, device=device, dtype=torch.float))
        result.backward(torch.ones_like(result))
        torch.xpu.synchronize()
    issue_24823_2()
TestNNDeviceType.test_grid_sample_large=_test_grid_sample_large

def _test_grid_sample_half_precision(self):
    def helper(shape_in, shape_out, align_corners):
        for mode in ('bilinear', 'nearest', 'bicubic'):
            if len(shape_in) != 4 and mode == 'bicubic':
                continue
            data = torch.randn(shape_in, device='xpu', dtype=torch.half)
            grid = torch.rand(shape_out, device='xpu', dtype=torch.half) * 2.0 - 1.0

            out_half = F.grid_sample(data, grid, mode=mode, padding_mode='zeros', align_corners=align_corners)
            out_double = F.grid_sample(data.double(), grid.double(), mode=mode, padding_mode='zeros',
                                        align_corners=align_corners)

            self.assertEqual(out_half, out_double.half(), msg=f"grid_sample with mode = {mode} doesn't match")

    helper((32, 64, 16, 16), (32, 8, 8, 2), True)
    # helper((32, 64, 16, 16, 16), (32, 8, 8, 8, 3), True) # grid_sampler_3d is not supported in xpu
    helper((32, 64, 16, 16), (32, 8, 8, 2), False)
    # helper((32, 64, 16, 16, 16), (32, 8, 8, 8, 3), False) # grid_sampler_3d is not supported in xpu
TestNNDeviceType.test_grid_sample_half_precision=_test_grid_sample_half_precision

def _test_grid_sample_bfloat16_precision(self):
    def helper(shape_in, shape_out, align_corners):
        for mode in ('bilinear', 'nearest', 'bicubic'):
            if len(shape_in) != 4 and mode == 'bicubic':
                continue
            data = torch.randn(shape_in, device='xpu', dtype=torch.bfloat16)
            grid = torch.rand(shape_out, device='xpu', dtype=torch.bfloat16) * 2.0 - 1.0

            out_half = F.grid_sample(data, grid, mode=mode, padding_mode='zeros', align_corners=align_corners)
            out_double = F.grid_sample(data.double(), grid.double(), mode=mode, padding_mode='zeros',
                                        align_corners=align_corners)

            self.assertEqual(out_half, out_double.bfloat16(), msg=f"grid_sample with mode = {mode} doesn't match")

    helper((32, 64, 16, 16), (32, 8, 8, 2), True)
    # helper((32, 64, 16, 16, 16), (32, 8, 8, 8, 3), True) # grid_sampler_3d is not supported in xpu
    helper((32, 64, 16, 16), (32, 8, 8, 2), False)
    # helper((32, 64, 16, 16, 16), (32, 8, 8, 8, 3), False) # grid_sampler_3d is not supported in xpu
TestNNDeviceType.test_grid_sample_bfloat16_precision=_test_grid_sample_bfloat16_precision

@parametrize_test("antialias", [True, False])
@parametrize_test("align_corners", [True, False])
@parametrize_test("mode", ["bilinear", "bicubic"])
@parametrize_test("memory_format", [torch.contiguous_format, torch.channels_last])
def upsamplingBiMode2d(self, device, antialias, align_corners, mode, memory_format):
    # Forward AD does not support XLA because XLA tensors don't have storage
    check_forward_ad = torch.device(device).type != 'xla'

    kwargs = dict(mode=mode, align_corners=align_corners, antialias=antialias)
    # test float scale factor up & downsampling
    for scale_factor in [0.5, 1.5, 2]:
        in_t = torch.ones(
            2, 3, 8, 8, device=device,
            dtype=torch.double).contiguous(memory_format=memory_format).requires_grad_()
        out_size = int(math.floor(in_t.shape[-1] * scale_factor))
        with warnings.catch_warnings(record=True) as w:
            out_t = F.interpolate(in_t, scale_factor=scale_factor, **kwargs)
        expected_out = torch.ones(2, 3, out_size, out_size, device=device, dtype=torch.double)
        self.assertEqual(expected_out, out_t)
        # Assert that memory format is carried through to the output
        self.assertTrue(out_t.is_contiguous(memory_format=memory_format))
        out_t.backward(torch.randn_like(out_t))
        self.assertTrue(in_t.grad.is_contiguous(memory_format=memory_format))

        if torch.device(device).type == 'xpu':
            # Bilinear backward is nondeterministic because of atomicAdd usage
            nondet_tol = 1e-5
        else:
            nondet_tol = 0.0

        input = torch.randn(
            2, 3, 8, 8, device=device,
            dtype=torch.double).contiguous(memory_format=memory_format).requires_grad_()
        gradcheck(
            lambda x: F.interpolate(x, out_size, **kwargs),
            [input],
            check_forward_ad=check_forward_ad, nondet_tol=nondet_tol
        )
        gradgradcheck(
            lambda x: F.interpolate(x, out_size, **kwargs),
            [input],
            check_fwd_over_rev=check_forward_ad, nondet_tol=nondet_tol
        )

        # Assert that cpu and cuda give same results
        if torch.device(device).type == 'xpu':
            for shapes in [
                (2, 2, 3, 4), (2, 3, 4, 5), (3, 1, 2, 2), (1, 5, 3, 2)
            ]:
                a_xpu = torch.randn(
                    *shapes, device=device, dtype=torch.double
                ).contiguous(memory_format=memory_format).requires_grad_()
                a_cpu = a_xpu.detach().cpu().requires_grad_()

                with warnings.catch_warnings(record=True):
                    out_xpu = F.interpolate(a_xpu, scale_factor=scale_factor, **kwargs)
                    out_cpu = F.interpolate(a_cpu, scale_factor=scale_factor, **kwargs)

                self.assertEqual(out_cpu, out_xpu.cpu())

                g_cuda = torch.randn_like(out_xpu)
                g_cpu = g_cuda.cpu()

                out_xpu.backward(g_cuda)
                out_cpu.backward(g_cpu)

                self.assertEqual(a_xpu.grad, a_cpu.grad)
TestNNDeviceType.test_upsamplingBiMode2d = upsamplingBiMode2d

@dtypes(torch.float16, torch.float32)
def _test_cross_entropy_loss_2d_out_of_bounds_class_index(self, device, dtype):
    from torch.testing._internal.common_utils import TestCase
    # Test for issue #117532
    # Run in a different process to prevent the device-side assert from affecting other tests
    stderr = TestCase.runWithPytorchAPIUsageStderr(f"""\
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
        """)
    self.assertIn('Assertion `cur_target >= 0 && cur_target < n_classes` failed', stderr)
TestNNDeviceType.test_cross_entropy_loss_2d_out_of_bounds_class_index = _test_cross_entropy_loss_2d_out_of_bounds_class_index

instantiate_device_type_tests(
    TestNNDeviceType, globals(), only_for="xpu", allow_xpu=True
)
instantiate_parametrized_tests(TestNN)


if __name__ == "__main__":
    run_tests()
