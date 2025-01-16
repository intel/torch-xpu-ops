# Owner(s): ["module: intel"]

import os
import sys
import unittest

import torch
from torch import multiprocessing as mp
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests
from torch.utils.data import DataLoader, IterDataPipe
from torch.utils.data.datapipes.iter import IterableWrapper

try:
    from xpu_test_utils import XPUPatchForImport
except Exception as e:
    from .xpu_test_utils import XPUPatchForImport


test_package = (
    os.path.dirname(os.path.abspath(__file__)) + "/../../../../test",
    os.path.dirname(os.path.abspath(__file__)) + "/../../../../test/nn",
)


with XPUPatchForImport(False):

    def _set_allocator_settings(device=None):
        pass

    torch.cuda.memory._set_allocator_settings = _set_allocator_settings
    from test_dataloader import (
        _clone_collate,
        collate_into_packed_sequence,
        collate_into_packed_sequence_batch_first,
        collate_wrapper,
        filter_len,
        row_processor,
        self_module,
        supported_multiprocessing_contexts,
        TEST_CUDA_IPC,
        TestCustomPinFn,
        TestDataLoader,
        TestDataLoaderDeviceType,
        TestDictDataLoader,
        TestStringDataLoader,
    )

    def _test_multiprocessing_iterdatapipe(self, with_dill):
        # Testing to make sure that function from global scope (e.g. imported from library) can be serialized
        # and used with multiprocess DataLoader

        reference = [
            torch.as_tensor([[2, 3, 4, 5]], dtype=torch.int64),
            torch.as_tensor([[2, 3, 4, 5]], dtype=torch.int64),
        ]
        datapipe: IterDataPipe = IterableWrapper([[1, 2, 3, 4], [1, 2, 3, 4, 5, 6]])
        datapipe = datapipe.map(row_processor)
        datapipe = (
            datapipe.filter(lambda row: len(row) == 4)
            if with_dill
            else datapipe.filter(filter_len)
        )

        dl_common_args = dict(
            num_workers=2, batch_size=2, shuffle=True, pin_memory=False
        )
        for ctx in supported_multiprocessing_contexts:
            self.assertEqual(
                reference,
                [
                    t.type(torch.int64)
                    for t in self._get_data_loader(
                        datapipe, multiprocessing_context=ctx, **dl_common_args
                    )
                ],
            )
            if ctx is not None:
                # test ctx object
                ctx = mp.get_context(ctx)
                self.assertEqual(
                    reference,
                    [
                        t.type(torch.int64)
                        for t in self._get_data_loader(
                            datapipe, multiprocessing_context=ctx, **dl_common_args
                        )
                    ],
                )

    def sequential_pin_memory(self):
        loader = self._get_data_loader(self.dataset, batch_size=2, pin_memory=True)
        for input, target in loader:
            self.assertTrue(input.is_pinned())
            self.assertTrue(target.is_pinned())

    def shuffle_pin_memory(self):
        loader = self._get_data_loader(
            self.dataset, batch_size=2, shuffle=True, num_workers=4, pin_memory=True
        )
        for input, target in loader:
            self.assertTrue(input.is_pinned())
            self.assertTrue(target.is_pinned())

    def string_shuffle_pin_memory(self):
        loader = DataLoader(
            self.dataset, batch_size=2, shuffle=True, num_workers=4, pin_memory=True
        )
        for s, n in loader:
            self.assertIsInstance(s[0], str)
            self.assertTrue(n.is_pinned())

    def pin_memory(self):
        loader = DataLoader(self.dataset, batch_size=2, pin_memory=True)
        for sample in loader:
            self.assertTrue(sample["a_tensor"].is_pinned())
            self.assertTrue(sample["another_dict"]["a_number"].is_pinned())

    def pin_memory_device(self):
        loader = DataLoader(
            self.dataset, batch_size=2, pin_memory=True, pin_memory_device="xpu"
        )
        for sample in loader:
            self.assertTrue(sample["a_tensor"].is_pinned(device="xpu"))
            self.assertTrue(sample["another_dict"]["a_number"].is_pinned(device="xpu"))

    def pin_memory_with_only_device(self):
        loader = DataLoader(self.dataset, batch_size=2, pin_memory_device="xpu")
        for sample in loader:
            self.assertFalse(sample["a_tensor"].is_pinned(device="xpu"))
            self.assertFalse(sample["another_dict"]["a_number"].is_pinned(device="xpu"))

    def custom_batch_pin(self):
        test_cases = [
            (collate_wrapper, self_module.SimpleCustomBatch),
            (collate_into_packed_sequence, torch.nn.utils.rnn.PackedSequence),
            (
                collate_into_packed_sequence_batch_first,
                torch.nn.utils.rnn.PackedSequence,
            ),
        ]
        for collate_fn, elem_cls in test_cases:
            loader = DataLoader(
                self.dataset, batch_size=2, collate_fn=collate_fn, pin_memory=True
            )
            for sample in loader:
                self.assertIsInstance(sample, elem_cls)
                self.assertTrue(sample.is_pinned())

    def custom_batch_pin_worker(self):
        test_cases = [
            (collate_wrapper, self_module.SimpleCustomBatch),
            (collate_into_packed_sequence, torch.nn.utils.rnn.PackedSequence),
            (
                collate_into_packed_sequence_batch_first,
                torch.nn.utils.rnn.PackedSequence,
            ),
        ]
        for collate_fn, elem_cls in test_cases:
            loader = DataLoader(
                self.dataset,
                batch_size=2,
                collate_fn=collate_fn,
                pin_memory=True,
                num_workers=1,
            )
            for sample in loader:
                self.assertIsInstance(sample, elem_cls)
                self.assertTrue(sample.is_pinned())

    @parametrize(
        "context",
        [ctx for ctx in supported_multiprocessing_contexts if ctx is not None],
    )
    @unittest.skipIf(not TEST_CUDA_IPC, "CUDA IPC not available")
    def nested_tensor_multiprocessing(self, device, context):
        # The 'fork' multiprocessing context doesn't work for CUDA so skip it
        if "xpu" in device and context == "fork":
            # TODO: Skip this better in a better way when the test framework allows
            return

        dataset = [
            torch.nested.nested_tensor([torch.randn(5)], device=device)
            for _ in range(10)
        ]

        pin_memory_settings = [False]
        if device == "cpu" and torch.xpu.is_available():
            pin_memory_settings.append(True)

        for pin_memory in pin_memory_settings:
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=1,
                num_workers=4,
                collate_fn=_clone_collate,
                pin_memory=pin_memory,
                multiprocessing_context=context,
            )

            for i, batch in enumerate(loader):
                self.assertEqual(batch[0], dataset[i])

        # Error case: default collate_fn doesn't currently support batches of nested tensors.
        # Following the current semantics, we'd need to stack them, which isn't possible atm.
        with self.assertRaisesRegex(
            RuntimeError, "not currently supported by the default collate_fn"
        ):
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=1,
                num_workers=4,
                multiprocessing_context=context,
            )

            next(iter(loader))

    TestDataLoader._test_multiprocessing_iterdatapipe = (
        _test_multiprocessing_iterdatapipe
    )
    TestDataLoader.test_sequential_pin_memory = sequential_pin_memory
    TestDataLoader.test_shuffle_pin_memory = shuffle_pin_memory
    TestStringDataLoader.test_shuffle_pin_memory = string_shuffle_pin_memory
    TestDictDataLoader.test_pin_memory = pin_memory
    TestDictDataLoader.test_pin_memory_device = pin_memory_device
    TestDictDataLoader.test_pin_memory_with_only_device = pin_memory_with_only_device
    TestCustomPinFn.test_custom_batch_pin = custom_batch_pin
    TestDataLoaderDeviceType.test_nested_tensor_multiprocessing = (
        nested_tensor_multiprocessing
    )


instantiate_device_type_tests(
    TestDataLoaderDeviceType, globals(), only_for="xpu", allow_xpu=True
)
original_path = sys.path.copy()
sys.path.extend(test_package)


if __name__ == "__main__":
    run_tests()
