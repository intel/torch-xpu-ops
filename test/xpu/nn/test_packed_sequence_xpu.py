# Owner(s): ["module: intel"]

from torch.testing._internal.common_utils import run_tests

try:
    from .xpu_test_utils import XPUPatchForImport
except Exception as e:
    from ..xpu_test_utils import XPUPatchForImport

with XPUPatchForImport(False):
    import types

    import torch
    import torch.nn.utils.rnn as rnn_utils
    from test_packed_sequence import PackedSequenceTest

    def myxpu(self, *args, **kwargs):
        ex = torch.tensor((), dtype=self.data.dtype, device=self.data.device).to(
            *args, **kwargs
        )
        if ex.device.type == "xpu":
            return self.to(*args, **kwargs)
        return self.to(*args, device="xpu", **kwargs)

    rnn_utils.PackedSequence.xpu = types.MethodType(rnn_utils.PackedSequence, myxpu)

    def my_test_to(self):
        for enforce_sorted in (True, False):
            padded, lengths = self._padded_sequence(torch.IntTensor)
            a = rnn_utils.pack_padded_sequence(
                padded, lengths, enforce_sorted=enforce_sorted
            ).cpu()

            self.assertIs(a, a.to("cpu"))
            self.assertIs(a, a.cpu())
            self.assertIs(a, a.to("cpu", dtype=torch.int32))
            self.assertEqual(a.long(), a.to(torch.int64))

            if torch.cuda.is_available():
                for cuda in [
                    "cuda",
                    "cuda:0" if torch.cuda.device_count() == 1 else "cuda:1",
                ]:
                    b = a.cuda(device=cuda)
                    self.assertIs(b, b.to(cuda))
                    self.assertIs(b, b.cuda())
                    self.assertEqual(a, b.to("cpu"))
                    self.assertEqual(b, a.to(cuda))
                    self.assertEqual(a, b.to("cpu", dtype=torch.int32))
                    self.assertIs(b, b.to(dtype=torch.int32))
                    self.assertEqual(b.long(), b.to(dtype=torch.int64))

            if torch.xpu.is_available():
                for xpu in [
                    "xpu",
                    "xpu:0" if torch.xpu.device_count() == 1 else "xpu:1",
                ]:
                    b = a.xpu()
                    self.assertIs(b, b.to(xpu))
                    self.assertIs(b, b.xpu())
                    self.assertEqual(a, b.to("cpu"))
                    self.assertEqual(b, a.to(xpu))
                    self.assertEqual(a, b.to("cpu", dtype=torch.int32))
                    self.assertIs(b, b.to(dtype=torch.int32))
                    self.assertEqual(b.long(), b.to(dtype=torch.int64))

    PackedSequenceTest.test_to = my_test_to

if __name__ == "__main__":
    run_tests()
