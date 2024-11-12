import torch
from torch.testing._internal.common_utils import TestCase


class TestTorchMethod(TestCase):
    def test_record_stream(self):
        t = torch.FloatTensor([1, 2, 3, 4]).pin_memory()
        result = torch.FloatTensor(t.size()).to("xpu")
        stream = torch.xpu.Stream()
        ptr = [None]

        # Performs the CPU->GPU copy in a background stream
        def perform_copy():
            x = torch.randn(256, 1024, 1024, device="xpu")
            y = torch.randn(256, 1024, 1024, device="xpu")
            with torch.xpu.stream(stream):
                tmp = t.xpu(non_blocking=True)
                ptr[0] = tmp.data_ptr()
            torch.xpu.current_stream().wait_stream(stream)
            tmp.record_stream(torch.xpu.current_stream())
            for i in range(30):  # delay the copy
                z = x + y
            result.copy_(tmp)

        perform_copy()
        with torch.xpu.stream(stream):
            tmp2 = torch.FloatTensor(t.size()).to("xpu")
            tmp2.zero_()
            self.assertNotEqual(
                tmp2.data_ptr(), ptr[0], msg="allocation re-used to soon"
            )

        self.assertEqual(result.tolist(), [1, 2, 3, 4])

        # In the native allocator, we expect "tmp"'s side-stream-tagged block will be reused
        # in that side stream after result.copy_(tmp) in the main stream finishes.
        torch.xpu.current_stream().synchronize()
        with torch.xpu.stream(stream):
            tmp3 = torch.FloatTensor(t.size()).to("xpu")
            self.assertEqual(tmp3.data_ptr(), ptr[0], msg="allocation not re-used")

    def test_record_stream_on_shifted_view(self):
        # See PyTorch issue #27366
        # This test detects unexpected block reallocation. For reliable test,
        # the stream to allocate tensors is isolated. The allocator will not
        # reuse free blocks which were allocated from another stream.
        x = torch.randn(256, 1024, 1024, device="xpu")
        y = torch.randn(256, 1024, 1024, device="xpu")

        stream_alloc = torch.xpu.Stream()
        with torch.xpu.stream(stream_alloc):
            base = torch.FloatTensor([10, 10]).xpu()

        # Record another stream on a shifted view tensor.
        view = base[5:]
        self.assertTrue(view.storage_offset() > 0)

        stream_record = torch.xpu.Stream()
        with torch.xpu.stream(stream_record):
            for i in range(30):
                z = x+y

        view.record_stream(stream_record)

        # Delete those tensors to make the block free soon.
        data_ptr = base.data_ptr()
        del base, view

        # A new tensor should not be allocated to the block above.
        stream_alloc.synchronize()

        with torch.xpu.stream(stream_alloc):
            try_realloc = torch.FloatTensor([10, 10]).xpu()

        self.assertNotEqual(try_realloc.data_ptr(), data_ptr)
