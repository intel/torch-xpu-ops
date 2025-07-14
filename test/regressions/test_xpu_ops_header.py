# Owner(s): ["module: intel"]
import os

import torch
from torch.testing._internal.common_utils import TestCase


class TestXpuOpsHeader(TestCase):
    def test_xpu_ops_header(self):
        include_dir = os.path.join(os.path.dirname(torch.__file__), "include")
        aten_ops_dir = os.path.join(include_dir, "ATen/ops")
        self.assertTrue(
            os.path.exists(os.path.join(aten_ops_dir, "cat_xpu_dispatch.h"))
        )
        self.assertTrue(
            os.path.exists(os.path.join(aten_ops_dir, "index_fill_xpu_dispatch.h"))
        )
        self.assertTrue(os.path.exists(os.path.join(aten_ops_dir, "col2im_native.h")))
        with open(os.path.join(aten_ops_dir, "col2im_native.h")) as fr:
            text = fr.read()
            self.assertTrue("col2im_xpu" in text)
