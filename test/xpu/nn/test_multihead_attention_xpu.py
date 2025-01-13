# Owner(s): ["module: intel"]

from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    run_tests,
)

try:
    from .xpu_test_utils import XPUPatchForImport
except Exception as e:
    from ..xpu_test_utils import XPUPatchForImport

with XPUPatchForImport(False):
    import unittest
    import unittest.mock as mock
    from typing import Optional

    import torch
    from test_multihead_attention import (
        TestMultiheadAttentionNN,
        TestMultiheadAttentionNNDeviceType,
    )
    from torch.testing._internal.common_utils import TEST_WITH_CROSSREF

    def _check_arg_device2(x: Optional[torch.Tensor]) -> bool:
        if x is not None:
            return x.device.type in [
                "cpu",
                "cuda",
                "xpu",
                torch.utils.backend_registration._privateuse1_backend_name,
            ]
        return True

    @torch.no_grad()
    @unittest.skipIf(
        TEST_WITH_CROSSREF,
        "CrossRef turns on TorchFunctionMode, and so disables fastpath.",
    )
    def multihead_self_attn_two_masks_fast_path_mock(self, device):
        """
        Multihead self-attention should take fast path when both attention mask (mask type 0)
        and key padding mask (mask type 1) are provided at the same time on CPU and CUDA and PrivateUse1
        """
        device = device.rstrip(":0123456789")
        if device not in [
            "cpu",
            "cuda",
            "xpu",
            torch._C._get_privateuse1_backend_name(),
        ]:
            self.skipTest("Fastpath only runs on CPU and CUDA and PrivateUse1.")

        with torch.autocast(device_type=device, enabled=False):
            embed_dim = 16
            num_heads = 8
            batch_size = 8
            src_len = 5

            query = value = key = torch.rand(batch_size, src_len, embed_dim).to(device)
            # Create masks of two different types
            attn_mask = torch.randint(0, 2, (src_len, src_len)).bool().to(device)
            key_padding_mask = (
                torch.randint(0, 2, (batch_size, src_len)).bool().to(device)
            )

            with mock.patch(
                "torch._native_multi_head_attention",
                new=mock.MagicMock(return_value=(torch.Tensor(), torch.Tensor())),
            ) as fastpath_mock:
                # Compute attention on the fast path
                mta_model = torch.nn.MultiheadAttention(
                    embed_dim, num_heads, batch_first=True, device=device
                ).eval()
                mta_model.training = False
                mta_model(
                    query,
                    key,
                    value,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                )
                # If mock was called, fastpath was taken
                self.assertTrue(fastpath_mock.called)

    TestMultiheadAttentionNNDeviceType.test_multihead_self_attn_two_masks_fast_path_mock = (
        multihead_self_attn_two_masks_fast_path_mock
    )
    torch.nn.modules.activation._check_arg_device = _check_arg_device2

instantiate_device_type_tests(
    TestMultiheadAttentionNNDeviceType, globals(), only_for="xpu", allow_xpu=True
)
instantiate_parametrized_tests(TestMultiheadAttentionNN)

if __name__ == "__main__":
    run_tests()
