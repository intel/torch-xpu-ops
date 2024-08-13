# Owner(s): ["module: intel"]

import numpy as np
import torch

from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)

from torch.testing._internal.common_dtype import all_types_and, floating_types_and
from torch.testing._internal.common_utils import run_tests

try:
    from xpu_test_utils import XPUPatchForImport
except Exception as e:
    from .xpu_test_utils import XPUPatchForImport

with XPUPatchForImport(False):
    from test_sort_and_select import TestSortAndSelect

    # FIXME: remove torch.bool from unsupported types once support is added for cub sort
    @dtypes(*all_types_and(torch.half, torch.bfloat16))
    def stable_sort_against_numpy(self, device, dtype):
        if dtype in floating_types_and(torch.float16, torch.bfloat16):
            inf = float("inf")
            neg_inf = -float("inf")
            nan = float("nan")
        else:
            if dtype != torch.bool:
                # no torch.iinfo support for torch.bool
                inf = torch.iinfo(dtype).max
                neg_inf = torch.iinfo(dtype).min
            else:
                inf = True
                neg_inf = ~inf
            # no nan for integral types, we use inf instead for simplicity
            nan = inf

        def generate_samples():
            from itertools import chain, combinations

            for sizes in [(1025,), (10000,)]:
                size = sizes[0]
                # binary strings
                yield (torch.tensor([0, 1] * size, dtype=dtype, device=device), 0)

            if self.device_type in ["cuda", "xpu"]:
                return

            yield (torch.tensor([0, 1] * 100, dtype=dtype, device=device), 0)

            def repeated_index_fill(t, dim, idxs, vals):
                res = t
                for idx, val in zip(idxs, vals):
                    res = res.index_fill(dim, idx, val)
                return res

            for sizes in [(1, 10), (10, 1), (10, 10), (10, 10, 10)]:
                size = min(*sizes)
                x = (torch.randn(*sizes, device=device) * size).to(dtype)
                yield (x, 0)

                # Generate tensors which are being filled at random locations
                # with values from the non-empty subsets of the set (inf, neg_inf, nan)
                # for each dimension.
                n_fill_vals = 3  # cardinality of (inf, neg_inf, nan)
                for dim in range(len(sizes)):
                    idxs = (
                        torch.randint(high=size, size=(size // 10,))
                        for i in range(n_fill_vals)
                    )
                    vals = (inf, neg_inf, nan)
                    subsets = chain.from_iterable(
                        combinations(list(zip(idxs, vals)), r)
                        for r in range(1, n_fill_vals + 1)
                    )
                    for subset in subsets:
                        idxs_subset, vals_subset = zip(*subset)
                        yield (
                            repeated_index_fill(x, dim, idxs_subset, vals_subset),
                            dim,
                        )

        for sample, dim in generate_samples():
            _, idx_torch = sample.sort(dim=dim, stable=True)
            if dtype is torch.bfloat16:
                sample_numpy = sample.float().cpu().numpy()
            else:
                sample_numpy = sample.cpu().numpy()
            idx_numpy = np.argsort(sample_numpy, axis=dim, kind="stable")
            self.assertEqual(idx_torch, idx_numpy)

    TestSortAndSelect.test_stable_sort_against_numpy = stable_sort_against_numpy

instantiate_device_type_tests(
    TestSortAndSelect, globals(), only_for="xpu", allow_xpu=True
)


if __name__ == "__main__":
    run_tests()
