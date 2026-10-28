# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""_fft_c2c in complex32 and complex64."""

from ..case import TableBench, torch, with_dtypes

CASES = [
    dict(shape=(64, 512, 768), dim=(1, 2), normalization=0, forward=True),
]


class FftC2c(TableBench):
    op = "_fft_c2c"
    family = "fft"
    expected = {"fwd": ("_fft_c2c",)}
    params = with_dtypes(CASES, ("c32", "c64"))
    pr = [dict(CASES[0], dtype="c64")]

    def fwd(self, s):
        return torch.ops.aten._fft_c2c(s.x, s.dim, s.normalization, s.forward)
