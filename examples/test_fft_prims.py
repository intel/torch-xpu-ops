import torch
from torch.testing._internal.common_utils import TestCase
import itertools

device = torch.device("xpu")

class TestTorchMethod(TestCase):
    def test_fft_r2c(self):
        test_dtypes = [torch.float, torch.double]

        def test_one_fft_r2c(dtype):
            size = (2, 3, 40, 40)
            input_ref = torch.randn(size, dtype=dtype, device="cpu")
            input = input_ref.to(device)

            res = torch._fft_r2c(input, (2, 3), normalization=0, onesided=True)
            res_ref = torch._fft_r2c(input_ref, (2, 3), normalization=0, onesided=True)
            self.assertEqual(res, res_ref)

            out = torch.empty_like(res)
            torch._fft_r2c(input, (2, 3), normalization=0, onesided=True, out=out)
            out_ref = torch.empty_like(res_ref)
            torch._fft_r2c(input_ref, (2, 3), normalization=0, onesided=True, out=out_ref)
            self.assertEqual(out, out_ref)

        for dtype in test_dtypes:
            test_one_fft_r2c(dtype)

    def test_fft_c2r(self):
        test_dtypes = [torch.cfloat, torch.cdouble]        

        def test_one_fft_c2r(dtype):
            size = (2, 3, 40, 40)

            input_ref = torch.randn(size, dtype=dtype, device="cpu")
            input = input_ref.to(device)

            res = torch._fft_c2r(input, (2, 3), normalization=0, last_dim_size=0)
            res_ref = torch._fft_c2r(input_ref, (2, 3), normalization=0, last_dim_size=0)
            self.assertEqual(res, res_ref)

            out = torch.empty_like(res)
            torch._fft_c2r(input, (2, 3), normalization=0, last_dim_size=0, out=out)
            out_ref = torch.empty_like(res_ref)
            torch._fft_c2r(input_ref, (2, 3), normalization=0, last_dim_size=0, out=out_ref)
            self.assertEqual(out, out_ref)
        
        for dtype in test_dtypes:
            test_one_fft_c2r(dtype)

    def test_fft_c2c(self):
        test_dtypes = [torch.cfloat, torch.cdouble]

        def test_one_fft_c2c(dtype):
            for fwd in [True, False] :
                
                size = (2, 3, 40, 40)

                input_ref = torch.randn(size, dtype=dtype, device="cpu")
                input = input_ref.to(device)

                res = torch._fft_c2c(input, (2, 3), normalization=0, forward=fwd)
                res_ref = torch._fft_c2c(input_ref, (2, 3), normalization=0, forward=fwd)
                self.assertEqual(res, res_ref)

                out = torch.empty_like(res)
                torch._fft_c2c(input, (2, 3), normalization=0, forward=fwd, out=out)
                out_ref = torch.empty_like(res_ref)
                torch._fft_c2c(input_ref, (2, 3), normalization=0, forward=fwd, out=out_ref)
                self.assertEqual(out, out_ref)
        
        for dtype in test_dtypes:
            test_one_fft_c2c(dtype)
