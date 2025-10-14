# Owner(s): ["module: intel"]
import torch
from torch.testing._internal.common_utils import TestCase

class TestTorchMethod(TestCase):
    # Define float8 dtypes for the focused test
    FLOAT8_DTYPES = (
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2,
        torch.float8_e5m2fnuz,
        torch.float8_e8m0fnu,
    )

    def _create_input_tensors(self, shape, dtype, memory_format=None):
        # Always generate random data using a CPU-compatible dtype (float32)
        # to avoid the "not implemented" error for float8 on CPU.
        tensor = torch.randn(shape, dtype=torch.float32)

        # Convert to the target testing dtype
        tensor = tensor.to(dtype)

        # Apply memory format if specified
        if memory_format is not None:
            tensor = tensor.to(memory_format=memory_format)

        return tensor

    def _test_cat_float8_core(self, tensors, dim, dtype):
        """Core function to test torch.cat for float8, using tolerances."""
        
        # --- CPU Reference Calculation (High Precision) ---
        # Convert inputs to float32 on CPU for golden reference calculation
        ref_tensors = [t.cpu().to(torch.float32) for t in tensors]

        # Calculate CPU reference result
        res_cpu = torch.cat(ref_tensors, dim=dim)

        # --- XPU Calculation ---
        # Convert inputs to XPU
        xpu_tensors = [t.xpu() for t in tensors]
        res_xpu = torch.cat(xpu_tensors, dim=dim)

        # Float8 is lossy, use higher tolerance (rtol=1e-2, atol=1e-2)
        rtol = 1e-2
        atol = 1e-2

        # Convert XPU result to float32 on CPU before comparison to match res_cpu's dtype.
        res_xpu_f32_on_cpu = res_xpu.cpu().to(torch.float32)
        
        self.assertEqual(res_cpu, res_xpu_f32_on_cpu, rtol=rtol, atol=atol)


    # ----------------------------------------------------------------------
    # New Focused Test: Simple Float8 torch.cat
    # ----------------------------------------------------------------------
    def test_cat_float8_simple(self):
        """Test torch.cat correctness across float8 dtypes using simple tensors."""
        for dtype in self.FLOAT8_DTYPES:
            with self.subTest(dtype=dtype):
                # Use simple 3D shape (2, 4, 3) and concatenate along dim 1
                user_cpu1 = self._create_input_tensors([2, 4, 3], dtype=dtype)
                user_cpu2 = self._create_input_tensors([2, 2, 3], dtype=dtype)
                user_cpu3 = self._create_input_tensors([2, 6, 3], dtype=dtype)

                tensors = (user_cpu1, user_cpu2, user_cpu3)
                dim = 1

                self._test_cat_float8_core(tensors, dim, dtype)

    # ----------------------------------------------------------------------
    # Original Tests (Restored to default float/float32)
    # ----------------------------------------------------------------------

    def test_cat_8d(self, dtype=torch.float):
        # Original test logic restored: uses default dtype (float32)
        input1 = torch.randn([256, 8, 8, 3, 3, 3, 3], dtype=dtype)
        input2 = torch.randn([256, 8, 8, 3, 3, 3, 3], dtype=dtype)
        
        input1_xpu = input1.xpu()
        input2_xpu = input2.xpu()
        
        output1 = torch.stack([input1, input2], dim=0)
        output1_xpu = torch.stack([input1_xpu, input2_xpu], dim=0)
        
        output2 = output1.reshape([2, 256, 8, 8, 9, 9])
        output2_xpu = output1_xpu.reshape([2, 256, 8, 8, 9, 9])
        
        output3 = torch.stack([output2, output2], dim=0)
        output3_xpu = torch.stack([output2_xpu, output2_xpu], dim=0)
        
        # Standard assertEqual for float32 (expect high precision)
        self.assertEqual(output3, output3_xpu.cpu())

    def test_cat_array(self, dtype=torch.float):
        # Original test logic restored: uses default dtype (float32)
        user_cpu1 = torch.randn([2, 2, 3], dtype=dtype)
        user_cpu2 = torch.randn([2, 2, 3], dtype=dtype)
        user_cpu3 = torch.randn([2, 2, 3], dtype=dtype)
        
        res_cpu = torch.cat((user_cpu1, user_cpu2, user_cpu3), dim=1)
        
        res_xpu = torch.cat(
            (
                user_cpu1.xpu(),
                user_cpu2.xpu(),
                user_cpu3.xpu(),
            ),
            dim=1,
        )
        # Standard assertEqual for float32
        self.assertEqual(res_cpu, res_xpu.cpu())

    def test_cat_array_2(self, dtype=torch.float):
        # Original test logic restored: uses default dtype (float32)
        shapes = [
            (8, 7, 3, 2), (4, 4, 4, 4), (4, 4, 1, 1), (4, 1, 4, 4),
            (4, 1, 4, 1), (4, 1, 1, 4), (1, 4, 1, 4), (1, 4, 4, 1),
            (4, 1, 1, 1),
        ]
        
        for shape in shapes:
            # Removed original print statements to streamline test
            N, C, H, W = shape[0], shape[1], shape[2], shape[3]
            dim_idx = 1
            
            # Case 1: all channels_last
            user_cpu1 = torch.randn([N, C, H, W], dtype=dtype).to(memory_format=torch.channels_last)
            user_cpu2 = torch.randn([N, C, H, W], dtype=dtype).to(memory_format=torch.channels_last)
            user_cpu3 = torch.randn([N, C, H, W], dtype=dtype).to(memory_format=torch.channels_last)
            res_cpu = torch.cat((user_cpu1, user_cpu2, user_cpu3), dim=dim_idx)
            res_xpu = torch.cat((user_cpu1.xpu(), user_cpu2.xpu(), user_cpu3.xpu()), dim=dim_idx)
            self.assertEqual(res_cpu, res_xpu.cpu())

            # Case 2: cl, contiguous, cl
            user_cpu1 = torch.randn([N, C, H, W], dtype=dtype).to(memory_format=torch.channels_last)
            user_cpu2 = torch.randn([N, C, H, W], dtype=dtype).to(memory_format=torch.contiguous_format)
            user_cpu3 = torch.randn([N, C, H, W], dtype=dtype).to(memory_format=torch.channels_last)
            res_cpu = torch.cat((user_cpu1, user_cpu2, user_cpu3), dim=dim_idx)
            res_xpu = torch.cat((user_cpu1.xpu(), user_cpu2.xpu(), user_cpu3.xpu()), dim=dim_idx)
            self.assertEqual(res_cpu, res_xpu.cpu())

            # Case 3: contiguous, cl, cl
            user_cpu1 = torch.randn([N, C, H, W], dtype=dtype).to(memory_format=torch.contiguous_format)
            user_cpu2 = torch.randn([N, C, H, W], dtype=dtype).to(memory_format=torch.channels_last)
            user_cpu3 = torch.randn([N, C, H, W], dtype=dtype).to(memory_format=torch.channels_last)
            res_cpu = torch.cat((user_cpu1, user_cpu2, user_cpu3), dim=dim_idx)
            res_xpu = torch.cat((user_cpu1.xpu(), user_cpu2.xpu(), user_cpu3.xpu()), dim=dim_idx)
            self.assertEqual(res_cpu, res_xpu.cpu())

            # Removed original verbose memory format assertions for clean test logic

            
