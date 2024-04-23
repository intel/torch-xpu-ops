import torch
from torch.testing._internal.common_utils import TestCase

cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")

# TODO: Low percision data type has accuracy issue on key_grad
floating_types = [torch.float32]
shapes = [[16, 16, 512, 512, 64], [1, 4, 384, 384, 64]]

class Dtypes(object):
    def __init__(self, include_dtypes, exclude_dtypes=[]):
        self.include_dtypes = include_dtypes
        self.exclude_dtypes = exclude_dtypes

    def __call__(self, fn):
        def fn_out(*args, **kwargs):
            for dtype in self.include_dtypes:
                if dtype in self.exclude_dtypes:
                    continue
                kwargs['dtype'] = dtype
                fn(*args, **kwargs)
        return fn_out

class Shapes(object):
    def __init__(self, shapes):
        self.shapes = shapes

    def __call__(self, fn):
        def fn_out(*args, **kwargs):
            for shape in self.shapes:
                kwargs['shape'] = shape
                fn(*args, **kwargs)
        return fn_out

class TestSDPBackward(TestCase):
    def _test_scale_dot_product_attention_backward(self, dtype, shape, is_causal=False, dropout_p=0.5):
        batch_size = shape[0]
        n_heads = shape[1]
        seq_len_q = shape[2]
        seq_len_k = shape[3]
        head_dim = shape[4] 
        scale = 1 / head_dim
        def drop_mask_gen(p):
            mask = torch.rand(batch_size, n_heads, seq_len_q, seq_len_k, device=cpu_device)
            mask = mask < p
            mask = mask.to(torch.float32)
            return mask

        query = torch.rand(batch_size, n_heads, seq_len_q, head_dim,
                           device=xpu_device, dtype=dtype, requires_grad=True)
        key = torch.rand(batch_size, n_heads, seq_len_k, head_dim, device=xpu_device,
                         dtype=dtype, requires_grad=True)
        value = torch.rand(batch_size, n_heads, seq_len_k, head_dim,
                           device=xpu_device, dtype=dtype, requires_grad=True)
        attn_mask = torch.rand(seq_len_q, seq_len_k, device=xpu_device, dtype=dtype, requires_grad=True) if is_causal==False else None

        query_cpu = query.detach().clone().to("cpu").requires_grad_()
        key_cpu = key.detach().clone().to("cpu").requires_grad_()
        value_cpu = value.detach().clone().to("cpu").requires_grad_()
        attn_mask_cpu = attn_mask.detach().clone().to("cpu").requires_grad_() if is_causal==False else None
        
        dropout_mask_cpu = drop_mask_gen(dropout_p)
        dropout_mask = dropout_mask_cpu.to("xpu")

        out = torch.ops.aten._scaled_dot_product_attention_math(
                query, key, value, attn_mask, dropout_p=dropout_p, is_causal=is_causal,
                scale=scale, dropout_mask=dropout_mask)[0]
        
        out_cpu = torch.ops.aten._scaled_dot_product_attention_math(
                query_cpu, key_cpu, value_cpu, attn_mask_cpu, dropout_p=dropout_p, is_causal=is_causal,
                scale=scale, dropout_mask=dropout_mask_cpu)[0]

        upstream_grad = torch.rand_like(out, requires_grad=False)
        upstream_grad_cpu = upstream_grad.clone().to("cpu")

        out.backward(upstream_grad)
        out_cpu.backward(upstream_grad_cpu)
        
        self.assertEqual(query.grad, query_cpu.grad, atol=1e-5, rtol=1e-5)
        self.assertEqual(key.grad, key_cpu.grad, atol=1e-5, rtol=1e-5)
        self.assertEqual(value.grad, value_cpu.grad, atol=1e-5, rtol=1e-5)

    @Dtypes(floating_types)
    @Shapes(shapes)
    def test_sdp_backward(self, dtype, shape):
        self._test_scale_dot_product_attention_backward(dtype, shape)

    @Dtypes(floating_types)
    @Shapes(shapes)
    def test_sdp_backward_with_causal(self, dtype, shape):
        self._test_scale_dot_product_attention_backward(dtype, shape, True)