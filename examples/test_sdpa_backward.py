import torch
from torch.testing._internal.common_utils import TestCase

cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")

class TestSDPBackward(TestCase):
    
    def test_scale_dot_product_attention_backward(self, device, batch_size: int, seq_len_q: int, seq_len_k: int,
                                                       head_dim: int, is_causal: bool, dropout_p: float, dtype: torch.dtype,
                                                       scale: str):
        def _get_mem_eff_drop_mask(batch_size, n_heads, q_len, kv_len, p, seed, offset):
            mask = torch.empty((batch_size, n_heads, q_len, kv_len), device=cpu_device, dtype=torch.float32)
            rand_uniform = torch._fill_mem_eff_dropout_mask_(mask, p, seed, offset)
            mask = (rand_uniform > p).to(torch.float32)
            return mask
        
        seed = 42
        scale = scale if scale is None else (1 / head_dim)
        n_heads = 4
        query = torch.rand(batch_size, n_heads, seq_len_q, head_dim,
                           device=xpu_device, dtype=dtype, requires_grad=True)
        key = torch.rand(batch_size, n_heads, seq_len_k, head_dim, device=xpu_device,
                         dtype=dtype, requires_grad=True)
        value = torch.rand(batch_size, n_heads, seq_len_k, head_dim,
                           device=xpu_device, dtype=dtype, requires_grad=True)
        attn_mask = torch.rand(seq_len_q, seq_len_k, device=xpu_device, dtype=dtype, requires_grad=True)

        query_cpu = query.to("cpu")
        key_cpu = key.to("cpu")
        value_cpu = value.to("cpu")

        attn_mask_cpu = attn_mask.to("cpu")

        dropout_mask_cpu = _get_mem_eff_drop_mask(batch_size, n_heads, seq_len_q, seq_len_k, dropout_p, seed, 0)
        dropout_mask = dropout_mask_cpu.to("xpu")

        out = torch.ops.aten._scaled_dot_product_attention_math(
                query, key, value, attn_mask, dropout_p=dropout_p, is_causal=is_causal,
                scale=scale, dropout_mask=dropout_mask)[0]
        
        out_cpu = torch.ops.aten._scaled_dot_product_attention_math(
                query_cpu, key_cpu, value_cpu, attn_mask_cpu, dropout_p=dropout_p, is_causal=is_causal,
                scale=scale, dropout_mask=dropout_mask_cpu)[0]

        upstream_grad = torch.rand_like(out, requires_grad=False)
        upstream_grad_cpu = upstream_grad.to("cpu")

        out.backward(upstream_grad)
        out_cpu.backward(upstream_grad_cpu)

        self.assertEqual(query.grad, query_cpu.grad)
        self.assertEqual(key.grad, key_cpu.grad)
        self.assertEqual(value.grad, value_cpu.grad)

