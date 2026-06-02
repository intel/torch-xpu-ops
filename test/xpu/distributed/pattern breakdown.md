# Pattern Breakdown

## 1. Original

### Layer 0

```text
01. hidden                                  # [full, H]
02. residual = hidden                       # [full, H]
03. hidden = norm(hidden)                   # [full, H] -> [full, H]
04. hidden = attn(hidden)                   # [full, H] -> [full, H]
05. hidden = allreduce(hidden)              # [full, H] -> [full, H]
06. residual = residual + hidden            # [full, H] + [full, H] -> [full, H]
07. hidden = norm(hidden)                   # [full, H] -> [full, H]
08. hidden = mlp(hidden)                    # [full, H] -> [full, H]
09. hidden = allreduce(hidden)              # [full, H] -> [full, H]
```

### Layer 1

```text
01. residual = residual + hidden            # [full, H] + [full, H] -> [full, H]
02. hidden = norm(hidden)                   # [full, H] -> [full, H]
03. hidden = attn(hidden)                   # [full, H] -> [full, H]
04. hidden = allreduce(hidden)              # [full, H] -> [full, H]
05. residual = residual + hidden            # [full, H] + [full, H] -> [full, H]
06. hidden = norm(hidden)                   # [full, H] -> [full, H]
07. hidden = mlp(hidden)                    # [full, H] -> [full, H]
08. hidden = allreduce(hidden)              # [full, H] -> [full, H]
```

## 2. Break allreduce into reducescatter + allgather

### Layer 0

```text
01. hidden                                  # [full, H]
02. residual = hidden                       # [full, H]
03. hidden = norm(hidden)                   # [full, H] -> [full, H]
04. hidden = attn(hidden)                   # [full, H] -> [full, H]
05. hidden = reducescatter_attn(hidden)     # [full, H] -> [full/TP, H]
06. hidden = allgather_attn(hidden)         # [full/TP, H] -> [full, H]
07. residual = residual + hidden            # [full, H] + [full, H] -> [full, H]
08. hidden = norm(hidden)                   # [full, H] -> [full, H]
09. hidden = mlp(hidden)                    # [full, H] -> [full, H]
10. hidden = reducescatter_mlp(hidden)      # [full, H] -> [full/TP, H]
11. hidden = allgather_mlp(hidden)          # [full/TP, H] -> [full, H]
```

### Layer 1

```text
01. residual = residual + hidden            # [full, H] + [full, H] -> [full, H]
02. hidden = norm(hidden)                   # [full, H] -> [full, H]
03. hidden = attn(hidden)                   # [full, H] -> [full, H]
04. hidden = reducescatter_attn(hidden)     # [full, H] -> [full/TP, H]
05. hidden = allgather_attn(hidden)         # [full/TP, H] -> [full, H]
06. residual = residual + hidden            # [full, H] + [full, H] -> [full, H]
07. hidden = norm(hidden)                   # [full, H] -> [full, H]
08. hidden = mlp(hidden)                    # [full, H] -> [full, H]
09. hidden = reducescatter_mlp(hidden)      # [full, H] -> [full/TP, H]
10. hidden = allgather_mlp(hidden)          # [full/TP, H] -> [full, H]
```

## 3. Pass down allgather across layers

### Layer 0

```text
01. hidden                                  # [full, H]
02. residual = hidden                       # [full, H]
03. hidden = norm(hidden)                   # [full, H] -> [full, H]
04. hidden = attn(hidden)                   # [full, H] -> [full, H]
05. hidden = reducescatter_attn_l0(hidden)  # [full, H] -> [full/TP, H]
06. <!-- hidden = allgather_attn_l0(hidden) # [full/TP, H] -> [full, H] -->
07. residual = residual + hidden            # [full/TP, H] + [full/TP, H] -> [full/TP, H]
08. hidden = norm(hidden)                   # [full/TP, H] -> [full/TP, H]
09. hidden = allgather_attn_l0(hidden)      # [full/TP, H] -> [full, H]
10. hidden = mlp(hidden)                    # [full, H] -> [full, H]
11. hidden = reducescatter_mlp_l0(hidden)   # [full, H] -> [full/TP, H]
12. <!-- hidden = allgather_mlp_l0(hidden) # [full/TP, H] -> [full/TP, H] -->
```

### Layer 1

```text
01. residual = residual + hidden            # [full/TP, H] + [full/TP, H] -> [full/TP, H]
02. hidden = norm(hidden)                   # [full/TP, H] -> [full/TP, H]
03. hidden = allgather_mlp_l0(hidden)       # [full/TP, H] -> [full, H]
04. hidden = attn(hidden)                   # [full, H] -> [full, H]
05. hidden = reducescatter_attn_l1(hidden)  # [full, H] -> [full/TP, H]
06. <!-- hidden = allgather_attn_l1(hidden) # [full/TP, H] -> [full, H] -->
07. residual = residual + hidden            # [full/TP, H] + [full/TP, H] -> [full/TP, H]
08. hidden = norm(hidden)                   # [full, H] -> [full, H]
09. hidden = allgather_attn_l1(hidden)      # [full/TP, H] -> [full, H]
10. hidden = mlp(hidden)                    # [full, H] -> [full, H]
11. hidden = reducescatter_mlp_l1(hidden)   # [full, H] -> [full/TP, H]
12. <!-- hidden = allgather_mlp(hidden) # [full/TP, H] -> [full/TP, H] --> move to next layers
```

## 4. Pass down allgather across layers with quantization

### Layer 0

```text
01. hidden                                  # [full, H]
02. residual = hidden                       # [full, H]
03. hidden = norm(hidden)                   # [full, H] -> [full, H]
04. hidden = attn(hidden)                   # [full, H] -> [full, H]
05. hidden = reducescatter_attn_l0(hidden)  # [full, H] -> [full/TP, H], **IN: 16bit, OUT: 16bit**
06. <!-- hidden = allgather_attn_l0(hidden) # [full/TP, H] -> [full, H] -->
07. residual = residual + hidden            # [full/TP, H] + [full/TP, H] -> [full/TP, H]
08. hidden = norm(hidden)                   # [full/TP, H] -> [full/TP, H]
09. hidden = quantize(hidden)               # [full/TP, H] -> [full/TP, H] **move MLP up proj quantization before allgather**
10. hidden = allgather_attn_l0(hidden)      # [full/TP, H] -> [full, H] **IN: 8bit, OUT: 8bit**
11. hidden = mlp(hidden)                    # [full, H] -> [full, H]
12. hidden = reducescatter_mlp_l0(hidden)   # [full, H] -> [full/TP, H]
13. <!-- hidden = allgather_mlp_l0(hidden) # [full/TP, H] -> [full, H] -->
```

### Layer 1

```text
01. residual = residual + hidden            # [full/TP, H] + [full/TP, H] -> [full/TP, H]
02. hidden = norm(hidden)                   # [full/TP, H] -> [full/TP, H]
03. hidden = quantize(hidden)               # [full/TP, H] -> [full/TP, H] **move attn QKV proj quantization before allgather**
04. hidden = allgather_mlp_l0(hidden)       # [full/TP, H] -> [full, H] **IN: 8bit, OUT: 8bit**
05. hidden = attn(hidden)                   # [full, H] -> [full, H]
06. hidden = reducescatter_attn_l1(hidden)  # [full, H] -> [full/TP, H] **IN: 16bit, OUT: 16bit**
07. <!-- hidden = allgather_attn_l1(hidden) # [full/TP, H] -> [full/TP, H] -->
08. residual = residual + hidden            # [full/TP, H] + [full/TP, H] -> [full/TP, H]
09. hidden = norm(hidden)                   # [full, H] -> [full, H]
10. hidden = allgather_attn_l1(hidden)      # [full/TP, H] -> [full, H] **move MLP up proj quantization before allgather**
11. hidden = mlp(hidden)                    # [full, H] -> [full, H] **IN: 8bit, OUT: 8bit**
12. hidden = reducescatter_mlp_l1(hidden)   # [full, H] -> [full/TP, H]
13. <!-- hidden = allgather_mlp(hidden) # [full/TP, H] -> [full/TP, H] --> move to next layers
```
