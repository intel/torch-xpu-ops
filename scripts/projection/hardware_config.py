from global_config import *
import os

pcie_discount = 0.6957
#pcie_discount = 1
compute_discount = 1
mem_bandwidth_discount = float(os.environ.get('MEM_BW_EFFICIENCY', 1))
#print(mem_bandwidth_discount)

T_fp8 = 117 * 1000 * 1000 * 1000 * 1000 * 2 / 2
T_bf16 = 117 * 1000 * 1000 * 1000 * 1000 / 2
V_bf16 = 28 * 1000 * 1000 * 1000 * 1000 / 2
V_fp32 = V_bf16 / 2
W_bf16 = 32
W_fp32 = W_bf16 / 2
B = 0.450 * 1000 * 1000 * 1000 * 1000
magic_m = 16 # Minimum m to reach full TFLOPS w/o considering memory access
magic_exp = 20 # Cycles in exp (latency)
magic_exp_eff = 0.25 # Throughput efficiency in exp
N_uni = 31.5 * pcie_discount * 1024 * 1024 * 1024
N_p2p_uni = -1
N_lat = 3 / 1000 / 1000
N_inter_node_uni = 100 / 8 * 1024 * 1024 * 1024
N_inter_node_p2p_uni = N_inter_node_uni
N_inter_node_lat = 8 / 1000 / 1000
mem_capacity = 12
#llc_capacity = 18 * 1024 * 1024
TVP = True

if Config.device.lower() == "crie":
    T_fp8 = 100 * 1000 * 1000 * 1000 * 1000 * 2 / 2
    T_bf16 = 100 * 1000 * 1000 * 1000 * 1000 / 2
    V_bf16 = 25 * 1000 * 1000 * 1000 * 1000 / 2
    V_fp32 = V_bf16 / 2
    B = 1.4 * 1000 * 1000 * 1000 * 1000
    magic_m = 16 # Minimum m to reach full TFLOPS w/o considering memory access
    magic_exp = 2 # Cycles in exp
    N_uni = 63 * 1024 * 1024 * 1024 * pcie_discount
    N_lat = 3 / 1000 / 1000
    mem_capacity = 256

if Config.device.lower() == "cri":
    T_peak = 655
    T_power = 459
    T = T_peak
    T_fp8 = T * 1000 * 1000 * 1000 * 1000 * 2 / 2
    T_bf16 = T * 1000 * 1000 * 1000 * 1000 / 2
    V_bf16 = T_bf16 / 4 * 1000 * 1000 * 1000 * 1000 / 2
    V_fp32 = V_bf16 / 2
    B = 1.536 * 1000 * 1000 * 1000 * 1000
    magic_m = 16 # Minimum m to reach full TFLOPS w/o considering memory access
    magic_exp = 2 # Cycles in exp
    N_uni = 63 * 1024 * 1024 * 1024 * pcie_discount
    N_lat = 3 / 1000 / 1000
    mem_capacity = 480

if Config.device.lower() == "h20":
    T_fp8 = 148 * 1000 * 1000 * 1000 * 1000 * 2 / 2
    T_bf16 = 148 * 1000 * 1000 * 1000 * 1000 / 2
    V_bf16 = 44 * 1000 * 1000 * 1000 * 1000 / 2
    V_fp32 = V_bf16
    W_bf16 = 32
    W_fp32 = W_bf16
    B = 4 * 1000 * 1000 * 1000 * 1000
    magic_m = 16 # Minimum m to reach full TFLOPS w/o considering memory access
    magic_exp = 2 # Cycles in exp
    N_uni = 450 * 1024 * 1024 * 1024
    N_lat = 3 / 1000 / 1000
    mem_capacity = 96

if Config.device.lower() == "crie2":
    T_fp8 = 100 * 1000 * 1000 * 1000 * 1000 * 2 / 2
    T_bf16 = 100 * 1000 * 1000 * 1000 * 1000 / 2
    V_bf16 = 25 * 1000 * 1000 * 1000 * 1000 / 2
    V_fp32 = V_bf16 / 2
    B = 1.4 * 1000 * 1000 * 1000 * 1000
    magic_m = 16 # Minimum m to reach full TFLOPS w/o considering memory access
    magic_exp = 2 # Cycles in exp
    N_uni = 63 * 1024 * 1024 * 1024 * pcie_discount / 2
    N_lat = 3 / 1000 / 1000
    mem_capacity = 256


if Config.device.lower() == "4070tisuper":
    T_fp8 = 89 * 1000 * 1000 * 1000 * 1000 * 2 / 2
    T_bf16 = 89 * 1000 * 1000 * 1000 * 1000 / 2
    V_bf16 = 40 * 1000 * 1000 * 1000 * 1000 / 2
    V_fp32 = V_bf16
    W_bf16 = 32
    W_fp32 = W_bf16
    B = 0.642 * 1000 * 1000 * 1000 * 1000
    magic_m = 16 # Minimum m to reach full TFLOPS w/o considering memory access
    magic_exp = 2 # Cycles in exp
    N_uni = 31.5 * 1024 * 1024 * 1024
    N_lat = 3 / 1000 / 1000
    mem_capacity = 256


if Config.device.lower() == "4090d":
    T_fp8 = 148 * 1000 * 1000 * 1000 * 1000 * 2 / 2
    T_bf16 = 148 * 1000 * 1000 * 1000 * 1000 / 2
    V_bf16 = 73.5 * 1000 * 1000 * 1000 * 1000 / 2
    V_fp32 = V_bf16
    W_bf16 = 32
    W_fp32 = W_bf16
    B = 0.939 * 1000 * 1000 * 1000 * 1000
    magic_m = 16 # Minimum m to reach full TFLOPS w/o considering memory access
    magic_exp = 2 # Cycles in exp
    N_uni = 31.5 * 1024 * 1024 * 1024
    N_lat = 3 / 1000 / 1000
    mem_capacity = 256

if Config.device.lower() == "bmg24":
    T_fp8 = 98 * 1000 * 1000 * 1000 * 1000 * 2 / 2
    T_bf16 = 98 * 1000 * 1000 * 1000 * 1000 / 2    
    mem_capacity = 24

if Config.device.lower() == "bmginf":
    T_fp8 = 98 * 1000 * 1000 * 1000 * 1000 * 2 / 2
    T_bf16 = 98 * 1000 * 1000 * 1000 * 1000 / 2    
    mem_capacity = 1024

if Config.device.lower() == "ai100u":
    T_fp8 = 64 * 3 * 1000 * 1000 * 1000 * 1000 / 2
    T_bf16 = 64 * 1000 * 1000 * 1000 * 1000 / 2
    V_bf16 = T_bf16 / 8
    V_fp32 = V_bf16 / 2
    W_bf16 = 256
    W_fp32 = W_bf16 / 2
    B = 0.137 * 1000 * 1000 * 1000 * 1000
    magic_m = 32 # Minimum m to reach full TFLOPS w/o considering memory access
    pcie_discount = 1
    N_uni = 15.754 * 1024 * 1024 * 1024 * pcie_discount
    N_lat = 3 / 1000 / 1000
    mem_capacity = 32

if Config.device.lower() == "ai100p":
    T_fp8 = 112 * 3 * 1000 * 1000 * 1000 * 1000 / 2
    T_bf16 = 112 * 1000 * 1000 * 1000 * 1000 / 2
    V_bf16 = T_bf16 / 8
    V_fp32 = V_bf16 / 2
    W_bf16 = 256
    W_fp32 = W_bf16 / 2
    B = 0.137 * 1000 * 1000 * 1000 * 1000
    magic_m = 32 # Minimum m to reach full TFLOPS w/o considering memory access
    N_uni = 15.754 * 1024 * 1024 * 1024 * pcie_discount
    N_lat = 3 / 1000 / 1000
    mem_capacity = 32

if Config.device.lower() == "rtx6000b":
    T_fp8 = 1000 * 2 * 1000 * 1000 * 1000 * 1000 / 2
    T_bf16 = 1000 * 1000 * 1000 * 1000 * 1000 / 2
    V_bf16 = 110 * 1000 * 1000 * 1000 * 1000 / 2 # another version is 126
    V_fp32 = V_bf16 # same as bf16
    W_bf16 = 32
    W_fp32 = W_bf16 # same as bf16
    B = 1.792 * 1000 * 1000 * 1000 * 1000
    magic_m = 32 # Minimum m to reach full TFLOPS w/o considering memory access
    N_uni = 63 * 1024 * 1024 * 1024 * pcie_discount
    N_lat = 3 / 1000 / 1000
    mem_capacity = 96

if Config.device.lower() == "rtx6000d":
    T_fp8 = 149 * 2 * 1000 * 1000 * 1000 * 1000 / 2
    T_bf16 = 149 * 1000 * 1000 * 1000 * 1000 / 2
    V_bf16 = 37.5 * 1000 * 1000 * 1000 * 1000 / 2 # another version is 126
    V_fp32 = V_bf16 # same as bf16
    W_bf16 = 32
    W_fp32 = W_bf16 # same as bf16
    B = 1.344 * 1000 * 1000 * 1000 * 1000
    magic_m = 32 # Minimum m to reach full TFLOPS w/o considering memory access
    N_uni = 63 * 1024 * 1024 * 1024 * pcie_discount
    N_lat = 3 / 1000 / 1000
    mem_capacity = 84

if Config.device.lower() == "g2dp":
    TVP = True
    T_fp8 = 149 * 1000 * 1000 * 1000 * 1000 * 2 / 2
    T_bf16 = 149 * 1000 * 1000 * 1000 * 1000 / 2
    V_bf16 = 11 * 1000 * 1000 * 1000 * 1000 / 2
    V_fp32 = V_bf16 / 2
    B = 2.4 * 1000 * 1000 * 1000 * 1000
    magic_m = 48 # Minimum m to reach full TFLOPS w/o considering memory access
    magic_exp = 5 # Cycles in exp
    N_p2p_uni = 200 * 6 / 8 * 1024 * 1024 * 1024 / 2
    N_lat = 5 / 1000 / 1000
    N_inter_node_p2p_uni = 100 / 8 * 1024 * 1024 * 1024
    N_inter_node_lat = 8 / 1000 / 1000
    mem_capacity = 96

if Config.device.lower() == "g2do":
    TVP = True
    T_fp8 = 149 * 1000 * 1000 * 1000 * 1000 * 2 / 2
    T_bf16 = 149 * 1000 * 1000 * 1000 * 1000 / 2
    V_bf16 = 11 * 1000 * 1000 * 1000 * 1000 / 2
    V_fp32 = V_bf16 / 2
    B = 2.4 * 1000 * 1000 * 1000 * 1000
    magic_m = 48 # Minimum m to reach full TFLOPS w/o considering memory access
    magic_exp = 5 # Cycles in exp
    N_p2p_uni = 200 * 3 / 8 * 1024 * 1024 * 1024 / 2
    N_lat = 5 / 1000 / 1000
    N_inter_node_p2p_uni = 100 / 8 * 1024 * 1024 * 1024
    N_inter_node_lat = 8 / 1000 / 1000
    mem_capacity = 96

B *= mem_bandwidth_discount 
T_fp8 *= compute_discount
T_bf16 *= compute_discount
V_bf16 *= compute_discount
V_fp32 *= compute_discount