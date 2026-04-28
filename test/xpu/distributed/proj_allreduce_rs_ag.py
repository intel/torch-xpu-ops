#!/usr/bin/env python3
"""
AllReduce roofline projection (ws=4, bf16) using the same constants as
sycl-tla/examples/00_bmg_gemm/projection.py.

Model: AllReduce = ReduceScatter + AllGather (ring)
    RS(S, N) = S / MemBW + (S/N / N_uni + N_lat) * (N-1)
    AG(S, N) = (S   / N_uni + N_lat) * (N-1)   # AG input = S/N shard
    AR(S, N) = RS(S, N) + AG(S/N, N)

Also reports one_shot pure-bandwidth lower bound: (N-1) * S / N_uni,
which is the PCIe-bound floor of our current one_shot kernel.

Both numbers are pure algorithm/hardware rooflines; they do NOT include
torch-op dispatcher (~13 us) or signal-barrier (~5 us each) overhead.
"""

# ---- constants (copied from projection.py) ---------------------------------
PCIE_DISCOUNT = 0.6957
N_UNI = 31.5 * PCIE_DISCOUNT * (1 << 30)   # bytes/s, BMG P2P unidirectional
MEM_BW = 0.450 * 1e12                      # bytes/s, HBM
N_LAT = 3e-6                               # s, per P2P hop latency
N_P2P_UNI = N_UNI


def proj_allgather(S, N):
    return (S / N_P2P_UNI + N_LAT) * (N - 1)


def proj_reduce_scatter(S, N):
    size_p2p = S / N
    time_p2p = (size_p2p / N_P2P_UNI + N_LAT) * (N - 1)
    return S / MEM_BW + time_p2p


def proj_allreduce_rs_ag(S, N):
    rs = proj_reduce_scatter(S, N)
    ag = proj_allgather(S / N, N)
    return rs, ag, rs + ag


def fmt_bytes(b):
    for u, f in [("MiB", 1 << 20), ("KiB", 1 << 10)]:
        if b >= f:
            return f"{b/f:6.1f} {u}"
    return f"{b:6d} B"


def main():
    N = 4
    dsz = 2  # bf16
    # Measured (docker ws=4 bf16, WARMUP=10 ITERS=50, USE_SIGNAL_BARRIER=1).
    # one_shot: our fused kernel, see instruction.md §12.6.
    # two_shot: our XPU two_shot_all_reduce_ (host-barrier'd), §12.6.
    # ccl:      oneCCL allreduce, docker baseline (§12.6 / §12.8).
    measured = {
        # numel  : (one_shot_us, two_shot_us, ccl_us)
        1024:     (26.5,  113.7,   17.3),
        4096:     (31.4,  112.0,   21.0),
        16384:    (31.0,  116.5,   21.7),
        65536:    (30.7,  112.8,   24.3),
        262144:   (83.7,  114.0,   52.0),
        1048576:  (265.7, 194.0,  166.4),
        4194304:  (1025.7,1054.3, 618.2),
    }

    print(f"AllReduce = ReduceScatter + AllGather, ws={N}, bf16")
    print(f"  N_uni = {N_UNI/(1<<30):.2f} GiB/s   MemBW = {MEM_BW/1e9:.0f} GB/s   "
          f"N_lat = {N_LAT*1e6:.1f} us   pcie_discount = {PCIE_DISCOUNT}")
    print()
    hdr = (f"{'size':>10s}  {'RS us':>8s}  {'AG us':>8s}  {'AR us':>8s}  "
           f"{'1shot_bw':>9s}    "
           f"{'ours 1shot':>10s}  {'ours 2shot':>10s}  {'ccl':>8s}   "
           f"{'1shot−AR':>9s}  {'ccl−AR':>7s}")
    print(hdr)
    print("-" * len(hdr))
    for k in range(10, 22):
        numel = 1 << k
        S = numel * dsz
        rs, ag, ar = proj_allreduce_rs_ag(S, N)
        bw_oneshot = (N - 1) * S / N_UNI * 1e6

        m = measured.get(numel)
        if m is None:
            one_s, two_s, ccl_s = "", "", ""
            d1, d2 = "", ""
        else:
            one_us, two_us, ccl_us = m
            one_s = f"{one_us:10.1f}"
            two_s = f"{two_us:10.1f}"
            ccl_s = f"{ccl_us:8.1f}"
            d1 = f"{one_us - ar*1e6:+9.1f}"
            d2 = f"{ccl_us - ar*1e6:+7.1f}"

        print(f"{fmt_bytes(S):>10s}  {rs*1e6:8.2f}  {ag*1e6:8.2f}  {ar*1e6:8.2f}  "
              f"{bw_oneshot:9.2f}    "
              f"{one_s:>10s}  {two_s:>10s}  {ccl_s:>8s}   "
              f"{d1:>9s}  {d2:>7s}")
    print()
    print("Legend:")
    print("  RS/AG/AR     : ring reduce-scatter / all-gather / AR=RS+AG projection (us)")
    print("  1shot_bw     : (N-1)*S/N_uni, pure-PCIe floor of our one_shot (us)")
    print("  ours 1shot   : measured one_shot_all_reduce (fused) in docker, §12.6 (us)")
    print("  ours 2shot   : measured two_shot_all_reduce_ in docker, §12.6 (us)")
    print("  ccl          : measured oneCCL allreduce in docker, §12.6 (us)")
    print("  Δ = measured - AR projection (positive = slower than roofline)")


if __name__ == "__main__":
    main()
