"""Standalone test for the hwloc-backed topology check (xpu_topology module).

Loads the pybind11 module `xpu_topology` (built by build.py), probes the local
node topology, validates the switch/numa containment invariant and PRINTS the
full result: per-device NUMA / PCIe-switch / nearest-NIC, discovered NICs, and
the same-node / same-switch / same-numa rank groups plus the chosen transport
for every peer.

Run (single process, enumerates all local GPUs):
    python test_topology_check.py

Run under a launcher (uses RANK / LOCAL_RANK / WORLD_SIZE / LOCAL_WORLD_SIZE):
    mpirun -np 4 --prepend-rank python test_topology_check.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CSRC = os.path.join(_HERE, "..", "csrc")
# Make the built xpu_topology<ext>.so importable.
if _CSRC not in sys.path:
    sys.path.insert(0, _CSRC)

import torch  # noqa: F401  (ensures the torch runtime / SYCL is loaded)

try:
    import xpu_topology as topo
except ImportError as exc:  # pragma: no cover - build guidance
    raise SystemExit(
        f"could not import xpu_topology ({exc}); build it first with "
        f"`python {os.path.join(_CSRC, 'build.py')}`"
    )


def _env_defaults():
    """Fill env so single-process runs still work (1 node, all local GPUs)."""
    ngpu = torch.xpu.device_count() if torch.xpu.is_available() else 1
    os.environ.setdefault("WORLD_SIZE", str(ngpu))
    os.environ.setdefault("LOCAL_WORLD_SIZE", os.environ["WORLD_SIZE"])
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")


def _transport_name(t):
    return "PCIe" if t == topo.Transport.PCIe else "NIC"


def main():
    _env_defaults()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # One-shot: read env + enumerate local GPU BDFs + hwloc probe. Validate the
    # switch ⊂ numa containment invariant (raises on violation).
    t = topo.Topology.from_env(validate=True)

    lines = []
    lines.append(f"===== topology check (rank {rank}/{world_size}) =====")
    layout = t.layout
    lines.append(
        f"layout: world_size={layout.world_size} "
        f"local_world_size={layout.local_world_size} "
        f"num_nodes={layout.num_nodes} node_id={layout.node_id} "
        f"local_rank={layout.local_rank}"
    )

    # Physical node topology dump (per-device numa/switch/nic + NIC list).
    lines.append(t.describe().rstrip("\n"))

    # ---- machine-wide grouping over ALL local cards (not just this rank) ----
    devices = list(t.node.devices)
    nics = list(t.node.nics)

    def _group_by(key):
        groups = {}
        for d in devices:
            groups.setdefault(key(d), []).append(d.local_rank)
        # stable order by first member
        return dict(sorted(groups.items(), key=lambda kv: min(kv[1])))

    switch_groups = _group_by(lambda d: d.pcie_switch)
    numa_groups = _group_by(lambda d: d.numa_node)

    lines.append("---- all-card grouping (local ranks) ----")
    lines.append(
        "numa groups     : "
        + "  ".join(f"numa{numa}={ranks}" for numa, ranks in numa_groups.items())
    )
    lines.append(
        "switch groups   : "
        + "  ".join(f"sw{sw}={ranks}" for sw, ranks in switch_groups.items())
    )

    # ---- per-card NIC (name + bdf) for every card on this machine ----
    def _nic_str(idx):
        if idx is None or idx < 0 or idx >= len(nics):
            return "none"
        n = nics[idx]
        return f"nic{n.index}:{n.name} ({n.pci_bdf}, numa{n.numa_node})"

    lines.append("---- per-card NIC ----")
    for d in devices:
        lines.append(
            f"card {d.local_rank} (bdf={d.pci_bdf}, numa{d.numa_node}) "
            f"-> {_nic_str(d.nearest_nic)}"
        )

    # Rank groups for the current device.
    lines.append(f"same_node_ranks   = {t.same_node_ranks()}")
    lines.append(f"same_switch_ranks = {t.same_switch_ranks()}")
    lines.append(f"same_numa_ranks   = {t.same_numa_ranks()}")
    lines.append(f"my_nic            = {t.my_nic()}")

    # Transport decision for every other global rank.
    decisions = []
    for peer in range(world_size):
        if peer == rank:
            continue
        decisions.append(f"{peer}:{_transport_name(t.transport_to(peer))}")
    lines.append("transport_to      = " + ", ".join(decisions))

    print("\n".join(lines) + "\n", flush=True)

    # ---- lightweight sanity assertions ----
    node_ranks = set(t.same_node_ranks())
    switch_ranks = set(t.same_switch_ranks())
    numa_ranks = set(t.same_numa_ranks())
    # Containment: switch ⊆ numa ⊆ node.
    assert switch_ranks <= numa_ranks <= node_ranks, (
        f"containment violated: switch={switch_ranks} numa={numa_ranks} "
        f"node={node_ranks}"
    )
    # Self is always in every group.
    assert rank in switch_ranks and rank in numa_ranks and rank in node_ranks
    # same-switch peers must resolve to PCIe transport.
    for peer in switch_ranks:
        if peer != rank:
            assert t.transport_to(peer) == topo.Transport.PCIe

    print(f"[rank {rank}] topology check PASSED", flush=True)


if __name__ == "__main__":
    main()
