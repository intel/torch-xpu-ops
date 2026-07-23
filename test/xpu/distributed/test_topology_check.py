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
