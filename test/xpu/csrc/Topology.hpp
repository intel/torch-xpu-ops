// Topology.hpp
//
// Two-layer node/device topology for hierarchical collectives.
//
//   Layer A (TopologyInfo): purely COMPUTED from the launcher env vars
//     (WORLD_SIZE / LOCAL_WORLD_SIZE / RANK / LOCAL_RANK). It only knows how
//     ranks map to (node_id, local_rank) and is communication-free.
//
//   Layer B (NodeTopology): PHYSICAL properties of the local node that cannot
//     be derived from env vars -- per-device NUMA node, PCIe switch group and
//     nearest NIC. These are probed from hwloc (see Topology.cpp).
//
// Combined in `Topology`, which routes cross-rank queries through Layer A so
// that physical relations are only consulted for intra-node peers.

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

// ===========================================================================
// Layer A -- computed rank layout.
// ===========================================================================
struct TopologyInfo {
  int world_size = 0;
  int local_world_size = 0;
  int rank = 0;
  int local_rank = 0;

  int num_nodes = 0;
  int node_id = 0;

  static TopologyInfo make(
      int world_size, int local_world_size, int rank, int local_rank);
  static TopologyInfo from_env();

  bool is_first_node() const { return node_id == 0; }
  bool is_last_node() const { return node_id == num_nodes - 1; }
  bool is_node_leader() const { return local_rank == 0; }
  bool is_global_root() const { return rank == 0; }
  bool is_single_node() const { return num_nodes == 1; }

  int node_of(int r) const { return r / local_world_size; }
  int local_rank_of(int r) const { return r % local_world_size; }
  bool same_node(int a, int b) const { return node_of(a) == node_of(b); }
  bool is_intra_node(int peer) const { return same_node(rank, peer); }

  int global_rank(int node, int lr) const {
    return node * local_world_size + lr;
  }
  int node_local_to_global(int lr) const { return global_rank(node_id, lr); }
};

// ===========================================================================
// Layer B -- physical per-device topology (hwloc-probed).
// ===========================================================================

// Per local device (indexed by local_rank) physical placement.
struct DeviceTopo {
  int local_rank = -1;
  int numa_node = -1;    // NUMA node id (-1 == unknown)
  int pcie_switch = -1;  // PCIe switch/bridge group id (-1 == unknown)
  int nearest_nic = -1;  // index into NodeTopology::nics (-1 == unknown)
  std::string pci_bdf;   // e.g. "0000:18:00.0"
};

// A NIC discovered on the node.
struct NicTopo {
  int index = -1;
  int numa_node = -1;
  std::string name;      // e.g. "mlx5_0"
  std::string pci_bdf;
};

struct NodeTopology {
  int local_world_size = 0;
  std::vector<DeviceTopo> devices;  // indexed by local_rank
  std::vector<NicTopo> nics;

  enum class Proximity {
    SameSwitch,  // fastest P2P
    SameNuma,    // cross-switch, same NUMA
    SameNode,    // cross-NUMA (over UPI)
    Unknown,
  };

  const DeviceTopo& dev(int lr) const { return devices.at(lr); }

  bool same_numa(int a, int b) const {
    int na = devices.at(a).numa_node, nb = devices.at(b).numa_node;
    return na >= 0 && na == nb;
  }
  bool same_switch(int a, int b) const {
    int sa = devices.at(a).pcie_switch, sb = devices.at(b).pcie_switch;
    return sa >= 0 && sa == sb;
  }
  int nearest_nic(int lr) const { return devices.at(lr).nearest_nic; }
  bool share_nic(int a, int b) const {
    int na = devices.at(a).nearest_nic, nb = devices.at(b).nearest_nic;
    return na >= 0 && na == nb;
  }

  Proximity proximity(int a, int b) const {
    if (same_switch(a, b)) return Proximity::SameSwitch;
    if (same_numa(a, b)) return Proximity::SameNuma;
    if (devices.at(a).numa_node >= 0 && devices.at(b).numa_node >= 0)
      return Proximity::SameNode;
    return Proximity::Unknown;
  }

  // ---- group listings (return LOCAL ranks) ----
  // All local ranks (optionally including `self`).
  std::vector<int> node_local_ranks(int self, bool include_self = true) const {
    std::vector<int> out;
    for (int lr = 0; lr < local_world_size; ++lr)
      if (include_self || lr != self) out.push_back(lr);
    return out;
  }
  // Local ranks under the same PCIe switch as `self`.
  std::vector<int> same_switch_local_ranks(
      int self, bool include_self = true) const {
    std::vector<int> out;
    for (int lr = 0; lr < local_world_size; ++lr) {
      if (lr == self) {
        if (include_self) out.push_back(lr);
      } else if (same_switch(self, lr)) {
        out.push_back(lr);
      }
    }
    return out;
  }
  // Local ranks on the same NUMA node as `self`.
  std::vector<int> same_numa_local_ranks(
      int self, bool include_self = true) const {
    std::vector<int> out;
    for (int lr = 0; lr < local_world_size; ++lr) {
      if (lr == self) {
        if (include_self) out.push_back(lr);
      } else if (same_numa(self, lr)) {
        out.push_back(lr);
      }
    }
    return out;
  }

  // Probe the local node topology from hwloc. `device_bdfs` must list the PCI
  // BDF of every local device ordered by local_rank (size == local_world_size).
  // Use device_bdf_from_sycl() below to obtain each entry.
  static NodeTopology probe(const std::vector<std::string>& device_bdfs);

  // Assert the switch ⊂ numa containment invariant; throws on violation.
  void validate() const;

  std::string describe() const;  // human-readable dump for logging
};

// ===========================================================================
// Combined view.
// ===========================================================================

// Physical link a transfer should use.
enum class Transport {
  PCIe,  // intra-node peer access (Xe-Link / P2P over PCIe)
  NIC,   // through the (nearest) NIC -- RDMA / ISHMEM put
};

// Per-proximity transport selection. same-switch peers are ALWAYS PCIe (there
// is no reason to route a same-switch transfer through a NIC), so it is exposed
// as a non-configurable constant while the other levels are env-overridable and
// default to PCIe.
struct TransportPolicy {
  // Always PCIe -- kept for completeness so all four proximity levels are
  // explicit; not configurable.
  static constexpr Transport same_switch = Transport::PCIe;

  Transport same_numa = Transport::PCIe;   // cross-switch, same NUMA
  Transport same_node = Transport::PCIe;   // cross-NUMA, same node
  Transport cross_node = Transport::NIC;   // different node (always NIC)

  // Build from env overrides. Value "nic" -> NIC, anything else -> PCIe.
  //   TOPO_TRANSPORT_SAME_NUMA, TOPO_TRANSPORT_SAME_NODE
  static TransportPolicy from_env();
};

struct Topology {
  TopologyInfo layout;
  NodeTopology node;
  TransportPolicy policy;

  // One-shot initialization: reads the launcher env vars (Layer A), enumerates
  // the local GPUs' PCI BDFs via SYCL and probes their physical placement with
  // hwloc (Layer B), reads the transport policy env overrides, and validates
  // the switch/numa containment invariant. This is the API to "get everything".
  //   validate == true  -> throw on a containment violation
  static Topology from_env(bool validate = true);

  // Cross-rank queries take GLOBAL ranks; physical relations are only
  // meaningful (and only consulted) for same-node peers.
  bool same_switch(int ga, int gb) const {
    return layout.same_node(ga, gb) &&
        node.same_switch(layout.local_rank_of(ga), layout.local_rank_of(gb));
  }
  bool same_numa(int ga, int gb) const {
    return layout.same_node(ga, gb) &&
        node.same_numa(layout.local_rank_of(ga), layout.local_rank_of(gb));
  }
  int my_nic() const { return node.nearest_nic(layout.local_rank); }

  // Which link to use to reach `peer` (a GLOBAL rank) from the current device.
  //   same switch  -> PCIe          (always)
  //   same NUMA    -> policy.same_numa   (default PCIe)
  //   same node    -> policy.same_node   (default PCIe)
  //   other node   -> policy.cross_node  (NIC)
  Transport transport_to(int peer) const {
    if (!layout.same_node(layout.rank, peer)) return policy.cross_node;
    if (same_switch(layout.rank, peer)) return TransportPolicy::same_switch;
    if (same_numa(layout.rank, peer)) return policy.same_numa;
    return policy.same_node;
  }

  // ---- group listings for the CURRENT device (return GLOBAL ranks) ----
  // Global ranks on the same node as us (our node's block of ranks).
  std::vector<int> same_node_ranks(bool include_self = true) const {
    std::vector<int> out;
    for (int lr : node.node_local_ranks(layout.local_rank, include_self))
      out.push_back(layout.node_local_to_global(lr));
    return out;
  }
  // Global ranks under the same PCIe switch as us.
  std::vector<int> same_switch_ranks(bool include_self = true) const {
    std::vector<int> out;
    for (int lr :
         node.same_switch_local_ranks(layout.local_rank, include_self))
      out.push_back(layout.node_local_to_global(lr));
    return out;
  }
  // Global ranks on the same NUMA node as us.
  std::vector<int> same_numa_ranks(bool include_self = true) const {
    std::vector<int> out;
    for (int lr : node.same_numa_local_ranks(layout.local_rank, include_self))
      out.push_back(layout.node_local_to_global(lr));
    return out;
  }
};

// Query the PCI BDF ("0000:18:00.0") of the SYCL/Level-Zero device with the
// given global device index. Returns empty string if unavailable.
std::string device_bdf_from_sycl(int device_index);
