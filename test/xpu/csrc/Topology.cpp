// Topology.cpp
//
// hwloc-backed implementation of the physical node topology declared in
// Topology.hpp. Every physical relation (NUMA node, PCIe switch group, nearest
// NIC) is discovered from the hwloc tree; the rank layout (Layer A) stays a
// pure computation.
//
// Data-source note: hwloc must be built/loaded with I/O discovery so PCI
// devices and NICs appear in the tree. We request that explicitly via
// hwloc_topology_set_io_types_filter(..., HWLOC_TYPE_FILTER_KEEP_IMPORTANT).

#include "Topology.hpp"

#include <hwloc.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <sstream>

// ---------------------------------------------------------------------------
// Layer A -- computed rank layout.
// ---------------------------------------------------------------------------
TopologyInfo TopologyInfo::make(int ws, int lws, int rank, int lrank) {
  if (lws <= 0 || ws <= 0)
    throw std::runtime_error("Topology: world/local size must be > 0");
  if (ws % lws != 0)
    throw std::runtime_error(
        "Topology: WORLD_SIZE must be divisible by LOCAL_WORLD_SIZE "
        "(non-uniform node layout not supported)");
  if (rank < 0 || rank >= ws)
    throw std::runtime_error("Topology: rank out of range");
  if (lrank < 0 || lrank >= lws)
    throw std::runtime_error("Topology: local_rank out of range");
  if (rank % lws != lrank)
    throw std::runtime_error(
        "Topology: LOCAL_RANK inconsistent with RANK/LOCAL_WORLD_SIZE");

  TopologyInfo t{};
  t.world_size = ws;
  t.local_world_size = lws;
  t.rank = rank;
  t.local_rank = lrank;
  t.num_nodes = ws / lws;
  t.node_id = rank / lws;
  return t;
}

static int env_int_or(const char* name, int fallback) {
  const char* v = std::getenv(name);
  if (v && *v) return std::atoi(v);
  return fallback;
}

TopologyInfo TopologyInfo::from_env() {
  const int ws = env_int_or("WORLD_SIZE", 1);
  const int lws = env_int_or("LOCAL_WORLD_SIZE", ws);
  const int rank = env_int_or("RANK", 0);
  const int lrank = env_int_or("LOCAL_RANK", rank % (lws > 0 ? lws : 1));
  return make(ws, lws, rank, lrank);
}

// ---------------------------------------------------------------------------
// Transport policy -- env overrides (default PCIe for intra-node).
// ---------------------------------------------------------------------------
static Transport transport_from_env(const char* name, Transport fallback) {
  const char* v = std::getenv(name);
  if (!v || !*v) return fallback;
  std::string s(v);
  for (auto& c : s) c = static_cast<char>(std::tolower(c));
  if (s == "nic" || s == "rdma" || s == "ishmem") return Transport::NIC;
  if (s == "pcie" || s == "p2p" || s == "xelink") return Transport::PCIe;
  return fallback;
}

TransportPolicy TransportPolicy::from_env() {
  TransportPolicy p;  // defaults: same_numa/same_node = PCIe, cross_node = NIC
  p.same_numa = transport_from_env("TOPO_TRANSPORT_SAME_NUMA", p.same_numa);
  p.same_node = transport_from_env("TOPO_TRANSPORT_SAME_NODE", p.same_node);
  return p;
}

// ---------------------------------------------------------------------------
// hwloc helpers.
// ---------------------------------------------------------------------------
namespace {

// Normalise a BDF to the canonical "domain:bus:dev.func" form hwloc uses.
// Accepts inputs with or without the 4-digit domain.
std::string normalize_bdf(const std::string& in) {
  // Expected canonical: 0000:18:00.0
  if (in.size() >= 12 && in[4] == ':') return in;  // already domain:bus:...
  if (in.size() >= 7 && in[2] == ':') return "0000:" + in;  // add domain
  return in;
}

// Parse "domain:bus:dev.func" into components; returns false on failure.
bool parse_bdf(
    const std::string& bdf, unsigned& domain, unsigned& bus, unsigned& dev,
    unsigned& func) {
  return std::sscanf(
             bdf.c_str(), "%x:%x:%x.%x", &domain, &bus, &dev, &func) == 4;
}

hwloc_obj_t pci_obj_for_bdf(hwloc_topology_t topo, const std::string& bdf) {
  unsigned domain, bus, dev, func;
  if (!parse_bdf(normalize_bdf(bdf), domain, bus, dev, func)) return nullptr;
  return hwloc_get_pcidev_by_busid(topo, domain, bus, dev, func);
}

// Nearest non-I/O ancestor (a bridge / host bridge / NUMA-bearing object).
hwloc_obj_t non_io_ancestor(hwloc_topology_t topo, hwloc_obj_t io) {
  if (!io) return nullptr;
  return hwloc_get_non_io_ancestor_obj(topo, io);
}

// The first NUMA node id covered by `obj`'s nodeset, or -1.
int numa_of(hwloc_obj_t obj) {
  if (!obj || !obj->nodeset) return -1;
  int id = hwloc_bitmap_first(obj->nodeset);
  return id;  // -1 if empty
}

// Is `obj` a PCI-to-PCI bridge (i.e. a switch port), as opposed to the host
// bridge that terminates a PCI hierarchy at the CPU/root complex?
bool is_pci_to_pci_bridge(hwloc_obj_t obj) {
  return obj && obj->type == HWLOC_OBJ_BRIDGE &&
      obj->attr->bridge.upstream_type == HWLOC_OBJ_BRIDGE_PCI;
}

// The PCIe "switch group" id of a PCI device: the gp_index of the OUTERMOST
// PCI-to-PCI bridge on the path from the device up to (but not including) the
// host bridge -- i.e. the upstream port of the switch the device sits behind.
// Two devices behind the same switch share this upstream-port bridge, so they
// get the same id; devices directly on a host bridge get -1 (no shared switch).
int pcie_switch_of(hwloc_obj_t pci) {
  int id = -1;
  for (hwloc_obj_t o = pci ? pci->parent : nullptr; o != nullptr;
       o = o->parent) {
    if (o->type == HWLOC_OBJ_BRIDGE) {
      if (is_pci_to_pci_bridge(o))
        id = static_cast<int>(o->gp_index);  // climb to the outermost
      else
        break;  // reached the host bridge
    }
  }
  return id;
}

bool is_network_osdev(hwloc_obj_t osdev) {
  if (!osdev || osdev->type != HWLOC_OBJ_OS_DEVICE) return false;
  const auto t = osdev->attr->osdev.type;
  return t == HWLOC_OBJ_OSDEV_NETWORK || t == HWLOC_OBJ_OSDEV_OPENFABRICS;
}

}  // namespace

// ---------------------------------------------------------------------------
// NodeTopology::probe
// ---------------------------------------------------------------------------
NodeTopology NodeTopology::probe(const std::vector<std::string>& device_bdfs) {
  NodeTopology nt;
  nt.local_world_size = static_cast<int>(device_bdfs.size());

  hwloc_topology_t topo;
  if (hwloc_topology_init(&topo) != 0)
    throw std::runtime_error("Topology: hwloc_topology_init failed");
  // Keep PCI devices and NICs in the tree.
  hwloc_topology_set_io_types_filter(topo, HWLOC_TYPE_FILTER_KEEP_IMPORTANT);
  if (hwloc_topology_load(topo) != 0) {
    hwloc_topology_destroy(topo);
    throw std::runtime_error("Topology: hwloc_topology_load failed");
  }

  // ---- discover NICs ----
  for (hwloc_obj_t os = hwloc_get_next_osdev(topo, nullptr); os != nullptr;
       os = hwloc_get_next_osdev(topo, os)) {
    if (!is_network_osdev(os)) continue;
    NicTopo nic;
    nic.index = static_cast<int>(nt.nics.size());
    nic.name = os->name ? os->name : "";
    // The NIC's PCI parent carries the BDF / NUMA locality.
    hwloc_obj_t pci = os->parent;
    while (pci && pci->type != HWLOC_OBJ_PCI_DEVICE) pci = pci->parent;
    if (pci) {
      char buf[32];
      std::snprintf(
          buf, sizeof(buf), "%04x:%02x:%02x.%01x",
          pci->attr->pcidev.domain, pci->attr->pcidev.bus,
          pci->attr->pcidev.dev, pci->attr->pcidev.func);
      nic.pci_bdf = buf;
    }
    nic.numa_node = numa_of(non_io_ancestor(topo, os));
    nt.nics.push_back(std::move(nic));
  }

  // ---- per-device placement ----
  nt.devices.resize(nt.local_world_size);
  for (int lr = 0; lr < nt.local_world_size; ++lr) {
    DeviceTopo d;
    d.local_rank = lr;
    d.pci_bdf = normalize_bdf(device_bdfs[lr]);

    hwloc_obj_t pci = pci_obj_for_bdf(topo, d.pci_bdf);
    if (pci) {
      d.numa_node = numa_of(non_io_ancestor(topo, pci));
      d.pcie_switch = pcie_switch_of(pci);

      // Nearest NIC by NUMA locality (same approach as oneCCL's hwloc wrapper,
      // which associates devices to NICs via NUMA/cpuset locality rather than
      // walking the PCIe graph): pick the first NIC on this device's NUMA node,
      // else fall back to the first NIC on the node.
      int best = -1;
      for (size_t n = 0; n < nt.nics.size(); ++n) {
        if (d.numa_node >= 0 && nt.nics[n].numa_node == d.numa_node) {
          best = static_cast<int>(n);
          break;
        }
      }
      if (best < 0 && !nt.nics.empty()) best = 0;
      d.nearest_nic = best;
    }
    nt.devices[lr] = std::move(d);
  }

  hwloc_topology_destroy(topo);
  return nt;
}

void NodeTopology::validate() const {
  // switch ⊂ numa: any two devices sharing a switch must share a NUMA node.
  for (size_t a = 0; a < devices.size(); ++a) {
    for (size_t b = a + 1; b < devices.size(); ++b) {
      const bool sw = same_switch(static_cast<int>(a), static_cast<int>(b));
      const bool nu = same_numa(static_cast<int>(a), static_cast<int>(b));
      if (sw && !nu) {
        std::ostringstream oss;
        oss << "Topology: containment violation -- local_rank " << a << " and "
            << b << " share PCIe switch but not NUMA node";
        throw std::runtime_error(oss.str());
      }
    }
  }
}

std::string NodeTopology::describe() const {
  std::ostringstream oss;
  oss << "NodeTopology(local_world_size=" << local_world_size << ")\n";
  for (const auto& d : devices) {
    oss << "  dev[" << d.local_rank << "] bdf=" << d.pci_bdf
        << " numa=" << d.numa_node << " switch=" << d.pcie_switch
        << " nic=" << d.nearest_nic << "\n";
  }
  for (const auto& n : nics) {
    oss << "  nic[" << n.index << "] name=" << n.name << " bdf=" << n.pci_bdf
        << " numa=" << n.numa_node << "\n";
  }
  return oss.str();
}

// ---------------------------------------------------------------------------
// SYCL BDF query.
// ---------------------------------------------------------------------------
#include <sycl/sycl.hpp>

std::string device_bdf_from_sycl(int device_index) {
  try {
    auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (device_index < 0 || device_index >= static_cast<int>(devices.size()))
      return {};
    auto& dev = devices[device_index];
#ifdef SYCL_EXT_INTEL_DEVICE_INFO
    if (dev.has(sycl::aspect::ext_intel_pci_address)) {
      return dev.get_info<sycl::ext::intel::info::device::pci_address>();
    }
#endif
    return {};
  } catch (...) {
    return {};
  }
}

// ---------------------------------------------------------------------------
// Topology::from_env -- one-shot "get everything".
// ---------------------------------------------------------------------------
Topology Topology::from_env(bool validate) {
  Topology t;
  t.layout = TopologyInfo::from_env();
  t.policy = TransportPolicy::from_env();

  // Enumerate the local GPUs' PCI BDFs (all GPUs on this node are visible to
  // the SYCL platform), ordered by device index == local_rank.
  std::vector<std::string> bdfs(t.layout.local_world_size);
  for (int lr = 0; lr < t.layout.local_world_size; ++lr) {
    bdfs[lr] = device_bdf_from_sycl(lr);
  }

  t.node = NodeTopology::probe(bdfs);
  if (validate) t.node.validate();
  return t;
}
