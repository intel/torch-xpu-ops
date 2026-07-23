// TopologyBindings.cpp
//
// pybind11 bindings for the hwloc-backed Topology (see Topology.hpp). This is a
// plain Python extension module -- NOT a torch custom op / torch.ops entry.
// Build produces `xpu_topology<ext>.so`; import as `import xpu_topology`.
//
//   import xpu_topology as topo
//   t = topo.Topology.from_env()          # reads env + probes hwloc
//   t.same_switch_ranks()                 # -> [global ranks]
//   t.same_numa_ranks()
//   t.same_node_ranks()
//   t.transport_to(peer)                  # -> topo.Transport.PCIe / NIC
//   print(t.describe())

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Topology.hpp"

namespace py = pybind11;

PYBIND11_MODULE(xpu_topology, m) {
  m.doc() = "hwloc-backed GPU/NIC node topology (non torch-op bindings)";

  py::enum_<Transport>(m, "Transport")
      .value("PCIe", Transport::PCIe)
      .value("NIC", Transport::NIC);

  py::class_<TopologyInfo>(m, "TopologyInfo")
      .def_static("from_env", &TopologyInfo::from_env)
      .def_static(
          "make", &TopologyInfo::make, py::arg("world_size"),
          py::arg("local_world_size"), py::arg("rank"), py::arg("local_rank"))
      .def_readonly("world_size", &TopologyInfo::world_size)
      .def_readonly("local_world_size", &TopologyInfo::local_world_size)
      .def_readonly("rank", &TopologyInfo::rank)
      .def_readonly("local_rank", &TopologyInfo::local_rank)
      .def_readonly("num_nodes", &TopologyInfo::num_nodes)
      .def_readonly("node_id", &TopologyInfo::node_id)
      .def("is_first_node", &TopologyInfo::is_first_node)
      .def("is_last_node", &TopologyInfo::is_last_node)
      .def("is_node_leader", &TopologyInfo::is_node_leader)
      .def("is_global_root", &TopologyInfo::is_global_root)
      .def("is_single_node", &TopologyInfo::is_single_node)
      .def("node_of", &TopologyInfo::node_of, py::arg("rank"))
      .def("local_rank_of", &TopologyInfo::local_rank_of, py::arg("rank"))
      .def("same_node", &TopologyInfo::same_node, py::arg("a"), py::arg("b"));

  py::class_<DeviceTopo>(m, "DeviceTopo")
      .def_readonly("local_rank", &DeviceTopo::local_rank)
      .def_readonly("numa_node", &DeviceTopo::numa_node)
      .def_readonly("pcie_switch", &DeviceTopo::pcie_switch)
      .def_readonly("nearest_nic", &DeviceTopo::nearest_nic)
      .def_readonly("pci_bdf", &DeviceTopo::pci_bdf);

  py::class_<NicTopo>(m, "NicTopo")
      .def_readonly("index", &NicTopo::index)
      .def_readonly("numa_node", &NicTopo::numa_node)
      .def_readonly("name", &NicTopo::name)
      .def_readonly("pci_bdf", &NicTopo::pci_bdf);

  py::class_<NodeTopology>(m, "NodeTopology")
      .def_static(
          "probe", &NodeTopology::probe, py::arg("device_bdfs"),
          "Probe hwloc for the given local device PCI BDFs (ordered by "
          "local_rank).")
      .def_readonly("local_world_size", &NodeTopology::local_world_size)
      .def_readonly("devices", &NodeTopology::devices)
      .def_readonly("nics", &NodeTopology::nics)
      .def("same_numa", &NodeTopology::same_numa, py::arg("a"), py::arg("b"))
      .def("same_switch", &NodeTopology::same_switch, py::arg("a"), py::arg("b"))
      .def("nearest_nic", &NodeTopology::nearest_nic, py::arg("local_rank"))
      .def("share_nic", &NodeTopology::share_nic, py::arg("a"), py::arg("b"))
      .def("validate", &NodeTopology::validate)
      .def("describe", &NodeTopology::describe);

  py::class_<Topology>(m, "Topology")
      .def_static(
          "from_env", &Topology::from_env, py::arg("validate") = true,
          "One-shot: read env vars, enumerate local GPU BDFs, probe hwloc.")
      .def_readonly("layout", &Topology::layout)
      .def_readonly("node", &Topology::node)
      .def("same_switch", &Topology::same_switch, py::arg("ga"), py::arg("gb"))
      .def("same_numa", &Topology::same_numa, py::arg("ga"), py::arg("gb"))
      .def("my_nic", &Topology::my_nic)
      .def("transport_to", &Topology::transport_to, py::arg("peer"))
      .def(
          "same_node_ranks", &Topology::same_node_ranks,
          py::arg("include_self") = true)
      .def(
          "same_switch_ranks", &Topology::same_switch_ranks,
          py::arg("include_self") = true)
      .def(
          "same_numa_ranks", &Topology::same_numa_ranks,
          py::arg("include_self") = true)
      .def(
          "describe",
          [](const Topology& t) { return t.node.describe(); });

  m.def(
      "device_bdf_from_sycl", &device_bdf_from_sycl, py::arg("device_index"),
      "Return the PCI BDF (e.g. '0000:18:00.0') of the given SYCL GPU index.");
}
