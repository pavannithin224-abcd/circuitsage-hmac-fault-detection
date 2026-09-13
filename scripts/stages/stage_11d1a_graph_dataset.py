#!/usr/bin/env python3
"""Build and freeze the deterministic graph of the frozen OpenTitan HMAC netlist."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

try:
    import numpy as np
except ImportError as error:
    raise SystemExit(
        "STOP: NumPy is required. Activate the frozen project .venv before running."
    ) from error


VERSION = "HMAC-GOLDEN-NETLIST-GRAPH-BUILDER-v1"
SCHEMA_VERSION = "HMAC-GOLDEN-NETLIST-GRAPH-SCHEMA-v1"
TOP_MODULE = "opentitan_hmac_sha256_msg32"
EXPECTED_NODES = 22_839
EXPECTED_FAULT_INSTANCES = 45_678
EXPECTED_DEV_TRAIN = 15_987
EXPECTED_DEV_CALIBRATION = 3_426
EXPECTED_DEV_SITE_TEST = 3_426

ROOT = Path.cwd().resolve()
SOURCE = Path(__file__).resolve()
RESULT_ROOT = ROOT / "results/hmac_fault_campaign_11d1"
GRAPH_ROOT = RESULT_ROOT / "graph_dataset_11d1a"
CONFIG_ROOT = ROOT / "config/diagnostic_model"

GOLDEN_NETLIST = ROOT / "build/hmac_synth_11b2a/opentitan_hmac_sha256_msg32_generic.json"
LEGAL_SITES = ROOT / "results/hmac_fault_campaign_11c1/hmac_legal_fault_sites_11c1b.json"
SITE_SPLIT = ROOT / "results/hmac_fault_campaign_11c5/hmac_fault_site_group_split_11c5d.csv"
SITE_SPLIT_MANIFEST = ROOT / "results/hmac_fault_campaign_11c5/hmac_fault_site_group_split_manifest_11c5d.json"
FEATURE_SCHEMA = (
    ROOT
    / "results/hmac_fault_campaign_11c5/feature_matrix_11c5e/hmac_leakage_safe_feature_matrix_schema_11c5e.json"
)
STAGE_11C5L_SOURCE = ROOT / "stage_11c5l_baseline_disposition_gnn_readiness.py"
DISPOSITION_POLICY = ROOT / "config/diagnostic_model/hmac_diagnostic_baseline_disposition_policy_11c5l.json"
GNN_READINESS_POLICY = ROOT / "config/diagnostic_model/hmac_gnn_readiness_policy_11c5l.json"
COMPARATOR_REGISTRY = ROOT / "results/hmac_fault_campaign_11c5/hmac_frozen_diagnostic_comparator_registry_11c5l.csv"
READINESS_AUDIT = ROOT / "results/hmac_fault_campaign_11c5/hmac_diagnostic_baseline_disposition_gnn_readiness_freeze_11c5l.json"

GRAPH_CONTRACT = CONFIG_ROOT / "hmac_gnn_graph_dataset_contract_11d1a.json"
NODE_TABLE = GRAPH_ROOT / "hmac_golden_netlist_graph_nodes_11d1a.csv"
EDGE_TABLE = GRAPH_ROOT / "hmac_golden_netlist_graph_edges_11d1a.csv"
BOUNDARY_TABLE = GRAPH_ROOT / "hmac_golden_netlist_graph_boundary_inputs_11d1a.csv"
GRAPH_NPZ = GRAPH_ROOT / "hmac_golden_netlist_graph_11d1a.npz"
GRAPH_SCHEMA = GRAPH_ROOT / "hmac_golden_netlist_graph_schema_11d1a.json"
MANIFEST = RESULT_ROOT / "hmac_golden_netlist_graph_manifest_11d1a.json"
AUDIT = RESULT_ROOT / "hmac_golden_netlist_graph_topology_integrity_freeze_11d1a.json"

EXPECTED_INPUTS = {
    GOLDEN_NETLIST: "0a33bb40ea331e8bc7fbc80b1c55c339a7687a694a650921222dd08775eb56a1",
    LEGAL_SITES: "d9a39366bb35a277125376c6dc26c83273b45c0d7732fb93a3403401a071cb45",
    SITE_SPLIT: "602fa310547f68d1e9b5ceb6f148d6b125c69df0589929f9edfde9d264922c61",
    SITE_SPLIT_MANIFEST: "1c96b7c7c7f5301e70037aaa9a0d18effd28cae6bf15e9f85cc85f0c413473fb",
    FEATURE_SCHEMA: "0bf1edffb8078b003a1116b276615d5544979d007712ab49870b1b93784e486c",
    STAGE_11C5L_SOURCE: "a10d17999b44353f6de4d79b210a518e7f9e35a1ad13dbd1d2b5f1b9aa5ff662",
    DISPOSITION_POLICY: "26553f2c402c2723c1a702fb23dc7ad8b897c0301a6c67d36a02f059ef171747",
    GNN_READINESS_POLICY: "702eb89f214aa2e70b1c95850d622ea15678ec71667639b0e7edc5f6cf3dd9fe",
    COMPARATOR_REGISTRY: "41dca9b233950636fbd93c0011e94fbc21fc237baf72faf242de410c8386bc2c",
    READINESS_AUDIT: "9fca71d7ab347e0c20c09ad2da5d42c0d4bbb7c47a9858e38680eb98dfcd49d1",
}

NODE_FIELDS = [
    "node_index",
    "site_id",
    "site_index",
    "partition",
    "partition_rank",
    "driver_cell",
    "driver_cell_type",
    "site_category",
    "output_port",
    "output_pin_index",
    "bit_id",
    "net_name",
    "cell_fanout",
    "derived_sink_branches",
    "is_primary_output_stem",
    "primary_output_names",
    "in_degree_branches",
    "out_degree_branches",
    "in_neighbor_count",
    "out_neighbor_count",
]

EDGE_FIELDS = [
    "edge_index",
    "source_node_index",
    "destination_node_index",
    "source_site_id",
    "destination_site_id",
    "multiplicity",
    "data_branches",
    "select_branches",
    "enable_branches",
    "clock_branches",
    "reset_branches",
    "other_branches",
    "is_self_loop",
]

BOUNDARY_FIELDS = [
    "boundary_index",
    "source_kind",
    "source_name",
    "source_pin_index",
    "bit_value",
    "destination_node_index",
    "destination_site_id",
    "destination_cell",
    "destination_port",
    "destination_pin_index",
    "edge_role",
]

PARTITION_CODE = {"DEV_TRAIN": 0, "DEV_CALIBRATION": 1, "DEV_SITE_TEST": 2}
SITE_CATEGORY_CODE = {"COMBINATIONAL_LOGIC_STEM": 0, "SEQUENTIAL_STATE_STEM": 1}
EDGE_ROLES = ("DATA", "SELECT", "ENABLE", "CLOCK", "RESET", "OTHER")


def stop(message: str) -> None:
    raise SystemExit(f"STOP: {message}")


def require(condition: bool, message: str) -> None:
    if not condition:
        stop(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        stop(f"artifact outside project root: {path}")


def record(path: Path) -> dict:
    return {"path": relative(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        stop(f"could not read JSON {path}: {error}")


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with zipfile.ZipFile(
        temporary,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as archive:
        for name in sorted(arrays):
            array = np.ascontiguousarray(arrays[name])
            payload = io.BytesIO()
            np.lib.format.write_array(payload, array, allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(
                info,
                payload.getvalue(),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=6,
            )
    temporary.replace(path)


def verify_input(path: Path, expected_sha: str) -> dict:
    require(path.is_file(), f"missing frozen input: {path}")
    require(path.stat().st_size > 0, f"empty frozen input: {path}")
    actual = sha256(path)
    require(
        actual == expected_sha,
        f"SHA mismatch for {path}: expected {expected_sha}, actual {actual}",
    )
    print(f"  {path.name:<74}: OK")
    return record(path)


def find_sites(value):
    if isinstance(value, dict):
        for child in value.values():
            result = find_sites(child)
            if result is not None:
                return result
    elif isinstance(value, list):
        if (
            value
            and isinstance(value[0], dict)
            and "fault_site_id" in value[0]
            and "driver_cell" in value[0]
        ):
            return value
        for child in value:
            result = find_sites(child)
            if result is not None:
                return result
    return None


def load_split() -> dict[str, dict]:
    rows: dict[str, dict] = {}
    counts = Counter()
    with SITE_SPLIT.open(newline="") as stream:
        reader = csv.DictReader(stream)
        require(
            reader.fieldnames
            == ["site_id", "site_index", "partition", "partition_rank", "split_hash_sha256"],
            "site-split header",
        )
        for line_number, raw in enumerate(reader, start=2):
            site_id = raw["site_id"]
            require(re.fullmatch(r"HMAC-STEM-\d{6}", site_id) is not None, f"site ID at split line {line_number}")
            require(site_id not in rows, f"duplicate split site {site_id}")
            site_index = int(raw["site_index"])
            partition = raw["partition"]
            require(partition in PARTITION_CODE, f"partition at split line {line_number}")
            rows[site_id] = {
                "site_id": site_id,
                "site_index": site_index,
                "partition": partition,
                "partition_rank": int(raw["partition_rank"]),
            }
            counts[partition] += 1
    require(len(rows) == EXPECTED_NODES, "site-split record count")
    require(
        sorted(item["site_index"] for item in rows.values()) == list(range(1, EXPECTED_NODES + 1)),
        "site-split site-index coverage",
    )
    require(counts == Counter({
        "DEV_TRAIN": EXPECTED_DEV_TRAIN,
        "DEV_CALIBRATION": EXPECTED_DEV_CALIBRATION,
        "DEV_SITE_TEST": EXPECTED_DEV_SITE_TEST,
    }), "site-split partition counts")
    return rows


def bit_key(value) -> tuple[str, object]:
    if isinstance(value, bool):
        stop(f"boolean Yosys bit identifier: {value}")
    if isinstance(value, int):
        return ("wire", value)
    if isinstance(value, str) and value in {"0", "1", "x", "z"}:
        return ("constant", value)
    stop(f"unsupported Yosys bit identifier: {value!r}")


def edge_role(cell_type: str, port: str) -> str:
    upper_port = port.upper().lstrip("\\")
    upper_type = cell_type.upper()
    if upper_port in {"C", "CLK", "CLOCK"}:
        return "CLOCK"
    if upper_port in {"R", "RN", "RESET", "RST", "CLR", "ARST", "SRST"}:
        return "RESET"
    if upper_port in {"E", "EN", "ENABLE", "CE"}:
        return "ENABLE"
    if upper_port in {"S", "SEL", "SELECT"} or ("MUX" in upper_type and upper_port.startswith("S")):
        return "SELECT"
    if upper_port in {"A", "B", "D", "I", "IN", "DATA"} or upper_port.startswith("A") or upper_port.startswith("B"):
        return "DATA"
    return "OTHER"


def output_names(module: dict) -> dict[int, list[str]]:
    names: dict[int, list[str]] = defaultdict(list)
    for port_name, port in sorted(module.get("ports", {}).items()):
        if port.get("direction") not in {"output", "inout"}:
            continue
        for index, raw_bit in enumerate(port.get("bits", [])):
            kind, value = bit_key(raw_bit)
            if kind == "wire":
                names[int(value)].append(f"{port_name}[{index}]")
    return {key: sorted(value) for key, value in names.items()}


def input_names(module: dict) -> dict[int, list[tuple[str, int]]]:
    names: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for port_name, port in sorted(module.get("ports", {}).items()):
        if port.get("direction") not in {"input", "inout"}:
            continue
        for index, raw_bit in enumerate(port.get("bits", [])):
            kind, value = bit_key(raw_bit)
            if kind == "wire":
                names[int(value)].append((port_name, index))
    return {key: sorted(value) for key, value in names.items()}


def bit_net_names(module: dict) -> dict[int, list[str]]:
    names: dict[int, list[str]] = defaultdict(list)
    for name, entry in sorted(module.get("netnames", {}).items()):
        for raw_bit in entry.get("bits", []):
            kind, value = bit_key(raw_bit)
            if kind == "wire":
                names[int(value)].append(name)
    return {key: sorted(set(value)) for key, value in names.items()}


def parse_netlist() -> tuple[dict, dict, dict, dict, dict]:
    raw = load_json(GOLDEN_NETLIST)
    modules = raw.get("modules")
    require(isinstance(modules, dict), "Yosys modules object")
    require(TOP_MODULE in modules, f"top module {TOP_MODULE}")
    require(len(modules) == 1, f"expected one flattened module, found {len(modules)}")
    module = modules[TOP_MODULE]
    cells = module.get("cells")
    require(isinstance(cells, dict), "Yosys cells object")
    require(len(cells) == EXPECTED_NODES, "golden netlist cell count")

    drivers: dict[int, tuple[str, str, int, str]] = {}
    cell_outputs: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    for cell_name, cell in sorted(cells.items()):
        directions = cell.get("port_directions", {})
        connections = cell.get("connections", {})
        cell_type = str(cell.get("type", ""))
        require(cell_type, f"cell type for {cell_name}")
        for port_name in sorted(connections):
            require(port_name in directions, f"missing direction for {cell_name}.{port_name}")
            if directions[port_name] != "output":
                continue
            for pin_index, raw_bit in enumerate(connections[port_name]):
                kind, value = bit_key(raw_bit)
                require(kind == "wire", f"constant cell output {cell_name}.{port_name}[{pin_index}]")
                bit_id = int(value)
                require(bit_id not in drivers, f"multiple drivers for bit {bit_id}")
                drivers[bit_id] = (cell_name, port_name, pin_index, cell_type)
                cell_outputs[cell_name].append((port_name, pin_index, bit_id))
    require(len(cell_outputs) == EXPECTED_NODES, "cells with output stems")
    require(all(len(outputs) == 1 for outputs in cell_outputs.values()), "one output stem per cell")
    return module, cells, drivers, cell_outputs, bit_net_names(module)


def load_legal_sites(
    split: dict[str, dict],
    cells: dict,
    cell_outputs: dict[str, list[tuple[str, int, int]]],
    net_names: dict[int, list[str]],
    primary_outputs: dict[int, list[str]],
) -> tuple[list[dict], dict[str, int], dict[int, int]]:
    raw_sites = find_sites(load_json(LEGAL_SITES))
    require(isinstance(raw_sites, list), "legal-site record list")
    require(len(raw_sites) == EXPECTED_NODES, "legal-site count")
    nodes = []
    cell_to_node: dict[str, int] = {}
    bit_to_node: dict[int, int] = {}
    required = {
        "fault_site_id",
        "site_index",
        "driver_cell",
        "driver_cell_type",
        "driver_port",
        "driver_pin_index",
        "bit_id",
        "net_name",
        "site_category",
        "cell_fanout",
    }
    for raw in raw_sites:
        site_id = str(raw.get("fault_site_id", ""))
        require(required <= set(raw), f"required legal-site fields for {site_id}")
        require(site_id in split, f"legal site absent from split: {site_id}")
        site_index = int(raw["site_index"])
        require(site_index == split[site_id]["site_index"], f"split index for {site_id}")
        cell_name = str(raw["driver_cell"])
        require(cell_name in cells, f"driver cell absent from golden netlist: {site_id}")
        require(cell_name not in cell_to_node, f"duplicate driver cell mapping: {cell_name}")
        cell_type = str(cells[cell_name].get("type"))
        require(cell_type == str(raw["driver_cell_type"]), f"driver type mismatch for {site_id}")
        output_port, output_pin, bit_id = cell_outputs[cell_name][0]
        require(output_port == str(raw["driver_port"]), f"output port mismatch for {site_id}")
        require(output_pin == int(raw["driver_pin_index"]), f"output pin mismatch for {site_id}")
        require(bit_id == int(raw["bit_id"]), f"output bit mismatch for {site_id}")
        require(bit_id not in bit_to_node, f"duplicate output bit mapping: {bit_id}")
        legal_net_name = str(raw["net_name"])
        if legal_net_name:
            aliases = set(net_names.get(bit_id, []))
            legal_base = re.sub(r"\[\d+(?::\d+)?\]$", "", legal_net_name)
            alias_bases = {
                re.sub(r"\[\d+(?::\d+)?\]$", "", alias)
                for alias in aliases
            }
            require(
                legal_net_name in aliases
                or legal_base in aliases
                or legal_base in alias_bases
                or legal_net_name == output_port
                or legal_net_name in primary_outputs.get(bit_id, []),
                f"net-name alias absent for {site_id}: {legal_net_name}",
            )
        category = str(raw["site_category"])
        require(category in SITE_CATEGORY_CODE, f"site category for {site_id}")
        is_primary = bool(raw.get("is_primary_output_stem", bool(primary_outputs.get(bit_id))))
        require(is_primary == bool(primary_outputs.get(bit_id)), f"primary-output status for {site_id}")
        node_index = site_index - 1
        node = {
            "node_index": node_index,
            "site_id": site_id,
            "site_index": site_index,
            "partition": split[site_id]["partition"],
            "partition_rank": split[site_id]["partition_rank"],
            "driver_cell": cell_name,
            "driver_cell_type": cell_type,
            "site_category": category,
            "output_port": output_port,
            "output_pin_index": output_pin,
            "bit_id": bit_id,
            "net_name": legal_net_name,
            "cell_fanout": int(raw["cell_fanout"]),
            "is_primary_output_stem": int(is_primary),
            "primary_output_names": "|".join(primary_outputs.get(bit_id, [])),
        }
        require(node["cell_fanout"] >= 0, f"negative fanout for {site_id}")
        nodes.append(node)
        cell_to_node[cell_name] = node_index
        bit_to_node[bit_id] = node_index
    nodes.sort(key=lambda item: item["node_index"])
    require([item["node_index"] for item in nodes] == list(range(EXPECTED_NODES)), "node-index coverage")
    require(len(cell_to_node) == EXPECTED_NODES, "cell-to-node bijection")
    require(len(bit_to_node) == EXPECTED_NODES, "bit-to-node bijection")
    return nodes, cell_to_node, bit_to_node


def build_edges(
    nodes: list[dict],
    cells: dict,
    cell_to_node: dict[str, int],
    bit_to_node: dict[int, int],
    top_inputs: dict[int, list[tuple[str, int]]],
) -> tuple[list[dict], list[dict], dict]:
    pair_roles: dict[tuple[int, int], Counter] = defaultdict(Counter)
    boundary = []
    constant_branches = Counter()
    undriven = []
    raw_internal_branches = 0
    total_input_branches = 0

    for destination_cell, cell in sorted(cells.items()):
        destination = cell_to_node[destination_cell]
        directions = cell["port_directions"]
        connections = cell["connections"]
        cell_type = str(cell["type"])
        for port_name in sorted(connections):
            if directions[port_name] != "input":
                continue
            role = edge_role(cell_type, port_name)
            for pin_index, raw_bit in enumerate(connections[port_name]):
                total_input_branches += 1
                kind, value = bit_key(raw_bit)
                if kind == "constant":
                    constant_branches[str(value)] += 1
                    continue
                bit_id = int(value)
                source = bit_to_node.get(bit_id)
                if source is not None:
                    pair_roles[(source, destination)][role] += 1
                    raw_internal_branches += 1
                    continue
                boundary_sources = top_inputs.get(bit_id)
                if boundary_sources:
                    for source_name, source_pin in boundary_sources:
                        boundary.append({
                            "source_kind": "TOP_INPUT",
                            "source_name": source_name,
                            "source_pin_index": source_pin,
                            "bit_value": bit_id,
                            "destination_node_index": destination,
                            "destination_site_id": nodes[destination]["site_id"],
                            "destination_cell": destination_cell,
                            "destination_port": port_name,
                            "destination_pin_index": pin_index,
                            "edge_role": role,
                        })
                    continue
                undriven.append((destination_cell, port_name, pin_index, bit_id))

    require(not undriven, f"undriven cell-input branches: {undriven[:5]}")
    edges = []
    for edge_index, ((source, destination), roles) in enumerate(sorted(pair_roles.items())):
        multiplicity = sum(roles.values())
        edges.append({
            "edge_index": edge_index,
            "source_node_index": source,
            "destination_node_index": destination,
            "source_site_id": nodes[source]["site_id"],
            "destination_site_id": nodes[destination]["site_id"],
            "multiplicity": multiplicity,
            "data_branches": roles["DATA"],
            "select_branches": roles["SELECT"],
            "enable_branches": roles["ENABLE"],
            "clock_branches": roles["CLOCK"],
            "reset_branches": roles["RESET"],
            "other_branches": roles["OTHER"],
            "is_self_loop": int(source == destination),
        })
    boundary.sort(key=lambda item: (
        item["source_name"],
        item["source_pin_index"],
        item["destination_node_index"],
        item["destination_port"],
        item["destination_pin_index"],
    ))
    for boundary_index, item in enumerate(boundary):
        item["boundary_index"] = boundary_index

    stats = {
        "total_cell_input_branches": total_input_branches,
        "raw_internal_branches": raw_internal_branches,
        "collapsed_directed_edges": len(edges),
        "top_input_boundary_branches": len(boundary),
        "constant_input_branches": sum(constant_branches.values()),
        "constant_input_values": dict(sorted(constant_branches.items())),
        "undriven_input_branches": len(undriven),
        "self_loop_edges": sum(item["is_self_loop"] for item in edges),
        "edge_role_branches": {
            role: sum(item[f"{role.lower()}_branches"] for item in edges)
            for role in EDGE_ROLES
        },
    }
    require(
        total_input_branches
        == raw_internal_branches + len(boundary) + sum(constant_branches.values()),
        "cell-input branch accounting",
    )
    return edges, boundary, stats


def enrich_nodes(nodes: list[dict], edges: list[dict]) -> dict:
    in_branches = [0] * len(nodes)
    out_branches = [0] * len(nodes)
    in_neighbors = [set() for _ in nodes]
    out_neighbors = [set() for _ in nodes]
    for edge in edges:
        source = edge["source_node_index"]
        destination = edge["destination_node_index"]
        count = edge["multiplicity"]
        out_branches[source] += count
        in_branches[destination] += count
        out_neighbors[source].add(destination)
        in_neighbors[destination].add(source)
    fanout_mismatches = []
    for node in nodes:
        index = node["node_index"]
        node["derived_sink_branches"] = out_branches[index]
        node["in_degree_branches"] = in_branches[index]
        node["out_degree_branches"] = out_branches[index]
        node["in_neighbor_count"] = len(in_neighbors[index])
        node["out_neighbor_count"] = len(out_neighbors[index])
        if node["cell_fanout"] != out_branches[index]:
            fanout_mismatches.append({
                "site_id": node["site_id"],
                "frozen": node["cell_fanout"],
                "derived": out_branches[index],
            })
    require(not fanout_mismatches, f"legal-site fanout mismatches: {fanout_mismatches[:5]}")
    return {
        "fanout_mismatches": len(fanout_mismatches),
        "zero_internal_indegree_nodes": sum(value == 0 for value in in_branches),
        "zero_internal_outdegree_nodes": sum(value == 0 for value in out_branches),
        "isolated_internal_nodes": sum(
            in_branches[index] == 0 and out_branches[index] == 0
            for index in range(len(nodes))
        ),
        "maximum_in_degree_branches": max(in_branches),
        "maximum_out_degree_branches": max(out_branches),
    }


def weak_components(node_count: int, edges: list[dict]) -> list[int]:
    parent = list(range(node_count))
    size = [1] * node_count

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if size[left_root] < size[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        size[left_root] += size[right_root]

    for edge in edges:
        union(edge["source_node_index"], edge["destination_node_index"])
    counts = Counter(find(index) for index in range(node_count))
    return sorted(counts.values(), reverse=True)


def strongly_connected_components(node_count: int, edges: list[dict]) -> list[int]:
    forward = [[] for _ in range(node_count)]
    reverse = [[] for _ in range(node_count)]
    for edge in edges:
        source = edge["source_node_index"]
        destination = edge["destination_node_index"]
        forward[source].append(destination)
        reverse[destination].append(source)
    for adjacency in (forward, reverse):
        for row in adjacency:
            row.sort()

    visited = bytearray(node_count)
    order = []
    for start in range(node_count):
        if visited[start]:
            continue
        visited[start] = 1
        stack = [(start, 0)]
        while stack:
            node, next_index = stack[-1]
            if next_index < len(forward[node]):
                neighbor = forward[node][next_index]
                stack[-1] = (node, next_index + 1)
                if not visited[neighbor]:
                    visited[neighbor] = 1
                    stack.append((neighbor, 0))
            else:
                order.append(node)
                stack.pop()

    assigned = bytearray(node_count)
    sizes = []
    for start in reversed(order):
        if assigned[start]:
            continue
        assigned[start] = 1
        stack = [start]
        count = 0
        while stack:
            node = stack.pop()
            count += 1
            for neighbor in reverse[node]:
                if not assigned[neighbor]:
                    assigned[neighbor] = 1
                    stack.append(neighbor)
        sizes.append(count)
    require(sum(sizes) == node_count, "SCC node accounting")
    return sorted(sizes, reverse=True)


def graph_commitment(nodes: list[dict], edges: list[dict], boundary: list[dict]) -> str:
    digest = hashlib.sha256()
    for label, fields, rows in (
        ("NODES", NODE_FIELDS, nodes),
        ("EDGES", EDGE_FIELDS, edges),
        ("BOUNDARY", BOUNDARY_FIELDS, boundary),
    ):
        digest.update((label + "\n").encode())
        digest.update(("\x1f".join(fields) + "\n").encode())
        for row in rows:
            digest.update(("\x1f".join(str(row[field]) for field in fields) + "\n").encode())
    return digest.hexdigest()


def build_graph() -> dict:
    split = load_split()
    module, cells, drivers, cell_outputs, net_names = parse_netlist()
    primary_outputs = output_names(module)
    top_inputs = input_names(module)
    nodes, cell_to_node, bit_to_node = load_legal_sites(
        split, cells, cell_outputs, net_names, primary_outputs
    )
    require(set(bit_to_node) == set(drivers), "legal-site/output-driver bit bijection")
    require(set(cell_to_node) == set(cells), "legal-site/cell bijection")
    edges, boundary, branch_stats = build_edges(
        nodes, cells, cell_to_node, bit_to_node, top_inputs
    )
    degree_stats = enrich_nodes(nodes, edges)
    commitment = graph_commitment(nodes, edges, boundary)
    return {
        "nodes": nodes,
        "edges": edges,
        "boundary": boundary,
        "branch_stats": branch_stats,
        "degree_stats": degree_stats,
        "commitment": commitment,
        "top_input_bit_count": len(top_inputs),
        "top_output_bit_count": len(primary_outputs),
    }


def verify_readiness() -> None:
    policy = load_json(GNN_READINESS_POLICY)
    audit = load_json(READINESS_AUDIT)
    split_manifest = load_json(SITE_SPLIT_MANIFEST)
    require(policy.get("status") == "PASS", "11C-5L GNN-readiness policy status")
    require(policy.get("readiness_status") == "READY_FOR_GRAPH_DATASET_CONSTRUCTION_ONLY", "graph readiness")
    require(policy.get("graph_dataset_construction_authorized") is True, "graph construction authorization")
    require(policy.get("gnn_training_authorized") is False, "prior GNN training gate")
    require(policy.get("hybrid_training_authorized") is False, "prior hybrid training gate")
    require(audit.get("status") == "PASS", "11C-5L audit status")
    require(audit.get("disposition_status") == "FROZEN", "11C-5L disposition freeze")
    require(audit.get("gnn_readiness_status") == "FROZEN", "11C-5L readiness freeze")
    require(audit.get("legal_graph_nodes_expected") == EXPECTED_NODES, "expected graph nodes")
    require(audit.get("persistent_fault_instances") == EXPECTED_FAULT_INSTANCES, "fault instances")
    require(audit.get("validation_vectors_exposed") == 0, "validation exposure")
    require(audit.get("holdout_vectors_exposed") == 0, "holdout exposure")
    require(split_manifest.get("overlapping_sites") == 0, "site split overlap")
    require(split_manifest.get("sa0_sa1_separations") == 0, "SA0/SA1 split separation")


def verify_outputs_absent() -> None:
    for path in (
        GRAPH_CONTRACT,
        NODE_TABLE,
        EDGE_TABLE,
        BOUNDARY_TABLE,
        GRAPH_NPZ,
        GRAPH_SCHEMA,
        MANIFEST,
        AUDIT,
    ):
        require(not path.exists(), f"refusing to overwrite Stage 11D-1A output: {path}")


def cleanup_partial_outputs() -> None:
    if AUDIT.exists():
        return
    for path in (
        GRAPH_CONTRACT,
        NODE_TABLE,
        EDGE_TABLE,
        BOUNDARY_TABLE,
        GRAPH_NPZ,
        GRAPH_SCHEMA,
        MANIFEST,
    ):
        if path.exists():
            path.unlink()
        temporary = path.with_name(path.name + ".tmp")
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    require(ROOT.name == "vlsi_fault_detection_v2", f"run from project root, not {ROOT}")
    require(SOURCE.parent == ROOT, "place Stage 11D-1A script in project root")
    verify_outputs_absent()

    print("STAGE 11D-1A — GOLDEN-NETLIST GRAPH DATASET CONSTRUCTION")
    print("FROZEN INPUT VERIFICATION", flush=True)
    inputs = {
        relative(path): verify_input(path, expected_sha)
        for path, expected_sha in EXPECTED_INPUTS.items()
    }
    verify_readiness()
    print("  Stage 11C-5L semantic authorization                                      : PASS")

    print("\nBUILDING CANONICAL GRAPH", flush=True)
    graph = build_graph()
    print("  Legal nodes mapped        :", len(graph["nodes"]))
    print("  Collapsed directed edges  :", len(graph["edges"]))
    print("  Internal edge branches    :", graph["branch_stats"]["raw_internal_branches"])
    print("  Boundary input branches   :", len(graph["boundary"]))

    print("\nDETERMINISTIC IN-MEMORY REPLAY", flush=True)
    replay = build_graph()
    require(graph["commitment"] == replay["commitment"], "graph replay commitment")
    require(graph["nodes"] == replay["nodes"], "graph replay nodes")
    require(graph["edges"] == replay["edges"], "graph replay edges")
    require(graph["boundary"] == replay["boundary"], "graph replay boundary inputs")
    print("  Exact replay              : PASS")
    print("  Graph commitment          :", graph["commitment"])

    weak_sizes = weak_components(EXPECTED_NODES, graph["edges"])
    scc_sizes = strongly_connected_components(EXPECTED_NODES, graph["edges"])
    cyclic_sccs = sum(size > 1 for size in scc_sizes)
    self_loops = graph["branch_stats"]["self_loop_edges"]

    driver_types = sorted({node["driver_cell_type"] for node in graph["nodes"]})
    driver_type_code = {name: index for index, name in enumerate(driver_types)}
    categories = sorted({node["site_category"] for node in graph["nodes"]})
    require(set(categories) == set(SITE_CATEGORY_CODE), "site-category vocabulary")

    GRAPH_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_csv(NODE_TABLE, NODE_FIELDS, graph["nodes"])
    atomic_csv(EDGE_TABLE, EDGE_FIELDS, graph["edges"])
    atomic_csv(BOUNDARY_TABLE, BOUNDARY_FIELDS, graph["boundary"])

    edge_index = np.asarray(
        [[edge["source_node_index"] for edge in graph["edges"]],
         [edge["destination_node_index"] for edge in graph["edges"]]],
        dtype="<i4",
    )
    arrays = {
        "edge_index": edge_index,
        "edge_multiplicity": np.asarray([edge["multiplicity"] for edge in graph["edges"]], dtype="<u2"),
        "edge_data_branches": np.asarray([edge["data_branches"] for edge in graph["edges"]], dtype="<u2"),
        "edge_select_branches": np.asarray([edge["select_branches"] for edge in graph["edges"]], dtype="<u2"),
        "edge_enable_branches": np.asarray([edge["enable_branches"] for edge in graph["edges"]], dtype="<u2"),
        "edge_clock_branches": np.asarray([edge["clock_branches"] for edge in graph["edges"]], dtype="<u2"),
        "edge_reset_branches": np.asarray([edge["reset_branches"] for edge in graph["edges"]], dtype="<u2"),
        "edge_other_branches": np.asarray([edge["other_branches"] for edge in graph["edges"]], dtype="<u2"),
        "edge_is_self_loop": np.asarray([edge["is_self_loop"] for edge in graph["edges"]], dtype="u1"),
        "node_site_index": np.asarray([node["site_index"] for node in graph["nodes"]], dtype="<u4"),
        "node_partition_code": np.asarray([PARTITION_CODE[node["partition"]] for node in graph["nodes"]], dtype="u1"),
        "node_partition_rank": np.asarray([node["partition_rank"] for node in graph["nodes"]], dtype="<u4"),
        "node_driver_type_code": np.asarray([driver_type_code[node["driver_cell_type"]] for node in graph["nodes"]], dtype="u1"),
        "node_site_category_code": np.asarray([SITE_CATEGORY_CODE[node["site_category"]] for node in graph["nodes"]], dtype="u1"),
        "node_cell_fanout": np.asarray([node["cell_fanout"] for node in graph["nodes"]], dtype="<u4"),
        "node_is_primary_output": np.asarray([node["is_primary_output_stem"] for node in graph["nodes"]], dtype="u1"),
        "node_in_degree_branches": np.asarray([node["in_degree_branches"] for node in graph["nodes"]], dtype="<u4"),
        "node_out_degree_branches": np.asarray([node["out_degree_branches"] for node in graph["nodes"]], dtype="<u4"),
        "node_in_neighbor_count": np.asarray([node["in_neighbor_count"] for node in graph["nodes"]], dtype="<u4"),
        "node_out_neighbor_count": np.asarray([node["out_neighbor_count"] for node in graph["nodes"]], dtype="<u4"),
    }
    deterministic_npz(GRAPH_NPZ, arrays)

    schema = {
        "stage": "11D-1A",
        "status": "PASS",
        "schema_version": SCHEMA_VERSION,
        "graph_type": "DIRECTED_COLLAPSED_MULTIGRAPH",
        "node_definition": "ONE_NODE_PER_LEGAL_PERSISTENT_CELL_OUTPUT_NET_STEM",
        "edge_definition": "GOLDEN_NETLIST_DRIVER_TO_CELL_INPUT_SINK",
        "parallel_edge_policy": "COLLAPSED_WITH_MULTIPLICITY_AND_ROLE_COUNTS",
        "self_loop_policy": "RETAINED_AND_FLAGGED",
        "node_indexing": "ZERO_BASED_SITE_INDEX_MINUS_ONE",
        "physical_site_count": EXPECTED_NODES,
        "fault_instance_count": EXPECTED_FAULT_INSTANCES,
        "partition_codes": PARTITION_CODE,
        "site_category_codes": SITE_CATEGORY_CODE,
        "driver_cell_type_codes": driver_type_code,
        "edge_roles": list(EDGE_ROLES),
        "npz_arrays": {
            name: {"shape": list(value.shape), "dtype": value.dtype.str}
            for name, value in sorted(arrays.items())
        },
        "node_table_fields": NODE_FIELDS,
        "edge_table_fields": EDGE_FIELDS,
        "boundary_table_fields": BOUNDARY_FIELDS,
        "target_in_graph_features": False,
        "post_simulation_features": 0,
        "validation_vectors_exposed": 0,
        "holdout_vectors_exposed": 0,
    }
    atomic_json(GRAPH_SCHEMA, schema)

    created_at = datetime.now(timezone.utc).isoformat()
    contract = {
        "stage": "11D-1A",
        "title": "GOLDEN-NETLIST GRAPH DATASET CONTRACT",
        "status": "PASS",
        "contract_status": "FROZEN",
        "created_at_utc": created_at,
        "builder_version": VERSION,
        "source_netlist": record(GOLDEN_NETLIST),
        "top_module": TOP_MODULE,
        "node_count": EXPECTED_NODES,
        "fault_instance_count": EXPECTED_FAULT_INSTANCES,
        "graph_commitment": graph["commitment"],
        "topology": {
            "collapsed_directed_edges": len(graph["edges"]),
            "raw_internal_branches": graph["branch_stats"]["raw_internal_branches"],
            "boundary_input_branches": len(graph["boundary"]),
            "constant_input_branches": graph["branch_stats"]["constant_input_branches"],
            "self_loop_edges": self_loops,
            "weak_components": len(weak_sizes),
            "largest_weak_component": weak_sizes[0],
            "strongly_connected_components": len(scc_sizes),
            "largest_strongly_connected_component": scc_sizes[0],
            "cyclic_strongly_connected_components": cyclic_sccs,
        },
        "partition_counts": {
            name: sum(node["partition"] == name for node in graph["nodes"])
            for name in PARTITION_CODE
        },
        "sa0_sa1_grouping": "PRESERVED_BY_PHYSICAL_SITE",
        "target": "DETECTED",
        "target_present_in_graph_features": False,
        "post_simulation_features": 0,
        "graph_dataset_construction": "COMPLETE",
        "gnn_architecture_contract_generation_authorized": True,
        "gnn_training_authorized": False,
        "hybrid_training_authorized": False,
        "validation_vectors_exposed": 0,
        "holdout_vectors_exposed": 0,
    }
    atomic_json(GRAPH_CONTRACT, contract)

    outputs = {
        "graph_contract": record(GRAPH_CONTRACT),
        "node_table": record(NODE_TABLE),
        "edge_table": record(EDGE_TABLE),
        "boundary_table": record(BOUNDARY_TABLE),
        "graph_npz": record(GRAPH_NPZ),
        "graph_schema": record(GRAPH_SCHEMA),
    }
    manifest = {
        "stage": "11D-1A",
        "title": "GOLDEN-NETLIST GRAPH DATASET MANIFEST",
        "status": "PASS",
        "dataset_status": "FROZEN",
        "schema_status": "FROZEN",
        "topology_status": "FROZEN",
        "created_at_utc": created_at,
        "builder_version": VERSION,
        "builder_source": record(SOURCE),
        "graph_commitment": graph["commitment"],
        "node_count": len(graph["nodes"]),
        "collapsed_edge_count": len(graph["edges"]),
        "raw_internal_branch_count": graph["branch_stats"]["raw_internal_branches"],
        "boundary_input_branch_count": len(graph["boundary"]),
        "fault_instance_count": EXPECTED_FAULT_INSTANCES,
        "branch_statistics": graph["branch_stats"],
        "degree_statistics": graph["degree_stats"],
        "connectivity_statistics": {
            "weak_component_count": len(weak_sizes),
            "weak_component_sizes_descending": weak_sizes,
            "strong_component_count": len(scc_sizes),
            "strong_component_sizes_descending": scc_sizes,
            "cyclic_strong_component_count": cyclic_sccs,
        },
        "mapping_checks": {
            "legal_sites_mapped": len(graph["nodes"]),
            "cell_bijection": "PASS",
            "output_bit_bijection": "PASS",
            "site_index_coverage": "PASS",
            "fanout_matches": "PASS",
            "fanout_mismatches": graph["degree_stats"]["fanout_mismatches"],
            "undriven_input_branches": graph["branch_stats"]["undriven_input_branches"],
        },
        "deterministic_replay": "PASS",
        "input_evidence": inputs,
        "outputs": outputs,
        "validation_vectors_exposed": 0,
        "holdout_vectors_exposed": 0,
        "frozen_rtl_modified": False,
        "golden_netlist_modified": False,
        "legal_site_inventory_modified": False,
        "site_split_modified": False,
        "feature_matrices_modified": False,
        "baseline_model_modified": False,
        "deep_model_modified": False,
    }
    atomic_json(MANIFEST, manifest)
    manifest_record = record(MANIFEST)

    audit = {
        "stage": "11D-1A",
        "title": "GOLDEN-NETLIST GRAPH DATASET CONSTRUCTION AND TOPOLOGY-INTEGRITY FREEZE",
        "status": "PASS",
        "graph_dataset_status": "FROZEN",
        "graph_schema_status": "FROZEN",
        "topology_integrity_status": "PASS",
        "deterministic_replay": "PASS",
        "graph_commitment": graph["commitment"],
        "golden_netlist_cells": EXPECTED_NODES,
        "legal_nodes": len(graph["nodes"]),
        "persistent_fault_instances": EXPECTED_FAULT_INSTANCES,
        "collapsed_directed_edges": len(graph["edges"]),
        "raw_internal_branches": graph["branch_stats"]["raw_internal_branches"],
        "boundary_input_branches": len(graph["boundary"]),
        "constant_input_branches": graph["branch_stats"]["constant_input_branches"],
        "undriven_input_branches": graph["branch_stats"]["undriven_input_branches"],
        "fanout_mismatches": graph["degree_stats"]["fanout_mismatches"],
        "weak_components": len(weak_sizes),
        "largest_weak_component": weak_sizes[0],
        "strongly_connected_components": len(scc_sizes),
        "largest_strongly_connected_component": scc_sizes[0],
        "cyclic_strongly_connected_components": cyclic_sccs,
        "self_loop_edges": self_loops,
        "dev_train_nodes": EXPECTED_DEV_TRAIN,
        "dev_calibration_nodes": EXPECTED_DEV_CALIBRATION,
        "dev_site_test_nodes": EXPECTED_DEV_SITE_TEST,
        "overlapping_sites": 0,
        "sa0_sa1_separations": 0,
        "post_simulation_features": 0,
        "target_present_in_graph_features": False,
        "validation_vectors_exposed": 0,
        "holdout_vectors_exposed": 0,
        "gnn_architecture_contract_generation": "AUTHORIZED",
        "gnn_training": "NOT_YET_AUTHORIZED",
        "hybrid_training": "NOT_YET_AUTHORIZED",
        "frozen_rtl_modified": False,
        "golden_netlist_modified": False,
        "legal_site_inventory_modified": False,
        "site_split_modified": False,
        "feature_matrices_modified": False,
        "baseline_model_modified": False,
        "deep_model_modified": False,
        "outputs": outputs,
        "manifest": manifest_record,
        "next_gate": "STAGE 11D-1B — GNN MODEL ARCHITECTURE AND TRAINING-CONTRACT FREEZE",
    }
    atomic_json(AUDIT, audit)
    audit_record = record(AUDIT)

    print("\nSTAGE 11D-1A — GOLDEN-NETLIST GRAPH DATASET CONSTRUCTION AND TOPOLOGY-INTEGRITY FREEZE")
    print("Status                         : PASS")
    print("Graph dataset status           : FROZEN")
    print("Graph schema status            : FROZEN")
    print("Topology integrity             : PASS")
    print("Deterministic replay           : PASS")
    print("Golden netlist cells           :", EXPECTED_NODES)
    print("Legal graph nodes              :", len(graph["nodes"]))
    print("Persistent fault instances     :", EXPECTED_FAULT_INSTANCES)
    print("Collapsed directed edges       :", len(graph["edges"]))
    print("Raw internal branches          :", graph["branch_stats"]["raw_internal_branches"])
    print("Boundary input branches        :", len(graph["boundary"]))
    print("Constant input branches        :", graph["branch_stats"]["constant_input_branches"])
    print("Undriven input branches        : 0")
    print("Fanout mismatches              : 0")
    print("Weak components                :", len(weak_sizes))
    print("Largest weak component         :", weak_sizes[0])
    print("Strongly connected components  :", len(scc_sizes))
    print("Largest strong component       :", scc_sizes[0])
    print("Cyclic strong components       :", cyclic_sccs)
    print("Self-loop edges                :", self_loops)
    print("DEV_TRAIN nodes                :", EXPECTED_DEV_TRAIN)
    print("DEV_CALIBRATION nodes          :", EXPECTED_DEV_CALIBRATION)
    print("DEV_SITE_TEST nodes            :", EXPECTED_DEV_SITE_TEST)
    print("SA0/SA1 site separations       : 0")
    print("Post-simulation features       : 0")
    print("Target present in graph X      : NO")
    print("VALIDATION vectors exposed     : 0")
    print("HOLDOUT vectors exposed        : 0")
    print("GNN architecture contract      : AUTHORIZED")
    print("GNN training                   : NOT YET AUTHORIZED")
    print("Hybrid training                : NOT YET AUTHORIZED")
    print("Frozen RTL modified            : NO")
    print("Golden netlist modified        : NO")
    print("Graph commitment               :", graph["commitment"])
    print("Graph NPZ                      :", GRAPH_NPZ)
    print("Graph NPZ SHA                  :", outputs["graph_npz"]["sha256"])
    print("Graph schema                   :", GRAPH_SCHEMA)
    print("Graph schema SHA               :", outputs["graph_schema"]["sha256"])
    print("Manifest                       :", MANIFEST)
    print("Manifest SHA                   :", manifest_record["sha256"])
    print("Audit                          :", AUDIT)
    print("Audit SHA                      :", audit_record["sha256"])
    print("Next gate                      : STAGE 11D-1B — GNN MODEL ARCHITECTURE AND TRAINING-CONTRACT FREEZE")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        cleanup_partial_outputs()
        raise
