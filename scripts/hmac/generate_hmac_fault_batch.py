#!/usr/bin/env python3

from pathlib import Path
from datetime import datetime, timezone
import argparse
import copy
import hashlib
import json
import os
import tempfile


GENERATOR_VERSION = "HMAC-FAULT-BATCH-GENERATOR-v1"
GOLD_MODULE = "opentitan_hmac_sha256_msg32"
TOTAL_LEGAL_SITES = 22839
BATCH_SIZE = 512
TOTAL_BATCHES = 45
SELECTOR_WIDTH = 9
SELECTOR_CAPACITY = 512

EXPECTED_GOLDEN_SHA256 = (
    "0a33bb40ea331e8bc7fbc80b1c55c339"
    "a7687a694a650921222dd08775eb56a1"
)

EXPECTED_LEGAL_SITES_SHA256 = (
    "d9a39366bb35a277125376c6dc26c832"
    "73b45c0d7732fb93a3403401a071cb45"
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )

    temporary_path = Path(temporary_name)

    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())

        temporary_path.replace(path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def write_json(path, value):
    atomic_write_text(
        path,
        json.dumps(value, indent=2, sort_keys=True) + "\n",
    )


def collect_integer_bits(module):
    bits = []

    for port in module.get("ports", {}).values():
        bits.extend(
            bit for bit in port.get("bits", [])
            if isinstance(bit, int)
        )

    for net in module.get("netnames", {}).values():
        bits.extend(
            bit for bit in net.get("bits", [])
            if isinstance(bit, int)
        )

    for cell in module.get("cells", {}).values():
        for connection in cell.get(
            "connections",
            {},
        ).values():
            bits.extend(
                bit for bit in connection
                if isinstance(bit, int)
            )

    return bits


def generate(args):
    project_root = Path(args.project_root).resolve()

    golden_path = (
        project_root /
        "build/hmac_synth_11b2a/"
        "opentitan_hmac_sha256_msg32_generic.json"
    )

    legal_path = (
        project_root /
        "results/hmac_fault_campaign_11c1/"
        "hmac_legal_fault_sites_11c1b.json"
    )

    output_dir = Path(args.output_dir)

    if not output_dir.is_absolute():
        output_dir = project_root / output_dir

    result_dir = Path(args.result_dir)

    if not result_dir.is_absolute():
        result_dir = project_root / result_dir

    batch_id = args.batch_id

    if not 0 <= batch_id < TOTAL_BATCHES:
        raise SystemExit(
            f"STOP: batch ID must be 0-{TOTAL_BATCHES - 1}"
        )

    batch_tag = f"{batch_id:03d}"
    batch_module = (
        "opentitan_hmac_sha256_msg32_"
        f"faultbatch{batch_tag}"
    )

    json_path = (
        output_dir /
        f"{batch_module}.json"
    )

    mapping_path = (
        result_dir /
        f"hmac_fault_batch_{batch_tag}_mapping.json"
    )

    record_path = (
        result_dir /
        f"hmac_fault_batch_{batch_tag}_generation.json"
    )

    generated_paths = (
        json_path,
        mapping_path,
        record_path,
    )

    existing = [
        path for path in generated_paths
        if path.exists()
    ]

    if existing and not args.overwrite:
        raise SystemExit(
            "STOP: generated artifacts already exist; "
            "use --overwrite only for controlled reproduction:\n" +
            "\n".join(str(path) for path in existing)
        )

    for path, expected in (
        (golden_path, EXPECTED_GOLDEN_SHA256),
        (legal_path, EXPECTED_LEGAL_SITES_SHA256),
    ):
        if not path.is_file():
            raise SystemExit(
                f"STOP: required source missing: {path}"
            )

        actual = sha256(path)

        if actual != expected:
            raise SystemExit(
                f"STOP: SHA mismatch for {path}\n"
                f"Expected: {expected}\n"
                f"Actual:   {actual}"
            )

    design = json.loads(golden_path.read_text())
    legal_data = json.loads(legal_path.read_text())

    if GOLD_MODULE not in design.get("modules", {}):
        raise SystemExit(
            f"STOP: golden module missing: {GOLD_MODULE}"
        )

    if "sites" not in legal_data:
        raise SystemExit(
            "STOP: legal-site list 'sites' is missing"
        )

    legal_sites = legal_data["sites"]

    if len(legal_sites) != TOTAL_LEGAL_SITES:
        raise SystemExit(
            f"STOP: expected {TOTAL_LEGAL_SITES} sites, "
            f"found {len(legal_sites)}"
        )

    start = batch_id * BATCH_SIZE
    stop = min(start + BATCH_SIZE, TOTAL_LEGAL_SITES)
    selected_sites = legal_sites[start:stop]

    expected_site_count = stop - start

    if len(selected_sites) != expected_site_count:
        raise SystemExit(
            "STOP: selected-site count mismatch"
        )

    for offset, site in enumerate(selected_sites):
        expected_index = start + offset + 1
        expected_id = f"HMAC-STEM-{expected_index:06d}"

        if site.get("site_index") != expected_index:
            raise SystemExit(
                f"STOP: site-index mismatch at {offset}"
            )

        if site.get("fault_site_id") != expected_id:
            raise SystemExit(
                f"STOP: site-ID mismatch at {offset}"
            )

        if (
            site.get("persistent_injection_status") !=
            "AUTHORIZED"
        ):
            raise SystemExit(
                f"STOP: unauthorized site: {expected_id}"
            )

    module = copy.deepcopy(
        design["modules"].pop(GOLD_MODULE)
    )

    module.setdefault("ports", {})
    module.setdefault("cells", {})
    module.setdefault("netnames", {})
    module.setdefault("attributes", {})

    module["attributes"]["top"] = (
        "00000000000000000000000000000001"
    )

    original_cell_count = len(module["cells"])

    if original_cell_count != 22839:
        raise SystemExit(
            f"STOP: golden cell count is "
            f"{original_cell_count}"
        )

    integer_bits = collect_integer_bits(module)
    next_bit = [max(integer_bits) + 1]

    def allocate_bit():
        bit = next_bit[0]
        next_bit[0] += 1
        return bit

    def add_netname(name, bits, hide_name=0):
        if name in module["netnames"]:
            raise SystemExit(
                f"STOP: duplicate netname: {name}"
            )

        module["netnames"][name] = {
            "hide_name": hide_name,
            "bits": list(bits),
            "attributes": {},
        }

    def add_port(name, direction, width):
        if name in module["ports"]:
            raise SystemExit(
                f"STOP: duplicate port: {name}"
            )

        bits = [
            allocate_bit()
            for _ in range(width)
        ]

        module["ports"][name] = {
            "direction": direction,
            "bits": bits,
        }

        add_netname(name, bits)

        return bits

    def add_cell(name, cell_type, connections, directions):
        if name in module["cells"]:
            raise SystemExit(
                f"STOP: duplicate cell: {name}"
            )

        module["cells"][name] = {
            "hide_name": 0,
            "type": cell_type,
            "parameters": {},
            "attributes": {},
            "port_directions": directions,
            "connections": {
                key: list(value)
                for key, value in connections.items()
            },
        }

    prefix = f"$faultbatch{batch_tag}$"

    fault_enable_bit = add_port(
        "fault_enable_i",
        "input",
        1,
    )[0]

    fault_selector_bits = add_port(
        "fault_selector_i",
        "input",
        SELECTOR_WIDTH,
    )

    fault_value_bit = add_port(
        "fault_value_i",
        "input",
        1,
    )[0]

    fault_raw_output_bit = add_port(
        "fault_raw_o",
        "output",
        1,
    )[0]

    inverted_selector_bits = []

    for bit_index, selector_bit in enumerate(
        fault_selector_bits
    ):
        inverted_bit = allocate_bit()
        inverted_selector_bits.append(inverted_bit)

        add_cell(
            f"{prefix}sel_inv${bit_index}",
            "$_NOT_",
            {
                "A": [selector_bit],
                "Y": [inverted_bit],
            },
            {
                "A": "input",
                "Y": "output",
            },
        )

    add_netname(
        f"{prefix}selector_inverted",
        inverted_selector_bits,
        hide_name=1,
    )

    decoder_level = [fault_enable_bit]
    decoder_cell_count = 0

    for selector_bit_index in reversed(
        range(SELECTOR_WIDTH)
    ):
        next_level = []

        for node_index, parent_bit in enumerate(
            decoder_level
        ):
            zero_bit = allocate_bit()
            one_bit = allocate_bit()

            add_cell(
                (
                    f"{prefix}decode$"
                    f"{selector_bit_index}$"
                    f"{node_index}$0"
                ),
                "$_AND_",
                {
                    "A": [parent_bit],
                    "B": [
                        inverted_selector_bits[
                            selector_bit_index
                        ]
                    ],
                    "Y": [zero_bit],
                },
                {
                    "A": "input",
                    "B": "input",
                    "Y": "output",
                },
            )

            add_cell(
                (
                    f"{prefix}decode$"
                    f"{selector_bit_index}$"
                    f"{node_index}$1"
                ),
                "$_AND_",
                {
                    "A": [parent_bit],
                    "B": [
                        fault_selector_bits[
                            selector_bit_index
                        ]
                    ],
                    "Y": [one_bit],
                },
                {
                    "A": "input",
                    "B": "input",
                    "Y": "output",
                },
            )

            decoder_cell_count += 2
            next_level.extend([zero_bit, one_bit])

        decoder_level = next_level

    decoder_leaves = decoder_level

    if len(decoder_leaves) != SELECTOR_CAPACITY:
        raise SystemExit(
            "STOP: decoder does not have 512 leaves"
        )

    if decoder_cell_count != 1022:
        raise SystemExit(
            f"STOP: decoder cell count is "
            f"{decoder_cell_count}"
        )

    add_netname(
        f"{prefix}fault_decode",
        decoder_leaves,
        hide_name=1,
    )

    raw_driver_bits = []
    mapping_records = []

    for selector_code, site in enumerate(selected_sites):
        driver_name = site["driver_cell"]
        driver_port = site["driver_port"]
        pin_index = int(site["driver_pin_index"])
        forced_bit = int(site["bit_id"])

        if driver_name not in module["cells"]:
            raise SystemExit(
                f"STOP: driver missing: {driver_name}"
            )

        driver = module["cells"][driver_name]

        if driver_port not in driver["connections"]:
            raise SystemExit(
                f"STOP: port missing: "
                f"{driver_name}.{driver_port}"
            )

        connection = driver["connections"][driver_port]

        if not 0 <= pin_index < len(connection):
            raise SystemExit(
                f"STOP: invalid pin index for "
                f"{site['fault_site_id']}"
            )

        if connection[pin_index] != forced_bit:
            raise SystemExit(
                f"STOP: driver-bit mismatch for "
                f"{site['fault_site_id']}"
            )

        raw_bit = allocate_bit()
        connection[pin_index] = raw_bit
        raw_driver_bits.append(raw_bit)

        injection_name = (
            f"{prefix}inject${selector_code:03d}"
        )

        add_cell(
            injection_name,
            "$_MUX_",
            {
                "A": [raw_bit],
                "B": [fault_value_bit],
                "S": [decoder_leaves[selector_code]],
                "Y": [forced_bit],
            },
            {
                "A": "input",
                "B": "input",
                "S": "input",
                "Y": "output",
            },
        )

        mapping_records.append({
            "batch_id": batch_id,
            "batch_name": f"HMAC-BATCH-{batch_tag}",
            "selector_code": selector_code,
            "selector_binary": (
                f"{selector_code:09b}"
            ),
            "fault_site_id": site["fault_site_id"],
            "site_index": site["site_index"],
            "site_category": site["site_category"],
            "driver_cell": driver_name,
            "driver_cell_type":
                site["driver_cell_type"],
            "driver_port": driver_port,
            "driver_pin_index": pin_index,
            "net_name": site["net_name"],
            "cell_fanout": site["cell_fanout"],
            "primary_outputs": site["primary_outputs"],
            "forced_net_bit": forced_bit,
            "raw_driver_bit": raw_bit,
            "decoder_leaf_bit":
                decoder_leaves[selector_code],
            "injection_cell": injection_name,
        })

    add_netname(
        f"{prefix}site_raw",
        raw_driver_bits,
        hide_name=1,
    )

    raw_level = (
        list(raw_driver_bits) +
        ["0"] * (
            SELECTOR_CAPACITY -
            len(raw_driver_bits)
        )
    )

    raw_mux_cell_count = 0

    for selector_bit_index in range(SELECTOR_WIDTH):
        next_level = []

        for pair_index in range(0, len(raw_level), 2):
            final_mux = (
                len(raw_level) == 2 and
                pair_index == 0
            )

            output_bit = (
                fault_raw_output_bit
                if final_mux
                else allocate_bit()
            )

            add_cell(
                (
                    f"{prefix}rawmux$"
                    f"{selector_bit_index}$"
                    f"{pair_index // 2}"
                ),
                "$_MUX_",
                {
                    "A": [raw_level[pair_index]],
                    "B": [raw_level[pair_index + 1]],
                    "S": [
                        fault_selector_bits[
                            selector_bit_index
                        ]
                    ],
                    "Y": [output_bit],
                },
                {
                    "A": "input",
                    "B": "input",
                    "S": "input",
                    "Y": "output",
                },
            )

            raw_mux_cell_count += 1
            next_level.append(output_bit)

        raw_level = next_level

    if raw_level != [fault_raw_output_bit]:
        raise SystemExit(
            "STOP: raw monitor tree is incomplete"
        )

    if raw_mux_cell_count != 511:
        raise SystemExit(
            f"STOP: raw mux count is "
            f"{raw_mux_cell_count}"
        )

    instrumentation_cells = (
        SELECTOR_WIDTH +
        decoder_cell_count +
        len(selected_sites) +
        raw_mux_cell_count
    )

    expected_total_cells = (
        original_cell_count +
        instrumentation_cells
    )

    if len(module["cells"]) != expected_total_cells:
        raise SystemExit(
            f"STOP: total cell count mismatch: "
            f"{len(module['cells'])} != "
            f"{expected_total_cells}"
        )

    design["modules"][batch_module] = module

    mapping = {
        "generator_version": GENERATOR_VERSION,
        "batch_id": batch_id,
        "batch_tag": batch_tag,
        "valid_site_count": len(selected_sites),
        "selector_width": SELECTOR_WIDTH,
        "valid_selector_first": 0,
        "valid_selector_last": len(selected_sites) - 1,
        "invalid_selector_first": len(selected_sites),
        "invalid_selector_last": 511,
        "sites": mapping_records,
    }

    write_json(json_path, design)
    write_json(mapping_path, mapping)

    generator_path = Path(__file__).resolve()

    record = {
        "generator_version": GENERATOR_VERSION,
        "status": "PASS",
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "batch_id": batch_id,
        "batch_tag": batch_tag,
        "first_site": (
            selected_sites[0]["fault_site_id"]
        ),
        "last_site": (
            selected_sites[-1]["fault_site_id"]
        ),
        "valid_sites": len(selected_sites),
        "fault_instances": len(selected_sites) * 2,
        "invalid_selectors": (
            SELECTOR_CAPACITY -
            len(selected_sites)
        ),
        "original_cells": original_cell_count,
        "selector_inverters": SELECTOR_WIDTH,
        "decoder_cells": decoder_cell_count,
        "injection_muxes": len(selected_sites),
        "raw_monitor_muxes": raw_mux_cell_count,
        "instrumentation_cells":
            instrumentation_cells,
        "total_cells": len(module["cells"]),
        "source_sha256": {
            "generator": sha256(generator_path),
            "golden_json": sha256(golden_path),
            "legal_sites": sha256(legal_path),
        },
        "generated_sha256": {
            "batch_json": sha256(json_path),
            "mapping": sha256(mapping_path),
        },
        "frozen_rtl_modified": False,
        "golden_netlist_modified": False,
    }

    write_json(record_path, record)

    print("HMAC REUSABLE FAULT-BATCH GENERATOR")
    print("Generator version   :", GENERATOR_VERSION)
    print("Status              : PASS")
    print("Batch ID            :", batch_id)
    print("Sites               :", len(selected_sites))
    print("Fault instances     :", len(selected_sites) * 2)
    print(
        "Invalid selectors   :",
        SELECTOR_CAPACITY - len(selected_sites),
    )
    print("Total cells         :", len(module["cells"]))
    print("Batch JSON          :", json_path)
    print("Batch JSON SHA      :", sha256(json_path))
    print("Mapping             :", mapping_path)
    print("Mapping SHA         :", sha256(mapping_path))
    print("Generation record   :", record_path)
    print("Record SHA          :", sha256(record_path))


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Generate one deterministic instrumented "
            "OpenTitan HMAC fault batch."
        )
    )

    parser.add_argument(
        "--batch-id",
        required=True,
        type=int,
    )

    parser.add_argument(
        "--project-root",
        default=str(
            Path(__file__).resolve().parents[2]
        ),
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--result-dir",
        required=True,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_arguments())
