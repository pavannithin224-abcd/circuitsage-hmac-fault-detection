#!/usr/bin/env python3
"""Stage 11E-1I — reproducible GitHub release bundle and sanitization freeze.

Builds an explicit, size-bounded release archive from frozen public artifacts.
It never contacts GitHub, trains or quantizes a model, runs inference, reads
HOLDOUT payloads, or copies private blinded-case material.
"""
import argparse
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tarfile
import tempfile
import zipfile


VERSION = "HMAC-REPRODUCIBLE-RELEASE-BUNDLE-v1"
RELEASE_NAME = "opentitan-hmac-vlsi-fault-detection-v1"
RESULT_ROOT = Path("results/hmac_project_release_11e1i")
README_TEMPLATE = Path("README_HMAC_VLSI_FAULT_DETECTION.md")
README_SHA = "fbdb5793c40a485b61aba4dfcd337462ff3e6c72d0e4b89258936a0d76f83341"

FROZEN_INPUTS = {
    "stage_11e1h_final_project_release_readiness.py":
        "1a28f4f56118af33572e64d23a6bb15281aa03ac1ad40c9e2562b5d9e312a9bc",
    "config/release/hmac_final_project_disposition_11e1h.json":
        "32f27f32e1a6102304e42a8cc9c886702fddaa65ce4081251ecb95edb4f5d6d7",
    "config/release/hmac_reproducible_release_policy_11e1h.json":
        "140419afe8d59989372cffb41188e1247d87892d6db8ad6931f95e95596f6b3e",
    "results/hmac_project_final_11e1h/hmac_reproducible_release_inventory_11e1h.json":
        "974f45a487e88f39bd458bd24f85cd3fc16182e9e1b6312b3041360eb5c9d275",
    "results/hmac_project_final_11e1h/hmac_final_research_prototype_model_card_11e1h.md":
        "5a4a5849a7b3caeafa9c125623b72e2ad0d086bc1f9abcb8f991d297449bbc8a",
    "results/hmac_project_final_11e1h/hmac_final_project_release_readiness_freeze_11e1h.json":
        "f7ad50d01f608614f634ee970b8f410b22b1cb632929a37122c8b66d45f95360",
    "results/hmac_fault_campaign_11d2/hybrid_training_11d2b/hmac_selected_hybrid_model_11d2b.joblib":
        "12fea5eabf4a6c605325b6ce4c1217f6a37cc59750065db8b85c3da14713f6b8",
    "config/diagnostic_model/hmac_final_diagnostic_model_lock_11d2d.json":
        "7985b534c62d93717a179d6d2b20f247a531ec0452003e35f0f65cf640d0b30d",
    "config/diagnostic_model/hmac_hybrid_architecture_11d2a.json":
        "d213a89c53769c217ca9cca71700b03abc23f94d5302bc4309ea63b035016063",
    "config/diagnostic_model/hmac_hybrid_training_contract_11d2a.json":
        "1ccd69a6ee179dbdbce9cc295324c85f28bc4fd98f22b5df010fdcbf21cda41e",
    "results/hmac_fault_campaign_11d2/hmac_hybrid_environment_11d2a.json":
        "1849274a8006e77cced8695be784e70102abceec940fee036163f0a62aa65a57",
    "results/hmac_fault_campaign_11c5/feature_matrix_11c5e/hmac_leakage_safe_feature_matrix_schema_11c5e.json":
        "0bf1edffb8078b003a1116b276615d5544979d007712ab49870b1b93784e486c",
    "results/hmac_fault_campaign_11d1/graph_dataset_11d1a/hmac_golden_netlist_graph_schema_11d1a.json":
        "d24c3dd285891e284d20d1adcd93f793bedfd3042f20159a8a486359bc25e393",
    "results/hmac_fault_campaign_11d3/canonical_dataset_11d3f/hmac_canonical_validation_dataset_schema_11d3f.json":
        "9c2fe65ea4e980ae7676c9a30457180b22783301710c81f2eea9fdef1ae57c22",
    "results/hmac_fault_campaign_11d3/feature_matrix_11d3g/hmac_validation_feature_schema_11d3g.json":
        "098d1df2aeba04085d8d70e733d69a96142ab7864bf3e942acb8ce535a4fc693",
    "results/hmac_fault_campaign_11d3/hmac_validation_failure_review_11d3j.md":
        "bf4c621a2c90bb4751a7305c7e128b88a7cb015d93cbdd61c17585d6876be774",
    "results/hmac_behavior_diagnosis_pilot/hmac_behavior_diagnosis_pilot_report_11e1g.md":
        "588b58ea47b046b99325daa912d6c370999a80346d0d39daf86a82ddbeb4f06b",
}

PRIVATE_COMPONENTS = {
    ".env", "entropy.json", "truth.json", "capture_plan.json", ".git",
    ".venv", "__pycache__", "private",
}

SECRET_PATTERNS = {
    "private key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "GitHub token": re.compile(rb"(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}"),
    "OpenAI key": re.compile(rb"sk-[A-Za-z0-9_-]{20,}"),
    "Hugging Face token": re.compile(rb"hf_[A-Za-z0-9]{20,}"),
    "AWS access key": re.compile(rb"AKIA[0-9A-Z]{16}"),
}

APACHE_2 = """Apache License
Version 2.0, January 2004
http://www.apache.org/licenses/

TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

1. Definitions.

"License" shall mean the terms and conditions for use, reproduction, and
distribution as defined by Sections 1 through 9 of this document.

"Licensor" shall mean the copyright owner or entity authorized by the
copyright owner that is granting the License.

"Legal Entity" shall mean the union of the acting entity and all other
entities that control, are controlled by, or are under common control with
that entity. For the purposes of this definition, "control" means (i) the
power, direct or indirect, to cause the direction or management of such
entity, whether by contract or otherwise, or (ii) ownership of fifty percent
(50%) or more of the outstanding shares, or (iii) beneficial ownership of
such entity.

"You" (or "Your") shall mean an individual or Legal Entity exercising
permissions granted by this License.

"Source" form shall mean the preferred form for making modifications,
including but not limited to software source code, documentation source, and
configuration files.

"Object" form shall mean any form resulting from mechanical transformation
or translation of a Source form, including but not limited to compiled object
code, generated documentation, and conversions to other media types.

"Work" shall mean the work of authorship, whether in Source or Object form,
made available under the License, as indicated by a copyright notice that is
included in or attached to the work.

"Derivative Works" shall mean any work, whether in Source or Object form,
that is based on (or derived from) the Work and for which the editorial
revisions, annotations, elaborations, or other modifications represent, as a
whole, an original work of authorship. Derivative Works shall not include
works that remain separable from, or merely link (or bind by name) to the
interfaces of, the Work and Derivative Works thereof.

"Contribution" shall mean any work of authorship, including the original
version of the Work and any modifications or additions to that Work or
Derivative Works thereof, that is intentionally submitted to Licensor for
inclusion in the Work by the copyright owner or by an individual or Legal
Entity authorized to submit on behalf of the copyright owner.

"Contributor" shall mean Licensor and any individual or Legal Entity on
behalf of whom a Contribution has been received by Licensor and subsequently
incorporated within the Work.

2. Grant of Copyright License. Subject to the terms and conditions of this
License, each Contributor hereby grants to You a perpetual, worldwide,
non-exclusive, no-charge, royalty-free, irrevocable copyright license to
reproduce, prepare Derivative Works of, publicly display, publicly perform,
sublicense, and distribute the Work and such Derivative Works in Source or
Object form.

3. Grant of Patent License. Subject to the terms and conditions of this
License, each Contributor hereby grants to You a perpetual, worldwide,
non-exclusive, no-charge, royalty-free, irrevocable (except as stated in this
section) patent license to make, have made, use, offer to sell, sell, import,
and otherwise transfer the Work, where such license applies only to those
patent claims licensable by such Contributor that are necessarily infringed
by their Contribution(s) alone or by combination of their Contribution(s)
with the Work to which such Contribution(s) was submitted. If You institute
patent litigation against any entity alleging that the Work or a Contribution
incorporated within the Work constitutes direct or contributory patent
infringement, then any patent licenses granted to You under this License for
that Work shall terminate as of the date such litigation is filed.

4. Redistribution. You may reproduce and distribute copies of the Work or
Derivative Works thereof in any medium, with or without modifications, and in
Source or Object form, provided that You meet the following conditions:

(a) You must give any other recipients of the Work or Derivative Works a copy
of this License; and

(b) You must cause any modified files to carry prominent notices stating that
You changed the files; and

(c) You must retain, in the Source form of any Derivative Works that You
distribute, all copyright, patent, trademark, and attribution notices from
the Source form of the Work, excluding those notices that do not pertain to
any part of the Derivative Works; and

(d) If the Work includes a "NOTICE" text file as part of its distribution,
then any Derivative Works that You distribute must include a readable copy of
the attribution notices contained within such NOTICE file.

You may add Your own copyright statement to Your modifications and may provide
additional or different license terms and conditions for use, reproduction,
or distribution of Your modifications, or for any such Derivative Works as a
whole, provided Your use, reproduction, and distribution of the Work otherwise
complies with the conditions stated in this License.

5. Submission of Contributions. Unless You explicitly state otherwise, any
Contribution intentionally submitted for inclusion in the Work by You to the
Licensor shall be under the terms and conditions of this License, without any
additional terms or conditions.

6. Trademarks. This License does not grant permission to use the trade names,
trademarks, service marks, or product names of the Licensor, except as required
for reasonable and customary use in describing the origin of the Work and
reproducing the content of the NOTICE file.

7. Disclaimer of Warranty. Unless required by applicable law or agreed to in
writing, Licensor provides the Work (and each Contributor provides its
Contributions) on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied, including, without limitation, any warranties
or conditions of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
PARTICULAR PURPOSE. You are solely responsible for determining the
appropriateness of using or redistributing the Work and assume any risks
associated with Your exercise of permissions under this License.

8. Limitation of Liability. In no event and under no legal theory, whether in
tort (including negligence), contract, or otherwise, unless required by
applicable law, shall any Contributor be liable to You for damages, including
any direct, indirect, special, incidental, or consequential damages arising as
a result of this License or out of the use or inability to use the Work.

9. Accepting Warranty or Additional Liability. While redistributing the Work
or Derivative Works thereof, You may choose to offer support, warranty,
indemnity, or other liability obligations and/or rights consistent with this
License. However, in accepting such obligations, You may act only on Your own
behalf and on Your sole responsibility, not on behalf of any other Contributor.

END OF TERMS AND CONDITIONS

Copyright 2026 Pavan Nithin

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy at
http://www.apache.org/licenses/LICENSE-2.0
"""

NOTICE = """# Third-party notice

This research project integrates and evaluates a wrapper around OpenTitan's
HMAC/SHA-256 RTL. OpenTitan is an upstream third-party project and is not
authored or owned by Pavan Nithin.

- Upstream project: OpenTitan
- Pinned revision: `83fc48ed3a727399056772d12be8c7d4a8a276f0`
- Upstream repository: https://github.com/lowRISC/opentitan

OpenTitan source is not copied into this release bundle. Obtain the pinned
revision directly from the upstream repository and follow its licenses and
notices. Python packages and OSS-CAD-Suite tools also retain their respective
licenses. The top-level project license applies only where Pavan Nithin owns
the copyright or has authority to license the contribution.
"""

CITATION = """cff-version: 1.2.0
message: "If you use this research prototype, please cite this software release."
title: "Graph-Aware Fault Detection for an OpenTitan HMAC Circuit"
type: software
authors:
  - family-names: "Nithin"
    given-names: "Pavan"
version: "1.0.0-research"
date-released: "2026-09-13"
license: "Apache-2.0"
repository-code: "ADD_GITHUB_REPOSITORY_URL_BEFORE_PUBLICATION"
keywords:
  - VLSI fault detection
  - OpenTitan
  - HMAC
  - graph neural network
  - stuck-at fault
"""

SECURITY = """# Security

Do not load untrusted `.joblib` or pickle files. Python object deserialization
can execute code. Verify the released model against `SHA256SUMS` before loading
it. Please report suspected credential exposure or malicious artifacts
privately to the repository maintainer instead of opening a public issue.
"""

GITIGNORE = """.venv/
__pycache__/
*.py[cod]
.env
.env.*
build/
third_party/
results/**/private/
**/entropy.json
**/truth.json
**/capture_plan.json
*.vcd
*.fst
*.log
"""


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def sanitize_json(value, root_string):
    if isinstance(value, dict):
        return {key: sanitize_json(item, root_string) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_json(item, root_string) for item in value]
    if isinstance(value, str):
        return value.replace(root_string, "${PROJECT_ROOT}")
    return value


def safe_destination(relative):
    parts = Path(relative).parts
    require(parts and not Path(relative).is_absolute() and ".." not in parts,
            "unsafe release destination")
    require(not any(part in PRIVATE_COMPONENTS for part in parts),
            "private destination component")


def copy_public(root, source, release_root, destination, source_records):
    source_path = (root / source).resolve()
    require(source_path.is_relative_to(root) and source_path.is_file() and
            not source_path.is_symlink(), "missing or unsafe release source: " + source)
    safe_destination(destination)
    destination_path = release_root / destination
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path.suffix == ".json":
        parsed = json.loads(source_path.read_text())
        write(destination_path, encode(sanitize_json(parsed, str(root))))
    else:
        shutil.copyfile(source_path, destination_path)
    source_records[destination] = {
        "source_path": source,
        "source_sha256": sha(source_path),
        "released_sha256": sha(destination_path),
        "released_bytes": destination_path.stat().st_size,
        "sanitized": sha(source_path) != sha(destination_path),
    }


def scan_release(release_root):
    files = []
    for path in sorted(release_root.rglob("*")):
        require(not path.is_symlink(), "release contains symlink: " + str(path))
        if path.is_dir():
            continue
        relative = path.relative_to(release_root)
        require(not any(part in PRIVATE_COMPONENTS for part in relative.parts),
                "release contains blocked path: " + str(relative))
        require(path.stat().st_size <= 50 * 1024 * 1024,
                "release file exceeds 50 MiB: " + str(relative))
        data = path.read_bytes()
        linux_home_marker = b"/home/" + b"pavan/"
        windows_home_marker = b"C:" + bytes([92]) + b"Users" + bytes([92])
        require(linux_home_marker not in data and windows_home_marker not in data,
                "machine-specific home path remains: " + str(relative))
        for label, pattern in SECRET_PATTERNS.items():
            require(pattern.search(data) is None,
                    f"possible {label} in release file: {relative}")
        files.append(path)
    require(files, "empty release")
    require(sum(path.stat().st_size for path in files) <= 100 * 1024 * 1024,
            "release payload exceeds 100 MiB")
    return files


def deterministic_zip(source_root, destination):
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as archive:
        for path in sorted(source_root.rglob("*")):
            if not path.is_file():
                continue
            relative = Path(RELEASE_NAME) / path.relative_to(source_root)
            info = zipfile.ZipInfo(str(relative).replace(os.sep, "/"),
                                   date_time=(2026, 9, 13, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o644 & 0xFFFF) << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED,
                             compresslevel=9)


def deterministic_tar_gz(source_root, destination):
    with destination.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0,
                           compresslevel=9) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for path in sorted(source_root.rglob("*")):
                    if not path.is_file():
                        continue
                    name = str(Path(RELEASE_NAME) / path.relative_to(source_root))
                    info = archive.gettarinfo(str(path), arcname=name)
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    info.mode = 0o644
                    with path.open("rb") as stream:
                        archive.addfile(info, stream)


def main(root):
    root = root.resolve()
    require(root.name == "vlsi_fault_detection_v2" and Path(__file__).resolve().parent == root,
            "copy this script and README template to ~/vlsi_fault_detection_v2")
    source_sha = sha(Path(__file__))

    print("STAGE 11E-1I — REPRODUCIBLE GITHUB RELEASE BUNDLE GENERATION")
    print("FROZEN INPUT VERIFICATION")
    for relative, expected in FROZEN_INPUTS.items():
        path = root / relative
        require(path.is_file() and not path.is_symlink(), "missing input: " + relative)
        require(sha(path) == expected, "SHA mismatch: " + relative)
        print("  " + Path(relative).name.ljust(76) + ": OK", flush=True)
    require((root / README_TEMPLATE).is_file() and sha(root / README_TEMPLATE) == README_SHA,
            "README template missing or SHA mismatch")
    print("  " + README_TEMPLATE.name.ljust(76) + ": OK", flush=True)

    final_audit = json.loads((root / "results/hmac_project_final_11e1h/"
                              "hmac_final_project_release_readiness_freeze_11e1h.json").read_text())
    values = json.dumps(final_audit, sort_keys=True)
    require('"status": "PASS"' in values and
            "COMPLETED_AND_FROZEN_RESEARCH_PROTOTYPE" in values and
            '"project_final_freeze": "DECLARED"' in values and
            '"original_validation_acceptance": "NOT_MET"' in values and
            '"public_github_push": "NOT_YET_AUTHORIZED"' in values,
            "Stage 11E-1H semantic authorization")

    output_parent = root / "release"
    final_release = output_parent / RELEASE_NAME
    zip_path = output_parent / (RELEASE_NAME + ".zip")
    tar_path = output_parent / (RELEASE_NAME + ".tar.gz")
    audit_path = root / RESULT_ROOT / "hmac_github_release_bundle_sanitization_freeze_11e1i.json"
    manifest_path = root / RESULT_ROOT / "hmac_github_release_bundle_manifest_11e1i.json"
    for path in (final_release, zip_path, tar_path, audit_path, manifest_path):
        require(not path.exists() and not path.is_symlink(),
                "output already exists; preserve it and inspect before rerunning: " + str(path))
    output_parent.mkdir(parents=True, exist_ok=True)
    (root / RESULT_ROOT).mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="hmac_release_11e1i_",
                                     dir=root / RESULT_ROOT) as temporary:
        release_root = Path(temporary) / RELEASE_NAME
        release_root.mkdir()
        source_records = {}
        copy_public(root, str(README_TEMPLATE), release_root, "README.md", source_records)

        # Frozen primary model and the records needed to interpret it.
        fixed_map = {
            "results/hmac_fault_campaign_11d2/hybrid_training_11d2b/hmac_selected_hybrid_model_11d2b.joblib":
                "models/hmac_hybrid_v1_original.joblib",
            "config/diagnostic_model/hmac_final_diagnostic_model_lock_11d2d.json":
                "models/hmac_final_diagnostic_model_lock_11d2d.json",
            "config/diagnostic_model/hmac_hybrid_architecture_11d2a.json":
                "config/hmac_hybrid_architecture_11d2a.json",
            "config/diagnostic_model/hmac_hybrid_training_contract_11d2a.json":
                "config/hmac_hybrid_training_contract_11d2a.json",
            "results/hmac_fault_campaign_11d2/hmac_hybrid_environment_11d2a.json":
                "environment/hmac_hybrid_environment_11d2a.json",
            "results/hmac_fault_campaign_11c5/feature_matrix_11c5e/hmac_leakage_safe_feature_matrix_schema_11c5e.json":
                "schemas/hmac_leakage_safe_feature_matrix_schema_11c5e.json",
            "results/hmac_fault_campaign_11d1/graph_dataset_11d1a/hmac_golden_netlist_graph_schema_11d1a.json":
                "schemas/hmac_golden_netlist_graph_schema_11d1a.json",
            "results/hmac_fault_campaign_11d3/canonical_dataset_11d3f/hmac_canonical_validation_dataset_schema_11d3f.json":
                "schemas/hmac_canonical_validation_dataset_schema_11d3f.json",
            "results/hmac_fault_campaign_11d3/feature_matrix_11d3g/hmac_validation_feature_schema_11d3g.json":
                "schemas/hmac_validation_feature_schema_11d3g.json",
            "results/hmac_project_final_11e1h/hmac_final_research_prototype_model_card_11e1h.md":
                "reports/MODEL_CARD.md",
            "results/hmac_fault_campaign_11d3/hmac_validation_failure_review_11d3j.md":
                "reports/VALIDATION_FAILURE_REVIEW.md",
            "results/hmac_behavior_diagnosis_pilot/hmac_behavior_diagnosis_pilot_report_11e1g.md":
                "reports/BEHAVIOR_DIAGNOSIS_PILOT.md",
            "config/release/hmac_final_project_disposition_11e1h.json":
                "config/hmac_final_project_disposition_11e1h.json",
            "config/release/hmac_reproducible_release_policy_11e1h.json":
                "config/hmac_reproducible_release_policy_11e1h.json",
        }
        for source, destination in fixed_map.items():
            copy_public(root, source, release_root, destination, source_records)

        # Project-authored wrappers/testbenches only; never copy third_party/.
        rtl_files = sorted((root / "rtl/hmac").glob("*.sv"))
        tb_files = sorted((root / "tb/hmac").glob("*.sv"))
        require(rtl_files and tb_files, "HMAC wrapper RTL or testbench sources are missing")
        for path in rtl_files + tb_files:
            relative = path.relative_to(root)
            copy_public(root, str(relative), release_root, str(relative), source_records)

        # Include the frozen workflow sources, while excluding caches and data.
        stage_sources = []
        stage_pattern = re.compile(r"stage_11(?:c5|d[123]|e1)[A-Za-z0-9_]*\.py$")
        for path in sorted(root.glob("stage_11*.py")):
            if path.is_file() and stage_pattern.fullmatch(path.name):
                stage_sources.append(path)
        behavior_source = root / "hmac_behavior_observability_audit.py"
        if behavior_source.is_file():
            stage_sources.append(behavior_source)
        require(len(stage_sources) >= 20 and
                any(path.name == "stage_11e1g_behavior_pilot_disposition.py"
                    for path in stage_sources),
                "insufficient final workflow sources for a reproducible release")
        stage_sources.append(Path(__file__).resolve())
        for path in sorted(set(stage_sources)):
            copy_public(root, str(path.relative_to(root)), release_root,
                        "scripts/stages/" + path.name, source_records)

        hmac_tools = sorted((root / "scripts/hmac").glob("*.py"))
        require(len(hmac_tools) >= 3, "reusable HMAC campaign tools are missing")
        for path in hmac_tools:
            copy_public(root, str(path.relative_to(root)), release_root,
                        "scripts/hmac/" + path.name, source_records)

        versions = []
        for package in ("numpy", "scipy", "scikit-learn", "joblib"):
            try:
                versions.append(f"{package}=={importlib.metadata.version(package)}")
            except importlib.metadata.PackageNotFoundError as error:
                raise RuntimeError("required Python package is missing: " + package) from error
        write(release_root / "requirements-lock.txt", ("\n".join(versions) + "\n").encode())
        write(release_root / "environment/PYTHON_VERSION.txt",
              (sys.version.replace("\n", " ") + "\n").encode())
        write(release_root / "LICENSE", APACHE_2.encode())
        write(release_root / "THIRD_PARTY_NOTICE.md", NOTICE.encode())
        write(release_root / "CITATION.cff", CITATION.encode())
        write(release_root / "SECURITY.md", SECURITY.encode())
        write(release_root / ".gitignore", GITIGNORE.encode())

        release_metadata = {
            "release": RELEASE_NAME,
            "stage": "11E-1I",
            "version": VERSION,
            "status": "PASS",
            "project_status": "COMPLETED_AND_FROZEN_RESEARCH_PROTOTYPE",
            "license": "Apache-2.0",
            "maintainer": "Pavan Nithin",
            "frozen_model": "models/hmac_hybrid_v1_original.joblib",
            "frozen_model_sha256": FROZEN_INPUTS[
                "results/hmac_fault_campaign_11d2/hybrid_training_11d2b/hmac_selected_hybrid_model_11d2b.joblib"],
            "quantized_model_included": False,
            "quantization_reason":
                "Quantization changes the frozen artifact and requires a separate derivative plus metric replay.",
            "validation_acceptance": "NOT_MET_MCC_RETENTION",
            "holdout": "NOT_EXECUTED",
            "independent_generalization": "NOT_ESTABLISHED",
            "production_readiness": "NOT_ESTABLISHED",
            "raw_datasets_included": False,
            "private_blinded_truth_included": False,
            "upstream_opentitan_source_included": False,
            "source_to_release_records": source_records,
        }
        write(release_root / "RELEASE_METADATA.json", encode(release_metadata))

        files = scan_release(release_root)
        lines = [f"{sha(path)}  {path.relative_to(release_root).as_posix()}"
                 for path in files if path.name != "SHA256SUMS"]
        write(release_root / "SHA256SUMS", ("\n".join(lines) + "\n").encode())
        files = scan_release(release_root)
        require(len(files) >= 35, "release inventory is unexpectedly small")

        # Ensure Python sources are syntactically valid without importing them.
        for path in files:
            if path.suffix == ".py":
                compile(path.read_bytes(), str(path), "exec")

        shutil.move(str(release_root), final_release)

    deterministic_zip(final_release, zip_path)
    deterministic_tar_gz(final_release, tar_path)
    with tempfile.TemporaryDirectory(prefix="hmac_archive_replay_", dir=root / RESULT_ROOT) as replay:
        replay_zip = Path(replay) / "replay.zip"
        replay_tar = Path(replay) / "replay.tar.gz"
        deterministic_zip(final_release, replay_zip)
        deterministic_tar_gz(final_release, replay_tar)
        require(sha(replay_zip) == sha(zip_path), "ZIP deterministic replay mismatch")
        require(sha(replay_tar) == sha(tar_path), "tar.gz deterministic replay mismatch")

    manifest = {
        "stage": "11E-1I",
        "version": VERSION,
        "status": "PASS",
        "release_name": RELEASE_NAME,
        "release_directory": str(final_release.relative_to(root)),
        "release_file_count": len([p for p in final_release.rglob("*") if p.is_file()]),
        "release_total_bytes": sum(p.stat().st_size for p in final_release.rglob("*") if p.is_file()),
        "zip": {"path": str(zip_path.relative_to(root)), "sha256": sha(zip_path),
                "bytes": zip_path.stat().st_size},
        "tar_gz": {"path": str(tar_path.relative_to(root)), "sha256": sha(tar_path),
                   "bytes": tar_path.stat().st_size},
        "readme_sha256": sha(final_release / "README.md"),
        "model_sha256": sha(final_release / "models/hmac_hybrid_v1_original.joblib"),
        "deterministic_archive_replay": "PASS",
        "secret_scan": "PASS",
        "private_artifact_scan": "PASS",
        "machine_path_scan": "PASS",
        "syntax_scan": "PASS",
        "github_contacted": False,
    }
    write(manifest_path, encode(manifest))

    audit = dict(manifest)
    audit.update({
        "audit_status": "FROZEN",
        "project_final_freeze": "PRESERVED",
        "frozen_model_modified": False,
        "model_training_calls": 0,
        "model_inference_calls": 0,
        "model_quantization_calls": 0,
        "holdout_payloads_read": 0,
        "raw_private_case_files_read": 0,
        "license": "Apache-2.0; OWNER CONFIRMATION REQUIRED BEFORE PUBLIC PUSH",
        "repository_url_placeholder_remaining": True,
        "public_github_push": "BLOCKED_PENDING_OWNER_LICENSE_CONFIRMATION_AND_CLEAN_ENVIRONMENT_SMOKE",
        "dashboard_frontend": "DEFERRED_UNTIL_VERIFIED_PUBLIC_RELEASE",
        "source_sha256": source_sha,
        "readme_template_sha256": README_SHA,
        "manifest": {"path": str(manifest_path.relative_to(root)),
                     "sha256": sha(manifest_path)},
        "next_gate":
            "STAGE 11E-1J — CLEAN-ENVIRONMENT RELEASE SMOKE, LICENSE CONFIRMATION, AND OPTIONAL QUANTIZED DERIVATIVE CONTRACT",
    })
    write(audit_path, encode(audit))

    print("\nSTAGE 11E-1I — REPRODUCIBLE GITHUB RELEASE BUNDLE GENERATION AND SANITIZATION FREEZE")
    print("Status                         : PASS")
    print("Release bundle status          : GENERATED AND FROZEN")
    print("Project freeze                 : PRESERVED")
    print("Original hybrid model          : INCLUDED UNCHANGED")
    print("Quantized derivative           : NOT CREATED")
    print("Secret/private/path scans      : PASS / PASS / PASS")
    print("Python syntax scan             : PASS")
    print("Deterministic archive replay   : PASS")
    print("HOLDOUT payloads read          : 0")
    print("GitHub contacted               : NO")
    print("Public push                    : NOT YET AUTHORIZED")
    print("Reason                         : CONFIRM LICENSE + CLEAN-ENVIRONMENT SMOKE")
    print("Release directory              :", final_release)
    print("ZIP                            :", zip_path)
    print("ZIP SHA                        :", sha(zip_path))
    print("tar.gz                         :", tar_path)
    print("tar.gz SHA                     :", sha(tar_path))
    print("Manifest                       :", manifest_path)
    print("Manifest SHA                   :", sha(manifest_path))
    print("Audit                          :", audit_path)
    print("Audit SHA                      :", sha(audit_path))
    print("Next gate                      :", audit["next_gate"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    try:
        main(args.project_root)
    except KeyboardInterrupt:
        print("STOP: interrupted; frozen project inputs were not modified.")
        raise SystemExit(130)
    except Exception as error:
        print("STOP:", error)
        print("Do not push a partial or unverified release bundle to GitHub.")
        raise SystemExit(1)
