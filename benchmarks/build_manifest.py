#!/usr/bin/env python3
"""Write a source-bound manifest for a dsmvc plugin build."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from paired_benchmark_support import sha256_file


def file_identity(path: Path) -> dict:
    output = subprocess.run(
        ["file", str(path)], capture_output=True, text=True, check=True,
    ).stdout.strip()
    return {
        "path": str(path),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
        "file": output,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", required=True,
                        choices=("candidate", "frozen-control"))
    parser.add_argument("--source-fingerprint-json", type=Path)
    parser.add_argument("--control-ref")
    parser.add_argument("--control-tree")
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--plugin", required=True, type=Path)
    parser.add_argument("--metallib", type=Path)
    parser.add_argument("--invocations", required=True, type=Path)
    parser.add_argument("--compile-commands", required=True, type=Path)
    parser.add_argument("--completed-at", required=True)
    parser.add_argument("--json-out", required=True, type=Path)
    options = parser.parse_args()

    for name in ("source_fingerprint_json", "build_dir", "plugin",
                 "metallib", "invocations", "compile_commands", "json_out"):
        value = getattr(options, name)
        if value is not None:
            setattr(options, name, value.expanduser().resolve())
    for path in (options.build_dir, options.plugin, options.invocations,
                 options.compile_commands):
        if not path.exists():
            raise FileNotFoundError(path)

    if options.kind == "candidate":
        if not options.source_fingerprint_json:
            raise ValueError("candidate manifest requires a source fingerprint")
        source_document = json.loads(
            options.source_fingerprint_json.read_text(encoding="utf-8"))
        source_fingerprint = source_document.get("source_fingerprint")
        source_identity = {
            "fingerprint_file": str(options.source_fingerprint_json),
            "head": source_document.get("head"),
            "tracked_diff_sha256": source_document.get("tracked_diff_sha256"),
            "untracked_sources": source_document.get("untracked_sources"),
        }
    else:
        if not options.control_ref or not options.control_tree:
            raise ValueError("frozen control requires commit and tree identities")
        canonical = json.dumps({
            "control_ref": options.control_ref,
            "control_tree": options.control_tree,
        }, sort_keys=True, separators=(",", ":")).encode("ascii")
        source_fingerprint = hashlib.sha256(canonical).hexdigest()
        source_identity = {
            "control_ref": options.control_ref,
            "control_tree": options.control_tree,
            "latest_candidate": False,
        }
    if not isinstance(source_fingerprint, str) or len(source_fingerprint) != 64:
        raise ValueError("source fingerprint is invalid")

    document = {
        "schema": "dsmvc-build-manifest-v1",
        "kind": options.kind,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": options.completed_at,
        "source_fingerprint": source_fingerprint,
        "source_identity": source_identity,
        "build_directory": str(options.build_dir),
        "plugin": file_identity(options.plugin),
        "metallib": file_identity(options.metallib)
        if options.metallib else None,
        "build_invocations": options.invocations.read_text(
            encoding="utf-8").splitlines(),
        "compile_commands_path": str(options.compile_commands),
        "compile_commands_sha256": sha256_file(options.compile_commands),
    }
    options.json_out.parent.mkdir(parents=True, exist_ok=True)
    options.json_out.write_text(
        json.dumps(document, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "manifest": str(options.json_out),
        "source_fingerprint": source_fingerprint,
        "plugin": document["plugin"],
    }, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
