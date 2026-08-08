#!/usr/bin/env python3
"""Create a reproducible source fingerprint for a dirty dsmvc checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


SOURCE_SCOPES = (
    "src", "include", "tests", "benchmarks", "cmake", "CMakeLists.txt",
)


def git(repo: Path, *arguments: str, text: bool = False):
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=text,
    ).stdout


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(repo: Path) -> dict:
    head = git(repo, "rev-parse", "HEAD", text=True).strip()
    tracked_diff = git(repo, "diff", "--binary", "HEAD")
    raw_untracked = git(
        repo, "ls-files", "--others", "--exclude-standard", "-z", "--",
        *SOURCE_SCOPES,
    )
    untracked_paths = sorted(
        item.decode("utf-8", errors="surrogateescape")
        for item in raw_untracked.split(b"\0") if item
    )
    untracked_sources = []
    for relative in untracked_paths:
        path = repo / relative
        if not path.is_file():
            raise RuntimeError(f"untracked source is not a regular file: {relative}")
        untracked_sources.append({
            "path": relative,
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    identity = {
        "head": head,
        "tracked_diff_sha256": sha256_bytes(tracked_diff),
        "untracked_sources": untracked_sources,
    }
    canonical = json.dumps(
        identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("ascii")
    return {
        "schema": "dsmvc-source-fingerprint-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repo": str(repo),
        "method": {
            "tracked": "git diff --binary HEAD",
            "untracked_scopes": list(SOURCE_SCOPES),
            "canonical_encoding": "sorted compact JSON over head/diff/untracked",
        },
        **identity,
        "source_fingerprint": sha256_bytes(canonical),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    options = parser.parse_args()
    repo = options.repo.expanduser().resolve()
    output = options.json_out.expanduser().resolve()
    if not (repo / ".git").exists():
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--git-dir"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise ValueError(f"not a Git checkout: {repo}")
    document = fingerprint(repo)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(document, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(document["source_fingerprint"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
