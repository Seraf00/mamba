#!/usr/bin/env python3
"""Prove a group downloaded from Drive is complete before deleting it there.

The notebook's offload_check(group) writes MANIFEST.json into the group folder:
the size and SHA-256 of every best_model.pth, computed on the Colab machine
from the files training produced. This checks your downloaded copy against it.

A browser or Drive download can be truncated or silently skip a file, and after
offload() the Drive copy is gone -- so "the folder looks right" is not enough.
Only ALL VERIFIED means it is safe to run offload(group, ...).

    python scripts/verify_offload.py results_revision/r1_canonical
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    d = Path(sys.argv[1])
    mf = d / 'MANIFEST.json'
    if not mf.exists():
        print(f'No MANIFEST.json in {d}. Run offload_check("{d.name}") in the '
              f'notebook, let it sync, then download the folder again.')
        return 1
    manifest = json.loads(mf.read_text())

    bad = 0
    total = 0
    for rel, want in sorted(manifest.items()):
        f = d / rel
        if not f.exists():
            print(f'  MISSING   {rel}')
            bad += 1
            continue
        size = f.stat().st_size
        if size != want['bytes']:
            print(f'  TRUNCATED {rel}: {size:,} of {want["bytes"]:,} bytes')
            bad += 1
            continue
        if sha256(f) != want['sha256']:
            print(f'  CORRUPT   {rel}: checksum differs')
            bad += 1
            continue
        total += size
        print(f'  ok        {rel}  ({size / 1e6:,.0f} MB)')

    # The small files the tables are built from must be here too.
    for need in ('all_results.json', 'evaluation/evaluation_results.json',
                 'baseline_ef_native.json'):
        if not (d / need).exists():
            print(f'  MISSING   {need}')
            bad += 1

    print()
    if bad:
        print(f'{bad} PROBLEM(S) -- do NOT offload. Download the folder again.')
        return 1
    print(f'ALL VERIFIED: {len(manifest)} checkpoints, {total / 1e9:.2f} GB. '
          f'Safe to run offload("{d.name}", i_have_verified_the_download=True).')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
