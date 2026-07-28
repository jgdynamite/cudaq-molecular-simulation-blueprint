#!/usr/bin/env python3
"""Assemble the dataset the public UI demo is exported from.

The demo shows two benchmarks that were run three months apart:

* ``akamai-blackwell-multiseed`` (2026-05-04) is the only source of H2 data.
* ``lih-active-space-followup-v02`` (2026-07-27) is the LiH source.

May's LiH runs are deliberately excluded. They used the legacy full-molecule
ansatz, and July's ``legacy_full`` arm reproduces exactly that configuration
on current code. Keeping both would place two same-named arms recorded on
different hosts, in different regions, three months apart into a single
bucket, which is the kind of silent merge the comparison report is written
to avoid.

Raw manifests and traces are not committed, so this only works on a machine
that still has the bench output on disk. Both source directories keep their
committed ``SUMMARY.csv`` / ``comparison.json`` either way.

Usage::

    python scripts/build_demo_dataset.py
    python -m app.ui.static_export --output-dir _site
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
MAY = RESULTS / "akamai-blackwell-multiseed"
JULY = RESULTS / "lih-active-space-followup-v02"
OUT = RESULTS / "demo-combined"


def _molecule_of(run_dir: Path) -> str:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    molecule = manifest.get("molecule")
    if isinstance(molecule, dict):
        molecule = molecule.get("name")
    return str(molecule or "?")


def _run_dirs(base: Path) -> list[Path]:
    if not base.exists():
        sys.exit(
            f"missing source directory: {base}\n"
            "Raw per-run manifests are not committed; re-run the bench or "
            "restore the tarball before building the demo dataset."
        )
    return sorted(p for p in base.iterdir() if p.is_dir() and (p / "manifest.json").exists())


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    h2_copied = 0
    lih_skipped = 0
    for run in _run_dirs(MAY):
        if _molecule_of(run) == "h2":
            shutil.copytree(run, OUT / run.name)
            h2_copied += 1
        else:
            lih_skipped += 1

    lih_copied = 0
    for run in _run_dirs(JULY):
        shutil.copytree(run, OUT / run.name)
        lih_copied += 1

    print(f"  H2 from {MAY.name}: {h2_copied}")
    print(f"  LiH skipped (superseded by July legacy_full arm): {lih_skipped}")
    print(f"  LiH from {JULY.name}: {lih_copied}")
    print(f"  total runs in {OUT.relative_to(ROOT)}: {len(_run_dirs(OUT))}")


if __name__ == "__main__":
    main()
