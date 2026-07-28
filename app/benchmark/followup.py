"""Reporting for the LiH active-space ansatz follow-up experiment.

This module is deliberately separate from :mod:`app.benchmark.compare`. The
generic comparison groups runs by ``(molecule, backend)``, which is exactly
the wrong thing to do here: the whole point of the follow-up is that two
sets of LiH runs share a molecule and a backend but use *different* ansatz
dimensions, and averaging them together would hide the effect under test.

Grouping here is always keyed on ``(experiment_variant, ansatz_mode,
backend)``, so a ``legacy_full`` run and a ``matched`` run can never land in
the same bucket.

Nothing in this module interprets the numbers. It computes descriptive
statistics and paired differences; whether those differences are meaningful
is a question for whoever reads the output.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.benchmark.metrics import summarize_run
from app.core.metadata import project_version
from app.storage.filesystem import list_runs, load_trace
from app.storage.manifests import (
    EXPERIMENT_VARIANT_LEGACY_FULL,
    EXPERIMENT_VARIANT_MATCHED,
    IterationTrace,
    RunManifest,
)

SUMMARY_CSV_FILENAME = "SUMMARY.csv"
COMPARISON_JSON_FILENAME = "comparison.json"

#: Final energies for the same seed on CPU vs GPU FP64 are treated as
#: numerically equivalent within this tolerance. Violations are reported
#: rather than discarded - a disagreement above this is a finding, not noise
#: to be filtered out.
ENERGY_EQUIVALENCE_TOLERANCE_HARTREE = 1e-8

HARTREE_TO_MILLIHARTREE = 1000.0

SUMMARY_COLUMNS = [
    "run_id",
    "experiment_variant",
    "ansatz_mode",
    "backend",
    "seed",
    "qubits",
    "parameters",
    "iterations",
    "function_evaluations",
    "wall_seconds",
    "time_per_evaluation_ms",
    "energy_hartree",
    "reference_hartree",
    "absolute_error_mhartree",
    "chemical_accuracy_reached",
    "gpu_model",
    "driver_version",
    "cudaq_version",
    "project_version",
    "git_commit_sha",
]


@dataclass(slots=True)
class FollowupRow:
    """One run, flattened into the fields SUMMARY.csv exposes."""

    run_id: str
    experiment_variant: str
    ansatz_mode: str
    backend: str
    seed: int
    qubits: int
    parameters: int
    iterations: int
    function_evaluations: int
    wall_seconds: float
    time_per_evaluation_ms: float
    energy_hartree: float
    reference_hartree: float | None
    absolute_error_mhartree: float | None
    chemical_accuracy_reached: bool | None
    gpu_model: str | None
    driver_version: str | None
    cudaq_version: str | None
    project_version: str
    git_commit_sha: str | None

    @property
    def group_key(self) -> str:
        return group_name(self.experiment_variant, self.ansatz_mode, self.backend)

    def as_csv_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "experiment_variant": self.experiment_variant,
            "ansatz_mode": self.ansatz_mode,
            "backend": self.backend,
            "seed": self.seed,
            "qubits": self.qubits,
            "parameters": self.parameters,
            "iterations": self.iterations,
            "function_evaluations": self.function_evaluations,
            "wall_seconds": f"{self.wall_seconds:.3f}",
            "time_per_evaluation_ms": f"{self.time_per_evaluation_ms:.4f}",
            # 12 dp keeps the ~1e-13 Ha cross-backend agreement visible; 9 dp
            # rounded it away. The error column is scientific for the same
            # reason: a 2.99e-5 mHa residual formatted to 4 dp reads as
            # exactly zero.
            "energy_hartree": f"{self.energy_hartree:.12f}",
            "reference_hartree": (
                "" if self.reference_hartree is None else f"{self.reference_hartree:.12f}"
            ),
            "absolute_error_mhartree": (
                ""
                if self.absolute_error_mhartree is None
                else f"{self.absolute_error_mhartree:.6e}"
            ),
            "chemical_accuracy_reached": (
                ""
                if self.chemical_accuracy_reached is None
                else str(self.chemical_accuracy_reached)
            ),
            "gpu_model": self.gpu_model or "",
            "driver_version": self.driver_version or "",
            "cudaq_version": self.cudaq_version or "",
            "project_version": self.project_version,
            "git_commit_sha": self.git_commit_sha or "",
        }


def group_name(experiment_variant: str, ansatz_mode: str, backend: str) -> str:
    """Stable, human-readable key for one experimental arm."""
    return f"{experiment_variant}::{ansatz_mode}::{backend}"


def _note(manifest: RunManifest, key: str, default: Any = None) -> Any:
    notes = manifest.notes or {}
    value = notes.get(key, default)
    return default if value is None else value


def _experiment_variant(manifest: RunManifest) -> str:
    """Variant label for a manifest, tolerating pre-v0.2 files.

    Manifests written before the ansatz mode existed carry no
    ``experiment_variant``. Rather than guessing which arm they belong to,
    they get their own bucket so they can never be silently averaged in with
    a labelled arm.
    """
    recorded = _note(manifest, "experiment_variant")
    if recorded:
        return str(recorded)
    return f"unlabelled_pre_v02_{manifest.molecule.name.value}"


def _first_gpu(manifest: RunManifest) -> dict[str, Any]:
    gpus = (manifest.system_info or {}).get("gpus") or []
    if not gpus:
        return {}
    first = gpus[0]
    return first if isinstance(first, dict) else {}


def row_from_run(manifest: RunManifest, trace: IterationTrace) -> FollowupRow:
    """Flatten one ``(manifest, trace)`` pair into a :class:`FollowupRow`."""
    metrics = summarize_run(manifest, trace)
    system_info = manifest.system_info or {}
    gpu = _first_gpu(manifest)

    error_ha = metrics.error_vs_reference_hartree
    abs_error_mha = None if error_ha is None else abs(error_ha) * HARTREE_TO_MILLIHARTREE

    return FollowupRow(
        run_id=manifest.run_id,
        experiment_variant=_experiment_variant(manifest),
        ansatz_mode=str(_note(manifest, "ansatz_mode", "unspecified")),
        backend=manifest.backend.value,
        seed=manifest.seed,
        qubits=manifest.qubit_count,
        parameters=manifest.parameter_count,
        iterations=metrics.iterations,
        function_evaluations=metrics.function_evaluations,
        wall_seconds=metrics.wall_time_seconds,
        time_per_evaluation_ms=metrics.time_per_evaluation_ms,
        energy_hartree=metrics.final_energy,
        reference_hartree=metrics.reference_energy,
        absolute_error_mhartree=abs_error_mha,
        chemical_accuracy_reached=metrics.chemical_accuracy_reached,
        gpu_model=gpu.get("name"),
        driver_version=gpu.get("driver_version"),
        cudaq_version=system_info.get("cudaq_version"),
        project_version=manifest.project_version or project_version(),
        git_commit_sha=system_info.get("git_sha"),
    )


def collect_rows(manifests: list[RunManifest] | None = None) -> list[FollowupRow]:
    """Build rows for every completed run in the configured results dir."""
    source = list_runs() if manifests is None else manifests
    rows: list[FollowupRow] = []
    for manifest in source:
        if manifest.result is None:
            continue
        rows.append(row_from_run(manifest, load_trace(manifest.run_id)))
    rows.sort(key=lambda r: (r.experiment_variant, r.ansatz_mode, r.backend, r.seed))
    return rows


def write_summary_csv(rows: list[FollowupRow], path: Path) -> Path:
    """Write one CSV row per run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_csv_dict())
    return path


def _stderr(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values) / math.sqrt(len(values))


def _stat_block(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"mean": None, "stderr": None, "values": []}
    return {
        "mean": statistics.fmean(values),
        "stderr": _stderr(values),
        "values": values,
    }


def _group_summary(rows: list[FollowupRow]) -> dict[str, Any]:
    wall = [r.wall_seconds for r in rows]
    per_eval = [r.time_per_evaluation_ms for r in rows]
    iters = [float(r.iterations) for r in rows]
    energies = [r.energy_hartree for r in rows]
    errors = [r.absolute_error_mhartree for r in rows if r.absolute_error_mhartree is not None]
    reached = [bool(r.chemical_accuracy_reached) for r in rows]

    n = len(rows)
    hits = sum(1 for r in reached if r)

    return {
        "n": n,
        "experiment_variant": rows[0].experiment_variant,
        "ansatz_mode": rows[0].ansatz_mode,
        "backend": rows[0].backend,
        "seeds": sorted(r.seed for r in rows),
        "qubit_count": rows[0].qubits,
        "parameter_count": rows[0].parameters,
        "wall_time_seconds": _stat_block(wall),
        "time_per_evaluation_ms": _stat_block(per_eval),
        "iterations": {"mean": statistics.fmean(iters) if iters else None},
        "absolute_error_mhartree": {
            "mean": statistics.fmean(errors) if errors else None,
            "median": statistics.median(errors) if errors else None,
            "min": min(errors) if errors else None,
            "max": max(errors) if errors else None,
            "values": errors,
        },
        "final_energy_hartree": {
            "stdev_across_seeds": statistics.stdev(energies) if len(energies) > 1 else 0.0,
            "values": energies,
        },
        "chemical_accuracy": {
            "count": hits,
            "rate": (hits / n) if n else None,
        },
    }


def _by_seed(rows: list[FollowupRow]) -> dict[int, FollowupRow]:
    return {r.seed: r for r in rows}


def _paired_differences(
    left: list[FollowupRow],
    right: list[FollowupRow],
) -> list[dict[str, Any]]:
    """Per-seed differences for seeds present in both arms (left minus right)."""
    lhs, rhs = _by_seed(left), _by_seed(right)
    shared = sorted(set(lhs) & set(rhs))
    pairs: list[dict[str, Any]] = []
    for seed in shared:
        a, b = lhs[seed], rhs[seed]
        entry: dict[str, Any] = {
            "seed": seed,
            "left_run_id": a.run_id,
            "right_run_id": b.run_id,
            "wall_seconds_left": a.wall_seconds,
            "wall_seconds_right": b.wall_seconds,
            "wall_seconds_difference": a.wall_seconds - b.wall_seconds,
            "wall_time_ratio_right_over_left": (
                b.wall_seconds / a.wall_seconds if a.wall_seconds > 0 else None
            ),
            "energy_hartree_left": a.energy_hartree,
            "energy_hartree_right": b.energy_hartree,
            "energy_difference_hartree": a.energy_hartree - b.energy_hartree,
        }
        if a.absolute_error_mhartree is not None and b.absolute_error_mhartree is not None:
            entry["absolute_error_difference_mhartree"] = (
                a.absolute_error_mhartree - b.absolute_error_mhartree
            )
        pairs.append(entry)
    return pairs


def _equivalence_check(
    cpu_rows: list[FollowupRow],
    gpu_rows: list[FollowupRow],
) -> dict[str, Any]:
    """Compare matched CPU vs matched GPU energies seed-by-seed."""
    pairs = _paired_differences(cpu_rows, gpu_rows)
    violations = [
        p
        for p in pairs
        if abs(p["energy_difference_hartree"]) > ENERGY_EQUIVALENCE_TOLERANCE_HARTREE
    ]
    return {
        "tolerance_hartree": ENERGY_EQUIVALENCE_TOLERANCE_HARTREE,
        "n_compared": len(pairs),
        "n_within_tolerance": len(pairs) - len(violations),
        "n_violations": len(violations),
        "all_within_tolerance": not violations,
        "max_absolute_energy_difference_hartree": (
            max((abs(p["energy_difference_hartree"]) for p in pairs), default=None)
        ),
        # Retained, not discarded: a disagreement here is a result.
        "violations": violations,
        "paired_energy_differences": pairs,
    }


def _mean_wall(group: dict[str, Any] | None) -> float | None:
    if not group:
        return None
    mean = group["wall_time_seconds"]["mean"]
    return float(mean) if mean else None


def build_comparison(rows: list[FollowupRow]) -> dict[str, Any]:
    """Build the variant-separated comparison report.

    Groups are keyed on ``(experiment_variant, ansatz_mode, backend)``, so
    the legacy and matched arms are never combined. Cross-arm relationships
    are expressed as explicit ratios and paired per-seed differences rather
    than by merging groups.
    """
    grouped: dict[str, list[FollowupRow]] = defaultdict(list)
    for row in rows:
        grouped[row.group_key].append(row)

    groups = {key: _group_summary(sorted(v, key=lambda r: r.seed)) for key, v in grouped.items()}

    legacy_gpu_key = group_name(EXPERIMENT_VARIANT_LEGACY_FULL, "legacy_full", "gpu_fp64")
    matched_gpu_key = group_name(EXPERIMENT_VARIANT_MATCHED, "matched", "gpu_fp64")
    matched_cpu_key = group_name(EXPERIMENT_VARIANT_MATCHED, "matched", "cpu")

    legacy_gpu_rows = grouped.get(legacy_gpu_key, [])
    matched_gpu_rows = grouped.get(matched_gpu_key, [])
    matched_cpu_rows = grouped.get(matched_cpu_key, [])

    matched_cpu_mean = _mean_wall(groups.get(matched_cpu_key))
    matched_gpu_mean = _mean_wall(groups.get(matched_gpu_key))
    legacy_gpu_mean = _mean_wall(groups.get(legacy_gpu_key))

    ratios: dict[str, Any] = {
        "matched_cpu_over_matched_gpu_wall_time": (
            matched_cpu_mean / matched_gpu_mean if matched_cpu_mean and matched_gpu_mean else None
        ),
        "matched_gpu_over_legacy_gpu_wall_time": (
            matched_gpu_mean / legacy_gpu_mean if matched_gpu_mean and legacy_gpu_mean else None
        ),
    }

    return {
        "schema_version": "1.0",
        "report": "lih_active_space_followup",
        "grouping_key": ["experiment_variant", "ansatz_mode", "backend"],
        "note": (
            "Groups are never merged across experiment_variant. Cross-arm "
            "relationships appear only as explicit ratios and paired "
            "per-seed differences."
        ),
        "group_keys": {
            "legacy_gpu": legacy_gpu_key,
            "matched_gpu": matched_gpu_key,
            "matched_cpu": matched_cpu_key,
        },
        "groups": dict(sorted(groups.items())),
        "ratios": ratios,
        "paired_matched_gpu_vs_legacy_gpu": _paired_differences(matched_gpu_rows, legacy_gpu_rows),
        "numerical_equivalence_matched_cpu_vs_matched_gpu": _equivalence_check(
            matched_cpu_rows, matched_gpu_rows
        ),
        "totals": {
            "total_runs": len(rows),
            "group_count": len(groups),
        },
    }


def write_comparison_json(report: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path


def generate_reports(
    output_dir: Path,
    manifests: list[RunManifest] | None = None,
) -> tuple[Path, Path]:
    """Write ``SUMMARY.csv`` and ``comparison.json`` into ``output_dir``.

    ``manifests`` defaults to every completed run in the configured results
    directory, which is what the suite runner passes after a sweep.
    """
    rows = collect_rows(manifests)
    csv_path = write_summary_csv(rows, output_dir / SUMMARY_CSV_FILENAME)
    json_path = write_comparison_json(build_comparison(rows), output_dir / COMPARISON_JSON_FILENAME)
    return csv_path, json_path


__all__ = [
    "COMPARISON_JSON_FILENAME",
    "ENERGY_EQUIVALENCE_TOLERANCE_HARTREE",
    "SUMMARY_COLUMNS",
    "SUMMARY_CSV_FILENAME",
    "FollowupRow",
    "build_comparison",
    "collect_rows",
    "generate_reports",
    "group_name",
    "row_from_run",
    "write_comparison_json",
    "write_summary_csv",
]
