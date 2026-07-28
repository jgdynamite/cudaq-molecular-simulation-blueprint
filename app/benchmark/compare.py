"""CPU vs GPU comparison report.

Reads every manifest under the configured ``results/`` directory, groups by
``(experiment, backend, ansatz_mode)``, computes mean and standard error for
the metrics the blog post quotes, and emits a structured report.

The ansatz mode is part of the grouping key on purpose. A ``legacy_full`` and
a ``matched`` LiH run share a molecule, a backend and a Hamiltonian, but they
execute different circuits (12 qubits / 92 parameters versus 10 / 24), so
averaging them together produces a mean that describes neither and a CPU/GPU
ratio that compares different amounts of work. Speedups are therefore only
computed between arms that share a mode.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Any

from app.benchmark.metrics import summarize_run
from app.storage.filesystem import list_runs, load_trace


def _stderr(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values) / math.sqrt(len(values))


def _ansatz_label(manifest: Any) -> str | None:
    """The run's ansatz mode, or ``None`` if it never declared one.

    ``None`` covers H2 (no active space, so the mode cannot change anything)
    and every manifest written before v0.2. Those are not assumed to be
    equivalent to any declared mode; they simply group among themselves.
    """
    mode = (manifest.notes or {}).get("ansatz_mode")
    return str(mode) if mode else None


def _arm_key(backend: str, mode: str | None) -> str:
    return backend if mode is None else f"{backend} · {mode}"


def compare_cpu_vs_gpu() -> dict[str, Any]:
    """Build a structured CPU-vs-GPU comparison from on-disk runs.

    Returns a dict with one entry per molecule, containing per-backend
    aggregates (n, mean, stderr) for wall time, time per evaluation,
    iterations, error vs reference, and the GPU/CPU speedup factor.
    """
    runs = list_runs()
    grouped: dict[tuple[str, str, str | None], list[Any]] = defaultdict(list)
    for manifest in runs:
        if manifest.result is None:
            continue
        trace = load_trace(manifest.run_id)
        m = summarize_run(manifest, trace)
        grouped[(m.molecule, m.backend, _ansatz_label(manifest))].append(m)

    report: dict[str, Any] = {"by_molecule": {}}
    for molecule in sorted({mol for mol, _, _ in grouped}):
        # Two circuits built for the same Hamiltonian on the same backend are
        # different experiments, so they only share an arm key when neither
        # declares a mode (H2, which has no active space, and pre-v0.2 runs).
        modes_present = any(mode is not None for mol, _, mode in grouped if mol == molecule)
        per_backend: dict[str, dict[str, Any]] = {}
        arm_modes: dict[str, str | None] = {}
        for (mol, backend, mode), metrics_list in grouped.items():
            if mol != molecule:
                continue
            arm = _arm_key(backend, mode) if modes_present else backend
            arm_modes[arm] = mode
            wall_times = [m.wall_time_seconds for m in metrics_list]
            per_eval = [m.time_per_evaluation_ms for m in metrics_list]
            iters = [m.iterations for m in metrics_list]
            errors = [
                abs(m.error_vs_reference_hartree)
                for m in metrics_list
                if m.error_vs_reference_hartree is not None
            ]
            per_backend[arm] = {
                "n": len(metrics_list),
                "backend": backend,
                "ansatz_mode": mode,
                "wall_time_seconds": {
                    "mean": statistics.fmean(wall_times),
                    "stderr": _stderr(wall_times),
                    "values": wall_times,
                },
                "time_per_evaluation_ms": {
                    "mean": statistics.fmean(per_eval),
                    "stderr": _stderr(per_eval),
                },
                "iterations": {
                    "mean": statistics.fmean(iters),
                    "stderr": _stderr([float(v) for v in iters]),
                },
                "error_hartree": {
                    "mean": statistics.fmean(errors) if errors else None,
                    "stderr": _stderr(errors) if errors else 0.0,
                },
                "qubit_count": metrics_list[0].qubit_count,
                "parameter_count": metrics_list[0].parameter_count,
                "target_strings": sorted({m.target_string for m in metrics_list}),
            }

        # A CPU/GPU ratio is only meaningful between two arms running the same
        # circuit, so pair arms within an ansatz mode and never across modes.
        speedups: dict[str, float] = {}
        for mode in {arm_modes[a] for a in per_backend}:
            arms_in_mode = {a: s for a, s in per_backend.items() if arm_modes[a] == mode}
            cpu_arm = next((a for a, s in arms_in_mode.items() if s["backend"] == "cpu"), None)
            if cpu_arm is None:
                continue
            cpu_mean = arms_in_mode[cpu_arm]["wall_time_seconds"]["mean"]
            for arm, stats in arms_in_mode.items():
                if arm == cpu_arm:
                    continue
                gpu_mean = stats["wall_time_seconds"]["mean"]
                if gpu_mean > 0:
                    label = f"cpu_over_{stats['backend']}_wall_time"
                    if mode is not None:
                        label = f"{mode}/{label}"
                    speedups[label] = cpu_mean / gpu_mean

        report["by_molecule"][molecule] = {
            "backends": per_backend,
            "speedups": speedups,
        }

    report["totals"] = {
        "molecules": list(report["by_molecule"].keys()),
        "total_runs": sum(len(v) for v in grouped.values()),
    }
    return report
