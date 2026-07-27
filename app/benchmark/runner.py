"""Benchmark suite runner.

Sweeps a list of ``(experiment, backend, seed)`` configurations, runs them
in series (we want clean GPU/CPU isolation, not interleaved), and persists
each run as a regular manifest + trace. Downstream comparison tooling reads
those directly off disk.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from app.core.logging import get_logger
from app.quantum.h2_vqe import run_h2
from app.quantum.lih_vqe import run_lih
from app.storage.filesystem import save_manifest, save_trace
from app.storage.manifests import (
    AnsatzMode,
    BackendIdentifier,
    IterationRecord,
    IterationTrace,
    RunManifest,
)

ExperimentName = Literal["h2", "lih"]

log = get_logger("benchmark")


@dataclass(slots=True)
class BenchmarkSpec:
    """One row of the benchmark suite.

    ``ansatz_mode`` is only meaningful for experiments with an active space
    (LiH). It is ignored for H2, which has no active space and therefore
    resolves to the same 4-qubit / 3-parameter circuit either way.
    """

    experiment: ExperimentName
    backend: BackendIdentifier
    seed: int = 42
    max_iterations: int = 200
    bond_distance: float | None = None
    label: str | None = None
    ansatz_mode: AnsatzMode = AnsatzMode.MATCHED


def run_one(
    spec: BenchmarkSpec,
    on_iteration: Callable[[IterationRecord], None] | None = None,
) -> tuple[RunManifest, IterationTrace]:
    """Execute a single benchmark configuration."""
    if spec.experiment == "h2":
        return run_h2(
            backend_id=spec.backend,
            bond_distance=spec.bond_distance if spec.bond_distance is not None else 0.7414,
            seed=spec.seed,
            max_iterations=spec.max_iterations,
            on_iteration=on_iteration,
        )
    if spec.experiment == "lih":
        return run_lih(
            backend_id=spec.backend,
            bond_distance=spec.bond_distance if spec.bond_distance is not None else 1.5957,
            seed=spec.seed,
            max_iterations=spec.max_iterations,
            on_iteration=on_iteration,
            ansatz_mode=spec.ansatz_mode,
        )
    raise ValueError(f"unknown experiment: {spec.experiment}")


def run_benchmark_suite(
    specs: list[BenchmarkSpec],
    *,
    on_iteration: Callable[[IterationRecord], None] | None = None,
) -> list[RunManifest]:
    """Run a list of benchmark specs and persist each as a regular run."""
    manifests: list[RunManifest] = []
    for spec in specs:
        log.info(
            "benchmark.spec_start",
            experiment=spec.experiment,
            backend=spec.backend.value,
            seed=spec.seed,
            label=spec.label,
        )
        manifest, trace = run_one(spec, on_iteration=on_iteration)
        save_manifest(manifest)
        save_trace(trace)
        manifests.append(manifest)
        log.info(
            "benchmark.spec_done",
            experiment=spec.experiment,
            backend=spec.backend.value,
            run_id=manifest.run_id,
            energy=manifest.result.energy if manifest.result else None,
            wall_time=manifest.result.wall_time_seconds if manifest.result else None,
        )
    return manifests


def lih_active_space_followup_suite(
    seeds: tuple[int, ...] = (42, 43, 44, 45, 46),
    *,
    max_iterations: int = 1500,
) -> list[BenchmarkSpec]:
    """Controlled follow-up comparing the legacy and matched LiH ansatz.

    Three arms, all LiH, all at the same iteration budget so the only thing
    that varies within a backend is the ansatz dimensioning:

    1. ``legacy_full`` on ``gpu_fp64`` - reproduces the v0.1 configuration
       (12-qubit / 92-parameter circuit against the active-space Hamiltonian).
    2. ``matched`` on ``gpu_fp64`` - the hypothesis arm (10-qubit /
       24-parameter circuit against the same Hamiltonian).
    3. ``matched`` on ``cpu`` - the CPU counterpart of arm 2, which gives
       both a CPU/GPU wall-time ratio for the matched ansatz and a numerical
       cross-check that the two backends agree seed-for-seed.

    Arms are emitted contiguously (all of arm 1, then all of arm 2, then all
    of arm 3) so the runner never interleaves CPU and GPU work.

    H2 and GPU FP32 are deliberately excluded: H2 has no active space so the
    ansatz mode cannot change anything, and FP32 would confound a precision
    difference with the ansatz difference under test.
    """
    arms: list[tuple[AnsatzMode, BackendIdentifier]] = [
        (AnsatzMode.LEGACY_FULL, BackendIdentifier.GPU_FP64),
        (AnsatzMode.MATCHED, BackendIdentifier.GPU_FP64),
        (AnsatzMode.MATCHED, BackendIdentifier.CPU),
    ]
    suite: list[BenchmarkSpec] = []
    for mode, backend in arms:
        for seed in seeds:
            suite.append(
                BenchmarkSpec(
                    experiment="lih",
                    backend=backend,
                    seed=seed,
                    max_iterations=max_iterations,
                    ansatz_mode=mode,
                    label=f"lih/{mode.value}/{backend.value}/seed{seed}",
                )
            )
    return suite


def default_blog_suite(
    seeds: tuple[int, ...] = (42, 43, 44),
    *,
    h2_max_iterations: int = 200,
    lih_max_iterations: int = 1500,
) -> list[BenchmarkSpec]:
    """The exact suite of runs that produces the blog-ready charts.

    For each backend we sweep multiple seeds so we can quote mean ± stderr.

    LiH defaults to ``max_iterations=1500``: with 92 UCCSD parameters on a
    12-qubit kernel (the LiH ansatz currently instantiates the full-molecule
    parameter space rather than restricting to the (2e, 5o) active space,
    even though the Hamiltonian itself does carry the active-space
    restriction), COBYLA needs the full 1500 iterations to settle into the
    converged basin. ``h2`` is left at 200 because COBYLA already plateaus
    there in ~75 evaluations.

    Three seeds is a deliberate trade-off. The 2026-05-04 multi-seed bench in
    Akamai's id-cgk (Jakarta) region, billed at the regional rate of $3.00/hr
    (a $0.50/hr uplift on the $2.50/hr base SKU), came in at ~$10.75 of total
    VM lifetime: ~2 h 27 min of bench compute plus ~1 h 8 min of provisioning
    + driver install + container build + results export + teardown overhead.
    Bumping the seed count from 3 to 5 would have added roughly an hour of
    LiH compute (each LiH seed is ~30 min on CPU and ~18 min on GPU), pushing
    the bench cycle past $13 without changing the qualitative conclusions.
    Three seeds gives a workable n=3 mean +/- stderr at acceptable cost.
    """
    suite: list[BenchmarkSpec] = []
    backends_h2 = [BackendIdentifier.CPU, BackendIdentifier.GPU_FP32, BackendIdentifier.GPU_FP64]
    backends_lih = [BackendIdentifier.CPU, BackendIdentifier.GPU_FP64]
    for backend in backends_h2:
        for seed in seeds:
            suite.append(
                BenchmarkSpec(
                    experiment="h2",
                    backend=backend,
                    seed=seed,
                    max_iterations=h2_max_iterations,
                    label=f"h2/{backend.value}/seed{seed}",
                )
            )
    for backend in backends_lih:
        for seed in seeds:
            suite.append(
                BenchmarkSpec(
                    experiment="lih",
                    backend=backend,
                    seed=seed,
                    max_iterations=lih_max_iterations,
                    label=f"lih/{backend.value}/seed{seed}",
                )
            )
    return suite
