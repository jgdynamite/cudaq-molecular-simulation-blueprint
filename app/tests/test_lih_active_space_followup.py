"""Tests for the LiH active-space ansatz follow-up (v0.2).

These cover the dimension resolver, the active-space validation, the
follow-up benchmark suite shape, and the variant-aware reporting. They run
without CUDA-Q installed: the resolver takes a ``HamiltonianBundle``, which
is a plain dataclass, and the parameter-count path is exercised against a
stub ``cudaq`` module so we can assert the count is *queried* rather than
hard-coded.
"""

from __future__ import annotations

import sys
import types
from datetime import UTC, datetime

import pytest

from app.benchmark.followup import (
    build_comparison,
    collect_rows,
    group_name,
    row_from_run,
)
from app.benchmark.runner import lih_active_space_followup_suite
from app.quantum.ansatz import resolve_ansatz_dimensions
from app.quantum.chemistry import HamiltonianBundle, Molecule, build_h2, build_lih
from app.quantum.lih_vqe import (
    EXPERIMENT_VARIANT_LEGACY_FULL,
    EXPERIMENT_VARIANT_MATCHED,
    lih_experiment_variant,
)
from app.storage.manifests import (
    AnsatzMode,
    BackendIdentifier,
    IterationRecord,
    IterationTrace,
    MoleculeSpec,
    OptimizerSpec,
    RunManifest,
    RunResult,
    RunStatus,
)

# LiH / STO-3G: CUDA-Q's chemistry module reports the full molecule as
# 4 electrons in 6 spatial orbitals (12 spin orbitals) even when an active
# space is requested. The default active space is (2 electrons, 5 orbitals).
LIH_FULL_ELECTRONS = 4
LIH_FULL_ORBITALS = 6
LIH_ACTIVE_ELECTRONS = 2
LIH_ACTIVE_ORBITALS = 5


def lih_bundle() -> HamiltonianBundle:
    """A bundle shaped like the default LiH (2e, 5o) active-space run."""
    return HamiltonianBundle(
        hamiltonian=object(),
        n_electrons=LIH_FULL_ELECTRONS,
        n_orbitals=LIH_FULL_ORBITALS,
        n_qubits=2 * LIH_FULL_ORBITALS,
        n_terms=0,
        active_electrons=LIH_ACTIVE_ELECTRONS,
        active_orbitals=LIH_ACTIVE_ORBITALS,
    )


def h2_bundle() -> HamiltonianBundle:
    """A bundle shaped like the full-space H2 run (no active space)."""
    return HamiltonianBundle(
        hamiltonian=object(),
        n_electrons=2,
        n_orbitals=2,
        n_qubits=4,
        n_terms=0,
    )


# --------------------------------------------------------------------------
# Dimension resolution
# --------------------------------------------------------------------------


def test_matched_lih_resolves_to_two_electrons_and_ten_qubits() -> None:
    dims = resolve_ansatz_dimensions(lih_bundle(), AnsatzMode.MATCHED)

    assert dims.electron_count == 2
    assert dims.qubit_count == 10
    assert dims.orbital_count == 5
    assert dims.mode is AnsatzMode.MATCHED


def test_legacy_full_lih_resolves_to_four_electrons_and_twelve_qubits() -> None:
    dims = resolve_ansatz_dimensions(lih_bundle(), AnsatzMode.LEGACY_FULL)

    assert dims.electron_count == 4
    assert dims.qubit_count == 12
    assert dims.orbital_count == 6
    assert dims.mode is AnsatzMode.LEGACY_FULL


def test_matched_is_the_default_mode() -> None:
    assert resolve_ansatz_dimensions(lih_bundle()).qubit_count == 10


def test_h2_is_unaffected_by_ansatz_mode() -> None:
    """H2 has no active space, so both modes must agree on 4 qubits."""
    matched = resolve_ansatz_dimensions(h2_bundle(), AnsatzMode.MATCHED)
    legacy = resolve_ansatz_dimensions(h2_bundle(), AnsatzMode.LEGACY_FULL)

    assert matched.electron_count == legacy.electron_count == 2
    assert matched.qubit_count == legacy.qubit_count == 4


# --------------------------------------------------------------------------
# Parameter count comes from CUDA-Q, not from a constant in our code
# --------------------------------------------------------------------------


class _StubUccsdKernels:
    """Records the arguments ``make_uccsd_kernel`` asks about."""

    def __init__(self, answer: int) -> None:
        self.answer = answer
        self.calls: list[tuple[int, int]] = []

    def uccsd_num_parameters(self, electron_count: int, qubit_count: int) -> int:
        self.calls.append((electron_count, qubit_count))
        return self.answer


def _install_stub_cudaq(monkeypatch: pytest.MonkeyPatch, answer: int) -> _StubUccsdKernels:
    """Install a minimal fake ``cudaq`` module and return its kernels stub.

    ``make_uccsd_kernel`` imports cudaq lazily and decorates the kernel with
    ``@cudaq.kernel``; a pass-through decorator is enough because we never
    execute the kernel here.
    """
    kernels = _StubUccsdKernels(answer)
    stub = types.ModuleType("cudaq")
    stub.kernels = kernels  # type: ignore[attr-defined]
    stub.kernel = lambda fn: fn  # type: ignore[attr-defined]
    stub.qvector = lambda n: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cudaq", stub)
    return kernels


def test_parameter_count_is_queried_from_cudaq_not_hard_coded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The count must come from CUDA-Q, with the resolved dimensions.

    A sentinel value CUDA-Q would never return proves we pass the number
    through rather than computing or hard-coding it ourselves.
    """
    from app.quantum.ansatz import make_uccsd_kernel

    sentinel = 4242
    kernels = _install_stub_cudaq(monkeypatch, sentinel)

    ansatz = make_uccsd_kernel(qubit_count=10, electron_count=2)

    assert ansatz.parameter_count == sentinel
    assert kernels.calls == [(2, 10)], "must query CUDA-Q with (electrons, qubits)"


def test_matched_lih_asks_cudaq_about_the_active_space_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.quantum.ansatz import make_uccsd_kernel

    kernels = _install_stub_cudaq(monkeypatch, 24)
    dims = resolve_ansatz_dimensions(lih_bundle(), AnsatzMode.MATCHED)

    make_uccsd_kernel(dims.qubit_count, dims.electron_count)

    assert kernels.calls == [(2, 10)]


def test_legacy_lih_asks_cudaq_about_the_full_molecule_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.quantum.ansatz import make_uccsd_kernel

    kernels = _install_stub_cudaq(monkeypatch, 92)
    dims = resolve_ansatz_dimensions(lih_bundle(), AnsatzMode.LEGACY_FULL)

    make_uccsd_kernel(dims.qubit_count, dims.electron_count)

    assert kernels.calls == [(4, 12)]


# --------------------------------------------------------------------------
# Active-space validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n_core", "n_active"),
    [(1, None), (None, 5), (0, None), (None, 0)],
)
def test_incomplete_active_space_raises_value_error(
    n_core: int | None, n_active: int | None
) -> None:
    with pytest.raises(ValueError, match="active space is under-specified"):
        Molecule(
            name="lih",
            geometry=(("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.5957))),
            n_core_orbitals=n_core,
            n_active_orbitals=n_active,
        )


def test_incomplete_active_space_raises_from_build_lih() -> None:
    with pytest.raises(ValueError, match="active space is under-specified"):
        build_lih(n_core_orbitals=1, n_active_orbitals=None)


@pytest.mark.parametrize(
    ("n_core", "n_active"),
    [(1, 5), (None, None), (0, 6)],
)
def test_complete_active_space_specifications_are_accepted(
    n_core: int | None, n_active: int | None
) -> None:
    molecule = build_lih(n_core_orbitals=n_core, n_active_orbitals=n_active)

    assert molecule.has_active_space is (n_core is not None)


def test_h2_has_no_active_space() -> None:
    assert build_h2().has_active_space is False


# --------------------------------------------------------------------------
# Follow-up suite shape
# --------------------------------------------------------------------------


def test_followup_suite_has_exactly_fifteen_specs_for_five_seeds() -> None:
    suite = lih_active_space_followup_suite()

    assert len(suite) == 15
    assert sorted({s.seed for s in suite}) == [42, 43, 44, 45, 46]


def test_followup_suite_has_the_three_expected_arms() -> None:
    suite = lih_active_space_followup_suite()
    arms = {(s.ansatz_mode, s.backend) for s in suite}

    assert arms == {
        (AnsatzMode.LEGACY_FULL, BackendIdentifier.GPU_FP64),
        (AnsatzMode.MATCHED, BackendIdentifier.GPU_FP64),
        (AnsatzMode.MATCHED, BackendIdentifier.CPU),
    }
    for arm in arms:
        matching = [s for s in suite if (s.ansatz_mode, s.backend) == arm]
        assert len(matching) == 5, f"arm {arm} should have one spec per seed"


def test_followup_suite_is_lih_only_and_excludes_fp32() -> None:
    suite = lih_active_space_followup_suite()

    assert {s.experiment for s in suite} == {"lih"}
    assert BackendIdentifier.GPU_FP32 not in {s.backend for s in suite}


def test_followup_suite_uses_one_iteration_budget_for_every_run() -> None:
    suite = lih_active_space_followup_suite()

    assert {s.max_iterations for s in suite} == {1500}


def test_followup_suite_does_not_interleave_cpu_and_gpu() -> None:
    """Each backend must appear as one contiguous block."""
    backends = [s.backend for s in lih_active_space_followup_suite()]
    blocks = [b for i, b in enumerate(backends) if i == 0 or backends[i - 1] != b]

    assert len(blocks) == len(set(blocks)), f"backend blocks are interleaved: {backends}"


def test_followup_suite_honours_custom_seeds() -> None:
    suite = lih_active_space_followup_suite(seeds=(1, 2))

    assert len(suite) == 6
    assert sorted({s.seed for s in suite}) == [1, 2]


def test_experiment_variant_labels() -> None:
    assert lih_experiment_variant(AnsatzMode.MATCHED) == EXPERIMENT_VARIANT_MATCHED
    assert lih_experiment_variant(AnsatzMode.LEGACY_FULL) == EXPERIMENT_VARIANT_LEGACY_FULL
    assert EXPERIMENT_VARIANT_MATCHED != EXPERIMENT_VARIANT_LEGACY_FULL


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def make_manifest(
    *,
    run_id: str,
    backend: BackendIdentifier,
    seed: int,
    energy: float,
    wall_time: float,
    qubits: int,
    parameters: int,
    ansatz_mode: str | None,
    experiment_variant: str | None,
    reference: float | None = -7.882164,
) -> RunManifest:
    """Build a completed LiH manifest.

    Passing ``ansatz_mode=None`` and ``experiment_variant=None`` produces a
    manifest shaped like a pre-v0.2 file, i.e. one with none of the new
    notes keys.
    """
    notes: dict[str, object] = {"experiment": "lih_vqe"}
    if ansatz_mode is not None:
        notes["ansatz_mode"] = ansatz_mode
        notes["ansatz_name"] = "uccsd"
    if experiment_variant is not None:
        notes["experiment_variant"] = experiment_variant

    error = None if reference is None else energy - reference

    return RunManifest(
        run_id=run_id,
        project_version="0.2.0",
        created_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        status=RunStatus.COMPLETED,
        backend=backend,
        target_string="nvidia-fp64" if backend != BackendIdentifier.CPU else "qpp-cpu",
        seed=seed,
        molecule=MoleculeSpec(
            name="lih",  # type: ignore[arg-type]
            geometry=[("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.5957))],
            active_electrons=2,
            active_orbitals=5,
        ),
        optimizer=OptimizerSpec(max_iterations=1500),
        qubit_count=qubits,
        parameter_count=parameters,
        system_info={
            "cudaq_version": "0.14.0",
            "git_sha": "abc1234",
            "gpus": [{"name": "RTX PRO 6000 Blackwell", "driver_version": "580.65"}],
        },
        result=RunResult(
            energy=energy,
            iterations=1500,
            parameters=[0.0] * parameters,
            wall_time_seconds=wall_time,
            converged=True,
            reference_energy=reference,
            error_vs_reference_hartree=error,
            chemical_accuracy_reached=None if error is None else abs(error) < 1.6e-3,
        ),
        notes=notes,
    )


def _trace(run_id: str, n: int = 10) -> IterationTrace:
    return IterationTrace(
        run_id=run_id,
        records=[
            IterationRecord(iteration=i, energy=-7.0 - i * 0.01, elapsed_seconds=float(i))
            for i in range(n)
        ],
    )


def _rows_for_two_arms() -> list:
    """One matched-GPU and one legacy-GPU run per seed, same seeds."""
    rows = []
    for seed in (42, 43):
        matched = make_manifest(
            run_id=f"matched-gpu-{seed}",
            backend=BackendIdentifier.GPU_FP64,
            seed=seed,
            energy=-7.8817,
            wall_time=300.0,
            qubits=10,
            parameters=24,
            ansatz_mode="matched",
            experiment_variant=EXPERIMENT_VARIANT_MATCHED,
        )
        legacy = make_manifest(
            run_id=f"legacy-gpu-{seed}",
            backend=BackendIdentifier.GPU_FP64,
            seed=seed,
            energy=-7.8752,
            wall_time=1080.0,
            qubits=12,
            parameters=92,
            ansatz_mode="legacy_full",
            experiment_variant=EXPERIMENT_VARIANT_LEGACY_FULL,
        )
        rows.append(row_from_run(matched, _trace(matched.run_id)))
        rows.append(row_from_run(legacy, _trace(legacy.run_id)))
    return rows


def test_comparison_never_combines_ansatz_variants() -> None:
    report = build_comparison(_rows_for_two_arms())
    groups = report["groups"]

    assert len(groups) == 2
    for summary in groups.values():
        assert summary["n"] == 2
        # A group that merged the two arms would show both qubit counts.
        assert summary["qubit_count"] in (10, 12)

    matched_key = group_name(EXPERIMENT_VARIANT_MATCHED, "matched", "gpu_fp64")
    legacy_key = group_name(EXPERIMENT_VARIANT_LEGACY_FULL, "legacy_full", "gpu_fp64")

    assert groups[matched_key]["qubit_count"] == 10
    assert groups[matched_key]["parameter_count"] == 24
    assert groups[legacy_key]["qubit_count"] == 12
    assert groups[legacy_key]["parameter_count"] == 92


def test_group_key_separates_variants_sharing_a_backend() -> None:
    matched = group_name(EXPERIMENT_VARIANT_MATCHED, "matched", "gpu_fp64")
    legacy = group_name(EXPERIMENT_VARIANT_LEGACY_FULL, "legacy_full", "gpu_fp64")

    assert matched != legacy


def test_paired_comparison_matches_runs_by_seed() -> None:
    report = build_comparison(_rows_for_two_arms())
    pairs = report["paired_matched_gpu_vs_legacy_gpu"]

    assert [p["seed"] for p in pairs] == [42, 43]
    for pair in pairs:
        seed = pair["seed"]
        assert pair["left_run_id"] == f"matched-gpu-{seed}"
        assert pair["right_run_id"] == f"legacy-gpu-{seed}"
        assert pair["wall_seconds_difference"] == pytest.approx(300.0 - 1080.0)


def test_paired_comparison_ignores_unmatched_seeds() -> None:
    """A seed present in only one arm must not produce a pair."""
    rows = _rows_for_two_arms()
    extra = make_manifest(
        run_id="matched-gpu-99",
        backend=BackendIdentifier.GPU_FP64,
        seed=99,
        energy=-7.88,
        wall_time=310.0,
        qubits=10,
        parameters=24,
        ansatz_mode="matched",
        experiment_variant=EXPERIMENT_VARIANT_MATCHED,
    )
    rows.append(row_from_run(extra, _trace(extra.run_id)))

    pairs = build_comparison(rows)["paired_matched_gpu_vs_legacy_gpu"]

    assert [p["seed"] for p in pairs] == [42, 43]


def test_cpu_gpu_energy_equivalence_within_tolerance() -> None:
    rows = []
    for seed in (42, 43):
        gpu = make_manifest(
            run_id=f"m-gpu-{seed}",
            backend=BackendIdentifier.GPU_FP64,
            seed=seed,
            energy=-7.8817,
            wall_time=300.0,
            qubits=10,
            parameters=24,
            ansatz_mode="matched",
            experiment_variant=EXPERIMENT_VARIANT_MATCHED,
        )
        cpu = make_manifest(
            run_id=f"m-cpu-{seed}",
            backend=BackendIdentifier.CPU,
            seed=seed,
            energy=-7.8817 + 1e-12,
            wall_time=600.0,
            qubits=10,
            parameters=24,
            ansatz_mode="matched",
            experiment_variant=EXPERIMENT_VARIANT_MATCHED,
        )
        rows.append(row_from_run(gpu, _trace(gpu.run_id)))
        rows.append(row_from_run(cpu, _trace(cpu.run_id)))

    check = build_comparison(rows)["numerical_equivalence_matched_cpu_vs_matched_gpu"]

    assert check["n_compared"] == 2
    assert check["all_within_tolerance"] is True
    assert check["n_violations"] == 0
    assert build_comparison(rows)["ratios"][
        "matched_cpu_over_matched_gpu_wall_time"
    ] == pytest.approx(2.0)


def test_energy_equivalence_violations_are_reported_not_discarded() -> None:
    gpu = make_manifest(
        run_id="m-gpu-42",
        backend=BackendIdentifier.GPU_FP64,
        seed=42,
        energy=-7.8817,
        wall_time=300.0,
        qubits=10,
        parameters=24,
        ansatz_mode="matched",
        experiment_variant=EXPERIMENT_VARIANT_MATCHED,
    )
    cpu = make_manifest(
        run_id="m-cpu-42",
        backend=BackendIdentifier.CPU,
        seed=42,
        energy=-7.8800,  # 1.7 mHa apart: far beyond the 1e-8 Ha tolerance
        wall_time=600.0,
        qubits=10,
        parameters=24,
        ansatz_mode="matched",
        experiment_variant=EXPERIMENT_VARIANT_MATCHED,
    )
    rows = [row_from_run(gpu, _trace(gpu.run_id)), row_from_run(cpu, _trace(cpu.run_id))]

    check = build_comparison(rows)["numerical_equivalence_matched_cpu_vs_matched_gpu"]

    assert check["all_within_tolerance"] is False
    assert check["n_violations"] == 1
    assert check["violations"][0]["seed"] == 42
    # The run still appears in its group; nothing is filtered out.
    matched_cpu_key = group_name(EXPERIMENT_VARIANT_MATCHED, "matched", "cpu")
    assert build_comparison(rows)["groups"][matched_cpu_key]["n"] == 1


def test_old_manifests_without_new_notes_remain_readable() -> None:
    """A pre-v0.2 manifest must parse and land in its own bucket."""
    old = make_manifest(
        run_id="20260503T224156Z-40afc5",
        backend=BackendIdentifier.CPU,
        seed=42,
        energy=-7.87524,
        wall_time=1795.294,
        qubits=12,
        parameters=92,
        ansatz_mode=None,
        experiment_variant=None,
    )

    row = row_from_run(old, _trace(old.run_id))

    assert row.ansatz_mode == "unspecified"
    assert row.experiment_variant == "unlabelled_pre_v02_lih"
    assert row.qubits == 12
    assert row.parameters == 92
    assert row.absolute_error_mhartree == pytest.approx(6.924, abs=1e-3)

    # It must not be merged into either labelled arm.
    report = build_comparison([row, *_rows_for_two_arms()])
    assert len(report["groups"]) == 3
    assert row.group_key in report["groups"]
    assert report["groups"][row.group_key]["n"] == 1


def test_old_manifest_is_excluded_from_paired_comparisons() -> None:
    old = make_manifest(
        run_id="old-42",
        backend=BackendIdentifier.GPU_FP64,
        seed=42,
        energy=-7.87524,
        wall_time=1795.0,
        qubits=12,
        parameters=92,
        ansatz_mode=None,
        experiment_variant=None,
    )
    rows = [*_rows_for_two_arms(), row_from_run(old, _trace(old.run_id))]

    pairs = build_comparison(rows)["paired_matched_gpu_vs_legacy_gpu"]

    assert all(p["right_run_id"] != "old-42" for p in pairs)


def test_summary_csv_round_trips(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import csv

    from app.benchmark.followup import SUMMARY_COLUMNS, write_summary_csv

    rows = _rows_for_two_arms()
    path = write_summary_csv(rows, tmp_path / "SUMMARY.csv")

    with path.open(encoding="utf-8") as handle:
        parsed = list(csv.DictReader(handle))

    assert list(parsed[0].keys()) == SUMMARY_COLUMNS
    assert len(parsed) == len(rows)
    assert {r["experiment_variant"] for r in parsed} == {
        EXPERIMENT_VARIANT_MATCHED,
        EXPERIMENT_VARIANT_LEGACY_FULL,
    }
    assert {r["gpu_model"] for r in parsed} == {"RTX PRO 6000 Blackwell"}
    assert {r["git_commit_sha"] for r in parsed} == {"abc1234"}
    assert {r["cudaq_version"] for r in parsed} == {"0.14.0"}


def test_collect_rows_reads_manifests_from_disk() -> None:
    from app.storage.filesystem import save_manifest, save_trace

    manifest = make_manifest(
        run_id="disk-42",
        backend=BackendIdentifier.GPU_FP64,
        seed=42,
        energy=-7.8817,
        wall_time=300.0,
        qubits=10,
        parameters=24,
        ansatz_mode="matched",
        experiment_variant=EXPERIMENT_VARIANT_MATCHED,
    )
    save_manifest(manifest)
    save_trace(_trace("disk-42"))

    rows = collect_rows()

    assert [r.run_id for r in rows] == ["disk-42"]
    assert rows[0].qubits == 10
    assert rows[0].parameters == 24
