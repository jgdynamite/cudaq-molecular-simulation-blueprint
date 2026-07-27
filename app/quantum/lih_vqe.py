"""LiH VQE experiment.

LiH / STO-3G with the default (2 electron, 5 orbital) active space. This is
the workload that makes the CPU-vs-GPU comparison meaningful: small enough
to run on CPU as a baseline, large enough that the GPU statevector simulator
pulls ahead noticeably.

Ansatz modes
------------
Through v0.1 the Hamiltonian carried the active-space restriction while the
UCCSD ansatz was instantiated against the *full* molecule (12 qubits,
4 electrons, 92 parameters). The active space itself only spans 10 qubits /
2 electrons / 24 parameters, so the circuit and the operator did not describe
the same orbital set.

``ansatz_mode`` makes that choice explicit and defaults to
:attr:`~app.storage.manifests.AnsatzMode.MATCHED`. Pass ``LEGACY_FULL`` to
reproduce the v0.1 configuration. The two ``experiment_variant`` labels below
are what downstream reporting groups on; they are never aggregated together.
"""

from __future__ import annotations

from collections.abc import Callable

from app.quantum.chemistry import LIH_DEFAULT_BOND_DISTANCE, build_lih
from app.quantum.experiment import run_vqe
from app.storage.manifests import (
    EXPERIMENT_VARIANT_LEGACY_FULL,
    EXPERIMENT_VARIANT_MATCHED,
    LIH_EXPERIMENT_VARIANT_BY_MODE,
    AnsatzMode,
    BackendIdentifier,
    IterationRecord,
    IterationTrace,
    RunManifest,
)

__all__ = [
    "EXPERIMENT_VARIANT_LEGACY_FULL",
    "EXPERIMENT_VARIANT_MATCHED",
    "lih_experiment_variant",
    "run_lih",
]


def lih_experiment_variant(ansatz_mode: AnsatzMode) -> str:
    """Return the ``experiment_variant`` label for an LiH ansatz mode."""
    return LIH_EXPERIMENT_VARIANT_BY_MODE[ansatz_mode]


def run_lih(
    *,
    backend_id: BackendIdentifier = BackendIdentifier.GPU_FP64,
    bond_distance: float = LIH_DEFAULT_BOND_DISTANCE,
    basis: str = "sto-3g",
    n_core_orbitals: int | None = 1,
    n_active_orbitals: int | None = 5,
    seed: int = 42,
    max_iterations: int = 500,
    tolerance: float = 1e-6,
    initial_parameters: list[float] | None = None,
    on_iteration: Callable[[IterationRecord], None] | None = None,
    run_id: str | None = None,
    ansatz_mode: AnsatzMode = AnsatzMode.MATCHED,
) -> tuple[RunManifest, IterationTrace]:
    """Run the LiH VQE experiment with the given backend + options."""
    molecule = build_lih(
        bond_distance=bond_distance,
        basis=basis,
        n_core_orbitals=n_core_orbitals,
        n_active_orbitals=n_active_orbitals,
    )
    return run_vqe(
        molecule=molecule,
        backend_id=backend_id,
        seed=seed,
        max_iterations=max_iterations,
        tolerance=tolerance,
        initial_parameters=initial_parameters,
        on_iteration=on_iteration,
        notes={
            "experiment": "lih_vqe",
            "experiment_variant": lih_experiment_variant(ansatz_mode),
        },
        run_id=run_id,
        ansatz_mode=ansatz_mode,
    )
