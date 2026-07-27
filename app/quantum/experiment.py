"""Generic VQE experiment driver shared by H2 and LiH.

This module composes the rest of the ``app.quantum`` package into one
cohesive function, :func:`run_vqe`, that takes a molecule + backend + options
and returns a fully populated :class:`RunManifest` (plus the iteration trace).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime

import numpy as np

from app.core.metadata import project_version
from app.core.system_info import collect_system_info
from app.quantum.ansatz import make_uccsd_kernel, resolve_ansatz_dimensions
from app.quantum.backends import select_backend
from app.quantum.chemistry import Molecule, create_hamiltonian
from app.quantum.optimizers import TracingCost, run_cobyla
from app.quantum.reference_data import (
    CHEMICAL_ACCURACY_HARTREE,
    lookup_reference,
)
from app.storage.filesystem import new_run_id
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
from app.storage.manifests import (
    Molecule as MoleculeName,
)


def _parse_active_space(molecule: Molecule) -> tuple[int, int] | None:
    if molecule.n_active_orbitals is None:
        return None
    n_active_orb = molecule.n_active_orbitals
    if molecule.n_core_orbitals is not None:
        ncore_e = 2 * molecule.n_core_orbitals
        n_active_e = _expected_total_electrons(molecule.name) - ncore_e
        return (n_active_e, n_active_orb)
    return None


def _expected_total_electrons(name: str) -> int:
    return {"h2": 2, "lih": 4}.get(name, 0)


def run_vqe(
    *,
    molecule: Molecule,
    backend_id: BackendIdentifier,
    seed: int = 42,
    max_iterations: int = 200,
    rhobeg: float = 0.5,
    tolerance: float = 1e-6,
    initial_parameters: list[float] | None = None,
    on_iteration: Callable[[IterationRecord], None] | None = None,
    notes: dict[str, object] | None = None,
    run_id: str | None = None,
    ansatz_mode: AnsatzMode = AnsatzMode.MATCHED,
) -> tuple[RunManifest, IterationTrace]:
    """Execute one VQE run end-to-end.

    Returns the populated ``RunManifest`` and the ``IterationTrace`` containing
    every cost-function evaluation in order. Both are persisted to disk by the
    caller (CLI / API / benchmark harness).

    Pass ``run_id`` to override the auto-generated identifier; the API uses
    this to allocate the id up front and return it to the client immediately.

    ``ansatz_mode`` selects whether the UCCSD ansatz is built for the
    active-space dimensions (``MATCHED``) or the full-molecule dimensions
    (``LEGACY_FULL``). It only has an effect when the molecule defines an
    active space; see :func:`app.quantum.ansatz.resolve_ansatz_dimensions`.
    """

    run_id = run_id or new_run_id()
    config = select_backend(backend_id)
    bundle = create_hamiltonian(molecule)
    dims = resolve_ansatz_dimensions(bundle, ansatz_mode)
    ansatz = make_uccsd_kernel(dims.qubit_count, dims.electron_count)

    rng = np.random.default_rng(seed)
    if initial_parameters is None:
        x0 = rng.normal(0.0, 0.1, size=ansatz.parameter_count)
    else:
        if len(initial_parameters) != ansatz.parameter_count:
            raise ValueError(
                f"initial_parameters has length {len(initial_parameters)} but "
                f"this ansatz expects {ansatz.parameter_count}"
            )
        x0 = np.asarray(initial_parameters, dtype=float)

    cost = TracingCost(
        kernel=ansatz.kernel,
        hamiltonian=bundle.hamiltonian,
        kernel_args=(ansatz.electron_count, ansatz.qubit_count),
    )
    started_at = datetime.now(UTC)

    optimizer_spec = OptimizerSpec(
        name="cobyla",
        max_iterations=max_iterations,
        tolerance=tolerance,
        initial_parameters=[float(v) for v in x0.tolist()],
    )

    active_space = _parse_active_space(molecule)
    active_e, active_o = active_space if active_space else (None, None)
    bond_distance = molecule.geometry[1][1][2] - molecule.geometry[0][1][2]
    reference = lookup_reference(
        molecule.name,
        molecule.basis,
        bond_distance,
        active_space,
    )

    manifest = RunManifest(
        run_id=run_id,
        project_version=project_version(),
        created_at=started_at,
        status=RunStatus.RUNNING,
        backend=backend_id,
        target_string=config.target_string(),
        seed=seed,
        molecule=MoleculeSpec(
            name=MoleculeName(molecule.name),
            geometry=[list(t) for t in molecule.geometry],  # type: ignore[arg-type]
            basis=molecule.basis,
            charge=molecule.charge,
            multiplicity=molecule.multiplicity,
            active_electrons=active_e,
            active_orbitals=active_o,
        ),
        optimizer=optimizer_spec,
        # These describe the circuit that was actually executed, which under
        # MATCHED is the active-space circuit rather than the full molecule.
        qubit_count=ansatz.qubit_count,
        parameter_count=ansatz.parameter_count,
        system_info=collect_system_info().to_dict(),
        notes={
            **(notes or {}),
            "hamiltonian_terms": bundle.n_terms,
            # Retained under their original keys for backward compatibility
            # with v0.1 manifests: these are the raw chemistry-reported
            # full-molecule numbers, NOT necessarily the executed circuit.
            "n_electrons": bundle.n_electrons,
            "n_orbitals": bundle.n_orbitals,
            "bond_distance_angstrom": bond_distance,
            "reference": asdict(reference) if reference else None,
            # v0.2 ansatz provenance. Explicit rather than derived so an
            # old manifest (which lacks these keys) is never mistaken for a
            # matched-ansatz run.
            "ansatz_name": "uccsd",
            "ansatz_mode": ansatz_mode.value,
            "ansatz_electron_count": ansatz.electron_count,
            "ansatz_orbital_count": dims.orbital_count,
            "ansatz_qubit_count": ansatz.qubit_count,
            "ansatz_parameter_count": ansatz.parameter_count,
            "chemistry_reported_electron_count": bundle.n_electrons,
            "chemistry_reported_orbital_count": bundle.n_orbitals,
            "active_electron_count": bundle.active_electrons,
            "active_orbital_count": bundle.active_orbitals,
        },
    )

    try:
        outcome = run_cobyla(
            cost,
            x0,
            max_iterations=max_iterations,
            rhobeg=rhobeg,
            tolerance=tolerance,
            on_iteration=on_iteration,
        )
        ref_energy = reference.energy_hartree if reference else None
        error = None if ref_energy is None else outcome.energy - ref_energy
        chem_acc = None if error is None else bool(abs(error) < CHEMICAL_ACCURACY_HARTREE)
        manifest = manifest.model_copy(
            update={
                "status": RunStatus.COMPLETED,
                "completed_at": datetime.now(UTC),
                "result": RunResult(
                    energy=outcome.energy,
                    iterations=outcome.iterations,
                    parameters=outcome.parameters,
                    wall_time_seconds=outcome.wall_time_seconds,
                    converged=outcome.converged,
                    reference_energy=ref_energy,
                    error_vs_reference_hartree=error,
                    chemical_accuracy_reached=chem_acc,
                ),
            }
        )
    except Exception as exc:  # pragma: no cover - exercised in error tests
        manifest = manifest.model_copy(
            update={
                "status": RunStatus.FAILED,
                "completed_at": datetime.now(UTC),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        raise
    finally:
        trace = IterationTrace(run_id=run_id, records=cost.records)

    return manifest, trace
