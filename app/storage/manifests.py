"""Pydantic schemas for run manifests and per-iteration traces.

Manifests are the durable, JSON-serializable record of every experiment. They
capture *enough* information to fully reproduce a result on a new machine.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class BackendIdentifier(StrEnum):
    """Logical backend identifiers exposed to users.

    The mapping to CUDA-Q target strings lives in :mod:`app.quantum.backends`.
    """

    CPU = "cpu"
    GPU_FP32 = "gpu_fp32"
    GPU_FP64 = "gpu_fp64"


class Molecule(StrEnum):
    """Supported molecules in v1."""

    H2 = "h2"
    LIH = "lih"


class AnsatzMode(StrEnum):
    """How the UCCSD ansatz dimensions are derived when an active space is used.

    ``MATCHED``
        Build the ansatz from the *active-space* dimensions, so the circuit
        acts on exactly the orbitals the Hamiltonian was restricted to. For
        the default LiH (2 electron, 5 orbital) active space this is a
        10-qubit / 2-electron ansatz.

    ``LEGACY_FULL``
        Reproduce the pre-v0.2 behavior: build the ansatz from the
        full-molecule dimensions reported by CUDA-Q's chemistry module even
        when the Hamiltonian carries an active-space restriction. For LiH
        this is a 12-qubit / 4-electron ansatz. Retained so the v0.1
        published numbers stay reproducible and so the two arms can be
        compared under otherwise identical conditions.

    Molecules without an active space (H2) resolve identically under both
    modes, because the active-space dimensions *are* the full dimensions.
    """

    MATCHED = "matched"
    LEGACY_FULL = "legacy_full"


#: ``experiment_variant`` labels recorded in LiH manifest notes. These are
#: part of the persisted schema: reporting groups on them and must never
#: merge two different variants into one aggregate.
EXPERIMENT_VARIANT_LEGACY_FULL = "lih_active_hamiltonian_legacy_full_ansatz_v01"
EXPERIMENT_VARIANT_MATCHED = "lih_active_space_matched_ansatz_v02"

LIH_EXPERIMENT_VARIANT_BY_MODE: dict[AnsatzMode, str] = {
    AnsatzMode.LEGACY_FULL: EXPERIMENT_VARIANT_LEGACY_FULL,
    AnsatzMode.MATCHED: EXPERIMENT_VARIANT_MATCHED,
}


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class MoleculeSpec(BaseModel):
    """Concrete molecule specification embedded in a manifest."""

    model_config = ConfigDict(frozen=True)

    name: Molecule
    geometry: list[tuple[str, tuple[float, float, float]]]
    basis: str = "sto-3g"
    charge: int = 0
    multiplicity: int = 1
    active_electrons: int | None = None
    active_orbitals: int | None = None


class OptimizerSpec(BaseModel):
    """Optimizer configuration embedded in a manifest."""

    model_config = ConfigDict(frozen=True)

    name: str = "cobyla"
    max_iterations: int = 200
    tolerance: float | None = None
    initial_parameters: list[float] | None = None


class IterationRecord(BaseModel):
    """A single VQE iteration: parameters in, expectation value out."""

    model_config = ConfigDict(frozen=True)

    iteration: int
    energy: float
    elapsed_seconds: float
    parameters: list[float] | None = None


class IterationTrace(BaseModel):
    """Ordered list of per-iteration records for one VQE run."""

    run_id: str
    records: list[IterationRecord] = Field(default_factory=list)


class RunResult(BaseModel):
    """The final outcome of a VQE run."""

    model_config = ConfigDict(frozen=True)

    energy: float
    iterations: int
    parameters: list[float]
    wall_time_seconds: float
    converged: bool
    reference_energy: float | None = None
    error_vs_reference_hartree: float | None = None
    chemical_accuracy_reached: bool | None = None


class RunManifest(BaseModel):
    """Top-level manifest written for every experiment run.

    Stored at ``results/<run_id>/manifest.json`` alongside the iteration trace.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    schema_version: str = "1.0"
    run_id: str
    project_name: str = "cudaq-molecular-simulation-blueprint"
    project_version: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    status: RunStatus = RunStatus.PENDING

    backend: BackendIdentifier
    target_string: str
    seed: int

    molecule: MoleculeSpec
    optimizer: OptimizerSpec
    qubit_count: int
    parameter_count: int

    system_info: dict[str, Any]

    result: RunResult | None = None
    error: str | None = None

    notes: dict[str, Any] = Field(default_factory=dict)
