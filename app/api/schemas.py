"""Request and response schemas exposed by the HTTP API.

These are intentionally separate from the durable manifest schemas in
``app.storage.manifests`` so the wire format and the on-disk format can
evolve independently.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.storage.manifests import AnsatzMode, BackendIdentifier, Molecule, RunStatus


class HealthResponse(BaseModel):
    status: str = "ok"
    project_version: str
    cudaq_version: str | None
    gpus: list[dict[str, Any]] = Field(default_factory=list)
    backends_available: list[BackendIdentifier]


class StartRunRequest(BaseModel):
    backend: BackendIdentifier = BackendIdentifier.CPU
    bond_distance: float | None = None
    max_iterations: int = 200
    seed: int = 42
    n_core_orbitals: int | None = 1
    n_active_orbitals: int | None = 5
    # Only consulted for molecules with an active space (LiH). H2 resolves
    # identically under both modes.
    ansatz_mode: AnsatzMode = AnsatzMode.MATCHED


class StartRunResponse(BaseModel):
    run_id: str
    status: RunStatus
    backend: BackendIdentifier
    molecule: Molecule
    started_at: datetime


class RunSummary(BaseModel):
    run_id: str
    molecule: Molecule
    backend: BackendIdentifier
    target_string: str
    status: RunStatus
    started_at: datetime
    completed_at: datetime | None
    energy: float | None
    iterations: int | None
    wall_time_seconds: float | None
    error_vs_reference_hartree: float | None
    chemical_accuracy_reached: bool | None


class IterationEvent(BaseModel):
    """SSE payload pushed to clients during a live VQE run."""

    iteration: int
    energy: float
    elapsed_seconds: float


class CompareResponse(BaseModel):
    by_molecule: dict[str, Any]
    totals: dict[str, Any]
