"""Endpoints to start a new VQE run."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException

from app.api.deps import get_coordinator
from app.api.schemas import StartRunRequest, StartRunResponse
from app.core.logging import get_logger
from app.storage.manifests import (
    IterationRecord,
    IterationTrace,
    Molecule,
    RunManifest,
    RunStatus,
)

router = APIRouter(prefix="/runs", tags=["runs"])
log = get_logger("api.run")


def _make_h2_runner(req: StartRunRequest):  # type: ignore[no-untyped-def]
    bond = req.bond_distance if req.bond_distance is not None else 0.7414

    def runner(run_id: str, on_iteration):  # type: ignore[no-untyped-def]
        from app.quantum.h2_vqe import run_h2

        return run_h2(
            backend_id=req.backend,
            bond_distance=bond,
            seed=req.seed,
            max_iterations=req.max_iterations,
            on_iteration=on_iteration,
            run_id=run_id,
        )

    return runner


def _make_lih_runner(req: StartRunRequest):  # type: ignore[no-untyped-def]
    bond = req.bond_distance if req.bond_distance is not None else 1.5957

    def runner(run_id: str, on_iteration):  # type: ignore[no-untyped-def]
        from app.quantum.lih_vqe import run_lih

        core = (
            req.n_core_orbitals
            if req.n_core_orbitals is not None and req.n_core_orbitals > 0
            else None
        )
        active = (
            req.n_active_orbitals
            if req.n_active_orbitals is not None and req.n_active_orbitals > 0
            else None
        )
        return run_lih(
            backend_id=req.backend,
            bond_distance=bond,
            n_core_orbitals=core,
            n_active_orbitals=active,
            seed=req.seed,
            max_iterations=req.max_iterations,
            on_iteration=on_iteration,
            run_id=run_id,
            ansatz_mode=req.ansatz_mode,
        )

    return runner


@router.post("/h2", response_model=StartRunResponse)
async def start_h2(req: StartRunRequest) -> StartRunResponse:
    coordinator = get_coordinator()
    runner = _make_h2_runner(req)
    try:
        run_id = await coordinator.spawn(runner)
    except RuntimeError as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    log.info("api.run_started", experiment="h2", run_id=run_id, backend=req.backend.value)
    return StartRunResponse(
        run_id=run_id,
        status=RunStatus.RUNNING,
        backend=req.backend,
        molecule=Molecule.H2,
        started_at=datetime.now(UTC),
    )


@router.post("/lih", response_model=StartRunResponse)
async def start_lih(req: StartRunRequest) -> StartRunResponse:
    coordinator = get_coordinator()
    runner = _make_lih_runner(req)
    try:
        run_id = await coordinator.spawn(runner)
    except RuntimeError as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    log.info("api.run_started", experiment="lih", run_id=run_id, backend=req.backend.value)
    return StartRunResponse(
        run_id=run_id,
        status=RunStatus.RUNNING,
        backend=req.backend,
        molecule=Molecule.LIH,
        started_at=datetime.now(UTC),
    )


# Re-exported for type checkers / tests; not registered as a route.
__all__ = ["IterationRecord", "IterationTrace", "RunManifest", "router"]
