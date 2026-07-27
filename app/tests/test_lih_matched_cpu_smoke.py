"""CUDA-Q CPU smoke test for the matched-ansatz LiH run.

Skipped when CUDA-Q is unavailable (notably on macOS, where the wheels are
Linux-x86_64 only). This does **not** check convergence: the iteration
budget is deliberately tiny. It exists to prove the plumbing works, namely
that the active-space Hamiltonian and the 10-qubit ansatz are dimensionally
compatible and that ``cudaq.observe`` will actually execute the pair.
"""

from __future__ import annotations

import pytest

from app.tests.conftest import requires_cudaq


@requires_cudaq
@pytest.mark.slow
def test_matched_lih_executes_on_cpu_with_ten_qubits() -> None:
    from app.quantum.lih_vqe import EXPERIMENT_VARIANT_MATCHED, run_lih
    from app.storage.manifests import AnsatzMode, BackendIdentifier

    manifest, trace = run_lih(
        backend_id=BackendIdentifier.CPU,
        seed=42,
        max_iterations=3,
        ansatz_mode=AnsatzMode.MATCHED,
    )

    # cudaq.observe ran the 10-qubit ansatz against the active-space
    # Hamiltonian without a dimension mismatch.
    assert manifest.result is not None
    assert len(trace.records) > 0

    assert manifest.qubit_count == 10, (
        f"matched LiH should execute a 10-qubit circuit, got {manifest.qubit_count}"
    )

    notes = manifest.notes
    assert notes["ansatz_mode"] == AnsatzMode.MATCHED.value
    assert notes["experiment_variant"] == EXPERIMENT_VARIANT_MATCHED
    assert notes["ansatz_qubit_count"] == 10
    assert notes["ansatz_electron_count"] == 2
    assert notes["ansatz_orbital_count"] == 5

    # The parameter count is whatever CUDA-Q says it is, recorded exactly.
    import cudaq

    expected = int(cudaq.kernels.uccsd_num_parameters(2, 10))
    assert manifest.parameter_count == expected
    assert notes["ansatz_parameter_count"] == expected
    assert len(manifest.result.parameters) == expected

    # Raw chemistry metadata is preserved alongside the executed dimensions.
    assert notes["chemistry_reported_electron_count"] == 4
    assert notes["chemistry_reported_orbital_count"] == 6
    assert notes["active_electron_count"] == 2
    assert notes["active_orbital_count"] == 5


@requires_cudaq
@pytest.mark.slow
def test_legacy_full_lih_still_executes_twelve_qubits() -> None:
    """The legacy arm must remain runnable so v0.1 stays reproducible."""
    from app.quantum.lih_vqe import EXPERIMENT_VARIANT_LEGACY_FULL, run_lih
    from app.storage.manifests import AnsatzMode, BackendIdentifier

    manifest, _ = run_lih(
        backend_id=BackendIdentifier.CPU,
        seed=42,
        max_iterations=2,
        ansatz_mode=AnsatzMode.LEGACY_FULL,
    )

    import cudaq

    assert manifest.qubit_count == 12
    assert manifest.parameter_count == int(cudaq.kernels.uccsd_num_parameters(4, 12))
    assert manifest.notes["experiment_variant"] == EXPERIMENT_VARIANT_LEGACY_FULL
