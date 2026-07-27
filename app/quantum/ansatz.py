"""UCCSD ansatz construction backed by CUDA-Q's built-in kernel.

We expose a :func:`make_uccsd_kernel` factory that returns a CUDA-Q kernel
along with its parameter count, plus :func:`resolve_ansatz_dimensions`, which
decides *which* dimensions the ansatz should be built for when the
Hamiltonian carries an active-space restriction.

The parameter count is always obtained from
``cudaq.kernels.uccsd_num_parameters`` rather than hard-coded, so it stays
correct if CUDA-Q changes its excitation enumeration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from app.storage.manifests import AnsatzMode

if TYPE_CHECKING:  # pragma: no cover
    from app.quantum.chemistry import HamiltonianBundle


@dataclass(slots=True)
class UccsdAnsatz:
    """A built UCCSD ansatz ready to be passed to :func:`cudaq.observe`."""

    kernel: Any
    qubit_count: int
    electron_count: int
    parameter_count: int


@dataclass(frozen=True, slots=True)
class AnsatzDimensions:
    """The electron/qubit counts an ansatz will be instantiated for."""

    electron_count: int
    qubit_count: int
    mode: AnsatzMode

    @property
    def orbital_count(self) -> int:
        return self.qubit_count // 2


def resolve_ansatz_dimensions(
    bundle: HamiltonianBundle,
    mode: AnsatzMode = AnsatzMode.MATCHED,
) -> AnsatzDimensions:
    """Pick the ansatz dimensions for ``bundle`` under ``mode``.

    ``MATCHED`` uses the active-space dimensions when the Hamiltonian has an
    active space, so the circuit spans exactly the orbitals the Hamiltonian
    was restricted to. ``LEGACY_FULL`` always uses the full-molecule
    dimensions CUDA-Q reports, reproducing the pre-v0.2 behavior.

    When there is no active space the two modes agree by construction, which
    is why H2 is unaffected by this setting.
    """
    active_orbitals = bundle.active_orbitals
    if mode is AnsatzMode.MATCHED and active_orbitals is not None:
        electron_count = (
            bundle.active_electrons if bundle.active_electrons is not None else bundle.n_electrons
        )
        return AnsatzDimensions(
            electron_count=electron_count,
            qubit_count=2 * active_orbitals,
            mode=mode,
        )
    return AnsatzDimensions(
        electron_count=bundle.n_electrons,
        qubit_count=bundle.n_qubits,
        mode=mode,
    )


def make_uccsd_kernel(qubit_count: int, electron_count: int) -> UccsdAnsatz:
    """Build the standard UCCSD ansatz on top of a Hartree-Fock reference.

    Returns a kernel whose signature is::

        kernel(thetas: list[float], electron_count: int, qubit_count: int) -> None

    The integer dimensions are passed in explicitly because CUDA-Q kernels do
    not support closing over Python ints from the enclosing scope.
    """
    import cudaq

    # Gate names like `x` and helpers like `qvector` are NOT importable from
    # the cudaq module - they are resolved by the kernel's AST bridge at
    # decoration time. We use bare names inside the kernel body.

    parameter_count = int(cudaq.kernels.uccsd_num_parameters(electron_count, qubit_count))

    @cudaq.kernel
    def kernel(thetas: list[float], n_electrons: int, n_qubits: int) -> None:
        qubits = cudaq.qvector(n_qubits)
        for i in range(n_electrons):
            x(qubits[i])  # noqa: F821 - resolved by @cudaq.kernel AST bridge
        cudaq.kernels.uccsd(qubits, thetas, n_electrons, n_qubits)

    return UccsdAnsatz(
        kernel=kernel,
        qubit_count=qubit_count,
        electron_count=electron_count,
        parameter_count=parameter_count,
    )
