"""Quantum chemistry preprocessing.

Wraps :func:`cudaq.chemistry.create_molecular_hamiltonian` with a typed
``Molecule`` dataclass and active-space helpers. CPU-only; this is the
classical preprocessing portion of the hybrid pipeline.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    pass


@dataclass(frozen=True, slots=True)
class Molecule:
    """Ground-truth molecule specification used by the experiment runners.

    An active space is defined by *both* ``n_core_orbitals`` and
    ``n_active_orbitals``. Supplying only one of the two is rejected at
    construction time rather than silently ignored: before v0.2 a partial
    specification produced a full-molecule Hamiltonian while still looking
    like an active-space experiment to the caller.
    """

    name: str
    geometry: tuple[tuple[str, tuple[float, float, float]], ...]
    basis: str = "sto-3g"
    charge: int = 0
    multiplicity: int = 1
    n_core_orbitals: int | None = None
    n_active_orbitals: int | None = None

    def __post_init__(self) -> None:
        core_set = self.n_core_orbitals is not None
        active_set = self.n_active_orbitals is not None
        if core_set != active_set:
            raise ValueError(
                "active space is under-specified for molecule "
                f"{self.name!r}: n_core_orbitals={self.n_core_orbitals!r}, "
                f"n_active_orbitals={self.n_active_orbitals!r}. "
                "Supply both values to request an active space, or neither "
                "to run the full molecule. Supplying only one is ambiguous: "
                "the Hamiltonian would be built for the full molecule while "
                "the run would be labelled as active-space."
            )

    @property
    def has_active_space(self) -> bool:
        return self.n_core_orbitals is not None and self.n_active_orbitals is not None


@dataclass(slots=True)
class HamiltonianBundle:
    """Result of building a molecular Hamiltonian.

    ``n_electrons`` / ``n_orbitals`` / ``n_qubits`` are what CUDA-Q's
    chemistry module reports for the *full* molecule. When an active space
    was requested these are **not** the dimensions the Hamiltonian acts on;
    ``active_electrons`` / ``active_orbitals`` / ``active_qubits`` carry
    those. Keeping both is what lets a run record the circuit it actually
    executed alongside the raw chemistry metadata.
    """

    hamiltonian: Any
    n_electrons: int
    n_orbitals: int
    n_qubits: int
    n_terms: int
    active_electrons: int | None = None
    active_orbitals: int | None = None

    @property
    def active_qubits(self) -> int | None:
        if self.active_orbitals is None:
            return None
        return 2 * self.active_orbitals


# Standard equilibrium geometries
H2_DEFAULT_BOND_DISTANCE = 0.7414
LIH_DEFAULT_BOND_DISTANCE = 1.5957


def h2_geometry(
    bond_distance: float = H2_DEFAULT_BOND_DISTANCE,
) -> tuple[tuple[str, tuple[float, float, float]], ...]:
    return (
        ("H", (0.0, 0.0, 0.0)),
        ("H", (0.0, 0.0, bond_distance)),
    )


def lih_geometry(
    bond_distance: float = LIH_DEFAULT_BOND_DISTANCE,
) -> tuple[tuple[str, tuple[float, float, float]], ...]:
    return (
        ("Li", (0.0, 0.0, 0.0)),
        ("H", (0.0, 0.0, bond_distance)),
    )


def build_h2(bond_distance: float = H2_DEFAULT_BOND_DISTANCE, basis: str = "sto-3g") -> Molecule:
    return Molecule(
        name="h2",
        geometry=h2_geometry(bond_distance),
        basis=basis,
        charge=0,
        multiplicity=1,
    )


def build_lih(
    bond_distance: float = LIH_DEFAULT_BOND_DISTANCE,
    basis: str = "sto-3g",
    n_core_orbitals: int | None = 1,
    n_active_orbitals: int | None = 5,
) -> Molecule:
    """LiH molecule with default (2 active electrons / 5 active orbitals) active space.

    The defaults freeze the lithium 1s orbital and keep 5 active spatial
    orbitals (10 qubits with Jordan-Wigner mapping). Pass ``n_core_orbitals=None``
    and ``n_active_orbitals=None`` for the full 12-qubit problem.
    """
    return Molecule(
        name="lih",
        geometry=lih_geometry(bond_distance),
        basis=basis,
        charge=0,
        multiplicity=1,
        n_core_orbitals=n_core_orbitals,
        n_active_orbitals=n_active_orbitals,
    )


_ATOMIC_NUMBERS: dict[str, int] = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
}


def _total_electrons(molecule: Molecule) -> int:
    z = sum(_ATOMIC_NUMBERS[atom] for atom, _ in molecule.geometry)
    return z - molecule.charge


_OPENFERMION_PATCH_FLAG = "_cudaq_bp_data_dir_patched"


def ensure_openfermion_data_dir_writable() -> str | None:
    """Redirect openfermion's MolecularData HDF5 cache to a writable directory.

    ``cudaq.chemistry.create_molecular_hamiltonian`` -> ``ofpyscf.run_pyscf``
    -> ``of.MolecularData.save()`` writes an HDF5 cache file to the directory
    given by ``openfermion.chem.molecular_data.DATA_DIRECTORY``, which defaults
    to a path inside the openfermion package itself. In our container the
    package is installed under root-owned ``/opt/venv/...`` while the runtime
    process runs as a non-root user, so ``save()`` raises ``PermissionError``.

    Returns the directory openfermion will use, or ``None`` if openfermion is
    not importable (which is fine: cudaq's own import will fail first and
    surface a clearer error).

    Idempotency is tracked via a sentinel attribute on the openfermion module
    itself, so multiple callers (CLI, API, benchmark harness) can safely call
    this without re-patching.
    """
    try:
        import openfermion.chem.molecular_data as _mod
    except ImportError:
        return None

    if getattr(_mod, _OPENFERMION_PATCH_FLAG, False):
        return str(_mod.DATA_DIRECTORY)

    current = getattr(_mod, "DATA_DIRECTORY", None)
    if current and os.access(current, os.W_OK):
        setattr(_mod, _OPENFERMION_PATCH_FLAG, True)
        return str(current)

    scratch = os.path.join(tempfile.gettempdir(), "cudaq-bp-of-data")
    os.makedirs(scratch, exist_ok=True)
    _mod.DATA_DIRECTORY = scratch
    setattr(_mod, _OPENFERMION_PATCH_FLAG, True)
    return scratch


def create_hamiltonian(molecule: Molecule) -> HamiltonianBundle:
    """Build a CUDA-Q molecular Hamiltonian from a ``Molecule``.

    ``cudaq.chemistry.create_molecular_hamiltonian`` describes an active space
    via ``n_active_electrons`` and ``n_active_orbitals``. We translate our
    ``n_core_orbitals`` / ``n_active_orbitals`` ergonomic representation into
    that pair: each frozen (closed-shell) core orbital removes 2 electrons
    from the active space.

    We also temporarily chdir to a writable directory while invoking
    ``create_molecular_hamiltonian`` because openfermion's
    ``MolecularData.save()`` writes a temporary ``<uuid>.hdf5`` to the current
    working directory before atomically renaming it to ``DATA_DIRECTORY``. In
    our container the WORKDIR is ``/app`` and the runtime user is ``cudaq``
    (uid 10001), so the temp write fails with ``PermissionError`` even though
    ``DATA_DIRECTORY`` itself is writable.
    """
    data_dir = ensure_openfermion_data_dir_writable()

    import cudaq

    geometry_list: list[tuple[str, tuple[float, float, float]]] = list(molecule.geometry)

    kwargs: dict[str, Any] = {}
    active_electrons: int | None = None
    active_orbitals: int | None = None
    ncore = molecule.n_core_orbitals
    nactive = molecule.n_active_orbitals
    if ncore is not None and nactive is not None:
        active_electrons = _total_electrons(molecule) - 2 * ncore
        active_orbitals = nactive
        kwargs["n_active_electrons"] = active_electrons
        kwargs["n_active_orbitals"] = active_orbitals

    prev_cwd = os.getcwd()
    target_cwd = data_dir if data_dir and os.access(data_dir, os.W_OK) else None
    if target_cwd is None and not os.access(prev_cwd, os.W_OK):
        target_cwd = tempfile.gettempdir()
    if target_cwd is not None:
        os.chdir(target_cwd)
    try:
        hamiltonian, data = cudaq.chemistry.create_molecular_hamiltonian(
            geometry_list,
            molecule.basis,
            molecule.multiplicity,
            molecule.charge,
            **kwargs,
        )
    finally:
        if target_cwd is not None:
            os.chdir(prev_cwd)

    n_electrons = int(data.n_electrons)
    n_orbitals = int(data.n_orbitals)
    n_qubits = 2 * n_orbitals

    try:
        n_terms = int(hamiltonian.term_count)
    except AttributeError:
        try:
            n_terms = len(list(hamiltonian))
        except (TypeError, AttributeError):
            n_terms = 0

    return HamiltonianBundle(
        hamiltonian=hamiltonian,
        n_electrons=n_electrons,
        n_orbitals=n_orbitals,
        n_qubits=n_qubits,
        n_terms=n_terms,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
    )
