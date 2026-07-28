# Results interpretation

## What the charts show

The `/compare` page (and the JSON written by `cudaq-bp bench compare`) renders
two views per molecule:

1. **Wall time per run** - mean ± standard error across seeds, per backend.
2. **Absolute error vs reference** - on a log scale, one bar per backend.

Both views are derived from the per-run manifests, which already contain
the fields. There is no derived "secret sauce" - the JSON is the chart.

## What the numbers mean

| field                          | meaning                                                              |
|--------------------------------|----------------------------------------------------------------------|
| `wall_time_seconds`            | total time the optimizer ran on that backend, end to end             |
| `time_per_evaluation_ms`       | wall_time / function_evaluations - per-statevector throughput proxy  |
| `iterations`                   | outer COBYLA iterations until termination                            |
| `error_vs_reference_hartree`   | `final_energy - reference_energy` (Hartree)                          |
| `chemical_accuracy_reached`    | `\|error\| < 1.6 mHa`                                                 |

`speedups` in the comparison report quote the ratio of CPU mean wall time
to each non-CPU backend's mean wall time, e.g. `cpu_over_gpu_fp64_wall_time`.

## What the numbers do NOT mean

This project is built around very specific framing. The charts are evidence
of a hybrid quantum-classical workflow being practical *today on GPU
infrastructure*, not evidence of any of the following:

- **Quantum advantage.** None of these workloads are at quantum-advantage
  scale. STO-3G H2 has a 4×4 Hamiltonian; the FCI energy can be computed
  in microseconds in literally any way. We are using H2 as a *correctness*
  smoke test, not as a benchmark target.
- **GPU vs QPU.** This project does not run on a QPU. The "QPU" in the
  blog title is a future-state stand-in; today, we simulate quantum circuits
  classically (which is what the entire industry does for practical work).
- **NVIDIA vs other vendors.** No comparison against other GPU vendors,
  cloud providers, or CPU architectures is published here. The CPU path
  uses CUDA-Q's own `qpp-cpu` so the only thing changing between backends is
  the simulator target string.
- **Akamai vs other clouds.** This project does not run on other clouds for
  v1 and makes no claim about cross-cloud performance.

## Reading the bond-distance results

The bond distance matters: VQE is a function minimizer of energy *at a fixed
geometry*. Picking a geometry far from equilibrium will give a higher
energy - that's not a bug. The reference energy in the manifest is matched
against the same geometry, so the *error* column is the apples-to-apples
quantity to compare.

## Reading active-space results

For LiH the default active space is `(2 active electrons, 5 active
orbitals)` with the lithium 1s frozen as core. The Hamiltonian itself
applies the active-space restriction.

Through v0.1 the **ansatz** did not: it was instantiated against the full
LiH molecule (`n_qubits=12`, `n_electrons=4`, 92 UCCSD parameters), so
every manifest under `results/akamai-blackwell-multiseed/` records a
12-qubit / 92-parameter circuit even though the operator spans 10 qubits.

As of v0.2 the ansatz dimensioning is an explicit choice, recorded in each
manifest as `notes.ansatz_mode`:

| mode | qubits | parameters | notes |
|------|--------|------------|-------|
| `matched` (default) | 10 | 24 | circuit spans exactly the active space |
| `legacy_full` | 12 | 92 | reproduces the v0.1 configuration |

Reference energies in `app/quantum/reference_data.py` are unchanged and
apply to both modes, because they are properties of the Hamiltonian rather
than the ansatz. The 12-qubit and 10-qubit runs are therefore scored
against the same CASCI(2e,5o) number and are directly comparable on
accuracy.

Which mode a run used is always recoverable: manifests written before v0.2
have no `ansatz_mode` key at all, and the reporting in
`app/benchmark/followup.py` places those in their own group rather than
guessing. It never merges a `legacy_full` group with a `matched` group.

The controlled comparison between the two modes is specified in
[experiment-methodology.md](experiment-methodology.md#the-ansatzhamiltonian-mismatch-v01).
It was run on 2026-07-27 on an Akamai Blackwell host in `us-east`: three
arms, five seeds each, 15 LiH runs at an identical 1500-iteration budget.
Full numbers are in
[`results/lih-active-space-followup-v02/`](https://github.com/jgdynamite/cudaq-molecular-simulation-blueprint/tree/main/results/lih-active-space-followup-v02).

| arm | iterations | wall (s) | \|error\| (mHa) | chem. accuracy |
|---|---|---|---|---|
| `legacy_full` / GPU | 1500 (cap hit, 5/5) | 1068.8 ± 3.5 | mean 31.18, median 6.92, max 126.01 | 0/5 |
| `matched` / GPU | 1030 mean (converged) | 719.4 ± 5.5 | 0.0000 | 5/5 |
| `matched` / CPU | 1055 mean (converged) | 773.4 ± 30.1 | 0.0000 | 5/5 |

Three conclusions follow, and they are worth keeping separate.

**Accuracy.** The mismatch was the whole story. Every `legacy_full` run
exhausted its budget without converging and none reached chemical
accuracy, with a 53 mHa spread across seeds. Every `matched` run
converged on its own and landed on `-7.8821640299` Ha, which is
CASCI(2e,5o) to ~1e-13 Ha, on all five seeds and both backends. This is
expected rather than lucky: with 2 active electrons, singles and doubles
already span the full CI space of the active space, so UCCSD is
equivalent to FCI for this problem and the correctly dimensioned circuit
is exact.

**Wall time.** The matched arm is 1.49x faster than legacy on the same
GPU, but that came from needing fewer and slightly cheaper evaluations
(1030 vs 1500, 698.4 vs 712.6 ms each), not from the hardware.

**The accelerator.** At this scale it contributes very little. The matched
CPU/GPU wall-time ratio is 1.075 and the per-evaluation ratio is 1.049,
with GPU utilization sampling 0% throughout. A 10-qubit statevector is
1024 amplitudes, so runtime is bound by Python and CUDA-Q per-`observe()`
overhead rather than linear algebra. Do not read the 1.49x as a GPU
result. Note also that this sweep has no `legacy_full` CPU arm, so it
does not reproduce or refute the 1.665x LiH figure from the May bench.

As a correctness check, CPU and GPU FP64 agreed on all 5 matched seeds to
a maximum absolute difference of 6.6e-13 Ha, with zero violations of a
1e-8 Ha tolerance.

The reference table in `app/quantum/reference_data.py` was recomputed
via PySCF on 2026-05-04 (`pyscf.mcscf.CASCI`) so chemical accuracy is
measured against PySCF-derived values rather than a literature
estimate. For LiH/STO-3G at the equilibrium geometry, CASCI(2e,5o) and
full FCI are only 0.227 mHa apart, so the active-space approximation
is not the limiting factor: any error larger than ~1 mHa is optimizer /
ansatz residual.

The 2026-05-04 multi-seed bench reflects this. With 1500 COBYLA
iterations, two of three seeds (42, 43) converge to within 1.1 mHa of
each other but stop ~5.8&ndash;6.9 mHa above CASCI(2e,5o); the third
seed (44) lands ~126 mHa above in a separate basin. None of these
runs reach chemical accuracy.

Those runs all used the `legacy_full` ansatz, so their 92-parameter search
space is one candidate explanation among several. The others are
gradient-based optimization (parameter-shift L-BFGS-B) and simply running
longer. Ranking them requires the controlled experiment described in
[experiment-methodology.md](experiment-methodology.md#the-ansatzhamiltonian-mismatch-v01);
until it runs, the cause is unestablished. None of these are vendor
problems; they are engineering choices the project will revisit.

## Sample size and noise

Each run is deterministic for a given seed (COBYLA is seeded; CUDA-Q
statevector simulation is deterministic). The "noise" reflected in
`stderr` columns therefore comes from optimizer-initial-condition variance
across different seeds, not stochastic shot noise. We deliberately use
`shots=0` (i.e. exact statevector observable) for this study; shot-based
runs would introduce another axis of variance that's outside the v1 scope.
