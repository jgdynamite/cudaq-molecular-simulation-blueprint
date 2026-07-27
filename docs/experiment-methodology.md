# Experiment methodology

## Why H2 and LiH

These two molecules give us a credible CPU-vs-GPU story without making
unsupportable claims:

- **H2 / STO-3G** is the textbook 4-qubit / 3-parameter UCCSD problem.
  Both backends converge to chemical accuracy in seconds to tens of seconds
  in this implementation; on the Blackwell host used for the multi-seed
  bench, H2 finishes in ~13–18 s wall time per run depending on backend
  and precision. H2 is the smoke test that proves the pipeline is correct
  end-to-end.
- **LiH / STO-3G** with the default (2 electron / 5 orbital) active space.
  Through v0.1 the ansatz was instantiated against the full molecule while
  the Hamiltonian carried the active-space restriction; see
  [the ansatz/Hamiltonian mismatch](#the-ansatzhamiltonian-mismatch-v01)
  below. The CPU statevector simulator is noticeably slower than the GPU
  here: in the v0.1 configuration CPU runs took ~30 minutes per seed (1500
  iterations of COBYLA) and GPU FP64 runs took ~18 minutes. The CPU
  baseline remains feasible &mdash; you can leave it running over a coffee
  &mdash; but it is long enough to expose meaningful GPU wall-time savings
  rather than being a sub-second curiosity.

We deliberately do **not** include molecules where CPU simulation becomes
intractable (BeH2, H2O, larger). That would force us to compare GPU runs
against extrapolations rather than measured CPU baselines, which is not what
this project is for.

## Method choices

| concern             | choice                                | why                                                 |
|---------------------|----------------------------------------|----------------------------------------------------|
| basis set           | STO-3G                                 | minimal, reproducible across QC papers              |
| Hamiltonian builder | `cudaq.chemistry.create_molecular_hamiltonian` | first-class CUDA-Q API; uses OpenFermion under the hood |
| ansatz              | UCCSD via `cudaq.kernels.uccsd`        | physically motivated for chemistry                  |
| ansatz dimensioning | `--ansatz-mode matched` (default) or `legacy_full` | `matched` builds the circuit for the active space; `legacy_full` reproduces v0.1. See [the mismatch](#the-ansatzhamiltonian-mismatch-v01) |
| parameter count     | `cudaq.kernels.uccsd_num_parameters()` | queried from CUDA-Q, never hard-coded               |
| reference state     | Hartree-Fock                           | standard initial state                              |
| optimizer           | COBYLA (SciPy)                         | gradient-free, exposes a real per-eval trace        |
| convergence test    | `\|E_VQE - E_FCI\| < 1.6e-3 Ha`         | chemical accuracy threshold                         |
| RNG seed            | configurable (default 42)              | stamped into every manifest for reproducibility     |
| seeds per backend   | 3 (default in `default_blog_suite`; H2 budget 200 iter, LiH 1500 iter) | mean +/- standard error in the comparison; LiH was bumped from 300 to 1500 after the v0.1.0 trace inspection showed COBYLA was still descending steadily at iter 300 |

## What is measured

For every run we record:

- **wall_time_seconds** - end-to-end time of the optimizer.
- **iterations** - number of outer COBYLA iterations.
- **function_evaluations** - number of `cudaq.observe` calls.
- **time_per_evaluation_ms** - wall_time / function_evaluations.
- **final_energy** - the optimizer's `result.fun`.
- **error_vs_reference_hartree** - `final_energy - E_reference`, where
  the reference comes from `app/quantum/reference_data.py` (published FCI
  values for the chosen geometry).
- **chemical_accuracy_reached** - `|error| < 1.6 mHa`.

The trace also stores every evaluation's `(iteration, energy,
elapsed_seconds, parameters)` so the UI can render convergence curves and the
benchmark harness can compute time-to-convergence.

## Reference values

`app/quantum/reference_data.py` ships with reference energies keyed by
`(molecule, basis, bond_distance, active_space)`. Bond distance is matched
within 0.01 Å so minor user perturbations still find the correct
reference.

The LiH references were recomputed via PySCF on 2026-05-04 (RHF + CASCI on
the equilibrium geometry) so any reader with `pip install pyscf` can
reproduce them locally. The earlier shipped LiH (2e, 5o) value of
-7.862500 Ha was effectively the HF energy and was off by ~19.7 mHa; it
has been replaced.

| molecule | basis  | R (Å)  | active space | method                            | E (Ha)        |
|----------|--------|--------|--------------|-----------------------------------|---------------|
| H2       | STO-3G | 0.7414 | full         | FCI                               | -1.137270     |
| H2       | STO-3G | 0.7474 | full         | FCI                               | -1.137275     |
| LiH      | STO-3G | 1.5957 | full         | FCI (pyscf 2026-05-04)            | -7.882391     |
| LiH      | STO-3G | 1.5957 | (2e, 5o)     | CASCI(2e,5o) (pyscf 2026-05-04)   | -7.882164     |

The CASCI(2e,5o) and full-FCI minima for LiH/STO-3G at this geometry are
only 0.227 mHa apart, so the (2e, 5o) active space already captures
essentially all of the FCI correlation. Any error larger than ~1 mHa
against CASCI(2e,5o) in this active space is therefore optimizer / ansatz
residual rather than active-space frozen-core error.

If a user picks an unusual geometry, the reference is `None` and only the
absolute energy is reported (no error column).

## The ansatz/Hamiltonian mismatch (v0.1)

The LiH Hamiltonian is built with one frozen core orbital and five active
spatial orbitals, i.e. **2 active electrons in 5 active orbitals**, which
under Jordan-Wigner is a **10-qubit** operator.

The v0.1 VQE driver did not use those dimensions. It built the UCCSD ansatz
from the full-molecule metadata that `cudaq.chemistry.create_molecular_hamiltonian`
reports (`n_electrons=4`, `n_orbitals=6`), producing a **12-qubit,
92-parameter** circuit. Every LiH manifest under
`results/akamai-blackwell-multiseed/` records those numbers.

So the circuit and the operator described different orbital sets. The runs
still executed &mdash; the two extra qubits are simply never acted on by the
Hamiltonian &mdash; but the ansatz carried 92 variational parameters and a
Hartree-Fock reference for 4 electrons to describe a 2-electron problem.

Parameter counts are not hard-coded anywhere; they come from
`cudaq.kernels.uccsd_num_parameters(electron_count, qubit_count)`. For
CUDA-Q 0.14 that function returns 92 for `(4, 12)` and 24 for `(2, 10)`,
which is where both figures above come from.

### Hypothesis

Restricting the ansatz to the active space should change the optimization
problem COBYLA is given: the same Hamiltonian, but a 24-parameter search
space instead of a 92-parameter one, with a Hartree-Fock reference matching
the active electron count.

Whether that changes accuracy, convergence behaviour, seed-to-seed
stability, or wall time is **an open question that this repository has not
yet measured**. The experiment below exists to answer it. Nothing in the
committed results speaks to it.

### Controlled experiment design

`cudaq-bp bench run-lih-active-space-followup` runs three arms over five
seeds (42-46), 15 runs total, every run at the same 1500-iteration COBYLA
budget:

| arm | ansatz mode | backend | qubits | parameters |
|-----|-------------|---------|--------|------------|
| 1   | `legacy_full` | `gpu_fp64` | 12 | 92 |
| 2   | `matched`     | `gpu_fp64` | 10 | 24 |
| 3   | `matched`     | `cpu`      | 10 | 24 |

Arms 1 and 2 differ **only** in ansatz dimensioning, which is the variable
under test. Arm 3 is the CPU counterpart of arm 2: it supplies a CPU/GPU
wall-time ratio for the matched ansatz, and a numerical cross-check that
the two backends agree seed-for-seed (final energies within 1e-8 Ha).
Disagreements above that tolerance are reported, not discarded.

Held constant across all three arms: geometry, basis, active-space
definition, CASCI(2e,5o) reference, the 1.6 mHa chemical-accuracy
threshold, COBYLA and its tolerance, the parameter-initialization
distribution, and the bond distance.

H2 is excluded because it has no active space, so the ansatz mode cannot
change anything about it. GPU FP32 is excluded because a precision
difference would confound the ansatz difference under test.

Arms run contiguously rather than interleaved, so CPU and GPU work never
overlap on the same host.

### Running it

```bash
# Defaults: seeds 42-46, 1500 iterations, output to
# <results_dir>/lih-active-space-followup-v02/
cudaq-bp bench run-lih-active-space-followup

# Or override any of them
cudaq-bp bench run-lih-active-space-followup \
  --seeds 42,43,44,45,46 \
  --max-iterations 1500 \
  --output-dir lih-active-space-followup-v02
```

This is a long sweep &mdash; 15 LiH runs at up to 1500 iterations each
&mdash; so budget GPU time before starting. On completion it writes
`SUMMARY.csv` (one row per run) and `comparison.json` into the output
directory.

`comparison.json` groups on `(experiment_variant, ansatz_mode, backend)`
and never merges the `legacy_full` and `matched` arms into a shared
aggregate. Cross-arm relationships appear only as explicit ratios and
paired per-seed differences.

Single runs can select the mode directly:

```bash
cudaq-bp run lih --backend gpu_fp64 --ansatz-mode matched      # default
cudaq-bp run lih --backend gpu_fp64 --ansatz-mode legacy_full  # v0.1 behavior
```

`matched` is the default for new LiH runs. The published v0.1 numbers were
produced under `legacy_full`, which is retained so they stay reproducible.

## Running the canonical suite

```bash
# CPU only - works everywhere
cudaq-bp run h2 --backend cpu

# CPU + GPU on the Blackwell host (after Akamai bootstrap)
cudaq-bp run h2  --backend gpu_fp64
cudaq-bp run lih --backend gpu_fp64

# Multi-seed sweep that drives the blog charts
python -m app.benchmark.runner  # or write your own driver script
cudaq-bp bench compare           # writes comparison report to results/
```
