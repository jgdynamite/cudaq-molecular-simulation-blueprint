# LiH active-space ansatz follow-up (2026-07-27)

Controlled follow-up that tests a single hypothesis: the ~6 mHa residual
left over from the May multi-seed bench was **ansatz overparametrization**,
not an active-space or hardware limitation.

In v0.1 the LiH Hamiltonian was built for a (2 electron, 5 orbital) active
space — a 10-qubit operator — but the UCCSD ansatz was dimensioned from
full-molecule metadata, producing a 12-qubit / 92-parameter circuit. This
sweep runs both dimensionings against the *same* Hamiltonian.

The full per-run manifests and traces (~9 MiB) are not committed; only the
two aggregate reports are.

## Files

| file | description |
|---|---|
| [`SUMMARY.csv`](SUMMARY.csv) | One row per run (15 rows), 20 columns including `ansatz_mode`, `qubits`, `parameters`, `iterations`, `wall_seconds`, `energy_hartree`, `absolute_error_mhartree`, and host fingerprint. |
| [`comparison.json`](comparison.json) | Per-arm aggregates (mean ± stderr), explicit cross-arm ratios, paired per-seed differences, and the CPU/GPU numerical-equivalence check. |

## Design

Three arms, five seeds each (42–46), 15 runs. All arms use LiH/STO-3G at
R = 1.5957 Å with the same (2e, 5o) Hamiltonian and an identical COBYLA
budget of 1500 iterations, so the only variable within a backend is how
the ansatz is dimensioned.

| arm | ansatz mode | backend | circuit |
|---|---|---|---|
| 1 | `legacy_full` | `gpu_fp64` | 12 qubits, 92 parameters |
| 2 | `matched` | `gpu_fp64` | 10 qubits, 24 parameters |
| 3 | `matched` | `cpu` | 10 qubits, 24 parameters |

H2 and GPU FP32 are deliberately excluded: H2 has no active space, so the
ansatz mode cannot change anything, and FP32 would confound a precision
difference with the ansatz difference under test.

## Results

Scored against `CASCI(2e,5o) = -7.882164 Ha` (PySCF). Chemical accuracy is
the standard 1.6 mHa threshold.

| arm | iterations | wall time (s) | \|error\| (mHa) | chemical accuracy |
|---|---|---|---|---|
| 1 · `legacy_full` / GPU | 1500 (cap hit, 5/5) | 1068.8 ± 3.5 | mean 31.18, median 6.92, max 126.01 | **0/5** |
| 2 · `matched` / GPU | 1030 mean (converged) | 719.4 ± 5.5 | **0.0000** | **5/5** |
| 3 · `matched` / CPU | 1055 mean (converged) | 773.4 ± 30.1 | **0.0000** | **5/5** |

The hypothesis is confirmed. Every `legacy_full` run exhausted the 1500
iteration budget without converging and none reached chemical accuracy,
with a seed-dependent spread of 53 mHa (stdev) and one seed landing in a
basin 126 mHa above the reference. Every `matched` run terminated on its
own well inside the budget and recovered the active-space energy to
`-7.8821640299` Ha — 0.0000 mHa error — on all five seeds and both
backends.

Why the matched ansatz is *exact* here rather than merely better: with
only 2 active electrons, singles and doubles already span the entire CI
space of the active space, so UCCSD is equivalent to FCI for this
problem. Recovering CASCI(2e,5o) to ~1e-13 Ha is the expected result once
the circuit is dimensioned correctly.

## The GPU contribution is small at this scale

| comparison | ratio |
|---|---|
| `matched` CPU wall / `matched` GPU wall | **1.075** |
| `matched` GPU wall / `legacy_full` GPU wall | **0.673** |

Per-evaluation cost was 712.6 ms (legacy, 12q), 698.4 ms (matched GPU,
10q), and 732.9 ms (matched CPU, 10q). GPU utilization sampled 0% with
621 MiB allocated throughout.

At 10–12 qubits a statevector is a few thousand amplitudes, so runtime is
dominated by Python and CUDA-Q per-`observe()` overhead rather than by
linear algebra. The GPU is only ~4.7% faster per evaluation and ~7.5%
faster on wall time. **The 1.49x wall-time win of arm 2 over arm 1 came
from needing fewer, cheaper evaluations — a correctness fix — not from
the accelerator.** Note also that the CPU/GPU ratio here is measured only
for the smaller matched circuit; this sweep has no `legacy_full` CPU arm,
so it does not reproduce the 1.665x LiH figure from the May bench.

## CPU/GPU numerical equivalence

For the matched arm, CPU (`qpp-cpu`) and GPU (`nvidia-fp64`) agreed on all
5 seeds within a 1e-8 Ha tolerance, with a maximum absolute difference of
**6.6e-13 Ha** and zero violations. FP64 statevector simulation is
reproducible across the two backends at this scale.

## Run conditions

- Hardware: Akamai Cloud `g3-gpu-rtxpro6000-blackwell-1` in `us-east` (Newark, NJ)
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887 MiB
- Driver: `nvidia-open-580.173.02`
- CUDA-Q: 0.14.2
- Host: `Linux-5.15.0-185-generic-x86_64-with-glibc2.35`
- VM lifetime 3h54m at $3.00/hr = **$11.74**

`us-east` was chosen over `id-cgk` (the May region) because Blackwell
carries a $1.50/hr surcharge in Jakarta; the same sweep there would have
cost $17.61. Region does not affect cross-arm validity, because arm 1
re-runs the v0.1 configuration on the same host in the same session.

## Reproducing

```bash
uv run cudaq-bp bench run-lih-active-space-followup \
  --seeds 42,43,44,45,46 --max-iterations 1500
```

Or a single run:

```bash
uv run cudaq-bp run lih --ansatz-mode matched     --backend gpu_fp64
uv run cudaq-bp run lih --ansatz-mode legacy_full --backend gpu_fp64
```

## Known gap

`git_commit_sha` is empty in `SUMMARY.csv`. The deployment syncs source to
the VM as a tarball without the `.git` directory, so the recorder had no
commit to read. The code state corresponds to `main` at the time of the
sweep; the column is populated when running from a git checkout.
