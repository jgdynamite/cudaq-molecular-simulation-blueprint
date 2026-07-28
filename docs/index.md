# cudaq-molecular-simulation-blueprint

> Hybrid quantum-classical molecular simulation reference using NVIDIA
> CUDA-Q and cuQuantum, validated end-to-end on Akamai Cloud NVIDIA RTX
> PRO 6000 Blackwell GPUs.

This site is the long-form companion to the
[GitHub repository](https://github.com/jgdynamite/cudaq-molecular-simulation-blueprint).
It is the reference implementation behind the article **"Quantum
Computing: What Every CIO Should Understand Before Investing"**, making
the hybrid quantum workflow concrete, runnable, and reproducible.

## Validated on Blackwell (Jakarta, multi-seed re-bench 2026-05-04)

A `g3-gpu-rtxpro6000-blackwell-1` VM in Akamai's `id-cgk` region. NVIDIA
driver `nvidia-open-580.159.03` (`nvidia-smi` reports max-supported
CUDA 13.0); container uses `cuda-quantum-cu13` wheels on top of the
CUDA 12.6 base image. 96 GB VRAM, 16 vCPU, 172 GB system RAM. 15 specs
total (3 RNG seeds per backend), 2 h 27 min of bench compute, ~3 h 35
min total VM lifetime, ~$10.75 at the Jakarta regional rate of $3.00/hr
in effect at the time. That surcharge rose before the July follow-up,
which is why the later run moved to `us-east`.

| Molecule | Backend | n | Wall (s) mean ± stderr | Energy mean (Ha) | min &#124;err&#124; (mHa) | chem. acc. |
|---|---|:-:|---:|---:|---:|:-:|
| H2  | `qpp-cpu`     | 3 | **16.87 ± 0.83** | -1.137270 | < 0.001 | 3 / 3 |
| H2  | `nvidia:fp32` | 3 | **12.98 ± 0.39** | -1.137267 | 0.002   | 3 / 3 |
| H2  | `nvidia:fp64` | 3 |   17.65 ± 1.08    | -1.137270 | < 0.001 | 3 / 3 |
| LiH | `qpp-cpu`     | 3 |   1809.12 ± 7.03 | -7.835907 |   5.84  | 0 / 3 |
| LiH | `nvidia:fp64` | 3 | **1086.56 ± 4.19** | -7.835907 |   5.84  | 0 / 3 |

H2 errors are vs FCI; LiH errors are vs PySCF-recomputed CASCI(2e,5o).
H2 / FP64 GPU is ~4% slower than CPU on a 4-qubit problem (well within
stderr) - host&hairsp;-&hairsp;device transfer dominates. H2 / FP32
gets a 1.30x speedup. **LiH / FP64 GPU is 1.665x faster than CPU**, with
~0.4% relative stderr on each backend, on identical COBYLA
trajectories. No LiH run reaches chemical accuracy; the converged seeds
sit ~6 mHa above CASCI(2e,5o) and one of three lands ~126 mHa above in a
separate basin. That seed-variance signal is the part single-seed
benchmarks hide. See [Results interpretation](results-interpretation.md)
for the full discussion.

## Documentation

- [Project charter](project-charter.md) - what this project is and isn't
- [Architecture](architecture.md) - module boundaries and the CUDA-Q backend abstraction
- [Experiment methodology](experiment-methodology.md) - VQE setup, basis, ansatz, optimizer
- [Akamai deployment](akamai-deployment.md) - Terraform + Ansible walkthrough
- [Results interpretation](results-interpretation.md) - what the bench numbers do and don't say
- [Scope and non-goals](scope-and-non-goals.md) - explicit guardrails for v1

## Quick start

```bash
git clone https://github.com/jgdynamite/cudaq-molecular-simulation-blueprint.git
cd cudaq-molecular-simulation-blueprint
make container-build
make container-run-cpu       # H2 VQE on qpp-cpu (~15-20 s)
make serve                   # demo UI at http://localhost:8000
```

CUDA-Q's Python wheel ships only Linux x86_64/ARM64 binaries. On macOS or
any other platform without those wheels, use the Docker path above; the
container handles the Linux runtime transparently.

## Reproducibility

Every run produces a JSON manifest capturing CUDA-Q version, target string,
GPU model, GPU UUID, driver version, OS, RNG seed, the realized initial
parameter vector, optimizer settings, basis set, geometry, and active
space. CI reproduces the H2 CPU result on every push.

Manifests do **not** capture a container image digest, a git SHA (the field
exists but is `null`, because the source tree reaches the host with `.git`
excluded), a hostname, or SciPy/NumPy versions. The release workflow
publishes a container image to
`ghcr.io/jgdynamite/cudaq-molecular-simulation-blueprint:<tag>` that
reproduces the *method*; it is not the exact binary that produced the
published 2026-07-27 numbers, which was a local untagged build on a
since-destroyed VM. See the [README](https://github.com/jgdynamite/cudaq-molecular-simulation-blueprint#reproducibility)
for the full provenance statement.
