# Why GPUs Matter to Quantum Before QPUs Do

> Using CUDA-Q, cuQuantum, and NVIDIA Blackwell GPUs for molecular
> simulation, validated end-to-end on Akamai Cloud.

> **Status: blog post draft.** This Markdown lives in the repo at
> [`docs/blog-post-draft.md`](https://github.com/jgdynamite/cudaq-molecular-simulation-blueprint/blob/main/docs/blog-post-draft.md).
> Edit freely before publishing.
> Companion repo: <https://github.com/jgdynamite/cudaq-molecular-simulation-blueprint>
> Live UI: <https://cudaq-blueprint-demo.website-us-east-1.linodeobjects.com/>

---

## The hybrid reality of quantum development today

There's a comfortable narrative about quantum computing that goes like this:
the machines are coming, we'll port our workloads when they arrive, and
in the meantime our job is to wait for the hardware to mature. That story
is wrong in two interesting ways.

First, the workflow that practitioners actually run today is hybrid:
classical CPUs orchestrate, classical GPUs simulate, and a QPU is one
optional execution target among several. The same Python you'd write to
target a real QPU runs on a CPU statevector or a GPU statevector with a
single argument change. Most of the development cycle for a near-term
quantum algorithm happens against simulators long before any QPU time is
booked.

Second, the simulators that matter aren't toy ones. They're built on
the same vendor stack that makes ML competitive on big GPUs: NVIDIA's
[CUDA-Q](https://nvidia.github.io/cuda-quantum/) for the language and
runtime, [cuQuantum](https://developer.nvidia.com/cuquantum-sdk)'s
`cuStateVec` for the statevector kernels. CUDA-Q's `nvidia-fp64` target
runs the whole VQE loop &mdash; circuit construction, expectation
evaluation, parameter update &mdash; on the GPU with the optimizer on the
CPU. You don't write any CUDA. You write quantum kernels.

So if your reaction to the headline "we should buy a Blackwell to run
quantum" is "what, why?", this post is for you.

---

## The setup: H2 and LiH on a real Blackwell

To make this concrete we built
[`cudaq-molecular-simulation-blueprint`](https://github.com/jgdynamite/cudaq-molecular-simulation-blueprint),
a small reference implementation of the variational quantum eigensolver
(VQE) for two textbook molecules:

- **H<sub>2</sub>** at 0.74 &Aring; bond length, sto-3g basis, 4 qubits,
  3 UCCSD parameters. Maps to a small Hamiltonian where everything fits
  comfortably in CPU registers.
- **LiH** at 1.5957 &Aring; bond length, sto-3g basis, 2-electron /
  5-orbital active space, 12 qubits, 92 UCCSD parameters. Big enough that
  the statevector starts to be interesting on its own.

The driver is plain CUDA-Q + Python. Chemistry preprocessing uses
[OpenFermion](https://quantumai.google/openfermion) and
[PySCF](https://pyscf.org/) to produce a CUDA-Q `Hamiltonian`; the ansatz
is the bundled `cudaq.kernels.uccsd`; the optimizer is SciPy's COBYLA so
the comparison stays in plain classical optimizer territory. The whole
loop is a couple hundred lines of typed Python with full manifest and
trace capture for every run.

What changes between the CPU and GPU runs is the value of one string:

```python
# CPU
cudaq.set_target("qpp-cpu")

# GPU (FP64 cuStateVec)
cudaq.set_target("nvidia", option="fp64")
```

The hardware host is an Akamai Cloud `g3-gpu-rtxpro6000-blackwell-1`
instance in the Jakarta (`id-cgk`) region, provisioned and torn down
with Terraform + Ansible: NVIDIA RTX PRO 6000 Blackwell Server Edition,
driver `580.159.03` (the open kernel module branch &mdash; required for
Blackwell), CUDA 13.0, 96 GB VRAM, 16 vCPU, 172 GB system RAM. The full
deployment is one Terraform apply plus one Ansible playbook, all gated
behind an SSH key that exists only for the bench cycle. The numbers
below come from a multi-seed re-bench done on 2026-05-04 (3 RNG seeds
per backend, 15 specs total): VM lifetime was 2 h 27 min of compute
plus ~30 min of bootstrap, billed at ~$3.00/hr.

---

## Results (multi-seed, n=3 per backend)

Every backend was run with seeds 42, 43, 44 on the same Blackwell host.
Full manifests and traces live under
[`results/akamai-blackwell-multiseed/`](https://github.com/jgdynamite/cudaq-molecular-simulation-blueprint/tree/main/results/akamai-blackwell-multiseed)
in the repo. Headline aggregates:

| Molecule | Backend | n | Wall (s) mean &plusmn; stderr | Energy mean (Ha) | min &#124;err vs ref&#124; (mHa) |
|---|---|:-:|---:|---:|---:|
| H<sub>2</sub>  | `qpp-cpu`     | 3 | **16.87 &plusmn; 0.83** | -1.137270 | < 0.001 |
| H<sub>2</sub>  | `nvidia:fp32` | 3 | **12.98 &plusmn; 0.39** | -1.137265 | 0.002 |
| H<sub>2</sub>  | `nvidia:fp64` | 3 |   17.65 &plusmn; 1.08    | -1.137270 | < 0.001 |
| LiH | `qpp-cpu`     | 3 |   1809.12 &plusmn; 7.03 | -7.835907 | 12.74 |
| LiH | `nvidia:fp64` | 3 | **1086.56 &plusmn; 4.19** | -7.835907 | 12.74 |

Two stories show up in the data, and they both matter.

### Small problem: GPU is no longer faster (and that's fine)

For a 4-qubit Hamiltonian the CPU statevector backend
(`qpp-cpu`, OpenMP-parallel, 16 cores) finished H<sub>2</sub> in
16.87 &plusmn; 0.83 s. The GPU FP64 path took 17.65 &plusmn; 1.08 s
&mdash; **0.96&times;** of CPU, well within stderr. The FP32 GPU path
finished in 12.98 &plusmn; 0.39 s for a 1.30&times; speedup; less
precision overhead pays off on a 16-amplitude statevector.

The takeaway with multi-seed visibility is sharper than the v0.1.0
single-seed cut allowed: at 4 qubits, the host&hairsp;-&hairsp;device
transfer and kernel-launch overhead is *enough by itself* to make a
top-of-line Blackwell trail a 16-core CPU. CUDA-Q is doing everything
correctly &mdash; the GPU just doesn't have enough actual work to chew
on to amortize its setup tax. This is the under-celebrated reality of
GPU acceleration in any domain: **there's a problem-size threshold
below which the device is the wrong tool**, and pretending otherwise
produces cargo-cult benchmarking.

### Bigger problem: GPU wins by 1.665&times; with tight error bars

LiH on a 12-qubit, 92-parameter UCCSD ansatz tells a different story.
Both backends ran to the 1500-iteration COBYLA cap. The wall-time gap
opens up cleanly:

- CPU: 1809.12 &plusmn; 7.03 s, 1206.08 ms per function evaluation.
- GPU FP64: **1086.56 &plusmn; 4.19 s**, **724.38 ms per function
  evaluation**.

That's a **1.665&times;** wall-time speedup, with stderr around 0.4%
relative on each backend &mdash; the speedup itself is a tight
measurement, not a single observation that happens to land where you
hope. The convergence chart is identical under the optimizer's view
(same x0 per seed, deterministic FP64 math), the GPU just gets through
each iteration faster:

![LiH convergence on Blackwell](images/04-result-lih-gpu.png)

This is the regime where the GPU pays its own freight. The 12-qubit
statevector is 4096 complex amplitudes per evaluation, the Hamiltonian
has hundreds of Pauli terms, and each function evaluation is doing
real work that benefits from the parallelism. The Blackwell's 96 GB of
VRAM is barely touched here &mdash; but it's the thing that makes the
next jumps (16, 18, 20-qubit active spaces with bigger basis sets)
tractable on a single card.

### Why multi-seed actually changed our mind

The v0.1.0 single-seed cut quoted a 1.71&times; LiH speedup on a 300-
iteration COBYLA cap. That number turns out to be honest *and*
incomplete. Looking at the trace inspection on the v0.1.0 manifest,
COBYLA was still descending steadily at iter 300 (the last quarter of
that run dropped 0.048 Ha by itself), so the +0.283 Ha residual error
versus FCI was *not enough iterations*, not *stuck in a local
minimum*. We bumped LiH `--max-iterations` from 300 to 1500 and re-ran
with three seeds:

- 2 of 3 seeds (42, 43) converged to within 1.1 mHa of each other,
  landing at -7.875 to -7.876 Ha. UCCSD with single and double
  excitations spans the full active CI space for a 2-electron system,
  so this *is* the active-space FCI minimum.
- 1 of 3 seeds (44) hit a different local minimum at -7.756 Ha, about
  120 mHa above the converged answer.

That third seed is the *real* signal multi-seed buys you. UCCSD with
COBYLA is sensitive to initialization, and 1-of-3 hits a basin you
don't want. A blog post that quoted only seed=42 would have been
misleading by omission. The variance is real, the median is honest,
and now the 1.665&times; speedup carries credible error bars.

The remaining ~6&ndash;7 mHa gap between our converged seeds and the
*full* FCI energy of -7.882362 Ha is the active-space frozen-core
error, not optimizer error. Closing it requires a bigger active space
or a richer basis, not a better optimizer.

---

## What this teaches

A few takeaways that generalize beyond this specific workload.

**1. Quantum development is hybrid CPU + GPU + (sometimes) QPU, not
QPU-or-bust.** If your platform story for quantum tomorrow involves
ignoring the GPU layer today, you're going to have to re-platform
twice: once to add GPU acceleration, once again to add a QPU
execution target. Building the hybrid runtime first is the cheaper
path.

**2. There's a problem-size threshold for GPU advantage.** It's not
where the marketing decks suggest. Below it, your GPU is a tax. Above
it, the GPU pays for itself in wall time and unlocks problem sizes a
CPU can't touch at all. Knowing where the threshold is &mdash; for your
specific Hamiltonian shape, your specific ansatz, your specific
optimizer &mdash; is the engineering work. We've published two data
points so others can extend the curve.

**3. Reproducibility is a feature, not a footnote.** Every run in this
project captures CUDA-Q version, GPU model, driver version, CUDA
version, OS, container digest, git SHA, and seed. The bench tarball
is attached to the GitHub release with a SHA256. Anyone can re-run
this on their own Blackwell, on their own CPU, on their own laptop,
on their own cloud. That's the only way "X is N times faster than
Y" claims become useful.

**4. A blog companion can be a real artifact.** The
[live UI](https://cudaq-blueprint-demo.website-us-east-1.linodeobjects.com/)
isn't a screenshot. It's the actual Jinja2/HTMX UI rendered as a
static bundle and served from Akamai Object Storage. Visitors see the
real Blackwell host fingerprint, drill into the convergence chart of
each individual run, and read the same comparison report the CLI
spits out. The "Run an experiment" form is intentionally inert in
static mode &mdash; clicking submit redirects you to the GitHub repo,
because running cudaq from a public web bucket without auth is
exactly the kind of thing that makes security reviewers cry.

---

## Reproduce it yourself

CPU path, runs anywhere with Docker (~17 seconds for H<sub>2</sub>):

```bash
git clone https://github.com/jgdynamite/cudaq-molecular-simulation-blueprint.git
cd cudaq-molecular-simulation-blueprint
make container-build
make container-run-cpu
make serve   # local UI at http://localhost:8000
```

GPU path on Akamai Cloud (Blackwell SKU is feature-gated; talk to your
account team):

```bash
export LINODE_TOKEN=...
cd infra/terraform/akamai
cp terraform.tfvars.example terraform.tfvars
$EDITOR terraform.tfvars   # SSH key, region (id-cgk or br-gru), label

terraform init && terraform apply
ansible-playbook -i inventory.ini ../../ansible/playbook.yml \
    --private-key ~/.ssh/your-key

ssh root@$(terraform output -raw public_ip) \
    "docker exec cudaq-blueprint cudaq-bp bench compare"

terraform destroy   # mandatory; cost meter stops
```

Total cost for the multi-seed bench cycle that produced the numbers
above: ~$10.75 for ~3.5 hours of `g3-gpu-rtxpro6000-blackwell-1` time
(15 specs &times; 3 seeds, including ~30 minutes of bootstrap and
teardown overhead). The original v0.1.0 single-seed cut was $3.84 for
1 h 17 min.

The full Terraform module, Ansible roles, and Dockerfile are in the
repo under `infra/`. The application code is provider-agnostic &mdash; you
can lift the Docker image to any GPU host that has the NVIDIA Container
Toolkit and the open kernel modules installed and the same binary will
run.

---

## What's next

Three things we'd do before scaling these numbers up in a follow-up
post:

- **Optimizer + ansatz upgrade.** The 1-of-3 LiH local minimum is
  fixable by replacing COBYLA with L-BFGS-B and parameter-shift
  gradients. The active-space frozen-core gap (6&ndash;7 mHa to full
  FCI) is fixable by enlarging the active space. Both swaps are recipe
  changes; the rest of the pipeline doesn't move.
- **Bigger active spaces.** The same 96 GB Blackwell that yawned
  through 12 qubits will hold 28&ndash;30 qubit statevectors comfortably.
  That's the region where exact diagonalization on a CPU stops being
  practical, so it's the natural next data point.
- **Multi-GPU.** Akamai's `g3-gpu-rtxpro6000-blackwell-2` SKU has two
  cards; CUDA-Q's `nvidia-mgpu` target slices the statevector across
  them. We held the v0.1.0/v0.1.1 comparison to a single card
  intentionally; the next data point goes the other way.

What's *not* on the list: a quantum-advantage claim, a cross-cloud
benchmark, or a Kubernetes story. Those are different posts, with
different threat models, on different timelines.

---

## Caveats and what this isn't

- **This is not a quantum-advantage claim.** The CPU runs the same VQE
  with the same convergence; the GPU just gets through each iteration
  faster on the larger of the two molecules.
- **This is not a positioning of Akamai as a dedicated quantum cloud.**
  The point is that Akamai already has Blackwell capacity, and the
  hybrid quantum workflow runs on Blackwell, so that hybrid quantum
  workflow runs on Akamai today &mdash; same as it would on any cloud
  with Blackwell capacity.
- **The application core is provider-agnostic.** Akamai-specific
  deployment lives entirely under `infra/`. Lift the Docker image
  anywhere that has Blackwell + the NVIDIA Container Toolkit.
- **The QPU column is empty by design.** Adding a QPU target is one
  CUDA-Q `set_target()` call away; we didn't ship it because it would
  shift the discussion off-topic. The point of this post is that the
  hybrid workflow is already worth standing up before the QPU column
  has a number in it.

---

## Links

- Repository: <https://github.com/jgdynamite/cudaq-molecular-simulation-blueprint>
- Documentation site: <https://jgdynamite.github.io/cudaq-molecular-simulation-blueprint/>
- Live UI snapshot: <https://cudaq-blueprint-demo.website-us-east-1.linodeobjects.com/>
- v0.1.0 release (with bench tarball + checksum):
  <https://github.com/jgdynamite/cudaq-molecular-simulation-blueprint/releases/tag/v0.1.0>
- Container image: `ghcr.io/jgdynamite/cudaq-molecular-simulation-blueprint:v0.1.0`
- CUDA-Q docs: <https://nvidia.github.io/cuda-quantum/>
- cuQuantum SDK: <https://developer.nvidia.com/cuquantum-sdk>
- Akamai Cloud GPU SKUs: <https://www.linode.com/products/gpu/>
