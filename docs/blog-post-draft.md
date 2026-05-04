# Why GPUs Matter to Quantum Before QPUs Do

> Using CUDA-Q, cuQuantum, and NVIDIA Blackwell GPUs for molecular
> simulation, validated end-to-end on Akamai Cloud.

> **Status: blog post draft.** This Markdown lives in the repo at
> [`docs/blog-post-draft.md`](https://github.com/jgdynamite/cudaq-molecular-simulation-blueprint/blob/main/docs/blog-post-draft.md).
> Edit freely before publishing.
> Companion repo: <https://github.com/jgdynamite/cudaq-molecular-simulation-blueprint>
> Live UI: <https://cudaq-blueprint-demo.website-us-east-1.linodeobjects.com/>

---

## Why this work, and why now

For most enterprise leaders, quantum computing still feels like a
future conversation. The infrastructure pattern underneath it,
however, is already here. Practical quantum development today is
hybrid: CPUs orchestrate, GPUs simulate, optimizers iterate, and a QPU
remains an optional future execution target. That is why this project
matters. It is not a claim that quantum advantage has arrived. It is a
small, reproducible example of how to evaluate the accelerated
workloads that are forming around the quantum software stack right
now.

The companion repository,
[`cudaq-molecular-simulation-blueprint`](https://github.com/jgdynamite/cudaq-molecular-simulation-blueprint),
is a reference implementation of the variational quantum eigensolver
(VQE) for two textbook molecules, built on NVIDIA's
[CUDA-Q](https://nvidia.github.io/cuda-quantum/) and
[cuQuantum](https://developer.nvidia.com/cuquantum-sdk)'s `cuStateVec`,
validated end-to-end on Akamai Cloud RTX PRO 6000 Blackwell GPUs. The
post that follows is a field note from running it: where current
GPUs help the hybrid quantum workflow, where they do not, and how to
evaluate the question carefully for your own organization without
leaning on vendor narrative. CUDA-Q's `nvidia-fp64` target runs the
whole VQE loop &mdash; circuit construction, expectation evaluation,
parameter update &mdash; on the GPU with the optimizer on the CPU; you
do not write any CUDA, you write quantum kernels in CUDA-Q's Python
frontend and pick a target.

The practical value of this kind of work starts with experimentation,
validation, and shortening time-to-decision, long before any QPU hours
are booked. The numbers below come from a multi-seed re-bench on a
single Blackwell GPU. They are honest about both the cases where the
GPU pays for itself and the cases where it does not.

---

## Why this matters to infrastructure leaders

Quantum computing usually enters executive conversations as a
future-tense topic. The near-term reality is more useful and more
mundane: the workflow is already running, on classical infrastructure,
mostly on GPUs that organizations are already provisioning for AI
training and inference.

Five points for someone evaluating where this fits in an infrastructure
portfolio:

- **This is a repeatable pattern for evaluating emerging accelerated
  workloads.** The structure &mdash; small typed Python core, container,
  Terraform-provisioned GPU host, multi-seed sweep, manifest-per-run,
  static UI snapshot &mdash; generalizes well past quantum. The same
  template can be pointed at any new accelerated stack you need to
  evaluate before committing portfolio dollars.
- **The value is not only speedup; it is reproducibility, auditability,
  and knowing where the GPU starts to pay for itself.** A 1.665&times;
  wall-time win on a 12-qubit problem is real, but the more useful
  artifact is the manifest: CUDA-Q version, GPU model, driver version,
  container digest, git SHA, RNG seed, optimizer settings, and full
  convergence trace, captured per run.
- **Akamai Cloud can host a Blackwell-class scientific workload with
  standard IaC and containers.** One Terraform apply, one Ansible
  playbook, a Docker container, and a systemd unit. No bespoke
  orchestration, no quantum-specific tooling, no custom kernels.
- **The application core remains provider-agnostic.** Akamai-specific
  deployment lives entirely under `infra/`. The same Docker image runs
  on any cloud or on-prem host that has a Blackwell-class GPU and the
  NVIDIA Container Toolkit installed.
- **A technical infrastructure thesis, not a platform claim.** The
  takeaway is that Blackwell-class GPUs are a viable platform for the
  hybrid quantum workloads that exist today, and Akamai Cloud is one
  validated path to running them. That is a technical thesis with
  reproducible evidence behind it. It is not a quantum-advantage claim,
  and it is not a positioning of Akamai as a dedicated quantum cloud.

The rest of this post walks through the experiment that supports those
takeaways and the places where the data complicated my own thinking.

---

## The setup: H2 and LiH on a real Blackwell

The blueprint targets two textbook molecules, picked to bracket the
regime where a small statevector simulation transitions from
CPU-dominated to GPU-dominated work:

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
96 GB VRAM, 16 vCPU, 172 GB system RAM. The host runs the open-source
NVIDIA driver branch (`nvidia-open-580.159.03`); `nvidia-smi` reports
max-supported CUDA 13.0 there. The container provides its own CUDA
runtime &mdash; CUDA-Q's `cu13` Python wheels on top of an
`nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04` base image &mdash; so
the host playbook installs the kernel modules and the NVIDIA Container
Toolkit, and the application's CUDA dependencies live entirely inside
the image. The full deployment is one Terraform apply plus one Ansible
playbook, all gated behind an SSH key that exists only for the bench
cycle.

The numbers below come from a multi-seed re-bench done on 2026-05-03/04
(3 RNG seeds per backend, 15 specs total). Bench compute time alone was
**2 h 27 min**. End-to-end VM lifetime &mdash; provisioning, NVIDIA
driver install plus reboot, container build, the bench itself, results
export, one mid-cycle reboot to clear an SSH `MaxStartups` state, and
teardown &mdash; was **~3 h 35 min**. Akamai's Jakarta and S&atilde;o
Paulo regions carry a $0.50/hr regional uplift over the base
`g3-gpu-rtxpro6000-blackwell-1` rate, putting the effective rate at
**$3.00/hr**, and the end-to-end VM cost was therefore **~$10.75**
&mdash; not 2 h 27 min &times; $2.50/hr. Treating cost-of-experiment as
"compute &times; base SKU rate" is the easiest way to under-estimate a
bench cycle by 2&times; or more.

---

## Results (multi-seed, n=3 per backend)

Every backend was run with seeds 42, 43, 44 on the same Blackwell host.
A 15-row summary CSV and aggregate `comparison.json` live under
[`results/akamai-blackwell-multiseed/`](https://github.com/jgdynamite/cudaq-molecular-simulation-blueprint/tree/main/results/akamai-blackwell-multiseed)
in the repo. Headline aggregates (errors quoted against PySCF
references recomputed on 2026-05-04):

| Molecule | Backend | n | Wall (s) mean &plusmn; stderr | Energy mean (Ha) | min &#124;err&#124; (mHa) | chem. acc. |
|---|---|:-:|---:|---:|---:|:-:|
| H<sub>2</sub>  | `qpp-cpu`     | 3 | **16.87 &plusmn; 0.83** | -1.137270 | < 0.001 | 3 / 3 |
| H<sub>2</sub>  | `nvidia:fp32` | 3 | **12.98 &plusmn; 0.39** | -1.137267 | 0.002   | 3 / 3 |
| H<sub>2</sub>  | `nvidia:fp64` | 3 |   17.65 &plusmn; 1.08    | -1.137270 | < 0.001 | 3 / 3 |
| LiH | `qpp-cpu`     | 3 |   1809.12 &plusmn; 7.03 | -7.835907 |   5.84  | 0 / 3 |
| LiH | `nvidia:fp64` | 3 | **1086.56 &plusmn; 4.19** | -7.835907 |   5.84  | 0 / 3 |

H<sub>2</sub> errors are vs FCI; LiH errors are vs CASCI(2e,5o), which
is essentially identical to full FCI for this geometry (the gap is
0.227 mHa). The "chem. acc." column counts how many of the three seeds
land within the 1.6 mHa chemical-accuracy threshold &mdash; all three
H<sub>2</sub> seeds do; none of the LiH seeds do, even with 1500
optimizer iterations.

Two stories show up in the data, and they both matter.

### Small problem: the GPU is not always the right tool

For a 4-qubit Hamiltonian the CPU statevector backend
(`qpp-cpu`, OpenMP-parallel, 16 cores) finished H<sub>2</sub> in
16.87 &plusmn; 0.83 s. The GPU FP64 path took 17.65 &plusmn; 1.08 s
&mdash; **0.96&times;** of CPU, well within stderr. The FP32 GPU path
finished in 12.98 &plusmn; 0.39 s for a 1.30&times; speedup; the
lighter precision overhead pays off on a 16-amplitude statevector.

Multi-seed visibility makes a useful observation precise: at 4 qubits,
host&hairsp;-&hairsp;device transfer and kernel-launch overhead is
enough by itself for a top-of-line Blackwell to trail a 16-core CPU.
CUDA-Q is doing the right thing; the GPU simply does not have enough
actual work on a 16-amplitude statevector to amortize its setup cost.
The general form of this finding is what matters: **GPU acceleration
has a problem-size threshold below which the device is overhead, not
acceleration.** Choosing the right tool for the size of the problem is
part of the engineering work, and reporting both regimes honestly is
part of evaluating the workload responsibly.

### Larger problem: GPU is 1.665&times; faster, with tight error bars

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
VRAM is barely touched here, which is why this benchmark should be read
as a disciplined baseline rather than a scale-limit test. The value of
this run is not that it exhausts the GPU; the value is that it shows
where the GPU begins to separate from CPU on the same hybrid
quantum-classical workflow.

### Why multi-seed mattered

A single-seed bench would have produced a tidy story for LiH: pick
seed 42, get a 1.665&times; speedup with the energy converged to
-7.875 Ha, mention the residual against the active-space reference,
ship the post. Multi-seed turns that tidy story into a more honest
one.

With three seeds and 1500 COBYLA iterations:

- 2 of 3 seeds (42, 43) converge to within 1.1 mHa of each other,
  landing at -7.875 to -7.876 Ha. The wall-time speedup between CPU
  and GPU is identical for both seeds (the optimizer takes the same
  trajectory; only execution time changes), and stderr on each
  backend's wall time is around 0.4% relative.
- 1 of 3 seeds (44) lands in a different basin at -7.756 Ha, about
  126 mHa above the converged sibling seeds. The CPU and GPU runs of
  seed 44 land at the same energy as each other, just on a different
  initial trajectory.

That third seed is the part of the picture single-seed benchmarks
hide. UCCSD with COBYLA is sensitive to initialization, and 1-in-3
lands somewhere you do not want. A version of this post that quoted
only seed 42 would have given a tidier picture than the workload
actually has. The variance is real, the median is honest, and the
1.665&times; speedup now carries credible error bars rather than a
single observation that happened to land where I hoped.

There is a related credibility story underneath the LiH numbers. The
converged seeds 42 and 43 sit ~5.8&ndash;6.9 mHa above the
PySCF-computed CASCI(2e, 5o) reference of -7.882164 Ha &mdash;
**none** of the three seeds reaches the 1.6 mHa chemical-accuracy
threshold. That gap is **not** an active-space frozen-core limitation:
PySCF reports the (2e, 5o) and full-FCI minima only 0.227 mHa apart
for this geometry, so the active space already captures essentially
all of the FCI correlation. The gap is optimizer / over-parametrization
residual: the LiH ansatz currently instantiates 92 UCCSD parameters on
a 12-qubit kernel even though the active-space Hamiltonian has support
on only 5 active orbitals, which leaves COBYLA optimizing in a much
larger parameter space than the problem needs. The remaining gap should
be read as part of the benchmark context, not as a failure of the GPU
result. The CPU and GPU runs agree on the optimization behavior; the
GPU simply gets through the same work faster. That is the
infrastructure point this post is making.

---

## An important caveat: the LiH ansatz is bigger than the active-space reference

One implementation detail matters for how to read the LiH result. The
benchmark is scored against a CASCI(2e, 5o) reference &mdash; the exact
energy for 2 active electrons in 5 active orbitals &mdash; but the
current CUDA-Q UCCSD path instantiates a 12-qubit, 92-parameter ansatz
(the full LiH molecule in STO-3G). That makes the workload useful for
stressing CPU-vs-GPU statevector execution, but it also means the
chemistry-convergence story is not yet as clean as it should be: the
optimizer is searching in a 92-dimensional parameter space for a minimum
that lives on a 5-orbital support, and COBYLA does not consistently find
it within 1500 iterations.

For this post, that caveat is enough. The LiH result should be read as
an infrastructure benchmark with a clearly disclosed
chemistry-convergence limitation, not as a final chemistry-accuracy
claim.

---

## What the numbers actually say

Three findings, all of them generalizable beyond this specific
workload:

1. **Tiny workloads can make GPU acceleration look bad because overhead
   dominates.** On the 4-qubit H<sub>2</sub> Hamiltonian (a 16-amplitude
   statevector), an FP64 Blackwell card finishes ~4% slower than a
   16-core CPU running the same kernel, well within stderr. CUDA-Q is
   doing the right thing; the GPU simply has not been given enough
   work to amortize host&hairsp;-&hairsp;device transfer and
   kernel-launch overhead. This is the part most marketing material
   skips.
2. **Larger statevector workloads begin to show the GPU advantage.**
   On the 12-qubit LiH UCCSD ansatz (~4096-amplitude statevector,
   ~hundreds of Pauli terms per evaluation), the FP64 GPU finishes
   ~1.665&times; faster than the CPU on the same 1500-iteration
   COBYLA budget. The wall-time stderr on each backend is under 0.4%
   relative, so the speedup is a tight measurement, not a single
   observation.
3. **Optimizer initialization matters; seed variance is not noise, it
   is part of the engineering reality.** With three seeds, two
   converge to nearly identical energies and one lands ~126 mHa above
   in a separate basin. A single-seed bench would have hidden this.
   The cost of being honest about it is one extra row in the results
   table; the value is a credible error bar on every claim that
   follows.

These three findings are not pro-GPU or anti-GPU. They are
descriptive: a responsible bench tells you *which* workloads benefit
from a Blackwell-class accelerator, by *how much*, with *what
variance*, against *what reference*. That is the artifact an
infrastructure team can act on.

---

## What this run teaches

A few takeaways that generalize beyond this specific workload.

**1. Quantum development today is hybrid by default.** The pipeline
that will eventually target a real QPU is the same pipeline that runs
on a CPU statevector or a GPU statevector now, with one configuration
value choosing the execution target. Standing up the hybrid runtime
first is the more economical path to readiness; it lets the QPU
column be added when QPUs reach an interesting working point, rather
than as a re-platforming exercise.

**2. Problem size, not vendor narrative, decides whether the GPU
helps.** Below the threshold the GPU is overhead; above it, the GPU
pays for itself in wall time and unlocks problem sizes a CPU cannot
practically reach. Knowing where the threshold is &mdash; for your
specific Hamiltonian shape, your specific ansatz, your specific
optimizer &mdash; is the engineering work. The two data points are enough for the narrow claim this post makes:
tiny workloads may not benefit from GPU acceleration, while the larger
LiH workload shows a measurable and repeatable GPU wall-time advantage
on the same host.

**3. Reproducibility belongs in the design, not the appendix.** Every
run in this project captures CUDA-Q version, GPU model, driver
version, CUDA version, OS, container digest, git SHA, RNG seed, and
optimizer settings. A 15-row summary CSV plus an aggregate JSON live
in the repository under `results/akamai-blackwell-multiseed/`, with
references recomputed from PySCF so a reader with `pip install pyscf`
can reproduce the reference values locally. Anyone can re-run the
bench on their own Blackwell, on their own CPU, on their own laptop,
on their own cloud. That kind of provenance is what turns an "X is N
times faster than Y" anecdote into something an infrastructure team
can act on.

**4. The artifact and the post belong together.** The
[live UI](https://cudaq-blueprint-demo.website-us-east-1.linodeobjects.com/)
is not a screenshot. It is the actual Jinja2/HTMX UI rendered as a
static bundle and served from Akamai Object Storage. Visitors see the
real Blackwell host fingerprint, drill into the convergence chart of
each individual run, and read the same comparison report the CLI
produces. The "Run an experiment" form is intentionally inert in
static mode &mdash; clicking submit redirects to the GitHub repo.
Public read-only artifacts are easy to reason about; live execution
belongs behind authentication, which is the appropriate place to
draw that line for a public companion site.

---

## Reproduce it yourself

This project is meant to be inspected and reproduced. The
[public UI](https://cudaq-blueprint-demo.website-us-east-1.linodeobjects.com/)
is a read-only static snapshot hosted from Akamai Object Storage. If
you clone the repository and run the container locally, the local
developer UI is served from your own machine at
`http://localhost:8000`.

CPU path, runs anywhere with Docker (~17 seconds for H<sub>2</sub>):

```bash
git clone https://github.com/jgdynamite/cudaq-molecular-simulation-blueprint.git
cd cudaq-molecular-simulation-blueprint
make container-build
make container-run-cpu
make serve   # serves the UI on your own machine at http://localhost:8000
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
above: **~$10.75** for **~3 h 35 min** of total VM lifetime on
`g3-gpu-rtxpro6000-blackwell-1` at the Jakarta regional rate of
$3.00/hr. That figure includes provisioning, NVIDIA driver install +
reboot, container build, the 2 h 27 min of bench compute itself,
results export, and teardown. The base SKU rate is $2.50/hr; the
Jakarta and S&atilde;o Paulo regions carry a $0.50/hr uplift, so a
"compute-time &times; base-rate" estimate would have under-shot the
real bill by roughly 2&times;. Always destroy the instance when not
actively using it.

The full Terraform module, Ansible roles, and Dockerfile are in the
repo under `infra/`. The application code is provider-agnostic &mdash; you
can lift the Docker image to any GPU host that has the NVIDIA Container
Toolkit and the open kernel modules installed and the same binary will
run.

---

## How to extend this work

This post is intentionally narrow: one reproducible hybrid
quantum-classical workflow, two molecules, one validated Blackwell
host, multiple seeds, and published benchmark artifacts.

There are several ways the artifact could be extended by others:

- **Different optimizers or ansatz choices.** The LiH result shows that
  optimizer initialization matters. A reader could experiment with
  different optimizer strategies or ansatz configurations while keeping
  the same benchmark harness.
- **Different molecule sizes or active spaces.** This benchmark does
  not claim to find the scale limit of Blackwell. It establishes a
  measured baseline that others can extend with larger chemistry
  problems.
- **Different GPU configurations.** The current published result is
  single-GPU. Multi-GPU would be a separate benchmark with its own
  measurement contract.
- **Different infrastructure targets.** The application core is
  provider-agnostic. Akamai Cloud is the validated path in this post,
  but the containerized workload can be evaluated on any compatible GPU
  host.

Those are extensions, not prerequisites for the current result. The
point of this post is the artifact as it stands today: a reproducible,
measured, end-to-end demonstration of how GPUs already matter to
practical quantum workflow development before QPUs enter the picture.

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
  CUDA-Q `set_target()` call away; I left it out so the post stays
  focused on the part of the workflow that runs on hardware available
  to anyone today. The point is that the hybrid workflow is already
  worth standing up before the QPU column has a number in it.

---

## Links

- Repository: <https://github.com/jgdynamite/cudaq-molecular-simulation-blueprint>
- Live UI snapshot: <https://cudaq-blueprint-demo.website-us-east-1.linodeobjects.com/>
- Container image: `ghcr.io/jgdynamite/cudaq-molecular-simulation-blueprint:v0.1.0`
- CUDA-Q docs: <https://nvidia.github.io/cuda-quantum/>
- cuQuantum SDK: <https://developer.nvidia.com/cuquantum-sdk>
- Akamai Cloud GPU SKUs: <https://www.linode.com/pricing/#compute-gpu>
