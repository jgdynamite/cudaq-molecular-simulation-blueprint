# Architecture

## High-level

The blueprint is a single Python package (`app/`) with thin CLI, API, and UI
front-doors over a shared core. Akamai-specific infrastructure lives entirely
under `infra/` and is invoked by `scripts/bootstrap_host.sh`.

```mermaid
flowchart LR
  subgraph clients [Clients]
    cli[Typer CLI cudaq-bp]
    ui[Browser UI]
    httpc[HTTP clients]
  end

  cli --> core
  ui --> api
  httpc --> api

  subgraph app [app/]
    api[FastAPI app/api]
    coord[Run coordinator app/api/deps]
    core[Experiment core app/quantum]
    bench[Benchmark harness app/benchmark]
    store[Storage app/storage]
  end

  api --> coord
  api --> store
  coord --> core
  bench --> core
  bench --> store
  core --> chem[chemistry preprocessing]
  core --> ans[UCCSD ansatz]
  core --> opt[COBYLA optimizer]
  core --> backends[Backend abstraction]
  backends --> cudaq[CUDA-Q runtime]
  cudaq --> custatevec[cuStateVec on GPU]
  cudaq --> qpp[OpenMP CPU statevector]
  core --> store

  subgraph akamai [infra/ Akamai-only]
    tf[Terraform linode_instance] --> vm[(Blackwell VM g3-gpu-rtxpro6000-blackwell-1)]
    ans2[Ansible nvidia_driver + docker + app] --> vm
    vm --> docker[(Docker container running app)]
  end
```

## Module boundaries

`app/` is provider-agnostic. It does not import anything from `infra/` and
has no awareness of Akamai. The only "deployment" knowledge inside `app/` is
the `nvidia` vs `qpp-cpu` backend mapping in
[`app/quantum/backends.py`](https://github.com/jgdynamite/cudaq-molecular-simulation-blueprint/blob/main/app/quantum/backends.py), which is a CUDA-Q
concept, not an Akamai concept.

`infra/` contains everything Akamai-specific:

- [`infra/terraform/akamai/`](https://github.com/jgdynamite/cudaq-molecular-simulation-blueprint/tree/main/infra/terraform/akamai) - Terraform stack
  using the official `linode/linode` provider.
- [`infra/ansible/`](https://github.com/jgdynamite/cudaq-molecular-simulation-blueprint/tree/main/infra/ansible) - Ansible playbook + roles that
  install drivers, Docker, and the application container.
- [`infra/k8s/future/`](https://github.com/jgdynamite/cudaq-molecular-simulation-blueprint/tree/main/infra/k8s/future) - placeholder; LKE is an
  explicit non-goal for v1.

## Hybrid pipeline

```mermaid
sequenceDiagram
  participant U as User
  participant API as FastAPI
  participant C as Coordinator
  participant T as Worker thread
  participant Q as CUDA-Q
  participant FS as results/

  U->>API: POST /api/runs/h2 {backend, seed, ...}
  API->>C: spawn(runner)
  C-->>API: run_id
  API-->>U: 200 {run_id, status: running}
  U->>API: GET /api/runs/{id}/stream (SSE)

  C->>T: asyncio.to_thread(runner)
  T->>Q: cudaq.set_target(qpp-cpu | nvidia)
  T->>Q: cudaq.chemistry.create_molecular_hamiltonian(...)
  T->>Q: cudaq.kernels.uccsd
  loop optimizer iteration
    T->>Q: cudaq.observe(kernel, hamiltonian, theta)
    Q-->>T: expectation
    T-->>API: on_iteration(record)
    API-->>U: SSE event "iteration"
  end
  T->>FS: write manifest.json + trace.json
  T-->>C: completed
  API-->>U: SSE event "completed"
```

## Result manifest

Every run writes `results/<run_id>/manifest.json` and
`results/<run_id>/trace.json`. The manifest captures the full
reproducibility context:

- CUDA-Q version, target string, RNG seed, optimizer settings.
- Geometry, basis set, charge, multiplicity, optional active-space.
- System info: OS, CPU, Python, NVIDIA driver, CUDA version, GPU model + UUID,
  memory.
- Git SHA of the code that produced it, when the process can see a `.git`
  directory. Container-based runs record `null` here; there is no container
  image ID or digest field in the schema. See the README provenance note.
- Final result: energy, parameters, evaluation count, wall time, error vs
  reference, whether chemical accuracy was reached.

This is the artifact downstream tools (`compare`, the UI, the blog post) all
read from. There is no database in v1 - the filesystem is the source of truth.

## Backend abstraction

`app/quantum/backends.py` exposes three logical backends:

| identifier  | CUDA-Q target   | hardware    | use case                               |
|-------------|-----------------|-------------|----------------------------------------|
| `cpu`       | `qpp-cpu`       | CPU/OpenMP  | local Mac/Linux, public reproducibility |
| `gpu_fp32`  | `nvidia` / fp32 | GPU         | speed-leaning runs                      |
| `gpu_fp64`  | `nvidia` / fp64 | GPU         | accuracy-leaning runs (the comparison)  |

Adding a new logical backend is a one-line change in `BACKEND_CONFIGS`. The
rest of the app (CLI, API, UI, benchmark) picks it up automatically through
the enum.
