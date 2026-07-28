# Project charter

## Project name

`cudaq-molecular-simulation-blueprint`

## Project purpose

Build a portable, public, reproducible reference implementation for hybrid
quantum-classical molecular simulation using NVIDIA CUDA-Q and cuQuantum,
validated initially on Akamai Cloud using NVIDIA RTX PRO 6000 Blackwell Server
Edition infrastructure.

## Primary narrative

This project exists to support a forthcoming technical article on
evaluating hybrid quantum-classical workloads.

(Working title through the May 2026 bench cycle: "Why GPUs Matter to
Quantum Before QPUs Do." The scope below is unchanged; the framing moved
from a GPU-performance argument to an investment-discipline argument
after the July active-space follow-up showed the headline speedup was an
artifact of ansatz misdimensioning.)

## Core thesis

Practical quantum development today is often hybrid rather than QPU-only. In
this context, hybrid means CPUs handle orchestration and chemistry
preprocessing, GPUs accelerate simulation and optimization, and QPUs are
optional rather than required.

## Why this project matters

1. It demonstrates a real near-term quantum workflow rather than a speculative
   one.
2. It shows why GPUs already matter to the quantum software stack today.
3. It uses molecular simulation as a concrete, credible workload.
4. It validates that Akamai Cloud can host this class of modern Blackwell GPU
   workload.
5. It does this in a portable way, without tightly coupling the core
   application to Akamai-specific code.

## Important positioning

1. This is not a quantum advantage claim.
2. This is not a claim that Akamai is a dedicated quantum cloud platform.
3. This is not the first cloud-based hybrid quantum workflow.
4. The differentiator is a portable, public, Akamai-validated Blackwell-era
   reference implementation for molecular simulation.

## What this project is trying to show

1. Quantum development today is often hybrid, not QPU-only.
2. GPUs are essential to the current practical quantum workflow.
3. Molecular simulation is a concrete and credible workload for this story.
4. Akamai Cloud can host this workload on modern Blackwell GPU infrastructure.
5. The implementation should be portable and not tightly coupled to
   Akamai-specific application code.

## Primary audience

1. Technical readers interested in quantum computing, CUDA-Q, and
   GPU-accelerated simulation.
2. Infrastructure and platform readers evaluating Blackwell-class GPU
   workloads.
3. Readers interested in Akamai Cloud as a serious GPU infrastructure option.
4. Future readers of the companion blog post.

## Design principles

1. Keep the core application provider-agnostic.
2. Keep Akamai-specific code isolated under `infra/`.
3. Make the repo runnable in CPU mode by the public.
4. Make GPU mode reproducible on Akamai.
5. Prefer correctness, clarity, and reproducibility over complexity.
6. Keep the MVP intentionally narrow.
7. Use containers so the runtime is portable even though the first validation
   target is Akamai.

## Definition of done for v1

1. Public GitHub repo.
2. CPU-capable local run path.
3. GPU-capable Akamai deployment path.
4. H2 VQE experiment implemented and documented.
5. LiH VQE experiment implemented and documented.
6. Benchmark outputs comparing CPU backend vs NVIDIA backend.
7. README with architecture, deployment, results methodology, and
   blog-supporting narrative.
8. Clean code, tests, and docs suitable for public viewing.
9. Minimal technical demonstration UI or simple browser page for running
   experiments and viewing results.
10. Blog-ready artifacts such as charts, result summaries, and screenshots.
