"""``cudaq-bp`` command-line entry point.

Subcommands:

- ``run h2``      Run an H2 VQE experiment.
- ``run lih``     Run an LiH VQE experiment.
- ``results list``     List previous runs.
- ``results show <id>`` Print a manifest.
- ``bench run-suite``  Run the full multi-seed benchmark suite.
- ``bench run-lih-active-space-followup``
                  Run the LiH matched-vs-legacy ansatz follow-up.
- ``bench compare``    Generate a CPU vs GPU comparison report.
- ``info``        Print system + GPU detection summary.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.core.system_info import collect_system_info
from app.storage.filesystem import (
    list_runs,
    load_manifest,
    load_trace,
    save_manifest,
    save_trace,
)
from app.storage.manifests import AnsatzMode, BackendIdentifier

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="cudaq-molecular-simulation-blueprint - hybrid quantum-classical VQE on CPU or GPU.",
)
run_app = typer.Typer(help="Run an experiment.", no_args_is_help=True)
results_app = typer.Typer(help="Inspect past runs.", no_args_is_help=True)
bench_app = typer.Typer(help="Benchmark utilities.", no_args_is_help=True)

app.add_typer(run_app, name="run")
app.add_typer(results_app, name="results")
app.add_typer(bench_app, name="bench")

console = Console()
log = get_logger("cli")

#: Default sub-directory (inside the configured results directory) for the
#: LiH active-space ansatz follow-up sweep.
DEFAULT_FOLLOWUP_OUTPUT_DIR = "lih-active-space-followup-v02"


def _parse_backend(value: str) -> BackendIdentifier:
    aliases = {
        "cpu": BackendIdentifier.CPU,
        "qpp-cpu": BackendIdentifier.CPU,
        "gpu": BackendIdentifier.GPU_FP64,
        "gpu_fp32": BackendIdentifier.GPU_FP32,
        "nvidia": BackendIdentifier.GPU_FP32,
        "gpu_fp64": BackendIdentifier.GPU_FP64,
        "nvidia-fp64": BackendIdentifier.GPU_FP64,
    }
    key = value.strip().lower()
    if key not in aliases:
        raise typer.BadParameter(f"unknown backend '{value}'. Valid: {', '.join(aliases)}")
    return aliases[key]


def _parse_ansatz_mode(value: str) -> AnsatzMode:
    key = value.strip().lower()
    try:
        return AnsatzMode(key)
    except ValueError:
        valid = ", ".join(m.value for m in AnsatzMode)
        raise typer.BadParameter(f"unknown ansatz mode '{value}'. Valid: {valid}") from None


def _summarize(manifest, trace) -> None:  # type: ignore[no-untyped-def]
    table = Table(title=f"Run {manifest.run_id}")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("backend", manifest.backend.value)
    table.add_row("target", manifest.target_string)
    table.add_row("molecule", manifest.molecule.name.value)
    table.add_row("qubits", str(manifest.qubit_count))
    table.add_row("parameters", str(manifest.parameter_count))
    if manifest.result is not None:
        r = manifest.result
        table.add_row("energy [Ha]", f"{r.energy:.6f}")
        if r.reference_energy is not None:
            table.add_row("reference [Ha]", f"{r.reference_energy:.6f}")
            table.add_row("error [Ha]", f"{r.error_vs_reference_hartree:+.2e}")
            table.add_row(
                "chemical accuracy",
                "yes" if r.chemical_accuracy_reached else "no",
            )
        table.add_row("iterations", str(r.iterations))
        table.add_row("wall time [s]", f"{r.wall_time_seconds:.3f}")
    table.add_row("function evaluations", str(len(trace.records)))
    console.print(table)


@app.callback()
def _root(
    log_level: str = typer.Option(None, "--log-level", help="Override CUDAQ_BP_LOG_LEVEL."),
) -> None:
    settings = get_settings()
    configure_logging(level=log_level or settings.log_level)


@app.command("info")
def info() -> None:
    """Print system + GPU detection summary."""
    info = collect_system_info()
    console.print(Panel.fit(json.dumps(info.to_dict(), indent=2), title="system info"))


@run_app.command("h2")
def run_h2_cmd(
    backend: str = typer.Option("cpu", "--backend", "-b", help="cpu | gpu_fp32 | gpu_fp64"),
    bond_distance: float = typer.Option(0.7414, "--bond-distance", help="H-H distance in Angstrom"),
    seed: int = typer.Option(42, "--seed"),
    max_iterations: int = typer.Option(200, "--max-iterations"),
) -> None:
    """Run the H2 VQE experiment."""
    from app.quantum.h2_vqe import run_h2  # imported lazily to avoid cudaq import cost

    backend_id = _parse_backend(backend)
    log.info("h2_run.start", backend=backend_id.value, seed=seed)
    manifest, trace = run_h2(
        backend_id=backend_id,
        bond_distance=bond_distance,
        seed=seed,
        max_iterations=max_iterations,
    )
    save_manifest(manifest)
    save_trace(trace)
    log.info("h2_run.completed", run_id=manifest.run_id)
    _summarize(manifest, trace)


@run_app.command("lih")
def run_lih_cmd(
    backend: str = typer.Option("gpu_fp64", "--backend", "-b", help="cpu | gpu_fp32 | gpu_fp64"),
    bond_distance: float = typer.Option(
        1.5957, "--bond-distance", help="Li-H distance in Angstrom"
    ),
    n_core_orbitals: int | None = typer.Option(
        1, "--core-orbitals", help="Number of frozen core orbitals; pass 0 to disable"
    ),
    n_active_orbitals: int | None = typer.Option(
        5, "--active-orbitals", help="Number of active spatial orbitals"
    ),
    seed: int = typer.Option(42, "--seed"),
    max_iterations: int = typer.Option(500, "--max-iterations"),
    ansatz_mode: str = typer.Option(
        AnsatzMode.MATCHED.value,
        "--ansatz-mode",
        help=(
            "matched = build UCCSD for the active space (10 qubits for default LiH); "
            "legacy_full = build it for the full molecule (12 qubits), reproducing v0.1."
        ),
    ),
) -> None:
    """Run the LiH VQE experiment."""
    from app.quantum.lih_vqe import run_lih  # imported lazily

    backend_id = _parse_backend(backend)
    mode = _parse_ansatz_mode(ansatz_mode)
    core = n_core_orbitals if (n_core_orbitals and n_core_orbitals > 0) else None
    active = n_active_orbitals if (n_active_orbitals and n_active_orbitals > 0) else None
    log.info("lih_run.start", backend=backend_id.value, seed=seed, ansatz_mode=mode.value)
    manifest, trace = run_lih(
        backend_id=backend_id,
        bond_distance=bond_distance,
        n_core_orbitals=core,
        n_active_orbitals=active,
        seed=seed,
        max_iterations=max_iterations,
        ansatz_mode=mode,
    )
    save_manifest(manifest)
    save_trace(trace)
    log.info("lih_run.completed", run_id=manifest.run_id)
    _summarize(manifest, trace)


@results_app.command("list")
def results_list() -> None:
    """List all runs in the configured results directory."""
    runs = list_runs()
    if not runs:
        console.print("[yellow]No runs found in results directory.[/yellow]")
        return
    table = Table(title=f"{len(runs)} runs")
    table.add_column("run_id")
    table.add_column("molecule")
    table.add_column("backend")
    table.add_column("status")
    table.add_column("energy")
    table.add_column("iters")
    table.add_column("wall (s)")
    for m in runs:
        e = f"{m.result.energy:.4f}" if m.result else "-"
        it = str(m.result.iterations) if m.result else "-"
        wt = f"{m.result.wall_time_seconds:.2f}" if m.result else "-"
        table.add_row(m.run_id, m.molecule.name.value, m.backend.value, m.status.value, e, it, wt)
    console.print(table)


@results_app.command("show")
def results_show(run_id: str = typer.Argument(...)) -> None:
    """Print the manifest for a single run."""
    manifest = load_manifest(run_id)
    trace = load_trace(run_id)
    _summarize(manifest, trace)


@bench_app.command("run-suite")
def bench_run_suite(
    seeds: str = typer.Option(
        "42,43,44",
        "--seeds",
        help="Comma-separated seed list (e.g. '42,43,44').",
    ),
    h2_max_iterations: int = typer.Option(
        200, "--h2-max-iterations", help="COBYLA max iterations for H2."
    ),
    lih_max_iterations: int = typer.Option(
        1500,
        "--lih-max-iterations",
        help="COBYLA max iterations for LiH (300 was the v0.1.0 cap; 1500 lets it converge).",
    ),
    skip_gpu: bool = typer.Option(
        False, "--skip-gpu", help="Skip GPU backends (e.g. for CPU-only smoke tests)."
    ),
) -> None:
    """Run the full multi-seed benchmark suite end-to-end.

    Defaults reproduce the post-v0.1.0 publication suite (3 seeds, LiH max
    iterations bumped from 300 to 1500 so the optimizer can reach chemical
    accuracy). Each spec is persisted as a regular run manifest + trace, so
    the standard ``cudaq-bp results list`` and ``cudaq-bp bench compare``
    commands continue to work afterwards.
    """
    from app.benchmark.runner import default_blog_suite, run_benchmark_suite

    seed_tuple = tuple(int(s.strip()) for s in seeds.split(",") if s.strip())
    if not seed_tuple:
        raise typer.BadParameter("--seeds must contain at least one integer")

    suite = default_blog_suite(
        seeds=seed_tuple,
        h2_max_iterations=h2_max_iterations,
        lih_max_iterations=lih_max_iterations,
    )
    if skip_gpu:
        suite = [s for s in suite if s.backend == BackendIdentifier.CPU]

    console.print(
        Panel.fit(
            f"Running {len(suite)} bench specs across {len(seed_tuple)} seed(s).\n"
            f"  seeds: {seed_tuple}\n"
            f"  H2  max_iterations: {h2_max_iterations}\n"
            f"  LiH max_iterations: {lih_max_iterations}\n"
            f"  skip_gpu: {skip_gpu}",
            title="bench run-suite",
            border_style="cyan",
        )
    )
    manifests = run_benchmark_suite(suite)
    console.print(f"\n[green]Done.[/green] Persisted {len(manifests)} runs.")
    table = Table(title="Suite summary")
    table.add_column("molecule")
    table.add_column("backend")
    table.add_column("seed", justify="right")
    table.add_column("status")
    table.add_column("energy", justify="right")
    table.add_column("iters", justify="right")
    table.add_column("wall (s)", justify="right")
    for m in manifests:
        e = f"{m.result.energy:.4f}" if m.result else "-"
        it = str(m.result.iterations) if m.result else "-"
        wt = f"{m.result.wall_time_seconds:.2f}" if m.result else "-"
        table.add_row(
            m.molecule.name.value, m.backend.value, str(m.seed), m.status.value, e, it, wt
        )
    console.print(table)


@bench_app.command("run-lih-active-space-followup")
def bench_run_lih_active_space_followup(
    seeds: str = typer.Option(
        "42,43,44,45,46",
        "--seeds",
        help="Comma-separated seed list.",
    ),
    max_iterations: int = typer.Option(
        1500,
        "--max-iterations",
        help="COBYLA max iterations, applied identically to all three arms.",
    ),
    output_dir: str = typer.Option(
        DEFAULT_FOLLOWUP_OUTPUT_DIR,
        "--output-dir",
        help=(
            "Directory for this sweep's runs and reports. Relative paths are "
            "resolved inside the configured results directory."
        ),
    ),
) -> None:
    """Run the LiH active-space ansatz follow-up experiment.

    Three arms at a fixed iteration budget: legacy_full on GPU FP64, matched
    on GPU FP64, and matched on CPU. Arms run contiguously so CPU and GPU
    work is never interleaved. Writes SUMMARY.csv and comparison.json into
    the output directory when the sweep finishes.

    This is a long sweep. With five seeds it is 15 LiH runs at up to 1500
    COBYLA iterations each; budget accordingly before starting a GPU VM.
    """
    from app.benchmark.followup import generate_reports
    from app.benchmark.runner import lih_active_space_followup_suite, run_benchmark_suite
    from app.core.config import get_settings, results_dir_override

    seed_tuple = tuple(int(s.strip()) for s in seeds.split(",") if s.strip())
    if not seed_tuple:
        raise typer.BadParameter("--seeds must contain at least one integer")

    target = Path(output_dir)
    if not target.is_absolute():
        target = get_settings().results_dir / target

    suite = lih_active_space_followup_suite(seeds=seed_tuple, max_iterations=max_iterations)

    console.print(
        Panel.fit(
            f"Running {len(suite)} LiH specs across {len(seed_tuple)} seed(s).\n"
            f"  seeds:          {seed_tuple}\n"
            f"  max_iterations: {max_iterations}\n"
            f"  arms:           legacy_full/gpu_fp64, matched/gpu_fp64, matched/cpu\n"
            f"  output_dir:     {target}",
            title="bench run-lih-active-space-followup",
            border_style="cyan",
        )
    )

    with results_dir_override(target):
        manifests = run_benchmark_suite(suite)
        csv_path, json_path = generate_reports(target, manifests)

    console.print(f"\n[green]Done.[/green] Persisted {len(manifests)} runs to {target}")

    table = Table(title="Follow-up suite summary")
    table.add_column("variant")
    table.add_column("backend")
    table.add_column("seed", justify="right")
    table.add_column("qubits", justify="right")
    table.add_column("params", justify="right")
    table.add_column("status")
    table.add_column("energy", justify="right")
    table.add_column("wall (s)", justify="right")
    for m in manifests:
        e = f"{m.result.energy:.6f}" if m.result else "-"
        wt = f"{m.result.wall_time_seconds:.2f}" if m.result else "-"
        table.add_row(
            str(m.notes.get("ansatz_mode", "-")),
            m.backend.value,
            str(m.seed),
            str(m.qubit_count),
            str(m.parameter_count),
            m.status.value,
            e,
            wt,
        )
    console.print(table)
    console.print(f"[green]Wrote[/green] {csv_path}")
    console.print(f"[green]Wrote[/green] {json_path}")


@bench_app.command("compare")
def bench_compare(
    output: str = typer.Option(
        "results/blog/cpu_vs_gpu.json",
        "--output",
        "-o",
        help="Output path for the comparison report.",
    ),
) -> None:
    """Generate a CPU vs GPU comparison report from existing runs."""
    from app.benchmark.compare import compare_cpu_vs_gpu  # imported lazily

    report = compare_cpu_vs_gpu()
    from pathlib import Path

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    console.print(f"[green]Wrote[/green] {path}")


if __name__ == "__main__":  # pragma: no cover
    app()
