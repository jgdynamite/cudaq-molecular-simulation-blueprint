"""Pre-render the FastAPI/Jinja2 UI to a static bucket-friendly tree.

The live UI is a server-rendered FastAPI app with HTMX/SSE for interactive
runs. For the public blog companion we want a static snapshot that:

* shows the **real** Jakarta benchmark data (4 manifests + traces +
  cpu_vs_gpu.json) without any Python/cudaq runtime;
* preserves layout fidelity but disables the run form and SSE plumbing
  with a clear "static demo, see GitHub for live runs" banner;
* lays files out as ``<page>/index.html`` so Akamai Object Storage's
  static-website hosting resolves clean URLs (``/run/`` -> ``/run/index.html``).

Usage::

    python -m app.ui.static_export \
        --results-dir results/akamai-jakarta \
        --output-dir _site

The script imports the same Jinja2 templates the live server uses; the only
template additions for static mode are guarded by ``{% if static_mode %}``,
so live behaviour is unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class StaticExportConfig:
    results_dir: Path
    output_dir: Path
    deployment_date: str = "2026-05-03"


def _configure_results_dir(results_dir: Path) -> None:
    """Point the cached Settings singleton at ``results_dir``.

    ``app.core.config.get_settings`` lazily caches a ``Settings`` instance on
    first call; we set the env var *before* importing anything that calls it.
    """
    os.environ["CUDAQ_BP_RESULTS_DIR"] = str(results_dir.resolve())


def _system_info_from_runs(runs: list[Any]) -> dict[str, Any]:
    """Pick the most informative ``system_info`` block from the Jakarta runs.

    Prefer a manifest whose target is ``nvidia-fp64`` so the home page shows
    the Blackwell GPU, driver, and CUDA version captured during deployment.
    """
    gpu_runs = [m for m in runs if "nvidia" in (m.target_string or "")]
    chosen = gpu_runs[0] if gpu_runs else (runs[0] if runs else None)
    if chosen is None:
        return {
            "cudaq_version": None,
            "python_version": "n/a",
            "platform": "static-demo",
            "gpus": [],
        }
    info = dict(chosen.system_info or {})
    info.setdefault("gpus", [])
    return info


def _write(path: Path, html: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


_RUN_ID_LINK_RE = re.compile(r'href="(/results/[^"/]+)"')


def _rewrite_links_for_static(html: str) -> str:
    """Rewrite live-server URLs to static, trailing-slash conventions.

    Object Storage's website hosting resolves ``/run/`` to ``/run/index.html``,
    but ``/run`` (no slash) just 404s. We patch the navigation, intra-site
    anchors, and dynamic ``/results/<run_id>`` links so every internal href
    ends with a slash.
    """
    replacements = [
        ('href="/run"', 'href="/run/"'),
        ('href="/results"', 'href="/results/"'),
        ('href="/compare"', 'href="/compare/"'),
        (
            'href="/docs"',
            'href="https://github.com/jgdynamite/cudaq-molecular-simulation-blueprint#documentation"',
        ),
        ('href="/run?', 'href="/run/?'),
    ]
    for old, new in replacements:
        html = html.replace(old, new)
    return _RUN_ID_LINK_RE.sub(lambda m: f'href="{m.group(1)}/"', html)


def _render_pages(out: Path, runs: list, ctx: dict[str, Any]) -> dict[str, Any]:
    from app.benchmark.compare import compare_cpu_vs_gpu
    from app.core.metadata import project_version
    from app.storage.filesystem import load_trace
    from app.storage.manifests import BackendIdentifier
    from app.ui.server import templates

    sys_info = _system_info_from_runs(runs)
    has_gpu = bool(sys_info.get("gpus"))
    backends = list(BackendIdentifier)

    home = templates.get_template("index.html").render(
        request=None,
        system_info=sys_info,
        project_version=project_version(),
        has_gpu=has_gpu,
        backends=backends,
        **ctx,
    )
    _write(out / "index.html", _rewrite_links_for_static(home))

    run_page = templates.get_template("run.html").render(
        request=None,
        has_gpu=has_gpu,
        backends=backends,
        **ctx,
    )
    _write(out / "run" / "index.html", _rewrite_links_for_static(run_page))

    results_page = templates.get_template("results.html").render(
        request=None,
        runs=runs,
        **ctx,
    )
    _write(out / "results" / "index.html", _rewrite_links_for_static(results_page))

    for manifest in runs:
        trace = load_trace(manifest.run_id)
        detail = templates.get_template("result_detail.html").render(
            request=None,
            manifest=manifest,
            trace_records=[r.model_dump(mode="json") for r in trace.records],
            **ctx,
        )
        _write(
            out / "results" / manifest.run_id / "index.html",
            _rewrite_links_for_static(detail),
        )

    report = compare_cpu_vs_gpu()
    compare_page = templates.get_template("compare.html").render(
        request=None,
        report=report,
        **ctx,
    )
    _write(out / "compare" / "index.html", _rewrite_links_for_static(compare_page))

    error_404 = (
        "<!doctype html><html><head><meta charset=utf-8>"
        "<title>Not found</title>"
        '<meta http-equiv="refresh" content="0; url=/"></head>'
        '<body><a href="/">cudaq-molecular-simulation-blueprint</a></body></html>'
    )
    _write(out / "404.html", error_404)
    return report


def _copy_assets(cfg: StaticExportConfig, runs: list, report: dict[str, Any]) -> None:
    from app.ui.server import STATIC_DIR

    out = cfg.output_dir
    if STATIC_DIR.exists():
        shutil.copytree(STATIC_DIR, out / "static")

    data_dir = out / "data"
    data_dir.mkdir(exist_ok=True)
    (data_dir / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for manifest in runs:
        run_data = data_dir / manifest.run_id
        run_data.mkdir(exist_ok=True)
        src = cfg.results_dir / manifest.run_id
        for filename in ("manifest.json", "trace.json"):
            if (src / filename).exists():
                shutil.copy(src / filename, run_data / filename)


def export(cfg: StaticExportConfig) -> None:
    _configure_results_dir(cfg.results_dir)

    from app.storage.filesystem import list_runs

    out = cfg.output_dir
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    runs = list_runs()
    if not runs:
        raise SystemExit(f"no runs found under {cfg.results_dir}; nothing to export")
    print(f"  found {len(runs)} runs")

    ctx: dict[str, Any] = {
        "static_mode": True,
        "static_deployment_date": cfg.deployment_date,
    }

    report = _render_pages(out, runs, ctx)
    print(f"  rendered home, run, results, {len(runs)} run details, compare, 404")

    _copy_assets(cfg, runs, report)
    print("  copied static assets and raw manifests/traces")

    total_files = sum(1 for _ in out.rglob("*") if _.is_file())
    total_bytes = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"\n  done: {total_files} files, {total_bytes / 1024:.1f} KiB total under {out}")


def _parse_args(argv: list[str]) -> StaticExportConfig:
    parser = argparse.ArgumentParser(
        description=__doc__.strip().splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/akamai-jakarta"),
        help="Directory containing the run manifests + traces to embed.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("_site"),
        help="Directory to write the static site into (will be wiped).",
    )
    parser.add_argument(
        "--deployment-date",
        default="2026-05-03",
        help="Date string surfaced in the static-mode banner.",
    )
    args = parser.parse_args(argv)
    return StaticExportConfig(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        deployment_date=args.deployment_date,
    )


def main(argv: list[str] | None = None) -> None:
    cfg = _parse_args(argv if argv is not None else sys.argv[1:])
    if not cfg.results_dir.exists():
        raise SystemExit(f"results-dir does not exist: {cfg.results_dir}")
    print(f"building static site from {cfg.results_dir} -> {cfg.output_dir}")
    export(cfg)


if __name__ == "__main__":
    main()
