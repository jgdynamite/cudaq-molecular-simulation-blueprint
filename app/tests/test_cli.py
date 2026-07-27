"""CLI smoke tests using Typer's runner.

We invoke each command with a wide terminal and disabled colour/markup so
Rich does not wrap option flags across lines or inject ANSI escapes that
trip simple substring assertions on narrow CI runners. We also use
``result.output`` (Click's combined stdout+stderr capture), which is more
reliable than ``result.stdout`` across Click 8.x point releases.
"""

from __future__ import annotations

from typer.testing import CliRunner

from app.cli.main import app

runner = CliRunner()

WIDE_ENV = {"COLUMNS": "200", "NO_COLOR": "1", "TERM": "dumb"}


def test_help_lists_subcommands() -> None:
    result = runner.invoke(app, ["--help"], env=WIDE_ENV)
    assert result.exit_code == 0
    assert "run" in result.output
    assert "results" in result.output
    assert "bench" in result.output


def test_info_runs() -> None:
    result = runner.invoke(app, ["info"], env=WIDE_ENV)
    assert result.exit_code == 0
    assert "python_version" in result.output


def test_run_help_documents_backend_option() -> None:
    result = runner.invoke(app, ["run", "h2", "--help"], env=WIDE_ENV)
    assert result.exit_code == 0
    assert "--backend" in result.output


def test_results_list_with_no_runs() -> None:
    result = runner.invoke(app, ["results", "list"], env=WIDE_ENV)
    assert result.exit_code == 0
    assert "No runs found" in result.output


def test_bench_run_suite_help_documents_seed_and_iterations() -> None:
    result = runner.invoke(app, ["bench", "run-suite", "--help"], env=WIDE_ENV)
    assert result.exit_code == 0
    assert "--seeds" in result.output
    assert "--lih-max-iterations" in result.output
    assert "--skip-gpu" in result.output


def test_run_lih_help_documents_ansatz_mode() -> None:
    result = runner.invoke(app, ["run", "lih", "--help"], env=WIDE_ENV)
    assert result.exit_code == 0
    assert "--ansatz-mode" in result.output
    assert "matched" in result.output
    assert "legacy_full" in result.output


def test_run_lih_rejects_unknown_ansatz_mode() -> None:
    result = runner.invoke(app, ["run", "lih", "--ansatz-mode", "bogus"], env=WIDE_ENV)
    assert result.exit_code != 0
    assert "unknown ansatz mode" in result.output


def test_bench_followup_help_documents_required_options() -> None:
    result = runner.invoke(app, ["bench", "run-lih-active-space-followup", "--help"], env=WIDE_ENV)
    assert result.exit_code == 0
    assert "--seeds" in result.output
    assert "--max-iterations" in result.output
    assert "--output-dir" in result.output
    assert "lih-active-space-followup-v02" in result.output
