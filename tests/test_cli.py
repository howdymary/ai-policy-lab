from pathlib import Path

from typer.testing import CliRunner

from ai_policy_lab.cli import app


def test_cli_run_writes_report_and_state(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("APL_USE_MOCK", "true")

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run",
            "--question",
            "How is AI adoption affecting occupational mobility?",
            "--constraint",
            "United States",
            "--quality-floor",
            "tier_2",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "state.json").exists()
