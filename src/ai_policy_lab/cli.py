from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import cast

import typer
from rich.console import Console
from rich.panel import Panel

from ai_policy_lab.graph import run_research
from ai_policy_lab.runtime import ResearchRuntime
from ai_policy_lab.sanitize import sanitize_user_input, sanitize_user_inputs
from ai_policy_lab.state import QualityFloor
from ai_policy_lab.utils import slugify

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


@app.callback()
def main() -> None:
    """AI Policy Research Lab CLI."""


@app.command()
def run(
    question: str = typer.Option(..., "--question", help="Root research question."),
    constraint: list[str] = typer.Option(
        [],
        "--constraint",
        help="Domain constraint. Repeat for multiple constraints.",
    ),
    quality_floor: str = typer.Option(
        "tier_2",
        "--quality-floor",
        help="Minimum acceptable source tier.",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        help="Directory for report and state artifacts.",
    ),
    model: str | None = typer.Option(None, "--model", help="Override the default LLM model."),
    verbose: bool = typer.Option(False, "--verbose", help="Enable verbose logging."),
    quiet: bool = typer.Option(False, "--quiet", help="Reduce log output."),
) -> None:
    if quality_floor not in {"tier_1", "tier_2", "tier_3"}:
        raise typer.BadParameter("quality floor must be one of: tier_1, tier_2, tier_3")
    if verbose and quiet:
        raise typer.BadParameter("Choose either --verbose or --quiet, not both.")

    logging.basicConfig(level=logging.DEBUG if verbose else logging.WARNING if quiet else logging.INFO)

    runtime = ResearchRuntime.from_env()
    if model:
        runtime.settings.default_model = model

    sanitized_question = sanitize_user_input(question)
    sanitized_constraints = sanitize_user_inputs(constraint)

    state = run_research(
        runtime=runtime,
        root_question=sanitized_question,
        domain_constraints=sanitized_constraints,
        quality_floor=cast(QualityFloor, quality_floor),
    )

    target_dir = _validate_output_dir(
        output_dir or _default_output_dir(runtime=runtime, question=sanitized_question),
        runs_dir=runtime.settings.runs_dir,
    )
    target_dir.mkdir(parents=True, exist_ok=True)

    report_path = target_dir / "report.md"
    state_path = target_dir / "state.json"

    report_path.write_text(state["full_report"], encoding="utf-8")
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    mode_label = "mock" if runtime.settings.use_mock else "live"
    console.print(
        Panel.fit(
            f"Run completed in [bold]{mode_label}[/bold] mode.\n"
            f"Report: {report_path}\n"
            f"State: {state_path}",
            title="AI Policy Lab",
        )
    )


def _default_output_dir(*, runtime: ResearchRuntime, question: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return runtime.settings.runs_dir / f"{timestamp}-{slugify(question)[:48]}"


def _validate_output_dir(target_dir: Path, *, runs_dir: Path) -> Path:
    resolved = target_dir.resolve()
    allowed_roots = [Path.cwd().resolve(), runs_dir.resolve()]
    if any(_is_within(resolved, root) for root in allowed_roots):
        return resolved
    raise typer.BadParameter(f"Output dir must be within {allowed_roots[0]} or {allowed_roots[1]}")


def _is_within(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False
