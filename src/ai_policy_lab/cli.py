from __future__ import annotations

import json
import logging
import os
import re
from datetime import date, datetime
from pathlib import Path
from tempfile import gettempdir
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
_MODEL_NAME_RE = re.compile(r"^[a-zA-Z0-9._:/-]{1,128}$")


@app.callback()
def main() -> None:
    """AI Policy Research Lab CLI."""


class _StateEncoder(json.JSONEncoder):
    def default(self, obj: object) -> object:
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


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
    allow_mock: bool = typer.Option(
        False,
        "--allow-mock",
        help="Explicitly allow mock-mode validation output instead of a live research run.",
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Enable verbose logging."),
    quiet: bool = typer.Option(False, "--quiet", help="Reduce log output."),
) -> None:
    if not question.strip():
        raise typer.BadParameter("Research question must not be empty.")
    if quality_floor not in {"tier_1", "tier_2", "tier_3"}:
        raise typer.BadParameter("quality floor must be one of: tier_1, tier_2, tier_3")
    if verbose and quiet:
        raise typer.BadParameter("Choose either --verbose or --quiet, not both.")

    logging.basicConfig(level=logging.DEBUG if verbose else logging.WARNING if quiet else logging.INFO)

    try:
        runtime = ResearchRuntime.from_env()
    except ValueError as exc:
        if allow_mock:
            os.environ["APL_USE_MOCK"] = "true"
            runtime = ResearchRuntime.from_env()
        else:
            raise typer.BadParameter(
                str(exc)
                + " Provide your own LLM credentials or local endpoint settings. "
                "For local Ollama, set OPENAI_BASE_URL=http://localhost:11434/v1, "
                "OPENAI_API_KEY=ollama, and choose a local model with --model."
            ) from exc

    if runtime.settings.use_mock and not allow_mock:
        raise typer.BadParameter(
            "Mock mode is disabled by default for research runs. "
            "Provide your own LLM credentials or local endpoint settings, or rerun with --allow-mock "
            "for clearly labeled validation output."
        )

    if model:
        if not _MODEL_NAME_RE.match(model):
            raise typer.BadParameter("Model name contains invalid characters.")
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
    state_path.write_text(json.dumps(state, indent=2, cls=_StateEncoder), encoding="utf-8")

    mode_label = "mock" if runtime.settings.use_mock else "live"
    mode_summary = (
        "THIS RUN USED EXPLICIT MOCK MODE.\nTreat every output as a validation-only draft."
        if runtime.settings.use_mock
        else "This run used live LLM generation and live retrieval where connectors were available."
    )
    console.print(
        Panel.fit(
            f"{mode_summary}\n\nRun completed in [bold]{mode_label}[/bold] mode.\n"
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
    allowed_roots = [
        Path.cwd().resolve(),
        runs_dir.resolve(),
        Path(os.sep, "tmp").resolve(),
        Path(gettempdir()).resolve(),
    ]
    if any(_is_within(resolved, root) for root in allowed_roots):
        return resolved
    raise typer.BadParameter(
        "Output dir must be within the current workspace, the configured runs directory, "
        "or the system temporary directory."
    )


def _is_within(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False
