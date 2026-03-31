from __future__ import annotations

from ai_policy_lab.state import ResearchQuestion, ResearchState


def _render_questions(questions: list[ResearchQuestion]) -> str:
    if not questions:
        return "- No research questions were generated.\n"
    return "".join(f"- {item['question']} ({item['priority']})\n" for item in questions)


def _render_findings(state: ResearchState) -> str:
    if not state["findings"]:
        return (
            "No validated findings have been produced yet. This run should be treated as a planning "
            "and scaffolding pass until live retrieval and analysis are enabled.\n"
        )

    lines = []
    for finding in state["findings"]:
        lines.append(
            f"- {finding['claim']} "
            f"[evidence: {finding['evidence_strength']}, confidence: {finding['confidence']:.2f}]"
        )
    return "\n".join(lines) + "\n"


def _render_datasets(state: ResearchState) -> str:
    if not state["datasets"]:
        return "- No datasets cataloged.\n"
    return "".join(
        f"- {dataset['name']} ({dataset['source_agency']}) — {dataset['url']}\n"
        for dataset in state["datasets"]
    )


def render_report(state: ResearchState) -> str:
    executive_summary = state["executive_summary"] or "Executive summary pending."
    literature = state["existing_literature_summary"] or "Literature review pending."
    policy = state["policy_landscape_summary"] or "Policy scan pending."
    methods = state["methodology_description"] or "Methodology plan pending."
    source_audit = state["source_audit_report"] or "Source audit pending."
    methodology_review = state["methodology_review"] or "Methodology review pending."
    limitations = state["flagged_issues"] or ["No issues logged."]

    limitation_lines = "".join(f"- {item}\n" for item in limitations)

    return f"""# AI Policy Research Lab Report

## 1. Executive Summary

{executive_summary}

## 2. Introduction and Motivation

Root question: {state["root_question"]}

Constraints: {", ".join(state["domain_constraints"]) if state["domain_constraints"] else "None provided"}

## 3. Data and Methods

{methods}

## 4. Literature Review

{literature}

## 5. Findings

{_render_findings(state)}

## 6. Discussion

{policy}

## 7. Limitations and Future Work

{limitation_lines}

## 8. Methodology Review Summary

{methodology_review}

## 9. Research Questions

{_render_questions(state["research_questions"])}

## 10. Data Appendix

{_render_datasets(state)}

## 11. Source Audit

{source_audit}
"""
