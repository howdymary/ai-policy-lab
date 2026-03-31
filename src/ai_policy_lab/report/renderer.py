from __future__ import annotations

from ai_policy_lab.state import AdversarialReviewItem, ResearchQuestion, ResearchState, SourceRecord


def _render_questions(questions: list[ResearchQuestion]) -> str:
    if not questions:
        return "- No research questions were generated.\n"
    return "".join(
        f"- {item.get('question', 'No question')} ({item.get('priority', 'unknown')})\n"
        for item in questions
    )


def _render_findings(state: ResearchState) -> str:
    if not state["findings"]:
        return (
            "No validated findings have been produced yet. Treat this run as an early research draft "
            "until live retrieval and analysis are available for the question being asked.\n"
        )

    lines = []
    for finding in state["findings"]:
        lines.append(
            f"- {finding.get('claim', 'No claim')} "
            f"[evidence: {finding.get('evidence_strength', 'unknown')}, "
            f"confidence: {float(finding.get('confidence', 0.0)):.2f}]"
        )
    return "\n".join(lines) + "\n"


def _render_datasets(state: ResearchState) -> str:
    if not state["datasets"]:
        return "- No datasets cataloged.\n"
    return "".join(
        f"- {dataset.get('name', 'Unnamed')} ({dataset.get('source_agency', 'Unknown')}) — "
        f"{dataset.get('url', 'N/A')}\n"
        for dataset in state["datasets"]
    )


def _render_adversarial_review(items: list[AdversarialReviewItem]) -> str:
    if not items:
        return "- No adversarial review items were logged.\n"
    lines = []
    for item in items:
        sources = ", ".join(item.get("supporting_sources") or []) or "none"
        lines.append(
            f"- [{item.get('recommendation', 'N/A')}] {item.get('finding_claim', 'N/A')} "
            f"Counterargument: {item.get('counterargument', 'N/A')} "
            f"[counter-evidence: {item.get('evidence_strength', 'unknown')}; sources: {sources}]"
        )
    return "\n".join(lines) + "\n"


def _render_sources(sources: list[SourceRecord]) -> str:
    if not sources:
        return "- No retrieved sources.\n"
    return "".join(
        f"- [{source.get('id', 'unknown')}] {source.get('title', 'Untitled')} "
        f"({source.get('publication', 'Unknown publication')}, {str(source.get('date', ''))[:4]}) "
        f"[{source.get('source_tier', 'unknown')}]\n"
        for source in sources
    )


def render_report(state: ResearchState) -> str:
    executive_summary = state["executive_summary"] or "Executive summary pending."
    literature = state["existing_literature_summary"] or "Literature review pending."
    policy = state["policy_landscape_summary"] or "Policy scan pending."
    methods = state["methodology_description"] or "Methodology plan pending."
    source_audit = state["source_audit_report"] or "Source audit pending."
    methodology_review = state["methodology_review"] or "Methodology review pending."
    limitations = state["flagged_issues"] or ["No issues logged."]
    mock_banner = ""
    if state.get("runtime_mode") == "mock":
        mock_banner = (
            "> WARNING: THIS REPORT WAS GENERATED IN EXPLICIT MOCK MODE. "
            "It is a validation-only draft and must not be treated as live research.\n\n"
        )

    limitation_lines = "".join(f"- {item}\n" for item in limitations)

    return f"""{mock_banner}# AI Policy Research Lab Report

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

## 9. Counterarguments and Rebuttals

{_render_adversarial_review(state["adversarial_review"])}

## 10. Research Questions

{_render_questions(state["research_questions"])}

## 11. Data Appendix

{_render_datasets(state)}

## 12. Source Audit

### Retrieved Sources

{_render_sources(state["citation_list"])}

### Audit Summary

{source_audit}
"""
