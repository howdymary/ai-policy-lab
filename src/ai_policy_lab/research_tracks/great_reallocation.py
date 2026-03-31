from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ai_policy_lab.catalog import default_dataset_catalog
from ai_policy_lab.config import Settings
from ai_policy_lab.connectors import (
    BLSConnector,
    CensusConnector,
    CrossrefConnector,
    FREDConnector,
    ONETConnector,
)
from ai_policy_lab.state import DatasetRecord, Finding, ResearchQuestion, SourceRecord
from ai_policy_lab.utils import compact_whitespace


class BLSLike(Protocol):
    def timeseries(
        self,
        *,
        series_ids: list[str],
        start_year: int,
        end_year: int,
    ) -> dict[str, Any]: ...


class CensusLike(Protocol):
    def dataset(
        self,
        *,
        dataset: str,
        year: int,
        variables: list[str],
        for_clause: str,
        in_clause: str | None = None,
        predicates: dict[str, str] | None = None,
    ) -> Any: ...


class FREDLike(Protocol):
    def series_observations(
        self,
        *,
        series_id: str,
        observation_start: str | None = None,
        observation_end: str | None = None,
    ) -> dict[str, Any]: ...


class CrossrefLike(Protocol):
    def works(self, *, query_title: str, rows: int = 5) -> dict[str, Any]: ...


class ONETLike(Protocol):
    def latest_text_release(self) -> str: ...

    def public_text_file(self, *, filename: str) -> str: ...


@dataclass(slots=True)
class DataDiscoveryResult:
    datasets: list[DatasetRecord]
    sources: list[SourceRecord]
    summary: str
    issues: list[str]


@dataclass(slots=True)
class LiteratureDiscoveryResult:
    sources: list[SourceRecord]
    summary: str
    issues: list[str]


@dataclass(slots=True)
class MetroExposureAnalysisResult:
    findings: list[Finding]
    quantitative_results: dict[str, Any]
    methodology_description: str
    issues: list[str]


_FLAGSHIP_QUESTION = (
    "how is ai adoption reshaping the occupational structure of the u.s. labor market"
)
_METRO_GEOID_FIELD = "metropolitan statistical area/micropolitan statistical area"
_ACS_BROAD_OCCUPATION_VARIABLES = {
    "management_business_science_arts": ("C24010_003E", "C24010_039E"),
    "service": ("C24010_019E", "C24010_055E"),
    "sales_office": ("C24010_027E", "C24010_063E"),
    "natural_resources_construction_maintenance": ("C24010_030E", "C24010_066E"),
    "production_transport_material": ("C24010_034E", "C24010_070E"),
}
_ACS_BROAD_OCCUPATION_LABELS = {
    "management_business_science_arts": "management, business, science, arts, and health professional work",
    "service": "service work",
    "sales_office": "sales and office work",
    "natural_resources_construction_maintenance": "natural resources, construction, and maintenance work",
    "production_transport_material": "production, transportation, and material-moving work",
}
_SOC_TO_ACS_BROAD_CATEGORY = {
    "11": "management_business_science_arts",
    "13": "management_business_science_arts",
    "15": "management_business_science_arts",
    "17": "management_business_science_arts",
    "19": "management_business_science_arts",
    "21": "management_business_science_arts",
    "23": "management_business_science_arts",
    "25": "management_business_science_arts",
    "27": "management_business_science_arts",
    "29": "management_business_science_arts",
    "31": "service",
    "33": "service",
    "35": "service",
    "37": "service",
    "39": "service",
    "41": "sales_office",
    "43": "sales_office",
    "45": "natural_resources_construction_maintenance",
    "47": "natural_resources_construction_maintenance",
    "49": "natural_resources_construction_maintenance",
    "51": "production_transport_material",
    "53": "production_transport_material",
}
_ONET_SKILL_TARGETS = {
    "Reading Comprehension",
    "Writing",
    "Critical Thinking",
    "Active Learning",
    "Complex Problem Solving",
    "Judgment and Decision Making",
    "Systems Analysis",
}
_ONET_AI_KEYWORDS = (
    "artificial intelligence",
    "machine learning",
    "deep learning",
    "neural",
    "natural language",
    "large language model",
    "llm",
    "generative ai",
    "genai",
    "computer vision",
    "predictive analytics",
    "data science",
    "chatgpt",
    "gpt-",
    "tensorflow",
    "pytorch",
    "vertex ai",
    "amazon bedrock",
    "azure ai",
    "openai",
)
_ONET_CACHE_FILENAMES = {
    "occupations": "Occupation Data.txt",
    "skills": "Skills.txt",
    "technology_skills": "Technology Skills.txt",
}
_ANALYSIS_MIN_EMPLOYED = 250_000


def is_great_reallocation_question(root_question: str) -> bool:
    normalized = root_question.lower()
    keywords = [
        "ai adoption",
        "occupational structure",
        "labor market",
        "metropolitan",
        "education levels",
        "distributional consequences",
    ]
    return _FLAGSHIP_QUESTION in normalized or sum(term in normalized for term in keywords) >= 3


def get_great_reallocation_subquestions() -> list[ResearchQuestion]:
    return [
        {
            "id": "rq-1",
            "question": (
                "Which occupations have the highest exposure to AI-driven task automation, "
                "and how does this differ from previous waves of computerization?"
            ),
            "parent_question": None,
            "status": "pending",
            "priority": "primary",
            "assigned_to": ["literature_review", "quantitative_analyst"],
        },
        {
            "id": "rq-2",
            "question": (
                "What does job postings and skill-taxonomy evidence suggest about changing "
                "skill demand in AI-exposed occupations?"
            ),
            "parent_question": None,
            "status": "pending",
            "priority": "primary",
            "assigned_to": ["data_scout", "quantitative_analyst"],
        },
        {
            "id": "rq-3",
            "question": (
                "Which metropolitan areas are most and least exposed, and how does that "
                "interact with industrial composition and economic complexity?"
            ),
            "parent_question": None,
            "status": "pending",
            "priority": "primary",
            "assigned_to": ["data_scout", "economic_complexity"],
        },
        {
            "id": "rq-4",
            "question": (
                "What occupational transition pathways exist for workers in high-exposure "
                "occupations, and how do transition costs vary across workers?"
            ),
            "parent_question": None,
            "status": "pending",
            "priority": "primary",
            "assigned_to": ["literature_review", "economic_complexity", "political_economy"],
        },
        {
            "id": "rq-5",
            "question": (
                "Which policy interventions have evidence of effectiveness for analogous "
                "transitions, and how applicable are they to AI-driven displacement?"
            ),
            "parent_question": None,
            "status": "pending",
            "priority": "secondary",
            "assigned_to": ["policy_scanner", "political_economy", "literature_review"],
        },
        {
            "id": "rq-6",
            "question": (
                "How do AI adoption patterns differ between large firms and SMEs, and what "
                "does that imply for labor market concentration and worker bargaining power?"
            ),
            "parent_question": None,
            "status": "pending",
            "priority": "exploratory",
            "assigned_to": ["data_scout", "political_economy"],
        },
    ]


def discover_great_reallocation_data(
    *,
    settings: Settings,
    use_live_lookup: bool,
    bls: BLSLike | None = None,
    census: CensusLike | None = None,
    fred: FREDLike | None = None,
) -> DataDiscoveryResult:
    datasets = default_dataset_catalog()
    sources = _official_data_sources()
    issues: list[str] = []
    notes: list[str] = []

    if not use_live_lookup:
        notes.append(
            "Mock mode uses the canonical dataset inventory for the Great Reallocation "
            "question without live API retrieval."
        )
        issues.append(
            "INFO: Live BLS, Census, and FRED retrieval is disabled in mock mode."
        )
        return DataDiscoveryResult(
            datasets=datasets,
            sources=sources,
            summary=_compose_data_summary(notes, issues),
            issues=issues,
        )

    bls_client = bls or BLSConnector(settings)
    census_client = census or CensusConnector(settings)
    fred_client = fred or FREDConnector(settings)

    try:
        bls_response = bls_client.timeseries(
            series_ids=[
                "LNS14000000",
                "CES0000000001",
                "CES0500000001",
                "JTS000000000000000JOL",
            ],
            start_year=2015,
            end_year=2025,
        )
        bls_snapshot = _parse_bls_snapshot(bls_response)
        _append_quality_note(
            datasets,
            "bls-jolts",
            "Live BLS API check: "
            f"latest unemployment {bls_snapshot['unemployment_rate']}, "
            f"job openings {bls_snapshot['job_openings']}, "
            f"private payroll employment {bls_snapshot['private_employment']}.",
        )
        notes.append(
            "BLS live retrieval succeeded. Latest available observations show "
            f"unemployment at {bls_snapshot['unemployment_rate']}, "
            f"total nonfarm employment at {bls_snapshot['total_nonfarm_employment']}, "
            f"private employment at {bls_snapshot['private_employment']}, and "
            f"job openings at {bls_snapshot['job_openings']}."
        )
    except Exception as exc:  # noqa: BLE001
        issues.append(f"WARNING: BLS live retrieval failed: {exc}")

    try:
        acs_rows = census_client.dataset(
            dataset="acs/acs1",
            year=2023,
            variables=[
                "NAME",
                "C24010_001E",
                "C24010_003E",
                "C24010_039E",
                "C24010_008E",
                "C24010_044E",
                "C24010_019E",
                "C24010_055E",
                "C24010_034E",
                "C24010_070E",
            ],
            for_clause="metropolitan statistical area/micropolitan statistical area:*",
        )
        acs_snapshot = _parse_acs_metro_snapshot(acs_rows)
        _append_quality_note(
            datasets,
            "census-acs",
            "Live ACS check: "
            f"{acs_snapshot['coverage_count']} metro/micro areas returned; "
            f"top computer/math share metros include {', '.join(acs_snapshot['top_metros'])}.",
        )
        notes.append(
            "ACS metro retrieval succeeded. The 2023 one-year file returned "
            f"{acs_snapshot['coverage_count']} metro and micro areas. Among labor markets "
            f"with at least 100,000 employed residents, the highest computer/math shares "
            f"appear in {', '.join(acs_snapshot['top_metros'])}."
        )
    except Exception as exc:  # noqa: BLE001
        issues.append(f"WARNING: Census ACS retrieval failed: {exc}")

    try:
        cbp_rows = census_client.dataset(
            dataset="cbp",
            year=2023,
            variables=["NAME", "ESTAB", "EMP", "PAYANN"],
            for_clause="metropolitan statistical area/micropolitan statistical area:*",
            predicates={"NAICS2017": "51", "EMPSZES": "001"},
        )
        cbp_snapshot = _parse_cbp_information_snapshot(cbp_rows)
        _append_quality_note(
            datasets,
            "census-cbp",
            "Live CBP check: "
            f"{cbp_snapshot['coverage_count']} metro/micro information-sector rows returned; "
            f"largest information-sector employment appears in {', '.join(cbp_snapshot['top_metros'])}.",
        )
        notes.append(
            "CBP information-sector retrieval succeeded. The 2023 file returned "
            f"{cbp_snapshot['coverage_count']} metro and micro rows for NAICS 51; "
            f"the largest information-sector labor markets are {', '.join(cbp_snapshot['top_metros'])}."
        )
    except Exception as exc:  # noqa: BLE001
        issues.append(f"WARNING: Census CBP retrieval failed: {exc}")

    if settings.fred_api_key:
        try:
            fred_response = fred_client.series_observations(
                series_id="UNRATE",
                observation_start="2015-01-01",
            )
            latest_unrate = _parse_latest_fred_observation(fred_response)
            _append_quality_note(
                datasets,
                "fred-labor",
                f"Live FRED API check: latest UNRATE observation is {latest_unrate}.",
            )
            notes.append(
                "FRED live retrieval succeeded. The latest UNRATE observation available via FRED is "
                f"{latest_unrate}."
            )
        except Exception as exc:  # noqa: BLE001
            issues.append(f"WARNING: FRED live retrieval failed: {exc}")
    else:
        issues.append(
            "NOTE: FRED_API_KEY is not configured, so FRED macro overlays were cataloged but not fetched live."
        )

    notes.append(
        "For the Great Reallocation question, the strongest current public-data path is "
        "O*NET plus ACS and OEWS for occupational structure, ACS plus CBP for metro variation, "
        "and JOLTS/FRED for macro labor-market context. Public data remains weakest on firm-level "
        "AI adoption, SME deployment, and worker-level transition costs."
    )

    return DataDiscoveryResult(
        datasets=datasets,
        sources=sources,
        summary=_compose_data_summary(notes, issues),
        issues=issues,
    )


def discover_great_reallocation_literature(
    *,
    settings: Settings,
    use_live_lookup: bool,
    crossref: CrossrefLike | None = None,
) -> LiteratureDiscoveryResult:
    sources = _great_reallocation_anchor_sources()
    issues: list[str] = []

    if use_live_lookup:
        crossref_client = crossref or CrossrefConnector(settings)
        sources = _enrich_sources_with_crossref(sources, crossref_client, issues)
    else:
        issues.append("INFO: Literature metadata enrichment is disabled in mock mode.")

    sources = _dedupe_sources(sources)
    summary = _compose_literature_summary(sources)
    return LiteratureDiscoveryResult(sources=sources, summary=summary, issues=issues)


def analyze_great_reallocation_exposure(
    *,
    settings: Settings,
    use_live_lookup: bool,
    census: CensusLike | None = None,
    onet: ONETLike | None = None,
) -> MetroExposureAnalysisResult:
    if not use_live_lookup:
        return MetroExposureAnalysisResult(
            findings=[],
            quantitative_results={
                "status": "not_executed",
                "mode": "mock",
                "analysis": "great_reallocation_metro_exposure_v1",
            },
            methodology_description=(
                "Mock mode did not execute the metro exposure index. In live mode, this step combines "
                "public O*NET occupation signals with ACS metro occupation shares and a CBP "
                "information-sector overlay."
            ),
            issues=["INFO: Live O*NET plus ACS/CBP exposure analysis is disabled in mock mode."],
        )

    issues: list[str] = []
    census_client = census or CensusConnector(settings)
    onet_client = onet or ONETConnector(settings)

    try:
        category_scores, onet_metadata = _compute_onet_broad_category_scores(settings, onet_client)
    except Exception as exc:  # noqa: BLE001
        return MetroExposureAnalysisResult(
            findings=[],
            quantitative_results={
                "status": "failed",
                "mode": "live",
                "analysis": "great_reallocation_metro_exposure_v1",
            },
            methodology_description=(
                "The intended descriptive index uses O*NET skills and technology-skill references, "
                "ACS occupational structure, and CBP information-sector employment."
            ),
            issues=[f"WARNING: O*NET public-text analysis failed: {exc}"],
        )

    acs_rows = census_client.dataset(
        dataset="acs/acs1",
        year=2023,
        variables=[
            "NAME",
            "C24010_001E",
            *[variable for pair in _ACS_BROAD_OCCUPATION_VARIABLES.values() for variable in pair],
        ],
        for_clause="metropolitan statistical area/micropolitan statistical area:*",
    )
    metros = _parse_acs_metro_exposure_rows(acs_rows, category_scores)
    large_metros = [metro for metro in metros if metro["total_employed"] >= _ANALYSIS_MIN_EMPLOYED]
    if not large_metros:
        return MetroExposureAnalysisResult(
            findings=[],
            quantitative_results={
                "status": "failed",
                "mode": "live",
                "analysis": "great_reallocation_metro_exposure_v1",
            },
            methodology_description=(
                "The metro exposure index could not be completed because no ACS metro observations passed "
                "the large-metro employment threshold."
            ),
            issues=["WARNING: ACS metro exposure analysis returned no sufficiently large metro areas."],
        )

    cbp_rows = census_client.dataset(
        dataset="cbp",
        year=2023,
        variables=["NAME", "EMP"],
        for_clause="metropolitan statistical area/micropolitan statistical area:*",
        predicates={"NAICS2017": "51", "EMPSZES": "001"},
    )
    info_employment = _parse_cbp_information_jobs(cbp_rows)
    metros_with_capability = _merge_information_capability(large_metros, info_employment)

    top_exposure = sorted(
        metros_with_capability,
        key=lambda metro: metro["exposure_score"],
        reverse=True,
    )[:5]
    lowest_exposure = sorted(
        metros_with_capability,
        key=lambda metro: metro["exposure_score"],
    )[:5]
    exposure_leaders = top_exposure[:]
    adaptation_ready = _high_exposure_capability_subset(exposure_leaders, descending=True)
    adaptation_codes = {metro["metro_code"] for metro in adaptation_ready}
    transition_candidates = [
        metro for metro in exposure_leaders if metro["metro_code"] not in adaptation_codes
    ]
    transition_pressure = _high_exposure_capability_subset(
        transition_candidates or exposure_leaders,
        descending=False,
    )

    findings = [
        _finding(
            agent="quantitative_analyst",
            claim=(
                "A first descriptive O*NET/ACS metro exposure index places "
                f"{_format_metro_series(top_exposure)} among the most AI-exposed large metro labor markets "
                "in the current public-data baseline."
            ),
            evidence_strength="moderate",
            supporting_sources=["src-onet", "src-census-acs"],
            confidence=0.69,
            methodology=(
                "Occupation-level O*NET cognitive and AI-technology signals were aggregated to broad ACS "
                "occupation groups and then weighted by 2023 ACS metro occupational composition."
            ),
            limitations=[
                "This is a descriptive composite index, not a direct measure of realized AI adoption or displacement.",
                "ACS broad occupation groups are coarser than the underlying O*NET occupation taxonomy.",
            ],
        ),
        _finding(
            agent="quantitative_analyst",
            claim=(
                "The lowest-scoring large metros in this baseline are "
                f"{_format_metro_series(lowest_exposure)}, reflecting a labor mix tilted more toward "
                "service, construction, production, and transportation work than toward high-cognitive "
                "professional occupations."
            ),
            evidence_strength="moderate",
            supporting_sources=["src-onet", "src-census-acs"],
            confidence=0.66,
            methodology=(
                "Large metros were ranked by the same ACS-weighted O*NET exposure index and compared at the "
                "bottom of the distribution."
            ),
            limitations=[
                "Low index scores do not imply immunity from labor-market disruption; they only indicate a different occupational mix.",
                "The measure captures potential exposure and complementarity, not wage or employment outcomes.",
            ],
        ),
    ]

    if adaptation_ready and transition_pressure:
        findings.append(
            _finding(
                agent="economic_complexity",
                claim=(
                    "Within the high-exposure metro cohort, "
                    f"{_format_metro_series(adaptation_ready)} pair elevated occupational exposure with deeper "
                    "information-sector employment intensity, while "
                    f"{_format_metro_series(transition_pressure)} look more exposed than locally capability-rich "
                    "on this first CBP overlay."
                ),
                evidence_strength="suggestive",
                supporting_sources=["src-onet", "src-census-acs", "src-census-cbp"],
                confidence=0.62,
                methodology=(
                    "High-exposure metros were compared on a CBP NAICS 51 employment-intensity overlay to "
                    "approximate local information-economy capability."
                ),
                limitations=[
                    "CBP information employment is an imperfect proxy for the broader capability stack used in full economic-complexity models.",
                    "This overlay does not yet compute RCA, industry proximity, or occupational transition networks.",
                ],
            )
        )
        issues.append(
            "NOTE: The place-based capability overlay currently uses CBP information-sector intensity as a proxy; full ECI/RCA network metrics remain a future step."
        )

    methodology_description = (
        "Executed a live descriptive metro exposure index for the Great Reallocation question. The pipeline "
        "downloads the current public O*NET text release, computes occupation-level signals from selected "
        "cognitive and decision-intensive skills plus AI-related technology mentions, aggregates those "
        "signals to broad ACS occupation groups, and weights them by 2023 ACS metro occupational structure. "
        "A CBP NAICS 51 employment overlay approximates local information-economy capability. This run is "
        "descriptive only: it does not identify causal effects, estimate standard errors, or directly measure "
        "firm-level AI adoption."
    )

    quantitative_results: dict[str, Any] = {
        "status": "completed",
        "mode": "live",
        "analysis": "great_reallocation_metro_exposure_v1",
        "analysis_type": "descriptive_index",
        "coverage": {
            "acs_large_metros": len(large_metros),
            "acs_total_metros": len(metros),
            "cbp_info_metros": len(info_employment),
            "onet_release": onet_metadata["release"],
            "onet_scored_occupations": onet_metadata["occupation_count"],
        },
        "broad_category_scores": {
            category: round(score, 4) for category, score in sorted(category_scores.items())
        },
        "top_exposure_metros": _compact_ranked_metros(top_exposure),
        "lowest_exposure_metros": _compact_ranked_metros(lowest_exposure),
        "adaptation_ready_metros": _compact_ranked_metros(adaptation_ready),
        "transition_pressure_metros": _compact_ranked_metros(transition_pressure),
    }
    return MetroExposureAnalysisResult(
        findings=findings,
        quantitative_results=quantitative_results,
        methodology_description=methodology_description,
        issues=issues,
    )


def _parse_bls_snapshot(response: dict[str, Any]) -> dict[str, str]:
    mapping = {
        "LNS14000000": "unemployment_rate",
        "CES0000000001": "total_nonfarm_employment",
        "CES0500000001": "private_employment",
        "JTS000000000000000JOL": "job_openings",
    }
    parsed: dict[str, str] = {}
    series_list = response.get("Results", {}).get("series", [])
    for series in series_list:
        series_id = str(series.get("seriesID"))
        for point in series.get("data", []):
            value = str(point.get("value", "")).strip()
            if value and value != "-":
                parsed[mapping[series_id]] = (
                    f"{value}%"
                    if series_id == "LNS14000000"
                    else f"{value} thousand"
                )
                break
    return parsed


def _parse_acs_metro_snapshot(rows: Any) -> dict[str, Any]:
    header = rows[0]
    entries = [dict(zip(header, row, strict=True)) for row in rows[1:]]
    enriched = []
    for row in entries:
        total = _safe_int(row["C24010_001E"])
        if total <= 0 or "Metro Area" not in row["NAME"]:
            continue
        mgmt = _safe_int(row["C24010_003E"]) + _safe_int(row["C24010_039E"])
        comp = _safe_int(row["C24010_008E"]) + _safe_int(row["C24010_044E"])
        service = _safe_int(row["C24010_019E"]) + _safe_int(row["C24010_055E"])
        production = _safe_int(row["C24010_034E"]) + _safe_int(row["C24010_070E"])
        enriched.append(
            {
                "name": row["NAME"],
                "total": total,
                "management_share": mgmt / total,
                "computer_share": comp / total,
                "service_share": service / total,
                "production_share": production / total,
            }
        )

    filtered = [item for item in enriched if item["total"] >= 100_000]
    top = sorted(filtered, key=lambda item: item["computer_share"], reverse=True)[:3]
    return {
        "coverage_count": len(rows) - 1,
        "top_metros": [
            f"{item['name']} ({item['computer_share']:.1%} computer/math share)" for item in top
        ],
    }


def _parse_cbp_information_snapshot(rows: Any) -> dict[str, Any]:
    header = rows[0]
    entries = [dict(zip(header, row, strict=True)) for row in rows[1:]]
    metros = [item for item in entries if "Metro Area" in item["NAME"]]
    ranked = sorted(metros, key=lambda item: _safe_int(item["EMP"]), reverse=True)[:3]
    return {
        "coverage_count": len(rows) - 1,
        "top_metros": [
            f"{item['NAME']} ({_safe_int(item['EMP']):,} information jobs)" for item in ranked
        ],
    }


def _parse_latest_fred_observation(response: dict[str, Any]) -> str:
    observations = response.get("observations", [])
    for item in reversed(observations):
        value = str(item.get("value", "")).strip()
        if value and value != ".":
            return value
    return "unavailable"


def _append_quality_note(datasets: list[DatasetRecord], dataset_id: str, note: str) -> None:
    for dataset in datasets:
        if dataset["id"] == dataset_id:
            dataset["quality_notes"] = f"{dataset['quality_notes']} {note}".strip()
            return


def _official_data_sources() -> list[SourceRecord]:
    return [
        _source(
            identifier="src-bls-oes",
            title="Occupational Employment and Wage Statistics",
            authors=["Bureau of Labor Statistics"],
            publication="Bureau of Labor Statistics",
            date="2026-03-30",
            url="https://www.bls.gov/oes/",
            source_tier="tier_1",
            source_type="government_data",
            notes="Official BLS occupation-by-place employment and wage program.",
        ),
        _source(
            identifier="src-bls-jolts",
            title="Job Openings and Labor Turnover Survey",
            authors=["Bureau of Labor Statistics"],
            publication="Bureau of Labor Statistics",
            date="2026-03-30",
            url="https://www.bls.gov/jlt/",
            source_tier="tier_1",
            source_type="government_data",
            notes="Official BLS labor demand and churn dataset.",
        ),
        _source(
            identifier="src-census-acs",
            title="American Community Survey",
            authors=["U.S. Census Bureau"],
            publication="U.S. Census Bureau",
            date="2026-03-30",
            url="https://www.census.gov/programs-surveys/acs",
            source_tier="tier_1",
            source_type="government_data",
            notes="Core demographic and occupation microdata program.",
        ),
        _source(
            identifier="src-census-cbp",
            title="County Business Patterns",
            authors=["U.S. Census Bureau"],
            publication="U.S. Census Bureau",
            date="2026-03-30",
            url="https://www.census.gov/programs-surveys/cbp.html",
            source_tier="tier_1",
            source_type="government_data",
            notes="Annual industry composition data used for place-based capability analysis.",
        ),
        _source(
            identifier="src-fred",
            title="Federal Reserve Economic Data",
            authors=["Federal Reserve Bank of St. Louis"],
            publication="Federal Reserve Bank of St. Louis",
            date="2026-03-30",
            url="https://fred.stlouisfed.org/",
            source_tier="tier_1",
            source_type="government_data",
            notes="Macro overlay and reference time-series platform.",
        ),
        _source(
            identifier="src-onet",
            title="O*NET Database",
            authors=["National Center for O*NET Development"],
            publication="O*NET Resource Center",
            date="2026-03-30",
            url="https://www.onetcenter.org/database.html",
            source_tier="tier_1",
            source_type="government_data",
            notes="Public Department of Labor-sponsored occupation, skills, and technology reference data.",
        ),
    ]


def _great_reallocation_anchor_sources() -> list[SourceRecord]:
    return [
        _source(
            identifier="doi-10.3386/w8337",
            title="The Skill Content of Recent Technological Change: An Empirical Exploration",
            authors=["David Autor", "Frank Levy", "Richard J. Murnane"],
            publication="NBER Working Paper 8337",
            date="2001-06-01",
            url="https://www.nber.org/papers/w8337",
            source_tier="tier_1",
            source_type="preprint",
            notes="Foundational task-based framework; use the later peer-reviewed QJE version when formal citation is needed.",
        ),
        _source(
            identifier="doi-10.1016/j.techfore.2016.08.019",
            title="The future of employment: How susceptible are jobs to computerisation?",
            authors=["Carl Benedikt Frey", "Michael A. Osborne"],
            publication="Technological Forecasting and Social Change",
            date="2017-01-01",
            url="https://doi.org/10.1016/j.techfore.2016.08.019",
            source_tier="tier_1",
            source_type="peer_reviewed",
            notes="Canonical occupation-level exposure benchmark, but heavily debated in later task-based literature.",
        ),
        _source(
            identifier="doi-10.1257/jep.33.2.3",
            title="Automation and New Tasks: How Technology Displaces and Reinstates Labor",
            authors=["Daron Acemoglu", "Pascual Restrepo"],
            publication="Journal of Economic Perspectives",
            date="2019-05-01",
            url="https://doi.org/10.1257/jep.33.2.3",
            source_tier="tier_1",
            source_type="peer_reviewed",
            notes="Key task-displacement and reinstatement framework for interpreting technology shocks.",
        ),
        _source(
            identifier="doi-10.1086/705716",
            title="Robots and Jobs: Evidence from US Labor Markets",
            authors=["Daron Acemoglu", "Pascual Restrepo"],
            publication="Journal of Political Economy",
            date="2020-04-01",
            url="https://doi.org/10.1086/705716",
            source_tier="tier_1",
            source_type="peer_reviewed",
            notes="Important evidence on geographic and distributional heterogeneity in automation shocks.",
        ),
        _source(
            identifier="doi-10.3386/w24196",
            title="Artificial Intelligence, Automation and Work",
            authors=["Daron Acemoglu", "Pascual Restrepo"],
            publication="NBER Working Paper 24196",
            date="2018-01-01",
            url="https://www.nber.org/papers/w24196",
            source_tier="tier_1",
            source_type="preprint",
            notes="Frames AI as a general-purpose technology with distributional consequences that depend on institutions and task design.",
        ),
        _source(
            identifier="src-webb-ai-labor-market",
            title="The Impact of Artificial Intelligence on the Labor Market",
            authors=["Michael Webb"],
            publication="Working paper",
            date="2020-01-01",
            url="https://www.michaelrwebb.com/research",
            source_tier="tier_3",
            source_type="preprint",
            notes="Frequently cited AI exposure measure based on patent-text matching; not peer reviewed in this form.",
        ),
        _source(
            identifier="doi-10.5089/9798400254802.001",
            title="Labor Market Exposure to AI: Cross-country Differences and Distributional Implications",
            authors=["Carlo Pizzinelli", "Antonio Spilimbergo", "Mariana Tavares"],
            publication="IMF Working Papers",
            date="2023-01-01",
            url="https://doi.org/10.5089/9798400254802.001",
            source_tier="tier_2",
            source_type="think_tank",
            notes="Useful for cross-country distributional framing and recent measurement approaches to AI exposure.",
        ),
        _source(
            identifier="src-brookings-moving-up",
            title="Moving up: Promoting workers’ upward mobility using network analysis",
            authors=["Marcela Escobari", "Ian Seyal", "Carlos Daboin Contreras"],
            publication="Brookings Institution",
            date="2021-06-14",
            url="https://www.brookings.edu/articles/moving-up-promoting-workers-upward-mobility-in-a-time-of-change/",
            source_tier="tier_2",
            source_type="think_tank",
            notes="Core Brookings mobility-pathways report using network analysis and transition data.",
        ),
        _source(
            identifier="src-brookings-growing-cities",
            title="Growing Cities that Work for All",
            authors=["Marcela Escobari", "Jose Morales-Arilla", "Ian Seyal"],
            publication="Brookings Institution",
            date="2019-05-01",
            url="https://www.brookings.edu/wp-content/uploads/2019/05/GrowingCitiesthatWorkforAll-FINALforWeb.pdf",
            source_tier="tier_2",
            source_type="think_tank",
            notes="Capability-based industrial development framing from the Workforce of the Future initiative.",
        ),
        _source(
            identifier="src-brookings-workforce-future",
            title="Workforce of the Future Initiative",
            authors=["Brookings Institution"],
            publication="Brookings Institution",
            date="2019-01-01",
            url="https://www.brookings.edu/wp-content/uploads/2019/01/Future-of-the-Workforce-Initiative1.pdf",
            source_tier="tier_2",
            source_type="think_tank",
            notes="Describes the industry-space and mobility toolkit agenda behind Brookings’ place-based labor market research.",
        ),
    ]


def _enrich_sources_with_crossref(
    sources: list[SourceRecord],
    crossref: CrossrefLike,
    issues: list[str],
) -> list[SourceRecord]:
    enriched: list[SourceRecord] = []
    for source in sources:
        if not source["id"].startswith(("doi-", "src-brookings", "src-webb", "src-")):
            enriched.append(source)
            continue
        try:
            result = crossref.works(query_title=source["title"], rows=3)
            items = result.get("message", {}).get("items", [])
            best = _pick_best_crossref_match(source["title"], items)
            if best is None:
                enriched.append(source)
                continue
            container = (best.get("container-title") or [source["publication"]])[0]
            date_parts = (
                best.get("published-print")
                or best.get("published-online")
                or best.get("published")
                or {}
            ).get("date-parts", [[source["date"][:4]]])[0]
            year = str(date_parts[0]) if date_parts else source["date"][:4]
            url = f"https://doi.org/{best['DOI']}" if best.get("DOI") else source["url"]
            enriched.append(
                {
                    **source,
                    "publication": container or source["publication"],
                    "date": f"{year}-01-01",
                    "url": url,
                }
            )
        except Exception as exc:  # noqa: BLE001
            issues.append(f"NOTE: Crossref enrichment failed for '{source['title']}': {exc}")
            enriched.append(source)
    return enriched


def _pick_best_crossref_match(title: str, items: list[dict[str, Any]]) -> dict[str, Any] | None:
    normalized_target = compact_whitespace(title).lower()
    for item in items:
        item_title = compact_whitespace((item.get("title") or [""])[0]).lower()
        if item_title == normalized_target:
            return item
    return items[0] if items else None


def _compose_data_summary(notes: list[str], issues: list[str]) -> str:
    body = "\n".join(f"- {note}" for note in notes)
    issue_lines = "\n".join(f"- {issue}" for issue in issues) or "- No issues logged."
    return (
        "Dataset inventory and availability assessment:\n"
        f"{body}\n\n"
        "Current blockers and caveats:\n"
        f"{issue_lines}"
    )


def _compose_literature_summary(sources: list[SourceRecord]) -> str:
    tier_1 = sum(1 for item in sources if item["source_tier"] == "tier_1")
    tier_2 = sum(1 for item in sources if item["source_tier"] == "tier_2")
    tier_3 = sum(1 for item in sources if item["source_tier"] == "tier_3")
    return (
        "Structured literature review for the Great Reallocation question:\n"
        f"- Anchor bibliography assembled with {tier_1} Tier 1, {tier_2} Tier 2, and {tier_3} Tier 3 sources.\n"
        "- Established consensus: task-based research treats automation and AI as shocks to bundles of tasks, "
        "not simple one-for-one occupation elimination.\n"
        "- Established consensus: displacement effects are geographically and distributionally uneven, with "
        "local industry structure and worker mobility pathways shaping adaptation.\n"
        "- Active debate: exposure measurement varies materially across Frey-Osborne style automation scores, "
        "task-based models, patent-text AI exposure measures, and employer adoption data.\n"
        "- Active debate: the balance between complementarity and displacement depends on whether AI creates "
        "new tasks fast enough, and whether institutions translate productivity gains into worker welfare.\n"
        "- Knowledge gaps: metro-level adaptation evidence, worker-level transition costs, SME adoption patterns, "
        "and public datasets linking AI uptake directly to occupational restructuring remain limited."
    )


def _dedupe_sources(sources: list[SourceRecord]) -> list[SourceRecord]:
    deduped: dict[str, SourceRecord] = {}
    for source in sources:
        deduped[source["id"]] = source
    return list(deduped.values())


def _safe_int(value: str | None) -> int:
    if value in (None, "", "null"):
        return 0
    return int(value)


def _compute_onet_broad_category_scores(
    settings: Settings,
    onet: ONETLike,
) -> tuple[dict[str, float], dict[str, Any]]:
    cache_dir = settings.runs_dir / "cache" / "onet" / onet.latest_text_release()
    occupation_text = _load_onet_text(cache_dir, onet, _ONET_CACHE_FILENAMES["occupations"])
    skills_text = _load_onet_text(cache_dir, onet, _ONET_CACHE_FILENAMES["skills"])
    technology_text = _load_onet_text(cache_dir, onet, _ONET_CACHE_FILENAMES["technology_skills"])

    skill_scores: dict[str, dict[str, float]] = {}
    for row in csv.DictReader(io.StringIO(skills_text), delimiter="\t"):
        if row["Scale ID"] != "LV":
            continue
        skill_name = row["Element Name"]
        if skill_name not in _ONET_SKILL_TARGETS:
            continue
        try:
            skill_scores.setdefault(row["O*NET-SOC Code"], {})[skill_name] = float(row["Data Value"])
        except ValueError:
            continue

    ai_pattern = re.compile("|".join(re.escape(keyword) for keyword in _ONET_AI_KEYWORDS), re.IGNORECASE)
    ai_hits: dict[str, int] = {}
    for row in csv.DictReader(io.StringIO(technology_text), delimiter="\t"):
        occupation_code = row["O*NET-SOC Code"]
        haystack = " ".join((row.get("Example", ""), row.get("Commodity Title", "")))
        if ai_pattern.search(haystack):
            ai_hits[occupation_code] = ai_hits.get(occupation_code, 0) + 1

    grouped_scores: dict[str, list[float]] = {category: [] for category in _ACS_BROAD_OCCUPATION_VARIABLES}
    scored_occupations = 0
    for row in csv.DictReader(io.StringIO(occupation_text), delimiter="\t"):
        occupation_code = row["O*NET-SOC Code"]
        soc_major = occupation_code.split("-")[0]
        category = _SOC_TO_ACS_BROAD_CATEGORY.get(soc_major)
        occupation_skills = skill_scores.get(occupation_code)
        if category is None or not occupation_skills:
            continue
        cognitive_score = sum(occupation_skills.values()) / (len(occupation_skills) * 7.0)
        ai_technology_score = min(1.0, ai_hits.get(occupation_code, 0) / 3.0)
        occupation_score = (0.8 * cognitive_score) + (0.2 * ai_technology_score)
        grouped_scores[category].append(occupation_score)
        scored_occupations += 1

    category_scores = {
        category: sum(values) / len(values)
        for category, values in grouped_scores.items()
        if values
    }
    if len(category_scores) != len(_ACS_BROAD_OCCUPATION_VARIABLES):
        missing = sorted(set(_ACS_BROAD_OCCUPATION_VARIABLES) - set(category_scores))
        raise RuntimeError(f"Incomplete O*NET broad-category coverage: {', '.join(missing)}")

    return category_scores, {
        "release": cache_dir.name,
        "occupation_count": scored_occupations,
    }


def _load_onet_text(cache_dir: Path, onet: ONETLike, filename: str) -> str:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / filename
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")
    content = onet.public_text_file(filename=filename)
    cache_path.write_text(content, encoding="utf-8")
    return content


def _parse_acs_metro_exposure_rows(
    rows: Any,
    category_scores: dict[str, float],
) -> list[dict[str, Any]]:
    header = rows[0]
    metros: list[dict[str, Any]] = []
    for raw_row in rows[1:]:
        row = dict(zip(header, raw_row, strict=True))
        if "Metro Area" not in row["NAME"]:
            continue
        total_employed = _safe_int(row["C24010_001E"])
        if total_employed <= 0:
            continue
        category_shares: dict[str, float] = {}
        for category, (male_variable, female_variable) in _ACS_BROAD_OCCUPATION_VARIABLES.items():
            count = _safe_int(row[male_variable]) + _safe_int(row[female_variable])
            category_shares[category] = count / total_employed

        metros.append(
            {
                "name": row["NAME"],
                "metro_code": row[_METRO_GEOID_FIELD],
                "total_employed": total_employed,
                "category_shares": category_shares,
                "exposure_score": sum(
                    category_shares[category] * category_scores[category]
                    for category in _ACS_BROAD_OCCUPATION_VARIABLES
                ),
            }
        )
    return metros


def _parse_cbp_information_jobs(rows: Any) -> dict[str, int]:
    header = rows[0]
    info_jobs: dict[str, int] = {}
    for raw_row in rows[1:]:
        row = dict(zip(header, raw_row, strict=True))
        if "Metro Area" not in row["NAME"]:
            continue
        info_jobs[row[_METRO_GEOID_FIELD]] = _safe_int(row["EMP"])
    return info_jobs


def _merge_information_capability(
    metros: list[dict[str, Any]],
    info_jobs: dict[str, int],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for metro in metros:
        info_employment = info_jobs.get(metro["metro_code"], 0)
        merged.append(
            {
                **metro,
                "information_jobs": info_employment,
                "info_intensity": info_employment / metro["total_employed"] if info_employment else 0.0,
            }
        )
    return merged


def _high_exposure_capability_subset(
    metros: list[dict[str, Any]],
    *,
    descending: bool,
) -> list[dict[str, Any]]:
    ranked = sorted(
        metros,
        key=lambda metro: (metro["info_intensity"], metro["exposure_score"]),
        reverse=descending,
    )
    return ranked[:3]


def _compact_ranked_metros(metros: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": metro["name"],
            "exposure_score": round(metro["exposure_score"], 4),
            "info_intensity": round(metro["info_intensity"], 4),
            "total_employed": metro["total_employed"],
        }
        for metro in metros
    ]


def _format_metro_series(metros: list[dict[str, Any]]) -> str:
    return ", ".join(
        f"{_short_metro_name(metro['name'])} ({metro['exposure_score']:.3f})" for metro in metros[:3]
    )


def _short_metro_name(name: str) -> str:
    return name.replace(" Metro Area", "")


def _finding(
    *,
    agent: str,
    claim: str,
    evidence_strength: str,
    supporting_sources: list[str],
    confidence: float,
    methodology: str,
    limitations: list[str],
) -> Finding:
    return {
        "agent": agent,
        "claim": claim,
        "evidence_strength": evidence_strength,  # type: ignore[typeddict-item]
        "supporting_sources": supporting_sources,
        "confidence": confidence,
        "methodology": methodology,
        "limitations": limitations,
        "contradicts": [],
    }


def _source(
    *,
    identifier: str,
    title: str,
    authors: list[str],
    publication: str,
    date: str,
    url: str,
    source_tier: str,
    source_type: str,
    notes: str,
) -> SourceRecord:
    return {
        "id": identifier,
        "title": title,
        "authors": authors,
        "publication": publication,
        "date": date,
        "url": url,
        "source_tier": source_tier,  # type: ignore[typeddict-item]
        "source_type": source_type,  # type: ignore[typeddict-item]
        "relevance_score": 0.95 if source_tier == "tier_1" else 0.85,
        "conflict_of_interest": None,
        "notes": notes,
    }
