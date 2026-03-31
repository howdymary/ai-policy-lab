from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from ai_policy_lab.catalog import default_dataset_catalog
from ai_policy_lab.config import Settings
from ai_policy_lab.connectors import BLSConnector, CensusConnector, CrossrefConnector, FREDConnector
from ai_policy_lab.state import DatasetRecord, ResearchQuestion, SourceRecord
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


_FLAGSHIP_QUESTION = (
    "how is ai adoption reshaping the occupational structure of the u.s. labor market"
)


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
