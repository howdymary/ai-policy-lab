from pathlib import Path

from ai_policy_lab.config import Settings
from ai_policy_lab.research_tracks import (
    discover_great_reallocation_data,
    discover_great_reallocation_literature,
    get_great_reallocation_subquestions,
    is_great_reallocation_question,
)


class FakeBLSConnector:
    def timeseries(self, *, series_ids: list[str], start_year: int, end_year: int):
        assert series_ids
        assert start_year == 2015
        assert end_year == 2025
        return {
            "Results": {
                "series": [
                    {"seriesID": "LNS14000000", "data": [{"value": "4.2"}]},
                    {"seriesID": "CES0000000001", "data": [{"value": "158432"}]},
                    {"seriesID": "CES0500000001", "data": [{"value": "135083"}]},
                    {"seriesID": "JTS000000000000000JOL", "data": [{"value": "6550"}]},
                ]
            }
        }


class FakeCensusConnector:
    def dataset(
        self,
        *,
        dataset: str,
        year: int,
        variables: list[str],
        for_clause: str,
        in_clause: str | None = None,
        predicates: dict[str, str] | None = None,
    ):
        assert year == 2023
        if dataset == "acs/acs1":
            return [
                [
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
                    "metropolitan statistical area/micropolitan statistical area",
                ],
                [
                    "San Jose-Sunnyvale-Santa Clara, CA Metro Area",
                    "1000000",
                    "150000",
                    "150000",
                    "90000",
                    "70000",
                    "100000",
                    "120000",
                    "50000",
                    "60000",
                    "41940",
                ],
                [
                    "Austin-Round Rock-San Marcos, TX Metro Area",
                    "900000",
                    "120000",
                    "130000",
                    "60000",
                    "50000",
                    "120000",
                    "130000",
                    "60000",
                    "70000",
                    "12420",
                ],
            ]
        return [
            [
                "NAME",
                "ESTAB",
                "EMP",
                "PAYANN",
                "NAICS2017",
                "EMPSZES",
                "metropolitan statistical area/micropolitan statistical area",
            ],
            [
                "New York-Newark-Jersey City, NY-NJ Metro Area",
                "10000",
                "500000",
                "999999",
                "51",
                "001",
                "35620",
            ],
            [
                "San Jose-Sunnyvale-Santa Clara, CA Metro Area",
                "8000",
                "300000",
                "888888",
                "51",
                "001",
                "41940",
            ],
        ]


class FakeCrossrefConnector:
    def works(self, *, query_title: str, rows: int = 5):
        return {
            "message": {
                "items": [
                    {
                        "title": [query_title],
                        "DOI": "10.1234/example",
                        "container-title": ["Journal or Working Paper"],
                        "published": {"date-parts": [[2020]]},
                    }
                ]
            }
        }


def _settings() -> Settings:
    return Settings(
        use_mock=True,
        runs_dir=Path("runs"),
        default_model="qwopus-q4km",
        default_temperature=0.2,
        openai_base_url="http://localhost:11434/v1",
        openai_api_key="ollama",
        http_timeout_seconds=30.0,
        bls_api_key=None,
        fred_api_key=None,
        census_api_key=None,
        onet_username=None,
        onet_password=None,
        semantic_scholar_api_key=None,
        web_search_api_key=None,
    )


def test_flagship_question_detection_and_subquestions() -> None:
    assert is_great_reallocation_question(
        "How is AI adoption reshaping the occupational structure of the U.S. labor market?"
    )
    subquestions = get_great_reallocation_subquestions()
    assert len(subquestions) == 6
    assert subquestions[0]["priority"] == "primary"


def test_live_data_discovery_can_use_fake_connectors() -> None:
    result = discover_great_reallocation_data(
        settings=_settings(),
        use_live_lookup=True,
        bls=FakeBLSConnector(),
        census=FakeCensusConnector(),
        fred=None,
    )

    assert result.datasets
    assert result.sources
    assert "BLS live retrieval succeeded" in result.summary
    assert "ACS metro retrieval succeeded" in result.summary
    assert "CBP information-sector retrieval succeeded" in result.summary


def test_literature_discovery_enriches_anchor_sources() -> None:
    result = discover_great_reallocation_literature(
        settings=_settings(),
        use_live_lookup=True,
        crossref=FakeCrossrefConnector(),
    )

    assert result.sources
    assert "Structured literature review" in result.summary
    assert any(source["url"].startswith("https://doi.org/10.1234/example") for source in result.sources)
