from pathlib import Path

from ai_policy_lab.config import Settings
from ai_policy_lab.research_tracks import (
    analyze_great_reallocation_exposure,
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
            full_rows = [
                {
                    "NAME": "San Jose-Sunnyvale-Santa Clara, CA Metro Area",
                    "C24010_001E": "1000000",
                    "C24010_003E": "330000",
                    "C24010_008E": "160000",
                    "C24010_019E": "35000",
                    "C24010_027E": "70000",
                    "C24010_030E": "30000",
                    "C24010_034E": "50000",
                    "C24010_039E": "290000",
                    "C24010_044E": "80000",
                    "C24010_055E": "55000",
                    "C24010_063E": "90000",
                    "C24010_066E": "10000",
                    "C24010_070E": "40000",
                    "metropolitan statistical area/micropolitan statistical area": "41940",
                },
                {
                    "NAME": "Austin-Round Rock-San Marcos, TX Metro Area",
                    "C24010_001E": "900000",
                    "C24010_003E": "240000",
                    "C24010_008E": "110000",
                    "C24010_019E": "50000",
                    "C24010_027E": "70000",
                    "C24010_030E": "40000",
                    "C24010_034E": "55000",
                    "C24010_039E": "210000",
                    "C24010_044E": "60000",
                    "C24010_055E": "70000",
                    "C24010_063E": "100000",
                    "C24010_066E": "20000",
                    "C24010_070E": "45000",
                    "metropolitan statistical area/micropolitan statistical area": "12420",
                },
                {
                    "NAME": "Bakersfield, CA Metro Area",
                    "C24010_001E": "300000",
                    "C24010_003E": "45000",
                    "C24010_008E": "10000",
                    "C24010_019E": "25000",
                    "C24010_027E": "20000",
                    "C24010_030E": "25000",
                    "C24010_034E": "35000",
                    "C24010_039E": "45000",
                    "C24010_044E": "8000",
                    "C24010_055E": "35000",
                    "C24010_063E": "35000",
                    "C24010_066E": "10000",
                    "C24010_070E": "25000",
                    "metropolitan statistical area/micropolitan statistical area": "12540",
                },
            ]
            header = [*variables, "metropolitan statistical area/micropolitan statistical area"]
            return [
                header,
                *[
                    [row[column] for column in header]
                    for row in full_rows
                ],
            ]
        return [
            [
                "NAME",
                "EMP",
                "metropolitan statistical area/micropolitan statistical area",
            ],
            [
                "New York-Newark-Jersey City, NY-NJ Metro Area",
                "500000",
                "35620",
            ],
            [
                "San Jose-Sunnyvale-Santa Clara, CA Metro Area",
                "300000",
                "41940",
            ],
            [
                "Austin-Round Rock-San Marcos, TX Metro Area",
                "120000",
                "12420",
            ],
            [
                "Bakersfield, CA Metro Area",
                "15000",
                "12540",
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


class FakeONETConnector:
    def latest_text_release(self) -> str:
        return "db_fake_text"

    def public_text_file(self, *, filename: str) -> str:
        files = {
            "Occupation Data.txt": (
                "O*NET-SOC Code\tTitle\tDescription\n"
                "15-1252.00\tSoftware Developers\tBuild software systems.\n"
                "13-2051.00\tFinancial Analysts\tAnalyze financial data.\n"
                "29-1141.00\tRegistered Nurses\tProvide patient care.\n"
                "35-3023.00\tFast Food and Counter Workers\tServe food.\n"
                "41-2011.00\tCashiers\tProcess customer purchases.\n"
                "43-9061.00\tOffice Clerks, General\tPerform clerical duties.\n"
                "47-2111.00\tElectricians\tInstall electrical systems.\n"
                "51-9199.00\tProduction Workers, All Other\tProduction support.\n"
                "53-3032.00\tHeavy and Tractor-Trailer Truck Drivers\tTransport freight.\n"
            ),
            "Skills.txt": (
                "O*NET-SOC Code\tElement ID\tElement Name\tScale ID\tData Value\tN\tStandard Error\tLower CI Bound\tUpper CI Bound\tRecommend Suppress\tNot Relevant\tDate\tDomain Source\n"
                "15-1252.00\t2.A.1.a\tReading Comprehension\tLV\t5.5\t8\t0\t0\t0\tN\tN\t08/2023\tAnalyst\n"
                "15-1252.00\t2.A.1.c\tWriting\tLV\t5.6\t8\t0\t0\t0\tN\tN\t08/2023\tAnalyst\n"
                "15-1252.00\t2.A.2.a\tCritical Thinking\tLV\t5.8\t8\t0\t0\t0\tN\tN\t08/2023\tAnalyst\n"
                "15-1252.00\t2.A.2.b\tActive Learning\tLV\t5.2\t8\t0\t0\t0\tN\tN\t08/2023\tAnalyst\n"
                "15-1252.00\t2.A.2.e\tComplex Problem Solving\tLV\t5.7\t8\t0\t0\t0\tN\tN\t08/2023\tAnalyst\n"
                "15-1252.00\t2.A.2.f\tJudgment and Decision Making\tLV\t4.8\t8\t0\t0\t0\tN\tN\t08/2023\tAnalyst\n"
                "15-1252.00\t2.B.3.e\tSystems Analysis\tLV\t5.4\t8\t0\t0\t0\tN\tN\t08/2023\tAnalyst\n"
                "13-2051.00\t2.A.1.a\tReading Comprehension\tLV\t5.1\t8\t0\t0\t0\tN\tN\t08/2023\tAnalyst\n"
                "13-2051.00\t2.A.1.c\tWriting\tLV\t4.9\t8\t0\t0\t0\tN\tN\t08/2023\tAnalyst\n"
                "13-2051.00\t2.A.2.a\tCritical Thinking\tLV\t5.0\t8\t0\t0\t0\tN\tN\t08/2023\tAnalyst\n"
                "13-2051.00\t2.A.2.b\tActive Learning\tLV\t4.8\t8\t0\t0\t0\tN\tN\t08/2023\tAnalyst\n"
                "13-2051.00\t2.A.2.e\tComplex Problem Solving\tLV\t4.9\t8\t0\t0\t0\tN\tN\t08/2023\tAnalyst\n"
                "13-2051.00\t2.A.2.f\tJudgment and Decision Making\tLV\t5.3\t8\t0\t0\t0\tN\tN\t08/2023\tAnalyst\n"
                "13-2051.00\t2.B.3.e\tSystems Analysis\tLV\t4.4\t8\t0\t0\t0\tN\tN\t08/2023\tAnalyst\n"
                "29-1141.00\t2.A.1.a\tReading Comprehension\tLV\t4.5\t8\t0\t0\t0\tN\tN\t08/2023\tAnalyst\n"
                "29-1141.00\t2.A.1.c\tWriting\tLV\t4.1\t8\t0\t0\t0\tN\tN\t08/2023\tAnalyst\n"
                "29-1141.00\t2.A.2.a\tCritical Thinking\tLV\t4.7\t8\t0\t0\t0\tN\tN\t08/2023\tAnalyst\n"
                "29-1141.00\t2.A.2.b\tActive Learning\tLV\t4.0\t8\t0\t0\t0\tN\tN\t08/2023\tAnalyst\n"
                "29-1141.00\t2.A.2.e\tComplex Problem Solving\tLV\t4.6\t8\t0\t0\t0\tN\tN\t08/2023\tAnalyst\n"
                "29-1141.00\t2.A.2.f\tJudgment and Decision Making\tLV\t4.8\t8\t0\t0\t0\tN\tN\t08/2023\tAnalyst\n"
                "29-1141.00\t2.B.3.e\tSystems Analysis\tLV\t3.5\t8\t0\t0\t0\tN\tN\t08/2023\tAnalyst\n"
                "35-3023.00\t2.A.1.a\tReading Comprehension\tLV\t2.2\t8\t0\t0\t0\tN\tN\t08/2023\tAnalyst\n"
                "35-3023.00\t2.A.1.c\tWriting\tLV\t1.8\t8\t0\t0\t0\tN\tN\t08/2023\tAnalyst\n"
                "35-3023.00\t2.A.2.a\tCritical Thinking\tLV\t2.4\t8\t0\t0\t0\tN\tN\t08/2023\tAnalyst\n"
                "35-3023.00\t2.A.2.b\tActive Learning\tLV\t2.1\t8\t0\t0\t0\tN\tN\t08/2023\tAnalyst\n"
                "35-3023.00\t2.A.2.e\tComplex Problem Solving\tLV\t2.0\t8\t0\t0\t0\tN\tN\t08/2023\tAnalyst\n"
                "35-3023.00\t2.A.2.f\tJudgment and Decision Making\tLV\t2.2\t8\t0\t0\t0\tN\tN\t08/2023\tAnalyst\n"
                "35-3023.00\t2.B.3.e\tSystems Analysis\tLV\t1.4\t8\t0\t0\t0\tN\tN\t08/2023\tAnalyst\n"
                "41-2011.00\t2.A.1.a\tReading Comprehension\tLV\t2.5\t8\t0\t0\t0\tN\tN\t08/2023\tAnalyst\n"
                "41-2011.00\t2.A.1.c\tWriting\tLV\t1.9\t8\t0\t0\t0\tN\tN\t08/2023\tAnalyst\n"
                "41-2011.00\t2.A.2.a\tCritical Thinking\tLV\t2.2\t8\t0\t0\t0\tN\tN\t08/2023\tAnalyst\n"
                "41-2011.00\t2.A.2.b\tActive Learning\tLV\t2.0\t8\t0\t0\t0\tN\tN\t08/2023\tAnalyst\n"
                "41-2011.00\t2.A.2.e\tComplex Problem Solving\tLV\t1.8\t8\t0\t0\t0\tN\tN\t08/2023\tAnalyst\n"
                "41-2011.00\t2.A.2.f\tJudgment and Decision Making\tLV\t2.0\t8\t0\t0\t0\tN\tN\t08/2023\tAnalyst\n"
                "41-2011.00\t2.B.3.e\tSystems Analysis\tLV\t1.5\t8\t0\t0\t0\tN\tN\t08/2023\tAnalyst\n"
                "43-9061.00\t2.A.1.a\tReading Comprehension\tLV\t3.8\t8\t0\t0\t0\tN\tN\t08/2023\tAnalyst\n"
                "43-9061.00\t2.A.1.c\tWriting\tLV\t3.5\t8\t0\t0\t0\tN\tN\t08/2023\tAnalyst\n"
                "43-9061.00\t2.A.2.a\tCritical Thinking\tLV\t3.4\t8\t0\t0\t0\tN\tN\t08/2023\tAnalyst\n"
                "43-9061.00\t2.A.2.b\tActive Learning\tLV\t3.0\t8\t0\t0\t0\tN\tN\t08/2023\tAnalyst\n"
                "43-9061.00\t2.A.2.e\tComplex Problem Solving\tLV\t2.9\t8\t0\t0\t0\tN\tN\t08/2023\tAnalyst\n"
                "43-9061.00\t2.A.2.f\tJudgment and Decision Making\tLV\t3.1\t8\t0\t0\t0\tN\tN\t08/2023\tAnalyst\n"
                "43-9061.00\t2.B.3.e\tSystems Analysis\tLV\t2.8\t8\t0\t0\t0\tN\tN\t08/2023\tAnalyst\n"
                "47-2111.00\t2.A.1.a\tReading Comprehension\tLV\t3.0\t8\t0\t0\t0\tN\tN\t08/2023\tAnalyst\n"
                "47-2111.00\t2.A.1.c\tWriting\tLV\t2.2\t8\t0\t0\t0\tN\tN\t08/2023\tAnalyst\n"
                "47-2111.00\t2.A.2.a\tCritical Thinking\tLV\t3.1\t8\t0\t0\t0\tN\tN\t08/2023\tAnalyst\n"
                "47-2111.00\t2.A.2.b\tActive Learning\tLV\t2.8\t8\t0\t0\t0\tN\tN\t08/2023\tAnalyst\n"
                "47-2111.00\t2.A.2.e\tComplex Problem Solving\tLV\t3.4\t8\t0\t0\t0\tN\tN\t08/2023\tAnalyst\n"
                "47-2111.00\t2.A.2.f\tJudgment and Decision Making\tLV\t3.0\t8\t0\t0\t0\tN\tN\t08/2023\tAnalyst\n"
                "47-2111.00\t2.B.3.e\tSystems Analysis\tLV\t2.4\t8\t0\t0\t0\tN\tN\t08/2023\tAnalyst\n"
                "51-9199.00\t2.A.1.a\tReading Comprehension\tLV\t2.8\t8\t0\t0\t0\tN\tN\t08/2023\tAnalyst\n"
                "51-9199.00\t2.A.1.c\tWriting\tLV\t1.9\t8\t0\t0\t0\tN\tN\t08/2023\tAnalyst\n"
                "51-9199.00\t2.A.2.a\tCritical Thinking\tLV\t2.7\t8\t0\t0\t0\tN\tN\t08/2023\tAnalyst\n"
                "51-9199.00\t2.A.2.b\tActive Learning\tLV\t2.6\t8\t0\t0\t0\tN\tN\t08/2023\tAnalyst\n"
                "51-9199.00\t2.A.2.e\tComplex Problem Solving\tLV\t2.9\t8\t0\t0\t0\tN\tN\t08/2023\tAnalyst\n"
                "51-9199.00\t2.A.2.f\tJudgment and Decision Making\tLV\t2.5\t8\t0\t0\t0\tN\tN\t08/2023\tAnalyst\n"
                "51-9199.00\t2.B.3.e\tSystems Analysis\tLV\t2.0\t8\t0\t0\t0\tN\tN\t08/2023\tAnalyst\n"
                "53-3032.00\t2.A.1.a\tReading Comprehension\tLV\t2.9\t8\t0\t0\t0\tN\tN\t08/2023\tAnalyst\n"
                "53-3032.00\t2.A.1.c\tWriting\tLV\t1.7\t8\t0\t0\t0\tN\tN\t08/2023\tAnalyst\n"
                "53-3032.00\t2.A.2.a\tCritical Thinking\tLV\t2.8\t8\t0\t0\t0\tN\tN\t08/2023\tAnalyst\n"
                "53-3032.00\t2.A.2.b\tActive Learning\tLV\t2.4\t8\t0\t0\t0\tN\tN\t08/2023\tAnalyst\n"
                "53-3032.00\t2.A.2.e\tComplex Problem Solving\tLV\t3.0\t8\t0\t0\t0\tN\tN\t08/2023\tAnalyst\n"
                "53-3032.00\t2.A.2.f\tJudgment and Decision Making\tLV\t2.7\t8\t0\t0\t0\tN\tN\t08/2023\tAnalyst\n"
                "53-3032.00\t2.B.3.e\tSystems Analysis\tLV\t2.1\t8\t0\t0\t0\tN\tN\t08/2023\tAnalyst\n"
            ),
            "Technology Skills.txt": (
                "O*NET-SOC Code\tExample\tCommodity Code\tCommodity Title\tHot Technology\tIn Demand\n"
                "15-1252.00\tTensorFlow\t43230000\tMachine learning software\tY\tY\n"
                "15-1252.00\tChatGPT\t43230001\tGenerative AI software\tY\tY\n"
                "13-2051.00\tPredictive analytics software\t43230002\tPredictive analytics software\tY\tN\n"
                "29-1141.00\tElectronic health records\t43230003\tClinical software\tN\tN\n"
                "43-9061.00\tSpreadsheet software\t43230004\tOffice suite software\tN\tN\n"
                "47-2111.00\tComputerized maintenance management system CMMS\t43230005\tMaintenance software\tN\tN\n"
                "51-9199.00\tManufacturing execution systems MES\t43230006\tManufacturing control software\tN\tN\n"
                "53-3032.00\tFleet management software\t43230007\tTransportation management software\tN\tN\n"
            ),
        }
        return files[filename]


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


def test_metro_exposure_analysis_uses_onet_acs_and_cbp() -> None:
    result = analyze_great_reallocation_exposure(
        settings=_settings(),
        use_live_lookup=True,
        census=FakeCensusConnector(),
        onet=FakeONETConnector(),
    )

    assert result.quantitative_results["status"] == "completed"
    assert result.findings
    assert any(finding["agent"] == "quantitative_analyst" for finding in result.findings)
    assert any(finding["agent"] == "economic_complexity" for finding in result.findings)
    assert result.quantitative_results["top_exposure_metros"][0]["name"].startswith("San Jose")
    assert "O*NET" in result.methodology_description
