from __future__ import annotations

from collections.abc import Sequence

from ai_policy_lab.state import DatasetRecord


def infer_dataset_domain(*, question: str = "", constraints: Sequence[str] | None = None) -> str:
    corpus = " ".join([question, *(constraints or [])]).lower()
    labor_keywords = {
        "labor",
        "workforce",
        "occupation",
        "employment",
        "wage",
        "job",
        "upskilling",
        "worker",
        "metro",
        "economic complexity",
        "ai adoption",
    }
    climate_keywords = {"climate", "environment", "emissions", "pollution", "epa", "energy"}
    crypto_keywords = {"crypto", "cryptocurrency", "bitcoin", "blockchain", "token"}
    demographic_election_keywords = {
        "immigration",
        "immigrant",
        "citizenship",
        "naturalization",
        "voting",
        "election",
        "electorate",
        "demographic",
        "demographics",
        "turnout",
    }

    if any(keyword in corpus for keyword in labor_keywords):
        return "labor_market"
    if any(keyword in corpus for keyword in demographic_election_keywords):
        return "demography_elections"
    if any(keyword in corpus for keyword in climate_keywords):
        return "environment"
    if any(keyword in corpus for keyword in crypto_keywords):
        return "crypto"
    return "general_policy"


def default_dataset_catalog(*, domain: str = "labor_market") -> list[DatasetRecord]:
    if domain == "demography_elections":
        return [
            {
                "id": "census-acs-demographics",
                "name": "American Community Survey Demographic Tables",
                "source_agency": "Census",
                "url": "https://www.census.gov/programs-surveys/acs",
                "format": "api",
                "temporal_coverage": "2005-present, annual",
                "geographic_coverage": "US national, state, county, tract",
                "key_variables": ["nativity", "citizenship", "race_ethnicity", "age", "education"],
                "update_frequency": "annual",
                "access_method": "public_download",
                "quality_notes": "Core demographic and citizenship composition source for electorate and population change analysis.",
                "normalization_status": "raw",
            },
            {
                "id": "cps-voting-registration",
                "name": "Current Population Survey Voting and Registration Supplement",
                "source_agency": "Census/BLS",
                "url": "https://www.census.gov/topics/public-sector/voting/data/tables.html",
                "format": "download",
                "temporal_coverage": "Biennial election-cycle supplement",
                "geographic_coverage": "US household microdata",
                "key_variables": ["registration_status", "turnout", "citizenship", "race_ethnicity", "age"],
                "update_frequency": "biennial",
                "access_method": "public_download",
                "quality_notes": "Best public source for turnout and registration patterns by demographic group.",
                "normalization_status": "raw",
            },
            {
                "id": "dhs-yearbook-immigration",
                "name": "DHS Yearbook of Immigration Statistics",
                "source_agency": "DHS",
                "url": "https://www.dhs.gov/immigration-statistics/yearbook",
                "format": "download",
                "temporal_coverage": "Annual historical series",
                "geographic_coverage": "US national and selected state measures",
                "key_variables": ["lawful_permanent_residents", "naturalizations", "refugees", "nonimmigrants"],
                "update_frequency": "annual",
                "access_method": "public_download",
                "quality_notes": "Primary federal administrative source for immigration flow and status trends.",
                "normalization_status": "raw",
            },
            {
                "id": "eac-eavs",
                "name": "Election Administration and Voting Survey",
                "source_agency": "EAC",
                "url": "https://www.eac.gov/research-and-data/studies-and-reports",
                "format": "download",
                "temporal_coverage": "Biennial federal election administration data",
                "geographic_coverage": "US state and local election administration",
                "key_variables": ["registration", "ballots_cast", "mail_voting", "provisional_ballots"],
                "update_frequency": "biennial",
                "access_method": "public_download",
                "quality_notes": "Federal election-administration data useful for turnout and electoral process context.",
                "normalization_status": "raw",
            },
        ]

    if domain != "labor_market":
        return []

    return [
        {
            "id": "bls-oews",
            "name": "Occupational Employment and Wage Statistics",
            "source_agency": "BLS",
            "url": "https://www.bls.gov/oes/",
            "format": "api",
            "temporal_coverage": "1997-present, annual",
            "geographic_coverage": "US national, state, and MSA",
            "key_variables": ["occupation", "employment", "wages"],
            "update_frequency": "annual",
            "access_method": "public_download",
            "quality_notes": "Core occupation-by-place wage and employment benchmark for labor market work.",
            "normalization_status": "raw",
        },
        {
            "id": "bls-jolts",
            "name": "Job Openings and Labor Turnover Survey",
            "source_agency": "BLS",
            "url": "https://www.bls.gov/jlt/",
            "format": "api",
            "temporal_coverage": "2000-present, monthly",
            "geographic_coverage": "US national and selected regions/industries",
            "key_variables": ["job_openings", "hires", "quits", "layoffs"],
            "update_frequency": "monthly",
            "access_method": "public_download",
            "quality_notes": "Useful for labor demand, churn, and labor market tightness.",
            "normalization_status": "raw",
        },
        {
            "id": "census-cbp",
            "name": "County Business Patterns",
            "source_agency": "Census",
            "url": "https://www.census.gov/programs-surveys/cbp.html",
            "format": "api",
            "temporal_coverage": "1986-present, annual",
            "geographic_coverage": "US national to county",
            "key_variables": ["naics", "employment", "establishments", "payroll"],
            "update_frequency": "annual",
            "access_method": "public_download",
            "quality_notes": "Canonical input for industry-space and regional capability mapping.",
            "normalization_status": "raw",
        },
        {
            "id": "census-acs",
            "name": "American Community Survey",
            "source_agency": "Census",
            "url": "https://www.census.gov/programs-surveys/acs",
            "format": "api",
            "temporal_coverage": "2005-present, annual",
            "geographic_coverage": "US national to tract",
            "key_variables": ["education", "occupation", "income", "demographics"],
            "update_frequency": "annual",
            "access_method": "public_download",
            "quality_notes": "Best broad demographic and labor-market microdata source for subgroup analysis.",
            "normalization_status": "raw",
        },
        {
            "id": "fred-labor",
            "name": "Federal Reserve Economic Data",
            "source_agency": "FRED",
            "url": "https://fred.stlouisfed.org/",
            "format": "api",
            "temporal_coverage": "Varies by series",
            "geographic_coverage": "US and international macro time series",
            "key_variables": ["unemployment", "participation", "productivity", "macro_indicators"],
            "update_frequency": "monthly",
            "access_method": "api_key_required",
            "quality_notes": "Convenient macro overlay and metadata source for labor market context.",
            "normalization_status": "raw",
        },
        {
            "id": "onet-skills",
            "name": "O*NET Occupation Data",
            "source_agency": "O*NET",
            "url": "https://www.onetcenter.org/database.html",
            "format": "csv",
            "temporal_coverage": "Rolling releases",
            "geographic_coverage": "US occupation-level",
            "key_variables": ["skills", "tasks", "knowledge", "abilities"],
            "update_frequency": "quarterly",
            "access_method": "public_download",
            "quality_notes": "Critical for task exposure, skill overlap, and transition-pathway analysis.",
            "normalization_status": "raw",
        },
        {
            "id": "ipums-cps",
            "name": "IPUMS CPS",
            "source_agency": "IPUMS",
            "url": "https://cps.ipums.org/cps/",
            "format": "csv",
            "temporal_coverage": "1962-present, monthly/annual extracts",
            "geographic_coverage": "US microdata",
            "key_variables": ["employment", "occupation", "industry", "demographics"],
            "update_frequency": "monthly",
            "access_method": "public_download",
            "quality_notes": "Best harmonized microdata layer for longitudinal labor market measurement.",
            "normalization_status": "raw",
        },
        {
            "id": "lightcast-postings",
            "name": "Lightcast Job Postings",
            "source_agency": "Lightcast",
            "url": "https://lightcast.io/open-skills",
            "format": "api",
            "temporal_coverage": "Vendor maintained",
            "geographic_coverage": "US regional job postings",
            "key_variables": ["skills", "titles", "locations", "employers"],
            "update_frequency": "daily",
            "access_method": "restricted",
            "quality_notes": "Highly valuable for changing skill demand; proprietary access is usually required.",
            "normalization_status": "raw",
        },
    ]
