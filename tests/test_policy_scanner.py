from ai_policy_lab.agents.policy_scanner import _run_live_policy_scan


class FakeFederalRegisterConnector:
    def search_documents(
        self,
        *,
        term: str,
        agency_slug: str | None = None,
        per_page: int = 5,
    ) -> dict:
        _ = per_page
        return {
            "results": [
                {
                    "title": f"{agency_slug} document for {term}",
                    "publication_date": "2026-03-30",
                    "html_url": f"https://www.federalregister.gov/documents/2026/03/30/{agency_slug}-{term.replace(' ', '-')}",
                    "document_number": f"{agency_slug}-{term.replace(' ', '-')}",
                    "agencies": [{"name": agency_slug.replace('-', ' ').title()}],
                }
            ]
        }


def test_live_policy_scan_uses_federal_register_sources() -> None:
    summary, sources, issues = _run_live_policy_scan(
        root_question="How is AI adoption disrupting established upskilling pathways?",
        federal_register=FakeFederalRegisterConnector(),
    )

    assert "policy scanner is not connected" not in summary.lower()
    assert "Labor Department AI/labor rulemaking" in summary
    assert sources
    assert all(source["source_tier"] == "tier_1" for source in sources)
    assert any("congress.gov" in issue.lower() for issue in issues)
