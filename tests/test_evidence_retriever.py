from verification.evidence_retriever import EvidenceRetriever

A = {"title": "Soch A", "href": "https://sochfactcheck.com/a", "body": "snippet a"}
AFP = {"title": "AFP B", "href": "https://factcheck.afp.com/doc/b", "body": "snippet b"}
DAWN = {"title": "Dawn C", "href": "https://www.dawn.com/c", "body": "snippet c"}
OTHER = {"title": "X D", "href": "https://example.com/d", "body": "snippet d"}


class RecordingSearch:
    def __init__(self, *batches):
        self.batches = list(batches)
        self.queries = []

    def __call__(self, query, region="pk-en", max_results=5):
        self.queries.append(query)
        if not self.batches:
            return []
        return self.batches.pop(0)


def test_runs_two_language_queries():
    search = RecordingSearch([], [])
    EvidenceRetriever(search_fn=search).retrieve("اردو دعویٰ", "English claim")
    assert search.queries == ["English claim fact check", "اردو دعویٰ"]


def test_dedupes_by_url_later_replaces_earlier():
    other = dict(A, href="https://sochfactcheck.com/a", body="earlier snippet")
    later = dict(A, href="https://sochfactcheck.com/a", body="later snippet")
    search = RecordingSearch([other], [later])
    results = EvidenceRetriever(search_fn=search).retrieve("u", "e")
    assert len(results) == 1
    assert results[0].snippet == "later snippet"


def test_strictly_prioritizes_soch_afp_dawn():
    search = RecordingSearch([DAWN, OTHER, AFP, A], [])
    results = EvidenceRetriever(search_fn=search).retrieve("u", "e")
    domains = [r.source_domain for r in results]
    assert domains.index("sochfactcheck.com") < domains.index("factcheck.afp.com")
    assert domains.index("factcheck.afp.com") < domains.index("www.dawn.com")
    assert domains.index("www.dawn.com") < domains.index("example.com")


def test_missing_href_results_are_dropped():
    search = RecordingSearch([{"title": "no url", "body": "x"}, A], [])
    results = EvidenceRetriever(search_fn=search).retrieve("u", "e")
    assert len(results) == 1


def test_rate_limit_degrades_to_empty():
    def boom(query, region="pk-en", max_results=5):
        raise RuntimeError("ratelimit")

    assert EvidenceRetriever(search_fn=boom).retrieve("u", "e") == []


def test_install_resolver_client_presets_shared_ddgs_client(monkeypatch):
    import ddgs.ddgs as ddgs_mod
    import primp

    monkeypatch.setattr(ddgs_mod, "_http_client", None)
    from verification.evidence_retriever import _install_resolver_client

    _install_resolver_client()
    assert isinstance(ddgs_mod._http_client, primp.Client)


def test_install_resolver_client_injects_dns_resolver_into_engine_clients(monkeypatch):
    import ddgs.http_client as http_mod
    import primp
    from ddgs.http_client import HttpClient

    from verification.evidence_retriever import DNS_RESOLVERS, _install_resolver_client

    captured = {}

    def fake_client(*args, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(http_mod, "_HAQEEQAT_DNS_PATCHED", False)
    monkeypatch.setattr(primp, "Client", fake_client)
    _install_resolver_client()
    HttpClient()
    assert captured.get("dns_resolver") == DNS_RESOLVERS


def test_default_get_search_returns_callable():
    retriever = EvidenceRetriever()
    search = retriever._get_search()
    assert callable(search)


def test_default_search_passes_configured_backend(monkeypatch):
    import ddgs.ddgs as ddgs_mod
    from verification.config import SEARCH_BACKEND

    captured = {}

    def fake_text(self, query, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(ddgs_mod.DDGS, "text", fake_text)
    retriever = EvidenceRetriever()
    search = retriever._get_search()
    search("q", region="pk-en", max_results=3)
    assert captured.get("backend") == SEARCH_BACKEND
