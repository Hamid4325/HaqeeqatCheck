from urllib.parse import urlparse

from .base import EvidenceItem
from .config import MAX_QUERY_RESULTS, SEARCH_BACKEND, SOURCE_PRIORITY, debug_enabled
from .debug_trace import trace

DNS_RESOLVERS = [
    "doh://cloudflare-dns.com/dns-query",
    "1.1.1.1",
    "system",
]


def _install_resolver_client() -> None:
    """Preset ddgs' HTTP clients with a reliable DNS resolver chain.

    ddgs search engines each build their own primp client through
    ``ddgs.http_client.HttpClient``, which would otherwise use the system
    resolver. On some networks that resolver intermittently refuses lookups
    for search backends (e.g. bing.com, startpage.com), so we wrap
    ``HttpClient.__init__`` to inject ``DNS_RESOLVERS`` into every engine
    client. Also presets the shared ddgs client for compatibility. Harmless
    no-op if ddgs is absent or already patched.
    """
    try:
        import ddgs.ddgs as ddgs_mod
        import ddgs.http_client as http_mod
        import primp as real_primp
    except ImportError:
        return
    if getattr(ddgs_mod, "_http_client", None) is None:
        try:
            ddgs_mod._http_client = real_primp.Client(
                timeout=5, dns_resolver=DNS_RESOLVERS
            )
        except Exception:
            pass
    if getattr(http_mod, "_HAQEEQAT_DNS_PATCHED", False):
        return

    original_init = http_mod.HttpClient.__init__

    def _patched_init(self, proxy=None, timeout=10, *, verify=True):
        original_init(self, proxy=proxy, timeout=timeout, verify=verify)
        try:
            self.client = real_primp.Client(
                proxy=proxy,
                timeout=timeout,
                impersonate="random",
                impersonate_os="random",
                verify=verify if isinstance(verify, bool) else True,
                ca_cert_file=verify if isinstance(verify, str) else None,
                dns_resolver=DNS_RESOLVERS,
            )
        except Exception:
            pass

    http_mod.HttpClient.__init__ = _patched_init
    http_mod._HAQEEQAT_DNS_PATCHED = True


class EvidenceRetriever:
    def __init__(self, search_fn=None, max_results=MAX_QUERY_RESULTS):
        self._search_fn = search_fn
        self.max_results = max_results

    def retrieve(self, urdu_claim: str, english_claim: str) -> list[EvidenceItem]:
        search = self._get_search()
        merged: dict[str, EvidenceItem] = {}
        english = english_claim.rstrip(".").strip()
        queries = (
            f"{english} fact check",
            urdu_claim,
            f"is it true that {english}",
        )
        trace(f"[Evidence] Search queries: {queries}")
        for query in queries:
            try:
                results = search(query, region="pk-en", max_results=self.max_results)
                trace(f"[Evidence] Query {query[:50]!r} -> {len(results or [])} results")
            except Exception as exc:
                trace(f"[Evidence] Query {query[:50]!r} -> ERROR: {exc}")
                continue
            for raw in results or []:
                item = self._to_item(raw)
                if item is not None:
                    merged[item.url] = item
        ranked = self._rank(list(merged.values()))
        trace(f"[Evidence] Total merged: {len(merged)}, ranked: {len(ranked)}")
        if debug_enabled():
            print(
                f"DEBUG retriever queries={list(queries)} "
                f"results={[item.source_domain for item in ranked]}"
            )
        return ranked

    def _to_item(self, raw: dict):
        url = (raw.get("href") or "").strip()
        title = (raw.get("title") or "").strip()
        snippet = (raw.get("body") or "").strip()
        if not url:
            return None
        return EvidenceItem(
            title=title,
            url=url,
            snippet=snippet,
            source_domain=urlparse(url).netloc.lower(),
        )

    def _rank(self, items: list[EvidenceItem]) -> list[EvidenceItem]:
        def priority(item: EvidenceItem):
            domain = item.source_domain
            for index, preferred in enumerate(SOURCE_PRIORITY):
                if domain == preferred or domain.endswith("." + preferred):
                    return (0, index)
            return (1, 0)

        return sorted(items, key=priority)

    def _get_search(self):
        if self._search_fn is None:
            from ddgs import DDGS

            _install_resolver_client()

            def search(query, region="pk-en", max_results=None, backend=SEARCH_BACKEND):
                return DDGS().text(
                    query, region=region, max_results=max_results, backend=backend
                )

            self._search_fn = search
        return self._search_fn
