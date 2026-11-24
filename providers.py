"""Search providers, shared result formatting, and retry helper.

Every provider gets the same question, asks for 10 results, and is capped at
the same amount of text per result, so the answer model sees comparable input.
"""
import os
import random
import re
import time
from urllib.parse import urlparse

import httpx

NUM_RESULTS = 10
MAX_CHARS_PER_RESULT = 10_000  # Keenable snippet_max_length; applied to every provider
TIMEOUT_S = 60

# List price in USD per search call (checked 2026-11-23 on each provider's pricing page).
SEARCH_PRICE_USD = {
    "keenable_pro": 4 / 1000,       # keenable.ai/pricing: $4 per 1k requests, pro and realtime
    "keenable_realtime": 4 / 1000,
    "exa": 7 / 1000,                # exa.ai/pricing: $7 per 1k /search (auto), up to 10 results
    "parallel": 5 / 1000,           # docs.parallel.ai pricing: basic mode $5 per 1k, 10 results
    "tavily": 0.008,                # docs.tavily.com: basic search = 1 credit, $0.008 pay-as-you-go
    "no_search": 0.0,
}
ALL_PROVIDERS = list(SEARCH_PRICE_USD)

_SECRET_ENV = ["KEENABLE_API_KEY", "EXA_API_KEY", "PARALLEL_API_KEY", "TAVILY_API_KEY", "OPENAI_API_KEY", "HF_TOKEN"]


def redact(text: str) -> str:
    """Remove any API key value from a string before it is printed or saved."""
    text = str(text)
    for name in _SECRET_ENV:
        value = os.getenv(name)
        if value and len(value) >= 8:
            text = text.replace(value, "***")
    return text


# ---------------------------------------------------------------- retries

def _status_code(exc):
    for attr in ("status_code", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    value = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(value, int):
        return value
    match = re.search(r"\b(429|50[0-4])\b", str(exc))
    return int(match.group(1)) if match else None


def _is_retryable(exc) -> bool:
    if type(exc).__name__ == "UsageLimitExceededError":  # Tavily raises this on HTTP 429
        return True
    code = _status_code(exc)
    if code in (429, 500, 502, 503, 504):
        return True
    name = type(exc).__name__.lower()
    return "timeout" in name or "connection" in name or "remoteprotocol" in name


def _retry_after(exc):
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    try:
        return float(headers.get("retry-after"))
    except (TypeError, ValueError):
        return None


def with_retries(fn, *args, max_attempts=6, **kwargs):
    """Call fn, retrying rate limits (429), 5xx and network errors with exponential backoff."""
    for attempt in range(1, max_attempts + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if attempt == max_attempts or not _is_retryable(exc):
                raise
            wait = _retry_after(exc) or min(60, 2 ** attempt) + random.uniform(0, 1)
            print(f"  retry {attempt}/{max_attempts - 1} in {wait:.0f}s ({type(exc).__name__}: {redact(exc)[:120]})")
            time.sleep(wait)


# ---------------------------------------------------------------- providers
# Each returns a list of {"title", "url", "date", "text"} dicts.

_clients = {}


def _client(name):
    if name not in _clients:
        if name == "keenable":
            _clients[name] = httpx.Client(timeout=TIMEOUT_S)
        elif name == "exa":
            from exa_py import Exa
            _clients[name] = Exa(api_key=os.environ["EXA_API_KEY"])
        elif name == "parallel":
            from parallel import Parallel
            # max_retries=0: we retry ourselves, so latency is measured on one attempt
            _clients[name] = Parallel(api_key=os.environ["PARALLEL_API_KEY"], max_retries=0, timeout=TIMEOUT_S)
        elif name == "tavily":
            from tavily import TavilyClient
            _clients[name] = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    return _clients[name]


# ---------------------------------------------------------------- domain exclusion
# Uses each API's own exclusion setting. Keenable and Parallel state that a plain domain also excludes
# its subdomains; Exa documents "*.domain" for subdomains, so it gets both forms; Tavily's docs don't say,
# but a test (2026-11-23) showed a plain domain excluded en./de./fr. subdomains too.

WIKIPEDIA = "wikipedia.org"
EXCLUDE_WIKIPEDIA_FOR = set()  # benchmarks that block Wikipedia: none, by choice

# Sites that host copies of public benchmarks (questions with gold answers). A search that returns one of
# these can read the answer key, so they are excluded for every provider on the public benchmarks.
BENCHMARK_LEAK_DOMAINS = [
    "huggingface.co", "hf.co", "kaggle.com", "aclanthology.org", "github.com", "githubusercontent.com",
    "openaipublic.blob.core.windows.net", "modelscope.cn", "opendatalab.com",
]
LEAK_PROTECTED_BENCHMARKS = {"simpleqa", "frames"}  # our own recent-events set is not published anywhere


def excluded_domains_for(provider, benchmark):
    """The domains a provider's search excludes on a benchmark (empty for the no-search baseline)."""
    if provider == "no_search":
        return []
    domains = list(BENCHMARK_LEAK_DOMAINS) if benchmark in LEAK_PROTECTED_BENCHMARKS else []
    if benchmark in EXCLUDE_WIKIPEDIA_FOR:
        domains.append(WIKIPEDIA)
    return domains


def host_matches(url, domains):
    host = (urlparse(url or "").hostname or "").lower()
    return any(host == d or host.endswith("." + d) for d in domains)


def is_excluded_host(url):
    """True for a Wikipedia URL (kept for the wikipedia_results count)."""
    return host_matches(url, [WIKIPEDIA])


def _keenable(query, mode, exclude):
    body = {"query": query, "mode": mode, "max_results": NUM_RESULTS, "snippet_max_length": MAX_CHARS_PER_RESULT}
    if exclude:
        body["exclude_domains"] = list(exclude)
    r = _client("keenable").post(
        "https://api.keenable.ai/v1/search",
        headers={"X-API-Key": os.environ["KEENABLE_API_KEY"]},
        json=body,
    )
    r.raise_for_status()
    return [
        {"title": x.get("title"), "url": x.get("url"), "date": x.get("published_at"), "text": x.get("snippet")}
        for x in r.json().get("results", [])
    ]


def _exa(query, exclude):
    extra = {"exclude_domains": list(exclude) + ["*." + d for d in exclude]} if exclude else {}
    resp = _client("exa").search(
        query,
        type="auto",
        num_results=NUM_RESULTS,
        contents={"text": {"max_characters": MAX_CHARS_PER_RESULT}},
        timeout=TIMEOUT_S,
        **extra,
    )
    return [{"title": x.title, "url": x.url, "date": x.published_date, "text": x.text} for x in resp.results]


def _parallel(query, exclude):
    settings = {"max_results": NUM_RESULTS, "excerpt_settings": {"max_chars_per_result": MAX_CHARS_PER_RESULT}}
    if exclude:
        settings["source_policy"] = {"exclude_domains": list(exclude)}
    resp = _client("parallel").search(
        objective=query,
        search_queries=[query],
        mode="basic",
        max_chars_total=NUM_RESULTS * MAX_CHARS_PER_RESULT,
        advanced_settings=settings,
    )
    return [
        {"title": x.title, "url": x.url, "date": x.publish_date, "text": "\n".join(x.excerpts or [])}
        for x in resp.results
    ]


def _tavily(query, exclude):
    extra = {"exclude_domains": list(exclude)} if exclude else {}
    resp = _client("tavily").search(query=query, search_depth="basic", max_results=NUM_RESULTS, timeout=TIMEOUT_S,
                                    **extra)
    return [
        {"title": x.get("title"), "url": x.get("url"), "date": x.get("published_date"), "text": x.get("content")}
        for x in resp.get("results", [])
    ]


_SEARCH_FNS = {
    "keenable_pro": lambda q, ex: _keenable(q, "pro", ex),
    "keenable_realtime": lambda q, ex: _keenable(q, "realtime", ex),
    "exa": _exa,
    "parallel": _parallel,
    "tavily": _tavily,
}


def search(provider, query, exclude=()):
    """One search call, excluding the given domains. Returns (results, latency_ms of the successful attempt)."""
    if provider == "no_search":
        return [], 0.0

    def timed():
        start = time.perf_counter()
        results = _SEARCH_FNS[provider](query, list(exclude))
        return results, (time.perf_counter() - start) * 1000

    return with_retries(timed)


# ---------------------------------------------------------------- formatting

def format_results(results):
    """Identical layout for every provider: numbered blocks of title, URL, date (if any) and text."""
    blocks = []
    for i, r in enumerate(results, 1):
        lines = [f"[{i}] {(r.get('title') or '').strip() or '(no title)'}", f"URL: {r.get('url') or ''}"]
        if r.get("date"):
            lines.append(f"Published: {str(r['date'])[:10]}")
        lines.append((r.get("text") or "").strip()[:MAX_CHARS_PER_RESULT])
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)
