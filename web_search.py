"""Búsqueda web mínima (sin API keys ni dependencias) para "Ampliar con internet".

Solo sale a internet el texto de la pregunta: nunca el audio ni la
transcripción. Primero DuckDuckGo (versión HTML) y, si no hay resultados
(o DuckDuckGo pide verificación), la API de Wikipedia.
"""
import json
import re
import urllib.parse
import urllib.request
from html import unescape
from html.parser import HTMLParser

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) VoiceLoggerMac/1.0"
TIMEOUT = 8
DUCKDUCKGO_URL = "https://html.duckduckgo.com/html/?"
WIKIPEDIA_URL = "https://{lang}.wikipedia.org/w/api.php?"
MAX_SNIPPET_CHARS = 600


class WebSearchError(Exception):
    pass


def _fetch(url, timeout=TIMEOUT):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept-Language": "en,es;q=0.8"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 (https fijo)
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace")


def _clean(text):
    return re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", "", text or ""))).strip()


class _DuckDuckGoParser(HTMLParser):
    """Lee los resultados de html.duckduckgo.com (a.result__a / .result__snippet)."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.results = []
        self._field = None  # "title" | "snippet"
        self._depth = 0
        self._buffer = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = (attrs.get("class") or "").split()
        if self._field:
            self._depth += 1
            return
        if tag == "a" and "result__a" in classes:
            self.results.append({"title": "", "url": _real_url(attrs.get("href") or ""), "snippet": ""})
            self._field, self._depth, self._buffer = "title", 0, []
        elif "result__snippet" in classes and self.results:
            self._field, self._depth, self._buffer = "snippet", 0, []

    def handle_endtag(self, tag):
        if not self._field:
            return
        if self._depth:
            self._depth -= 1
            return
        self.results[-1][self._field] = re.sub(r"\s+", " ", "".join(self._buffer)).strip()
        self._field = None

    def handle_data(self, data):
        if self._field:
            self._buffer.append(data)


def _real_url(href):
    """DuckDuckGo envuelve los enlaces: //duckduckgo.com/l/?uddg=<url real>."""
    if href.startswith("//"):
        href = "https:" + href
    parsed = urllib.parse.urlparse(href)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        target = urllib.parse.parse_qs(parsed.query).get("uddg", [""])[0]
        return target or href
    return href


def parse_duckduckgo(html_text, limit=4):
    parser = _DuckDuckGoParser()
    parser.feed(html_text or "")
    results = []
    for item in parser.results:
        url = item["url"]
        if not item["title"] or not url.startswith("http") or "duckduckgo.com/y.js" in url:
            continue  # anuncios y enlaces internos
        results.append({"title": item["title"], "url": url, "snippet": item["snippet"][:MAX_SNIPPET_CHARS]})
        if len(results) >= limit:
            break
    return results


def parse_wikipedia(json_text, limit=4):
    pages = (json.loads(json_text or "{}").get("query") or {}).get("pages") or {}
    if isinstance(pages, dict):
        pages = list(pages.values())
    pages = sorted(pages, key=lambda page: page.get("index", 0))
    results = []
    for page in pages[:limit]:
        title = page.get("title") or ""
        url = page.get("fullurl") or ""
        if title and url:
            results.append({"title": title, "url": url, "snippet": _clean(page.get("extract"))[:MAX_SNIPPET_CHARS]})
    return results


def search(query, language="en", limit=4, fetch=_fetch):
    """Hasta ``limit`` resultados [{"title", "url", "snippet"}]. ``fetch`` se puede sustituir en tests."""
    query = (query or "").strip()
    if not query:
        return []
    results, errors = [], []
    try:
        results = parse_duckduckgo(fetch(DUCKDUCKGO_URL + urllib.parse.urlencode({"q": query})), limit)
    except Exception as exc:  # red, CAPTCHA, HTML inesperado: probamos Wikipedia
        errors.append(exc)
    if len(results) < 2:
        params = {
            "action": "query", "format": "json", "generator": "search", "gsrsearch": query,
            "gsrlimit": str(limit), "prop": "extracts|info", "exintro": "1", "explaintext": "1",
            "exlimit": str(limit), "inprop": "url",
        }
        wiki_lang = "es" if language == "es" else "en"
        try:
            seen = {r["url"] for r in results}
            for item in parse_wikipedia(fetch(WIKIPEDIA_URL.format(lang=wiki_lang) + urllib.parse.urlencode(params)), limit):
                if item["url"] not in seen and len(results) < limit:
                    results.append(item)
        except Exception as exc:
            errors.append(exc)
    if not results and len(errors) == 2:
        raise WebSearchError(f"no pude buscar en internet ({errors[-1]})")
    return results
