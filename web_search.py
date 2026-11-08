"""Búsqueda web mínima (sin API keys ni dependencias) para las respuestas con internet.

Solo sale a internet el texto de la pregunta (más operadores ``site:``):
nunca el audio ni la transcripción. Pasos:

1. DuckDuckGo (versión HTML; sus resultados vienen sobre todo de Bing), sin
   redes sociales ni sitios que piden login (``-site:pinterest.com``…).
2. A la vez, una búsqueda en cada sitio preferido (``… site:hbr.org``;
   DuckDuckGo no admite ``OR`` entre ``site:``) y, según el tema de la
   pregunta, en un sitio de referencia: frontend → MDN, backend →
   Stack Overflow, marketing → Think with Google.
3. Si hay menos de 2 resultados (o DuckDuckGo pide verificación), Wikipedia.
4. Orden: sitios preferidos, luego fuentes confiables (Wikipedia, gobierno,
   universidades, organismos, ciencia, prensa seria, documentación oficial),
   luego el resto.
5. Opcional (``read_pages``): se abren las mejores páginas y se guardan sus
   párrafos más relacionados con la pregunta, no solo el resumen del buscador.
"""
import ipaddress
import json
import re
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, wait
from html import unescape
from html.parser import HTMLParser

from meeting_memory import tokenize

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) VoiceLoggerMac/1.0"
TIMEOUT = 8
DUCKDUCKGO_URL = "https://html.duckduckgo.com/html/?"
WIKIPEDIA_URL = "https://{lang}.wikipedia.org/w/api.php?"
MAX_SNIPPET_CHARS = 600
CANDIDATES = 8  # primeros resultados del buscador que se reordenan (más abajo suelen ser menos relevantes)
MAX_SITES = 3  # una búsqueda por sitio: más serían demasiadas peticiones a DuckDuckGo
PER_SITE = 2  # resultados que se toman de cada sitio preferido

# Lectura de páginas.
PAGE_TIMEOUT = 5
PAGE_DEADLINE = 7  # segundos como máximo para todas las páginas (se leen a la vez)
MAX_PAGE_BYTES = 1_500_000
PAGE_EXCERPT_CHARS = 2500

# Redes sociales y sitios con login: casi nunca se pueden leer y no aportan.
EXCLUDED_SITES = ("pinterest.com", "facebook.com", "instagram.com", "tiktok.com", "x.com", "twitter.com")

# Fuentes confiables (además de .gov/.edu/.int y gob.xx/gov.xx/edu.xx/ac.xx).
TRUSTED_DOMAINS = (
    "wikipedia.org", "britannica.com", "who.int", "un.org", "oecd.org", "worldbank.org", "imf.org",
    "europa.eu", "nature.com", "science.org", "sciencedirect.com", "springer.com", "arxiv.org",
    "thelancet.com", "nejm.org", "bmj.com", "mayoclinic.org", "clevelandclinic.org",
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "economist.com", "ft.com", "nytimes.com",
    "theguardian.com", "elpais.com", "hbr.org", "mckinsey.com", "investopedia.com",
    "developer.apple.com", "support.apple.com", "learn.microsoft.com", "developer.mozilla.org",
    "docs.python.org", "cloud.google.com", "aws.amazon.com",
    # Desarrollo (documentación oficial y referencias)
    "web.dev", "developer.chrome.com", "react.dev", "nextjs.org", "vuejs.org", "angular.dev", "svelte.dev",
    "nodejs.org", "typescriptlang.org", "python.org", "djangoproject.com", "fastapi.tiangolo.com",
    "postgresql.org", "docs.docker.com", "kubernetes.io", "git-scm.com", "w3.org", "owasp.org",
    "stackoverflow.com", "martinfowler.com", "developers.google.com", "developer.android.com",
    "rust-lang.org", "go.dev", "scikit-learn.org", "pandas.pydata.org",
    # Marketing y marketing science
    "thinkwithgoogle.com", "ama.org", "marketingscience.info", "nielsen.com", "moz.com",
    "searchengineland.com", "support.google.com", "bcg.com", "gartner.com", "forrester.com",
    "pewresearch.org", "statista.com",
)

# Sitio de referencia que se busca con site: según el tema de la pregunta
# (palabras ya en minúsculas y sin tildes; solo las que no se confunden con
# lenguaje normal: nada de «next», «rest», «go» o «lead»).
TOPIC_SITES = (
    ("developer.mozilla.org", {
        "javascript", "typescript", "react", "nextjs", "vue", "angular", "svelte", "css", "html", "dom",
        "tailwind", "frontend", "webpack", "vite", "navegador", "browser", "accesibilidad", "accessibility",
        "jsx", "tsx", "hooks", "flexbox",
    }),
    ("stackoverflow.com", {
        "python", "django", "flask", "fastapi", "nodejs", "java", "golang", "rust", "sql", "postgres",
        "postgresql", "mysql", "sqlite", "mongodb", "redis", "api", "apis", "graphql", "docker", "kubernetes",
        "k8s", "aws", "azure", "gcp", "backend", "microservicios", "microservices", "oauth", "jwt", "kafka",
        "linux", "bash", "git", "endpoint", "endpoints", "orm", "serverless", "pandas", "numpy",
    }),
    ("thinkwithgoogle.com", {
        "marketing", "mercadeo", "campaign", "campaigns", "campana", "campanas", "brand", "branding", "seo",
        "sem", "ctr", "cpc", "cpm", "cpa", "roas", "roi", "attribution", "atribucion", "conversion",
        "conversiones", "conversions", "funnel", "embudo", "segmentation", "segmentacion", "ltv", "cac",
        "churn", "retention", "retencion", "audience", "audiencia", "ads", "anuncios", "ga4", "cohort",
        "cohorte", "uplift", "incrementality", "incrementalidad", "mmm", "pricing", "engagement", "leads",
        "crm", "influencer", "influencers",
    }),
)
MAX_TOPIC_SITES = 2
_TRUSTED_TLDS = {"gov", "edu", "int", "mil"}
_TRUSTED_SECOND_LEVEL = {"gov", "gob", "edu", "ac", "go", "govt"}  # gov.uk, gob.mx, edu.co, ac.uk…
_UNREADABLE = ("youtube.com", "youtu.be", "vimeo.com", "spotify.com", "linkedin.com") + EXCLUDED_SITES
_DOMAIN_RE = re.compile(r"^[a-z0-9-]+(?:\.[a-z0-9-]+)+$")


class WebSearchError(Exception):
    pass


def _fetch(url, timeout=TIMEOUT):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept-Language": "en,es;q=0.8"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 (https fijo)
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace")


def _clean(text):
    return re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", "", text or ""))).strip()


# ---------- Dominios ----------

def host_of(url):
    """Dominio de una URL, en minúsculas y sin «www.»."""
    host = (urllib.parse.urlparse(url or "").hostname or "").lower().rstrip(".")
    return host[4:] if host.startswith("www.") else host


def _matches(host, domain):
    return host == domain or host.endswith("." + domain)


def is_trusted(url):
    """True para Wikipedia, gobierno, universidades, organismos, ciencia,
    prensa seria y documentación oficial."""
    host = host_of(url)
    if not host:
        return False
    labels = host.split(".")
    if labels[-1] in _TRUSTED_TLDS:
        return True
    if len(labels[-1]) == 2 and len(labels) >= 2 and labels[-2] in _TRUSTED_SECOND_LEVEL:
        # gov.uk / gob.mx son el propio gobierno; edu.co, ac.uk… solo con algo delante
        if len(labels) >= 3 or labels[-2] in ("gov", "gob"):
            return True
    return any(_matches(host, domain) for domain in TRUSTED_DOMAINS)


def parse_sites(text, limit=MAX_SITES):
    """«hbr.org, https://www.mckinsey.com/insights site:who.int» → ["hbr.org", "mckinsey.com", "who.int"]."""
    if isinstance(text, (list, tuple)):
        text = " ".join(text)
    sites = []
    for raw in re.split(r"[\s,;]+", (text or "").lower()):
        if raw.startswith("site:"):
            raw = raw[5:]
        if "://" in raw:
            raw = raw.split("://", 1)[1]
        raw = raw.split("/", 1)[0].strip(".")
        if raw.startswith("www."):
            raw = raw[4:]
        if _DOMAIN_RE.match(raw) and raw not in sites:
            sites.append(raw)
    return sites[:limit]


def topic_sites(query):
    """Sitios de referencia para el tema de la pregunta (como mucho ``MAX_TOPIC_SITES``)."""
    words = set(tokenize(query))
    return [site for site, keywords in TOPIC_SITES if words & keywords][:MAX_TOPIC_SITES]


def build_queries(query, sites=()):
    """([una consulta «… site:x» por sitio preferido], consulta general sin redes sociales)."""
    general = query + " " + " ".join(f"-site:{site}" for site in EXCLUDED_SITES)
    return [f"{query} site:{site}" for site in sites], general


# ---------- DuckDuckGo / Wikipedia ----------

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


def _duckduckgo(query, fetch):
    return parse_duckduckgo(fetch(DUCKDUCKGO_URL + urllib.parse.urlencode({"q": query})), CANDIDATES)


def _wikipedia(query, language, limit, fetch):
    params = {
        "action": "query", "format": "json", "generator": "search", "gsrsearch": query,
        "gsrlimit": str(limit), "prop": "extracts|info", "exintro": "1", "explaintext": "1",
        "exlimit": str(limit), "inprop": "url",
    }
    wiki_lang = "es" if language == "es" else "en"
    return parse_wikipedia(fetch(WIKIPEDIA_URL.format(lang=wiki_lang) + urllib.parse.urlencode(params)), limit)


def _url_key(url):
    return url.split("#", 1)[0].rstrip("/").replace("://www.", "://").lower()


def rank(results, sites=()):
    """Preferidos primero, luego confiables, luego el resto; sin repetidos ni
    redes sociales. Dentro de cada grupo se respeta el orden del buscador."""
    unique, seen = [], set()
    for result in results:
        key = _url_key(result["url"])
        host = host_of(result["url"])
        if key in seen or not host or any(_matches(host, site) for site in EXCLUDED_SITES):
            continue
        seen.add(key)
        unique.append(result)

    def score(result):
        host = host_of(result["url"])
        if any(_matches(host, site) for site in sites):
            return 2
        return 1 if is_trusted(result["url"]) else 0

    return sorted(unique, key=lambda result: -score(result))  # sorted es estable


# ---------- Leer las páginas ----------

def _is_public_https(url):
    parsed = urllib.parse.urlparse(url or "")
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or not host or host == "localhost" or host.endswith((".local", ".internal")):
        return False
    try:
        ipaddress.ip_address(host)
        return False  # IP literal: nunca (evita la red local)
    except ValueError:
        return True


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    """Solo sigue redirecciones a otra página https pública."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not _is_public_https(newurl):
            return None
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_PAGE_OPENER = urllib.request.build_opener(_SafeRedirect)


def _fetch_page(url, timeout=PAGE_TIMEOUT):
    request = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml", "Accept-Language": "en,es;q=0.8",
    })
    with _PAGE_OPENER.open(request, timeout=timeout) as response:
        if response.headers.get_content_type() not in ("text/html", "application/xhtml+xml"):
            return ""  # PDF, imagen, descarga…
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read(MAX_PAGE_BYTES).decode(charset, errors="replace")


_SKIP_TAGS = {"script", "style", "noscript", "nav", "header", "footer", "aside", "form", "svg", "button",
              "select", "template", "iframe"}
_BLOCK_TAGS = {"p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "br", "tr", "td", "section", "article",
               "main", "blockquote", "dd", "dt", "pre", "ul", "ol", "table"}


class _TextExtractor(HTMLParser):
    """Texto de la página en bloques (párrafos, ítems, títulos), sin menús,
    cabeceras, pies, scripts ni formularios."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.blocks = []
        self._buffer = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self._skip += 1
        elif tag in _BLOCK_TAGS:
            self._flush()

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS:
            self._skip = max(0, self._skip - 1)
        elif tag in _BLOCK_TAGS:
            self._flush()

    def handle_data(self, data):
        if not self._skip:
            self._buffer.append(data)

    def _flush(self):
        text = re.sub(r"\s+", " ", "".join(self._buffer)).strip()
        self._buffer = []
        if text:
            self.blocks.append(text)

    def close(self):
        super().close()
        self._flush()


def _cut(text, max_chars):
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "…"


def extract_relevant_text(html_text, query, max_chars=PAGE_EXCERPT_CHARS):
    """Los párrafos de la página más relacionados con ``query`` (hasta
    ``max_chars``), en el orden en que aparecen. Sin coincidencias, el comienzo."""
    parser = _TextExtractor()
    try:
        parser.feed(html_text or "")
        parser.close()
    except Exception:  # HTML muy roto: lo que se haya leído
        pass
    blocks, seen = [], set()
    for block in parser.blocks:
        if len(block) >= 60 and len(block.split()) >= 8 and block not in seen:
            seen.add(block)
            blocks.append(block)
    if not blocks:
        return ""
    terms = set(tokenize(query))
    hits = [len(terms & set(tokenize(block))) for block in blocks]
    order = sorted(range(len(blocks)), key=lambda i: (-hits[i], i))
    chosen, total = [], 0
    for i in order:
        if total >= max_chars:
            break
        room = max_chars - total
        if len(blocks[i]) > room and chosen:
            continue  # no cabe entero: quizá uno más corto sí
        chosen.append(i)
        total += min(len(blocks[i]), room) + 1
    return "\n".join(_cut(blocks[i], max_chars) for i in sorted(chosen))[:max_chars + 1]


def _readable(url):
    host = host_of(url)
    return _is_public_https(url) and not url.lower().split("?", 1)[0].endswith(".pdf") and \
        not any(_matches(host, site) for site in _UNREADABLE)


def read_pages(results, query, count, fetch_page=_fetch_page, deadline=PAGE_DEADLINE):
    """Añade ``content`` (párrafos relevantes) a las primeras ``count`` páginas
    legibles. Las que fallen o tarden más de ``deadline`` se quedan con el resumen."""
    targets = [result for result in results if _readable(result["url"])][:count]
    if not targets:
        return results
    pool = ThreadPoolExecutor(max_workers=len(targets))
    futures = {pool.submit(fetch_page, result["url"]): result for result in targets}
    done, _ = wait(futures, timeout=deadline)
    pool.shutdown(wait=False, cancel_futures=True)
    for future in done:
        try:
            excerpt = extract_relevant_text(future.result(), query)
        except Exception:
            continue
        if excerpt:
            futures[future]["content"] = excerpt
    return results


# ---------- Búsqueda ----------

def search(query, language="en", limit=4, fetch=_fetch, sites=(), read=0, fetch_page=_fetch_page, topics=True):
    """Hasta ``limit`` resultados [{"title", "url", "snippet", "trusted", "content"?}].

    ``sites``: dominios preferidos (lista o texto «hbr.org, who.int»).
    ``read``: cuántas de las mejores páginas abrir y leer (0 = solo el resumen).
    ``topics``: añadir el sitio de referencia del tema (MDN, Stack Overflow…).
    ``fetch`` / ``fetch_page`` se pueden sustituir en tests.
    """
    query = (query or "").strip()
    if not query:
        return []
    sites = parse_sites(sites)
    if topics:
        for site in topic_sites(query):
            if len(sites) < MAX_SITES and site not in sites:
                sites.append(site)
    preferred_queries, general_query = build_queries(query, sites)

    def run(search_query):
        try:
            return _duckduckgo(search_query, fetch), None
        except Exception as exc:  # red, CAPTCHA, HTML inesperado
            return [], exc

    preferred = []
    if preferred_queries:
        with ThreadPoolExecutor(max_workers=len(preferred_queries)) as pool:  # a la vez que la general
            futures = [pool.submit(run, preferred_query) for preferred_query in preferred_queries]
            general, error = run(general_query)
            for future in futures:
                preferred += future.result()[0][:PER_SITE]
    else:
        general, error = run(general_query)
    if not general and error is None:
        general, error = run(query)  # por si los -site: dejaron la búsqueda vacía
    results = rank(preferred + general[:CANDIDATES], sites)

    if len(results) < 2:
        try:
            results = rank(results + _wikipedia(query, language, limit, fetch), sites)
        except Exception as exc:
            if not results and error is not None:
                raise WebSearchError(f"no pude buscar en internet ({exc})") from exc

    results = results[:limit]
    for result in results:
        result["trusted"] = is_trusted(result["url"])
    if read:
        read_pages(results, query, read, fetch_page)
    return results
