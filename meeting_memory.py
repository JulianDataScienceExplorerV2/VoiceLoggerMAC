"""Memoria en disco de una reunión para la IA.

La transcripción se guarda por fragmentos (~3000 caracteres) en
``<memory_dir>/<sesion>.jsonl``. Al preguntar, se recuperan solo los
fragmentos relevantes con BM25 (búsqueda por palabras clave, Python puro, sin
otro modelo en memoria), así el prompt tiene tamaño fijo aunque la reunión
dure horas.
"""
import json
import math
import os
import re
import unicodedata

_SESSION_RE = re.compile(r"^[\w.-]{1,64}$")
_WORD_RE = re.compile(r"\w+", re.UNICODE)

# Palabras muy frecuentes (EN/ES, sin tildes) que no ayudan a encontrar nada.
STOPWORDS = set(
    """
    a about after again all also am an and any are as at be because been before being but by can
    could did do does doing for from had has have having he her here hers him his how i if in into
    is it its just me more most my no not now of on once only or other our out over own same she
    should so some such than that the their them then there these they this those through to too
    under until up very was we were what when where which while who whom why will with would you
    your yours yeah okay ok um uh like going gonna get got know think right
    al algo como con cual cuales cuando de del desde donde el ella ellas ellos en entre era eran es
    esa esas ese eso esos esta estaba estamos estan estar este esto estos fue fueron ha han hay la
    las le les lo los mas me mi mis muy nada ni no nos nosotros o para pero por porque que quien se
    sea ser si sin sobre son su sus tambien te tiene tienen todo todos tu tus un una uno unos y ya yo
    bueno pues entonces vale
    """.split()
)

BM25_K1 = 1.5
BM25_B = 0.75


def _fold(text):
    """Minúsculas y sin tildes, para que 'decisión' encuentre 'decision'."""
    decomposed = unicodedata.normalize("NFKD", (text or "").lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def tokenize(text):
    return [w for w in _WORD_RE.findall(_fold(text)) if len(w) > 1 and w not in STOPWORDS]


class MeetingMemory:
    def __init__(self, directory, session):
        if not _SESSION_RE.match(session or ""):
            raise ValueError(f"invalid session id: {session!r}")
        if not directory:
            raise ValueError("missing memory directory")
        os.makedirs(directory, exist_ok=True)
        self.path = os.path.join(directory, f"{session}.jsonl")

    def segments(self):
        """Fragmentos guardados, en orden; el primero con cada índice gana."""
        found = {}
        try:
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        index = int(item["index"])
                    except (ValueError, KeyError, TypeError):
                        continue  # línea corrupta (p. ej. corte de luz a medio escribir)
                    found.setdefault(index, item)
        except FileNotFoundError:
            return []
        return [found[i] for i in sorted(found)]

    def add(self, index, start, end, text):
        """Guarda el fragmento ``index``. Idempotente: repetirlo no duplica."""
        index = int(index)
        text = (text or "").strip()
        if not text or any(s["index"] == index for s in self.segments()):
            return False
        record = {"index": index, "start": start or "", "end": end or "", "text": text}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return True

    def search(self, query, k=3, exclude=None):
        """Los ``k`` fragmentos más relevantes para ``query`` (BM25), en orden cronológico."""
        query_terms = set(tokenize(query))
        if not query_terms:
            return []
        candidates = [s for s in self.segments() if not (exclude and exclude(s))]
        if not candidates:
            return []
        docs = [tokenize(s["text"]) for s in candidates]
        avg_len = sum(len(d) for d in docs) / len(docs) or 1.0
        doc_freq = {t: sum(1 for d in docs if t in d) for t in query_terms}
        n = len(docs)
        scored = []
        for segment, words in zip(candidates, docs):
            if not words:
                continue
            score = 0.0
            for term in query_terms:
                tf = words.count(term)
                if not tf:
                    continue
                idf = math.log(1 + (n - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
                score += idf * tf * (BM25_K1 + 1) / (tf + BM25_K1 * (1 - BM25_B + BM25_B * len(words) / avg_len))
            if score > 0:
                scored.append((score, segment))
        best = sorted(scored, key=lambda item: item[0], reverse=True)[:k]
        return sorted((segment for _, segment in best), key=lambda s: s["index"])
