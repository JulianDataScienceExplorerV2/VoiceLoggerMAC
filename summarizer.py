"""Resumen, preguntas clave y respuestas locales con Qwen (mlx-lm).

Se usa de tres formas:
- Desde la app Python (``build_summary``).
- Como proceso persistente para la app nativa: ``python summarizer.py serve``
  (Qwen se carga una sola vez y queda en memoria; protocolo JSON por líneas).
- Como CLI puntual: ``python summarizer.py summary|ask ...``.

El contexto/persona del modelo se lee de ``qwen_context.txt`` en cada petición.

Reuniones largas sin reventar la RAM: lo que gasta memoria no es el texto
(2 h de reunión son ~100 KB) sino el modelo y el prompt mientras genera. Por
eso cada prompt tiene un presupuesto fijo de tokens, lo antiguo vive en disco
(``meeting_memory``) y solo se trae lo relevante, el prefill va en pasos
pequeños y Qwen se descarga tras un rato sin uso.
"""
import datetime
import gc
import json
import os
import re
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Se puede cambiar sin tocar código, p. ej. para más calidad con más RAM:
#   VOICELOGGER_QWEN_MODEL=mlx-community/Qwen2.5-3B-Instruct-4bit
SUMMARY_MODEL = os.environ.get("VOICELOGGER_QWEN_MODEL") or "mlx-community/Qwen2.5-1.5B-Instruct-4bit"

# Qwen2.5 admite 32k tokens de contexto; ~12k caracteres (~3-4k tokens) caben de
# sobra con margen para la respuesta. Por encima se resume por tramos.
SINGLE_PASS_CHARS = 12000
CHUNK_CHARS = 8000
MAX_NOTES_CHARS = 9000
# Preguntas: notas del resumen en vivo + la parte más reciente de la reunión.
# Antes eran 24k caracteres de transcripción: el doble de prompt que el
# resumen, más lento y con picos de memoria/GPU (sobre todo con modelos 4B).
QA_MAX_CHARS = 12000
MAX_KEY_QUESTIONS = 6
MAX_TASKS = 10

# Presupuestos de tokens por bloque del prompt: el tamaño del prompt (y la
# memoria que usa al generar) no crece con la duración de la reunión.
CHARS_PER_TOKEN = 3.5  # estimación si el tokenizer aún no está cargado
LIVE_TEXT_TOKENS = 3000
LIVE_NOTES_TOKENS = 1000
LIVE_NEW_TOKENS = 2500
QA_NOTES_TOKENS = 800
QA_MEMORY_TOKENS = 1500
QA_RECENT_TOKENS = 1500
QA_MEMORY_HITS = 3

# Memoria de MLX:
# - el prompt se procesa en pasos de 512 tokens (menos pico de memoria en GPU);
# - la caché de buffers libres de MLX se limita y se vacía tras cada respuesta;
# - Qwen se descarga tras VOICELOGGER_QWEN_IDLE_UNLOAD segundos sin peticiones
#   (0 = nunca) y se recarga solo en la siguiente.
PREFILL_STEP = 512
MLX_CACHE_LIMIT_MB = int(os.environ.get("VOICELOGGER_MLX_CACHE_MB") or 512)
IDLE_UNLOAD_SECONDS = float(os.environ.get("VOICELOGGER_QWEN_IDLE_UNLOAD") or 600)

# Anti-bucles: los modelos pequeños a veces repiten la misma línea hasta
# agotar max_tokens ("- Toasting whole pork shoulder" ×80). La penalización
# de repetición lo frena al generar, y las líneas repetidas se quitan después
# (si no, el acta rota sería la base de la siguiente actualización en vivo).
REPETITION_PENALTY = 1.1
REPETITION_CONTEXT = 64
MAX_LINES_PER_SECTION = 25
ANSWER_MAX_LINES = 60

# Persona/contexto para Qwen. Se lee en cada petición: basta con editar y
# guardar. qwen_context.txt no se sube a git (puede tener datos privados) y se
# crea a partir del ejemplo la primera vez.
CONTEXT_FILE = os.path.join(BASE_DIR, "qwen_context.txt")
CONTEXT_EXAMPLE_FILE = os.path.join(BASE_DIR, "qwen_context.example.txt")
MAX_CONTEXT_CHARS = 4000

_lock = threading.Lock()
_model = None
_tokenizer = None
_last_error = None

# MLX asocia sus streams al hilo que carga el modelo. Todas las llamadas al
# modelo pasan por este único hilo para que carga y generación coincidan,
# aunque la GUI lance un QThread distinto en cada resumen.
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="summarizer")


PROMPTS = {
    "en": {
        "role": "You are a meeting assistant.",
        "job": (
            "Your job is to help me follow a meeting from its live transcript. "
            "Always answer in English, even if the transcript is in another language. "
            "Use only what the transcript says; do not invent names, dates or numbers. "
            "Reply only with the final result: do not show your reasoning or repeat these instructions."
        ),
        "job_open": (
            "Your job is to help me during a brainstorming meeting. "
            "Always answer in English, even if the transcript is in another language. "
            "Use the meeting as context and add your own knowledge and ideas, but never present them "
            "as something said in the meeting, and do not invent facts about the meeting. "
            "Reply only with the final result: do not show your reasoning or repeat these instructions."
        ),
        "minutes": (
            "Write short meeting notes in Markdown from the transcript below, "
            "using exactly these sections:\n"
        ),
        "sections": (
            "## Summary\n(2-3 sentences)\n"
            "## Key topics\n(bullets)\n"
            "## Key questions\n(bullets: the most important questions asked or raised, at most 5; "
            "if someone asked it, copy it word for word)\n"
            "## Decisions\n(bullets)\n"
            "## Action items\n(bullets, with owner if mentioned)\n"
        ),
        "transcript": "Transcript:",
        "chunk": (
            "Extract the key ideas, questions, decisions and action items (with owner "
            "if mentioned) from this part of the meeting, as short bullets.\n\nPart {index}:\n{chunk}"
        ),
        "part": "Part",
        "consolidated": "(Consolidated notes from the whole meeting)",
        "update": (
            "These are the current notes of a meeting in progress:\n\n{previous}\n\n"
            "This is the new part of the transcript:\n{new}\n\n"
            "Update the notes so they cover the whole meeting so far: keep the important "
            "earlier points, add the new ones and drop questions that were already answered. "
            "Use exactly these sections:\n"
        ),
        "ask": {
            "notes": "Notes of the meeting so far:",
            "memory": "Earlier parts of the meeting related to the question:",
            "recent": "Most recent part of the transcript:",
            "transcript": "Meeting transcript:",
            "question": "Question:",
            "web": "Web search results:",
            "reliable": "reliable source",
            "open_instruction": (
                "Answer as an expert who explains things clearly, so the answer can be shared as it is. Adapt the "
                "format to the question: for how-to questions, give numbered step-by-step instructions; for "
                "'which is best' or 'what should I choose' questions, give a clear recommendation first and then "
                "explain why, comparing it with the alternatives; for other questions, explain the reasoning in "
                "well-written paragraphs with concrete examples. Explain the why behind each point and do not use "
                "labels such as 'Key insight' or 'Next step'. Use the meeting as context only where it is "
                "relevant (say so briefly), add your own knowledge, and cite facts from the web results with "
                "their number, like [1] or [2]. If sources disagree, trust the ones marked (reliable source)."
            ),
            "open_instruction_noweb": (
                "Answer as an expert who explains things clearly, so the answer can be shared as it is. Adapt the "
                "format to the question: for how-to questions, give numbered step-by-step instructions; for "
                "'which is best' or 'what should I choose' questions, give a clear recommendation first and then "
                "explain why, comparing it with the alternatives; for other questions, explain the reasoning in "
                "well-written paragraphs with concrete examples. Explain the why behind each point and do not use "
                "labels such as 'Key insight' or 'Next step'. Use the meeting as context only where it is "
                "relevant (say so briefly) and add your own knowledge."
            ),
            "given": "Answer already given:",
            "complement": (
                "Using only the web results, add what they contribute to the answer already given: new facts, "
                "figures, examples or corrections, explained in the same style (steps or explanatory paragraphs, "
                "no labels). Do not repeat the answer already given. Cite web facts with their number, like [1] "
                "or [2]. If sources disagree, trust the ones marked (reliable source). If the web results add "
                "nothing useful, reply only: NOTHING NEW"
            ),
            "previous": "Answer based on the meeting:",
            "extend": (
                "Extend the answer with useful information from the web results. Keep what the "
                "meeting said, clearly separate it from what comes from the web, and cite web facts "
                "with their number, like [1] or [2]. If sources disagree, trust the ones marked "
                "(reliable source). If the web results are not relevant, say so."
            ),
            "instruction_short": (
                "Answer concisely (2-4 sentences) using only this information. If the answer "
                "is not there, say so clearly instead of making it up."
            ),
            "instruction_long": (
                "Give a thorough, well-organized answer using only this information: explain the "
                "context, what was said (and by whom, if known), any decisions, numbers or dates, "
                "and the open points or next steps. Use short paragraphs or bullet points "
                "(about 150-300 words). If the answer is not there, say so clearly instead of "
                "making it up."
            ),
        },
        "tasks": (
            "From the meeting notes and transcript below, list the action items: concrete things "
            "someone has to do after the meeting. Write one per line in exactly this format:\n"
            "- TASK | OWNER | DUE\n"
            "TASK: short imperative sentence. OWNER: who is responsible, or - if not mentioned. "
            "DUE: the deadline as YYYY-MM-DD if one was mentioned (today is {today}), or -.\n"
            "Write at most 10 lines and nothing else. If there are no action items, write: - none"
        ),
        "questions": (
            "From the latest part of the meeting below, list the important questions that were asked or "
            "raised. If someone asked it, copy it word for word. Do not repeat the questions already "
            "listed, even reworded. One per line, starting with '- ', at most {limit}, and nothing else. "
            "If there are no new questions, write: - none"
        ),
        "known": "Questions already listed:",
        "latest": "Latest part of the meeting:",
        "trimmed": "(… transcript trimmed: only the most recent part …)",
    },
    "es": {
        "role": "Eres un asistente de reuniones.",
        "job": (
            "Tu trabajo es ayudarme a seguir una reunión a partir de su transcripción en vivo. "
            "Responde SIEMPRE en español. Usa solo lo que dice la transcripción; "
            "no inventes nombres, fechas ni cifras. "
            "Responde solo con el resultado final: no muestres tu razonamiento ni repitas estas instrucciones."
        ),
        "job_open": (
            "Tu trabajo es ayudarme durante una reunión de lluvia de ideas. "
            "Responde SIEMPRE en español. Usa la reunión como contexto y añade tu propio conocimiento e "
            "ideas, pero nunca los presentes como algo dicho en la reunión ni inventes datos de la reunión. "
            "Responde solo con el resultado final: no muestres tu razonamiento ni repitas estas instrucciones."
        ),
        "minutes": (
            "A partir de la siguiente transcripción de reunión, genera un acta breve "
            "en formato Markdown con estas secciones exactas:\n"
        ),
        "sections": (
            "## Resumen\n(2-3 frases)\n"
            "## Temas tratados\n(viñetas)\n"
            "## Preguntas clave\n(viñetas: las preguntas más importantes que se han hecho o planteado, máximo 5; "
            "si alguien la hizo, cópiala tal cual)\n"
            "## Decisiones\n(viñetas)\n"
            "## Tareas / pendientes\n(viñetas, con responsable si se menciona)\n"
        ),
        "transcript": "Transcripción:",
        "chunk": (
            "Extrae las ideas clave, preguntas, decisiones y tareas (con responsable si se "
            "menciona) de este fragmento de reunión, en viñetas cortas.\n\nFragmento {index}:\n{chunk}"
        ),
        "part": "Parte",
        "consolidated": "(Notas consolidadas de toda la reunión)",
        "update": (
            "Estas son las notas actuales de una reunión en curso:\n\n{previous}\n\n"
            "Esta es la parte nueva de la transcripción:\n{new}\n\n"
            "Actualiza las notas para que cubran toda la reunión hasta ahora: conserva lo "
            "importante de antes, añade lo nuevo y quita las preguntas que ya se respondieron. "
            "Usa exactamente estas secciones:\n"
        ),
        "ask": {
            "notes": "Notas de la reunión hasta ahora:",
            "memory": "Partes anteriores de la reunión relacionadas con la pregunta:",
            "recent": "Parte más reciente de la transcripción:",
            "transcript": "Transcripción de la reunión:",
            "question": "Pregunta:",
            "web": "Resultados de la búsqueda en internet:",
            "reliable": "fuente confiable",
            "open_instruction": (
                "Responde como un experto que explica con claridad, para que la respuesta se pueda compartir tal "
                "cual. Adapta el formato a la pregunta: si es de cómo hacer algo, da instrucciones paso a paso "
                "numeradas; si es de cuál es mejor o qué elegir, da primero una recomendación clara y luego "
                "explica por qué, comparándola con las alternativas; en las demás, explica el razonamiento en "
                "párrafos bien escritos con ejemplos concretos. Explica el porqué de cada punto y no uses "
                "etiquetas como «Idea clave» o «Próximo paso». Usa la reunión como contexto solo donde sea "
                "relevante (dilo brevemente), añade tu propio conocimiento y cita los datos de internet con su "
                "número, como [1] o [2]. Si las fuentes no coinciden, confía en las marcadas (fuente confiable)."
            ),
            "open_instruction_noweb": (
                "Responde como un experto que explica con claridad, para que la respuesta se pueda compartir tal "
                "cual. Adapta el formato a la pregunta: si es de cómo hacer algo, da instrucciones paso a paso "
                "numeradas; si es de cuál es mejor o qué elegir, da primero una recomendación clara y luego "
                "explica por qué, comparándola con las alternativas; en las demás, explica el razonamiento en "
                "párrafos bien escritos con ejemplos concretos. Explica el porqué de cada punto y no uses "
                "etiquetas como «Idea clave» o «Próximo paso». Usa la reunión como contexto solo donde sea "
                "relevante (dilo brevemente) y añade tu propio conocimiento."
            ),
            "given": "Respuesta ya dada:",
            "complement": (
                "Usando solo los resultados de internet, agrega lo que aportan a la respuesta ya dada: datos "
                "nuevos, cifras, ejemplos o correcciones, explicados con el mismo estilo (pasos o párrafos "
                "explicativos, sin etiquetas). No repitas la respuesta ya dada. Cita los datos de internet con su "
                "número, como [1] o [2]. Si las fuentes no coinciden, confía en las marcadas (fuente confiable). "
                "Si los resultados no aportan nada útil, responde solo: NADA NUEVO"
            ),
            "previous": "Respuesta basada en la reunión:",
            "extend": (
                "Amplía la respuesta con información útil de los resultados de internet. Conserva lo "
                "que dijo la reunión, sepáralo claramente de lo que viene de internet y cita los datos "
                "de internet con su número, como [1] o [2]. Si las fuentes no coinciden, confía en las "
                "marcadas (fuente confiable). Si los resultados no son relevantes, dilo."
            ),
            "instruction_short": (
                "Responde de forma concisa (2-4 frases) usando solo esta información. Si la "
                "respuesta no aparece, dilo claramente en vez de inventarla."
            ),
            "instruction_long": (
                "Da una respuesta completa y bien organizada usando solo esta información: explica "
                "el contexto, qué se dijo (y quién, si se sabe), las decisiones, cifras o fechas, y "
                "los puntos abiertos o próximos pasos. Usa párrafos cortos o viñetas (unas 150-300 "
                "palabras). Si la respuesta no aparece, dilo claramente en vez de inventarla."
            ),
        },
        "tasks": (
            "A partir de las notas y la transcripción de abajo, lista las tareas: cosas concretas "
            "que alguien tiene que hacer después de la reunión. Escribe una por línea con este "
            "formato exacto:\n"
            "- TAREA | RESPONSABLE | FECHA\n"
            "TAREA: frase corta en infinitivo. RESPONSABLE: quién la hace, o - si no se dijo. "
            "FECHA: la fecha límite como AAAA-MM-DD si se mencionó (hoy es {today}), o -.\n"
            "Escribe como máximo 10 líneas y nada más. Si no hay tareas, escribe: - ninguna"
        ),
        "questions": (
            "De la parte más reciente de la reunión, abajo, lista las preguntas importantes que se hicieron "
            "o plantearon. Si alguien la hizo, cópiala tal cual. No repitas las preguntas ya listadas, ni "
            "con otras palabras. Una por línea, empezando con '- ', como máximo {limit}, y nada más. "
            "Si no hay preguntas nuevas, escribe: - ninguna"
        ),
        "known": "Preguntas ya listadas:",
        "latest": "Parte más reciente de la reunión:",
        "trimmed": "(… transcripción recortada: solo la parte más reciente …)",
    },
}


def _prompts(language):
    return PROMPTS.get(language, PROMPTS["en"])


def _mx_call(name, *args):
    """Llama a mlx.core.<name> (o mx.metal.<name> en versiones viejas) si existe."""
    try:
        import mlx.core as mx
    except ImportError:
        return None
    fn = getattr(mx, name, None)
    if fn is None and hasattr(mx, "metal"):
        fn = getattr(mx.metal, name, None)
    if fn is None:
        return None
    try:
        return fn(*args)
    except Exception:
        return None


def _ensure_model():
    global _model, _tokenizer
    with _lock:
        if _model is None:
            from mlx_lm import load

            try:
                _model, _tokenizer = load(SUMMARY_MODEL)
            except ValueError as exc:
                # p. ej. "Model type lfm2 not supported": modelos nuevos (LFM2.5)
                # necesitan un mlx-lm reciente.
                if "not supported" in str(exc).lower():
                    raise RuntimeError(
                        f"{exc} Actualiza mlx-lm: uv pip install --python .venv/bin/python -U mlx-lm"
                    ) from exc
                raise
            _mx_call("set_cache_limit", MLX_CACHE_LIMIT_MB * 1024 * 1024)


def _unload_model():
    """Libera Qwen de la memoria (se recarga solo en la siguiente petición)."""
    global _model, _tokenizer
    with _lock:
        if _model is None:
            return False
        _model = None
        _tokenizer = None
    gc.collect()
    _mx_call("clear_cache")
    return True


def _count_tokens(text):
    if _tokenizer is not None:
        try:
            return len(_tokenizer.encode(text))
        except Exception:
            pass
    return int(len(text) / CHARS_PER_TOKEN) + 1


def fit_tokens(text, max_tokens, keep="tail"):
    """Recorta ``text`` a ``max_tokens`` sin partir líneas.

    ``keep="tail"`` conserva el final (lo más reciente); ``"head"`` el principio
    (las notas empiezan por el resumen).
    """
    text = (text or "").strip()
    for _ in range(4):
        count = _count_tokens(text)
        if count <= max_tokens:
            return text
        cut = max(1, int(len(text) * max_tokens / count * 0.95))
        if keep == "tail":
            text = text[-cut:]
            newline = text.find("\n")
            if 0 <= newline < len(text) - 1:
                text = text[newline + 1:]
        else:
            text = text[:cut]
            newline = text.rfind("\n")
            if newline > 0:
                text = text[:newline]
    return text


def load_context(path=None, example_path=None):
    """Lee qwen_context.txt sin comentarios (#) ni líneas con XXXX sin rellenar."""
    path = path or CONTEXT_FILE
    example_path = example_path if example_path is not None else CONTEXT_EXAMPLE_FILE
    if not os.path.exists(path) and example_path and os.path.exists(example_path):
        try:
            shutil.copyfile(example_path, path)
        except OSError:
            path = example_path
    try:
        with open(path, encoding="utf-8") as f:
            raw = f.read()
    except OSError:
        return ""
    lines = []
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or "XXXX" in stripped.upper():
            continue
        lines.append(line.rstrip())
    return "\n".join(lines).strip()[:MAX_CONTEXT_CHARS]


def _system_prompt(language, mode="meeting"):
    prompts = _prompts(language)
    job = prompts["job_open"] if mode == "open" else prompts["job"]
    context = load_context()
    if context:
        return f"{context}\n\n{job}"
    return f"{prompts['role']} {job}"


def split_text(text, max_chars=CHUNK_CHARS):
    """Divide el texto en tramos de ~max_chars cortando por saltos de línea."""
    chunks, current, size = [], [], 0
    for line in text.splitlines():
        # Una línea gigante (sin saltos) se parte a la fuerza.
        while len(line) > max_chars:
            if current:
                chunks.append("\n".join(current))
                current, size = [], 0
            chunks.append(line[:max_chars])
            line = line[max_chars:]
        if size + len(line) + 1 > max_chars and current:
            chunks.append("\n".join(current))
            current, size = [], 0
        current.append(line)
        size += len(line) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


def _tail_text(text, max_chars, language="es"):
    """Se queda con lo más reciente sin partir la primera línea."""
    if len(text) <= max_chars:
        return text
    tail = text[-max_chars:]
    newline = tail.find("\n")
    if 0 <= newline < len(tail) - 1:
        tail = tail[newline + 1:]
    return _prompts(language)["trimmed"] + "\n" + tail


_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def _strip_thinking(result):
    """Quita el razonamiento <think>…</think> de modelos como Qwen3/Qwen3.5."""
    result = _THINK_RE.sub("", result or "")
    if "</think>" in result:  # la plantilla abrió <think> en el prompt: solo llega el cierre
        result = result.split("</think>", 1)[1]
    if "<think>" in result:  # razonamiento cortado por max_tokens
        result = result.split("<think>", 1)[0]
    return result.strip()


class ModelOutputError(RuntimeError):
    """El modelo devolvió su razonamiento (o nada útil) en vez del resultado."""


REASONING_MESSAGE = (
    "El modelo respondió con su razonamiento en vez del resultado (pasa con modelos que «piensan»). "
    "Vuelve a intentarlo o usa Qwen3 4B 2507."
)
_REASONING_OPENERS = (
    "the user", "okay, so", "okay, let", "ok, so", "alright, so", "let me", "let's see", "i need to",
    "we need to", "hmm", "wait,", "el usuario", "necesito", "vale, ", "bueno, el usuario", "veamos",
)
_SUMMARY_HEADING_RE = re.compile(r"^#{1,3}\s*(summary|resumen)\b", re.IGNORECASE)


def looks_like_reasoning(text):
    """True si el texto parece el razonamiento del modelo y no una respuesta."""
    head = (text or "").strip().lower()[:240]
    if head.startswith(_REASONING_OPENERS):
        return True
    return any(phrase in head for phrase in (
        "the user wants", "the user is asking", "the user asked", "user wants me to",
        "el usuario quiere", "el usuario pregunta", "me pide que",
    ))


def clean_minutes(markdown):
    """Deja solo el acta: desde el último encabezado de resumen (o el primero de
    sección) en adelante. Si no hay acta, sino razonamiento, lanza ModelOutputError."""
    lines = (markdown or "").strip().splitlines()
    summary_starts = [i for i, line in enumerate(lines) if _SUMMARY_HEADING_RE.match(line.strip())]
    if summary_starts:
        start = summary_starts[-1]  # la versión final suele ir después del borrador
    else:
        start = next((i for i, line in enumerate(lines) if line.lstrip().startswith("#")), None)
    if start is None:
        raise ModelOutputError(REASONING_MESSAGE)
    cleaned = "\n".join(lines[start:]).strip()
    body = "\n".join(lines[start + 1:]).strip()  # lo que va bajo el primer encabezado
    if looks_like_reasoning(cleaned) or looks_like_reasoning(body):
        raise ModelOutputError(REASONING_MESSAGE)
    return dedupe_lines(cleaned)


_FINAL_ANSWER_RE = re.compile(r"^\**\s*(final answer|answer|respuesta final|respuesta)\s*:\**\s*", re.IGNORECASE)


def clean_answer(text):
    """Respuesta sin razonamiento. Si el modelo pensó en voz alta pero dejó una
    línea «Answer:/Respuesta:», se usa lo que va después; si no, ModelOutputError."""
    text = dedupe_lines(text, max_per_section=ANSWER_MAX_LINES)
    if not looks_like_reasoning(text):
        return text
    lines = text.splitlines()
    marks = [i for i, line in enumerate(lines) if _FINAL_ANSWER_RE.match(line.strip())]
    if marks:
        i = marks[-1]
        tail = [_FINAL_ANSWER_RE.sub("", lines[i].strip())] + lines[i + 1:]
        answer = "\n".join(tail).strip()
        if answer and not looks_like_reasoning(answer):
            return answer
    raise ModelOutputError(REASONING_MESSAGE)


def _strip_code_fence(result):
    result = result.strip()
    if result.startswith("```"):
        lines = result.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        result = "\n".join(lines).strip()
    return result


def _repetition_processors():
    """Logits processors con penalización de repetición, o None si la versión
    de mlx-lm no los tiene."""
    try:
        from mlx_lm.sample_utils import make_logits_processors

        return make_logits_processors(
            repetition_penalty=REPETITION_PENALTY, repetition_context_size=REPETITION_CONTEXT
        )
    except (ImportError, TypeError):
        return None


_BULLET_RE = re.compile(r"^(?:[-*•+]|\d+[.)])\s*")
_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _line_key(line):
    """Forma normalizada de una línea para detectar repeticiones."""
    text = _BULLET_RE.sub("", line.strip())
    return " ".join(_WORD_RE.findall(text.replace("*", "").lower()))


def dedupe_lines(text, max_per_section=MAX_LINES_PER_SECTION):
    """Quita las líneas repetidas de cada sección (aunque cambien viñeta,
    negritas o puntuación), las viñetas vacías y lo que pase de
    ``max_per_section`` líneas por sección.

    Una línea corta (menos de 3 palabras, p. ej. «Pros:») solo se quita si
    repite la anterior: puede volver a salir con sentido en la misma sección.
    Los encabezados, líneas en blanco y separadores (``---``) se conservan.
    Sirve para que un bucle del modelo no llene el acta ni se arrastre a la
    siguiente actualización en vivo.
    """
    kept, seen = [], set()
    previous_key, in_section = None, 0
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            seen, previous_key, in_section = set(), None, 0
            kept.append(line)
            continue
        if not stripped:
            if not (kept and not kept[-1].strip()):  # sin líneas en blanco dobles
                kept.append(line)
            continue
        key = _line_key(stripped)
        if not key:
            if _BULLET_RE.sub("", stripped):
                kept.append(line)  # separador o símbolo suelto: se deja
            continue  # viñeta vacía ("-" cortado por max_tokens)
        repeated = key == previous_key or (key in seen and len(key.split()) >= 3)
        previous_key = key
        if repeated or in_section >= max_per_section:
            continue
        seen.add(key)
        in_section += 1
        kept.append(line)
    return "\n".join(kept).strip()


def _generate(system, user, max_tokens, temp=0.3, penalize_repeats=True):
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    prompt = _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        # Qwen3/Qwen3.5 "piensan" por defecto: en vivo solo añade latencia.
        # Las plantillas que no usan esta variable (Qwen2.5) la ignoran.
        enable_thinking=False,
    )
    options = dict(prompt=prompt, max_tokens=max_tokens, sampler=make_sampler(temp=temp), verbose=False)
    processors = _repetition_processors() if penalize_repeats else None
    extra = {"logits_processors": processors} if processors else {}
    # De la opción más completa a la más básica: las versiones viejas de
    # mlx-lm no aceptan prefill_step_size ni logits_processors.
    attempts = [dict(prefill_step_size=PREFILL_STEP, **extra), extra, {}]
    try:
        for index, attempt in enumerate(attempts):
            try:
                result = generate(_model, _tokenizer, **options, **attempt)
                break
            except TypeError:
                if index == len(attempts) - 1:
                    raise
    finally:
        _mx_call("clear_cache")
    return _strip_code_fence(_strip_thinking(result))


def _minutes_prompt(text, language="es"):
    prompts = _prompts(language)
    return f"{prompts['minutes']}{prompts['sections']}\n{prompts['transcript']}\n{text}"


def _summarize_blocking(text, language):
    system = _system_prompt(language)
    prompts = _prompts(language)

    if len(text) <= SINGLE_PASS_CHARS:
        return _generate(system, _minutes_prompt(text, language), max_tokens=700)

    # Reunión larga: notas por tramo y acta final sobre las notas, de modo que
    # el final de la reunión (decisiones, pendientes) nunca se pierde.
    notes = []
    for index, chunk in enumerate(split_text(text), start=1):
        note = _generate(
            system, prompts["chunk"].format(index=index, chunk=chunk), max_tokens=400
        )
        notes.append(f"{prompts['part']} {index}:\n{note}")
    joined = "\n\n".join(notes)
    if len(joined) > MAX_NOTES_CHARS:
        joined = joined[-MAX_NOTES_CHARS:]
    return _generate(
        system,
        _minutes_prompt(prompts["consolidated"] + "\n" + joined, language),
        max_tokens=700,
    )


def _run(text, language):
    _ensure_model()
    return _summarize_blocking(text, language)


def build_summary(text, language="es"):
    global _last_error
    text = (text or "").strip()
    if not text:
        return "", "no hay texto para resumir"

    try:
        result = clean_minutes(_executor.submit(_run, text, language).result())
        _last_error = None
        return result, None
    except Exception as exc:
        _last_error = f"no se pudo generar el resumen: {exc}"
        return "", _last_error


def get_last_error():
    return _last_error


# ---------- Resumen en vivo con preguntas clave ----------

_NUMBERED_RE = re.compile(r"^\d+[.)]\s+")
_NO_QUESTIONS = {
    "none", "n/a", "na", "none yet", "none so far", "no questions", "no key questions",
    "ninguna", "ninguno", "sin preguntas", "no hay preguntas", "ninguna por ahora",
}


def _is_empty_item(item):
    lowered = item.strip(" .").lower()
    if lowered in _NO_QUESTIONS:
        return True
    if lowered.startswith("(") and lowered.endswith(")"):
        return True  # el modelo repitió la instrucción, p. ej. "(bullets)"
    return (lowered.startswith("no ") and "question" in lowered) or lowered.startswith("no hay pregunta")


def _is_questions_title(title):
    return "question" in title or "pregunta" in title


def _is_actions_title(title):
    return any(word in title for word in ("action", "task", "to-do", "todo", "tarea", "pendiente"))


def _extract_section(markdown, matches_title, limit):
    """Saca de ``markdown`` las secciones cuyo título cumple ``matches_title``.

    Devuelve (resto_markdown, [viñetas de esas secciones]).
    """
    kept, items = [], []
    inside = False
    for line in (markdown or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            inside = matches_title(stripped.lstrip("#").strip().lower())
            if inside:
                continue
        elif inside:
            item = stripped
            for bullet in ("- ", "* ", "• "):
                if item.startswith(bullet):
                    item = item[len(bullet):].strip()
                    break
            else:
                item = _NUMBERED_RE.sub("", item).strip()
            if item and not _is_empty_item(item) and len(items) < limit:
                items.append(item)
            continue
        kept.append(line)
    return "\n".join(kept).strip(), items


def split_key_questions(markdown, limit=MAX_KEY_QUESTIONS):
    """Separa la sección de preguntas clave del acta: (resto_markdown, [preguntas])."""
    return _extract_section(markdown, _is_questions_title, limit)


def split_action_items(markdown, limit=MAX_TASKS):
    """Separa la sección de tareas / action items del acta: (resto_markdown, [tareas])."""
    return _extract_section(markdown, _is_actions_title, limit)


def _live_blocking(text, new_text, previous, language):
    if text.strip():
        if len(text) <= SINGLE_PASS_CHARS:
            text = fit_tokens(text, LIVE_TEXT_TOKENS, keep="tail")
        return _summarize_blocking(text, language)
    if previous.strip() and new_text.strip():
        # Incremental: notas anteriores + solo lo nuevo, ambos con presupuesto
        # fijo, así el costo no crece con la duración de la reunión. Las notas
        # se limpian de repeticiones para que un bucle no se arrastre.
        previous = dedupe_lines(previous)
        prompts = _prompts(language)
        user = prompts["update"].format(
            previous=fit_tokens(previous, LIVE_NOTES_TOKENS, keep="head"),
            new=fit_tokens(new_text, LIVE_NEW_TOKENS, keep="tail"),
        ) + prompts["sections"]
        return _generate(_system_prompt(language), user, max_tokens=700)
    return _summarize_blocking(fit_tokens(new_text, LIVE_TEXT_TOKENS, keep="tail"), language)


def _run_live(text, new_text, previous, language):
    _ensure_model()
    return _live_blocking(text, new_text, previous, language)


def live_insights(text="", new_text="", previous="", language="en"):
    """Resumen en vivo + preguntas clave.

    - ``text``: transcripción completa (la app la manda solo si es corta) → acta desde cero.
    - ``previous`` + ``new_text``: actualización incremental de las notas anteriores.
    Devuelve {"markdown": acta completa, "summary": acta sin preguntas ni tareas,
    "questions": [...], "action_items": [...]} (las pestañas Resumen / Preguntas / Tareas).
    """
    text, new_text, previous = text or "", new_text or "", previous or ""
    if not (text.strip() or new_text.strip()):
        if not previous.strip():
            raise ValueError("no transcript to summarize yet")
        markdown = dedupe_lines(previous)
    else:
        markdown = clean_minutes(_executor.submit(_run_live, text, new_text, previous, language).result())
    rest, questions = split_key_questions(markdown)
    summary, action_items = split_action_items(rest)
    return {"markdown": markdown, "summary": summary, "questions": questions, "action_items": action_items}


# ---------- Preguntas ----------

# Respuestas: "long" (por defecto) explica con detalle; "short" va al grano.
ANSWER_MAX_TOKENS = {"long": 700, "short": 300}


def _ask_blocking(text, question, language, notes="", memory="", detail="long"):
    labels = _prompts(language)["ask"]
    notes = fit_tokens(notes, QA_NOTES_TOKENS, keep="head")
    memory = fit_tokens(memory, QA_MEMORY_TOKENS, keep="head")
    text = fit_tokens(text, QA_RECENT_TOKENS, keep="tail")
    blocks = []
    if notes:
        blocks.append(f"{labels['notes']}\n{notes}")
    if memory:
        blocks.append(f"{labels['memory']}\n{memory}")
    blocks.append(f"{labels['recent'] if (notes or memory) else labels['transcript']}\n{text}")
    detail = detail if detail in ANSWER_MAX_TOKENS else "long"
    blocks.append(f"{labels['question']} {question}\n{labels['instruction_' + detail]}")
    return _generate(
        _system_prompt(language), "\n\n".join(blocks), max_tokens=ANSWER_MAX_TOKENS[detail], temp=0.2
    )


def _run_question(text, question, language, notes="", memory="", detail="long"):
    _ensure_model()
    return _ask_blocking(text, question, language, notes, memory, detail)


def ask_question(text, question, language="es", notes="", memory="", detail="long"):
    """Responde una pregunta sobre la reunión. Lanza excepción si falla.

    ``notes`` (resumen en vivo) y ``memory`` (fragmentos antiguos relevantes,
    traídos del disco) cubren lo que ya no cabe de la transcripción, que se
    recorta a su parte más reciente. Cada bloque tiene su presupuesto de tokens.
    Pasa por el mismo hilo que el resumen (MLX asocia sus streams al hilo que
    cargó el modelo).
    """
    text = _tail_text((text or "").strip(), QA_MAX_CHARS, language)
    return clean_answer(_executor.submit(
        _run_question, text, question, language, notes or "", memory or "", detail or "long"
    ).result())


# Respuestas en modo "Ideas" (lluvia de ideas): reunión + conocimiento + internet.
OPEN_LENGTH = {
    "en": {"long": "Be as complete as the question needs (about 250-500 words).", "short": "Keep it short: 3-5 sentences, or at most 5 steps."},
    "es": {"long": "Sé tan completo como la pregunta lo necesite (unas 250-500 palabras).", "short": "Sé breve: 3-5 frases, o como mucho 5 pasos."},
}


# ---------- Resultados de internet en el prompt ----------

# Presupuesto del bloque de internet: con páginas leídas hay más texto útil.
WEB_TOKENS = 3200
WEB_RESULTS = 6  # resultados que se le pasan a Qwen
WEB_READ_PAGES = 3  # páginas que se abren y leen (además del resumen del buscador)


def _web_block(results, labels):
    """Resultados numerados [1], [2]… para el prompt: el texto leído de la
    página si lo hay, si no el resumen del buscador; las fuentes confiables
    van marcadas."""
    items = []
    for n, r in enumerate(results, start=1):
        mark = f" ({labels['reliable']})" if r.get("trusted") else ""
        body = r.get("content") or r.get("snippet") or ""
        items.append(f"[{n}] {r['title']}{mark} — {r['url']}\n{body}")
    return fit_tokens("\n\n".join(items), WEB_TOKENS, keep="head")


def _sources(results):
    return [
        {"title": r["title"], "url": r["url"], "trusted": bool(r.get("trusted")), "read": bool(r.get("content"))}
        for r in results
    ]


def _open_blocking(text, question, language, notes, memory, results, detail):
    labels = _prompts(language)["ask"]
    detail = detail if detail in ANSWER_MAX_TOKENS else "long"
    blocks = []
    notes = fit_tokens(notes, QA_NOTES_TOKENS, keep="head")
    memory = fit_tokens(memory, QA_MEMORY_TOKENS, keep="head")
    text = fit_tokens(text, QA_RECENT_TOKENS, keep="tail")
    if notes:
        blocks.append(f"{labels['notes']}\n{notes}")
    if memory:
        blocks.append(f"{labels['memory']}\n{memory}")
    if text:
        blocks.append(f"{labels['recent'] if (notes or memory) else labels['transcript']}\n{text}")
    if results:
        blocks.append(f"{labels['web']}\n{_web_block(results, labels)}")
    length = OPEN_LENGTH.get(language, OPEN_LENGTH["en"])[detail]
    instruction = labels["open_instruction"] if results else labels["open_instruction_noweb"]
    blocks.append(f"{labels['question']} {question}\n{instruction} {length}")
    return _generate(
        _system_prompt(language, mode="open"), "\n\n".join(blocks),
        max_tokens=ANSWER_MAX_TOKENS[detail] + 100, temp=0.4,
    )


def _run_open(text, question, language, notes, memory, results, detail):
    _ensure_model()
    return _open_blocking(text, question, language, notes, memory, results, detail)


def open_question(text, question, language="en", notes="", memory="", detail="long", web=True,
                  sites="", read=WEB_READ_PAGES, search=None):
    """Modo «Ideas»: responde con la reunión como contexto + conocimiento propio + internet.

    Solo el texto de la pregunta sale a internet (si ``web``). ``sites``: sitios
    preferidos (se buscan con site:); ``read``: cuántas páginas leer enteras.
    Si la búsqueda falla, responde igual con lo que sabe. Devuelve {"answer", "sources"}.
    """
    results = []
    if web:
        if search is None:
            from web_search import search
        try:
            results = search(question, language=language, sites=sites, read=read, limit=WEB_RESULTS)
        except Exception:
            results = []  # sin internet: responde con la reunión y lo que sabe
    text = _tail_text((text or "").strip(), QA_MAX_CHARS, language)
    answer = clean_answer(_executor.submit(
        _run_open, text, question, language, notes or "", memory or "", results, detail or "long"
    ).result())
    return {"answer": answer, "sources": _sources(results)}


def recall(memory_dir, session, question, recent_text):
    """Fragmentos antiguos de la reunión (en disco) relevantes para la pregunta."""
    if not (memory_dir and session):
        return ""
    from meeting_memory import MeetingMemory

    try:
        hits = MeetingMemory(memory_dir, session).search(
            question, k=QA_MEMORY_HITS, exclude=lambda segment: segment["text"] in recent_text
        )
    except (OSError, ValueError):
        return ""
    return "\n\n".join(f"[{h['start']} – {h['end']}]\n{h['text']}" for h in hits)


# ---------- Ampliar una respuesta con internet ----------

EXTEND_NOTES_TOKENS = 600
EXTEND_PREVIOUS_TOKENS = 700


def _extend_blocking(question, previous, notes, results, language, detail):
    labels = _prompts(language)["ask"]
    blocks = [f"{labels['question']} {question}"]
    previous = fit_tokens(previous, EXTEND_PREVIOUS_TOKENS, keep="head")
    notes = fit_tokens(notes, EXTEND_NOTES_TOKENS, keep="head")
    if previous:
        blocks.append(f"{labels['previous']}\n{previous}")
    if notes:
        blocks.append(f"{labels['notes']}\n{notes}")
    blocks.append(f"{labels['web']}\n{_web_block(results, labels)}")
    detail = detail if detail in ANSWER_MAX_TOKENS else "long"
    blocks.append(labels["extend"])
    return _generate(
        _system_prompt(language), "\n\n".join(blocks), max_tokens=ANSWER_MAX_TOKENS[detail] + 100, temp=0.2
    )


def _run_extend(question, previous, notes, results, language, detail):
    _ensure_model()
    return _extend_blocking(question, previous, notes, results, language, detail)


def extend_with_web(question, previous="", notes="", language="en", detail="long", sites="",
                    read=WEB_READ_PAGES, search=None):
    """Amplía una respuesta con resultados de internet.

    Solo se envía a internet el texto de ``question``; la reunión no sale del
    Mac. Devuelve {"answer", "sources": [{"title", "url"}]}.
    """
    question = (question or "").strip()
    if not question:
        raise ValueError("empty question")
    if search is None:
        from web_search import search
    results = search(question, language=language, sites=sites, read=read, limit=WEB_RESULTS)
    if not results:
        empty = "No encontré resultados en internet para esa pregunta." if language == "es" else \
            "I couldn't find web results for that question."
        return {"answer": empty, "sources": []}
    answer = clean_answer(_executor.submit(
        _run_extend, question, previous or "", notes or "", results, language, detail or "long"
    ).result())
    return {"answer": answer, "sources": _sources(results)}


# ---------- Modo Ideas en dos pasos: respuesta rápida + complemento web ----------
#
# 1. "ask" (modo open, sin web, con "prefetch"): la búsqueda arranca en otro
#    hilo (solo red) mientras Qwen responde con la reunión y lo que sabe.
# 2. "complement": usa esa búsqueda (casi siempre ya lista) y Qwen agrega solo
#    lo que internet aporta: «Puede complementar tu pregunta».

COMPLEMENT_MAX_TOKENS = {"long": 650, "short": 250}
COMPLEMENT_LENGTH = {
    "en": {
        "long": "About 150-250 words: build more elaborate ideas, examples or steps with what the web adds, "
                "in short paragraphs or bullets.",
        "short": "2-4 sentences.",
    },
    "es": {
        "long": "Unas 150-250 palabras: arma ideas más elaboradas, ejemplos o pasos con lo que aporta internet, "
                "en párrafos cortos o viñetas.",
        "short": "2-4 frases.",
    },
}
PREFETCH_TIMEOUT = 40  # segundos; la búsqueda tiene sus propios límites (~20 s en el peor caso)
_NOTHING_NEW_RE = re.compile(r"^\W*(nothing new|nada nuevo)\b", re.IGNORECASE)
_web_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="web")
_web_prefetch = {}


def _prefetch_key(question, language, sites, read):
    return ((question or "").strip(), language, str(sites or ""), int(read or 0))


def prefetch_web(question, language="en", sites="", read=WEB_READ_PAGES, search=None):
    """Empieza la búsqueda en segundo plano (solo red: no usa Qwen)."""
    if search is None:
        from web_search import search
    key = _prefetch_key(question, language, sites, read)
    if key in _web_prefetch:
        return
    while len(_web_prefetch) >= 8:  # las que nunca se usaron (p. ej. la respuesta falló)
        _web_prefetch.pop(next(iter(_web_prefetch)))
    _web_prefetch[key] = _web_pool.submit(search, key[0], language=language, sites=sites, read=read, limit=WEB_RESULTS)


def _web_results(question, language, sites, read, search):
    """Resultados de la búsqueda adelantada, o una búsqueda nueva si no la hay."""
    future = _web_prefetch.pop(_prefetch_key(question, language, sites, read), None)
    if future is None:
        if search is None:
            from web_search import search
        return search(question, language=language, sites=sites, read=read, limit=WEB_RESULTS)
    return future.result(timeout=PREFETCH_TIMEOUT)


def _complement_blocking(question, previous, notes, results, language, detail):
    labels = _prompts(language)["ask"]
    detail = detail if detail in COMPLEMENT_MAX_TOKENS else "long"
    blocks = [f"{labels['question']} {question}"]
    previous = fit_tokens(previous, EXTEND_PREVIOUS_TOKENS, keep="head")
    notes = fit_tokens(notes, EXTEND_NOTES_TOKENS, keep="head")
    if notes:
        blocks.append(f"{labels['notes']}\n{notes}")
    if previous:
        blocks.append(f"{labels['given']}\n{previous}")
    blocks.append(f"{labels['web']}\n{_web_block(results, labels)}")
    length = COMPLEMENT_LENGTH.get(language, COMPLEMENT_LENGTH["en"])[detail]
    blocks.append(f"{labels['complement']} {length}")
    return _generate(
        _system_prompt(language, mode="open"), "\n\n".join(blocks),
        max_tokens=COMPLEMENT_MAX_TOKENS[detail] + 60, temp=0.2,
    )


def _run_complement(question, previous, notes, results, language, detail):
    _ensure_model()
    return _complement_blocking(question, previous, notes, results, language, detail)


def complement_with_web(question, previous="", notes="", language="en", detail="long", sites="",
                        read=WEB_READ_PAGES, search=None):
    """Lo que internet agrega a una respuesta ya dada (modo Ideas, segundo paso).

    Solo la pregunta sale a internet. Devuelve {"answer", "sources"}; "answer"
    vacío si internet no aporta nada nuevo (o no hubo resultados).
    """
    question = (question or "").strip()
    if not question:
        raise ValueError("empty question")
    results = _web_results(question, language, sites, read, search)
    if not results:
        return {"answer": "", "sources": []}
    answer = clean_answer(_executor.submit(
        _run_complement, question, previous or "", notes or "", results, language, detail or "long"
    ).result())
    if _NOTHING_NEW_RE.match(answer) and len(answer) < 80:
        answer = ""
    return {"answer": answer, "sources": _sources(results)}


# ---------- Preguntas en vivo (cada pocos segundos) ----------
#
# Petición liviana, aparte del resumen: solo la parte más reciente de la
# reunión y las preguntas ya listadas; responde con pocas líneas, así cuesta
# unos pocos segundos y puede correr seguido.

QUESTIONS_TEXT_TOKENS = 450
QUESTIONS_KNOWN_TOKENS = 250
QUESTIONS_MAX = 4
QUESTIONS_MAX_TOKENS = 160


def parse_questions(raw, limit=QUESTIONS_MAX):
    """Líneas «- pregunta» del modelo → lista limpia (sin «ninguna», razonamiento ni repetidas)."""
    questions, seen = [], set()
    for line in (raw or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        item = _LIST_PREFIX_RE.sub("", line).strip().strip('"“”«»*').strip()
        if not item or _is_empty_item(item) or looks_like_reasoning(item):
            continue
        if len(item.split()) < 3:
            continue
        if not item.endswith("?"):
            item = item.rstrip(".:;") + "?"
        key = re.sub(r"\W+", "", item.lower())
        if key in seen:
            continue
        seen.add(key)
        questions.append(item[:300])
        if len(questions) >= limit:
            break
    return questions


def _questions_blocking(text, known, language):
    prompts = _prompts(language)
    blocks = []
    if known:
        listed = "\n".join(f"- {question}" for question in known)
        blocks.append(f"{prompts['known']}\n{fit_tokens(listed, QUESTIONS_KNOWN_TOKENS, keep='tail')}")
    blocks.append(f"{prompts['latest']}\n{fit_tokens(text, QUESTIONS_TEXT_TOKENS, keep='tail')}")
    blocks.append(prompts["questions"].format(limit=QUESTIONS_MAX))
    raw = _generate(_system_prompt(language), "\n\n".join(blocks), max_tokens=QUESTIONS_MAX_TOKENS, temp=0.2)
    return parse_questions(raw)


def _run_questions(text, known, language):
    _ensure_model()
    return _questions_blocking(text, known, language)


def live_questions(text="", known=(), language="en"):
    """Preguntas nuevas de la parte más reciente de la reunión (sin repetir ``known``)."""
    text = (text or "").strip()
    if not text:
        return []
    known = [question.strip() for question in known if question and question.strip()]
    return _executor.submit(_run_questions, text, known, language).result()


# ---------- Tareas para Recordatorios ----------

TASKS_NOTES_TOKENS = 1000
TASKS_RECENT_TOKENS = 2500
_LIST_PREFIX_RE = re.compile(r"^(?:[-*•]|\d+[.)])\s*(?:\[[ xX]\]\s*)?")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_NO_VALUE = {"", "-", "none", "n/a", "na", "unknown", "tbd", "ninguno", "ninguna", "nadie", "sin fecha"}


def parse_tasks(raw, limit=MAX_TASKS):
    """Convierte las líneas 'TAREA | RESPONSABLE | FECHA' del modelo en una lista limpia."""
    tasks, seen = [], set()
    for line in (raw or "").splitlines():
        line = _LIST_PREFIX_RE.sub("", line.strip()).strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split("|")]
        title = parts[0].strip(" .").strip()
        if not title or _is_empty_item(title) or title.upper() in ("TASK", "TAREA"):
            continue  # vacío, "- none" o la fila de cabecera repetida
        owner = parts[1] if len(parts) > 1 else ""
        owner = "" if owner.strip(" .").lower() in _NO_VALUE else owner
        due = parts[2] if len(parts) > 2 else ""
        if _DATE_RE.match(due):
            try:
                datetime.date.fromisoformat(due)
            except ValueError:
                due = ""
        else:
            due = ""
        key = title.lower()
        if key in seen:
            continue
        seen.add(key)
        tasks.append({"title": title[:200], "owner": owner[:80], "due": due})
        if len(tasks) >= limit:
            break
    return tasks


def _tasks_blocking(notes, text, language, today):
    labels = _prompts(language)["ask"]
    blocks = [_prompts(language)["tasks"].format(today=today)]
    notes = fit_tokens(notes, TASKS_NOTES_TOKENS, keep="head")
    text = fit_tokens(text, TASKS_RECENT_TOKENS, keep="tail")
    if notes:
        blocks.append(f"{labels['notes']}\n{notes}")
    if text:
        blocks.append(f"{labels['recent'] if notes else labels['transcript']}\n{text}")
    # Sin penalización: el formato "tarea | responsable | fecha" se repite a propósito.
    raw = _generate(
        _system_prompt(language), "\n\n".join(blocks), max_tokens=500, temp=0.1, penalize_repeats=False
    )
    return parse_tasks(raw)


def _run_tasks(notes, text, language, today):
    _ensure_model()
    return _tasks_blocking(notes, text, language, today)


def extract_tasks(notes="", text="", language="en", today=None):
    """Tareas de la reunión: [{"title", "owner", "due" (AAAA-MM-DD o "")}]."""
    if not ((notes or "").strip() or (text or "").strip()):
        raise ValueError("no transcript to extract tasks from yet")
    today = today or datetime.date.today().isoformat()
    text = _tail_text((text or "").strip(), QA_MAX_CHARS, language)
    return _executor.submit(_run_tasks, notes or "", text, language, today).result()


# ---------- Proceso persistente para la app nativa ----------

def _web_options(request):
    """Sitios preferidos («hbr.org, who.int») y si se leen las páginas ("read": "0" = no)."""
    return {
        "sites": request.get("sites") or "",
        "read": WEB_READ_PAGES if request.get("read", "1") != "0" else 0,
    }


def handle_request(request):
    kind = request.get("type")
    language = request.get("lang") or "en"
    if kind == "warmup":
        _executor.submit(_ensure_model).result()
        return {"ready": True, "model": SUMMARY_MODEL}
    if kind == "live":
        return live_insights(
            request.get("text", ""), request.get("new_text", ""), request.get("previous", ""), language
        )
    if kind == "ask":
        question = (request.get("question") or "").strip()
        text = request.get("text") or ""
        if not question:
            raise ValueError("empty question")
        if not text.strip():
            raise ValueError("no transcript to answer from yet")
        memory = recall(request.get("memory_dir"), request.get("session"), question, text)
        detail = request.get("detail") or "long"
        if request.get("mode") == "open":
            if request.get("prefetch") == "1":
                # La búsqueda corre mientras Qwen responde; la usa "complement".
                prefetch_web(question, language, **_web_options(request))
            reply = open_question(
                text, question, language, request.get("notes") or "", memory, detail,
                web=request.get("web", "1") != "0", **_web_options(request),
            )
            reply["used_memory"] = bool(memory)
            return reply
        answer = ask_question(text, question, language, request.get("notes") or "", memory, detail=detail)
        return {"answer": answer, "used_memory": bool(memory), "sources": []}
    if kind == "remember":
        from meeting_memory import MeetingMemory

        memory = MeetingMemory(request.get("memory_dir"), request.get("session"))
        stored = memory.add(int(request.get("index")), request.get("start"), request.get("end"), request.get("text"))
        return {"stored": stored, "segments": len(memory.segments())}
    if kind == "extend":
        return extend_with_web(
            request.get("question"), request.get("previous"), request.get("notes"), language,
            request.get("detail") or "long", **_web_options(request),
        )
    if kind == "questions":
        known = (request.get("known") or "").splitlines()
        return {"questions": live_questions(request.get("text"), known, language)}
    if kind == "complement":
        return complement_with_web(
            request.get("question"), request.get("previous"), request.get("notes"), language,
            request.get("detail") or "long", **_web_options(request),
        )
    if kind == "tasks":
        return {"tasks": extract_tasks(request.get("notes"), request.get("text"), language, request.get("today"))}
    if kind == "unload":
        return {"unloaded": _executor.submit(_unload_model).result()}
    if kind == "summary":
        result, error = build_summary(request.get("text", ""), language)
        if error:
            raise RuntimeError(error)
        return {"markdown": result}
    raise ValueError(f"unknown request type: {kind!r}")


_idle_timer = None


def _schedule_idle_unload():
    """Reinicia la cuenta atrás para liberar Qwen tras un rato sin peticiones."""
    global _idle_timer
    if IDLE_UNLOAD_SECONDS <= 0:
        return
    if _idle_timer is not None:
        _idle_timer.cancel()
    _idle_timer = threading.Timer(IDLE_UNLOAD_SECONDS, lambda: _executor.submit(_unload_model))
    _idle_timer.daemon = True
    _idle_timer.start()


def serve(stdin=None, stdout=None):
    """Proceso persistente: Qwen se carga una vez y queda en memoria.

    Protocolo: un JSON por línea en stdin; por cada uno, una línea JSON en
    stdout con el mismo "id": {"id", "ok": true, "result": {...}} o
    {"id", "ok": false, "error": "..."}. Termina al cerrarse stdin (al salir la app).
    """
    stdin = stdin or sys.stdin
    out = stdout or sys.stdout
    if stdout is None:
        # Cualquier print de librerías va a stderr para no romper el protocolo.
        sys.stdout = sys.stderr
    for raw in stdin:
        raw = raw.strip()
        if not raw:
            continue
        request_id = None
        try:
            request = json.loads(raw)
            request_id = request.get("id")
            reply = {"id": request_id, "ok": True, "result": handle_request(request)}
        except Exception as exc:
            reply = {"id": request_id, "ok": False, "error": str(exc) or exc.__class__.__name__}
        out.write(json.dumps(reply, ensure_ascii=False) + "\n")
        out.flush()
        _schedule_idle_unload()
    if _idle_timer is not None:
        _idle_timer.cancel()
    return 0


def main(argv=None, stdin=None, stdout=None, stderr=None):
    """CLI. La transcripción llega por stdin.

        python summarizer.py serve
        python summarizer.py summary --lang en < transcripcion.txt
        python summarizer.py ask --lang en --question "What was decided?" < transcripcion.txt

    summary/ask escriben el resultado en stdout; ante un error escriben el
    motivo en stderr y terminan con código 1.
    """
    import argparse

    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr

    parser = argparse.ArgumentParser(
        prog="summarizer.py", description="Resumen y preguntas locales con Qwen (mlx-lm)."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("serve", help="proceso persistente (JSON por líneas) para la app nativa")
    summary = commands.add_parser("summary", help="acta de la reunión en Markdown")
    summary.add_argument("--lang", choices=("es", "en"), default="en")
    ask = commands.add_parser("ask", help="responde una pregunta sobre la reunión")
    ask.add_argument("--lang", choices=("es", "en"), default="en")
    ask.add_argument("--question", required=True)
    ask.add_argument("--detail", choices=("long", "short"), default="long")
    args = parser.parse_args(argv)

    if args.command == "serve":
        return serve()

    text = stdin.read()
    if args.command == "summary":
        result, error = build_summary(text, args.lang)
    else:
        result, error = "", None
        question = args.question.strip()
        if not text.strip():
            error = "no hay transcripción sobre la que preguntar"
        elif not question:
            error = "la pregunta está vacía"
        else:
            try:
                result = ask_question(text, question, args.lang, detail=args.detail)
            except Exception as exc:
                error = f"no se pudo responder: {exc}"

    result = (result or "").strip()
    if error or not result:
        stderr.write((error or "el modelo no devolvió texto") + "\n")
        return 1
    stdout.write(result + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
