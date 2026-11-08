"""Resumen, preguntas clave y respuestas locales con Qwen (mlx-lm).

Se usa de tres formas:
- Desde la app Python (``build_summary``).
- Como proceso persistente para la app nativa: ``python summarizer.py serve``
  (Qwen se carga una sola vez y queda en memoria; protocolo JSON por líneas).
- Como CLI puntual: ``python summarizer.py summary|ask ...``.

El contexto/persona del modelo se lee de ``qwen_context.txt`` en cada petición.
"""
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
# Preguntas: se manda la parte más reciente de la reunión (~6-8k tokens).
QA_MAX_CHARS = 24000
MAX_KEY_QUESTIONS = 6

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
            "Use only what the transcript says; do not invent names, dates or numbers."
        ),
        "minutes": (
            "Write short meeting notes in Markdown from the transcript below, "
            "using exactly these sections:\n"
        ),
        "sections": (
            "## Summary\n(2-3 sentences)\n"
            "## Key topics\n(bullets)\n"
            "## Key questions\n(bullets: the most important questions asked or raised, at most 5)\n"
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
        "ask": (
            "Meeting transcript:\n{text}\n\nQuestion: {question}\n"
            "Answer concisely and usefully. If the answer is not in the transcript, "
            "say so clearly instead of making it up."
        ),
        "trimmed": "(… transcript trimmed: only the most recent part …)",
    },
    "es": {
        "role": "Eres un asistente de reuniones.",
        "job": (
            "Tu trabajo es ayudarme a seguir una reunión a partir de su transcripción en vivo. "
            "Responde SIEMPRE en español. Usa solo lo que dice la transcripción; "
            "no inventes nombres, fechas ni cifras."
        ),
        "minutes": (
            "A partir de la siguiente transcripción de reunión, genera un acta breve "
            "en formato Markdown con estas secciones exactas:\n"
        ),
        "sections": (
            "## Resumen\n(2-3 frases)\n"
            "## Temas tratados\n(viñetas)\n"
            "## Preguntas clave\n(viñetas: las preguntas más importantes que se han hecho o planteado, máximo 5)\n"
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
        "ask": (
            "Transcripción de la reunión:\n{text}\n\nPregunta: {question}\n"
            "Responde de forma concisa y útil. Si la respuesta no aparece en la "
            "transcripción, dilo claramente en vez de inventarla."
        ),
        "trimmed": "(… transcripción recortada: solo la parte más reciente …)",
    },
}


def _prompts(language):
    return PROMPTS.get(language, PROMPTS["en"])


def _ensure_model():
    global _model, _tokenizer
    with _lock:
        if _model is None:
            from mlx_lm import load

            _model, _tokenizer = load(SUMMARY_MODEL)


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


def _system_prompt(language):
    prompts = _prompts(language)
    context = load_context()
    if context:
        return f"{context}\n\n{prompts['job']}"
    return f"{prompts['role']} {prompts['job']}"


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


def _generate(system, user, max_tokens, temp=0.3):
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
    result = generate(
        _model,
        _tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=make_sampler(temp=temp),
        verbose=False,
    )
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
        result = _executor.submit(_run, text, language).result()
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


def split_key_questions(markdown, limit=MAX_KEY_QUESTIONS):
    """Separa la sección de preguntas clave del acta: (resto_markdown, [preguntas])."""
    kept, questions = [], []
    in_questions = False
    for line in (markdown or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            title = stripped.lstrip("#").strip().lower()
            in_questions = "question" in title or "pregunta" in title
            if in_questions:
                continue
        elif in_questions:
            item = stripped
            for bullet in ("- ", "* ", "• "):
                if item.startswith(bullet):
                    item = item[len(bullet):].strip()
                    break
            else:
                item = _NUMBERED_RE.sub("", item).strip()
            if item and not _is_empty_item(item) and len(questions) < limit:
                questions.append(item)
            continue
        kept.append(line)
    return "\n".join(kept).strip(), questions


def _live_blocking(text, new_text, previous, language):
    if text.strip():
        return _summarize_blocking(text, language)
    if previous.strip() and new_text.strip():
        # Incremental: notas anteriores + solo lo nuevo, así el costo no crece
        # con la duración de la reunión.
        prompts = _prompts(language)
        user = prompts["update"].format(
            previous=previous.strip(),
            new=_tail_text(new_text.strip(), SINGLE_PASS_CHARS, language),
        ) + prompts["sections"]
        return _generate(_system_prompt(language), user, max_tokens=700)
    return _summarize_blocking(_tail_text(new_text.strip(), SINGLE_PASS_CHARS, language), language)


def _run_live(text, new_text, previous, language):
    _ensure_model()
    return _live_blocking(text, new_text, previous, language)


def live_insights(text="", new_text="", previous="", language="en"):
    """Resumen en vivo + preguntas clave.

    - ``text``: transcripción completa (la app la manda solo si es corta) → acta desde cero.
    - ``previous`` + ``new_text``: actualización incremental de las notas anteriores.
    Devuelve {"markdown": acta completa, "summary": acta sin preguntas, "questions": [...]}.
    """
    text, new_text, previous = text or "", new_text or "", previous or ""
    if not (text.strip() or new_text.strip()):
        if not previous.strip():
            raise ValueError("no transcript to summarize yet")
        markdown = previous.strip()
    else:
        markdown = (_executor.submit(_run_live, text, new_text, previous, language).result() or "").strip()
    summary, questions = split_key_questions(markdown)
    return {"markdown": markdown, "summary": summary, "questions": questions}


# ---------- Preguntas ----------

def _ask_blocking(text, question, language):
    user = _prompts(language)["ask"].format(text=text, question=question)
    return _generate(_system_prompt(language), user, max_tokens=400, temp=0.2)


def _run_question(text, question, language):
    _ensure_model()
    return _ask_blocking(text, question, language)


def ask_question(text, question, language="es"):
    """Responde una pregunta sobre la transcripción. Lanza excepción si falla.

    Pasa por el mismo hilo que el resumen (MLX asocia sus streams al hilo
    que cargó el modelo) y recorta reuniones muy largas para no exceder el
    contexto del modelo.
    """
    text = _tail_text((text or "").strip(), QA_MAX_CHARS, language)
    return _executor.submit(_run_question, text, question, language).result()


# ---------- Proceso persistente para la app nativa ----------

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
        return {"answer": ask_question(text, question, language)}
    if kind == "summary":
        result, error = build_summary(request.get("text", ""), language)
        if error:
            raise RuntimeError(error)
        return {"markdown": result}
    raise ValueError(f"unknown request type: {kind!r}")


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
                result = ask_question(text, question, args.lang)
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
