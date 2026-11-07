import threading
from concurrent.futures import ThreadPoolExecutor

SUMMARY_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"

# Qwen2.5 admite 32k tokens de contexto; ~12k caracteres (~3-4k tokens) caben de
# sobra con margen para la respuesta. Por encima se resume por tramos.
SINGLE_PASS_CHARS = 12000
CHUNK_CHARS = 8000
MAX_NOTES_CHARS = 9000

_lock = threading.Lock()
_model = None
_tokenizer = None
_last_error = None

# MLX asocia sus streams al hilo que carga el modelo. Todas las llamadas al
# modelo pasan por este único hilo para que carga y generación coincidan,
# aunque la GUI lance un QThread distinto en cada resumen.
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="summarizer")


def _ensure_model():
    global _model, _tokenizer
    with _lock:
        if _model is None:
            from mlx_lm import load

            _model, _tokenizer = load(SUMMARY_MODEL)


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
        messages, tokenize=False, add_generation_prompt=True
    )
    result = generate(
        _model,
        _tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=make_sampler(temp=temp),
        verbose=False,
    )
    return _strip_code_fence(result)


def _minutes_prompt(text):
    return (
        "A partir de la siguiente transcripción de reunión, genera un acta breve "
        "en formato Markdown con estas secciones exactas:\n"
        "## Resumen\n(2-3 frases)\n"
        "## Temas tratados\n(viñetas)\n"
        "## Decisiones\n(viñetas)\n"
        "## Tareas / pendientes\n(viñetas, con responsable si se menciona)\n\n"
        "Transcripción:\n"
        f"{text}"
    )


def _summarize_blocking(text, language):
    lang_name = "español" if language == "es" else "English"
    system = f"Eres un asistente de reuniones. Responde SIEMPRE en {lang_name}."

    if len(text) <= SINGLE_PASS_CHARS:
        return _generate(system, _minutes_prompt(text), max_tokens=700)

    # Reunión larga: notas por tramo y acta final sobre las notas, de modo que
    # el final de la reunión (decisiones, pendientes) nunca se pierde.
    notes = []
    for index, chunk in enumerate(split_text(text), start=1):
        note = _generate(
            system,
            "Extrae las ideas clave, decisiones y tareas (con responsable si se "
            "menciona) de este fragmento de reunión, en viñetas cortas.\n\n"
            f"Fragmento {index}:\n{chunk}",
            max_tokens=400,
        )
        notes.append(f"Parte {index}:\n{note}")
    joined = "\n\n".join(notes)
    if len(joined) > MAX_NOTES_CHARS:
        joined = joined[-MAX_NOTES_CHARS:]
    return _generate(
        system,
        _minutes_prompt("(Notas consolidadas de toda la reunión)\n" + joined),
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
