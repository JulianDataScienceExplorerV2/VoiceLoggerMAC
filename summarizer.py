import threading

SUMMARY_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"

_lock = threading.Lock()
_model = None
_tokenizer = None
_last_error = None


def _ensure_model():
    global _model, _tokenizer
    with _lock:
        if _model is None:
            from mlx_lm import load

            _model, _tokenizer = load(SUMMARY_MODEL)


def build_summary(text, language="es"):
    global _last_error
    text = (text or "").strip()
    if not text:
        return "", "no hay texto para resumir"

    if len(text) > 6000:
        text = text[:6000]

    try:
        _ensure_model()
    except Exception as exc:
        _last_error = f"no se pudo cargar el modelo de resumen: {exc}"
        return "", _last_error

    lang_name = "español" if language == "es" else "English"
    system = (
        f"Eres un asistente de reuniones. Responde SIEMPRE en {lang_name}."
    )
    user = (
        "A partir de la siguiente transcripción de reunión, genera un acta breve "
        "en formato Markdown con estas secciones exactas:\n"
        "## Resumen\n(2-3 frases)\n"
        "## Temas tratados\n(viñetas)\n"
        "## Decisiones\n(viñetas)\n"
        "## Tareas / pendientes\n(viñetas, con responsable si se menciona)\n\n"
        "Transcripción:\n"
        f"{text}"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    try:
        prompt = _tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        result = generate(
            _model,
            _tokenizer,
            prompt=prompt,
            max_tokens=700,
            sampler=make_sampler(temp=0.3),
            verbose=False,
        )
        result = result.strip()
        if result.startswith("```"):
            lines = result.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            result = "\n".join(lines).strip()
        _last_error = None
        return result, None
    except Exception as exc:
        _last_error = str(exc)
        return "", _last_error


def get_last_error():
    return _last_error
