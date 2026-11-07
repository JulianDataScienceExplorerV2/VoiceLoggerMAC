import re
import threading
from collections import deque

import numpy as np

SAMPLE_RATE = 16000

DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo-q4"

MODEL_CHOICES = [
    ("Whisper Large v3 Turbo Q4 (recomendado, bajo consumo)", "mlx-community/whisper-large-v3-turbo-q4"),
    ("Whisper Large v3 Turbo (máxima calidad)", "mlx-community/whisper-large-v3-turbo"),
    ("Whisper Small English (ligero)", "mlx-community/whisper-small.en-mlx"),
    ("Whisper Base English (ultra ligero)", "mlx-community/whisper-base.en-mlx"),
    ("Whisper Tiny (solo pruebas)", "mlx-community/whisper-tiny"),
]

LANGUAGE_CHOICES = [
    ("Inglés", "en"),
    ("Auto-detectar", "auto"),
    ("Español", "es"),
    ("Portugués", "pt"),
]

# Frases típicas que Whisper "alucina" con silencio o ruido (EN / ES / PT).
# Se comparan normalizadas: minúsculas, sin signos ni espacios sobrantes.
BAD_TRANSCRIPTS = {
    "",
    "thank you",
    "thanks",
    "thanks for watching",
    "thank you for watching",
    "please subscribe",
    "subscribe to my channel",
    "like and subscribe",
    "you",
    "bye",
    "gracias",
    "gracias por ver el video",
    "gracias por ver el vídeo",
    "gracias por vernos",
    "suscribete",
    "suscríbete",
    "suscribete al canal",
    "suscríbete al canal",
    "hasta la proxima",
    "hasta la próxima",
    "obrigado",
    "obrigada",
    "obrigado por assistir",
    "inscreva-se no canal",
    "inscreva se no canal",
}

# Fragmentos que descartan la frase aunque haya más texto alrededor.
BAD_SUBSTRINGS = (
    "amara.org",
    "subtitles by",
    "subtítulos realizados por",
    "subtitulos realizados por",
    "subtítulos por la comunidad",
    "legendas pela comunidade",
)

_NORMALIZE_RE = re.compile(r"[^\w\s-]", re.UNICODE)


def _normalize(text):
    return " ".join(_NORMALIZE_RE.sub(" ", (text or "").lower()).split())


def is_bad_transcript(text):
    """True si el texto es vacío, trivial o una alucinación conocida de Whisper."""
    raw = (text or "").strip()
    if len(raw) < 2:
        return True
    lowered = raw.lower()
    if any(fragment in lowered for fragment in BAD_SUBSTRINGS):
        return True
    return _normalize(raw) in BAD_TRANSCRIPTS


def extract_text(result, max_no_speech=0.65, min_avg_logprob=-1.3):
    """Une los segmentos fiables de un resultado de Whisper.

    Si todos los segmentos se descartan por baja confianza, devuelve "" en vez
    de recurrir al texto completo sin filtrar (eso anulaba el filtro).
    """
    parts = []
    for segment in result.get("segments") or []:
        if segment.get("no_speech_prob", 0.0) > max_no_speech:
            continue
        avg_logprob = segment.get("avg_logprob")
        if avg_logprob is not None and avg_logprob < min_avg_logprob:
            continue
        text = (segment.get("text") or "").strip()
        if text:
            parts.append(text)
    if parts:
        return " ".join(parts).strip()
    if not result.get("segments"):
        # Sin información de segmentos no hay nada que filtrar.
        return (result.get("text") or "").strip()
    return ""


BASE_PROMPTS = {
    "en": "Business meeting, software engineering, data analytics, clear English punctuation.",
    "es": "Reunión de trabajo, ingeniería de software, analítica de datos, puntuación clara en español.",
    "pt": "Reunião de trabalho, engenharia de software, análise de dados, pontuação clara em português.",
    "auto": "Meeting conversation, software engineering, data analytics, clear punctuation.",
}


class SpeechTranscriber:
    """Envoltorio de mlx-whisper.

    Usa dos locks: ``_settings_lock`` (corto, lo toma la GUI al cambiar ajustes)
    y ``_infer_lock`` (largo, solo durante la inferencia). Así cambiar un ajuste
    nunca bloquea la interfaz esperando a que termine una transcripción.
    """

    def __init__(self, model_size=DEFAULT_MODEL, language="en", extra_context=""):
        self.model_size = model_size
        self.language = language
        self.extra_context = extra_context.strip()
        self._settings_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._rolling_context = deque(maxlen=5)
        self._is_loaded = False
        self._needs_cache_reset = False
        self._last_error = None

    @property
    def model_loaded(self):
        return self._is_loaded

    def reset_context(self):
        self._rolling_context.clear()

    def apply_settings(self, model_size=None, language=None, extra_context=None):
        changed = False
        with self._settings_lock:
            if model_size and model_size != self.model_size:
                self.model_size = model_size
                self._is_loaded = False
                # La caché de MLX se limpia en el hilo de inferencia, justo antes
                # de la siguiente transcripción (nunca en el hilo de la GUI).
                self._needs_cache_reset = True
                changed = True
            if language is not None and language != self.language:
                self.language = language
                changed = True
            if extra_context is not None and extra_context.strip() != self.extra_context:
                self.extra_context = extra_context.strip()
                changed = True
            if changed:
                self._rolling_context.clear()
        return changed

    def _snapshot(self):
        with self._settings_lock:
            return self.model_size, self.language, self.extra_context

    def _reset_model_cache_if_needed(self):
        with self._settings_lock:
            needed = self._needs_cache_reset
            self._needs_cache_reset = False
        if not needed:
            return
        try:
            from mlx_whisper.transcribe import ModelHolder

            ModelHolder.model = None
            ModelHolder.model_path = None
            import mlx.core as mx

            mx.clear_cache()
        except Exception:
            pass

    def warmup(self):
        """Carga el modelo (descargándolo si hace falta) con 0.5 s de silencio.

        Debe llamarse desde el mismo hilo que luego transcribe, para que MLX
        use siempre el mismo stream.
        """
        model_size, _, _ = self._snapshot()
        with self._infer_lock:
            self._reset_model_cache_if_needed()
            try:
                import mlx_whisper

                mlx_whisper.transcribe(
                    np.zeros(SAMPLE_RATE // 2, dtype=np.float32),
                    path_or_hf_repo=model_size,
                    language="en",
                    temperature=0.0,
                    verbose=None,
                )
                self._is_loaded = True
                return True
            except Exception as exc:
                self._last_error = f"no se pudo precargar el modelo: {exc}"
                return False

    def _build_prompt(self, language, extra_context):
        lang = "en" if language in (None, "", "auto") else language
        parts = [BASE_PROMPTS.get(lang, BASE_PROMPTS["auto"])]
        if extra_context:
            parts.append(extra_context[:300])
        history = " ".join(list(self._rolling_context)).strip()
        if history:
            parts.append(history[-220:])
        return " ".join(parts).strip()

    def transcribe(self, audio_data, sample_rate=SAMPLE_RATE):
        if audio_data is None or len(audio_data) == 0:
            return "", self.language or "en"

        audio = np.asarray(audio_data, dtype=np.float32)
        if sample_rate != SAMPLE_RATE:
            from scipy.signal import resample_poly
            from math import gcd

            g = gcd(SAMPLE_RATE, sample_rate)
            audio = resample_poly(audio, SAMPLE_RATE // g, sample_rate // g).astype(np.float32)

        with self._infer_lock:
            try:
                import mlx_whisper
            except ImportError as exc:
                self._last_error = f"mlx-whisper no está instalado: {exc}"
                return "", self.language or "en"

            self._reset_model_cache_if_needed()
            model_size, language, extra_context = self._snapshot()

            target_lang = None if language in (None, "", "auto") else language
            if model_size.endswith(".en"):
                target_lang = "en"

            prompt = self._build_prompt(language, extra_context)

            try:
                result = mlx_whisper.transcribe(
                    audio,
                    path_or_hf_repo=model_size,
                    language=target_lang,
                    initial_prompt=prompt,
                    # Con fallback de temperatura Whisper reintenta si detecta
                    # repeticiones (compression ratio) en vez de quedarse en bucle.
                    temperature=(0.0, 0.2, 0.4),
                    condition_on_previous_text=False,
                    word_timestamps=False,
                    verbose=None,
                )
            except Exception as exc:
                self._last_error = str(exc)
                return "", self.language or "en"

            self._is_loaded = True
            full_text = extract_text(result)
            detected_lang = result.get("language") or target_lang or "en"

            if is_bad_transcript(full_text):
                return "", detected_lang

            self._rolling_context.append(full_text)
            return full_text, detected_lang

    def get_last_error(self):
        return self._last_error
