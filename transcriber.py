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

BAD_TRANSCRIPTS = {
    "",
    ".",
    "..",
    "...",
    "!",
    "?",
    "Thank you.",
    "Thanks for watching.",
    "Thank you for watching.",
    "Thank you for watching!",
    "Please subscribe.",
    "Subscribe to my channel.",
    "Subtitles by the Amara.org community",
}

BASE_PROMPTS = {
    "en": "Business meeting, software engineering, data analytics, clear English punctuation.",
    "es": "Reunión de trabajo, ingeniería de software, analítica de datos, puntuación clara en español.",
    "pt": "Reunião de trabalho, engenharia de software, análise de dados, pontuação clara em português.",
    "auto": "Meeting conversation, software engineering, data analytics, clear punctuation.",
}


class SpeechTranscriber:
    def __init__(self, model_size=DEFAULT_MODEL, language="en", extra_context=""):
        self.model_size = model_size
        self.language = language
        self.extra_context = extra_context.strip()
        self._lock = threading.Lock()
        self._rolling_context = deque(maxlen=5)
        self._is_loaded = False
        self._last_error = None

    @property
    def model_loaded(self):
        return self._is_loaded

    def reset_context(self):
        self._rolling_context.clear()

    def apply_settings(self, model_size=None, language=None, extra_context=None):
        changed = False
        with self._lock:
            if model_size and model_size != self.model_size:
                self.model_size = model_size
                self._is_loaded = False
                changed = True
                try:
                    from mlx_whisper.transcribe import ModelHolder
                    ModelHolder.model = None
                    ModelHolder.model_path = None
                    import mlx.core as mx
                    mx.clear_cache()
                except Exception:
                    pass
            if language is not None and language != self.language:
                self.language = language
                changed = True
            if extra_context is not None and extra_context.strip() != self.extra_context:
                self.extra_context = extra_context.strip()
                changed = True
            if changed:
                self._rolling_context.clear()
        return changed

    def load_model(self):
        import mlx_whisper  # noqa: F401
        self._is_loaded = True
        return True

    def _build_prompt(self):
        lang = "en" if self.language in (None, "", "auto") else self.language
        base = BASE_PROMPTS.get(lang, BASE_PROMPTS["auto"])
        parts = [base]
        if self.extra_context:
            parts.append(self.extra_context[:300])
        history = " ".join(self._rolling_context).strip()
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

        with self._lock:
            try:
                import mlx_whisper
            except ImportError as exc:
                self._last_error = f"mlx-whisper no está instalado: {exc}"
                return "", self.language or "en"

            if not self._is_loaded:
                self._is_loaded = True

            target_lang = None if self.language in (None, "", "auto") else self.language
            if self.model_size.endswith(".en"):
                target_lang = "en"

            prompt = self._build_prompt()

            try:
                result = mlx_whisper.transcribe(
                    audio,
                    path_or_hf_repo=self.model_size,
                    language=target_lang,
                    initial_prompt=prompt,
                    temperature=0.0,
                    condition_on_previous_text=False,
                    word_timestamps=False,
                    verbose=None,
                )
            except Exception as exc:
                self._last_error = str(exc)
                return "", self.language or "en"

            text_parts = []
            for segment in result.get("segments") or []:
                if segment.get("no_speech_prob", 0.0) > 0.65:
                    continue
                avg_logprob = segment.get("avg_logprob")
                if avg_logprob is not None and avg_logprob < -1.3:
                    continue
                text = (segment.get("text") or "").strip()
                if text:
                    text_parts.append(text)

            full_text = " ".join(text_parts).strip() or (result.get("text") or "").strip()
            detected_lang = result.get("language") or target_lang or "en"

            if full_text in BAD_TRANSCRIPTS or len(full_text) < 2:
                return "", detected_lang

            self._rolling_context.append(full_text)
            return full_text, detected_lang

    def get_last_error(self):
        return self._last_error
