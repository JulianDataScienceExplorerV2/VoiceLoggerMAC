"""Pruebas del núcleo (sin Qt ni MLX): `python3 -m unittest discover -s tests`."""
import io
import os
import sys
import time
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_capture import CHUNK_SAMPLES, SAMPLE_RATE, PhraseSegmenter, prepare_phrase  # noqa: E402
import summarizer  # noqa: E402
from summarizer import _tail_text, split_text  # noqa: E402
from transcriber import extract_text, is_bad_transcript  # noqa: E402


class ExtractTextTests(unittest.TestCase):
    def test_keeps_reliable_segments(self):
        result = {"segments": [
            {"text": " Hola equipo", "no_speech_prob": 0.01, "avg_logprob": -0.2},
            {"text": " ruido", "no_speech_prob": 0.9, "avg_logprob": -0.2},
        ]}
        self.assertEqual(extract_text(result), "Hola equipo")

    def test_all_filtered_returns_empty_not_raw_text(self):
        # Regresión: antes caía a result["text"] y se colaba la alucinación.
        result = {
            "text": " Thanks for watching",
            "segments": [{"text": " Thanks for watching", "no_speech_prob": 0.95, "avg_logprob": -1.8}],
        }
        self.assertEqual(extract_text(result), "")

    def test_no_segments_uses_text(self):
        self.assertEqual(extract_text({"text": " hola "}), "hola")


class BadTranscriptTests(unittest.TestCase):
    def test_known_hallucinations(self):
        for text in ["", ".", "Thank you.", "Thanks for watching!", "Gracias por ver el video.",
                     "¡Suscríbete!", "Subtítulos realizados por la comunidad de Amara.org",
                     "Subtitles by the Amara.org community"]:
            self.assertTrue(is_bad_transcript(text), text)

    def test_real_speech_passes(self):
        for text in ["Vamos a revisar el sprint de UPTC", "Thank you for the update on the API"]:
            self.assertFalse(is_bad_transcript(text), text)


class SplitTextTests(unittest.TestCase):
    def test_chunks_respect_limit_and_keep_tail(self):
        lines = [f"[00:00:{i % 60:02d}] frase número {i}" for i in range(500)]
        text = "\n".join(lines)
        chunks = split_text(text, max_chars=1000)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c) <= 1000 for c in chunks))
        self.assertIn("frase número 499", chunks[-1])  # el final no se pierde
        self.assertEqual("\n".join(chunks), text)

    def test_giant_line_is_split(self):
        chunks = split_text("x" * 2500, max_chars=1000)
        self.assertEqual([len(c) for c in chunks], [1000, 1000, 500])


class TailTextTests(unittest.TestCase):
    def test_short_text_is_untouched(self):
        self.assertEqual(_tail_text("hola\nequipo", 100), "hola\nequipo")

    def test_long_text_keeps_recent_whole_lines(self):
        text = "\n".join(f"[00:00:{i:02d}] linea {i}" for i in range(50))
        out = _tail_text(text, 120)
        self.assertTrue(out.startswith("(… transcripción recortada"))
        body = out.split("\n", 1)[1]
        self.assertTrue(body.startswith("[00:00:"))  # no empieza a mitad de línea
        self.assertTrue(out.endswith("linea 49"))


class CliTests(unittest.TestCase):
    """CLI que llama la app nativa: sin MLX, con el modelo simulado."""

    def _run(self, argv, stdin_text):
        out, err = io.StringIO(), io.StringIO()
        code = summarizer.main(argv, stdin=io.StringIO(stdin_text), stdout=out, stderr=err)
        return code, out.getvalue(), err.getvalue()

    def test_summary_ok(self):
        with mock.patch.object(summarizer, "build_summary", return_value=("## Resumen\nok", None)) as fake:
            code, out, err = self._run(["summary", "--lang", "en"], "[10:00:00] hola")
        self.assertEqual((code, out, err), (0, "## Resumen\nok\n", ""))
        fake.assert_called_once_with("[10:00:00] hola", "en")

    def test_summary_error_goes_to_stderr(self):
        with mock.patch.object(summarizer, "build_summary", return_value=("", "sin modelo")):
            code, out, err = self._run(["summary"], "texto")
        self.assertEqual((code, out, err), (1, "", "sin modelo\n"))

    def test_ask_passes_question_starting_with_dash(self):
        with mock.patch.object(summarizer, "ask_question", return_value="Respuesta") as fake:
            code, out, _ = self._run(["ask", "--lang", "es", "--question=-¿qué se decidió?"], "texto")
        self.assertEqual((code, out), (0, "Respuesta\n"))
        fake.assert_called_once_with("texto", "-¿qué se decidió?", "es")

    def test_ask_without_transcript_fails(self):
        with mock.patch.object(summarizer, "ask_question") as fake:
            code, _, err = self._run(["ask", "--question", "hola"], "   ")
        self.assertEqual(code, 1)
        self.assertIn("no hay transcripción", err)
        fake.assert_not_called()

    def test_ask_exception_is_reported(self):
        with mock.patch.object(summarizer, "ask_question", side_effect=RuntimeError("boom")):
            code, _, err = self._run(["ask", "--question", "hola"], "texto")
        self.assertEqual(code, 1)
        self.assertIn("boom", err)


def _tone(seconds, amp=0.1, freq=220.0):
    t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class SegmenterTests(unittest.TestCase):
    def _run(self, signal):
        import queue

        phrases = []
        seg = PhraseSegmenter(on_phrase=phrases.append)
        q = queue.Queue()
        seg.start(q)
        for i in range(0, len(signal) - CHUNK_SAMPLES + 1, CHUNK_SAMPLES):
            q.put(signal[i:i + CHUNK_SAMPLES])
        deadline = time.time() + 5
        while not q.empty() and time.time() < deadline:
            time.sleep(0.01)  # stop() corta de inmediato: dejamos que consuma todo
        time.sleep(0.05)
        seg.stop()
        return phrases

    def test_speech_between_silences_makes_one_phrase(self):
        silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
        phrases = self._run(np.concatenate([silence, _tone(1.5), silence]))
        self.assertEqual(len(phrases), 1)
        audio, _ts = phrases[0]
        self.assertGreater(len(audio), SAMPLE_RATE)

    def test_pure_silence_makes_nothing(self):
        self.assertEqual(self._run(np.zeros(SAMPLE_RATE * 3, dtype=np.float32)), [])

    def test_prepare_phrase_is_bounded(self):
        out = prepare_phrase(_tone(1.0, amp=0.9))
        self.assertLessEqual(float(np.max(np.abs(out))), 0.95)


if __name__ == "__main__":
    unittest.main()
