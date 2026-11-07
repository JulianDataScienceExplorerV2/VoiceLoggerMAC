"""Pruebas del núcleo (sin Qt ni MLX): `python3 -m unittest discover -s tests`."""
import os
import sys
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_capture import CHUNK_SAMPLES, SAMPLE_RATE, PhraseSegmenter, prepare_phrase  # noqa: E402
from summarizer import split_text  # noqa: E402
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
