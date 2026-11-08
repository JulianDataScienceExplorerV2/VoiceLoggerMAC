"""Pruebas del núcleo (sin Qt ni MLX): `python3 -m unittest discover -s tests`."""
import io
import json
import os
import tempfile
import sys
import time
import types
import unittest
import urllib.parse
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_capture import CHUNK_SAMPLES, SAMPLE_RATE, PhraseSegmenter, prepare_phrase  # noqa: E402
import summarizer  # noqa: E402
import web_search  # noqa: E402
from meeting_memory import MeetingMemory, tokenize  # noqa: E402
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
        fake.assert_called_once_with("texto", "-¿qué se decidió?", "es", detail="long")

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


class ContextFileTests(unittest.TestCase):
    def test_ignores_comments_and_unfilled_placeholders(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ctx.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write("# comentario\nYou are an expert in XXXXXXXX.\nYou are an expert in data analytics.\n\nBe concise.\n")
            self.assertEqual(
                summarizer.load_context(path, example_path=""),
                "You are an expert in data analytics.\n\nBe concise.",
            )

    def test_creates_file_from_example(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ctx.txt")
            example = os.path.join(tmp, "example.txt")
            with open(example, "w", encoding="utf-8") as f:
                f.write("Be concise.\n")
            self.assertEqual(summarizer.load_context(path, example_path=example), "Be concise.")
            self.assertTrue(os.path.exists(path))

    def test_repo_template_has_placeholders_disabled(self):
        context = summarizer.load_context(
            os.path.join(tempfile.mkdtemp(), "ctx.txt"), example_path=summarizer.CONTEXT_EXAMPLE_FILE
        )
        self.assertNotIn("XXXX", context)
        self.assertIn("concise", context)

    def test_system_prompt_uses_context_and_language(self):
        with mock.patch.object(summarizer, "load_context", return_value="You are an expert in fintech."):
            prompt = summarizer._system_prompt("en")
        self.assertTrue(prompt.startswith("You are an expert in fintech."))
        self.assertIn("Always answer in English", prompt)
        with mock.patch.object(summarizer, "load_context", return_value=""):
            self.assertIn("Responde SIEMPRE en español", summarizer._system_prompt("es"))


class KeyQuestionsTests(unittest.TestCase):
    def test_extracts_questions_and_removes_section(self):
        md = (
            "## Summary\nWe reviewed Q3.\n"
            "## Key questions\n- Do we ship before the freeze?\n* Who owns the fix?\n1. What about pricing?\n"
            "## Decisions\n- Ship it"
        )
        summary, questions = summarizer.split_key_questions(md)
        self.assertEqual(questions, ["Do we ship before the freeze?", "Who owns the fix?", "What about pricing?"])
        self.assertNotIn("Key questions", summary)
        self.assertIn("## Decisions\n- Ship it", summary)

    def test_filters_empty_answers(self):
        for filler in ["- None", "- N/A.", "- No questions were raised yet.", "- (bullets)", "- Ninguna"]:
            _, questions = summarizer.split_key_questions(f"## Preguntas clave\n{filler}\n## Decisiones\n- x")
            self.assertEqual(questions, [], filler)

    def test_limit(self):
        md = "## Key questions\n" + "\n".join(f"- Q{i}?" for i in range(10))
        self.assertEqual(len(summarizer.split_key_questions(md, limit=3)[1]), 3)


class LiveInsightsTests(unittest.TestCase):
    def test_full_text_goes_to_model_and_questions_are_split(self):
        md = "## Summary\nOk.\n## Key questions\n- Why?\n"
        with mock.patch.object(summarizer, "_run_live", return_value=md) as fake:
            result = summarizer.live_insights(text="[10:00] hi", language="en")
        fake.assert_called_once_with("[10:00] hi", "", "", "en")
        self.assertEqual(result, {"markdown": md.strip(), "summary": "## Summary\nOk.", "questions": ["Why?"],
                                  "action_items": []})

    def test_nothing_new_returns_previous_without_model(self):
        with mock.patch.object(summarizer, "_run_live") as fake:
            result = summarizer.live_insights(previous="## Summary\nOld", new_text="  ")
        fake.assert_not_called()
        self.assertEqual(result["summary"], "## Summary\nOld")

    def test_no_text_at_all_fails(self):
        with self.assertRaises(ValueError):
            summarizer.live_insights()


class ThinkingModelTests(unittest.TestCase):
    def test_strip_thinking(self):
        self.assertEqual(summarizer._strip_thinking("<think>hmm\nok</think>\n## Summary\nX"), "## Summary\nX")
        self.assertEqual(summarizer._strip_thinking("razono...</think>Respuesta"), "Respuesta")
        self.assertEqual(summarizer._strip_thinking("Respuesta<think>cortado"), "Respuesta")
        self.assertEqual(summarizer._strip_thinking("## Summary\nSin pensar"), "## Summary\nSin pensar")

    def test_generate_disables_thinking_and_cleans_output(self):
        captured = {}

        class FakeTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                captured.update(kwargs)
                return "PROMPT"

        fake_mlx = types.ModuleType("mlx_lm")
        fake_mlx.generate = lambda model, tokenizer, prompt, max_tokens, sampler, verbose: (
            "<think>planning</think>\n```markdown\n## Summary\nOk\n```"
        )
        fake_utils = types.ModuleType("mlx_lm.sample_utils")
        fake_utils.make_sampler = lambda temp: None
        with mock.patch.dict(sys.modules, {"mlx_lm": fake_mlx, "mlx_lm.sample_utils": fake_utils}), \
                mock.patch.object(summarizer, "_tokenizer", FakeTokenizer()), \
                mock.patch.object(summarizer, "_model", object()):
            out = summarizer._generate("system", "user", max_tokens=10)
        self.assertEqual(out, "## Summary\nOk")
        self.assertIs(captured["enable_thinking"], False)


class RepetitionTests(unittest.TestCase):
    LOOP = "# Resumen\n- **Tema**: Sandwich\n- **Tareas**:\n  - Slicing pork\n" + \
        "  - Toasting whole pork shoulder\n" * 80 + "  -"

    def test_loop_is_collapsed_in_minutes(self):
        out = summarizer.clean_minutes(self.LOOP)
        self.assertEqual(out.count("Toasting whole pork shoulder"), 1)
        self.assertEqual(out.splitlines()[-1], "  - Toasting whole pork shoulder")  # sin la viñeta vacía

    def test_same_line_with_other_bullet_or_bold_is_a_repeat(self):
        text = "## Notes\n- **Budget** is approved.\n* Budget is approved\n1. budget IS approved!"
        self.assertEqual(summarizer.dedupe_lines(text), "## Notes\n- **Budget** is approved.")

    def test_repeats_in_other_sections_and_short_labels_are_kept(self):
        text = ("## Option A\n**Pros:**\n- fast\n**Cons:**\n- pricey\n**Pros:**\n- more\n---\n"
                "## Key questions\n- Who pays for the new office?\n"
                "## Summary\n- Who pays for the new office?")
        self.assertEqual(summarizer.dedupe_lines(text), text)

    def test_consecutive_short_repeats_and_long_sections_are_cut(self):
        self.assertEqual(summarizer.dedupe_lines("- Pork\n- Pork\n- Pork"), "- Pork")
        many = "## Points\n" + "\n".join(f"- point number {i}" for i in range(40))
        self.assertEqual(len(summarizer.dedupe_lines(many).splitlines()), 1 + summarizer.MAX_LINES_PER_SECTION)

    def test_broken_previous_notes_are_cleaned_before_reuse(self):
        with mock.patch.object(summarizer, "_generate", return_value="## Summary\nNew") as fake:
            summarizer._live_blocking("", "more talk", self.LOOP, "en")
        prompt = fake.call_args.args[1]
        self.assertEqual(prompt.count("Toasting whole pork shoulder"), 1)
        result = summarizer.live_insights(previous=self.LOOP, new_text=" ")
        self.assertEqual(result["markdown"].count("Toasting whole pork shoulder"), 1)

    def test_generate_penalizes_repeats_except_for_tasks(self):
        calls = []

        class FakeTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return "PROMPT"

        def fake_generate(model, tokenizer, prompt, max_tokens, sampler, verbose, prefill_step_size=None,
                          logits_processors=None):
            calls.append(logits_processors)
            return "ok"

        fake_mlx = types.ModuleType("mlx_lm")
        fake_mlx.generate = fake_generate
        fake_utils = types.ModuleType("mlx_lm.sample_utils")
        fake_utils.make_sampler = lambda temp: None
        fake_utils.make_logits_processors = lambda **kwargs: [("penalty", kwargs["repetition_penalty"])]
        with mock.patch.dict(sys.modules, {"mlx_lm": fake_mlx, "mlx_lm.sample_utils": fake_utils}), \
                mock.patch.object(summarizer, "_tokenizer", FakeTokenizer()), \
                mock.patch.object(summarizer, "_model", object()):
            summarizer._generate("system", "user", max_tokens=10)
            summarizer._generate("system", "user", max_tokens=10, penalize_repeats=False)
        self.assertEqual(calls, [[("penalty", summarizer.REPETITION_PENALTY)], None])


class AskWithNotesTests(unittest.TestCase):
    def test_notes_go_into_the_prompt(self):
        with mock.patch.object(summarizer, "_generate", return_value="ok") as fake, \
                mock.patch.object(summarizer, "load_context", return_value=""):
            summarizer._ask_blocking("[10:00] hi", "Who owns it?", "en", notes="## Summary\nQ3 review")
        user_prompt = fake.call_args[0][1]
        self.assertIn("Notes of the meeting so far:\n## Summary\nQ3 review", user_prompt)
        self.assertIn("Most recent part of the transcript:\n[10:00] hi", user_prompt)

    def test_without_notes_uses_plain_prompt(self):
        with mock.patch.object(summarizer, "_generate", return_value="ok") as fake, \
                mock.patch.object(summarizer, "load_context", return_value=""):
            summarizer._ask_blocking("[10:00] hi", "Who?", "en")
        self.assertTrue(fake.call_args[0][1].startswith("Meeting transcript:"))

    def test_serve_forwards_notes(self):
        with mock.patch.object(summarizer, "ask_question", return_value="42") as fake:
            summarizer.handle_request({"type": "ask", "text": "t", "question": "q", "lang": "en", "notes": "n"})
        fake.assert_called_once_with("t", "q", "en", "n", "", detail="long")


class AnswerDetailTests(unittest.TestCase):
    def _prompt_and_tokens(self, **kwargs):
        with mock.patch.object(summarizer, "_generate", return_value="ok") as fake, \
                mock.patch.object(summarizer, "load_context", return_value=""):
            summarizer._ask_blocking("[10:00] hi", "Who?", "en", **kwargs)
        return fake.call_args[0][1], fake.call_args[1]["max_tokens"]

    def test_long_is_default_and_elaborate(self):
        prompt, tokens = self._prompt_and_tokens()
        self.assertIn("thorough, well-organized answer", prompt)
        self.assertEqual(tokens, 700)

    def test_short_answers(self):
        prompt, tokens = self._prompt_and_tokens(detail="short")
        self.assertIn("Answer concisely (2-4 sentences)", prompt)
        self.assertEqual(tokens, 300)

    def test_unknown_detail_falls_back_to_long(self):
        self.assertEqual(self._prompt_and_tokens(detail="weird")[1], 700)

    def test_serve_forwards_detail(self):
        with mock.patch.object(summarizer, "ask_question", return_value="x") as fake:
            summarizer.handle_request({"type": "ask", "text": "t", "question": "q", "lang": "es", "detail": "short"})
        self.assertEqual(fake.call_args[1], {"detail": "short"})

    def test_cli_detail_flag(self):
        with mock.patch.object(summarizer, "ask_question", return_value="x") as fake:
            summarizer.main(["ask", "--question", "q", "--detail", "short"], stdin=io.StringIO("texto"),
                            stdout=io.StringIO(), stderr=io.StringIO())
        self.assertEqual(fake.call_args[1], {"detail": "short"})


class TasksTests(unittest.TestCase):
    def test_parse_tasks_formats(self):
        raw = (
            "- TASK | OWNER | DUE\n"
            "- Fix the permissions copy | Julian | 2026-11-13\n"
            "* [ ] Share the draft in the channel | - | -\n"
            "2. Review pricing | Ana | next week\n"
            "- Fix the permissions copy | Julian | 2026-11-13\n"
            "- Send budget | Bob | 2026-02-30\n"
        )
        self.assertEqual(summarizer.parse_tasks(raw), [
            {"title": "Fix the permissions copy", "owner": "Julian", "due": "2026-11-13"},
            {"title": "Share the draft in the channel", "owner": "", "due": ""},
            {"title": "Review pricing", "owner": "Ana", "due": ""},
            {"title": "Send budget", "owner": "Bob", "due": ""},
        ])

    def test_parse_tasks_none(self):
        for raw in ["- none", "- ninguna", "", "- (bullets)"]:
            self.assertEqual(summarizer.parse_tasks(raw), [], raw)

    def test_parse_tasks_limit(self):
        raw = "\n".join(f"- Task {i} | - | -" for i in range(20))
        self.assertEqual(len(summarizer.parse_tasks(raw)), summarizer.MAX_TASKS)

    def test_tasks_request_uses_today_and_notes(self):
        with mock.patch.object(summarizer, "_generate", return_value="- Ship it | Ana | 2026-11-14") as fake, \
                mock.patch.object(summarizer, "load_context", return_value=""), \
                mock.patch.object(summarizer, "_ensure_model"):
            reply = summarizer.handle_request({"type": "tasks", "notes": "## Summary\nX", "text": "[10:00] ship",
                                               "lang": "en", "today": "2026-11-08"})
        self.assertEqual(reply, {"tasks": [{"title": "Ship it", "owner": "Ana", "due": "2026-11-14"}]})
        prompt = fake.call_args[0][1]
        self.assertIn("today is 2026-11-08", prompt)
        self.assertIn("Notes of the meeting so far:\n## Summary\nX", prompt)

    def test_tasks_without_text_fails(self):
        with self.assertRaises(ValueError):
            summarizer.extract_tasks("", "  ")


class SummaryTabsTests(unittest.TestCase):
    def test_live_splits_summary_questions_and_tasks(self):
        md = ("## Summary\nQ3 review.\n## Key questions\n- Ship before freeze?\n"
              "## Decisions\n- Ship it\n## Action items\n- Julian: fix copy by Friday\n- None\n")
        with mock.patch.object(summarizer, "_run_live", return_value=md):
            result = summarizer.live_insights(text="[10:00] x")
        self.assertEqual(result["summary"], "## Summary\nQ3 review.\n## Decisions\n- Ship it")
        self.assertEqual(result["questions"], ["Ship before freeze?"])
        self.assertEqual(result["action_items"], ["Julian: fix copy by Friday"])
        self.assertEqual(result["markdown"], md.strip())  # full notes kept for the next update

    def test_spanish_headings(self):
        md = "## Resumen\nX\n## Tareas / pendientes\n1. Enviar acta\n## Preguntas clave\n- ¿Fecha?"
        rest, tasks = summarizer.split_action_items(md)
        self.assertEqual(tasks, ["Enviar acta"])
        self.assertNotIn("Tareas", rest)


DDG_SAMPLE = """
<div class="result results_links results_links_deep web-result">
  <h2 class="result__title"><a rel="nofollow" class="result__a"
     href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fopenai.com%2Findex%2Fwhisper%2F&amp;rut=abc">Introducing <b>Whisper</b></a></h2>
  <a class="result__snippet" href="#">Whisper is a <b>speech recognition</b> model &amp; more.</a>
</div>
<div class="result result--ad"><a class="result__a" href="https://duckduckgo.com/y.js?ad=1">Ad</a></div>
<div class="result"><h2><a class="result__a" href="https://github.com/openai/whisper">openai/whisper</a></h2>
  <div class="result__snippet">Robust Speech Recognition via Large-Scale Weak Supervision</div></div>
"""

WIKI_SAMPLE = json.dumps({"query": {"pages": {
    "2": {"index": 2, "title": "Speech recognition", "fullurl": "https://en.wikipedia.org/wiki/Speech_recognition",
          "extract": "Speech recognition is an interdisciplinary subfield."},
    "1": {"index": 1, "title": "Whisper (speech recognition system)",
          "fullurl": "https://en.wikipedia.org/wiki/Whisper_(speech_recognition_system)",
          "extract": "Whisper is a machine learning model for speech recognition."},
}}})


class WebSearchTests(unittest.TestCase):
    def test_parse_duckduckgo_unwraps_links_and_skips_ads(self):
        results = web_search.parse_duckduckgo(DDG_SAMPLE)
        self.assertEqual([r["url"] for r in results], ["https://openai.com/index/whisper/", "https://github.com/openai/whisper"])
        self.assertEqual(results[0]["title"], "Introducing Whisper")
        self.assertEqual(results[0]["snippet"], "Whisper is a speech recognition model & more.")

    def test_parse_wikipedia_keeps_ranking(self):
        results = web_search.parse_wikipedia(WIKI_SAMPLE)
        self.assertEqual(results[0]["title"], "Whisper (speech recognition system)")
        self.assertIn("machine learning model", results[0]["snippet"])

    def test_falls_back_to_wikipedia_when_duckduckgo_fails(self):
        calls = []

        def fetch(url):
            calls.append(url)
            if "duckduckgo" in url:
                raise OSError("blocked")
            self.assertIn("es.wikipedia.org", url)
            return WIKI_SAMPLE

        results = web_search.search("whisper", language="es", fetch=fetch)
        self.assertEqual(len(results), 2)
        self.assertEqual(len(calls), 2)

    def test_only_the_question_goes_out(self):
        seen = []
        web_search.search("¿Qué es Whisper?", fetch=lambda url: seen.append(url) or DDG_SAMPLE)
        self.assertTrue(all("Whisper" in url for url in seen))

    def test_all_sources_down_raises(self):
        def fetch(url):
            raise OSError("offline")
        with self.assertRaises(web_search.WebSearchError):
            web_search.search("whisper", fetch=fetch)


PAGE_HTML = """<html><head><title>T</title><script>var tracking = "whisper whisper whisper whisper";</script>
<style>p { color: red }</style></head><body>
<nav><ul><li>Home</li><li>About Whisper and every other product we sell today online</li></ul></nav>
<header><p>Subscribe to our newsletter for weekly updates on many different topics</p></header>
<article>
<p>Our company was founded many years ago in a small garage with three people.</p>
<p>Whisper is a speech recognition model trained on 680,000 hours of multilingual audio data.</p>
<p>It handles accents, background noise and technical language better than older systems.</p>
<p>Short.</p>
</article>
<footer><p>Copyright notice and the cookie policy for this website and its partners</p></footer>
</body></html>"""


class WebSearchQualityTests(unittest.TestCase):
    """site: (preferidos y exclusiones), fuentes confiables primero y lectura de páginas."""

    def test_parse_sites_accepts_urls_and_operators(self):
        self.assertEqual(web_search.parse_sites("hbr.org, https://www.McKinsey.com/insights site:who.int hbr.org nada x.org"),
                         ["hbr.org", "mckinsey.com", "who.int"])  # máximo 3
        self.assertEqual(web_search.parse_sites(""), [])

    def test_trusted_domains(self):
        for url in ("https://en.wikipedia.org/wiki/X", "https://www.nasa.gov/a", "https://mit.edu/x",
                    "https://www.gov.uk/y", "https://www.minsalud.gov.co/z", "https://ox.ac.uk/a", "https://who.int/b",
                    "https://www.reuters.com/c"):
            self.assertTrue(web_search.is_trusted(url), url)
        for url in ("https://randomblog.com/x", "https://abc.go.com/show", "https://fakewikipedia.org/x", ""):
            self.assertFalse(web_search.is_trusted(url), url)

    def test_queries_exclude_social_sites_and_target_preferred_ones(self):
        preferred, general = web_search.build_queries("team motivation", ["hbr.org", "mckinsey.com"])
        self.assertEqual(preferred, ["team motivation site:hbr.org", "team motivation site:mckinsey.com"])
        self.assertTrue(general.startswith("team motivation -site:pinterest.com"))
        self.assertEqual(web_search.build_queries("x")[0], [])

    def test_rank_preferred_then_trusted_then_rest(self):
        results = [{"url": u, "title": u} for u in (
            "https://blog.example.com/a", "https://www.pinterest.com/pin/1", "https://en.wikipedia.org/wiki/A",
            "https://hbr.org/2020/x", "https://blog.example.com/a/", "https://other.com/b")]
        ranked = [r["url"] for r in web_search.rank(results, ["hbr.org"])]
        self.assertEqual(ranked, ["https://hbr.org/2020/x", "https://en.wikipedia.org/wiki/A",
                                  "https://blog.example.com/a", "https://other.com/b"])

    def test_search_with_preferred_sites(self):
        queries = []

        def fetch(url):
            query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)["q"][0]
            queries.append(query)
            if "site:hbr.org" in query:
                return ('<a class="result__a" href="https://hbr.org/motivation">HBR on motivation</a>'
                        '<a class="result__snippet">How leaders motivate</a>')
            return DDG_SAMPLE

        results = web_search.search("motivation", fetch=fetch, sites="hbr.org")
        self.assertEqual(sorted(queries), sorted([
            "motivation site:hbr.org",
            "motivation " + " ".join(f"-site:{s}" for s in web_search.EXCLUDED_SITES)]))
        self.assertEqual(results[0]["url"], "https://hbr.org/motivation")
        self.assertEqual([r["trusted"] for r in results], [True, False, False])

    def test_empty_search_with_exclusions_is_retried_plain(self):
        queries = []

        def fetch(url):
            query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)["q"][0]
            queries.append(query)
            return "" if "-site:" in query else DDG_SAMPLE

        results = web_search.search("whisper", fetch=fetch)
        self.assertEqual(queries[-1], "whisper")
        self.assertEqual(len(results), 2)

    def test_extract_relevant_text_skips_menus_scripts_and_footers(self):
        text = web_search.extract_relevant_text(PAGE_HTML, "What is Whisper speech recognition?")
        self.assertIn("Whisper is a speech recognition model", text)
        self.assertNotIn("tracking", text)
        self.assertNotIn("newsletter", text)
        self.assertNotIn("cookie", text)
        self.assertNotIn("every other product", text)
        self.assertNotIn("Short.", text)
        # el párrafo más relevante entra aunque no quepa todo
        small = web_search.extract_relevant_text(PAGE_HTML, "speech recognition multilingual audio", max_chars=100)
        self.assertTrue(small.startswith("Whisper is a speech recognition model"))
        self.assertLessEqual(len(small), 101)

    def test_read_pages_only_public_https(self):
        results = [{"url": u, "title": "t", "snippet": "s"} for u in (
            "http://insecure.com/a", "https://192.168.1.1/admin", "https://localhost/x",
            "https://www.youtube.com/watch?v=1", "https://site.com/doc.pdf", "https://good.com/page")]
        fetched = []
        web_search.read_pages(results, "whisper", 2, fetch_page=lambda url: fetched.append(url) or PAGE_HTML)
        self.assertEqual(fetched, ["https://good.com/page"])
        self.assertIn("speech recognition", results[-1]["content"])
        self.assertNotIn("content", results[0])

    def test_read_pages_failure_or_timeout_keeps_snippet(self):
        results = [{"url": "https://slow.com/a", "title": "t", "snippet": "s"},
                   {"url": "https://broken.com/b", "title": "t", "snippet": "s"}]

        def fetch_page(url):
            if "slow" in url:
                time.sleep(0.5)
                return PAGE_HTML
            raise OSError("404")

        start = time.monotonic()
        web_search.read_pages(results, "whisper", 2, fetch_page=fetch_page, deadline=0.1)
        self.assertLess(time.monotonic() - start, 0.4)
        self.assertFalse(any("content" in r for r in results))

    def test_redirects_only_to_public_https(self):
        self.assertTrue(web_search._is_public_https("https://example.com/a"))
        for url in ("http://example.com", "https://10.0.0.1/", "https://[::1]/", "https://printer.local/", "file:///etc"):
            self.assertFalse(web_search._is_public_https(url), url)

    def test_prompt_uses_page_text_and_marks_reliable_sources(self):
        labels = summarizer._prompts("en")["ask"]
        block = summarizer._web_block([
            {"title": "Wiki", "url": "https://en.wikipedia.org/wiki/W", "snippet": "short", "trusted": True,
             "content": "Full paragraph from the page."},
            {"title": "Blog", "url": "https://blog.com/x", "snippet": "Blog snippet", "trusted": False},
        ], labels)
        self.assertIn("[1] Wiki (reliable source) — https://en.wikipedia.org/wiki/W\nFull paragraph from the page.", block)
        self.assertIn("[2] Blog — https://blog.com/x\nBlog snippet", block)
        self.assertEqual(summarizer._sources([{"title": "Wiki", "url": "u", "trusted": True, "content": "x"}]),
                         [{"title": "Wiki", "url": "u", "trusted": True, "read": True}])

    def test_serve_passes_sites_and_reading(self):
        with mock.patch.object(summarizer, "open_question", return_value={"answer": "o", "sources": []}) as op:
            summarizer.handle_request({"type": "ask", "text": "t", "question": "q", "mode": "open",
                                       "sites": "hbr.org", "read": "0"})
        self.assertEqual(op.call_args[1]["sites"], "hbr.org")
        self.assertEqual(op.call_args[1]["read"], 0)


class ExtendWithWebTests(unittest.TestCase):
    def test_extend_cites_sources(self):
        results = [{"title": "Whisper", "url": "https://openai.com/w", "snippet": "ASR model"}]
        with mock.patch.object(summarizer, "_generate", return_value="Meeting: X. Web: ASR [1].") as fake, \
                mock.patch.object(summarizer, "load_context", return_value=""), \
                mock.patch.object(summarizer, "_ensure_model"):
            reply = summarizer.extend_with_web("What is Whisper?", previous="They use Whisper.", notes="## Summary\nY",
                                               language="en", search=lambda q, **kwargs: results)
        self.assertEqual(reply, {"answer": "Meeting: X. Web: ASR [1].", "sources": [
            {"title": "Whisper", "url": "https://openai.com/w", "trusted": False, "read": False}]})
        prompt = fake.call_args[0][1]
        self.assertIn("Answer based on the meeting:\nThey use Whisper.", prompt)
        self.assertIn("[1] Whisper — https://openai.com/w\nASR model", prompt)

    def test_no_results(self):
        reply = summarizer.extend_with_web("x", language="es", search=lambda q, **kwargs: [])
        self.assertEqual(reply["sources"], [])
        self.assertIn("No encontré", reply["answer"])

    def test_serve_extend(self):
        with mock.patch.object(summarizer, "extend_with_web", return_value={"answer": "a", "sources": []}) as fake:
            reply = summarizer.handle_request({"type": "extend", "question": "q", "previous": "p", "notes": "n",
                                               "lang": "en", "detail": "short"})
        self.assertEqual(reply, {"answer": "a", "sources": []})
        fake.assert_called_once_with("q", "p", "n", "en", "short", sites="", read=summarizer.WEB_READ_PAGES)


class ModelLoadTests(unittest.TestCase):
    def test_old_mlx_lm_gives_update_hint(self):
        fake_mlx = types.ModuleType("mlx_lm")

        def load(name):
            raise ValueError("Model type lfm2 not supported.")

        fake_mlx.load = load
        with mock.patch.dict(sys.modules, {"mlx_lm": fake_mlx}), mock.patch.object(summarizer, "_model", None):
            with self.assertRaises(RuntimeError) as ctx:
                summarizer._ensure_model()
        self.assertIn("-U mlx-lm", str(ctx.exception))


LEAKED = (
    "Where to grab the fish (mouth vs other parts) - mentioned in the\n\n"
    "The user wants me to create meeting notes from a transcript. The transcript appears to be from a cooking "
    "show.\n1. ## Summary (2-3 sentences)\n2. ## Key topics (bullets)\n\nLet me analyze the transcript:\n"
    "- The speaker is preparing snake head fish noodles\n\nSo the key question is about where to grab the fish (mouth"
)


class ReasoningLeakTests(unittest.TestCase):
    def test_leaked_reasoning_is_rejected(self):
        with self.assertRaises(summarizer.ModelOutputError):
            summarizer.clean_minutes(LEAKED)
        with self.assertRaises(summarizer.ModelOutputError):
            summarizer.clean_minutes("The user wants me to summarize.\n## Summary\nI need to think")

    def test_reasoning_then_final_notes_keeps_the_final_notes(self):
        md = ("Okay, so let me draft.\n## Summary\ndraft\nWait, better:\n"
              "## Summary\nCooking show about snakehead fish noodles.\n## Key questions\n- Where to grab the fish?")
        self.assertEqual(summarizer.clean_minutes(md),
                         "## Summary\nCooking show about snakehead fish noodles.\n## Key questions\n- Where to grab the fish?")

    def test_normal_notes_untouched(self):
        md = "## Summary\nQ3 review.\n## Decisions\n- Ship it"
        self.assertEqual(summarizer.clean_minutes(md), md)
        self.assertEqual(summarizer.clean_minutes("## Resumen\nX"), "## Resumen\nX")

    def test_answers(self):
        self.assertEqual(summarizer.clean_answer("They grab it by the skull."), "They grab it by the skull.")
        self.assertEqual(summarizer.clean_answer("The user is asking where.\nLet me check.\nAnswer: By the skull."),
                         "By the skull.")
        with self.assertRaises(summarizer.ModelOutputError):
            summarizer.clean_answer("The user wants to know where to grab the fish. Let me look at the (mouth")

    def test_live_keeps_previous_on_leak(self):
        with mock.patch.object(summarizer, "_run_live", return_value=LEAKED):
            with self.assertRaises(summarizer.ModelOutputError):
                summarizer.live_insights(text="[10:00] x")

    def test_build_summary_reports_leak_as_error(self):
        with mock.patch.object(summarizer, "_run", return_value=LEAKED):
            result, error = summarizer.build_summary("[10:00] x", "en")
        self.assertEqual(result, "")
        self.assertIn("razonamiento", error)

    def test_system_prompt_forbids_reasoning(self):
        with mock.patch.object(summarizer, "load_context", return_value=""):
            self.assertIn("do not show your reasoning", summarizer._system_prompt("en"))


class EvaluateTranscriptTests(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
        import evaluar_transcripcion
        self.ev = evaluar_transcripcion

    def test_normalize_matches_spoken_and_written_forms(self):
        self.assertEqual(self.ev.normalize("It's gonna be 7 pounds, cuz it's 8:30 a.m. [music]"),
                         self.ev.normalize("it's going to be seven pounds because it's 8 30 am"))

    def test_trims_reference_and_finds_lost_chunk(self):
        intro = "this video starts long before the recording began with plenty of words "
        said = ("so this is borough market the most famous place to eat in london where you will find "
                "a huge array of street food and then we try the oysters from mersea island which are great ")
        lost = "this grilled cheese sandwich has reached a mythical status level in the whole market today "
        rest = "and next up we are trying the black pig sandwich with pork crackling and salsa verde on top"
        ref = self.ev.normalize(intro + said + lost + rest)
        blocks = [("10:00:00", self.ev.normalize(said)), ("10:00:30", self.ev.normalize(rest))]
        result = self.ev.evaluate(ref, blocks)
        self.assertEqual(result["ref_words"], len(self.ev.normalize(said + lost + rest)))  # intro trimmed
        self.assertEqual(result["missing"][0][0], len(lost.split()))
        self.assertIn("mythical status", result["missing"][0][2])
        self.assertAlmostEqual(result["accuracy"], 1 - len(lost.split()) / result["ref_words"])


class OpenModeTests(unittest.TestCase):
    """Modo «Ideas»: reunión como contexto + conocimiento propio + internet."""

    def _run(self, **kwargs):
        with mock.patch.object(summarizer, "_generate", return_value="Idea: try a pilot [1].") as fake, \
                mock.patch.object(summarizer, "load_context", return_value=""), \
                mock.patch.object(summarizer, "_ensure_model"):
            reply = summarizer.open_question("[10:00] we need more users", "How do we grow?", "en",
                                             notes="## Summary\nGrowth", **kwargs)
        return reply, fake.call_args

    def test_uses_web_and_brainstorm_prompts(self):
        results = [{"title": "Growth loops", "url": "https://x.org/g", "snippet": "Referrals work"}]
        reply, call = self._run(search=lambda q, **kwargs: results)
        self.assertEqual(reply, {"answer": "Idea: try a pilot [1].", "sources": [
            {"title": "Growth loops", "url": "https://x.org/g", "trusted": False, "read": False}]})
        system, user = call[0][0], call[0][1]
        self.assertIn("brainstorming meeting", system)
        self.assertNotIn("Use only what the transcript says", system)
        self.assertIn("[1] Growth loops — https://x.org/g", user)
        self.assertIn("numbered step-by-step instructions", user)
        self.assertIn("give a clear recommendation first and then explain why", user)
        self.assertIn("do not use labels", user)
        self.assertIn("Notes of the meeting so far", user)

    def test_offline_still_answers(self):
        def broken(q, **kwargs):
            raise OSError("offline")
        reply, call = self._run(search=broken)
        self.assertEqual(reply["sources"], [])
        self.assertEqual(reply["answer"], "Idea: try a pilot [1].")
        self.assertNotIn("Web search results", call[0][1])

    def test_web_disabled_never_searches(self):
        def must_not_search(q, **kwargs):
            raise AssertionError("searched with web disabled")
        reply, _ = self._run(search=must_not_search, web=False)
        self.assertEqual(reply["sources"], [])

    def test_serve_routes_by_mode(self):
        with mock.patch.object(summarizer, "open_question", return_value={"answer": "o", "sources": []}) as op, \
                mock.patch.object(summarizer, "ask_question", return_value="m") as mt:
            summarizer.handle_request({"type": "ask", "text": "t", "question": "q", "mode": "open", "web": "0"})
            summarizer.handle_request({"type": "ask", "text": "t", "question": "q"})
        self.assertFalse(op.call_args[1]["web"])
        mt.assert_called_once()

    def test_meeting_mode_keeps_strict_prompt(self):
        with mock.patch.object(summarizer, "load_context", return_value=""):
            self.assertIn("Use only what the transcript says", summarizer._system_prompt("en"))
            self.assertIn("lluvia de ideas", summarizer._system_prompt("es", mode="open"))


class TwoStepIdeasTests(unittest.TestCase):
    """Modo Ideas: respuesta rápida de Qwen + «Puede complementar tu pregunta» con internet."""

    RESULTS = [{"title": "Growth", "url": "https://en.wikipedia.org/wiki/G", "snippet": "s", "trusted": True,
                "content": "Referral programs grow users."}]

    def setUp(self):
        summarizer._web_prefetch.clear()

    def _patched(self, output):
        return mock.patch.object(summarizer, "_generate", return_value=output), \
            mock.patch.object(summarizer, "load_context", return_value=""), \
            mock.patch.object(summarizer, "_ensure_model")

    def test_quick_answer_does_not_mention_web(self):
        gen, ctx, ensure = self._patched("Try a pilot.")
        with gen as fake, ctx, ensure:
            summarizer.open_question("[10:00] we need users", "How do we grow?", "en", web=False)
        self.assertIn("add your own knowledge.", fake.call_args[0][1])
        self.assertNotIn("web results", fake.call_args[0][1])

    def test_ask_prefetches_while_answering_and_complement_reuses_it(self):
        calls = []

        def search(q, **kwargs):
            calls.append((q, kwargs))
            return self.RESULTS

        gen, ctx, ensure = self._patched("Referrals help [1].")
        with gen as fake, ctx, ensure, mock.patch.dict(sys.modules, {"web_search": types.SimpleNamespace(search=search)}):
            quick = summarizer.handle_request({
                "type": "ask", "mode": "open", "web": "0", "prefetch": "1", "text": "[10:00] we need users",
                "question": "How do we grow?", "sites": "hbr.org", "read": "1",
            })
            self.assertEqual(quick["sources"], [])
            extra = summarizer.handle_request({
                "type": "complement", "question": "How do we grow?", "previous": "Try a pilot.",
                "notes": "", "sites": "hbr.org", "read": "1",
            })
        self.assertEqual(calls, [("How do we grow?", {"language": "en", "sites": "hbr.org",
                                                      "read": summarizer.WEB_READ_PAGES,
                                                      "limit": summarizer.WEB_RESULTS})])
        self.assertEqual(extra, {"answer": "Referrals help [1].", "sources": [
            {"title": "Growth", "url": "https://en.wikipedia.org/wiki/G", "trusted": True, "read": True}]})
        prompt = fake.call_args[0][1]
        self.assertIn("Answer already given:\nTry a pilot.", prompt)
        self.assertIn("Do not repeat the answer already given", prompt)
        self.assertIn("(reliable source)", prompt)
        self.assertEqual(summarizer._web_prefetch, {})

    def test_complement_without_prefetch_searches_now(self):
        gen, ctx, ensure = self._patched("NADA NUEVO.")
        with gen, ctx, ensure:
            reply = summarizer.complement_with_web("¿Cuánto es 2+2?", previous="4", language="es",
                                                   search=lambda q, **kwargs: self.RESULTS)
        self.assertEqual(reply["answer"], "")  # internet no aportó nada
        self.assertEqual(len(reply["sources"]), 1)

    def test_complement_with_no_results_skips_the_model(self):
        with mock.patch.object(summarizer, "_generate") as fake:
            reply = summarizer.complement_with_web("x", search=lambda q, **kwargs: [])
        fake.assert_not_called()
        self.assertEqual(reply, {"answer": "", "sources": []})

    def test_old_prefetches_are_dropped(self):
        for n in range(12):
            summarizer.prefetch_web(f"q{n}", search=lambda q, **kwargs: [])
        self.assertLessEqual(len(summarizer._web_prefetch), 8)


class LiveQuestionsTests(unittest.TestCase):
    """Preguntas en vivo cada pocos segundos (petición liviana, aparte del resumen)."""

    def test_parse_questions(self):
        raw = ("- Why does your organization exist?\n- none\n1. **What is the budget for Q3**\n"
               "- ok?\n- Why does your organization exist?\n- The user wants questions\n## Notes\n"
               "- How do we measure ROAS?\n- Who owns the launch?\n- When is the deadline?")
        self.assertEqual(summarizer.parse_questions(raw), [
            "Why does your organization exist?", "What is the budget for Q3?",
            "How do we measure ROAS?", "Who owns the launch?"])  # máximo 4
        self.assertEqual(summarizer.parse_questions("- ninguna"), [])

    def test_prompt_has_recent_text_and_known_questions(self):
        with mock.patch.object(summarizer, "_generate", return_value="- Who pays?  \n- How much does it cost?") as fake, \
                mock.patch.object(summarizer, "load_context", return_value=""), \
                mock.patch.object(summarizer, "_ensure_model"):
            reply = summarizer.handle_request({"type": "questions", "text": "[10:00] How much does it cost?",
                                               "known": "Who pays for it?\nWhen do we start?", "lang": "en"})
        self.assertEqual(reply, {"questions": ["How much does it cost?"]})  # «Who pays?» es muy corta
        prompt = fake.call_args[0][1]
        self.assertIn("Questions already listed:\n- Who pays for it?\n- When do we start?", prompt)
        self.assertIn("Latest part of the meeting:\n[10:00] How much does it cost?", prompt)
        self.assertIn("copy it word for word", prompt)
        self.assertEqual(fake.call_args[1]["max_tokens"], summarizer.QUESTIONS_MAX_TOKENS)

    def test_no_text_no_model(self):
        with mock.patch.object(summarizer, "_generate") as fake:
            self.assertEqual(summarizer.live_questions("  "), [])
        fake.assert_not_called()


class TopicSitesTests(unittest.TestCase):
    def test_topics(self):
        self.assertEqual(web_search.topic_sites("How do I memoize a React component with hooks?"), ["developer.mozilla.org"])
        self.assertEqual(web_search.topic_sites("Cómo mido el ROAS de una campaña con GA4"), ["thinkwithgoogle.com"])
        self.assertEqual(web_search.topic_sites("Paginate an API in FastAPI and track conversions"),
                         ["stackoverflow.com", "thinkwithgoogle.com"])
        self.assertEqual(web_search.topic_sites("What are the next steps? Let's rest and lead"), [])

    def test_search_adds_topic_site_after_preferred(self):
        queries = []

        def fetch(url):
            queries.append(urllib.parse.parse_qs(urllib.parse.urlparse(url).query)["q"][0])
            return DDG_SAMPLE

        web_search.search("React hooks", fetch=fetch, sites="hbr.org")
        self.assertIn("React hooks site:hbr.org", queries)
        self.assertIn("React hooks site:developer.mozilla.org", queries)
        queries.clear()
        web_search.search("React hooks", fetch=fetch, topics=False)
        self.assertEqual(len(queries), 1)

    def test_developer_and_marketing_sources_are_trusted(self):
        for url in ("https://react.dev/learn", "https://stackoverflow.com/q/1", "https://www.thinkwithgoogle.com/x",
                    "https://docs.docker.com/a", "https://www.postgresql.org/docs/"):
            self.assertTrue(web_search.is_trusted(url), url)


class FitTokensTests(unittest.TestCase):
    def test_keeps_short_text(self):
        self.assertEqual(summarizer.fit_tokens("hola equipo", 100), "hola equipo")

    def test_tail_keeps_recent_whole_lines(self):
        text = "\n".join(f"[10:{i:02d}] linea numero {i}" for i in range(200))
        out = summarizer.fit_tokens(text, 100, keep="tail")
        self.assertLessEqual(summarizer._count_tokens(out), 100)
        self.assertTrue(out.endswith("linea numero 199"))
        self.assertTrue(out.startswith("[10:"))

    def test_head_keeps_beginning(self):
        text = "## Summary\nFirst\n" + "\n".join(f"- item {i}" for i in range(300))
        out = summarizer.fit_tokens(text, 50, keep="head")
        self.assertTrue(out.startswith("## Summary\nFirst"))
        self.assertLessEqual(summarizer._count_tokens(out), 50)


class MeetingMemoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.memory = MeetingMemory(self.tmp.name, "2026-11-08_10-00-00")
        self.memory.add(0, "10:00:00", "10:04:00", "[10:00:00] Welcome everyone, agenda is the Q3 roadmap.")
        self.memory.add(1, "10:04:00", "10:08:00", "[10:04:00] The marketing budget for Q3 is 40 thousand dollars.")
        self.memory.add(2, "10:08:00", "10:12:00", "[10:08:00] Julian owns the permissions fix, deadline Friday.")

    def tearDown(self):
        self.tmp.cleanup()

    def test_add_is_idempotent_and_persisted(self):
        self.assertFalse(self.memory.add(1, "x", "y", "duplicate"))
        reopened = MeetingMemory(self.tmp.name, "2026-11-08_10-00-00")
        self.assertEqual([s["index"] for s in reopened.segments()], [0, 1, 2])

    def test_search_finds_relevant_segment(self):
        hits = self.memory.search("What was the budget for marketing?", k=1)
        self.assertEqual([h["index"] for h in hits], [1])

    def test_search_ignores_accents_and_excludes(self):
        self.memory.add(3, "10:12", "10:16", "[10:12] La decisión final fue lanzar el viernes.")
        self.assertEqual([h["index"] for h in self.memory.search("decision lanzar")], [3])
        self.assertEqual(self.memory.search("decision lanzar", exclude=lambda s: s["index"] == 3), [])

    def test_stopword_only_query_returns_nothing(self):
        self.assertEqual(self.memory.search("what is the"), [])
        self.assertNotIn("the", tokenize("the budget"))

    def test_rejects_unsafe_session_names(self):
        for bad in ["../etc", "a/b", "", "x" * 80]:
            with self.assertRaises(ValueError):
                MeetingMemory(self.tmp.name, bad)

    def test_corrupt_line_is_skipped(self):
        with open(self.memory.path, "a", encoding="utf-8") as f:
            f.write("{roto\n")
        self.assertEqual(len(self.memory.segments()), 3)


class MemoryRequestsTests(unittest.TestCase):
    def test_remember_then_ask_uses_disk_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = {"memory_dir": tmp, "session": "s1"}
            reply = summarizer.handle_request(dict(base, type="remember", index="0", start="10:00",
                                                   end="10:04", text="[10:00] Budget for Q3 is 40k."))
            self.assertEqual(reply, {"stored": True, "segments": 1})
            with mock.patch.object(summarizer, "ask_question", return_value="40k") as fake:
                reply = summarizer.handle_request(dict(base, type="ask", text="[11:00] other topic",
                                                       question="What is the Q3 budget?", lang="en"))
            self.assertEqual(reply, {"answer": "40k", "used_memory": True, "sources": []})
            memory = fake.call_args[0][4]
            self.assertIn("Budget for Q3 is 40k.", memory)
            self.assertIn("[10:00 – 10:04]", memory)

    def test_segment_already_in_recent_text_is_not_repeated(self):
        with tempfile.TemporaryDirectory() as tmp:
            summarizer.handle_request({"type": "remember", "memory_dir": tmp, "session": "s1",
                                       "index": "0", "start": "a", "end": "b", "text": "[10:00] Budget is 40k."})
            self.assertEqual(summarizer.recall(tmp, "s1", "budget", "[10:00] Budget is 40k.\n[10:01] ok"), "")

    def test_memory_and_notes_go_into_prompt(self):
        with mock.patch.object(summarizer, "_generate", return_value="ok") as fake, \
                mock.patch.object(summarizer, "load_context", return_value=""):
            summarizer._ask_blocking("[11:00] now", "Budget?", "en", notes="## Summary\nX", memory="[10:00 – 10:04]\nold")
        prompt = fake.call_args[0][1]
        self.assertLess(prompt.index("Notes of the meeting so far:"), prompt.index("Earlier parts of the meeting"))
        self.assertLess(prompt.index("Earlier parts of the meeting"), prompt.index("Most recent part of the transcript:"))

    def test_unload_request(self):
        with mock.patch.object(summarizer, "_model", None):
            self.assertEqual(summarizer.handle_request({"type": "unload"}), {"unloaded": False})
        with mock.patch.object(summarizer, "_model", object()), mock.patch.object(summarizer, "_tokenizer", object()):
            self.assertEqual(summarizer.handle_request({"type": "unload"}), {"unloaded": True})
            self.assertIsNone(summarizer._model)


class ServeTests(unittest.TestCase):
    def _serve(self, *lines):
        out = io.StringIO()
        summarizer.serve(stdin=io.StringIO("\n".join(lines) + "\n"), stdout=out)
        return [json.loads(line) for line in out.getvalue().splitlines()]

    def test_live_and_ask_round_trip(self):
        live = {"markdown": "m", "summary": "s", "questions": ["q?"]}
        with mock.patch.object(summarizer, "live_insights", return_value=live), \
                mock.patch.object(summarizer, "ask_question", return_value="42"):
            replies = self._serve(
                json.dumps({"id": "a", "type": "live", "text": "hola", "lang": "en"}),
                "",
                json.dumps({"id": "b", "type": "ask", "text": "hola", "question": "¿qué?", "lang": "en"}),
            )
        self.assertEqual(replies, [
            {"id": "a", "ok": True, "result": live},
            {"id": "b", "ok": True, "result": {"answer": "42", "used_memory": False, "sources": []}},
        ])

    def test_errors_do_not_stop_the_server(self):
        replies = self._serve(
            "esto no es json",
            json.dumps({"id": "x", "type": "nope"}),
            json.dumps({"id": "y", "type": "ask", "text": "", "question": "hola"}),
        )
        self.assertEqual([r["ok"] for r in replies], [False, False, False])
        self.assertEqual([r["id"] for r in replies], [None, "x", "y"])
        self.assertIn("unknown request type", replies[1]["error"])


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
