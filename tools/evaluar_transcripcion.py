"""Califica una transcripción de VoiceLogger contra un texto de referencia.

    python3 tools/evaluar_transcripcion.py reunion_2026-11-08.txt referencia.txt

- La transcripción es el TXT que guarda la app ("Guardar transcripción…").
- La referencia es el texto "correcto" (p. ej. los subtítulos de YouTube
  descargados en .txt). Se recorta sola al tramo que cubre la grabación.

Muestra la precisión (1 - WER), palabras cambiadas/omitidas/añadidas, los
tramos perdidos y el texto añadido que no está en la referencia.
"""
import argparse
import difflib
import re
import sys

NUMBERS = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
           "seven": "7", "eight": "8", "nine": "9", "ten": "10", "twelve": "12", "thirty": "30"}
FILLERS = {"uh", "um", "mm", "mhm", "mmm", "ooh", "oh"}


def normalize(text):
    t = text.lower().replace("’", "'")
    t = re.sub(r"\[(music|laughter|singing|applause| __ )\]|>>", " ", t)
    t = re.sub(r"(\d+):(\d+)", r"\1 \2", t)
    t = re.sub(r"\bgonna\b", "going to", t)
    t = re.sub(r"\bgotta\b", "got to", t)
    t = re.sub(r"\bwanna\b", "want to", t)
    t = re.sub(r"'cause\b|\bcuz\b", "because", t)
    t = re.sub(r"\ba\.m\.?", "am", t)
    t = re.sub(r"\bp\.m\.?", "pm", t)
    t = t.replace("-", " ")
    t = re.sub(r"[^\w\s']", " ", t)
    return [NUMBERS.get(w, w) for w in t.split() if w not in FILLERS]


def read_app_transcript(path):
    """[(hora, palabras)] de un TXT de la app ("[HH:MM:SS] [EN] texto")."""
    blocks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = re.match(r"\[(\d\d:\d\d:\d\d)\] \[\w+\] (.*)", line.strip())
            if m:
                blocks.append((m.group(1), normalize(m.group(2))))
    return blocks


def word_errors(ref, hyp):
    """Distancia de Levenshtein entre listas de palabras."""
    prev = list(range(len(hyp) + 1))
    for i, rw in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, hw in enumerate(hyp, 1):
            a, b, c = prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw)
            cur[j] = a if a < b and a < c else (b if b < c else c)
        prev = cur
    return prev[-1]


def trim_to_overlap(ref, hyp, min_block=5):
    """Recorta la referencia al tramo que cubre la transcripción."""
    blocks = [b for b in difflib.SequenceMatcher(None, ref, hyp, autojunk=False).get_matching_blocks()
              if b.size >= min_block]
    if not blocks:
        return ref
    return ref[blocks[0].a: blocks[-1].a + blocks[-1].size]


def evaluate(ref_words, blocks):
    hyp, hyp_time = [], []
    for ts, words in blocks:
        hyp += words
        hyp_time += [ts] * len(words)
    ref = trim_to_overlap(ref_words, hyp)
    errors = word_errors(ref, hyp)
    sm = difflib.SequenceMatcher(None, ref, hyp, autojunk=False)
    subs = dels = ins = 0
    missing, extra = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        n_ref, n_hyp = i2 - i1, j2 - j1
        when = hyp_time[min(j1, len(hyp_time) - 1)] if hyp_time else "?"
        if tag == "replace":
            subs += min(n_ref, n_hyp)
            dels += max(0, n_ref - n_hyp)
            ins += max(0, n_hyp - n_ref)
            if n_ref - n_hyp >= 8:
                missing.append((n_ref - n_hyp, when, " ".join(ref[i1:i2])))
        elif tag == "delete":
            dels += n_ref
            if n_ref >= 8:
                missing.append((n_ref, when, " ".join(ref[i1:i2])))
        elif tag == "insert":
            ins += n_hyp
            if n_hyp >= 4:
                extra.append((n_hyp, when, " ".join(hyp[j1:j2])))
    return {
        "ref_words": len(ref), "hyp_words": len(hyp), "errors": errors,
        "accuracy": 1 - errors / max(1, len(ref)), "substituted": subs, "omitted": dels, "added": ins,
        "matched": sum(b.size for b in sm.get_matching_blocks()) / max(1, len(ref)),
        "missing": sorted(missing, reverse=True), "extra": sorted(extra, reverse=True),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("transcripcion", help="TXT guardado por VoiceLogger")
    parser.add_argument("referencia", help="texto de referencia (p. ej. subtítulos de YouTube en .txt)")
    args = parser.parse_args(argv)

    with open(args.referencia, encoding="utf-8") as f:
        ref_words = normalize(f.read())
    blocks = read_app_transcript(args.transcripcion)
    if not blocks:
        sys.exit("No encontré líneas «[HH:MM:SS] [EN] texto» en la transcripción.")
    r = evaluate(ref_words, blocks)

    print(f"Referencia (tramo cubierto): {r['ref_words']} palabras · transcripción: {r['hyp_words']} palabras")
    print(f"\nPRECISIÓN: {r['accuracy'] * 100:.1f}%  (WER {100 - r['accuracy'] * 100:.1f}%)")
    print(f"  cambiadas {r['substituted']} · omitidas {r['omitted']} · añadidas {r['added']}")
    print(f"  palabras de la referencia que aparecen igual: {r['matched'] * 100:.1f}%")
    lost = sum(m[0] for m in r["missing"])
    print(f"\nTramos perdidos (8+ palabras seguidas): {len(r['missing'])}, {lost} palabras")
    for n, when, text in r["missing"][:8]:
        print(f"  −{n:3d} cerca de [{when}]: «{text[:100]}{'…' if len(text) > 100 else ''}»")
    print(f"\nTexto añadido que no está en la referencia (4+ palabras): {len(r['extra'])}")
    for n, when, text in r["extra"][:6]:
        print(f"  +{n:3d} en [{when}]: «{text[:100]}»")
    print("\nOjo: si la referencia son subtítulos automáticos, también tienen errores;"
          " parte de lo «añadido» puede ser voz real que ellos no captaron.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
