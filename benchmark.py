import argparse
import json
import os
import resource
import subprocess
import sys
import time

MODELS = [
    ("Turbo-Q4", "mlx-community/whisper-large-v3-turbo-q4"),
    ("Turbo", "mlx-community/whisper-large-v3-turbo"),
    ("Small.en", "mlx-community/whisper-small.en-mlx"),
    ("Base.en", "mlx-community/whisper-base.en-mlx"),
]

DURATIONS = [10.0, 30.0]


def worker(model, audio_path, seconds):
    import numpy as np
    import scipy.io.wavfile as wavfile
    import mlx_whisper

    sample_rate, data = wavfile.read(audio_path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    else:
        data = data.astype(np.float32)
    if sample_rate != 16000:
        from math import gcd
        from scipy.signal import resample_poly

        g = gcd(16000, sample_rate)
        data = resample_poly(data, 16000 // g, sample_rate // g).astype(np.float32)

    target = int(seconds * 16000)
    if len(data) < target:
        data = np.pad(data, (0, target - len(data)))
    else:
        data = data[:target]

    start = time.perf_counter()
    mlx_whisper.transcribe(
        data, path_or_hf_repo=model, language="en", temperature=0.0,
        condition_on_previous_text=False, verbose=None,
    )
    cold = time.perf_counter() - start

    usage_start = resource.getrusage(resource.RUSAGE_SELF)
    start = time.perf_counter()
    mlx_whisper.transcribe(
        data, path_or_hf_repo=model, language="en", temperature=0.0,
        condition_on_previous_text=False, verbose=None,
    )
    warm = time.perf_counter() - start
    usage_end = resource.getrusage(resource.RUSAGE_SELF)

    cpu_seconds = (
        (usage_end.ru_utime - usage_start.ru_utime)
        + (usage_end.ru_stime - usage_start.ru_stime)
    )
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = peak_rss if peak_rss > 10_000_000 else peak_rss * 1024

    print(json.dumps({
        "model": model,
        "seconds": seconds,
        "cold_s": round(cold, 3),
        "warm_s": round(warm, 3),
        "rtf": round(warm / seconds, 4),
        "peak_rss_gb": round(peak_bytes / (1024 ** 3), 2),
        "cpu_cores": round(cpu_seconds / warm, 2) if warm > 0 else 0,
    }))


def thermal_state():
    try:
        output = subprocess.run(
            ["pmset", "-g", "therm"], capture_output=True, text=True, timeout=5
        ).stdout.strip()
    except Exception:
        return "no disponible"
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return " | ".join(lines) if lines else "sin alertas"


def parent(audio_path, models):
    if not os.path.exists(audio_path):
        print(f"No existe el audio de prueba: {audio_path}")
        sys.exit(1)

    print(f"Audio: {audio_path}")
    print(f"Termico antes: {thermal_state()}")
    print()
    header = f"{'Modelo':<10} {'Audio':>6} {'Cold(s)':>9} {'Warm(s)':>9} {'RTF':>7} {'RAM(GB)':>8} {'CPU cores':>10}"
    print(header)
    print("-" * len(header))

    rows = []
    for label, repo in models:
        for seconds in DURATIONS:
            env = dict(os.environ)
            env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
            proc = subprocess.run(
                [
                    sys.executable,
                    os.path.abspath(__file__),
                    "--worker",
                    "--model",
                    repo,
                    "--audio",
                    audio_path,
                    "--seconds",
                    str(seconds),
                ],
                capture_output=True,
                text=True,
                env=env,
            )
            payload = None
            for line in proc.stdout.splitlines():
                line = line.strip()
                if line.startswith("{"):
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        pass
            if not payload:
                print(f"{label:<10} {int(seconds):>5}s  ERROR: {proc.stderr.strip()[-200:]}")
                continue
            rows.append(payload)
            print(
                f"{label:<10} {int(seconds):>5}s {payload['cold_s']:>9.2f} "
                f"{payload['warm_s']:>9.2f} {payload['rtf']:>7.3f} "
                f"{payload['peak_rss_gb']:>8.2f} {payload['cpu_cores']:>10.2f}"
            )

    print()
    print(f"Termico despues: {thermal_state()}")
    if rows:
        print()
        print("Lectura rapida:")
        for payload in rows:
            speed = 1.0 / payload["rtf"] if payload["rtf"] else 0
            print(
                f"  {payload['model'].split('/')[-1]:<32} {speed:>5.1f}x tiempo real, "
                f"{payload['peak_rss_gb']:.2f} GB RAM"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--audio", default="/tmp/vl_bench.wav")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--models", default="")
    args = parser.parse_args()

    if args.worker:
        worker(args.model, args.audio, args.seconds)
    else:
        selected = MODELS
        if args.models:
            wanted = {label.strip().lower() for label in args.models.split(",")}
            selected = [item for item in MODELS if item[0].lower() in wanted]
        parent(args.audio, selected)
