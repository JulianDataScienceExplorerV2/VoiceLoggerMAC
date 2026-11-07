# VoiceLoggerMac

[English](README.md) | [Español](README.es.md)

Local macOS app that transcribes meetings and content in real time (Zoom, Teams, Meet, DingTalk, YouTube, etc.) without sending audio to the cloud.

It uses **MLX Whisper** on Apple Silicon, system-audio capture via **ScreenCaptureKit** (Swift helper) or microphone, voice-activity segmentation (VAD), a floating caption overlay, keyword alerts, exportable history, and a **local AI summary** with Qwen.

## Requirements

- macOS 13 or newer on **Apple Silicon** (M1/M2/M3/M4...). MLX does not work on Intel.
- Xcode Command Line Tools: `xcode-select --install`
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/): `brew install uv`
- Python 3.11 for the virtual environment.
  > `setup.sh` looks for Python at `/opt/homebrew/bin/python3.11`. If your installation uses a different path, install `python@3.11` with Homebrew (`brew install python@3.11`) or edit that line in `setup.sh`.
- An internet connection **only the first time**, to download dependencies and models.

## Installation

```bash
git clone https://github.com/JulianDataScienceExplorerV2/VoiceLoggerMAC.git
cd VoiceLoggerMAC
chmod +x setup.sh run.sh
./setup.sh
```

`setup.sh` does 3 things:

1. Creates `.venv` with Python 3.11.
2. Installs the dependencies from `requirements.txt` (PySide6, numpy, scipy, sounddevice, mlx-whisper, mlx-lm).
3. Compiles `swift/SystemAudioCapture.swift` to `.build/system_audio_capture` (the ScreenCaptureKit system-audio capturer).

## Running

```bash
open VoiceLoggerMac.app
```

or

```bash
./run.sh
```

or directly:

```bash
.venv/bin/python main.py
```

## macOS permissions (important)

Without these, the app cannot capture audio. After granting them, **quit and reopen the app**.

1. **System audio** (for Zoom/Teams/Meet/YouTube...):
   `System Settings > Privacy & Security > Screen Recording & System Audio Recording` → enable **VoiceLoggerMac**.
   If you run it from Terminal, also enable **Terminal** (or your terminal app).
2. **Microphone** (only if you choose the "Microphone" source):
   `System Settings > Privacy & Security > Microphone` → enable **VoiceLoggerMac**.

## Models: no manual download needed

Models are downloaded **automatically from Hugging Face the first time** they are used and kept in the local Hugging Face cache (`~/.cache/huggingface/`). After that they work offline. The first download can take several minutes and a few GB in total.

### Transcription (Whisper MLX)

The default model is `mlx-community/whisper-large-v3-turbo-q4` (recommended balance of quality and power consumption). You can change it in the GUI, under "WHISPER MODEL (MLX)":

| GUI option | Hugging Face repo |
|---|---|
| Whisper Large v3 Turbo Q4 (recommended) | `mlx-community/whisper-large-v3-turbo-q4` |
| Whisper Large v3 Turbo (best quality) | `mlx-community/whisper-large-v3-turbo` |
| Whisper Small English (lightweight) | `mlx-community/whisper-small.en-mlx` |
| Whisper Base English (ultra lightweight) | `mlx-community/whisper-base.en-mlx` |
| Whisper Tiny (testing only) | `mlx-community/whisper-tiny` |

When you switch models, the first transcription with that model will take longer because it downloads/loads it.

Languages available in the GUI: English, Auto-detect, Spanish, Portuguese.

### AI summary (local Qwen)

The "AI Summary" button uses `mlx-community/Qwen2.5-1.5B-Instruct-4bit` (defined in `summarizer.py`). It is also downloaded only the first time you press the button.

### Optional manual download (pre-download)

If you prefer to download the models before opening the app:

```bash
.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('mlx-community/whisper-large-v3-turbo-q4')"
.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('mlx-community/Qwen2.5-1.5B-Instruct-4bit')"
```

## Basic usage

1. Choose the source: **System audio** or **Microphone**.
2. Choose the Whisper model and language.
3. (Optional) Enter vocabulary/context: names, products, jargon. This helps Whisper get words right.
4. Adjust voice sensitivity if speech gets cut off or noise comes through.
5. Click **Start listening**. **Captions** shows the floating window with the latest phrases.
6. You can enable **alerts** with keywords (e.g. `julian, deadline, urgent`).
7. **Save TXT** (Cmd+S), **Copy all**, **AI Summary**, and **History** (exports to Markdown or PDF).

Logs are saved locally in `logs/transcripcion_YYYY-MM-DD.txt` and `.jsonl`.

## Privacy

- Everything is processed on your Mac: audio, transcription, and summaries never leave your machine.
- Transcripts are private meeting content. The `.gitignore` already excludes `logs/`, `.venv/`, `.build/`, and caches, but **do not share the `logs/` folder or upload exports with real transcripts**.

## Project structure

```
.
├── main.py               # Entry point
├── app_gui.py            # PySide6 GUI (window, overlay, alerts, history)
├── audio_capture.py      # System/microphone capture + VAD segmentation
├── transcriber.py        # MLX Whisper transcription
├── summarizer.py         # Local AI summary with Qwen (mlx-lm)
├── logger.py             # TXT/JSONL logging in logs/
├── history_dialog.py     # History and MD/PDF export
├── benchmark.py          # Whisper model benchmark
├── swift/                # ScreenCaptureKit helper
├── VoiceLoggerMac.app/   # macOS app launcher
├── setup.sh              # Setup (venv + dependencies + Swift helper)
└── run.sh                # Opens the app
```

## Benchmark (optional)

Compare Whisper model speed/RAM with a test WAV file:

```bash
.venv/bin/python benchmark.py --audio /tmp/vl_bench.wav
```

## Troubleshooting

- **"Audio capturer not found"** → run `./setup.sh` to compile `.build/system_audio_capture`.
- **macOS blocks capture** → check the permissions in the section above and restart the app.
- **First transcription takes a long time** → it is downloading the Whisper model. This is normal, and only happens once per model.
- **Summary error (`mlx_lm` not installed)** → reinstall dependencies: `uv pip install --python .venv/bin/python -r requirements.txt` (includes `mlx-lm`).
- **Python path error in `setup.sh`** → check that `/opt/homebrew/bin/python3.11` exists or adjust the path.
- **Voice not detected / too much noise** → raise/lower the VAD sensitivity in the GUI and try the "Voice clarity" filter with the microphone.
