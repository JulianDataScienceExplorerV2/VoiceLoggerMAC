#!/bin/bash
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "Instala uv primero: brew install uv"
  exit 1
fi

if [ ! -d .venv ]; then
  uv venv --python /opt/homebrew/bin/python3.11 .venv
fi

uv pip install --python .venv/bin/python -r requirements.txt

mkdir -p .build
xcrun swiftc -O -swift-version 5 -parse-as-library swift/SystemAudioCapture.swift \
  -o .build/system_audio_capture \
  -framework ScreenCaptureKit -framework CoreMedia -framework Foundation -framework AVFoundation

chmod +x .build/system_audio_capture
echo "Listo. Abre la app con: open VoiceLoggerMac.app"
