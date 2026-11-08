# VoiceLogger Mac — Native (Swift)

Versión nativa para macOS en SwiftUI, conviviendo con la app Python/PySide6 de la raíz. Mismo formato de logs (`transcripcion_AAAA-MM-DD.{txt,jsonl}`).

## Estado

- [x] Swift Package + app SwiftUI que compila: `swift build` (requiere macOS 15+, Apple Silicon)
- [x] Captura de audio del sistema con ScreenCaptureKit en proceso (`SystemAudioCapture.swift`, port del helper `swift/SystemAudioCapture.swift`)
- [x] Segmentador VAD por energía portado de Python (`AudioSegmenter.swift`, con tests)
- [x] Logger TXT/JSONL compatible con Python (`TranscriptLogger.swift`)
- [x] Interfaz mínima: iniciar/detener, nivel, lista de transcripciones
- [x] **Transcripción real con WhisperKit** (CoreML en Apple Neural Engine, modelo `large-v3-v20240930_626MB` de Argmax OSS, vía SPM `argmax-oss-swift` ≥ 1.1.0)
- [x] Selector de modelo en la UI, como en Python (`large-v3-turbo` recomendado → Small/Base/Tiny); se descarga solo la primera vez y queda persistido
- [x] Ventana siempre encima + modo compacto PiP (340×130)
- [ ] Subtítulos flotantes (PiP), bandeja, atajos, ajustes persistentes
- [ ] Resumen local (Foundation Models o MLX Swift)
- [ ] Bundle `.app` firmado/notarizado + DMG

### Prueba real (sin datos personales)

Audio sintético TTS de 13 s, M3 Pro, modelo descargado la primera vez (~3 min) y en caché después (~5 s por proceso):

```bash
cd native
./.build/debug/VoiceLoggerMacNative --transcribe /tmp/vl_bench.wav
# LANG=en
# Hello, this is a synthetic test. Voice logger Mac transcribes meetings locally, ...
```

El modelo se descarga solo la primera vez desde Hugging Face (caché de WhisperKit). Puedes cambiarlo con `VOICELOGGER_MODEL=large-v3-v20240930_turbo`.

## Ejecutar

```bash
cd native
swift run
swift test
```

## Roadmap

1. **Captura dual**: sumar micrófono (AVAudioEngine) como segunda pista y etiquetar sistema = ellos / micrófono = tú.
2. **UI completa**: portar lo bonito del Python — ventana con waveform, overlay flotante siempre visible, alertas por palabras clave, historial con búsqueda y exportación MD/PDF/SRT, ajustes persistentes y atajos.
3. **Resumen**: Foundation Models (`LanguageModelSession`) en macOS 26, con chunking como el Python actual.
4. **Distribución**: Xcode project, firma, notarización, DMG y Sparkle para auto-actualización.

> Nota: MLX/WhisperKit son solo Apple Silicon. La versión Python queda como referencia y fallback; Linux queda fuera del alcance nativo (requeriría otro backend).
