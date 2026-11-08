# VoiceLogger Mac — Native (Swift)

Versión nativa para macOS en SwiftUI, conviviendo con la app Python/PySide6 de la raíz. Mismo formato de logs (`transcripcion_AAAA-MM-DD.{txt,jsonl}`).

## Estado

- [x] Swift Package + app SwiftUI que compila: `swift build` (requiere macOS 15+, Apple Silicon)
- [x] Captura de audio del sistema con ScreenCaptureKit en proceso (`SystemAudioCapture.swift`, port del helper `swift/SystemAudioCapture.swift`)
- [x] Segmentador VAD por energía portado de Python (`AudioSegmenter.swift`, con tests)
- [x] Logger TXT/JSONL compatible con Python (`TranscriptLogger.swift`)
- [x] **Transcripción real con WhisperKit** (CoreML en Apple Neural Engine, modelo `large-v3-v20240930_626MB` de Argmax OSS, vía SPM `argmax-oss-swift` ≥ 1.1.0)
- [x] Selector de modelo en la UI, como en Python (`large-v3-turbo` recomendado → Small/Base/Tiny); se descarga solo la primera vez y queda persistido
- [x] Filtro de alucinaciones de Whisper y limpieza de tokens especiales (`TextFormatting.swift`, port de `is_bad_transcript`)
- [x] Interfaz de lectura: barra superior de una sola línea, transcripción a la izquierda y **Resumen IA** + **Preguntas** apilados a la derecha (paneles redimensionables)
- [x] Ajustes en un popover (⚙), subtítulos flotantes arrastrables, modo compacto, ventana siempre encima, atajos en el menú *Sesión*
- [x] Resumen y preguntas con Qwen local vía `summarizer.py` (puente Python, sin rutas fijas)
- [x] **Resumen en vivo** incremental + **preguntas clave detectadas en el audio**, con Qwen cargado una sola vez en un proceso persistente
- [x] Contexto/persona para Qwen en `qwen_context.txt` (se relee en cada petición)
- [ ] Micrófono como fuente (hoy solo audio del sistema)
- [ ] Contexto/vocabulario para WhisperKit (`promptTokens`)
- [ ] Resumen 100 % nativo (Foundation Models o MLX Swift)
- [ ] Bundle `.app` firmado/notarizado + DMG

## Interfaz

```
┌ ● Escuchando  00:12:34 ▮▮▮▯▯   ·   Large v3 Turbo · EN  [■ Detener] [⏸] │ CC ⧉ ⋯ ⚙ ┐
├──────────────────────────────────────┬───────────────────────────────────┤
│ Transcripción          🔍 Filtrar A- A+│ ✨ Resumen IA        Actualizado ↻ │
│                                      │                                   │
│  10:02:11 EN                         ├───────────────────────────────────┤
│  Texto grande y seleccionable…       │ ? Preguntas  (clave del audio + tuyas)│
│                                      │  ¿Qué se decidió? → respuesta     │
│                                      │  [Pregunta algo…            ➤]    │
└──────────────────────────────────────┴───────────────────────────────────┘
```

Atajos: ⌘↩ iniciar/detener · ⌘⇧P pausar · ⌘S guardar TXT · ⌘⇧C copiar · ⌘R actualizar resumen · ⌘F filtrar · ⌘+ / ⌘- letra de lectura · ⌘⌥+ / ⌘⌥- tamaño de toda la interfaz.

Tamaño: ⚙ → *Tamaño de la interfaz* (Normal / Grande / Muy grande / Enorme) agranda barra, cabeceras,
botones y ajustes; *Letra de lectura* (A- / A+) agranda transcripción, resumen y respuestas.

## IA en vivo: resumen + preguntas clave (Qwen local)

- Al pulsar **Iniciar**, la app arranca `summarizer.py serve`: un proceso Python que carga
  Qwen **una sola vez** y lo deja en memoria (~1 GB con el modelo 1.5B 4-bit). Habla con la
  app por JSON, una línea por mensaje; se cierra solo al salir de la app. Su log queda en
  `~/Library/Application Support/VoiceLoggerMac/qwen_server.log`.
- **En vivo**: cada *N* segundos (15 s / 30 s / 1 min / 2 min, en ⚙) si hay texto nuevo.
  Nunca se solapan dos actualizaciones y el intervalo cuenta desde que termina la anterior.
- **Incremental**: mientras la reunión cabe en una pasada (~12 000 caracteres) el acta se
  rehace entera; después se actualiza con *notas anteriores + solo lo nuevo*, así cada
  actualización cuesta lo mismo a los 5 minutos que a las 2 horas.
- **Preguntas clave**: salen del propio resumen y se muestran arriba del panel *Preguntas*.
  Al tocar una, Qwen la responde con lo dicho en la reunión.
- Whisper también se precarga al abrir la app; los archivos de ambos modelos ya quedan en la
  caché de disco tras la primera descarga.
- Respuestas en **inglés** por defecto (⚙ → *Idioma de las respuestas*).
- **Modelo de IA** en ⚙ (se descarga la primera vez y se reinicia el proceso Qwen):

  | Opción | Repo | Descarga aprox. | Para qué |
  |---|---|---|---|
  | Qwen2.5 1.5B (por defecto) | `mlx-community/Qwen2.5-1.5B-Instruct-4bit` | ~0.9 GB | ligero, rápido |
  | Qwen3 4B 2507 | `mlx-community/Qwen3-4B-Instruct-2507-4bit` | ~2.3 GB | mejor calidad, va bien con 16 GB |
  | Qwen3.5 4B (experimental) | `mlx-community/Qwen3.5-4B-MLX-4bit` | ~3 GB | el más nuevo; necesita `mlx-lm` reciente |

  Los modelos que "piensan" (Qwen3/3.5) se usan con el razonamiento apagado y se limpia cualquier `<think>`.
  Desde la terminal también vale `VOICELOGGER_QWEN_MODEL=<repo>`.

### Reuniones largas sin reventar la RAM

Lo que gasta RAM no es el texto (2 h de reunión ≈ 100 KB), sino el modelo cargado y el
prompt mientras genera. Por eso:

- **Presupuesto fijo de tokens por prompt**: resumen en vivo = notas (≤1000) + texto nuevo
  (≤2500); pregunta = notas (≤800) + memoria (≤1500) + lo reciente (≤1500). El prompt no
  crece con la duración de la reunión.
- **Memoria en disco**: cada ~3000 caracteres (3-4 min de charla) se cierra un fragmento y se
  guarda en `~/Library/Application Support/VoiceLoggerMac/memory/<sesión>.jsonl`. Al
  preguntar, se traen del disco solo los 3 fragmentos más relevantes (búsqueda BM25 por
  palabras clave, sin modelos extra) — sirve para «¿qué presupuesto dijeron al principio?».
  *Limpiar* empieza una sesión nueva; los archivos viejos quedan en disco.
- **MLX contenido**: el prompt se procesa en pasos de 512 tokens (menos pico de memoria), la
  caché de MLX se limita a 512 MB y se vacía tras cada respuesta.
- **Qwen se descarga solo tras 10 min sin uso** y se recarga al volver a necesitarlo
  (`VOICELOGGER_QWEN_IDLE_UNLOAD=0` lo desactiva). También a mano: ⋯ → *Liberar memoria de la IA*.

### Contexto / persona: `qwen_context.txt`

En la raíz del repo (⋯ → *Editar contexto de la IA…* lo abre). Va antes de cada resumen y
pregunta, como un system prompt. Se crea desde `qwen_context.example.txt` la primera vez y
**no se sube a git**. Las líneas con `#` y las que aún tengan `XXXX` se ignoran: rellena
`You are an expert in XXXXXXXX…` y se activa sola en la siguiente respuesta.

### Dónde encuentra Python

La app ejecuta `.venv/bin/python summarizer.py serve`. Busca la raíz del repo en este orden:

1. Variable de entorno `VOICELOGGER_ROOT`
2. `defaults write VoiceLoggerMacNative python_root /ruta/al/repo`
3. Subiendo desde el directorio actual y desde el ejecutable (`swift run` dentro de `native/` la encuentra sola)
4. `~/Desktop/VoiceLoggerMac` y `~/VoiceLoggerMac`

Necesita haber ejecutado `./setup.sh` en la raíz (crea `.venv` con `mlx-lm`).

Prueba del protocolo a mano:

```bash
printf '%s\n' '{"id":"1","type":"live","text":"[10:00] We ship on Friday?","lang":"en"}' | .venv/bin/python summarizer.py serve
```

### Prueba real (sin datos personales)

Audio sintético TTS de 13 s, M3 Pro, modelo descargado la primera vez (~3 min) y en caché después (~5 s por proceso):

> El modo `--transcribe` usado en esta prueba no está en el código actual de la app.

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
