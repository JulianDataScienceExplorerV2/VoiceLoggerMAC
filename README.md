# VoiceLoggerMac

App local para macOS que transcribe reuniones y contenido en tiempo real (Zoom, Teams, Meet, DingTalk, YouTube, etc.) sin enviar audio a la nube.

Usa **MLX Whisper** en Apple Silicon, captura de audio del sistema con **ScreenCaptureKit** (helper en Swift) o micrófono, segmentación por voz (VAD), subtítulos flotantes, alertas por palabras clave, historial exportable y **resumen IA local** con Qwen.

## Requisitos

- macOS 13 o superior, en **Apple Silicon** (M1/M2/M3/M4...). MLX no funciona en Intel.
- Xcode Command Line Tools: `xcode-select --install`
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/): `brew install uv`
- Python 3.11 para el entorno virtual.
  > `setup.sh` busca Python en `/opt/homebrew/bin/python3.11`. Si tu instalación usa otra ruta, instala `python@3.11` con Homebrew (`brew install python@3.11`) o edita esa línea en `setup.sh`.
- Conexión a internet **solo la primera vez**, para descargar dependencias y modelos.

## Instalación

```bash
git clone https://github.com/JulianDataScienceExplorerV2/VoiceLoggerMAC.git
cd VoiceLoggerMAC
chmod +x setup.sh run.sh
./setup.sh
```

`setup.sh` hace 3 cosas:

1. Crea `.venv` con Python 3.11.
2. Instala las dependencias de `requirements.txt` (PySide6, numpy, scipy, sounddevice, mlx-whisper, mlx-lm).
3. Compila `swift/SystemAudioCapture.swift` a `.build/system_audio_capture` (el capturador de audio del sistema con ScreenCaptureKit).

## Ejecutar

```bash
open VoiceLoggerMac.app
```

o

```bash
./run.sh
```

o directamente:

```bash
.venv/bin/python main.py
```

## Permisos de macOS (importante)

Sin esto la app no puede capturar audio. Después de concederlos, **cierra y vuelve a abrir la app**.

1. **Audio del sistema** (para Zoom/Teams/Meet/YouTube...):
   `Ajustes del Sistema > Privacidad y Seguridad > Grabación de pantalla y audio del sistema` → activa **VoiceLoggerMac**.
   Si ejecutas desde Terminal, activa también **Terminal** (o tu terminal).
2. **Micrófono** (solo si eliges la fuente "Micrófono"):
   `Ajustes del Sistema > Privacidad y Seguridad > Micrófono` → activa **VoiceLoggerMac**.

## Modelos: no hay que descargarlos a mano

Los modelos se descargan **automáticamente desde Hugging Face la primera vez** que se usan y quedan en la caché local de Hugging Face (`~/.cache/huggingface/`). Después funcionan sin internet. La primera descarga puede tardar varios minutos y ocupar unos pocos GB en total.

### Transcripción (Whisper MLX)

El modelo por defecto es `mlx-community/whisper-large-v3-turbo-q4` (equilibrio recomendado entre calidad y consumo). Puedes cambiarlo en la GUI, en "MODELO WHISPER (MLX)":

| Opción en la GUI | Repo en Hugging Face |
|---|---|
| Whisper Large v3 Turbo Q4 (recomendado) | `mlx-community/whisper-large-v3-turbo-q4` |
| Whisper Large v3 Turbo (máxima calidad) | `mlx-community/whisper-large-v3-turbo` |
| Whisper Small English (ligero) | `mlx-community/whisper-small.en-mlx` |
| Whisper Base English (ultra ligero) | `mlx-community/whisper-base.en-mlx` |
| Whisper Tiny (solo pruebas) | `mlx-community/whisper-tiny` |

Al cambiar de modelo, la primera transcripción con ese modelo tardará más porque lo descarga/carga.

Idiomas disponibles en la GUI: Inglés, Auto-detectar, Español, Portugués.

### Resumen IA (Qwen local)

El botón "Resumen IA" usa `mlx-community/Qwen2.5-1.5B-Instruct-4bit` (definido en `summarizer.py`). También se descarga solo la primera vez que pulsas el botón.

### Descarga manual opcional (pre-descargar)

Si prefieres descargar los modelos antes de abrir la app:

```bash
.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('mlx-community/whisper-large-v3-turbo-q4')"
.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('mlx-community/Qwen2.5-1.5B-Instruct-4bit')"
```

## Uso básico

1. Elige la fuente: **Audio del sistema** o **Micrófono**.
2. Elige modelo Whisper e idioma.
3. (Opcional) Escribe vocabulario/contexto: nombres, productos, jerga. Ayuda a que Whisper acierte palabras.
4. Ajusta la sensibilidad de voz si se corta o si entra ruido.
5. Pulsa **Iniciar escucha**. **Subtítulos** muestra la ventana flotante con las últimas frases.
6. Puedes activar **alertas** con palabras clave (ej: `julian, deadline, urgente`).
7. **Guardar TXT** (Cmd+S), **Copiar todo**, **Resumen IA** e **Historial** (exporta a Markdown o PDF).

Los logs se guardan localmente en `logs/transcripcion_AAAA-MM-DD.txt` y `.jsonl`.

## Privacidad

- Todo se procesa en tu Mac: audio, transcripción y resumen no salen a ningún servidor.
- Las transcripciones son contenido privado de tus reuniones. El `.gitignore` ya excluye `logs/`, `.venv/`, `.build/` y cachés, pero **no compartas la carpeta `logs/` ni subas exports con transcripciones reales**.

## Estructura

```
.
├── main.py               # Punto de entrada
├── app_gui.py            # Interfaz PySide6 (ventana, overlay, alertas, historial)
├── audio_capture.py      # Captura sistema/micrófono + segmentación VAD
├── transcriber.py        # Transcripción con MLX Whisper
├── summarizer.py         # Resumen IA local con Qwen (mlx-lm)
├── logger.py             # Guardado TXT/JSONL en logs/
├── history_dialog.py     # Historial y exportación MD/PDF
├── benchmark.py          # Benchmark de modelos Whisper
├── swift/                # Helper ScreenCaptureKit
├── VoiceLoggerMac.app/   # Lanzador de la app para macOS
├── setup.sh              # Instalación (venv + dependencias + helper Swift)
└── run.sh                # Abre la app
```

## Benchmark (opcional)

Compara velocidad/RAM de los modelos Whisper con un WAV de prueba:

```bash
.venv/bin/python benchmark.py --audio /tmp/vl_bench.wav
```

## Problemas frecuentes

- **"No se encontró el capturador de audio"** → ejecuta `./setup.sh` para compilar `.build/system_audio_capture`.
- **macOS bloquea la captura** → revisa los permisos de la sección anterior y reinicia la app.
- **La primera transcripción tarda mucho** → está descargando el modelo de Whisper. Es normal, solo pasa una vez por modelo.
- **Error con el resumen (`mlx_lm` no instalado)** → reinstala dependencias: `uv pip install --python .venv/bin/python -r requirements.txt` (incluye `mlx-lm`).
- **Error de ruta de Python en `setup.sh`** → comprueba que existe `/opt/homebrew/bin/python3.11` o ajusta la ruta.
- **No detecta la voz / entra ruido** → sube/baja la sensibilidad (VAD) en la GUI y prueba el filtro "Claridad de voz" con micrófono.
