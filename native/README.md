# VoiceLogger Mac — Native (Swift)

Versión nativa para macOS en SwiftUI, conviviendo con la app Python/PySide6 de la raíz. Mismo formato de logs (`transcripcion_AAAA-MM-DD.{txt,jsonl}`).

## Estado

- [x] Swift Package + app SwiftUI que compila: `swift build` (requiere macOS 15+, Apple Silicon)
- [x] Captura de audio del sistema con ScreenCaptureKit en proceso (`SystemAudioCapture.swift`, port del helper `swift/SystemAudioCapture.swift`)
- [x] Segmentador VAD por energía portado de Python (`AudioSegmenter.swift`, con tests). Re-corta el audio en bloques exactos de 30 ms (como Python): corta una frase tras 0,65 s de silencio o a los 8 s de habla seguida, sin importar el tamaño de los buffers de ScreenCaptureKit
- [x] Logger TXT/JSONL compatible con Python (`TranscriptLogger.swift`)
- [x] **Transcripción real con WhisperKit** (CoreML en Apple Neural Engine, modelo `large-v3-v20240930_626MB` de Argmax OSS, vía SPM `argmax-oss-swift` ≥ 1.1.0)
- [x] Selector de modelo en la UI, como en Python (`large-v3-turbo` recomendado → Small/Base/Tiny); se descarga solo la primera vez y queda persistido
- [x] Filtro de alucinaciones de Whisper y limpieza de tokens especiales (`TextFormatting.swift`, port de `is_bad_transcript`)
- [x] Interfaz de lectura: barra superior de una sola línea, transcripción a la izquierda y **Resumen IA** + **Preguntas** apilados a la derecha (paneles redimensionables)
- [x] Ajustes en un popover (⚙), subtítulos flotantes arrastrables, modo compacto, ventana siempre encima, atajos en el menú *Sesión*
- [x] Resumen y preguntas con Qwen local vía `summarizer.py` (puente Python, sin rutas fijas)
- [x] **Resumen en vivo** incremental + **preguntas clave detectadas en el audio**, con Qwen cargado una sola vez en un proceso persistente
- [x] Contexto/persona para Qwen en `qwen_context.txt` (se relee en cada petición)
- [x] Respuestas **largas y detalladas** por defecto, o cortas (selector *Corta / Larga* en el panel Preguntas)
- [x] Velocidad del texto en vivo en ⚙: Rápida (pausa 0,4 s, frases ≤ 5 s) / Normal (0,65 s, ≤ 8 s) / Precisa (1 s, ≤ 12 s); el texto aparece tras esa pausa + ~1 s de Whisper
- [x] Preguntas detectadas: botón **×** para ocultarlas y **?** en la cabecera para volver a verlas
- [x] Ícono propio (`assets/AppIcon.svg`; dibujado en SwiftUI para el Dock y `.icns` para `VoiceLoggerMac.app`)
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

Atajos: ⌘↩ iniciar/detener · ⌘⇧P pausar · ⌘S guardar TXT · ⌘⇧C copiar · ⌘R actualizar resumen · ⌘F filtrar · ⌘+ / ⌘- letra de lectura · ⌘⌥+ / ⌘⌥- tamaño de toda la interfaz · ⌘⇧L leer respuestas en grande.

Tamaño: ⚙ → *Tamaño de la interfaz* (Normal / Grande / Muy grande / Enorme) agranda barra, cabeceras,
botones y ajustes; *Letra de lectura* (A- / A+) agranda transcripción, resumen y respuestas.

## Calidad de la transcripción

- **Cortes en el silencio:** una frase termina tras una pausa, o al llegar al máximo (5/8/12 s según la velocidad);
  en ese caso se corta en el momento más silencioso de su último 1,5 s, entre palabras, y el resto empieza la frase
  siguiente: no se pierde audio ni se parten palabras.
- **Whisper afinado para ir en paralelo con la IA:** codificador y decodificador en el Neural Engine y el espectrograma
  en la CPU (no en la GPU), así la transcripción no se frena mientras la IA escribe en la GPU; sin marcas de tiempo
  (los bloques son cortos y solo se usa el texto: menos tokens y menos bloques vacíos) y como mucho 2 reintentos a
  más temperatura por bloque (antes 5: en música o ruido podían sumar segundos).
- **Contexto para Whisper:** recibe la frase anterior y tu **vocabulario** (⚙ → Vocabulario: nombres, productos,
  siglas) para escribir bien los nombres y no perder el hilo.
- **Bloques vacíos (rescate):** si Whisper devuelve solo «[Music]» o nada en un bloque de 2 s o más, se reintenta
  sin contexto, sin filtros de calidad (no voz, probabilidad, compresión) y sin marcas de tiempo; si sigue vacío y
  dura 4 s o más, se transcribe cada mitad por separado. Lo rescatado solo se guarda si es creíble (≈1 palabra por
  segundo, mínimo 3): así no se cuelan inventos como «It's a good one.» en 7 s de charla. Se quitan las etiquetas de
  sonido ([Music], (laughter), ♪…) y los bucles («X… X… X…»).
- **Diagnóstico:** cada bloque queda en `~/Library/Application Support/VoiceLoggerMac/logs/diagnostico_AAAA-MM-DD.jsonl`
  (texto crudo de Whisper, qué se guardó, `result` ok / recuperado / descartado y `attempts`: normal, rescate, mitades).
- **Medir:** `python3 tools/evaluar_transcripcion.py reunion.txt referencia.txt` compara con un texto de referencia
  (p. ej. subtítulos de YouTube) y da precisión, tramos perdidos y texto añadido. Línea base con el video de Borough
  Market antes de estas mejoras: **81,6 %**, con 17 tramos perdidos (393 palabras).
- Resultados con las mejoras: Borough Market (comiendo, música, mercado) **82,9 % → 87,6 %** en los mismos 20 min;
  charla TED de Simon Sinek (voz clara, transcripción oficial de TED) **91,9 %**, con el 96 % de las palabras igual.

## IA en vivo: resumen + preguntas clave (Qwen local)

- Al pulsar **Iniciar**, la app arranca `summarizer.py serve`: un proceso Python que carga
  Qwen **una sola vez** y lo deja en memoria (~1 GB con el modelo 1.5B 4-bit). Habla con la
  app por JSON, una línea por mensaje; se cierra solo al salir de la app. Su log queda en
  `~/Library/Application Support/VoiceLoggerMac/qwen_server.log`.
- **Dos ritmos (⚙):** las **preguntas** cada 5 s (10 / 15 / 30 s o «con el resumen») con una petición liviana
  —solo lo último que se dijo (~1-2 min) y las preguntas ya listadas, para no repetirlas—, y el **resumen** cada
  1 min (30 s / 2 min / 5 min o **«Solo al final»**, que lo escribe una vez al detener). Al detener siempre se hace
  una última actualización con el final de la reunión. Solo corren si hay texto nuevo, nunca se solapan, el
  intervalo cuenta desde que termina la anterior, y Whisper y tus preguntas van primero.
- **Incremental**: mientras la reunión cabe en una pasada (~12 000 caracteres) el acta se
  rehace entera; después se actualiza con *notas anteriores + solo lo nuevo*, así cada
  actualización cuesta lo mismo a los 5 minutos que a las 2 horas.
- **Anti-bucles**: Qwen genera con penalización de repetición y el acta se limpia de líneas
  repetidas (y de lo que pase de 25 líneas por sección) antes de mostrarse y de usarse como
  *notas anteriores*: un resumen roto («X» ×80) ya no se arrastra a la siguiente actualización.
- **Preguntas clave**: salen del propio resumen y se muestran arriba del panel *Preguntas*.
  En **naranja** (💬) las que alguien dijo tal cual en la reunión (se comprueba contra la transcripción, aunque
  la pregunta quede partida entre dos bloques); en azul las que la IA redactó a partir de lo que se habló.
  ⚙ → *Preguntas sacadas por la IA*: **Pocas (1) / Medias (3, por defecto) / Muchas (6)** azules por actualización;
  las naranjas salen siempre y primero, para que no se pierdan.
  Al tocar una, Qwen la responde con lo dicho en la reunión.
- Whisper también se precarga al abrir la app; los archivos de ambos modelos ya quedan en la
  caché de disco tras la primera descarga.
- Respuestas en **inglés** por defecto (⚙ → *Idioma de las respuestas*).
- **Elige los modelos al abrir**: la app pregunta qué modelo de transcripción y de IA usar (vienen marcados los
  recomendados; *Aceptar* y listo). Casilla para no volver a preguntar; se reabre en ⋯ → *Elegir modelos…*.

  | IA | Repo | Descarga aprox. | Para qué |
  |---|---|---|---|
  | **Qwen3 4B 2507** (recomendado) | `mlx-community/Qwen3-4B-Instruct-2507-4bit` | ~2.3 GB | mejor calidad, va bien con 16 GB |
  | Qwen2.5 1.5B | `mlx-community/Qwen2.5-1.5B-Instruct-4bit` | ~0.9 GB | el más rápido |
  | LFM2.5 2.6B (experimental) | `LiquidAI/LFM2.5-2.6B-MLX-4bit` | ~1.5 GB | rápido, bueno extrayendo datos; licencia LFM 1.0; necesita `mlx-lm` reciente |

  Transcripción recomendada: **Large v3 Turbo** (máxima calidad); el *626MB* queda como opción más liviana.

  Solo modelos de respuesta directa (sin razonamiento previo) para que todo vaya rápido. Si usas uno que "piensa"
  vía `VOICELOGGER_QWEN_MODEL`, la app le apaga el razonamiento y limpia cualquier `<think>`.
  Si un modelo nuevo dice *not supported*: `uv pip install --python .venv/bin/python -U mlx-lm`.
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

## App de verdad (.app / DMG)

Para abrirla con doble click (sin Terminal), con ícono en Finder y permisos propios:

```bash
native/package.sh             # → native/dist/VoiceLogger Mac.app
native/package.sh --install   # además la copia a /Aplicaciones
native/package.sh --dmg       # además native/dist/VoiceLoggerMac.dmg para compartir
```

- Compila con **`xcodebuild`** (hace falta **Xcode** completo, no solo las Command Line Tools): es lo único que
  compila los shaders Metal de MLX que usa el motor «Swift nativo». La primera vez tarda varios minutos (compila MLX).
- Con el motor **Python** la IA usa el `.venv` del repo: la app guarda dónde está el repo (no lo muevas o vuelve a
  empaquetar). Con el motor **Swift nativo** no hace falta Python.
- Firma: usa el primer certificado de firma que encuentre (Developer ID, Apple Development o uno llamado
  «VoiceLogger»); si no hay ninguno, firma ad-hoc. También se puede forzar: `SIGN_IDENTITY="…" native/package.sh`.

### Permisos de macOS: «me lo vuelve a pedir pero ya lo tiene»

- **Firma ad-hoc (sin certificado):** macOS guarda el permiso para *esa* versión exacta de la app. Al volver a
  empaquetar es «otra app» para macOS: Ajustes sigue mostrando el interruptor activado (de la versión vieja) pero
  no vale. `package.sh` lo reinicia solo para que lo pida limpio; si ya lo tenías trabado:
  `tccutil reset ScreenCapture com.julian.voiceloggermac.native` y vuelve a abrir la app.
- **Para que no pase más (una sola vez):** *Acceso a Llaveros* → menú *Acceso a Llaveros* → *Asistente para
  certificados* → *Crear un certificado…* → Nombre **VoiceLogger**, Tipo de identidad *Raíz autofirmada*, Tipo de
  certificado *Firma de código* → Crear. `package.sh` lo detecta y los permisos sobreviven a los re-empaquetados.
  (Si tienes cuenta de desarrollador, un certificado *Apple Development* de Xcode también sirve.)
- **macOS 15 o posterior** vuelve a pedir confirmación **cada mes** a las apps que graban pantalla/audio aunque ya
  tengan permiso: es del sistema; elige *Permitir*.
- La app Python se llama **«VoiceLogger Mac (Python)»** en esas listas, para no confundirla con la nativa.
- En la app: ⋯ → *Ayuda con permisos…* explica esto y abre el panel correcto de Ajustes.
- La primera vez copia tus ajustes de `swift run`.

## Pestañas del Resumen IA

**Resumen** (por defecto) · **Preguntas** · **Tareas**. El resumen en vivo se separa solo: las preguntas clave
(tócalas para que la IA las responda) y las tareas (botón *Agregar a Recordatorios…*). El botón copiar copia la
pestaña que estás viendo.

## Modos de respuesta: Ideas o Reunión

Junto a la caja de preguntas eliges cómo responde la IA:

- **💡 Ideas** (por defecto, para lluvia de ideas): usa la reunión como contexto **más lo que sabe el modelo y
  resultados de internet**; propone opciones, ejemplos y próximos pasos, separa lo dicho en la reunión del
  conocimiento general y cita las fuentes web [1], [2]. Sin internet (o con el permiso apagado) responde igual con
  la reunión y lo que sabe.
  **En dos pasos, para no esperar:** Qwen responde **primero** con la reunión y lo que sabe; mientras escribe, la
  búsqueda en internet ya está corriendo (es solo red, no usa la IA). Al terminar aparece debajo
  **«🌐 Puede complementar tu pregunta»** con lo que internet agrega (datos, cifras, ejemplos o correcciones, sin
  repetir) y sus fuentes. Si internet no aporta nada (p. ej. «¿cuánto es 2 + 2?»), lo dice en una línea.
- **📋 Reunión:** responde **solo** con lo que se dijo; si no está, lo dice.

El modo por defecto se cambia en ⚙. A internet solo sale el texto de la pregunta.

Las preguntas más nuevas salen **arriba** (con su respuesta), las anteriores debajo.

**Leer en grande:** el botón ⤢ de la esquina del panel *Preguntas* agranda las respuestas sobre toda la ventana, con
letra más grande (+3) y líneas de largo cómodo, y abre en la última pregunta respondida. Deja el mouse sobre ⤢ un momento para echar un vistazo (se cierra al
sacar el mouse) o haz clic / ⌘⇧L para que se quede; 📌 lo fija y Esc lo cierra. La transcripción sigue debajo.

## Ampliar con internet 🌐

Qwen es local y no navega: responde con lo dicho en la reunión. Cada respuesta tiene **Ampliar con internet**:
busca **solo el texto de la pregunta** en DuckDuckGo (o Wikipedia si falla) y Qwen amplía la respuesta citando
[1], [2]… con las fuentes como enlaces. La reunión no sale de tu Mac. Se puede quitar en ⚙. Los resultados de un
modelo pequeño pueden fallar: revisa las fuentes.

### Calidad de la búsqueda (modo Ideas y Ampliar con internet)

- **Sin redes sociales:** la búsqueda va con `-site:pinterest.com -site:facebook.com…` (Instagram, TikTok, X):
  piden login y no aportan. Si así no sale nada, se repite sin filtros.
- **Sitios preferidos** (⚙, hasta 3, p. ej. `hbr.org, mckinsey.com, who.int`): se busca en cada uno con
  `site:` (DuckDuckGo no admite `OR` entre `site:`) y van primero en las fuentes.
- **Sitio de referencia según el tema** (automático, si queda cupo): frontend (React, CSS, TypeScript…) →
  `site:developer.mozilla.org`; backend (Python, SQL, Docker, APIs…) → `site:stackoverflow.com`; marketing
  (ROAS, atribución, SEO, embudo, GA4…) → `site:thinkwithgoogle.com`.
- **Fuentes confiables primero:** de los 8 primeros resultados suben Wikipedia, gobierno (.gov, gob.xx), universidades
  (.edu, ac.uk), organismos (WHO, ONU, OCDE…), ciencia (Nature, arXiv…), prensa seria, documentación oficial de
  desarrollo (MDN, react.dev, Python, Docker, PostgreSQL, OWASP, Stack Overflow…) y fuentes de marketing (Think with
  Google, AMA, Ehrenberg-Bass, Nielsen, Gartner, Forrester…). A Qwen
  le llegan marcadas «fuente confiable» y, si las fuentes no coinciden, se le pide confiar en esas.
- **Lee las páginas** (⚙, activado): abre las 3 mejores páginas (solo https públicas, máx. 7 s) y le pasa a Qwen sus
  párrafos más relacionados con la pregunta (hasta 2500 caracteres por página), sin menús, cabeceras, pies ni scripts;
  no solo el resumen corto del buscador. Qwen recibe hasta 6 resultados y el complemento de Ideas (respuesta larga)
  arma ideas más elaboradas, ejemplos o pasos (~150-250 palabras).
- En cada fuente se ve su **dominio**, ✓ si es confiable y 🔍 si la IA leyó la página.

## Tareas → Recordatorios y exportar

- Al pulsar **Detener**, Qwen saca las tareas de la reunión (con responsable y fecha límite si se dijeron) y pregunta
  *«Se detectaron N tareas… ¿agregarlas a Recordatorios?»*. Marcas/desmarcas y van a la lista **Reuniones**
  (se crea sola). También: ⋯ → *Tareas → Recordatorios…* (⌘⇧T). Se puede apagar en ⚙.
  Necesita la app empaquetada (macOS exige la descripción de permiso en el Info.plist); con `swift run` puedes copiarlas.
- ⋯ → **Exportar reunión (Markdown)…** (⌘E): resumen, preguntas clave, tareas, preguntas y respuestas y transcripción.
  Se abre tal cual en Obsidian y se importa en Notion.

## Log de la sesión

Cada vez que abres la app y cada vez que pulsas **Iniciar** se crea `logs/sesion_AAAA-MM-DD_HH-mm-ss.log`
(⋯ → *Abrir log de la sesión*, o «Log de la sesión» abajo a la derecha). Arriba van el Mac (chip, RAM, macOS) y
los ajustes; después, una línea por evento:

```
10:42:03.120 [WHISPER] 7.8 s de audio en 640 ms (×12 tiempo real) · codificar 180 ms · decodificar 410 ms · reintentos 0 · normal
10:42:03.125 [BLOQUE] 7.8 s · normal · ok · 21 palabras · listo en 655 ms
10:42:10.402 [IA] 214 tokens en 4.31 s (58 tok/s escribiendo) · 1.ª palabra 520 ms · prompt 1630 car · ayudante sí · KV 8 bits desde 1024 · …
10:42:10.410 [IA] respuesta Ideas (Swift) en 4.40 s · 1187 car · 0 fuentes
10:42:14.900 [WEB] 6 resultados (3 confiables, 3 páginas leídas) en 2.85 s · dominios: react.dev, …
10:43:00.001 [MEMORIA] app 5210 MB · 0 bloques esperando a Whisper
```

Registra tiempos de Whisper (por bloque y reintentos), de cada petición a la IA (resumen, preguntas, respuestas,
complemento, tareas), carga de modelos, uso del ayudante, búsquedas web, memoria cada minuto
y errores. **No guarda lo que se dice en la reunión** (ni transcripción, ni preguntas, ni respuestas): solo tiempos,
tamaños y conteos, así se puede compartir para depurar.

## Motor de IA: Python o Swift nativo (beta)

⚙ → **Motor de IA**:

- **Python (estable, por defecto):** `summarizer.py serve` con mlx-lm, como hasta ahora.
- **Swift nativo (beta):** la IA corre **dentro de la app** con [MLX Swift](https://github.com/ml-explore/mlx-swift-lm),
  sin Python. Es un port de `summarizer.py`, `meeting_memory.py` y `web_search.py`: **mismos prompts** (comprobados
  letra por letra contra el Python), mismos presupuestos de tokens, limpieza (razonamiento, bucles), resumen en
  vivo, preguntas cada pocos segundos, respuestas Reunión/Ideas en dos pasos, «Ampliar con internet» (`site:`,
  fuentes confiables, lectura de páginas), tareas y memoria en disco (mismo formato de archivo).
  - Usa **los mismos modelos y la misma caché** (`~/.cache/huggingface/hub`): no vuelve a descargarlos.
  - Una generación a la vez (como el hilo único de Python), caché de MLX limitada a 512 MB y vaciada tras cada
    respuesta, y el modelo se libera tras 10 min sin uso.
  - **Optimizaciones que el motor Python no tiene:**
    1. **Respuesta en vivo:** el texto aparece mientras se escribe («Escribiendo…»), también el complemento de
       Ideas y «Ampliar con internet»; las primeras palabras salen en ~1 s.
    2. **Decodificación especulativa** (⚙ «Acelerar con un modelo ayudante», activada): Qwen3 0.6B (o Qwen2.5 0.5B)
       propone 3 palabras y el modelo principal las confirma de una vez. Misma calidad, escribe más rápido;
       +~0,4 GB de RAM y una descarga la primera vez. No aplica a LFM2.5.
    3. **Memoria de trabajo (KV) en 8 bits** a partir de 1024 tokens: la mitad de memoria en prompts largos
       (resumen, respuestas con páginas web), casi sin pérdida; los cortos (preguntas en vivo) no cambian.
    - La caché del prompt fijo (procesar las instrucciones una sola vez) queda pendiente: `ChatSession.fork` aún no
      está en la versión publicada de MLX Swift.
    - Si alguna optimización falla en tu Mac, esa respuesta se repite en modo normal y el modelo sigue así hasta
      cambiarlo o liberarlo: nunca se queda sin responder por una optimización.
    - Debajo de cada respuesta sale el motor y los segundos (p. ej. «Swift · 3,1 s») para compararlos.
  - **Necesita la app empaquetada con `native/package.sh`** (xcodebuild). Con `swift run` los shaders de MLX no se
    compilan: la app lo detecta y avisa en vez de cerrarse; usa el motor Python en ese caso.
  - El archivo de contexto es el mismo `qwen_context.txt` del repo (o uno en Application Support si no hay repo).
- Cambiar de motor libera el modelo del anterior: nunca quedan los dos cargados.

**Cómo probarlo:** empaqueta (`native/package.sh --install`), abre la app, elige ⚙ → Motor de IA → Swift nativo, y
compara con el mismo video que con Python (resumen, preguntas, una respuesta en Reunión y otra en Ideas).

## Ejecutar

```bash
cd native
swift run      # interfaz y transcripción; para el motor Swift nativo usa native/package.sh
swift test     # incluye las pruebas del motor Swift (texto, memoria y búsqueda; sin modelo)
```

## Roadmap

1. **Captura dual**: sumar micrófono (AVAudioEngine) como segunda pista y etiquetar sistema = ellos / micrófono = tú.
2. **UI completa**: portar lo bonito del Python — ventana con waveform, overlay flotante siempre visible, alertas por palabras clave, historial con búsqueda y exportación MD/PDF/SRT, ajustes persistentes y atajos.
3. **Resumen**: Foundation Models (`LanguageModelSession`) en macOS 26, con chunking como el Python actual.
4. **Distribución**: Xcode project, firma, notarización, DMG y Sparkle para auto-actualización.

> Nota: MLX/WhisperKit son solo Apple Silicon. La versión Python queda como referencia y fallback; Linux queda fuera del alcance nativo (requeriría otro backend).
