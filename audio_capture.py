import math
import os
import queue
import subprocess
import threading
import time
from collections import deque
from math import gcd

import numpy as np

SAMPLE_RATE = 16000
CHUNK_MS = 30
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_MS / 1000)
BYTES_PER_SAMPLE = 4
BYTES_PER_CHUNK = CHUNK_SAMPLES * BYTES_PER_SAMPLE

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HELPER_PATH = os.path.join(BASE_DIR, ".build", "system_audio_capture")


class AudioSourceError(Exception):
    pass


def prepare_phrase(audio, highpass=False):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return audio
    audio = audio - float(np.mean(audio))
    if highpass and audio.size > 160:
        try:
            from scipy.signal import butter, sosfilt
            sos = butter(4, 85, "hp", fs=SAMPLE_RATE, output="sos")
            audio = sosfilt(sos, audio).astype(np.float32)
        except Exception:
            pass
    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms > 1e-4:
        gain = min(0.07 / rms, 10.0)
        audio = audio * gain
    return np.clip(audio, -0.95, 0.95).astype(np.float32)


class SystemAudioSource:
    def __init__(self, helper_path=HELPER_PATH, on_error=None, on_status=None):
        self.helper_path = helper_path
        self.on_error = on_error
        self.on_status = on_status
        self._proc = None
        self._thread = None
        self._stderr_thread = None
        self._running = threading.Event()
        self._error_reported = False

    @property
    def is_running(self):
        return self._running.is_set()

    def start(self, audio_queue):
        if not os.path.exists(self.helper_path):
            raise AudioSourceError(
                "No se encontró el capturador de audio. Ejecuta setup.sh para compilarlo."
            )
        try:
            self._proc = subprocess.Popen(
                [self.helper_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except OSError as exc:
            raise AudioSourceError(f"No se pudo iniciar la captura de audio: {exc}") from exc

        self._running.set()
        self._thread = threading.Thread(
            target=self._read_audio, args=(audio_queue,), daemon=True
        )
        self._thread.start()
        self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._stderr_thread.start()

        if self.on_status:
            self.on_status("Capturando audio del sistema...")

    def _read_audio(self, audio_queue):
        buffer = bytearray()
        stream = self._proc.stdout
        try:
            while self._running.is_set():
                data = stream.read(4096)
                if not data:
                    break
                buffer.extend(data)
                while len(buffer) >= BYTES_PER_CHUNK:
                    raw = bytes(buffer[:BYTES_PER_CHUNK])
                    del buffer[:BYTES_PER_CHUNK]
                    chunk = np.frombuffer(raw, dtype=np.float32).copy()
                    try:
                        audio_queue.put_nowait(chunk)
                    except queue.Full:
                        pass
        except Exception as exc:
            if self._running.is_set() and self.on_error and not self._error_reported:
                self._error_reported = True
                self.on_error(f"Error leyendo audio del sistema: {exc}")
        finally:
            if self._running.is_set() and self.on_error and not self._error_reported:
                self._error_reported = True
                self.on_error(
                    "La captura de audio del sistema se detuvo. Revisa el permiso en "
                    "Ajustes del Sistema > Privacidad y Seguridad > Grabación de pantalla y audio del sistema."
                )

    def _read_stderr(self):
        stream = self._proc.stderr
        try:
            for raw_line in iter(stream.readline, b""):
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                if "SCREEN_RECORDING_PERMISSION_DENIED" in line:
                    if self.on_error and not self._error_reported:
                        self._error_reported = True
                        self.on_error(
                            "macOS bloqueó la grabación de audio del sistema.\n\n"
                            "Abre Ajustes del Sistema > Privacidad y Seguridad > "
                            "Grabación de pantalla y audio del sistema, activa VoiceLogger Mac "
                            "y vuelve a abrir la app."
                        )
                elif "NO_DISPLAY_FOUND" in line:
                    if self.on_error and not self._error_reported:
                        self._error_reported = True
                        self.on_error(
                            "No se encontró ninguna pantalla para capturar el audio del sistema.\n\n"
                            "Conecta o activa una pantalla e inténtalo de nuevo."
                        )
                elif "UNSUPPORTED_MACOS" in line:
                    if self.on_error and not self._error_reported:
                        self._error_reported = True
                        self.on_error("La captura de audio del sistema requiere macOS 13 o superior.")
                elif "STREAM_STOPPED_WITH_ERROR" in line:
                    if self.on_error and not self._error_reported:
                        self._error_reported = True
                        self.on_error(
                            "macOS detuvo la captura de audio del sistema.\n\n"
                            f"Detalle: {line.split(':', 1)[-1].strip()}"
                        )
                elif "CAPTURE_STARTED" in line:
                    if self.on_status:
                        self.on_status("Escuchando audio del sistema...")
        except Exception:
            pass

    def stop(self):
        self._running.clear()
        if self._proc:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=2)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        if self._stderr_thread and self._stderr_thread.is_alive():
            self._stderr_thread.join(timeout=1.0)
        self._stderr_thread = None


class MicrophoneSource:
    def __init__(self, on_error=None, on_status=None):
        self.on_error = on_error
        self.on_status = on_status
        self._stream = None
        self._running = threading.Event()
        self._native_sr = SAMPLE_RATE
        self._up = 1
        self._down = 1

    @property
    def is_running(self):
        return self._running.is_set()

    def start(self, audio_queue):
        import sounddevice as sd

        errors = []
        for samplerate in (SAMPLE_RATE, 48000, 44100):
            try:
                g = gcd(SAMPLE_RATE, samplerate)
                self._up = SAMPLE_RATE // g
                self._down = samplerate // g
                self._native_sr = samplerate

                def callback(indata, frames, time_info, status):
                    chunk = indata[:, 0].copy()
                    if samplerate != SAMPLE_RATE:
                        from scipy.signal import resample_poly
                        chunk = resample_poly(chunk, self._up, self._down).astype(np.float32)
                    try:
                        audio_queue.put_nowait(chunk)
                    except queue.Full:
                        pass

                self._stream = sd.InputStream(
                    channels=1,
                    samplerate=samplerate,
                    blocksize=int(samplerate * CHUNK_MS / 1000),
                    dtype="float32",
                    callback=callback,
                )
                self._stream.start()
                self._running.set()
                if self.on_status:
                    self.on_status("Escuchando micrófono...")
                return
            except Exception as exc:
                errors.append(str(exc))
                self._stream = None

        raise AudioSourceError(
            "No se pudo abrir el micrófono. Revisa el permiso en "
            "Ajustes del Sistema > Privacidad y Seguridad > Micrófono.\n\n"
            + "\n".join(errors[-2:])
        )

    def stop(self):
        self._running.clear()
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None


class PhraseSegmenter:
    def __init__(
        self,
        energy_threshold=0.0035,
        silence_timeout_sec=0.65,
        min_speech_sec=0.5,
        max_speech_sec=10.0,
        on_phrase=None,
        on_level=None,
        on_status=None,
        highpass=False,
    ):
        self.energy_threshold = energy_threshold
        self.silence_timeout_sec = silence_timeout_sec
        self.min_speech_sec = min_speech_sec
        self.max_speech_sec = max_speech_sec
        self.on_phrase = on_phrase
        self.on_level = on_level
        self.on_status = on_status
        self.highpass = highpass
        self._running = threading.Event()
        self._thread = None
        self._queue = None

    @property
    def is_running(self):
        return self._running.is_set()

    def set_energy_threshold(self, threshold):
        self.energy_threshold = max(0.0005, min(0.05, threshold))

    def start(self, audio_queue):
        if self._running.is_set():
            return
        self._queue = audio_queue
        self._running.set()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running.clear()
        if self._queue is not None:
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

    def _emit(self, chunks, timestamp):
        if not chunks:
            return
        audio = prepare_phrase(np.concatenate(chunks), highpass=self.highpass)
        if audio.size and self.on_phrase:
            self.on_phrase((audio, timestamp))

    def _run(self):
        chunk_sec = CHUNK_MS / 1000.0
        max_silence_chunks = max(3, int(self.silence_timeout_sec / chunk_sec))
        max_phrase_chunks = max(10, int(self.max_speech_sec / chunk_sec))
        min_voiced_chunks = max(2, int(self.min_speech_sec / chunk_sec))

        pre_roll = deque(maxlen=6)
        speaking = False
        streak = 0
        current = []
        voiced = 0
        silence = 0
        timestamp = None

        while self._running.is_set():
            try:
                chunk = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if chunk is None:
                break
            if chunk.size == 0:
                continue

            rms = float(np.sqrt(np.mean(chunk ** 2)))
            if self.on_level:
                self.on_level(min(1.0, rms * 40.0))
            above = rms >= self.energy_threshold

            if not speaking:
                pre_roll.append(chunk)
                if above:
                    streak += 1
                    if streak >= 2:
                        speaking = True
                        streak = 0
                        silence = 0
                        voiced = len(pre_roll)
                        current = list(pre_roll)
                        timestamp = time.strftime("%H:%M:%S")
                        if self.on_status:
                            self.on_status("Voz detectada...")
                else:
                    streak = 0
            else:
                current.append(chunk)
                if above:
                    voiced += 1
                    silence = 0
                else:
                    silence += 1

                silence_reached = silence >= max_silence_chunks
                max_reached = len(current) >= max_phrase_chunks

                if silence_reached or max_reached:
                    if max_reached and above and len(current) > 20:
                        tail = current[-15:]
                        tail_rms = [float(np.sqrt(np.mean(c ** 2))) for c in tail]
                        cut = len(current) - 15 + int(np.argmin(tail_rms)) + 1
                        if voiced >= min_voiced_chunks:
                            self._emit(current[:cut], timestamp)
                        current = list(current[cut:])
                        timestamp = time.strftime("%H:%M:%S")
                        voiced = sum(
                            1 for c in current
                            if float(np.sqrt(np.mean(c ** 2))) >= self.energy_threshold
                        )
                        silence = 0
                        pre_roll.clear()
                    else:
                        speaking = False
                        if voiced >= min_voiced_chunks:
                            self._emit(current, timestamp)
                        current = []
                        voiced = 0
                        silence = 0
                        pre_roll.clear()
                        if self.on_status and self._running.is_set():
                            self.on_status("Escuchando...")

        if speaking and current and voiced >= max(2, min_voiced_chunks):
            self._emit(current, timestamp)
