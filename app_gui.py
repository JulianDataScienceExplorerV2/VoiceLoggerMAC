import html
import os
import queue
import re
import sys
import time
from collections import deque
from datetime import datetime

from PySide6.QtCore import Qt, Signal, Slot, QSettings, QThread, QTimer, QUrl, QRectF
from PySide6.QtGui import QColor, QDesktopServices, QKeySequence, QPainter, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSystemTrayIcon,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from audio_capture import (
    AudioSourceError,
    MicrophoneSource,
    PhraseSegmenter,
    SystemAudioSource,
)
from history_dialog import HistoryDialog
from logger import TranscriptLogger
from transcriber import (
    DEFAULT_MODEL,
    LANGUAGE_CHOICES,
    MODEL_CHOICES,
    SpeechTranscriber,
)


class WaveformVisualizer(QWidget):
    def __init__(self, parent=None, bar_count=36):
        super().__init__(parent)
        self.bar_count = bar_count
        self.values = deque([0.04] * self.bar_count, maxlen=self.bar_count)
        self.setFixedHeight(24)
        self.setMinimumWidth(160)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._decay_timer = QTimer(self)
        self._decay_timer.setInterval(40)
        self._decay_timer.timeout.connect(self._decay_step)
        self._decay_timer.start()

    def push_level(self, level):
        scaled = min(1.0, max(0.04, (level ** 0.55) * 1.5))
        self.values.append(scaled)
        self.update()

    def _decay_step(self):
        changed = False
        new_vals = []
        for value in self.values:
            decayed = max(0.04, value * 0.86)
            if abs(decayed - value) > 0.005:
                changed = True
            new_vals.append(decayed)
        if changed:
            self.values = deque(new_vals, maxlen=self.bar_count)
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        width = self.width()
        height = self.height()
        spacing = 3.0
        bar_width = max(2.5, (width - (self.bar_count - 1) * spacing) / self.bar_count)

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#13131d"))
        painter.drawRoundedRect(0, 0, width, height, 6, 6)

        mid_y = height / 2.0
        for index, value in enumerate(self.values):
            x = index * (bar_width + spacing)
            bar_height = max(3.0, value * (height - 6))
            y = mid_y - (bar_height / 2.0)
            if value > 0.72:
                color = QColor("#ef4444")
            elif value > 0.45:
                color = QColor("#f59e0b")
            elif value > 0.20:
                color = QColor("#06b6d4")
            else:
                color = QColor("#10b981")
            painter.setBrush(color)
            painter.drawRoundedRect(QRectF(x, y, bar_width, bar_height), 2.0, 2.0)


class CaptionOverlay(QWidget):
    def __init__(self):
        super().__init__(None, Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setMinimumSize(520, 120)
        self.resize(820, 170)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self.card = QFrame()
        self.card.setObjectName("overlayCard")
        card_layout = QVBoxLayout(self.card)
        card_layout.setContentsMargins(18, 11, 18, 13)
        card_layout.setSpacing(6)

        header = QHBoxLayout()
        header.setSpacing(6)
        self.dot = QLabel("●")
        self.dot.setStyleSheet("color: #10b981; font-size: 10px;")
        title = QLabel("VOICE LOGGER")
        title.setStyleSheet(
            "color: #94a3b8; font-size: 10px; font-weight: 800; letter-spacing: 1.5px;"
        )
        hint = QLabel("arrastra para mover")
        hint.setStyleSheet("color: #475569; font-size: 10px;")
        header.addWidget(self.dot)
        header.addWidget(title)
        header.addStretch()
        header.addWidget(hint)
        card_layout.addLayout(header)

        self.text_label = QLabel("Esperando audio...")
        self.text_label.setWordWrap(True)
        self.text_label.setTextFormat(Qt.RichText)
        self.text_label.setStyleSheet(
            "color: #f8fafc; font-size: 19px; line-height: 1.4; font-weight: 500;"
        )
        self.text_label.setMinimumHeight(84)
        self.text_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        card_layout.addWidget(self.text_label, 1)

        self.lines = deque(maxlen=3)
        self._font_size = 21
        self._drag_offset = None
        self._positioned = False

        outer.addWidget(self.card)
        self.setStyleSheet(
            """
            QFrame#overlayCard {
                background-color: rgba(9, 9, 14, 235);
                border: 1px solid #2c2c40;
                border-left: 3px solid #10b981;
                border-radius: 12px;
            }
            """
        )

    def move_to_default_position(self):
        screen = QApplication.primaryScreen()
        if not screen:
            return
        geometry = screen.availableGeometry()
        x = geometry.center().x() - self.width() // 2
        self.move(max(0, x), geometry.top() + 70)
        self._positioned = True

    def showEvent(self, event):
        super().showEvent(event)
        if not self._positioned:
            self.move_to_default_position()

    def set_text_size(self, size):
        self._font_size = max(16, min(40, int(size)))
        self.text_label.setStyleSheet(
            f"color: #f8fafc; font-size: {self._font_size}px; "
            "line-height: 1.4; font-weight: 500;"
        )

    def set_entries(self, entries):
        self.lines.clear()
        for entry in entries[-3:]:
            self.lines.append((entry["timestamp"], entry["text"]))
        if not self.lines:
            self.text_label.setText("Esperando audio...")
            return
        timestamp_size = max(11, self._font_size - 9)
        parts = []
        for timestamp, text in self.lines:
            parts.append(
                f'<div style="margin-bottom: 3px;">'
                f'<span style="color: #10b981; font-family: Menlo, monospace; '
                f'font-size: {timestamp_size}px;">[{timestamp}]</span> '
                f'<span>{html.escape(text)}</span></div>'
            )
        self.text_label.setText("".join(parts))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if self._drag_offset is not None and event.buttons() & Qt.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_offset)
            event.accept()

    def mouseReleaseEvent(self, event):
        self._drag_offset = None
        event.accept()


class AlertToast(QWidget):
    def __init__(self):
        super().__init__(None, Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        frame = QFrame()
        frame.setObjectName("alertToast")
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(16, 10, 16, 10)
        self.label = QLabel("")
        self.label.setStyleSheet(
            "color: #fff1f2; font-size: 14px; font-weight: 700; background: transparent;"
        )
        layout.addWidget(self.label)
        root.addWidget(frame)
        self.setStyleSheet(
            "QFrame#alertToast { background-color: #7f1d1d; "
            "border: 1px solid #ef4444; border-radius: 10px; }"
        )
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.hide)

    def show_message(self, text):
        self.label.setText(f"ALERTA: {text}")
        self.adjustSize()
        screen = QApplication.primaryScreen()
        if screen:
            geometry = screen.availableGeometry()
            self.move(geometry.right() - self.width() - 30, geometry.top() + 30)
        self.show()
        self.raise_()
        self._timer.start(4000)


class SummaryWorker(QThread):
    summary_ready = Signal(str, str)

    def __init__(self, text, language):
        super().__init__()
        self.text = text
        self.language = language

    def run(self):
        from summarizer import build_summary

        result, error = build_summary(self.text, self.language)
        self.summary_ready.emit(result or "", error or "")


class SummaryDialog(QDialog):
    def __init__(self, text, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Resumen de la reunion")
        self.resize(760, 640)
        self.summary_text = text

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        self.viewer = QTextEdit()
        self.viewer.setReadOnly(True)
        if hasattr(self.viewer, "setMarkdown"):
            self.viewer.setMarkdown(text)
        else:
            self.viewer.setPlainText(text)
        root.addWidget(self.viewer)

        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        btn_copy = QPushButton("Copiar")
        btn_copy.clicked.connect(self._copy)
        btn_save = QPushButton("Guardar MD")
        btn_save.clicked.connect(self._save)
        buttons.addWidget(btn_copy)
        buttons.addWidget(btn_save)
        buttons.addStretch()
        btn_close = QPushButton("Cerrar")
        btn_close.clicked.connect(self.accept)
        buttons.addWidget(btn_close)
        root.addLayout(buttons)

    def _copy(self):
        QApplication.clipboard().setText(self.summary_text)

    def _save(self):
        default = os.path.join(
            os.path.expanduser("~"),
            "Desktop",
            f"resumen_reunion_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.md",
        )
        path, _ = QFileDialog.getSaveFileName(
            self, "Guardar resumen", default, "Markdown (*.md)"
        )
        if not path:
            return
        if not path.lower().endswith(".md"):
            path += ".md"
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.summary_text)
        except OSError as exc:
            QMessageBox.critical(self, "Error", str(exc))
            return
        QMessageBox.information(self, "Guardado", f"Resumen guardado en:\n{path}")


class TranscribeWorker(QThread):
    phrase_transcribed = Signal(dict)
    status_updated = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, transcriber, logger):
        super().__init__()
        self.transcriber = transcriber
        self.logger = logger
        self.queue = queue.Queue(maxsize=40)
        self._running = True

    def enqueue_audio(self, phrase):
        if self.queue.full():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self.queue.put_nowait(phrase)
        except queue.Full:
            pass

    def stop(self):
        self._running = False
        try:
            while True:
                self.queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            pass
        self.wait(3000)

    def run(self):
        while self._running:
            try:
                item = self.queue.get(timeout=0.25)
            except queue.Empty:
                continue
            if item is None:
                break

            audio_data, capture_time = item
            if not self.transcriber.model_loaded:
                self.status_updated.emit("Cargando modelo (la primera vez puede tardar)...")
            else:
                self.status_updated.emit("Transcribiendo...")

            try:
                text, lang = self.transcriber.transcribe(audio_data)
                if text:
                    model_label = self.transcriber.model_size.split("/")[-1]
                    entry = self.logger.log(
                        text,
                        model_name=f"mlx:{model_label}",
                        lang=lang,
                        timestamp=capture_time,
                    )
                    if entry:
                        self.phrase_transcribed.emit(entry)
            except Exception as exc:
                self.error_occurred.emit(f"Error de transcripción: {exc}")
            finally:
                self.status_updated.emit("Escuchando...")


class VoiceLoggerWindow(QMainWindow):
    sig_audio_level = Signal(float)
    sig_phrase_done = Signal(object)
    sig_recorder_status = Signal(str)
    sig_source_error = Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("VoiceLogger Mac - Log de reuniones y contenido")
        self.resize(1180, 800)
        self.setMinimumSize(760, 440)

        self.logger = TranscriptLogger()
        self.transcriber = SpeechTranscriber(model_size=DEFAULT_MODEL, language="en")
        self.all_entries = deque(maxlen=3000)
        self.start_session_time = None
        self._last_entry_wall_ts = 0.0
        self._source = None
        self._segmenter = None
        self._paused = False
        self._is_pip_mode = False
        self._normal_geometry = None

        self.qsettings = QSettings("VoiceLoggerMac", "VoiceLoggerMac")
        try:
            self.transcript_font_size = int(self.qsettings.value("font_size", 17))
        except (TypeError, ValueError):
            self.transcript_font_size = 17
        self.transcript_font_size = max(12, min(28, self.transcript_font_size))

        self.overlay = CaptionOverlay()
        self.overlay.set_text_size(self.transcript_font_size + 5)
        self.overlay.hide()

        self.alert_toast = AlertToast()
        self._last_alert_at = 0.0
        self._summary_worker = None

        self.search_timer = QTimer(self)
        self.search_timer.setSingleShot(True)
        self.search_timer.setInterval(180)
        self.search_timer.timeout.connect(self._do_search_filter)

        self.worker = TranscribeWorker(self.transcriber, self.logger)
        self.worker.phrase_transcribed.connect(self.on_phrase_transcribed)
        self.worker.status_updated.connect(self.update_status)
        self.worker.error_occurred.connect(self.show_error)
        self.worker.start()

        self.sig_audio_level.connect(self.update_audio_level)
        self.sig_phrase_done.connect(self.worker.enqueue_audio)
        self.sig_recorder_status.connect(self.update_status)
        self.sig_source_error.connect(self.on_source_error)

        self.init_ui()
        self.apply_dark_theme()
        self.setup_tray_icon()
        QShortcut(QKeySequence.StandardKey.Save, self, activated=self.save_meeting_txt)

        self.timer_session = QTimer(self)
        self.timer_session.setInterval(1000)
        self.timer_session.timeout.connect(self.update_session_clock)
        self.timer_session.start()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        self.main_layout = QVBoxLayout(central)
        self.main_layout.setContentsMargins(18, 16, 18, 14)
        self.main_layout.setSpacing(11)

        self.header_card = QFrame()
        self.header_card.setObjectName("headerCard")
        header_layout = QHBoxLayout(self.header_card)
        header_layout.setContentsMargins(14, 10, 14, 10)

        title_box = QVBoxLayout()
        title_box.setSpacing(1)
        title = QLabel("VoiceLogger Mac")
        title.setStyleSheet(
            "font-size: 20px; font-weight: 800; color: #10b981; letter-spacing: 0.5px;"
        )
        subtitle = QLabel("ScreenCaptureKit + Whisper MLX - DingTalk, Zoom, Teams, Meet y YouTube")
        subtitle.setStyleSheet("font-size: 11px; color: #94a3b8;")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header_layout.addLayout(title_box)
        header_layout.addStretch()

        self.lbl_active_pill = QLabel("large-v3-turbo-q4 - EN")
        self.lbl_active_pill.setStyleSheet(
            "background-color: #1e1b4b; color: #a5b4fc; border: 1px solid #3730a3; "
            "padding: 5px 10px; border-radius: 12px; font-size: 11px; font-weight: bold;"
        )
        header_layout.addWidget(self.lbl_active_pill)

        self.lbl_timer = QLabel("00:00:00")
        self.lbl_timer.setStyleSheet(
            "background-color: #161622; color: #e2e8f0; border: 1px solid #27273a; "
            "padding: 5px 12px; border-radius: 12px; font-family: Menlo, monospace; "
            "font-size: 12px; font-weight: bold;"
        )
        header_layout.addWidget(self.lbl_timer)

        self.btn_pip = QPushButton("Subtitulos")
        self.btn_pip.setObjectName("btn_header_action")
        self.btn_pip.setToolTip("Ventana flotante siempre visible con las ultimas frases")
        self.btn_pip.clicked.connect(self.toggle_overlay)
        header_layout.addWidget(self.btn_pip)

        self.btn_toggle_settings = QPushButton("Ajustes ▾")
        self.btn_toggle_settings.setObjectName("btn_header_action")
        self.btn_toggle_settings.clicked.connect(self.toggle_settings_panel)
        header_layout.addWidget(self.btn_toggle_settings)

        self.main_layout.addWidget(self.header_card)

        self.settings_group = QGroupBox("Captura y motor de transcripcion")
        self.settings_group.setObjectName("settingsGroup")
        settings_layout = QVBoxLayout(self.settings_group)
        settings_layout.setSpacing(10)
        settings_layout.setContentsMargins(14, 14, 14, 12)

        row1 = QHBoxLayout()
        row1.setSpacing(10)

        source_box = QVBoxLayout()
        source_label = QLabel("FUENTE DE AUDIO")
        source_label.setObjectName("fieldLabel")
        self.combo_source = QComboBox()
        self.combo_source.addItem("Audio del sistema (DingTalk, Zoom, Teams, Meet)", "system")
        self.combo_source.addItem("Microfono", "mic")
        source_box.addWidget(source_label)
        source_box.addWidget(self.combo_source)
        row1.addLayout(source_box, 3)

        model_box = QVBoxLayout()
        model_label = QLabel("MODELO WHISPER (MLX)")
        model_label.setObjectName("fieldLabel")
        self.combo_model = QComboBox()
        for label, repo in MODEL_CHOICES:
            self.combo_model.addItem(label, repo)
        model_box.addWidget(model_label)
        model_box.addWidget(self.combo_model)
        row1.addLayout(model_box, 4)

        lang_box = QVBoxLayout()
        lang_label = QLabel("IDIOMA")
        lang_label.setObjectName("fieldLabel")
        self.combo_lang = QComboBox()
        for label, code in LANGUAGE_CHOICES:
            self.combo_lang.addItem(label, code)
        lang_box.addWidget(lang_label)
        lang_box.addWidget(self.combo_lang)
        row1.addLayout(lang_box, 2)

        settings_layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.setSpacing(10)

        vocab_box = QVBoxLayout()
        vocab_label = QLabel("CONTEXTO / VOCABULARIO (nombres, productos, jerga)")
        vocab_label.setObjectName("fieldLabel")
        self.edit_context = QLineEdit()
        self.edit_context.setPlaceholderText(
            "Ej: DingTalk, sprint planning, API, customer success, Julian, UPTC..."
        )
        vocab_box.addWidget(vocab_label)
        vocab_box.addWidget(self.edit_context)
        row2.addLayout(vocab_box, 5)

        sens_box = QVBoxLayout()
        sens_label = QLabel("SENSIBILIDAD DE VOZ (VAD)")
        sens_label.setObjectName("fieldLabel")
        slider_row = QHBoxLayout()
        self.slider_sens = QSlider(Qt.Horizontal)
        self.slider_sens.setRange(10, 150)
        self.slider_sens.setValue(35)
        self.slider_sens.setToolTip("Umbral de energia para detectar voz. Sube si se corta, baja si entra ruido.")
        self.slider_sens.valueChanged.connect(self.on_sensitivity_changed)
        self.lbl_sens_val = QLabel("Recomendada")
        self.lbl_sens_val.setStyleSheet(
            "color: #38bdf8; font-weight: bold; min-width: 85px;"
        )
        slider_row.addWidget(self.slider_sens, 1)
        slider_row.addWidget(self.lbl_sens_val)
        sens_box.addWidget(sens_label)
        sens_box.addLayout(slider_row)
        row2.addLayout(sens_box, 4)

        settings_layout.addLayout(row2)

        row3 = QHBoxLayout()
        row3.setSpacing(16)
        self.chk_enhance = QCheckBox("Claridad de voz (filtro pasa-altos en microfono)")
        self.chk_enhance.setChecked(False)
        self.chk_enhance.setStyleSheet("color: #38bdf8; font-weight: 600;")
        row3.addWidget(self.chk_enhance)
        row3.addStretch()
        settings_layout.addLayout(row3)

        row4 = QHBoxLayout()
        row4.setSpacing(10)

        alerts_box = QVBoxLayout()
        alerts_label = QLabel("ALERTAS EN VIVO (palabras separadas por coma)")
        alerts_label.setObjectName("fieldLabel")
        self.edit_alerts = QLineEdit()
        self.edit_alerts.setPlaceholderText(
            "ej: julian, action item, deadline, urgente"
        )
        alerts_box.addWidget(alerts_label)
        alerts_box.addWidget(self.edit_alerts)
        row4.addLayout(alerts_box, 5)

        summary_box = QVBoxLayout()
        summary_label = QLabel("IDIOMA DEL RESUMEN IA")
        summary_label.setObjectName("fieldLabel")
        self.combo_summary_lang = QComboBox()
        self.combo_summary_lang.addItem("Espanol", "es")
        self.combo_summary_lang.addItem("Ingles", "en")
        summary_box.addWidget(summary_label)
        summary_box.addWidget(self.combo_summary_lang)
        row4.addLayout(summary_box, 2)

        self.chk_alert_sound = QCheckBox("Sonido")
        self.chk_alert_sound.setChecked(True)
        self.chk_alert_sound.setStyleSheet("color: #cbd5e1;")
        row4.addWidget(self.chk_alert_sound)

        settings_layout.addLayout(row4)

        self.main_layout.addWidget(self.settings_group)

        self.combo_model.currentIndexChanged.connect(self.on_settings_changed)
        self.combo_lang.currentIndexChanged.connect(self.on_settings_changed)
        self.edit_context.editingFinished.connect(self.on_settings_changed)

        control_strip = QHBoxLayout()
        control_strip.setSpacing(10)
        self.lbl_status_badge = QLabel("● Detenido")
        self._set_badge_style("#ef4444")
        control_strip.addWidget(self.lbl_status_badge)
        self.waveform = WaveformVisualizer(bar_count=36)
        control_strip.addWidget(self.waveform, 1)
        self.vu_bar = QProgressBar()
        self.vu_bar.setRange(0, 100)
        self.vu_bar.setValue(0)
        self.vu_bar.setTextVisible(False)
        self.vu_bar.setFixedWidth(70)
        self.vu_bar.setFixedHeight(18)
        self.vu_bar.setStyleSheet(
            """
            QProgressBar {
                background-color: #151520;
                border: 1px solid #282838;
                border-radius: 9px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #059669, stop:0.6 #10b981, stop:0.85 #f59e0b, stop:1 #ef4444);
                border-radius: 8px;
            }
            """
        )
        control_strip.addWidget(self.vu_bar)
        self.main_layout.addLayout(control_strip)

        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)

        self.btn_start = QPushButton("Iniciar escucha")
        self.btn_start.setObjectName("btn_start")
        self.btn_start.clicked.connect(self.start_listening)

        self.btn_pause = QPushButton("Pausar")
        self.btn_pause.setObjectName("btn_secondary")
        self.btn_pause.setEnabled(False)
        self.btn_pause.clicked.connect(self.toggle_pause)

        self.btn_stop = QPushButton("Detener")
        self.btn_stop.setObjectName("btn_danger")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_listening)

        button_layout.addWidget(self.btn_start)
        button_layout.addWidget(self.btn_pause)
        button_layout.addWidget(self.btn_stop)
        button_layout.addStretch()

        self.btn_save_txt = QPushButton("Guardar TXT")
        self.btn_save_txt.setObjectName("btn_secondary")
        self.btn_save_txt.setToolTip("Guarda la reunión actual como archivo de texto (Cmd+S)")
        self.btn_save_txt.clicked.connect(self.save_meeting_txt)

        self.btn_summary = QPushButton("Resumen IA")
        self.btn_summary.setObjectName("btn_summary")
        self.btn_summary.setToolTip("Genera un acta con temas, decisiones y tareas")
        self.btn_summary.clicked.connect(self.generate_summary)

        self.btn_history = QPushButton("Historial")
        self.btn_history.setObjectName("btn_secondary")
        self.btn_history.setToolTip("Busca sesiones anteriores y exporta a Markdown o PDF")
        self.btn_history.clicked.connect(self.open_history)

        self.btn_open_folder = QPushButton("Ver logs")
        self.btn_open_folder.setObjectName("btn_secondary")
        self.btn_open_folder.clicked.connect(self.open_logs_folder)

        self.btn_copy = QPushButton("Copiar todo")
        self.btn_copy.setObjectName("btn_secondary")
        self.btn_copy.clicked.connect(self.copy_transcript)

        self.btn_clear = QPushButton("Limpiar")
        self.btn_clear.setObjectName("btn_secondary")
        self.btn_clear.clicked.connect(self.clear_display)

        button_layout.addWidget(self.btn_save_txt)
        button_layout.addWidget(self.btn_summary)
        button_layout.addWidget(self.btn_history)
        button_layout.addWidget(self.btn_open_folder)
        button_layout.addWidget(self.btn_copy)
        button_layout.addWidget(self.btn_clear)
        self.main_layout.addLayout(button_layout)

        filter_layout = QHBoxLayout()
        filter_layout.setSpacing(10)
        self.edit_search = QLineEdit()
        self.edit_search.setPlaceholderText("Filtrar por palabra clave (resaltado automatico)...")
        self.edit_search.textChanged.connect(self.search_timer.start)
        filter_layout.addWidget(self.edit_search, 1)

        self.chk_autoscroll = QCheckBox("Autodesplazar")
        self.chk_autoscroll.setChecked(True)
        self.chk_autoscroll.setStyleSheet("color: #94a3b8; font-size: 12px;")
        filter_layout.addWidget(self.chk_autoscroll)

        self.btn_font_minus = QPushButton("A-")
        self.btn_font_minus.setObjectName("btn_font")
        self.btn_font_minus.setFixedWidth(38)
        self.btn_font_minus.setToolTip("Reducir tamano de letra")
        self.btn_font_minus.clicked.connect(lambda: self.change_font_size(-1))

        self.btn_font_plus = QPushButton("A+")
        self.btn_font_plus.setObjectName("btn_font")
        self.btn_font_plus.setFixedWidth(38)
        self.btn_font_plus.setToolTip("Aumentar tamano de letra")
        self.btn_font_plus.clicked.connect(lambda: self.change_font_size(1))

        filter_layout.addWidget(self.btn_font_minus)
        filter_layout.addWidget(self.btn_font_plus)
        self.main_layout.addLayout(filter_layout)

        self.text_transcript = QTextEdit()
        self.text_transcript.setReadOnly(True)
        self.text_transcript.document().setMaximumBlockCount(3000)
        self.main_layout.addWidget(self.text_transcript)

        footer = QHBoxLayout()
        self.lbl_phrase_count = QLabel("Bloques: 0")
        self.lbl_phrase_count.setStyleSheet(
            "color: #64748b; font-size: 12px; font-weight: 600;"
        )
        self.lbl_file_info = QLabel(
            f"Guardado en: {os.path.basename(self.logger.get_current_txt_path())}"
        )
        self.lbl_file_info.setStyleSheet("color: #64748b; font-size: 12px;")
        footer.addWidget(self.lbl_phrase_count)
        footer.addStretch()
        footer.addWidget(self.lbl_file_info)
        self.main_layout.addLayout(footer)

    def _set_badge_style(self, color_hex):
        self.lbl_status_badge.setStyleSheet(
            f"""
            background-color: #14141e;
            color: {color_hex};
            font-weight: 800;
            font-size: 12px;
            padding: 5px 12px;
            border-radius: 8px;
            border: 1px solid #2b2b3d;
            """
        )

    def toggle_settings_panel(self):
        visible = self.settings_group.isVisible()
        self.settings_group.setVisible(not visible)
        self.btn_toggle_settings.setText("Ajustes ▾" if visible else "Ajustes ▴")

    def toggle_overlay(self):
        self._is_pip_mode = not self._is_pip_mode
        if self._is_pip_mode:
            self.overlay.set_entries(list(self.all_entries))
            self.overlay.show()
            self.overlay.raise_()
            self.btn_pip.setText("Ocultar subtitulos")
        else:
            self.overlay.hide()
            self.btn_pip.setText("Subtitulos")

    def setup_tray_icon(self):
        self.tray_icon = QSystemTrayIcon(self)
        self.tray_icon.setIcon(
            self.style().standardIcon(self.style().StandardPixmap.SP_MediaVolume)
        )
        self.tray_icon.setToolTip("VoiceLogger Mac")
        tray_menu = QMenu()
        act_show = tray_menu.addAction("Mostrar ventana")
        act_show.triggered.connect(self.showNormal)
        act_save = tray_menu.addAction("Guardar TXT de la reunion")
        act_save.triggered.connect(self.save_meeting_txt)
        act_overlay = tray_menu.addAction("Subtitulos flotantes")
        act_overlay.triggered.connect(self.toggle_overlay)
        act_toggle = tray_menu.addAction("Iniciar / Detener")
        act_toggle.triggered.connect(self.toggle_listening)
        tray_menu.addSeparator()
        act_exit = tray_menu.addAction("Salir")
        act_exit.triggered.connect(self.close)
        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.show()

    def toggle_listening(self):
        if self._source and self._source.is_running:
            self.stop_listening()
        else:
            self.start_listening()

    def on_settings_changed(self):
        model = self.combo_model.currentData()
        language = self.combo_lang.currentData() or "en"
        context = self.edit_context.text()
        self.transcriber.apply_settings(
            model_size=model, language=language, extra_context=context
        )
        self.transcriber.reset_context()
        short = (model or "").split("/")[-1]
        self.lbl_active_pill.setText(f"{short} - {language.upper()}")

    def on_sensitivity_changed(self, value):
        threshold = value / 10000.0
        if self._segmenter:
            self._segmenter.set_energy_threshold(threshold)
        if value < 25:
            self.lbl_sens_val.setText("Muy alta")
        elif value < 50:
            self.lbl_sens_val.setText("Recomendada")
        elif value < 85:
            self.lbl_sens_val.setText("Media")
        else:
            self.lbl_sens_val.setText("Baja (ruidosa)")

    def change_font_size(self, delta):
        new_size = max(12, min(28, self.transcript_font_size + delta))
        if new_size == self.transcript_font_size:
            return
        self.transcript_font_size = new_size
        self.qsettings.setValue("font_size", new_size)
        self.overlay.set_text_size(new_size + 5)
        self._do_search_filter()

    def generate_summary(self):
        text = "\n".join(
            f"[{entry['timestamp']}] {entry['text']}" for entry in self.all_entries
        )
        if not text.strip():
            QMessageBox.information(
                self, "Sin contenido", "No hay transcripcion para resumir todavia."
            )
            return
        language = self.combo_summary_lang.currentData() or "es"
        self.btn_summary.setEnabled(False)
        self.btn_summary.setText("Generando...")
        self._summary_worker = SummaryWorker(text, language)
        self._summary_worker.summary_ready.connect(self.on_summary_ready)
        self._summary_worker.start()

    @Slot(str, str)
    def on_summary_ready(self, summary, error):
        self.btn_summary.setEnabled(True)
        self.btn_summary.setText("Resumen IA")
        if not summary:
            QMessageBox.warning(
                self, "Error de resumen", error or "No se pudo generar el resumen."
            )
            return
        dialog = SummaryDialog(summary, self)
        dialog.exec()

    def open_history(self):
        dialog = HistoryDialog(self.logger.get_logs_directory(), self)
        dialog.exec()

    def _apply_alerts(self, entry):
        keywords = [
            keyword.strip().lower()
            for keyword in self.edit_alerts.text().split(",")
            if keyword.strip()
        ]
        entry.pop("alert", None)
        entry.pop("matched", None)
        if not keywords:
            return
        lowered = entry["text"].lower()
        matched = [keyword for keyword in keywords if keyword in lowered]
        if matched:
            entry["alert"] = True
            entry["matched"] = matched
            now = time.time()
            if now - self._last_alert_at >= 5.0:
                self._last_alert_at = now
                self.show_alert(matched)

    def show_alert(self, matched):
        self.alert_toast.show_message(", ".join(matched))
        if self.chk_alert_sound.isChecked():
            QApplication.beep()

    def _build_source(self):
        is_system = self.combo_source.currentData() == "system"
        if is_system:
            return SystemAudioSource(
                on_error=lambda message: self.sig_source_error.emit(message),
                on_status=lambda message: self.sig_recorder_status.emit(message),
            )
        return MicrophoneSource(
            on_error=lambda message: self.sig_source_error.emit(message),
            on_status=lambda message: self.sig_recorder_status.emit(message),
        )

    def start_listening(self):
        if self._source and self._source.is_running:
            return

        self.on_settings_changed()
        audio_queue = queue.Queue(maxsize=300)

        self._segmenter = PhraseSegmenter(
            energy_threshold=self.slider_sens.value() / 10000.0,
            on_phrase=self.sig_phrase_done.emit,
            on_level=self.sig_audio_level.emit,
            on_status=self.sig_recorder_status.emit,
            highpass=self.chk_enhance.isChecked()
            and self.combo_source.currentData() == "mic",
        )
        self._source = self._build_source()

        try:
            self._source.start(audio_queue)
            self._segmenter.start(audio_queue)
        except AudioSourceError as exc:
            self._source = None
            self._segmenter = None
            QMessageBox.critical(self, "No se pudo iniciar la captura", str(exc))
            self.update_status("Detenido")
            return

        self._paused = False
        self.start_session_time = time.time()
        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.btn_pause.setText("Pausar")
        self.update_status("Escuchando...")

    def toggle_pause(self):
        if self._paused:
            self._paused = False
            self.btn_pause.setText("Pausar")
            started = self.start_session_time
            self.start_listening()
            if started:
                self.start_session_time = started
        else:
            self._paused = True
            self._stop_capture()
            self.btn_pause.setText("Reanudar")
            self.btn_start.setEnabled(False)
            self.btn_pause.setEnabled(True)
            self.btn_stop.setEnabled(True)
            self.update_status("Pausado")

    def _stop_capture(self):
        if self._segmenter:
            self._segmenter.stop()
            self._segmenter = None
        if self._source:
            self._source.stop()
            self._source = None

    def stop_listening(self):
        self._stop_capture()
        self._paused = False
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.btn_pause.setText("Pausar")
        self.vu_bar.setValue(0)
        self.update_status("Detenido")

    @Slot(float)
    def update_audio_level(self, level):
        self.waveform.push_level(level)
        self.vu_bar.setValue(int(level * 100))

    @Slot(str)
    def update_status(self, status):
        running = bool(self._source and self._source.is_running)
        if self._paused:
            self.lbl_status_badge.setText("Pausado")
            self._set_badge_style("#eab308")
            return
        if not running:
            self.lbl_status_badge.setText("● Detenido")
            self._set_badge_style("#ef4444")
            return
        if "Transcribiendo" in status:
            self.lbl_status_badge.setText("Transcribiendo...")
            self._set_badge_style("#38bdf8")
        elif "Cargando" in status:
            self.lbl_status_badge.setText("Cargando modelo...")
            self._set_badge_style("#a855f7")
        elif "Voz" in status:
            self.lbl_status_badge.setText("Voz detectada...")
            self._set_badge_style("#f59e0b")
        elif "Capturando" in status:
            self.lbl_status_badge.setText("Capturando audio...")
            self._set_badge_style("#10b981")
        else:
            self.lbl_status_badge.setText("● Escuchando...")
            self._set_badge_style("#10b981")

    @Slot(dict)
    def on_phrase_transcribed(self, entry):
        now = time.time()
        gap = now - self._last_entry_wall_ts
        self._last_entry_wall_ts = now

        query = self.edit_search.text().strip().lower()
        merged = False
        if (
            self.all_entries
            and gap <= 6.5
            and self.all_entries[-1].get("lang") == entry.get("lang")
            and len(self.all_entries[-1]["text"]) < 450
            and not query
        ):
            previous = self.all_entries[-1]
            new_chunk = entry["text"].strip()
            if not previous["text"].endswith((".", "!", "?")) and len(new_chunk) > 1:
                if new_chunk[0].isupper() and not new_chunk.startswith(
                    ("I ", "I'", "OK", "API", "SQL", "BI", "DingTalk")
                ):
                    new_chunk = new_chunk[0].lower() + new_chunk[1:]
            previous["text"] = f"{previous['text']} {new_chunk}".strip()
            target = previous
            merged = True
        else:
            self.all_entries.append(entry)
            target = entry

        self._apply_alerts(target)

        if merged:
            self._do_search_filter()
        else:
            if not query or query in target["text"].lower():
                self.render_entry(target, search_query=query)

        self.lbl_phrase_count.setText(f"Bloques: {len(self.all_entries)}")
        self.lbl_file_info.setText(
            f"Guardado en: {os.path.basename(self.logger.get_current_txt_path())}"
        )
        if self.overlay.isVisible():
            self.overlay.set_entries(list(self.all_entries))

    def render_entry(self, entry, search_query=None):
        timestamp = entry["timestamp"]
        escaped = html.escape(entry["text"])
        lang = entry.get("lang", "EN").upper()
        text_size = self.transcript_font_size
        timestamp_size = max(11, text_size - 5)
        badge_size = max(10, text_size - 7)

        if search_query:
            pattern = re.compile(re.escape(search_query), re.IGNORECASE)
            highlighted = pattern.sub(
                lambda match: (
                    '<span style="background-color: #eab308; color: #09090b; '
                    f'font-weight: 800; border-radius: 3px; padding: 1px 4px;">'
                    f"{match.group(0)}</span>"
                ),
                escaped,
            )
        else:
            highlighted = escaped

        badge_bg = {
            "EN": "linear-gradient(135deg, #0284c7, #2563eb)",
            "ES": "linear-gradient(135deg, #059669, #10b981)",
            "PT": "linear-gradient(135deg, #7c3aed, #9333ea)",
        }.get(lang, "linear-gradient(135deg, #475569, #64748b)")
        accent = {"EN": "#38bdf8", "ES": "#10b981", "PT": "#c084fc"}.get(lang, "#64748b")
        if entry.get("alert"):
            accent = "#ef4444"
            alert_badge = (
                '<span style="background: linear-gradient(135deg, #b91c1c, #ef4444); '
                'color: #ffffff; padding: 2px 7px; border-radius: 4px; '
                f'font-size: {badge_size}px; font-weight: 800; letter-spacing: 0.5px; '
                'margin-left: 6px;">ALERTA</span>'
            )
        else:
            alert_badge = ""

        self.text_transcript.append(
            f"""
        <div style="margin-bottom: 8px; padding: 9px 13px; background-color: #13131c; border: 1px solid #232334; border-left: 3px solid {accent}; border-radius: 8px;">
            <div style="margin-bottom: 4px;">
                <span style="color: #06b6d4; font-family: Menlo, monospace; font-size: {timestamp_size}px; font-weight: bold; margin-right: 8px;">[{timestamp}]</span>
                <span style="background: {badge_bg}; color: #ffffff; padding: 2px 7px; border-radius: 4px; font-size: {badge_size}px; font-weight: 800; letter-spacing: 0.5px;">{lang}</span>
                {alert_badge}
            </div>
            <div style="color: #f1f5f9; font-size: {text_size}px; line-height: 1.6; font-family: -apple-system, sans-serif;">
                {highlighted}
            </div>
        </div>
        """
        )
        if self.chk_autoscroll.isChecked():
            cursor = self.text_transcript.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            self.text_transcript.setTextCursor(cursor)

    @Slot()
    def _do_search_filter(self):
        query = self.edit_search.text().strip().lower()
        self.text_transcript.clear()
        for entry in self.all_entries:
            if not query or query in entry["text"].lower():
                self.render_entry(entry, search_query=query)

    def copy_transcript(self):
        lines = [
            f"[{entry['timestamp']}] [{entry.get('lang', 'EN')}] {entry['text']}"
            for entry in self.all_entries
        ]
        QApplication.clipboard().setText("\n".join(lines))
        original = self.btn_copy.text()
        self.btn_copy.setText(f"Copiados {len(lines)} bloques")
        QTimer.singleShot(1600, lambda: self.btn_copy.setText(original))

    def save_meeting_txt(self):
        if not self.all_entries:
            QMessageBox.information(
                self, "Sin contenido", "Todavia no hay transcripcion para guardar."
            )
            return

        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        default_dir = desktop if os.path.isdir(desktop) else os.path.expanduser("~")
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        default_path = os.path.join(default_dir, f"reunion_{stamp}.txt")
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Guardar transcripcion de la reunion",
            default_path,
            "Texto (*.txt)",
        )
        if not path:
            return
        if not path.lower().endswith(".txt"):
            path += ".txt"

        model = (self.combo_model.currentData() or "").split("/")[-1]
        language = (self.combo_lang.currentData() or "en").upper()
        source = (
            "Audio del sistema"
            if self.combo_source.currentData() == "system"
            else "Microfono"
        )

        lines = [
            "=" * 60,
            "VOICE LOGGER MAC - TRANSCRIPCION DE REUNION",
            "=" * 60,
            f"Guardado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Duracion de la sesion: {self.lbl_timer.text()}",
            f"Modelo: {model} | Idioma: {language} | Fuente: {source}",
            f"Bloques: {len(self.all_entries)}",
            "=" * 60,
            "",
        ]
        for entry in self.all_entries:
            lines.append(
                f"[{entry['timestamp']}] [{entry.get('lang', 'EN')}] {entry['text']}"
            )
            lines.append("")

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
        except OSError as exc:
            QMessageBox.critical(self, "No se pudo guardar", str(exc))
            return

        QMessageBox.information(
            self, "Reunion guardada", f"Archivo guardado en:\n{path}"
        )

    def clear_display(self):
        self.text_transcript.clear()
        self.all_entries.clear()
        self.lbl_phrase_count.setText("Bloques: 0")
        self.transcriber.reset_context()
        if self.overlay.isVisible():
            self.overlay.set_entries([])

    def open_logs_folder(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(self.logger.get_logs_directory()))

    def update_session_clock(self):
        if self.start_session_time and self._source and self._source.is_running and not self._paused:
            elapsed = int(time.time() - self.start_session_time)
            hours = elapsed // 3600
            minutes = (elapsed % 3600) // 60
            seconds = elapsed % 60
            self.lbl_timer.setText(f"{hours:02d}:{minutes:02d}:{seconds:02d}")

    @Slot(str)
    def show_error(self, message):
        print(f"[VoiceLogger] {message}")
        self.lbl_status_badge.setText("● Error")
        self._set_badge_style("#ef4444")
        self.lbl_status_badge.setToolTip(message)

    @Slot(str)
    def on_source_error(self, message):
        self.show_error(message)
        if self._source:
            self.stop_listening()
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Permiso o error de captura")
        dialog.setIcon(QMessageBox.Warning)
        dialog.setText(message)
        open_settings = dialog.addButton("Abrir Ajustes del Sistema", QMessageBox.ActionRole)
        dialog.addButton("Cerrar", QMessageBox.RejectRole)
        dialog.exec()
        if dialog.clickedButton() == open_settings:
            QDesktopServices.openUrl(
                QUrl(
                    "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture"
                )
            )

    def apply_dark_theme(self):
        self.setStyleSheet(
            """
            QMainWindow { background-color: #0b0b10; color: #f8fafc; }
            QWidget {
                background-color: #0b0b10;
                color: #f8fafc;
                font-family: -apple-system, 'SF Pro Text', 'Helvetica Neue', sans-serif;
            }
            QFrame#headerCard {
                background-color: #12121a;
                border: 1px solid #242436;
                border-radius: 12px;
            }
            QGroupBox#settingsGroup {
                background-color: #101017;
                border: 1px solid #242436;
                border-radius: 12px;
                margin-top: 14px;
                padding-top: 16px;
                font-weight: 700;
                color: #e2e8f0;
                font-size: 12px;
            }
            QGroupBox#settingsGroup::title {
                subcontrol-origin: margin;
                left: 14px;
                padding: 0 6px;
                color: #38bdf8;
            }
            QLabel#fieldLabel {
                color: #94a3b8;
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 0.5px;
            }
            QComboBox {
                background-color: #171722;
                border: 1px solid #2e2e42;
                border-radius: 8px;
                padding: 6px 12px;
                color: #f1f5f9;
                font-size: 12px;
                font-weight: 500;
            }
            QComboBox:hover { border-color: #10b981; background-color: #1a1a27; }
            QComboBox::drop-down { border: none; width: 20px; }
            QComboBox QAbstractItemView {
                background-color: #171722;
                border: 1px solid #33334d;
                border-radius: 8px;
                selection-background-color: #27273d;
                selection-color: #10b981;
                color: #f1f5f9;
                padding: 4px;
            }
            QLineEdit {
                background-color: #14141e;
                border: 1px solid #2b2b3d;
                border-radius: 8px;
                padding: 7px 12px;
                color: #f1f5f9;
                font-size: 12px;
            }
            QLineEdit:focus { border-color: #6366f1; background-color: #171724; }
            QTextEdit {
                background-color: #0d0d14;
                border: 1px solid #222233;
                border-radius: 10px;
                padding: 10px;
                color: #f1f5f9;
                selection-background-color: #6366f1;
                selection-color: #ffffff;
            }
            QPushButton {
                font-size: 12px;
                font-weight: 700;
                padding: 7px 16px;
                border-radius: 8px;
                letter-spacing: 0.3px;
            }
            QPushButton#btn_start {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #059669, stop:1 #10b981);
                color: white;
                border: none;
            }
            QPushButton#btn_start:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #10b981, stop:1 #34d399);
            }
            QPushButton#btn_start:disabled { background: #1e293b; color: #64748b; }
            QPushButton#btn_secondary {
                background-color: #171722;
                color: #e2e8f0;
                border: 1px solid #2e2e42;
            }
            QPushButton#btn_secondary:hover { background-color: #242436; border-color: #3f3f5a; }
            QPushButton#btn_secondary:disabled {
                background-color: #111119;
                color: #475569;
                border-color: #1c1c28;
            }
            QPushButton#btn_header_action {
                background-color: #181826;
                color: #e2e8f0;
                border: 1px solid #2f2f45;
                padding: 5px 12px;
                font-size: 11px;
                font-weight: bold;
                border-radius: 8px;
            }
            QPushButton#btn_header_action:hover {
                background-color: #27273d;
                border-color: #4f46e5;
                color: #ffffff;
            }
            QPushButton#btn_font {
                background-color: #171722;
                color: #e2e8f0;
                border: 1px solid #2e2e42;
                padding: 3px 8px;
                font-size: 12px;
                font-weight: 700;
                border-radius: 6px;
            }
            QPushButton#btn_font:hover {
                border-color: #10b981;
                color: #10b981;
            }
            QPushButton#btn_summary {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #7c3aed, stop:1 #a855f7);
                color: white;
                border: none;
            }
            QPushButton#btn_summary:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #8b5cf6, stop:1 #c084fc);
            }
            QPushButton#btn_summary:disabled { background: #1e293b; color: #64748b; }
            QPushButton#btn_danger {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #b91c1c, stop:1 #ef4444);
                color: white;
                border: none;
            }
            QPushButton#btn_danger:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #dc2626, stop:1 #f87171);
            }
            QPushButton#btn_danger:disabled { background: #1e293b; color: #64748b; }
            QSlider::groove:horizontal {
                height: 6px;
                background: #1f1f2e;
                border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #059669, stop:1 #10b981);
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #ffffff;
                border: 2px solid #10b981;
                width: 14px;
                margin-top: -5px;
                margin-bottom: -5px;
                border-radius: 7px;
            }
            QCheckBox { color: #cbd5e1; }
            QCheckBox::indicator {
                width: 15px;
                height: 15px;
                border-radius: 4px;
                border: 1px solid #3f3f56;
                background: #14141e;
            }
            QCheckBox::indicator:checked { background-color: #10b981; border-color: #059669; }
            QScrollBar:vertical { background: #0e0e15; width: 10px; margin: 0; }
            QScrollBar::handle:vertical {
                background: #252538;
                min-height: 25px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical:hover { background: #3b3b57; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
            """
        )

    def closeEvent(self, event):
        self.stop_listening()
        if self.worker.isRunning():
            self.worker.stop()
        if self._summary_worker and self._summary_worker.isRunning():
            self._summary_worker.wait(1500)
        self.overlay.close()
        self.alert_toast.close()
        if hasattr(self, "tray_icon"):
            self.tray_icon.hide()
        event.accept()
        QApplication.quit()


def run_app():
    app = QApplication(sys.argv)
    app.setApplicationName("VoiceLogger Mac")
    app.setApplicationDisplayName("VoiceLogger Mac")
    app.setQuitOnLastWindowClosed(True)
    window = VoiceLoggerWindow()
    window.show()
    if "--autostart" in sys.argv:
        QTimer.singleShot(800, window.start_listening)
    sys.exit(app.exec())
