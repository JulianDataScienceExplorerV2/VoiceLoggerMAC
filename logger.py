import json
import os
from datetime import datetime
from threading import Lock


class TranscriptLogger:
    def __init__(self, logs_dir=None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.logs_dir = os.path.abspath(logs_dir or os.path.join(base_dir, "logs"))
        os.makedirs(self.logs_dir, exist_ok=True)
        self._lock = Lock()
        self._current_date = None
        self._txt_path = None
        self._jsonl_path = None

    def _get_file_paths(self):
        today = datetime.now().strftime("%Y-%m-%d")
        if today != self._current_date or not self._txt_path:
            self._current_date = today
            self._txt_path = os.path.join(self.logs_dir, f"transcripcion_{today}.txt")
            self._jsonl_path = os.path.join(self.logs_dir, f"transcripcion_{today}.jsonl")
            if not os.path.exists(self._txt_path):
                with open(self._txt_path, "a", encoding="utf-8") as f:
                    f.write(f"=== Meeting Log - {today} ===\n\n")
        return self._txt_path, self._jsonl_path

    def log(self, text, model_name="mlx-whisper", lang="en", timestamp=None):
        text = (text or "").strip()
        if not text:
            return None

        now = datetime.now()
        timestamp_str = timestamp or now.strftime("%H:%M:%S")
        lang_tag = (lang or "en").upper()

        entry = {
            "timestamp": timestamp_str,
            "iso": now.isoformat(),
            "lang": lang_tag,
            "model": model_name,
            "text": text,
        }

        with self._lock:
            txt_path, jsonl_path = self._get_file_paths()
            with open(txt_path, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp_str}] [{lang_tag}] {text}\n")
                f.flush()
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                f.flush()

        return entry

    def get_logs_directory(self):
        return self.logs_dir

    def get_current_txt_path(self):
        with self._lock:
            txt_path, _ = self._get_file_paths()
            return txt_path

    def get_session_text(self):
        with self._lock:
            txt_path, _ = self._get_file_paths()
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                return f.read()
        except OSError:
            return ""
