import os
import re

from PySide6.QtCore import Qt
from PySide6.QtGui import QTextDocument
from PySide6.QtPrintSupport import QPrinter
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class HistoryDialog(QDialog):
    def __init__(self, logs_dir, parent=None):
        super().__init__(parent)
        self.logs_dir = logs_dir
        self.setWindowTitle("Historial de reuniones")
        self.resize(1000, 660)

        self._content = ""
        self._current_file = None

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        splitter = QSplitter(Qt.Horizontal)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)
        left_layout.addWidget(QLabel("Sesiones guardadas"))
        self.list_files = QListWidget()
        self.list_files.setMinimumWidth(240)
        self.list_files.currentItemChanged.connect(self._load_selected)
        left_layout.addWidget(self.list_files)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        self.edit_search = QLineEdit()
        self.edit_search.setPlaceholderText("Buscar dentro del archivo...")
        self.edit_search.textChanged.connect(self._find_in_content)
        right_layout.addWidget(self.edit_search)

        self.viewer = QTextEdit()
        self.viewer.setReadOnly(True)
        right_layout.addWidget(self.viewer)

        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        self.btn_md = QPushButton("Exportar Markdown")
        self.btn_md.clicked.connect(self._export_markdown)
        self.btn_pdf = QPushButton("Exportar PDF")
        self.btn_pdf.clicked.connect(self._export_pdf)
        self.btn_copy = QPushButton("Copiar")
        self.btn_copy.clicked.connect(self._copy_all)
        buttons.addWidget(self.btn_md)
        buttons.addWidget(self.btn_pdf)
        buttons.addWidget(self.btn_copy)
        buttons.addStretch()
        right_layout.addLayout(buttons)

        splitter.addWidget(right)
        splitter.setSizes([300, 700])
        root.addWidget(splitter)

        close_row = QHBoxLayout()
        close_row.addStretch()
        btn_close = QPushButton("Cerrar")
        btn_close.clicked.connect(self.accept)
        close_row.addWidget(btn_close)
        root.addLayout(close_row)

        self._populate()

    def _populate(self):
        try:
            files = [
                name
                for name in os.listdir(self.logs_dir)
                if name.endswith(".txt") and name.startswith("transcripcion_")
            ]
        except OSError:
            files = []
        files.sort(reverse=True)
        self.list_files.clear()
        for name in files:
            self.list_files.addItem(name)
        if files:
            self.list_files.setCurrentRow(0)

    def _load_selected(self, item):
        if not item:
            return
        path = os.path.join(self.logs_dir, item.text())
        try:
            with open(path, "r", encoding="utf-8") as f:
                self._content = f.read()
        except OSError as exc:
            self._content = ""
            self.viewer.setPlainText(f"No se pudo leer el archivo: {exc}")
            return
        self._current_file = path
        self.viewer.setPlainText(self._content)
        self.edit_search.clear()

    def _find_in_content(self, query):
        query = query.strip()
        if not query or not self._content:
            return
        found = self.viewer.find(query)
        if not found:
            cursor = self.viewer.textCursor()
            cursor.movePosition(cursor.MoveOperation.Start)
            self.viewer.setTextCursor(cursor)
            self.viewer.find(query)

    def _current_text(self):
        return self.viewer.toPlainText() or self._content

    def _to_markdown(self):
        lines = self._current_text().splitlines()
        out = []
        for line in lines:
            match = re.match(r"^\[(\d{2}:\d{2}:\d{2})\] \[([A-Z]{2})\] (.*)$", line.strip())
            if match:
                out.append(f"**{match.group(1)}** · {match.group(2)}")
                out.append("")
                out.append(match.group(3))
                out.append("")
            elif line.strip():
                out.append(line)
        return "\n".join(out)

    def _export_markdown(self):
        if not self._current_text().strip():
            QMessageBox.information(self, "Sin contenido", "No hay texto para exportar.")
            return
        default = os.path.splitext(os.path.basename(self._current_file or "reunion"))[0] + ".md"
        path, _ = QFileDialog.getSaveFileName(
            self, "Exportar Markdown", default, "Markdown (*.md)"
        )
        if not path:
            return
        if not path.lower().endswith(".md"):
            path += ".md"
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self._to_markdown())
        except OSError as exc:
            QMessageBox.critical(self, "Error", str(exc))
            return
        QMessageBox.information(self, "Exportado", f"Archivo guardado en:\n{path}")

    def _export_pdf(self):
        if not self._current_text().strip():
            QMessageBox.information(self, "Sin contenido", "No hay texto para exportar.")
            return
        default = os.path.splitext(os.path.basename(self._current_file or "reunion"))[0] + ".pdf"
        path, _ = QFileDialog.getSaveFileName(
            self, "Exportar PDF", default, "PDF (*.pdf)"
        )
        if not path:
            return
        if not path.lower().endswith(".pdf"):
            path += ".pdf"
        document = QTextDocument()
        document.setPlainText(self._current_text())
        printer = QPrinter(QPrinter.HighResolution)
        printer.setOutputFormat(QPrinter.OutputFormat.PdfFormat)
        printer.setOutputFileName(path)
        document.print_(printer)
        QMessageBox.information(self, "Exportado", f"Archivo guardado en:\n{path}")

    def _copy_all(self):
        from PySide6.QtWidgets import QApplication

        QApplication.clipboard().setText(self._current_text())
