import AppKit
import SwiftUI
import UniformTypeIdentifiers
import WhisperKit

struct TranscriptEntry: Identifiable, Equatable {
    let id: UUID
    let timestamp: String
    let text: String
    let lang: String
}

@MainActor
final class AppState: ObservableObject {
    static let shared = AppState()

    @Published var isListening = false
    @Published var isPaused = false
    @Published var status = "Detenido"
    @Published var level: Float = 0
    @Published var compact = false
    @Published var alwaysOnTop = true { didSet { applyOnTop(); save() } }
    @Published var showSettings = true
    @Published var searchText = ""
    @Published var toastMessage: String?
    @Published var toastKind = "info"
    @Published var overlayVisible = false
    @Published var entries: [TranscriptEntry] = []
    @Published var liveSummary: String = ""
    @Published var qaQuestion: String = ""
    @Published var qaAnswer: String = ""
    private var summaryCounter = 0
    @Published var sessionElapsed = "00:00:00"
    @Published var transcriptFontSize: Int = 17 { didSet { save() } }

    // Settings
    @Published var source = "system" { didSet { save() } }
    @Published var modelName = WhisperKitTranscriber.defaultModel { didSet { transcriber = nil; save() } }
    @Published var language = "en" { didSet { transcriber = nil; save() } }
    @Published var context = "" { didSet { save() } }
    @Published var sensitivity = 35 { didSet { segmenter?.setEnergyThreshold(Float(sensitivity) / 10000.0); save() } }
    @Published var alerts = "" { didSet { save() } }
    @Published var alertSound = true { didSet { save() } }
    @Published var enhance = false { didSet { save() } }
    @Published var autoscroll = true { didSet { save() } }
    @Published var summaryLang = "es" { didSet { save() } }

    private let capture = SystemAudioCapture()
    private let logger = TranscriptLogger()
    private var transcriber: Transcriber?
    private var segmenter: PhraseSegmenter?
    private var sessionTimer: Timer?
    private var sessionStart: Date?
    private var lastAlertAt: TimeInterval = 0

    init() {
        load()
        capture.onAudio = { [weak self] samples in
            guard let self else { return }
            if self.isPaused { return }
            self.segmenter?.consume(samples)
        }
        capture.onStatus = { [weak self] msg in Task { @MainActor in self?.status = msg } }
        capture.onError = { [weak self] msg in Task { @MainActor in self?.status = "Error: \(msg)"; self?.toast(msg, kind: "error") } }
    }

    func applyOnTop() {
        for window in NSApp.windows where window.isVisible {
            if alwaysOnTop {
                window.level = .floating
                window.collectionBehavior.insert(.canJoinAllSpaces)
                window.collectionBehavior.insert(.stationary)
                window.collectionBehavior.insert(.fullScreenAuxiliary)
            } else {
                window.level = .normal
            }
        }
    }

    func toast(_ message: String, kind: String = "info") {
        toastMessage = message
        toastKind = kind
        Task { @MainActor in
            try? await Task.sleep(nanoseconds: 3_000_000_000)
            if self.toastMessage == message { self.toastMessage = nil }
        }
    }

    func start() async {
        guard !isListening else { return }
        if transcriber == nil {
            status = "Cargando modelo (la primera vez puede tardar)..."
            do {
                transcriber = try await WhisperKitTranscriber(model: modelName)
            } catch {
                status = "Error cargando modelo: \(error.localizedDescription)"
                toast(status, kind: "error")
                return
            }
        }
        let seg = PhraseSegmenter(
            energyThreshold: Float(sensitivity) / 10000.0,
            onPhrase: { [weak self] audio, ts in
                Task { @MainActor in
                    guard let self else { return }
                    self.status = "Transcribiendo..."
                    do {
                        guard let transcriber = self.transcriber else { return }
                        let (text, lang) = try await transcriber.transcribe(audio, language: self.language)
                        if !text.isEmpty {
                            self.logger.log(text: text, model: "whisperkit:\(self.modelName)", lang: lang, timestamp: ts)
                            let entry = TranscriptEntry(id: UUID(), timestamp: ts, text: text, lang: lang)
                            self.entries.append(entry)
                            self.checkAlerts(text)
                            self.summaryCounter += 1
                            if self.summaryCounter % 5 == 0 { self.refreshSummary() }
                            if self.overlayVisible {
                                // keep last 5
                            }
                        }
                    } catch {
                        self.status = "Error: \(error.localizedDescription)"
                        self.toast(error.localizedDescription, kind: "error")
                    }
                    self.status = "Escuchando..."
                }
            },
            onLevel: { [weak self] lvl in
                Task { @MainActor in self?.level = lvl }
            }
        )
        segmenter = seg
        do {
            try await capture.start()
            isListening = true
            isPaused = false
            status = "Escuchando audio del sistema..."
            sessionStart = Date()
            startTimer()
        } catch {
            status = "No se pudo iniciar la captura"
            toast(status, kind: "error")
        }
    }

    func stop() async {
        segmenter?.flush()
        await capture.stop()
        isListening = false
        isPaused = false
        status = "Detenido"
        level = 0
        stopTimer()
    }

    func togglePause() {
        guard isListening else { return }
        isPaused.toggle()
        status = isPaused ? "Pausado" : "Escuchando..."
    }

    func startTimer() {
        stopTimer()
        sessionTimer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { [weak self] _ in
            Task { @MainActor in
                guard let self, let start = self.sessionStart else { return }
                let elapsed = Date().timeIntervalSince(start)
                let h = Int(elapsed) / 3600
                let m = (Int(elapsed) % 3600) / 60
                let s = Int(elapsed) % 60
                self.sessionElapsed = String(format: "%02d:%02d:%02d", h, m, s)
            }
        }
    }

    func stopTimer() {
        sessionTimer?.invalidate()
        sessionTimer = nil
    }

    func checkAlerts(_ text: String) {
        let keywords = alerts.split(separator: ",").map { $0.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() }.filter { !$0.isEmpty }
        guard !keywords.isEmpty else { return }
        let lower = text.lowercased()
        let matched = keywords.filter { lower.contains($0) }
        guard !matched.isEmpty else { return }
        let now = Date().timeIntervalSince1970
        guard now - lastAlertAt > 2 else { return }
        lastAlertAt = now
        toast("Alerta: \(matched.joined(separator: ", "))", kind: "alert")
        if alertSound {
            NSSound.beep()
        }
    }

    func refreshSummary() {
        let text = entries.map { "[\($0.timestamp)] \($0.text)" }.joined(separator: "\n")
        guard !text.isEmpty else { return }
        Task.detached(priority: .utility) {
            let process = Process()
            process.executableURL = URL(fileURLWithPath: "/Users/julianurrego/Desktop/VoiceLoggerMac/.venv/bin/python")
            process.arguments = ["-c", "import sys; sys.path.insert(0,'/Users/julianurrego/Desktop/VoiceLoggerMac'); from summarizer import build_summary; print(build_summary(sys.stdin.read(),'en')[0])"]
            let input = Pipe()
            let output = Pipe()
            process.standardInput = input
            process.standardOutput = output
            do {
                try process.run()
                if let data = text.data(using: .utf8) { try input.fileHandleForWriting.write(contentsOf: data); try input.fileHandleForWriting.close() }
                process.waitUntilExit()
                let out = String(data: output.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
                await MainActor.run { AppState.shared.liveSummary = out.trimmingCharacters(in: .whitespacesAndNewlines) }
            } catch { }
        }
    }

    func askQuestion() {
        let q = qaQuestion.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !q.isEmpty else { return }
        let text = entries.map { "[\($0.timestamp)] \($0.text)" }.joined(separator: "\n")
        Task.detached(priority: .utility) {
            let process = Process()
            process.executableURL = URL(fileURLWithPath: "/Users/julianurrego/Desktop/VoiceLoggerMac/.venv/bin/python")
            process.arguments = ["-c", "import sys; sys.path.insert(0,'/Users/julianurrego/Desktop/VoiceLoggerMac'); from summarizer import ask_question; d=sys.stdin.read().split('|||'); print(ask_question(d[0], d[1].strip(), 'en'))"]
            // above split approach is fragile; instead pass via env not. Use simpler: write text+question to stdin with delimiter
            let input = Pipe()
            let output = Pipe()
            process.standardInput = input
            process.standardOutput = output
            do {
                try process.run()
                let payload = text + "\n|||\n" + q
                if let data = payload.data(using: .utf8) { try input.fileHandleForWriting.write(contentsOf: data); try input.fileHandleForWriting.close() }
                process.waitUntilExit()
                let out = String(data: output.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
                await MainActor.run { AppState.shared.qaAnswer = out.trimmingCharacters(in: .whitespacesAndNewlines) }
            } catch { }
        }
    }

    func saveMeetingTxt() {
        let text = entries.map { "[\($0.timestamp)] [\($0.lang)] \($0.text)" }.joined(separator: "\n")
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.plainText]
        panel.nameFieldStringValue = "reunion.txt"
        if panel.runModal() == .OK, let url = panel.url {
            try? text.write(to: url, atomically: true, encoding: .utf8)
            toast("Guardado: \(url.lastPathComponent)", kind: "info")
        }
    }

    func openLogsFolder() {
        NSWorkspace.shared.open(logger.logsDirectory)
    }

    func copyAll() {
        let text = entries.map { "[\($0.timestamp)] \($0.text)" }.joined(separator: "\n")
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
        toast("Copiado al portapapeles", kind: "info")
    }

    func clearDisplay() {
        entries.removeAll()
    }

    func changeFontSize(_ delta: Int) {
        transcriptFontSize = max(12, min(28, transcriptFontSize + delta))
    }

    func save() {
        let d = UserDefaults.standard
        d.set(source, forKey: "source")
        d.set(modelName, forKey: "model")
        d.set(language, forKey: "language")
        d.set(context, forKey: "context")
        d.set(sensitivity, forKey: "sensitivity")
        d.set(alerts, forKey: "alerts")
        d.set(alertSound, forKey: "alert_sound")
        d.set(enhance, forKey: "enhance")
        d.set(autoscroll, forKey: "autoscroll")
        d.set(summaryLang, forKey: "summary_lang")
        d.set(transcriptFontSize, forKey: "font_size")
        d.set(alwaysOnTop, forKey: "always_on_top")
        d.set(showSettings, forKey: "settings_visible")
    }

    private func load() {
        let d = UserDefaults.standard
        source = d.string(forKey: "source") ?? "system"
        modelName = d.string(forKey: "model") ?? WhisperKitTranscriber.defaultModel
        language = d.string(forKey: "language") ?? "en"
        context = d.string(forKey: "context") ?? ""
        sensitivity = d.object(forKey: "sensitivity") as? Int ?? 35
        alerts = d.string(forKey: "alerts") ?? ""
        alertSound = d.object(forKey: "alert_sound") as? Bool ?? true
        enhance = d.object(forKey: "enhance") as? Bool ?? false
        autoscroll = d.object(forKey: "autoscroll") as? Bool ?? true
        summaryLang = d.string(forKey: "summary_lang") ?? "es"
        transcriptFontSize = d.object(forKey: "font_size") as? Int ?? 17
        alwaysOnTop = d.object(forKey: "always_on_top") as? Bool ?? true
        showSettings = d.object(forKey: "settings_visible") as? Bool ?? true
    }

    var filteredEntries: [TranscriptEntry] {
        if searchText.trimmingCharacters(in: .whitespaces).isEmpty { return entries }
        let q = searchText.lowercased()
        return entries.filter { $0.text.lowercased().contains(q) || $0.timestamp.contains(q) }
    }
}

// MARK: - Views

struct ToastView: View {
    let message: String
    let kind: String
    var body: some View {
        let color: Color = kind == "alert" ? .red : (kind == "error" ? .orange : .blue)
        Text(message)
            .font(.callout.bold())
            .foregroundStyle(.white)
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
            .background(color.opacity(0.9), in: RoundedRectangle(cornerRadius: 10))
            .shadow(radius: 6)
    }
}

struct FullView: View {
    @ObservedObject var state: AppState
    @Environment(\.openWindow) private var openWindow
    @State private var overlayEntries: [TranscriptEntry] = []

    var body: some View {
        ZStack(alignment: .top) {
            VStack(spacing: 12) {
                header
                if state.showSettings { settingsPanel }
                controls
                searchRow
                HSplitView {
                    transcriptList
                    VStack(alignment: .leading) {
                        Text("Resumen en vivo").font(.headline)
                        ScrollView { Text(state.liveSummary).font(.system(size: CGFloat(state.transcriptFontSize))).frame(maxWidth: .infinity, alignment: .leading) }
                        Button("Actualizar ahora") { state.refreshSummary() }
                    }.frame(minWidth: 220)
                    VStack(alignment: .leading) {
                        Text("Preguntar al modelo").font(.headline)
                        TextField("Tu pregunta...", text: $state.qaQuestion).textFieldStyle(.roundedBorder)
                        Button("Preguntar") { state.askQuestion() }
                        ScrollView { Text(state.qaAnswer).font(.system(size: CGFloat(state.transcriptFontSize))).frame(maxWidth: .infinity, alignment: .leading) }
                    }.frame(minWidth: 220)
                }
                footer
            }
            .padding()
            .frame(minWidth: 760, minHeight: 440)
            .onHover { hovering in for w in NSApp.windows where w.isVisible { w.alphaValue = hovering ? 1.0 : 0.3 } }
            if let msg = state.toastMessage {
                ToastView(message: msg, kind: state.toastKind)
                    .padding(.top, 8)
                    .transition(.move(edge: .top).combined(with: .opacity))
            }
        }
        .background(Color(red: 0.07, green: 0.07, blue: 0.11))
        .preferredColorScheme(.dark)
    }

    private var header: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text("VoiceLogger Mac")
                    .font(.title2.bold())
                    .foregroundStyle(Color(red: 0.06, green: 0.73, blue: 0.51))
                Text("ScreenCaptureKit + WhisperKit · nativo")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Text("\(state.modelName.split(separator: "/").last ?? "") · \(state.language.uppercased())")
                .font(.caption.bold())
                .padding(.horizontal, 10).padding(.vertical, 5)
                .background(Color.indigo.opacity(0.3), in: Capsule())
            Text(state.sessionElapsed)
                .font(.system(.body, design: .monospaced).bold())
                .padding(.horizontal, 12).padding(.vertical, 5)
                .background(Color(red: 0.09, green: 0.09, blue: 0.13), in: Capsule())
                .overlay(Capsule().stroke(Color.gray.opacity(0.3)))
            Button("Subtítulos") { state.overlayVisible.toggle(); if state.overlayVisible { openWindow(id: "overlay") } }
                .buttonStyle(.bordered)
            Button(state.showSettings ? "Ajustes ▴" : "Ajustes ▾") { state.showSettings.toggle(); state.save() }
                .buttonStyle(.bordered)
        }
    }

    private var settingsPanel: some View {
        VStack(spacing: 10) {
            HStack {
                VStack(alignment: .leading) {
                    Text("FUENTE DE AUDIO").font(.caption).foregroundStyle(.secondary)
                    Picker("", selection: $state.source) {
                        Text("Sistema").tag("system")
                        Text("Micrófono").tag("mic")
                    }.pickerStyle(.segmented)
                }
                VStack(alignment: .leading) {
                    Text("MODELO WHISPER").font(.caption).foregroundStyle(.secondary)
                    Picker("", selection: $state.modelName) {
                        ForEach(WhisperKitTranscriber.modelChoices, id: \.id) { c in
                            Text(c.label).tag(c.id)
                        }
                    }.pickerStyle(.menu)
                }
                VStack(alignment: .leading) {
                    Text("IDIOMA").font(.caption).foregroundStyle(.secondary)
                    Picker("", selection: $state.language) {
                        Text("Inglés").tag("en")
                        Text("Auto").tag("auto")
                        Text("Español").tag("es")
                        Text("Portugués").tag("pt")
                    }.pickerStyle(.menu)
                }
            }
            HStack {
                VStack(alignment: .leading) {
                    Text("CONTEXTO / VOCABULARIO").font(.caption).foregroundStyle(.secondary)
                    TextField("nombres, productos, jerga...", text: $state.context)
                        .textFieldStyle(.roundedBorder)
                }
                VStack(alignment: .leading) {
                    Text("SENSIBILIDAD VAD").font(.caption).foregroundStyle(.secondary)
                    HStack {
                        Slider(value: Binding(get: { Double(state.sensitivity) }, set: { state.sensitivity = Int($0) }), in: 10...150)
                        Text(state.sensitivity < 25 ? "Muy alta" : state.sensitivity < 50 ? "Recomendada" : state.sensitivity < 85 ? "Media" : "Baja")
                            .font(.caption).foregroundStyle(.cyan).frame(width: 80)
                    }
                }
            }
            HStack {
                VStack(alignment: .leading) {
                    Text("ALERTAS (coma)").font(.caption).foregroundStyle(.secondary)
                    TextField("julian, deadline...", text: $state.alerts)
                        .textFieldStyle(.roundedBorder)
                }
                Toggle("Sonido", isOn: $state.alertSound)
                Toggle("Autoscroll", isOn: $state.autoscroll)
                Stepper("Fuente \(state.transcriptFontSize)", value: $state.transcriptFontSize, in: 12...28)
            }
        }
        .padding(12)
        .background(Color(red: 0.1, green: 0.1, blue: 0.15), in: RoundedRectangle(cornerRadius: 10))
    }

    private var controls: some View {
        HStack(spacing: 10) {
            Text(state.isListening ? "● Escuchando" : "● Detenido")
                .font(.caption.bold())
                .foregroundStyle(state.isListening ? .green : .red)
            ProgressView(value: Double(state.level))
                .frame(width: 120)
            Spacer()
            Button(state.isListening ? "Detener" : "Iniciar") {
                Task { if state.isListening { await state.stop() } else { await state.start() } }
            }
            .keyboardShortcut(.return, modifiers: .command)
            .buttonStyle(.borderedProminent)
            Button("Pausar") { state.togglePause() }
                .disabled(!state.isListening)
            Button("Guardar TXT") { state.saveMeetingTxt() }
                .keyboardShortcut("s", modifiers: .command)
            Button("Copiar") { state.copyAll() }
            Button("Limpiar") { state.clearDisplay() }
            Button("Ver logs") { state.openLogsFolder() }
        }
    }

    private var searchRow: some View {
        HStack {
            TextField("Filtrar...", text: $state.searchText)
                .textFieldStyle(.roundedBorder)
            Button("A-") { state.changeFontSize(-1) }
            Button("A+") { state.changeFontSize(1) }
        }
    }

    private var transcriptList: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 6) {
                    ForEach(state.filteredEntries) { e in
                        VStack(alignment: .leading, spacing: 2) {
                            Text("[\(e.timestamp)]")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            Text(e.text)
                                .font(.system(size: CGFloat(state.transcriptFontSize)))
                        }
                        .padding(6)
                        .background(Color.white.opacity(0.04), in: RoundedRectangle(cornerRadius: 6))
                        .id(e.id)
                    }
                }
            }
            .onChange(of: state.entries.count) { _ in
                if state.autoscroll, let last = state.filteredEntries.last {
                    withAnimation { proxy.scrollTo(last.id, anchor: .bottom) }
                }
            }
        }
        .frame(minHeight: 200)
    }

    private var footer: some View {
        HStack {
            Text("\(state.entries.count) bloques")
                .font(.caption).foregroundStyle(.secondary)
            Spacer()
            Text("Guardado en logs")
                .font(.caption).foregroundStyle(.secondary)
        }
    }
}

struct CompactView: View {
    @ObservedObject var state: AppState
    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Circle().fill(state.isListening ? Color.green : Color.red).frame(width: 8, height: 8)
                Text("VOICE LOGGER").font(.caption.bold()).foregroundStyle(.secondary)
                Spacer()
                Button("Expandir") { state.compact = false }
                    .buttonStyle(.link).font(.caption)
            }
            Text(state.entries.last?.text ?? "Esperando audio…")
                .font(.body).lineLimit(2)
            ProgressView(value: Double(state.level))
            HStack(spacing: 8) {
                Button(state.isListening ? "Detener" : "Iniciar") {
                    Task { if state.isListening { await state.stop() } else { await state.start() } }
                }.buttonStyle(.borderedProminent).controlSize(.small)
                Text(state.status).font(.caption).foregroundStyle(.secondary).lineLimit(1)
            }
        }
        .padding(12)
        .frame(width: 340, height: 130)
        .onHover { hovering in for w in NSApp.windows where w.isVisible { w.alphaValue = hovering ? 1.0 : 0.3 } }
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
    }
}

struct OverlayView: View {
    @ObservedObject var state = AppState.shared
    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Circle().fill(state.isListening ? Color.green : Color.red).frame(width: 8, height: 8)
                Text("Subtítulos").font(.caption.bold()).foregroundStyle(.secondary)
                Spacer()
            }
            Text(state.entries.suffix(5).map(\.text).joined(separator: "\n"))
                .font(.system(size: CGFloat(state.transcriptFontSize + 4)))
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(12)
        .frame(width: 420, height: 160)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
    }
}

struct ContentView: View {
    @StateObject private var state = AppState.shared

    var body: some View {
        Group {
            if state.compact {
                CompactView(state: state)
            } else {
                FullView(state: state)
            }
        }
        .onAppear { state.applyOnTop() }
    }
}

@main
struct VoiceLoggerMacNativeApp: App {
    @Environment(\.openWindow) private var openWindow

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        .windowResizability(.contentMinSize)

        Window("Subtítulos", id: "overlay") {
            OverlayView()
        }
        .windowStyle(.plain)
        .windowResizability(.contentSize)

        MenuBarExtra("VoiceLogger", systemImage: "waveform") {
            Button("Mostrar") { NSApp.activate(ignoringOtherApps: true) }
            Button("Iniciar/Detener") { Task { if AppState.shared.isListening { await AppState.shared.stop() } else { await AppState.shared.start() } } }
            Button("Subtítulos") { AppState.shared.overlayVisible.toggle(); if AppState.shared.overlayVisible { openWindow(id: "overlay") } }
            Divider()
            Button("Salir") { NSApp.terminate(nil) }
        }
    }
}
