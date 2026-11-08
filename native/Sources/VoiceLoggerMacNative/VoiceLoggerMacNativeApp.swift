import AppKit
import SwiftUI
import UniformTypeIdentifiers

struct TranscriptEntry: Identifiable, Equatable {
    let id: UUID
    let timestamp: String
    let text: String
    let lang: String
    var alert: Bool = false
}

struct QAItem: Identifiable, Equatable {
    let id = UUID()
    let question: String
    var answer: String?
    var failed = false
}

/// Values that change many times per second (audio level, session clock).
/// Kept apart from `AppState` so they only re-render the meter, not the
/// whole window with the transcript list.
@MainActor
final class LiveMetrics: ObservableObject {
    @Published var level: Float = 0
    @Published var elapsed = "00:00:00"
}

@MainActor
final class AppState: ObservableObject {
    static let shared = AppState()

    // Session
    @Published var isListening = false
    @Published var isStarting = false
    @Published var isPaused = false
    @Published var status = "Detenido"
    @Published var compact = false
    @Published var overlayVisible = false
    @Published var entries: [TranscriptEntry] = []
    /// Bumped on every transcript change (append or merge) to drive autoscroll.
    @Published var transcriptRevision = 0
    @Published var searchText = ""
    @Published var toastMessage: String?
    @Published var toastKind = "info"
    let metrics = LiveMetrics()

    // AI (summary + questions)
    @Published var liveSummary = ""
    @Published var summaryUpdatedAt: String?
    @Published var summaryError: String?
    @Published var isSummarizing = false
    @Published var qaQuestion = ""
    @Published var qaItems: [QAItem] = []
    @Published var qaRevision = 0

    // Settings (persisted)
    @Published var modelName = WhisperKitTranscriber.defaultModel {
        didSet {
            if oldValue != modelName { transcriber = nil }
            save()
        }
    }
    /// The language is passed on every `transcribe` call: changing it does
    /// not need to reload the model.
    @Published var language = "en" { didSet { save() } }
    @Published var sensitivity = 35 {
        didSet {
            segmenter?.setEnergyThreshold(Float(sensitivity) / 10000.0)
            save()
        }
    }
    @Published var alerts = "" { didSet { save() } }
    @Published var alertSound = true { didSet { save() } }
    @Published var autoscroll = true { didSet { save() } }
    @Published var summaryLang = "es" { didSet { save() } }
    @Published var autoSummary = true { didSet { save() } }
    @Published var transcriptFontSize = 17 { didSet { save() } }
    @Published var alwaysOnTop = true { didSet { applyOnTop(); save() } }
    @Published var dimWhenIdle = true { didSet { save() } }

    private let capture = SystemAudioCapture()
    private let logger = TranscriptLogger()
    private var transcriber: (any Transcriber)?
    private var transcriberTask: Task<any Transcriber, Error>?
    private var transcriberTaskModel: String?
    private var segmenter: PhraseSegmenter?
    private var bridge: PythonBridge?
    private var sessionTimer: Timer?
    private var sessionStart: Date?
    private var lastAlertAt: TimeInterval = 0
    private var lastEntryAt: Date?
    private var phrasesSinceSummary = 0
    private var lastSummaryAt: Date?
    private var isLoading = false

    init() {
        load()
        capture.onAudio = { [weak self] samples in
            guard let self else { return }
            if self.isPaused { return }
            self.segmenter?.consume(samples)
        }
        capture.onStatus = { [weak self] msg in Task { @MainActor in self?.status = msg } }
        capture.onError = { [weak self] msg in
            Task { @MainActor in
                self?.status = "Error: \(msg)"
                self?.toast(msg, kind: "error")
            }
        }
    }

    // MARK: Window helpers

    func applyOnTop() {
        guard !isLoading, let app = NSApp else { return }
        for window in app.windows where window.isVisible && window.styleMask.contains(.titled) {
            if alwaysOnTop {
                window.level = .floating
                window.collectionBehavior.insert([.canJoinAllSpaces, .fullScreenAuxiliary])
            } else {
                window.level = .normal
            }
        }
    }

    func toast(_ message: String, kind: String = "info") {
        withAnimation(.easeOut(duration: 0.2)) {
            toastMessage = message
            toastKind = kind
        }
        Task { @MainActor in
            try? await Task.sleep(for: .seconds(kind == "error" ? 5 : 3))
            if self.toastMessage == message {
                withAnimation(.easeIn(duration: 0.2)) { self.toastMessage = nil }
            }
        }
    }

    // MARK: Capture

    func toggleListening() async {
        if isListening { await stop() } else { await start() }
    }

    func start() async {
        guard !isListening, !isStarting else { return }
        isStarting = true
        defer { isStarting = false }
        do {
            _ = try await ensureTranscriber()
        } catch {
            status = "Error cargando modelo: \(error.localizedDescription)"
            toast(status, kind: "error")
            return
        }
        let seg = PhraseSegmenter(
            energyThreshold: Float(sensitivity) / 10000.0,
            onPhrase: { [weak self] audio, ts in
                Task { @MainActor in
                    await self?.handlePhrase(audio, timestamp: ts)
                }
            },
            onLevel: { [weak self] lvl in
                Task { @MainActor in self?.updateLevel(lvl) }
            }
        )
        segmenter = seg
        do {
            try await capture.start()
            isListening = true
            isPaused = false
            status = "Escuchando..."
            sessionStart = Date()
            metrics.elapsed = "00:00:00"
            startTimer()
        } catch {
            segmenter = nil
            status = "No se pudo iniciar la captura"
            toast(
                "No se pudo iniciar la captura. Revisa el permiso de Grabación de pantalla y audio del sistema.",
                kind: "error"
            )
        }
    }

    func stop() async {
        guard isListening else { return }
        // Stop the audio first so the segmenter isn't fed while it flushes.
        await capture.stop()
        segmenter?.flush()
        segmenter = nil
        isListening = false
        isPaused = false
        status = "Detenido"
        metrics.level = 0
        stopTimer()
    }

    func togglePause() {
        guard isListening else { return }
        isPaused.toggle()
        status = isPaused ? "Pausado" : "Escuchando..."
        if isPaused { metrics.level = 0 }
    }

    /// Loads the WhisperKit model once; phrases that arrive while it loads
    /// (e.g. after switching models mid-session) wait for the same task
    /// instead of being dropped.
    private func ensureTranscriber() async throws -> any Transcriber {
        if let transcriber { return transcriber }
        let model = modelName
        let task: Task<any Transcriber, Error>
        if let pending = transcriberTask, transcriberTaskModel == model {
            task = pending
        } else {
            status = "Cargando modelo (la primera vez puede tardar)..."
            task = Task<any Transcriber, Error> {
                let loaded: any Transcriber = try await WhisperKitTranscriber(model: model)
                return loaded
            }
            transcriberTask = task
            transcriberTaskModel = model
        }
        do {
            let loaded = try await task.value
            if transcriberTaskModel == model {
                transcriberTask = nil
                transcriberTaskModel = nil
            }
            if modelName == model { transcriber = loaded }
            return loaded
        } catch {
            if transcriberTaskModel == model {
                transcriberTask = nil
                transcriberTaskModel = nil
            }
            throw error
        }
    }

    private func handlePhrase(_ audio: [Float], timestamp ts: String) async {
        do {
            let transcriber = try await ensureTranscriber()
            status = "Transcribiendo..."
            let (raw, lang) = try await transcriber.transcribe(audio, language: language)
            if let text = TranscriptCleaner.clean(raw) {
                logger.log(text: text, model: "whisperkit:\(modelName)", lang: lang, timestamp: ts)
                appendEntry(text: text, lang: lang, timestamp: ts)
            }
        } catch {
            // Keep the error visible instead of overwriting it right away.
            status = "Error: \(error.localizedDescription)"
            toast(error.localizedDescription, kind: "error")
            return
        }
        if isListening {
            status = isPaused ? "Pausado" : "Escuchando..."
        } else {
            status = "Detenido"
        }
    }

    private func updateLevel(_ value: Float) {
        guard isListening, !isPaused else { return }
        if abs(metrics.level - value) >= 0.02 { metrics.level = value }
    }

    /// Joins consecutive short phrases into one paragraph (same rule as the
    /// Python app) so the transcript reads like text, not like a log.
    private func appendEntry(text: String, lang: String, timestamp: String) {
        let now = Date()
        let gap = lastEntryAt.map { now.timeIntervalSince($0) } ?? .infinity
        lastEntryAt = now
        let matched = matchedAlerts(in: text)

        if let last = entries.last, gap <= 6.5, last.lang == lang, last.text.count < 450 {
            entries[entries.count - 1] = TranscriptEntry(
                id: last.id,
                timestamp: last.timestamp,
                text: last.text + " " + text,
                lang: lang,
                alert: last.alert || !matched.isEmpty
            )
        } else {
            entries.append(TranscriptEntry(
                id: UUID(), timestamp: timestamp, text: text, lang: lang, alert: !matched.isEmpty
            ))
        }
        transcriptRevision += 1
        notifyAlert(matched)
        phrasesSinceSummary += 1
        maybeAutoSummarize()
    }

    private func startTimer() {
        stopTimer()
        sessionTimer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.tick() }
        }
    }

    private func stopTimer() {
        sessionTimer?.invalidate()
        sessionTimer = nil
    }

    private func tick() {
        guard let start = sessionStart else { return }
        let elapsed = Int(Date().timeIntervalSince(start))
        metrics.elapsed = String(format: "%02d:%02d:%02d", elapsed / 3600, (elapsed % 3600) / 60, elapsed % 60)
    }

    // MARK: Alerts

    private func matchedAlerts(in text: String) -> [String] {
        let keywords = alerts.split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() }
            .filter { !$0.isEmpty }
        guard !keywords.isEmpty else { return [] }
        let lower = text.lowercased()
        return keywords.filter { lower.contains($0) }
    }

    private func notifyAlert(_ matched: [String]) {
        guard !matched.isEmpty else { return }
        let now = Date().timeIntervalSince1970
        guard now - lastAlertAt > 2 else { return }
        lastAlertAt = now
        toast("Alerta: \(matched.joined(separator: ", "))", kind: "alert")
        if alertSound { NSSound.beep() }
    }

    // MARK: AI (Python bridge)

    private func resolveBridge() -> PythonBridge? {
        if let bridge { return bridge }
        bridge = PythonBridge.locate()
        return bridge
    }

    private func transcriptForAI() -> String {
        entries.map { "[\($0.timestamp)] \($0.text)" }.joined(separator: "\n")
    }

    /// Every run starts a Python process that loads Qwen, so the automatic
    /// refresh waits for 5 new phrases and at least 45 s since the last one,
    /// and never overlaps with a run in progress.
    private func maybeAutoSummarize() {
        guard autoSummary, !isSummarizing, phrasesSinceSummary >= 5 else { return }
        if let last = lastSummaryAt, Date().timeIntervalSince(last) < 45 { return }
        refreshSummary()
    }

    func refreshSummary(force: Bool = false) {
        guard !isSummarizing else { return }
        let text = transcriptForAI()
        guard !text.isEmpty else {
            if force { toast("Todavía no hay transcripción para resumir.") }
            return
        }
        guard let bridge = resolveBridge() else {
            summaryError = PythonBridge.missingMessage
            return
        }
        isSummarizing = true
        summaryError = nil
        phrasesSinceSummary = 0
        let lang = summaryLang
        Task {
            do {
                let result = try await bridge.run(["summary", "--lang", lang], input: text)
                self.liveSummary = result
                self.summaryUpdatedAt = Self.clock()
            } catch {
                self.summaryError = error.localizedDescription
            }
            self.isSummarizing = false
            self.lastSummaryAt = Date()
        }
    }

    var canAsk: Bool {
        !qaQuestion.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && !entries.isEmpty
    }

    func askQuestion() {
        let question = qaQuestion.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !question.isEmpty else { return }
        let text = transcriptForAI()
        guard !text.isEmpty else {
            toast("Todavía no hay transcripción sobre la que preguntar.")
            return
        }
        guard let bridge = resolveBridge() else {
            toast(PythonBridge.missingMessage, kind: "error")
            return
        }
        qaQuestion = ""
        let item = QAItem(question: question)
        qaItems.append(item)
        qaRevision += 1
        let lang = summaryLang
        Task {
            let answer: String
            let failed: Bool
            do {
                // `--question=` keeps questions that start with "-" intact.
                answer = try await bridge.run(["ask", "--lang", lang, "--question=\(question)"], input: text)
                failed = false
            } catch {
                answer = error.localizedDescription
                failed = true
            }
            if let index = self.qaItems.firstIndex(where: { $0.id == item.id }) {
                self.qaItems[index].answer = answer
                self.qaItems[index].failed = failed
                self.qaRevision += 1
            }
        }
    }

    // MARK: Actions

    func saveMeetingTxt() {
        guard !entries.isEmpty else {
            toast("Todavía no hay transcripción para guardar.")
            return
        }
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.plainText]
        panel.nameFieldStringValue = "reunion_\(Self.fileStamp()).txt"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        let rule = String(repeating: "=", count: 60)
        var lines = [
            rule,
            "VOICE LOGGER MAC - TRANSCRIPCIÓN DE REUNIÓN",
            rule,
            "Duración de la sesión: \(metrics.elapsed)",
            "Modelo: \(modelName) | Idioma: \(language.uppercased())",
            "Bloques: \(entries.count) | Palabras: \(wordCount)",
            rule,
            "",
        ]
        for entry in entries {
            lines.append("[\(entry.timestamp)] [\(entry.lang.uppercased())] \(entry.text)")
            lines.append("")
        }
        do {
            try lines.joined(separator: "\n").write(to: url, atomically: true, encoding: .utf8)
            toast("Guardado: \(url.lastPathComponent)")
        } catch {
            toast("No se pudo guardar: \(error.localizedDescription)", kind: "error")
        }
    }

    func openLogsFolder() {
        NSWorkspace.shared.open(logger.logsDirectory)
    }

    func copyAll() {
        guard !entries.isEmpty else {
            toast("Todavía no hay transcripción para copiar.")
            return
        }
        let text = entries.map { "[\($0.timestamp)] [\($0.lang.uppercased())] \($0.text)" }.joined(separator: "\n")
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
        toast("Copiados \(entries.count) bloques")
    }

    func copySummary() {
        guard !liveSummary.isEmpty else { return }
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(liveSummary, forType: .string)
        toast("Resumen copiado")
    }

    func clearDisplay() {
        entries.removeAll()
        transcriptRevision += 1
        lastEntryAt = nil
        liveSummary = ""
        summaryUpdatedAt = nil
        summaryError = nil
        phrasesSinceSummary = 0
        qaItems.removeAll()
        qaRevision += 1
    }

    func changeFontSize(_ delta: Int) {
        transcriptFontSize = max(12, min(28, transcriptFontSize + delta))
    }

    // MARK: Derived values for the UI

    var filteredEntries: [TranscriptEntry] {
        let query = searchText.trimmingCharacters(in: .whitespacesAndNewlines)
        if query.isEmpty { return entries }
        return entries.filter {
            $0.text.range(of: query, options: [.caseInsensitive, .diacriticInsensitive]) != nil
                || $0.timestamp.contains(query)
        }
    }

    var wordCount: Int {
        entries.reduce(0) { $0 + $1.text.split(whereSeparator: \.isWhitespace).count }
    }

    var captionText: String {
        let recent = entries.suffix(2).map(\.text).joined(separator: " ")
        return recent.isEmpty ? "Esperando audio…" : recent
    }

    var modelSummary: String {
        let label = WhisperKitTranscriber.modelChoices.first(where: { $0.id == modelName })?.label ?? modelName
        let short = (label.components(separatedBy: " (").first ?? label)
            .components(separatedBy: " —").first ?? label
        return "\(short) · \(language == "auto" ? "AUTO" : language.uppercased())"
    }

    var sensitivityValue: Double {
        get { Double(sensitivity) }
        set { sensitivity = Int(newValue) }
    }

    var sensitivityLabel: String {
        sensitivity < 25 ? "Muy alta" : sensitivity < 50 ? "Recomendada" : sensitivity < 85 ? "Media" : "Baja"
    }

    var statusLabel: String {
        if isStarting { return "Cargando modelo…" }
        if status.hasPrefix("Error") || status.hasPrefix("No se pudo") { return "Error" }
        guard isListening else { return "Detenido" }
        if isPaused { return "Pausado" }
        if status.hasPrefix("Cargando") { return "Cargando modelo…" }
        if status.hasPrefix("Transcribiendo") { return "Transcribiendo" }
        return "Escuchando"
    }

    var statusColor: Color {
        switch statusLabel {
        case "Escuchando": return Theme.brand
        case "Transcribiendo": return Theme.qa
        case "Pausado": return .yellow
        case "Cargando modelo…": return Theme.ai
        case "Error": return .red
        default: return .gray
        }
    }

    private static func clock() -> String {
        let f = DateFormatter()
        f.dateFormat = "HH:mm"
        return f.string(from: Date())
    }

    private static func fileStamp() -> String {
        let f = DateFormatter()
        f.dateFormat = "yyyy-MM-dd_HH-mm"
        return f.string(from: Date())
    }

    // MARK: Persistence

    func save() {
        guard !isLoading else { return }
        let d = UserDefaults.standard
        d.set(modelName, forKey: "model")
        d.set(language, forKey: "language")
        d.set(sensitivity, forKey: "sensitivity")
        d.set(alerts, forKey: "alerts")
        d.set(alertSound, forKey: "alert_sound")
        d.set(autoscroll, forKey: "autoscroll")
        d.set(summaryLang, forKey: "summary_lang")
        d.set(autoSummary, forKey: "auto_summary")
        d.set(transcriptFontSize, forKey: "font_size")
        d.set(alwaysOnTop, forKey: "always_on_top")
        d.set(dimWhenIdle, forKey: "dim_when_idle")
    }

    private func load() {
        isLoading = true
        defer { isLoading = false }
        let d = UserDefaults.standard
        modelName = d.string(forKey: "model") ?? WhisperKitTranscriber.defaultModel
        language = d.string(forKey: "language") ?? "en"
        sensitivity = d.object(forKey: "sensitivity") as? Int ?? 35
        alerts = d.string(forKey: "alerts") ?? ""
        alertSound = d.object(forKey: "alert_sound") as? Bool ?? true
        autoscroll = d.object(forKey: "autoscroll") as? Bool ?? true
        summaryLang = d.string(forKey: "summary_lang") ?? "es"
        autoSummary = d.object(forKey: "auto_summary") as? Bool ?? true
        transcriptFontSize = max(12, min(28, d.object(forKey: "font_size") as? Int ?? 17))
        alwaysOnTop = d.object(forKey: "always_on_top") as? Bool ?? true
        dimWhenIdle = d.object(forKey: "dim_when_idle") as? Bool ?? true
    }
}

// MARK: - Theme & building blocks

enum Theme {
    static let background = Color(red: 0.055, green: 0.055, blue: 0.082)
    static let card = Color(red: 0.086, green: 0.086, blue: 0.122)
    static let cardBorder = Color.white.opacity(0.07)
    static let row = Color.white.opacity(0.035)
    static let brand = Color(red: 0.06, green: 0.73, blue: 0.51)
    static let ai = Color(red: 0.66, green: 0.45, blue: 0.98)
    static let qa = Color(red: 0.22, green: 0.74, blue: 0.97)

    static func langColor(_ lang: String) -> Color {
        switch lang.uppercased() {
        case "EN": return Color(red: 0.22, green: 0.74, blue: 0.97)
        case "ES": return brand
        case "PT": return Color(red: 0.75, green: 0.52, blue: 0.99)
        default: return .gray
        }
    }
}

/// Card with a header row (title + accessory controls) and a content area.
struct Panel<Accessory: View, Content: View>: View {
    let title: String
    let systemImage: String
    let tint: Color
    let accessory: Accessory
    let content: Content

    init(
        title: String,
        systemImage: String,
        tint: Color,
        @ViewBuilder accessory: () -> Accessory,
        @ViewBuilder content: () -> Content
    ) {
        self.title = title
        self.systemImage = systemImage
        self.tint = tint
        self.accessory = accessory()
        self.content = content()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 8) {
                Label(title, systemImage: systemImage)
                    .font(.headline)
                    .foregroundStyle(tint)
                Spacer(minLength: 8)
                accessory
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 9)
            Divider().opacity(0.5)
            content
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        }
        .background(Theme.card, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 12, style: .continuous).strokeBorder(Theme.cardBorder)
        }
        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
    }
}

struct IconButton: View {
    let systemImage: String
    let help: String
    var active = false
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Image(systemName: systemImage)
                .font(.system(size: 14, weight: .medium))
                .foregroundStyle(active ? Theme.brand : Color.secondary)
                .frame(width: 28, height: 26)
                .background(active ? Theme.brand.opacity(0.15) : Color.clear, in: RoundedRectangle(cornerRadius: 6))
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help(help)
    }
}

struct StatusPill: View {
    let label: String
    let color: Color

    var body: some View {
        HStack(spacing: 6) {
            Circle().fill(color).frame(width: 8, height: 8)
            Text(label)
                .font(.callout.weight(.semibold))
                .lineLimit(1)
        }
        .foregroundStyle(color)
        .padding(.horizontal, 10)
        .padding(.vertical, 4)
        .background(color.opacity(0.14), in: Capsule())
    }
}

struct LevelMeter: View {
    let level: Float
    static let segments = 12

    var body: some View {
        let lit = Int((Double(min(max(level, 0), 1)) * Double(Self.segments)).rounded(.up))
        HStack(spacing: 2) {
            ForEach(0..<Self.segments, id: \.self) { index in
                RoundedRectangle(cornerRadius: 1.5)
                    .fill(index < lit ? color(for: index) : Color.white.opacity(0.08))
            }
        }
        .animation(.easeOut(duration: 0.12), value: lit)
    }

    private func color(for index: Int) -> Color {
        let fraction = Double(index) / Double(Self.segments)
        if fraction > 0.8 { return .red }
        if fraction > 0.55 { return .orange }
        return Theme.brand
    }
}

/// Clock + level meter. Observes `LiveMetrics` only.
struct SessionMeter: View {
    @ObservedObject var metrics: LiveMetrics

    var body: some View {
        HStack(spacing: 10) {
            Text(metrics.elapsed)
                .font(.system(.callout, design: .monospaced).weight(.semibold))
                .foregroundStyle(.secondary)
            LevelMeter(level: metrics.level)
                .frame(width: 84, height: 12)
        }
    }
}

struct EmptyHint: View {
    let systemImage: String
    let text: String

    var body: some View {
        VStack(spacing: 10) {
            Image(systemName: systemImage)
                .font(.system(size: 28))
                .foregroundStyle(.tertiary)
            Text(text)
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)
        }
        .padding(24)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

struct ToastView: View {
    let message: String
    let kind: String

    var body: some View {
        let color: Color = kind == "alert" ? .red : (kind == "error" ? .orange : Theme.brand)
        let icon = kind == "alert" ? "bell.fill" : (kind == "error" ? "exclamationmark.triangle.fill" : "checkmark.circle.fill")
        Label(message, systemImage: icon)
            .font(.callout.weight(.semibold))
            .foregroundStyle(.white)
            .lineLimit(2)
            .padding(.horizontal, 14)
            .padding(.vertical, 9)
            .background(color.opacity(0.92), in: Capsule())
            .shadow(color: .black.opacity(0.35), radius: 10, y: 4)
            .frame(maxWidth: 560)
    }
}

/// Renders headings / bullets / paragraphs from the model's Markdown.
struct MarkdownText: View {
    let markdown: String
    let fontSize: Int
    var accent: Color = Theme.ai

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            ForEach(Array(MarkdownBlock.parse(markdown).enumerated()), id: \.offset) { item in
                switch item.element {
                case .heading(let text):
                    Text(Self.inline(text))
                        .font(.system(size: CGFloat(fontSize) + 1, weight: .bold))
                        .foregroundStyle(accent)
                        .padding(.top, item.offset == 0 ? 0 : 6)
                case .bullet(let text):
                    HStack(alignment: .firstTextBaseline, spacing: 8) {
                        Text("•").foregroundStyle(accent)
                        Text(Self.inline(text))
                    }
                case .paragraph(let text):
                    Text(Self.inline(text))
                }
            }
        }
        .font(.system(size: CGFloat(fontSize)))
        .lineSpacing(3)
        .textSelection(.enabled)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    static func inline(_ text: String) -> AttributedString {
        (try? AttributedString(
            markdown: text,
            options: AttributedString.MarkdownParsingOptions(interpretedSyntax: .inlineOnlyPreservingWhitespace)
        )) ?? AttributedString(text)
    }
}

struct TranscriptRow: View {
    let entry: TranscriptEntry
    let fontSize: Int
    let query: String

    var body: some View {
        let accent = entry.alert ? Color.red : Theme.langColor(entry.lang)
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 8) {
                Text(entry.timestamp)
                    .font(.system(size: max(11, CGFloat(fontSize) - 5), design: .monospaced))
                    .foregroundStyle(.secondary)
                Text(entry.lang.uppercased())
                    .font(.system(size: max(10, CGFloat(fontSize) - 7), weight: .bold))
                    .foregroundStyle(accent)
                if entry.alert {
                    Text("ALERTA")
                        .font(.system(size: max(10, CGFloat(fontSize) - 7), weight: .heavy))
                        .foregroundStyle(.white)
                        .padding(.horizontal, 5)
                        .padding(.vertical, 1)
                        .background(Color.red, in: RoundedRectangle(cornerRadius: 3))
                }
            }
            Text(Self.highlighted(entry.text, query: query))
                .font(.system(size: CGFloat(fontSize)))
                .lineSpacing(3)
                .foregroundStyle(Color.primary.opacity(0.92))
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(.vertical, 9)
        .padding(.leading, 14)
        .padding(.trailing, 12)
        .background(alignment: .leading) {
            Rectangle().fill(accent).frame(width: 3)
        }
        .background(entry.alert ? Color.red.opacity(0.1) : Theme.row)
        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
    }

    static func highlighted(_ text: String, query: String) -> AttributedString {
        var attributed = AttributedString(text)
        let needle = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !needle.isEmpty else { return attributed }
        var searchRange = text.startIndex..<text.endIndex
        while let found = text.range(of: needle, options: [.caseInsensitive, .diacriticInsensitive], range: searchRange) {
            if let target = Range<AttributedString.Index>(found, in: attributed) {
                attributed[target].backgroundColor = Color.yellow.opacity(0.85)
                attributed[target].foregroundColor = Color.black
            }
            searchRange = found.upperBound..<text.endIndex
        }
        return attributed
    }
}

struct QARow: View {
    let item: QAItem
    let fontSize: Int

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Image(systemName: "person.fill")
                    .font(.system(size: 12))
                    .foregroundStyle(Theme.qa)
                Text(item.question)
                    .font(.system(size: CGFloat(fontSize), weight: .semibold))
                    .textSelection(.enabled)
            }
            Group {
                if let answer = item.answer {
                    if item.failed {
                        Label(answer, systemImage: "exclamationmark.triangle")
                            .font(.system(size: CGFloat(fontSize) - 2))
                            .foregroundStyle(Color.orange)
                    } else {
                        MarkdownText(markdown: answer, fontSize: fontSize, accent: Theme.qa)
                    }
                } else {
                    HStack(spacing: 8) {
                        ProgressView().controlSize(.small)
                        Text("Pensando…").foregroundStyle(.secondary)
                    }
                }
            }
            .padding(.leading, 20)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Theme.row, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
    }
}

/// Reports the NSWindow hosting a SwiftUI view.
final class WindowTrackingView: NSView {
    var onWindow: (@MainActor (NSWindow?) -> Void)?

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        onWindow?(window)
    }
}

struct WindowReader: NSViewRepresentable {
    let onWindow: @MainActor (NSWindow?) -> Void

    func makeNSView(context: Context) -> WindowTrackingView {
        let view = WindowTrackingView()
        view.onWindow = onWindow
        return view
    }

    func updateNSView(_ nsView: WindowTrackingView, context: Context) {
        nsView.onWindow = onWindow
    }
}

/// Holds the main window to dim it / resize it for compact mode, touching
/// only that window (the old code dimmed every window, overlay included).
@MainActor
final class WindowBox {
    weak var window: NSWindow?
    private var expandedFrame: NSRect?

    func setDimmed(_ dimmed: Bool) {
        window?.animator().alphaValue = dimmed ? 0.5 : 1.0
    }

    func setCompact(_ compact: Bool) {
        guard let window else { return }
        if compact {
            expandedFrame = window.frame
            window.setContentSize(NSSize(width: 360, height: 150))
        } else if let frame = expandedFrame {
            window.setFrame(frame, display: true, animate: true)
            expandedFrame = nil
        }
    }
}

// MARK: - Main window

struct FullView: View {
    @ObservedObject var state: AppState
    @Environment(\.openWindow) private var openWindow
    @Environment(\.dismissWindow) private var dismissWindow
    @State private var showSettings = false
    @FocusState private var searchFocused: Bool

    var body: some View {
        VStack(spacing: 10) {
            topBar
            HSplitView {
                transcriptPanel
                    .padding(.trailing, 5)
                    .frame(minWidth: 340, idealWidth: 640, maxWidth: .infinity, maxHeight: .infinity)
                VSplitView {
                    summaryPanel
                        .padding(.bottom, 5)
                        .frame(minHeight: 150, idealHeight: 320, maxHeight: .infinity)
                    questionsPanel
                        .padding(.top, 5)
                        .frame(minHeight: 190, idealHeight: 320, maxHeight: .infinity)
                }
                .padding(.leading, 5)
                .frame(minWidth: 300, idealWidth: 440, maxWidth: .infinity, maxHeight: .infinity)
            }
            footer
        }
        .padding(12)
        .frame(minWidth: 780, minHeight: 500)
        .background(Theme.background)
        .overlay(alignment: .bottom) {
            if let message = state.toastMessage {
                ToastView(message: message, kind: state.toastKind)
                    .padding(.bottom, 36)
                    .transition(.move(edge: .bottom).combined(with: .opacity))
            }
        }
        .preferredColorScheme(.dark)
    }

    // One compact row: status · clock · level | model · start/pause | tools.
    private var topBar: some View {
        HStack(spacing: 10) {
            StatusPill(label: state.statusLabel, color: state.statusColor)
                .help(state.status)
            SessionMeter(metrics: state.metrics)

            Spacer(minLength: 12)

            Button { showSettings = true } label: {
                Text(state.modelSummary)
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .padding(.horizontal, 9)
                    .padding(.vertical, 4)
                    .background(Color.white.opacity(0.06), in: Capsule())
            }
            .buttonStyle(.plain)
            .help("Modelo e idioma de transcripción (clic para cambiar)")

            Button {
                Task { await state.toggleListening() }
            } label: {
                Label(
                    state.isListening ? "Detener" : "Iniciar",
                    systemImage: state.isListening ? "stop.fill" : "record.circle"
                )
                .frame(minWidth: 78)
            }
            .buttonStyle(.borderedProminent)
            .tint(state.isListening ? Color.red : Theme.brand)
            .disabled(state.isStarting)
            .help("Iniciar / detener la escucha (⌘↩)")

            Button { state.togglePause() } label: {
                Image(systemName: state.isPaused ? "play.fill" : "pause.fill")
            }
            .disabled(!state.isListening)
            .help(state.isPaused ? "Reanudar (⌘⇧P)" : "Pausar (⌘⇧P)")

            Divider().frame(height: 18)

            HStack(spacing: 2) {
                IconButton(systemImage: "captions.bubble", help: "Subtítulos flotantes", active: state.overlayVisible) {
                    toggleOverlay()
                }
                IconButton(systemImage: "pip.enter", help: "Modo compacto") {
                    state.compact = true
                }
                moreMenu
                IconButton(systemImage: "gearshape", help: "Ajustes") {
                    showSettings.toggle()
                }
                .popover(isPresented: $showSettings, arrowEdge: .bottom) {
                    SettingsPopover(state: state)
                }
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 7)
        .background(Theme.card, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 12, style: .continuous).strokeBorder(Theme.cardBorder)
        }
    }

    private var moreMenu: some View {
        Menu {
            Button("Guardar transcripción…", systemImage: "square.and.arrow.down") { state.saveMeetingTxt() }
            Button("Copiar transcripción", systemImage: "doc.on.doc") { state.copyAll() }
            Button("Abrir carpeta de logs", systemImage: "folder") { state.openLogsFolder() }
            Divider()
            Toggle("Siempre encima", isOn: $state.alwaysOnTop)
            Toggle("Atenuar al quitar el ratón", isOn: $state.dimWhenIdle)
            Divider()
            Button("Limpiar transcripción", systemImage: "trash", role: .destructive) { state.clearDisplay() }
        } label: {
            Image(systemName: "ellipsis.circle")
                .font(.system(size: 14, weight: .medium))
                .foregroundStyle(.secondary)
        }
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
        .fixedSize()
        .frame(width: 28, height: 26)
        .help("Más acciones")
    }

    private func toggleOverlay() {
        if state.overlayVisible {
            dismissWindow(id: "overlay")
        } else {
            openWindow(id: "overlay")
        }
        state.overlayVisible.toggle()
    }

    // MARK: Transcript

    private var transcriptPanel: some View {
        Panel(title: "Transcripción", systemImage: "waveform", tint: Theme.brand) {
            HStack(spacing: 4) {
                searchField
                IconButton(systemImage: "textformat.size.smaller", help: "Reducir letra (⌘-)") {
                    state.changeFontSize(-1)
                }
                IconButton(systemImage: "textformat.size.larger", help: "Aumentar letra (⌘+)") {
                    state.changeFontSize(1)
                }
            }
        } content: {
            transcriptList
        }
    }

    private var searchField: some View {
        HStack(spacing: 5) {
            Image(systemName: "magnifyingglass").foregroundStyle(.secondary)
            TextField("Filtrar (⌘F)", text: $state.searchText)
                .textFieldStyle(.plain)
                .focused($searchFocused)
            if !state.searchText.isEmpty {
                Button { state.searchText = "" } label: {
                    Image(systemName: "xmark.circle.fill")
                }
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
            }
        }
        .font(.callout)
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .frame(width: 190)
        .background(Color.white.opacity(0.06), in: Capsule())
        .background {
            // Invisible button that only carries the ⌘F shortcut.
            Button("") { searchFocused = true }
                .keyboardShortcut("f", modifiers: .command)
                .opacity(0)
                .allowsHitTesting(false)
        }
    }

    @ViewBuilder
    private var transcriptList: some View {
        let entries = state.filteredEntries
        if entries.isEmpty {
            EmptyHint(
                systemImage: state.searchText.isEmpty ? "waveform" : "magnifyingglass",
                text: state.searchText.isEmpty
                    ? "La transcripción aparecerá aquí.\nPulsa Iniciar (⌘↩) para capturar el audio del sistema."
                    : "Sin resultados para «\(state.searchText)»."
            )
        } else {
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 8) {
                        ForEach(entries) { entry in
                            TranscriptRow(entry: entry, fontSize: state.transcriptFontSize, query: state.searchText)
                                .id(entry.id)
                        }
                    }
                    .padding(12)
                }
                .onChange(of: state.transcriptRevision) { _, _ in
                    guard state.autoscroll, let last = state.filteredEntries.last else { return }
                    withAnimation(.easeOut(duration: 0.25)) {
                        proxy.scrollTo(last.id, anchor: .bottom)
                    }
                }
                .onAppear {
                    if let last = entries.last { proxy.scrollTo(last.id, anchor: .bottom) }
                }
            }
        }
    }

    // MARK: AI summary

    private var summaryPanel: some View {
        Panel(title: "Resumen IA", systemImage: "sparkles", tint: Theme.ai) {
            HStack(spacing: 6) {
                if state.isSummarizing {
                    ProgressView().controlSize(.small)
                    Text("Generando…").font(.caption).foregroundStyle(.secondary)
                } else if let updated = state.summaryUpdatedAt {
                    Text("Actualizado \(updated)").font(.caption).foregroundStyle(.secondary)
                }
                IconButton(systemImage: "doc.on.doc", help: "Copiar resumen") {
                    state.copySummary()
                }
                .disabled(state.liveSummary.isEmpty)
                IconButton(systemImage: "arrow.clockwise", help: "Actualizar ahora (⌘R)") {
                    state.refreshSummary(force: true)
                }
                .disabled(state.isSummarizing)
            }
        } content: {
            ScrollView {
                Group {
                    if let error = state.summaryError {
                        Label(error, systemImage: "exclamationmark.triangle")
                            .foregroundStyle(Color.orange)
                    } else if state.liveSummary.isEmpty {
                        Text(
                            state.autoSummary
                                ? "El resumen se genera solo cada pocos bloques de transcripción, o pulsa ↻."
                                : "Pulsa ↻ para generar el resumen de lo transcrito."
                        )
                        .foregroundStyle(.secondary)
                    } else {
                        MarkdownText(markdown: state.liveSummary, fontSize: state.transcriptFontSize)
                    }
                }
                .padding(14)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }

    // MARK: Questions

    private var questionsPanel: some View {
        Panel(title: "Preguntas", systemImage: "questionmark.bubble", tint: Theme.qa) {
            if !state.qaItems.isEmpty {
                Button("Borrar") {
                    state.qaItems.removeAll()
                }
                .buttonStyle(.plain)
                .font(.caption)
                .foregroundStyle(.secondary)
            }
        } content: {
            VStack(spacing: 0) {
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 10) {
                            if state.qaItems.isEmpty {
                                Text("Pregunta lo que quieras sobre la reunión: «¿qué se decidió?», «¿qué tareas tengo?»…")
                                    .foregroundStyle(.secondary)
                            }
                            ForEach(state.qaItems) { item in
                                QARow(item: item, fontSize: state.transcriptFontSize)
                                    .id(item.id)
                            }
                        }
                        .padding(14)
                    }
                    .onChange(of: state.qaRevision) { _, _ in
                        guard let last = state.qaItems.last else { return }
                        withAnimation(.easeOut(duration: 0.25)) {
                            proxy.scrollTo(last.id, anchor: .bottom)
                        }
                    }
                }
                HStack(spacing: 8) {
                    TextField("Pregunta algo sobre la reunión…", text: $state.qaQuestion)
                        .textFieldStyle(.plain)
                        .font(.system(size: 14))
                        .onSubmit { state.askQuestion() }
                    Button { state.askQuestion() } label: {
                        Image(systemName: "paperplane.fill").font(.system(size: 14))
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(state.canAsk ? Theme.qa : Color.secondary)
                    .disabled(!state.canAsk)
                    .help("Preguntar (↩)")
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 9)
                .background(Color.white.opacity(0.06), in: RoundedRectangle(cornerRadius: 8, style: .continuous))
                .padding(10)
            }
        }
    }

    private var footer: some View {
        HStack(spacing: 6) {
            Text("\(state.entries.count) bloques · \(state.wordCount) palabras")
            Spacer()
            Button { state.openLogsFolder() } label: {
                Label("Logs locales", systemImage: "folder")
            }
            .buttonStyle(.plain)
        }
        .font(.caption)
        .foregroundStyle(.secondary)
        .padding(.horizontal, 4)
    }
}

struct SettingsPopover: View {
    @ObservedObject var state: AppState

    var body: some View {
        Form {
            Section("Transcripción") {
                Picker("Modelo", selection: $state.modelName) {
                    ForEach(WhisperKitTranscriber.modelChoices, id: \.id) { choice in
                        Text(choice.label).tag(choice.id)
                    }
                }
                Picker("Idioma", selection: $state.language) {
                    Text("Inglés").tag("en")
                    Text("Español").tag("es")
                    Text("Portugués").tag("pt")
                    Text("Auto-detectar").tag("auto")
                }
                LabeledContent("Sensibilidad de voz") {
                    HStack {
                        Slider(value: $state.sensitivityValue, in: 10...150)
                        Text(state.sensitivityLabel)
                            .foregroundStyle(.secondary)
                            .frame(width: 86, alignment: .trailing)
                    }
                }
            }
            Section("Resumen y preguntas (IA local)") {
                Picker("Idioma de respuesta", selection: $state.summaryLang) {
                    Text("Español").tag("es")
                    Text("Inglés").tag("en")
                }
                Toggle("Actualizar el resumen automáticamente", isOn: $state.autoSummary)
            }
            Section("Alertas") {
                TextField("Palabras clave", text: $state.alerts, prompt: Text("julian, deadline, urgente"))
                Toggle("Sonido al detectar", isOn: $state.alertSound)
            }
            Section("Lectura") {
                Stepper("Tamaño de letra: \(state.transcriptFontSize) pt", value: $state.transcriptFontSize, in: 12...28)
                Toggle("Autodesplazar al final", isOn: $state.autoscroll)
            }
        }
        .formStyle(.grouped)
        .frame(width: 420, height: 500)
    }
}

// MARK: - Compact, overlay, menu bar

struct CompactView: View {
    @ObservedObject var state: AppState

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Circle().fill(state.statusColor).frame(width: 8, height: 8)
                Text(state.statusLabel.uppercased())
                    .font(.caption2.bold())
                    .foregroundStyle(.secondary)
                Spacer()
                Button { state.compact = false } label: {
                    Image(systemName: "arrow.up.left.and.arrow.down.right")
                }
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
                .help("Expandir")
            }
            Text(state.captionText)
                .font(.system(size: 14))
                .lineLimit(3)
                .truncationMode(.head)
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottomLeading)
            HStack(spacing: 10) {
                Button {
                    Task { await state.toggleListening() }
                } label: {
                    Image(systemName: state.isListening ? "stop.fill" : "record.circle")
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
                .tint(state.isListening ? Color.red : Theme.brand)
                .disabled(state.isStarting)
                SessionMeter(metrics: state.metrics)
            }
        }
        .padding(12)
        .frame(width: 360, height: 150)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
        .preferredColorScheme(.dark)
    }
}

struct OverlayView: View {
    @ObservedObject var state = AppState.shared
    @Environment(\.dismissWindow) private var dismissWindow

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                Circle().fill(state.statusColor).frame(width: 8, height: 8)
                Text("SUBTÍTULOS").font(.caption2.bold()).foregroundStyle(.secondary)
                Text("arrastra para mover").font(.caption2).foregroundStyle(.tertiary)
                Spacer()
                Button { dismissWindow(id: "overlay") } label: {
                    Image(systemName: "xmark")
                }
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
                .help("Ocultar subtítulos")
            }
            Text(state.captionText)
                .font(.system(size: CGFloat(state.transcriptFontSize + 5), weight: .medium))
                .lineLimit(3)
                .truncationMode(.head)
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottomLeading)
        }
        .padding(14)
        .frame(width: 640, height: 170)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14))
        .background(WindowReader { window in
            window?.isMovableByWindowBackground = true
            window?.level = .floating
            window?.collectionBehavior.insert([.canJoinAllSpaces, .fullScreenAuxiliary])
        })
        .onAppear { state.overlayVisible = true }
        .onDisappear { state.overlayVisible = false }
        .preferredColorScheme(.dark)
    }
}

struct MenuBarContent: View {
    @ObservedObject var state = AppState.shared
    @Environment(\.openWindow) private var openWindow
    @Environment(\.dismissWindow) private var dismissWindow

    var body: some View {
        Button("Mostrar ventana") { NSApp.activate() }
        Button(state.isListening ? "Detener" : "Iniciar") {
            Task { await state.toggleListening() }
        }
        Button(state.overlayVisible ? "Ocultar subtítulos" : "Subtítulos") {
            if state.overlayVisible {
                dismissWindow(id: "overlay")
            } else {
                openWindow(id: "overlay")
            }
        }
        Divider()
        Button("Salir") { NSApp.terminate(nil) }
    }
}

struct ContentView: View {
    @StateObject private var state = AppState.shared
    @State private var windowBox = WindowBox()

    var body: some View {
        Group {
            if state.compact {
                CompactView(state: state)
            } else {
                FullView(state: state)
            }
        }
        .background(WindowReader { window in windowBox.window = window })
        .onAppear { state.applyOnTop() }
        .onHover { hovering in windowBox.setDimmed(state.dimWhenIdle && !hovering) }
        .onChange(of: state.dimWhenIdle) { _, enabled in
            if !enabled { windowBox.setDimmed(false) }
        }
        .onChange(of: state.compact) { _, compact in
            windowBox.setCompact(compact)
        }
    }
}

struct SessionCommands: Commands {
    var body: some Commands {
        CommandGroup(replacing: .saveItem) {
            Button("Guardar transcripción…") { AppState.shared.saveMeetingTxt() }
                .keyboardShortcut("s", modifiers: .command)
        }
        CommandMenu("Sesión") {
            Button("Iniciar / Detener") {
                Task { await AppState.shared.toggleListening() }
            }
            .keyboardShortcut(.return, modifiers: .command)
            Button("Pausar / Reanudar") { AppState.shared.togglePause() }
                .keyboardShortcut("p", modifiers: [.command, .shift])
            Divider()
            Button("Copiar transcripción") { AppState.shared.copyAll() }
                .keyboardShortcut("c", modifiers: [.command, .shift])
            Button("Actualizar resumen IA") { AppState.shared.refreshSummary(force: true) }
                .keyboardShortcut("r", modifiers: .command)
            Divider()
            Button("Aumentar letra") { AppState.shared.changeFontSize(1) }
                .keyboardShortcut("+", modifiers: .command)
            Button("Reducir letra") { AppState.shared.changeFontSize(-1) }
                .keyboardShortcut("-", modifiers: .command)
        }
    }
}

@main
struct VoiceLoggerMacNativeApp: App {
    var body: some Scene {
        WindowGroup("VoiceLogger Mac") {
            ContentView()
        }
        .windowResizability(.contentMinSize)
        .defaultSize(width: 1180, height: 780)
        .commands { SessionCommands() }

        Window("Subtítulos", id: "overlay") {
            OverlayView()
        }
        .windowStyle(.plain)
        .windowResizability(.contentSize)

        MenuBarExtra("VoiceLogger", systemImage: "waveform") {
            MenuBarContent()
        }
    }
}
