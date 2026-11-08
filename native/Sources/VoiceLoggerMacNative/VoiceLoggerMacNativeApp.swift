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

/// Qwen models for the summary / questions (MLX 4-bit, downloaded once to the
/// Hugging Face cache). Sizes are approximate download sizes.
enum QwenModels {
    static let `default` = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
    static let choices: [(label: String, id: String)] = [
        ("Qwen2.5 1.5B · ligero, ~0.9 GB (actual)", "mlx-community/Qwen2.5-1.5B-Instruct-4bit"),
        ("Qwen3 4B 2507 · mejor calidad, ~2.3 GB (bien con 16 GB)", "mlx-community/Qwen3-4B-Instruct-2507-4bit"),
        ("Qwen3.5 4B · el más nuevo, ~3 GB (experimental)", "mlx-community/Qwen3.5-4B-MLX-4bit"),
    ]
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
    @Published var isLoadingModel = false
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
    @Published var isWarmingAI = false
    /// Most important questions heard so far, extracted with the live summary.
    @Published var keyQuestions: [String] = []
    /// Transcript fragments saved in the meeting memory on disk.
    @Published var memorySegments = 0
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
    /// Language of AI answers (summary, key questions, Q&A). English for now.
    @Published var summaryLang = "en" { didSet { save() } }
    @Published var autoSummary = true { didSet { save() } }
    /// Seconds between live summary updates (counted from the end of the previous one).
    @Published var liveInterval = 30 { didSet { save() } }
    @Published var qwenModel = QwenModels.default {
        didSet {
            guard oldValue != qwenModel else { return }
            save()
            switchQwenModel()
        }
    }
    /// Interface size: 1.0 normal … 1.45 enorme.
    @Published var uiScale = 1.15 { didSet { save() } }
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
    private var qwen: QwenServer?
    private var sessionTimer: Timer?
    private var sessionStart: Date?
    private var lastAlertAt: TimeInterval = 0
    private var lastEntryAt: Date?
    /// Phrases waiting for (or in) Whisper. The live transcript comes first:
    /// Qwen updates only start when this is 0.
    private var pendingTranscriptions = 0
    private var lastSummaryAt: Date?
    /// Full notes from the last live update (fed back for incremental updates).
    private var liveMarkdown = ""
    /// Transcript already covered by `liveMarkdown`.
    private var summarizedText = ""
    private var hasUnsummarizedText = false
    /// Bumped by "Limpiar" so late AI results for the old transcript are ignored.
    private var aiGeneration = 0

    /// Transcript sent whole to Qwen while it fits in one pass (~3-4k tokens).
    static let singlePassChars = 12000
    /// Most recent part of the transcript sent with each question. Older parts
    /// come from the live notes and the meeting memory on disk, so the prompt
    /// stays the same size however long the meeting is.
    static let questionChars = 6000
    /// Size of each fragment saved to the meeting memory (~3-4 minutes of talk).
    static let segmentChars = 3000

    // Meeting memory on disk (Application Support/VoiceLoggerMac/memory/<session>.jsonl)
    private var sessionID: String?
    /// Exclusive end entry index of each closed fragment.
    private var segmentEnds: [Int] = []
    private var segmentsSent = 0
    private var isRemembering = false
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
            if autoSummary { warmUpAI() }
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

    /// Loads Whisper into memory as soon as the app opens, so "Iniciar" is
    /// instant (the model files are already cached on disk after the first run).
    func prewarm() {
        guard transcriber == nil, transcriberTask == nil else { return }
        Task {
            do {
                _ = try await self.ensureTranscriber()
            } catch {
                self.status = "Error cargando modelo: \(error.localizedDescription)"
                return
            }
            if !self.isListening, !self.isStarting, self.status.hasPrefix("Cargando") {
                self.status = "Detenido"
            }
        }
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
            isLoadingModel = true
        }
        do {
            let loaded = try await task.value
            if transcriberTaskModel == model {
                transcriberTask = nil
                transcriberTaskModel = nil
                isLoadingModel = false
            }
            if modelName == model { transcriber = loaded }
            return loaded
        } catch {
            if transcriberTaskModel == model {
                transcriberTask = nil
                transcriberTaskModel = nil
                isLoadingModel = false
            }
            throw error
        }
    }

    private func handlePhrase(_ audio: [Float], timestamp ts: String) async {
        pendingTranscriptions += 1
        defer { pendingTranscriptions -= 1 }
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
        hasUnsummarizedText = true
        closeSegments()
        rememberPendingSegments()
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
        // Also checked every second so the last phrases get summarized even
        // when nobody speaks for a while, and failed memory writes are retried.
        rememberPendingSegments()
        maybeAutoSummarize()
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

    // MARK: AI (Qwen, persistent process)

    private func resolveBridge() -> PythonBridge? {
        if let bridge { return bridge }
        bridge = PythonBridge.locate()
        return bridge
    }

    private func resolveServer() -> QwenServer? {
        if let qwen { return qwen }
        guard let bridge = resolveBridge() else { return nil }
        let log = logger.logsDirectory.deletingLastPathComponent().appendingPathComponent("qwen_server.log")
        let server = QwenServer(bridge: bridge, logURL: log, model: qwenModel)
        qwen = server
        return server
    }

    /// Restarts the Qwen process with the selected model (it loads on the next
    /// request, or right away if a session is running).
    private func switchQwenModel() {
        guard !isLoading, let qwen else { return }
        let model = qwenModel
        Task {
            await qwen.setModel(model)
            if self.isListening, self.autoSummary { self.warmUpAI() }
        }
        toast("Modelo de IA: se carga en la próxima respuesta (la primera vez se descarga).")
    }

    private func transcriptForAI() -> String {
        entries.map { "[\($0.timestamp)] \($0.text)" }.joined(separator: "\n")
    }

    private func warmUpAI() {
        guard !isWarmingAI, let server = resolveServer() else { return }
        isWarmingAI = true
        Task {
            do {
                try await server.warmup()
            } catch {
                self.summaryError = error.localizedDescription
            }
            self.isWarmingAI = false
        }
    }

    /// Live mode: an update runs when there is new text and `liveInterval`
    /// seconds have passed since the previous one finished. Never overlaps,
    /// so Qwen is busy at most a fraction of the time.
    private func maybeAutoSummarize() {
        guard autoSummary, !isSummarizing, hasUnsummarizedText else { return }
        // Don't start Qwen while Whisper has phrases to transcribe: both share
        // the chip and the live transcript on the left must not lag.
        guard pendingTranscriptions == 0 else { return }
        // A question waiting for its answer goes first: Qwen handles one
        // request at a time and the user is looking at the screen.
        guard !qaItems.contains(where: { $0.answer == nil }) else { return }
        if liveMarkdown.isEmpty, entries.reduce(0, { $0 + $1.text.count }) < 150 { return }
        if let last = lastSummaryAt, Date().timeIntervalSince(last) < Double(liveInterval) { return }
        refreshSummary()
    }

    func refreshSummary(force: Bool = false) {
        guard !isSummarizing else { return }
        let full = transcriptForAI()
        guard !full.isEmpty else {
            if force { toast("Todavía no hay transcripción para resumir.") }
            return
        }
        guard let server = resolveServer() else {
            summaryError = PythonBridge.missingMessage
            return
        }
        var newText = full
        if !summarizedText.isEmpty, full.hasPrefix(summarizedText) {
            newText = String(full.dropFirst(summarizedText.count))
        }
        if !force, !liveMarkdown.isEmpty, newText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            hasUnsummarizedText = false
            return
        }
        // Short meetings: notes rebuilt from the whole transcript (best quality).
        // Long ones: previous notes + only the new part, so every update costs
        // about the same no matter how long the meeting is.
        let rebuild = full.count <= Self.singlePassChars || liveMarkdown.isEmpty
        let fullText = rebuild ? String(full.suffix(Self.singlePassChars)) : ""
        let previous = rebuild ? "" : liveMarkdown
        let delta = rebuild ? "" : String(newText.suffix(Self.singlePassChars))
        let lang = summaryLang
        let generation = aiGeneration

        isSummarizing = true
        summaryError = nil
        hasUnsummarizedText = false
        Task {
            let outcome: Result<LiveInsights, Error>
            do {
                outcome = .success(try await server.liveInsights(
                    text: fullText, newText: delta, previous: previous, lang: lang
                ))
            } catch {
                outcome = .failure(error)
            }
            self.isSummarizing = false
            self.lastSummaryAt = Date()
            guard generation == self.aiGeneration else { return }
            switch outcome {
            case .success(let insights):
                self.liveMarkdown = insights.markdown
                self.liveSummary = insights.summary
                self.keyQuestions = insights.questions
                self.summarizedText = full
                self.summaryUpdatedAt = Self.clock()
            case .failure(let error):
                self.summaryError = error.localizedDescription
                self.hasUnsummarizedText = true
            }
        }
    }

    private var memoryDirectory: URL {
        logger.logsDirectory.deletingLastPathComponent().appendingPathComponent("memory", isDirectory: true)
    }

    private func currentSession() -> String {
        if let sessionID { return sessionID }
        let f = DateFormatter()
        f.dateFormat = "yyyy-MM-dd_HH-mm-ss"
        let id = f.string(from: Date())
        sessionID = id
        return id
    }

    /// Groups finished entries into fragments of ~`segmentChars`. The last
    /// entry is never included: it can still grow when phrases are merged, and
    /// a saved fragment must not change.
    private func closeSegments() {
        var size = 0
        var index = segmentEnds.last ?? 0
        while index < entries.count - 1 {
            size += entries[index].text.count + 1
            index += 1
            if size >= Self.segmentChars {
                segmentEnds.append(index)
                size = 0
            }
        }
    }

    /// Sends closed fragments to the meeting memory on disk, one at a time.
    private func rememberPendingSegments() {
        guard !isRemembering, segmentsSent < segmentEnds.count, let server = resolveServer() else { return }
        let index = segmentsSent
        let start = index == 0 ? 0 : segmentEnds[index - 1]
        let end = segmentEnds[index]
        guard start < end, end <= entries.count else { return }
        let slice = entries[start..<end]
        let text = slice.map { "[\($0.timestamp)] \($0.text)" }.joined(separator: "\n")
        let firstStamp = slice.first?.timestamp ?? ""
        let lastStamp = slice.last?.timestamp ?? ""
        let session = currentSession()
        let directory = memoryDirectory.path
        let generation = aiGeneration
        isRemembering = true
        Task {
            var stored = false
            do {
                try await server.remember(
                    session: session, memoryDir: directory, index: index,
                    start: firstStamp, end: lastStamp, text: text
                )
                stored = true
            } catch {
                // Retried on the next tick.
            }
            self.isRemembering = false
            guard generation == self.aiGeneration else { return }
            if stored {
                self.segmentsSent = index + 1
                self.memorySegments = index + 1
                self.rememberPendingSegments()
            }
        }
    }

    /// Frees Qwen from RAM now (it reloads by itself on the next request).
    func unloadAI() {
        guard let qwen else {
            toast("La IA no está cargada.")
            return
        }
        Task {
            let freed = await qwen.unload()
            self.toast(freed ? "Memoria de la IA liberada." : "La IA no está cargada.")
        }
    }

    var canAsk: Bool {
        !qaQuestion.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && !entries.isEmpty
    }

    func askQuestion() {
        let question = qaQuestion.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !question.isEmpty else { return }
        if ask(question) { qaQuestion = "" }
    }

    /// Asks Qwen about the meeting (typed question or a detected key question).
    /// True while that question is waiting for Qwen's answer.
    func isAnswering(_ question: String) -> Bool {
        qaItems.contains { $0.question == question && $0.answer == nil }
    }

    @discardableResult
    func ask(_ question: String) -> Bool {
        // Double clicks would queue the same heavy request twice.
        guard !isAnswering(question) else { return false }
        let text = transcriptForAI()
        guard !text.isEmpty else {
            toast("Todavía no hay transcripción sobre la que preguntar.")
            return false
        }
        guard let server = resolveServer() else {
            toast(PythonBridge.missingMessage, kind: "error")
            return false
        }
        let item = QAItem(question: question)
        qaItems.append(item)
        qaRevision += 1
        let lang = summaryLang
        let context = String(text.suffix(Self.questionChars))
        let notes = liveMarkdown
        let session = currentSession()
        let directory = memoryDirectory.path
        Task {
            let answer: String
            let failed: Bool
            do {
                answer = try await server.ask(
                    question: question, text: context, notes: notes,
                    session: session, memoryDir: directory, lang: lang
                )
                failed = false
            } catch {
                answer = error.localizedDescription
                failed = true
            }
            if let index = self.qaItems.firstIndex(where: { $0.id == item.id }) {
                self.qaItems[index].answer = answer.isEmpty ? "—" : answer
                self.qaItems[index].failed = failed
                self.qaRevision += 1
            }
        }
        return true
    }

    /// Opens qwen_context.txt (persona / context prepended to every prompt),
    /// creating it from the template the first time. Python re-reads it on
    /// every request, so saving is enough.
    func openAIContext() {
        guard let bridge = resolveBridge() else {
            toast(PythonBridge.missingMessage, kind: "error")
            return
        }
        let fm = FileManager.default
        if !fm.fileExists(atPath: bridge.contextFile.path) {
            if fm.fileExists(atPath: bridge.contextExample.path) {
                try? fm.copyItem(at: bridge.contextExample, to: bridge.contextFile)
            } else {
                try? "You are an expert in XXXXXXXX.\n".write(to: bridge.contextFile, atomically: true, encoding: .utf8)
            }
        }
        NSWorkspace.shared.open(bridge.contextFile)
        toast("Guarda el archivo: se aplica en la siguiente respuesta de la IA.")
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
        liveMarkdown = ""
        summarizedText = ""
        keyQuestions = []
        // A cleared transcript starts a new memory session (the old file stays on disk).
        sessionID = nil
        segmentEnds = []
        segmentsSent = 0
        memorySegments = 0
        hasUnsummarizedText = false
        lastSummaryAt = nil
        aiGeneration += 1
        summaryUpdatedAt = nil
        summaryError = nil
        qaItems.removeAll()
        qaRevision += 1
    }

    func changeFontSize(_ delta: Int) {
        transcriptFontSize = max(12, min(28, transcriptFontSize + delta))
    }

    func changeUIScale(_ step: Int) {
        let scales = Theme.uiScales
        let current = scales.enumerated().min(by: { abs($0.element - uiScale) < abs($1.element - uiScale) })?.offset ?? 1
        uiScale = scales[max(0, min(scales.count - 1, current + step))]
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
        if isStarting || isLoadingModel { return "Cargando modelo…" }
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
        d.set(summaryLang, forKey: "ai_lang")
        d.set(autoSummary, forKey: "auto_summary")
        d.set(liveInterval, forKey: "live_interval")
        d.set(qwenModel, forKey: "qwen_model")
        d.set(uiScale, forKey: "ui_scale")
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
        summaryLang = d.string(forKey: "ai_lang") ?? "en"
        autoSummary = d.object(forKey: "auto_summary") as? Bool ?? true
        liveInterval = d.object(forKey: "live_interval") as? Int ?? 30
        qwenModel = d.string(forKey: "qwen_model") ?? QwenModels.default
        uiScale = d.object(forKey: "ui_scale") as? Double ?? 1.15
        transcriptFontSize = max(12, min(28, d.object(forKey: "font_size") as? Int ?? 17))
        alwaysOnTop = d.object(forKey: "always_on_top") as? Bool ?? true
        dimWhenIdle = d.object(forKey: "dim_when_idle") as? Bool ?? true
    }
}

// MARK: - Interface scale

/// Global interface size (⚙ → Tamaño de la interfaz). Chrome fonts and
/// controls read it; reading text has its own size (A- / A+).
struct UIScaleKey: EnvironmentKey {
    static let defaultValue: CGFloat = 1
}

extension EnvironmentValues {
    var uiScale: CGFloat {
        get { self[UIScaleKey.self] }
        set { self[UIScaleKey.self] = newValue }
    }
}

struct ScaledFont: ViewModifier {
    @Environment(\.uiScale) private var scale
    let size: CGFloat
    let weight: Font.Weight
    let design: Font.Design

    func body(content: Content) -> some View {
        content.font(.system(size: (size * scale).rounded(), weight: weight, design: design))
    }
}

extension View {
    /// System font of `size` points at 100 %, scaled by the interface size.
    func scaledFont(_ size: CGFloat, weight: Font.Weight = .regular, design: Font.Design = .default) -> some View {
        modifier(ScaledFont(size: size, weight: weight, design: design))
    }
}

// MARK: - Theme & building blocks

enum Theme {
    /// Interface size steps (⌘⌥+ / ⌘⌥-): normal, grande, muy grande, enorme.
    static let uiScales: [Double] = [1.0, 1.15, 1.3, 1.45]
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
                    .scaledFont(14, weight: .bold)
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
    @Environment(\.uiScale) private var scale

    var body: some View {
        Button(action: action) {
            Image(systemName: systemImage)
                .scaledFont(14, weight: .medium)
                .foregroundStyle(active ? Theme.brand : Color.secondary)
                .frame(width: 28 * scale, height: 26 * scale)
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
                .scaledFont(12, weight: .semibold)
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
                .scaledFont(12, weight: .semibold, design: .monospaced)
                .foregroundStyle(.secondary)
            LevelMeter(level: metrics.level)
                .frame(width: 84, height: 12)
        }
    }
}

/// Pulsing "EN VIVO" tag shown while the summary updates itself.
struct LiveBadge: View {
    let interval: Int
    @State private var pulse = false

    var body: some View {
        HStack(spacing: 4) {
            Circle()
                .fill(Theme.ai)
                .frame(width: 6, height: 6)
                .opacity(pulse ? 0.25 : 1)
                .animation(.easeInOut(duration: 0.9).repeatForever(autoreverses: true), value: pulse)
            Text("EN VIVO · \(interval) s")
                .scaledFont(10, weight: .bold)
        }
        .foregroundStyle(Theme.ai)
        .padding(.horizontal, 7)
        .padding(.vertical, 3)
        .background(Theme.ai.opacity(0.14), in: Capsule())
        .onAppear { pulse = true }
        .help("El resumen y las preguntas clave se actualizan solos mientras escuchas")
    }
}

struct EmptyHint: View {
    let systemImage: String
    let text: String

    var body: some View {
        VStack(spacing: 10) {
            Image(systemName: systemImage)
                .scaledFont(28)
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
            .scaledFont(12, weight: .semibold)
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
                    .scaledFont(12)
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
    @Environment(\.uiScale) private var uiScale

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
        .frame(minWidth: 780 * uiScale, minHeight: 500)
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
                    .scaledFont(11, weight: .semibold)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.tail)
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
            Button("Editar contexto de la IA…", systemImage: "brain") { state.openAIContext() }
            Button("Liberar memoria de la IA", systemImage: "memorychip") { state.unloadAI() }
            Divider()
            Toggle("Siempre encima", isOn: $state.alwaysOnTop)
            Toggle("Atenuar al quitar el ratón", isOn: $state.dimWhenIdle)
            Divider()
            Button("Limpiar transcripción", systemImage: "trash", role: .destructive) { state.clearDisplay() }
        } label: {
            Image(systemName: "ellipsis.circle")
                .scaledFont(14, weight: .medium)
                .foregroundStyle(.secondary)
        }
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
        .fixedSize()
        .frame(width: 28 * uiScale, height: 26 * uiScale)
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
        .scaledFont(12)
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .frame(width: 190 * uiScale)
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
                if state.autoSummary && state.isListening {
                    LiveBadge(interval: state.liveInterval)
                }
                if state.isWarmingAI {
                    ProgressView().controlSize(.small)
                    Text("Cargando Qwen…").scaledFont(11).foregroundStyle(.secondary)
                } else if state.isSummarizing {
                    ProgressView().controlSize(.small)
                    Text("Actualizando…").scaledFont(11).foregroundStyle(.secondary)
                } else if let updated = state.summaryUpdatedAt {
                    Text("Actualizado \(updated)").scaledFont(11).foregroundStyle(.secondary)
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
                VStack(alignment: .leading, spacing: 10) {
                    // Errors show above the last good summary instead of hiding it.
                    if let error = state.summaryError {
                        Label(error, systemImage: "exclamationmark.triangle")
                            .scaledFont(12)
                            .foregroundStyle(Color.orange)
                    }
                    if !state.liveSummary.isEmpty {
                        MarkdownText(markdown: state.liveSummary, fontSize: state.transcriptFontSize)
                    } else if state.summaryError == nil {
                        Text(
                            state.autoSummary
                                ? "El resumen se escribe solo mientras escuchas (cada \(state.liveInterval) s) y saca las preguntas clave al panel de abajo. También puedes pulsar ↻."
                                : "Pulsa ↻ para generar el resumen de lo transcrito."
                        )
                        .foregroundStyle(.secondary)
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
                .scaledFont(11)
                .foregroundStyle(.secondary)
            }
        } content: {
            VStack(spacing: 0) {
                if !state.keyQuestions.isEmpty {
                    keyQuestionsList
                    Divider().opacity(0.5)
                }
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 10) {
                            if state.qaItems.isEmpty {
                                Text(
                                    state.keyQuestions.isEmpty
                                        ? "Aquí aparecen las preguntas clave que se escuchen en el audio. También puedes preguntar lo que quieras: «what was decided?», «what are my action items?»…"
                                        : "Toca una pregunta detectada para que la IA la responda con lo dicho en la reunión, o escribe la tuya."
                                )
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
                // Answers always keep some room, even with several detected questions.
                .frame(minHeight: 90)
                HStack(spacing: 8) {
                    TextField("Pregunta algo sobre la reunión…", text: $state.qaQuestion)
                        .textFieldStyle(.plain)
                        .scaledFont(14)
                        .onSubmit { state.askQuestion() }
                    Button { state.askQuestion() } label: {
                        Image(systemName: "paperplane.fill").scaledFont(14)
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

    /// Key questions extracted from the audio by the live summary; tapping one
    /// asks Qwen to answer it from the transcript.
    private var keyQuestionsList: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("DETECTADAS EN EL AUDIO")
                .scaledFont(10, weight: .bold)
                .foregroundStyle(.secondary)
            ForEach(Array(state.keyQuestions.prefix(5).enumerated()), id: \.offset) { item in
                let answering = state.isAnswering(item.element)
                Button {
                    state.ask(item.element)
                } label: {
                    HStack(alignment: .firstTextBaseline, spacing: 8) {
                        Image(systemName: "questionmark.circle.fill")
                            .foregroundStyle(Theme.qa)
                        Text(MarkdownText.inline(item.element))
                            .font(.system(size: CGFloat(max(12, state.transcriptFontSize - 2))))
                            .lineLimit(2)
                            .multilineTextAlignment(.leading)
                            .frame(maxWidth: .infinity, alignment: .leading)
                        if answering {
                            ProgressView().controlSize(.small)
                            Text("Respondiendo…").scaledFont(11).foregroundStyle(Theme.qa)
                        } else {
                            Image(systemName: "arrow.right.circle")
                                .foregroundStyle(.secondary)
                        }
                    }
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .disabled(answering)
                .help("Pedir a la IA que la responda con lo dicho en la reunión (la respuesta sale abajo)")
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Theme.qa.opacity(0.07))
    }

    private var footer: some View {
        HStack(spacing: 6) {
            Text("\(state.entries.count) bloques · \(state.wordCount) palabras")
            if state.memorySegments > 0 {
                Text("· \(state.memorySegments) fragmentos en memoria")
                    .help("Partes antiguas de la reunión guardadas en disco; la IA trae solo las relevantes al responder")
            }
            Spacer()
            Button { state.openLogsFolder() } label: {
                Label("Logs locales", systemImage: "folder")
            }
            .buttonStyle(.plain)
        }
        .scaledFont(11)
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
            Section("IA local (Qwen)") {
                Picker("Modelo", selection: $state.qwenModel) {
                    ForEach(QwenModels.choices, id: \.id) { choice in
                        Text(choice.label).tag(choice.id)
                    }
                }
                Picker("Idioma de las respuestas", selection: $state.summaryLang) {
                    Text("Inglés").tag("en")
                    Text("Español").tag("es")
                }
                Toggle("Resumen y preguntas clave en vivo", isOn: $state.autoSummary)
                Picker("Actualizar cada", selection: $state.liveInterval) {
                    Text("15 s").tag(15)
                    Text("30 s").tag(30)
                    Text("1 min").tag(60)
                    Text("2 min").tag(120)
                }
                .disabled(!state.autoSummary)
                Button("Editar contexto de la IA (qwen_context.txt)…") { state.openAIContext() }
            }
            Section("Alertas") {
                TextField("Palabras clave", text: $state.alerts, prompt: Text("julian, deadline, urgente"))
                Toggle("Sonido al detectar", isOn: $state.alertSound)
            }
            Section("Lectura") {
                Picker("Tamaño de la interfaz", selection: $state.uiScale) {
                    Text("Normal").tag(1.0)
                    Text("Grande").tag(1.15)
                    Text("Muy grande").tag(1.3)
                    Text("Enorme").tag(1.45)
                }
                Stepper("Letra de lectura: \(state.transcriptFontSize) pt", value: $state.transcriptFontSize, in: 12...28)
                Toggle("Autodesplazar al final", isOn: $state.autoscroll)
            }
        }
        .formStyle(.grouped)
        .font(.system(size: (13 * state.uiScale).rounded()))
        .controlSize(state.uiScale >= 1.15 ? .large : .regular)
        .frame(width: 440 * state.uiScale, height: 600 * state.uiScale)
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
        .environment(\.uiScale, CGFloat(state.uiScale))
        .font(.system(size: (13 * state.uiScale).rounded()))
        .controlSize(state.uiScale >= 1.15 ? .large : .regular)
        .background(WindowReader { window in windowBox.window = window })
        .onAppear {
            state.applyOnTop()
            state.prewarm()
        }
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
            Button("Agrandar interfaz") { AppState.shared.changeUIScale(1) }
                .keyboardShortcut("+", modifiers: [.command, .option])
            Button("Reducir interfaz") { AppState.shared.changeUIScale(-1) }
                .keyboardShortcut("-", modifiers: [.command, .option])
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
