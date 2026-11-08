import Foundation

/// Result of a live update.
struct LiveInsights: Decodable, Sendable {
    /// Full notes, sent back as `previous` on the next incremental update.
    let markdown: String
    /// Notes without the key-questions and action-items sections ("Resumen" tab).
    let summary: String
    /// Most important questions heard so far ("Preguntas" tab and Questions panel).
    let questions: [String]
    /// Action items so far ("Tareas" tab).
    let actionItems: [String]

    enum CodingKeys: String, CodingKey {
        case markdown, summary, questions
        case actionItems = "action_items"
    }

    init(markdown: String, summary: String, questions: [String], actionItems: [String]) {
        self.markdown = markdown
        self.summary = summary
        self.questions = questions
        self.actionItems = actionItems
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        markdown = try container.decode(String.self, forKey: .markdown)
        summary = try container.decode(String.self, forKey: .summary)
        questions = try container.decodeIfPresent([String].self, forKey: .questions) ?? []
        actionItems = try container.decodeIfPresent([String].self, forKey: .actionItems) ?? []
    }
}

/// A web page used by an answer with internet (Ideas mode or "Ampliar con internet").
struct WebSource: Decodable, Sendable, Equatable {
    let title: String
    let url: String
    /// Wikipedia, government, universities, science, serious press, official docs.
    var trusted: Bool? = nil
    /// The AI read the page text, not only the search engine snippet.
    var read: Bool? = nil

    /// "wikipedia.org", "hbr.org"… (without "www.").
    var domain: String {
        guard let host = URL(string: url)?.host() else { return "" }
        return host.hasPrefix("www.") ? String(host.dropFirst(4)) : host
    }
}

/// Splits a byte stream into newline-terminated lines. Thread-safe: it is fed
/// from the pipe's background queue.
final class LineBuffer: @unchecked Sendable {
    private let lock = NSLock()
    private var data = Data()

    func append(_ chunk: Data) -> [Data] {
        lock.lock()
        defer { lock.unlock() }
        data.append(chunk)
        var lines: [Data] = []
        while let newline = data.firstIndex(of: 0x0A) {
            let line = Data(data[data.startIndex..<newline])
            data = Data(data[data.index(after: newline)...])
            if !line.isEmpty { lines.append(line) }
        }
        return lines
    }
}

/// Keeps one `summarizer.py serve` process alive, so Qwen is loaded once and
/// stays in memory: live summaries and answers no longer pay the model load
/// (~1 GB) on every call.
///
/// Protocol: one JSON request per line on stdin, one JSON reply per line on
/// stdout, matched by `id`. Python handles requests one after another. The
/// process exits on its own when the app quits (its stdin closes).
actor QwenServer {
    private struct Reply: Decodable {
        let id: String?
        let ok: Bool
        let error: String?
    }

    private struct Envelope<Payload: Decodable>: Decodable {
        let result: Payload
    }

    private struct Answer: Decodable {
        let answer: String
        let sources: [WebSource]?
    }

    private struct Extended: Decodable {
        let answer: String
        let sources: [WebSource]
    }

    private struct TasksPayload: Decodable {
        let tasks: [MeetingTask]
    }

    private struct QuestionsPayload: Decodable {
        let questions: [String]
    }

    private let bridge: PythonBridge
    private let logURL: URL
    private var model: String
    private var process: Process?
    private var input: FileHandle?
    private var pending: [String: CheckedContinuation<Data, Error>] = [:]

    init(bridge: PythonBridge, logURL: URL, model: String) {
        self.bridge = bridge
        self.logURL = logURL
        self.model = model
    }

    /// Stops the current process; the next request starts one with `newModel`.
    func setModel(_ newModel: String) {
        guard newModel != model else { return }
        model = newModel
        try? input?.close()
        process?.terminate()
        process = nil
        input = nil
        failPending(reason: "Se cambió el modelo de IA; vuelve a intentarlo.")
    }

    /// Loads Qwen ahead of time so the first live summary is quick.
    func warmup() async throws {
        _ = try await send(["type": "warmup"])
    }

    func liveInsights(text: String, newText: String, previous: String, lang: String) async throws -> LiveInsights {
        let data = try await send([
            "type": "live", "text": text, "new_text": newText, "previous": previous, "lang": lang,
        ])
        return try JSONDecoder().decode(Envelope<LiveInsights>.self, from: data).result
    }

    /// `notes` (the live summary) and the meeting memory on disk (`session`
    /// in `memoryDir`) cover what no longer fits in `text`.
    /// `detail`: "long" (thorough, default) or "short" (2-4 sentences).
    /// `mode`: "meeting" answers only from the meeting; "open" (brainstorming)
    /// adds the model's own knowledge and, if `web`, web results (only the
    /// question text is searched). `sites`: preferred sites, searched with
    /// `site:`; `readPages`: open the best pages instead of only the snippets.
    /// `prefetchWeb`: start the web search now, in the background, for a later
    /// `complement` (Ideas mode answers first and complements afterwards).
    func ask(
        question: String, text: String, notes: String, session: String, memoryDir: String,
        lang: String, detail: String, mode: String, web: Bool, sites: String, readPages: Bool,
        prefetchWeb: Bool = false
    ) async throws -> (answer: String, sources: [WebSource]) {
        let data = try await send([
            "type": "ask", "question": question, "text": text, "notes": notes,
            "session": session, "memory_dir": memoryDir, "lang": lang, "detail": detail,
            "mode": mode, "web": web ? "1" : "0", "sites": sites, "read": readPages ? "1" : "0",
            "prefetch": prefetchWeb ? "1" : "0",
        ])
        let result = try JSONDecoder().decode(Envelope<Answer>.self, from: data).result
        return (result.answer.trimmingCharacters(in: .whitespacesAndNewlines), result.sources ?? [])
    }

    /// Extends an answer with web results. Only `question` is searched on the
    /// internet; the meeting stays on the Mac (notes go to the local model only).
    func extend(
        question: String, previous: String, notes: String, lang: String, detail: String,
        sites: String, readPages: Bool
    ) async throws -> (answer: String, sources: [WebSource]) {
        let data = try await send([
            "type": "extend", "question": question, "previous": previous, "notes": notes,
            "lang": lang, "detail": detail, "sites": sites, "read": readPages ? "1" : "0",
        ])
        let result = try JSONDecoder().decode(Envelope<Extended>.self, from: data).result
        return (result.answer.trimmingCharacters(in: .whitespacesAndNewlines), result.sources)
    }

    /// Ideas mode, second step: what the web adds to an answer already given
    /// ("Puede complementar tu pregunta"). Uses the search started by `ask`
    /// with `prefetchWeb`. Empty answer = the web added nothing new.
    func complement(
        question: String, previous: String, notes: String, lang: String, detail: String,
        sites: String, readPages: Bool
    ) async throws -> (answer: String, sources: [WebSource]) {
        let data = try await send([
            "type": "complement", "question": question, "previous": previous, "notes": notes,
            "lang": lang, "detail": detail, "sites": sites, "read": readPages ? "1" : "0",
        ])
        let result = try JSONDecoder().decode(Envelope<Extended>.self, from: data).result
        return (result.answer.trimmingCharacters(in: .whitespacesAndNewlines), result.sources)
    }

    /// New questions in the latest part of the meeting (`text`), not repeating
    /// `known`. Light request (short prompt, a few lines out) for the live loop.
    func questions(text: String, known: [String], lang: String) async throws -> [String] {
        let data = try await send([
            "type": "questions", "text": text, "known": known.joined(separator: "\n"), "lang": lang,
        ])
        return try JSONDecoder().decode(Envelope<QuestionsPayload>.self, from: data).result.questions
    }

    /// Action items of the meeting (`today` = YYYY-MM-DD, to resolve "next Friday").
    func tasks(notes: String, text: String, lang: String, today: String) async throws -> [MeetingTask] {
        let data = try await send(["type": "tasks", "notes": notes, "text": text, "lang": lang, "today": today])
        return try JSONDecoder().decode(Envelope<TasksPayload>.self, from: data).result.tasks
    }

    /// Stores a closed transcript fragment in the meeting memory on disk
    /// (idempotent by `index`). No model involved: it is instant.
    func remember(session: String, memoryDir: String, index: Int, start: String, end: String, text: String) async throws {
        _ = try await send([
            "type": "remember", "session": session, "memory_dir": memoryDir,
            "index": String(index), "start": start, "end": end, "text": text,
        ])
    }

    /// Frees Qwen from RAM if the process is running (it reloads on the next request).
    func unload() async -> Bool {
        guard let process, process.isRunning else { return false }
        return (try? await send(["type": "unload"])) != nil
    }

    private func send(_ fields: [String: String]) async throws -> Data {
        try startIfNeeded()
        guard let input else { throw PythonBridge.Failure(message: "Qwen no está disponible.") }
        let id = UUID().uuidString
        var message = fields
        message["id"] = id
        var line = try JSONEncoder().encode(message)
        line.append(0x0A)
        return try await withCheckedThrowingContinuation { continuation in
            pending[id] = continuation
            do {
                try input.write(contentsOf: line)
            } catch {
                pending[id] = nil
                continuation.resume(throwing: PythonBridge.Failure(
                    message: "No se pudo hablar con Qwen: \(error.localizedDescription)"
                ))
            }
        }
    }

    private func startIfNeeded() throws {
        if let process, process.isRunning, input != nil { return }
        // A previous process died before its exit was handled: release its waiters.
        failPending(reason: "Qwen se reinició.")
        // Writing to a dead process' stdin must return an error, not kill the app.
        signal(SIGPIPE, SIG_IGN)

        let process = Process()
        process.executableURL = bridge.python
        process.arguments = [bridge.script.path, "serve"]
        process.currentDirectoryURL = bridge.root
        // Background priority: the live transcription (Whisper, in the app)
        // must win when both want the chip at the same time.
        process.qualityOfService = .utility
        var environment = ProcessInfo.processInfo.environment
        environment["PYTHONIOENCODING"] = "utf-8"
        environment["PYTHONUNBUFFERED"] = "1"
        environment["TOKENIZERS_PARALLELISM"] = "false"
        environment["VOICELOGGER_QWEN_MODEL"] = model
        process.environment = environment

        let stdinPipe = Pipe()
        let stdoutPipe = Pipe()
        try? FileManager.default.createDirectory(
            at: logURL.deletingLastPathComponent(), withIntermediateDirectories: true
        )
        FileManager.default.createFile(atPath: logURL.path, contents: nil)
        process.standardInput = stdinPipe
        process.standardOutput = stdoutPipe
        // stderr (model download progress, warnings) goes to a log file: a
        // pipe nobody reads would fill up and freeze Python.
        process.standardError = try FileHandle(forWritingTo: logURL)

        let buffer = LineBuffer()
        // Both callbacks run on background threads. They are explicitly
        // @Sendable so Swift 6 never treats them as isolated to this actor
        // (that would crash the whole app, e.g. when Python exits).
        stdoutPipe.fileHandleForReading.readabilityHandler = { @Sendable [weak self] handle in
            let chunk = handle.availableData
            guard !chunk.isEmpty else {
                handle.readabilityHandler = nil
                return
            }
            for line in buffer.append(chunk) {
                Task { await self?.deliver(line) }
            }
        }
        process.terminationHandler = { @Sendable [weak self] finished in
            let token = ObjectIdentifier(finished)
            let status = finished.terminationStatus
            Task { await self?.processEnded(token: token, status: status) }
        }
        try process.run()
        self.process = process
        self.input = stdinPipe.fileHandleForWriting
    }

    private func deliver(_ line: Data) {
        guard let reply = try? JSONDecoder().decode(Reply.self, from: line),
              let id = reply.id,
              let continuation = pending.removeValue(forKey: id)
        else { return }
        if reply.ok {
            continuation.resume(returning: line)
        } else {
            continuation.resume(throwing: PythonBridge.Failure(message: reply.error ?? "Error desconocido en Qwen."))
        }
    }

    private func processEnded(token: ObjectIdentifier, status: Int32) {
        guard let process, ObjectIdentifier(process) == token else { return }
        self.process = nil
        input = nil
        let log = (try? String(contentsOf: logURL, encoding: .utf8)) ?? ""
        failPending(reason: PythonBridge.lastMeaningfulLine(of: log) ?? "Qwen se cerró (código \(status)).")
    }

    private func failPending(reason: String) {
        let waiting = pending
        pending.removeAll()
        for continuation in waiting.values {
            continuation.resume(throwing: PythonBridge.Failure(message: reason))
        }
    }
}
