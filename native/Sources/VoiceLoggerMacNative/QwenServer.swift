import Foundation

/// Result of a live update.
struct LiveInsights: Decodable, Sendable {
    /// Full notes, sent back as `previous` on the next incremental update.
    let markdown: String
    /// Notes without the key-questions section (shown in the summary panel).
    let summary: String
    /// Most important questions heard so far (shown in the questions panel).
    let questions: [String]
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

    /// `notes` (the live summary) covers what no longer fits in `text`.
    func ask(question: String, text: String, notes: String, lang: String) async throws -> String {
        let data = try await send([
            "type": "ask", "question": question, "text": text, "notes": notes, "lang": lang,
        ])
        return try JSONDecoder().decode(Envelope<Answer>.self, from: data).result.answer
            .trimmingCharacters(in: .whitespacesAndNewlines)
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
