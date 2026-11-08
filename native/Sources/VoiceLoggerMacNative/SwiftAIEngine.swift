import Foundation

/// What the app asks the local AI. Two implementations: `QwenServer` (Python
/// process with mlx-lm) and `SwiftAIEngine` (MLX Swift inside the app). The
/// app only talks to this protocol, so ⚙ can switch engines.
protocol AIEngine: Sendable {
    func setModel(_ newModel: String) async
    func warmup() async throws
    func liveInsights(text: String, newText: String, previous: String, lang: String) async throws -> LiveInsights
    /// `partial` receives the answer while it is written (engines that cannot
    /// stream it just never call it).
    func ask(
        question: String, text: String, notes: String, session: String, memoryDir: String,
        lang: String, detail: String, mode: String, web: Bool, sites: String, readPages: Bool,
        prefetchWeb: Bool, partial: PartialText?
    ) async throws -> (answer: String, sources: [WebSource])
    func extend(
        question: String, previous: String, notes: String, lang: String, detail: String,
        sites: String, readPages: Bool, partial: PartialText?
    ) async throws -> (answer: String, sources: [WebSource])
    func complement(
        question: String, previous: String, notes: String, lang: String, detail: String,
        sites: String, readPages: Bool, partial: PartialText?
    ) async throws -> (answer: String, sources: [WebSource])
    func questions(text: String, known: [String], lang: String) async throws -> [String]
    func tasks(notes: String, text: String, lang: String, today: String) async throws -> [MeetingTask]
    func remember(session: String, memoryDir: String, index: Int, start: String, end: String, text: String) async throws
    func unload() async -> Bool
}

/// The Python process answers in one piece (one JSON line per request): no streaming.
extension QwenServer: AIEngine {
    func ask(
        question: String, text: String, notes: String, session: String, memoryDir: String,
        lang: String, detail: String, mode: String, web: Bool, sites: String, readPages: Bool,
        prefetchWeb: Bool, partial: PartialText?
    ) async throws -> (answer: String, sources: [WebSource]) {
        try await ask(
            question: question, text: text, notes: notes, session: session, memoryDir: memoryDir,
            lang: lang, detail: detail, mode: mode, web: web, sites: sites, readPages: readPages,
            prefetchWeb: prefetchWeb
        )
    }

    func extend(
        question: String, previous: String, notes: String, lang: String, detail: String,
        sites: String, readPages: Bool, partial: PartialText?
    ) async throws -> (answer: String, sources: [WebSource]) {
        try await extend(
            question: question, previous: previous, notes: notes, lang: lang, detail: detail,
            sites: sites, readPages: readPages
        )
    }

    func complement(
        question: String, previous: String, notes: String, lang: String, detail: String,
        sites: String, readPages: Bool, partial: PartialText?
    ) async throws -> (answer: String, sources: [WebSource]) {
        try await complement(
            question: question, previous: previous, notes: notes, lang: lang, detail: detail,
            sites: sites, readPages: readPages
        )
    }
}

/// The AI engine without Python: a port of `summarizer.py` (same prompts,
/// token budgets, cleaning and web search) on top of `LocalLLM` (MLX Swift).
actor SwiftAIEngine: AIEngine {
    struct Failure: LocalizedError {
        let message: String
        var errorDescription: String? { message }
    }

    // Token budgets per prompt block (same as summarizer.py): the prompt size
    // does not grow with the length of the meeting.
    static let liveTextTokens = 3000
    static let liveNotesTokens = 1000
    static let liveNewTokens = 2500
    static let qaNotesTokens = 800
    static let qaMemoryTokens = 1500
    static let qaRecentTokens = 1500
    static let qaMemoryHits = 3
    static let extendNotesTokens = 600
    static let extendPreviousTokens = 700
    static let webTokens = 2200  // the AI reads 1-3 chosen pages, not all of them
    static let webResults = 6
    static let webReadPages = 3
    static let tasksNotesTokens = 1000
    static let tasksRecentTokens = 2500
    static let questionsTextTokens = 450
    static let questionsKnownTokens = 250
    static let questionsMaxTokens = 160
    static let answerMaxTokens = ["long": 700, "short": 300]
    static let complementMaxTokens = ["long": 650, "short": 250]
    /// The model is freed after this long without requests (it reloads on the next one).
    static let idleUnloadSeconds: Double = 600
    static let maxPrefetched = 8

    private let llm: LocalLLM
    private let contextFile: URL?
    private let contextExample: URL?
    /// Ideas mode: web searches started while the model writes the quick answer.
    private var prefetched: [String: Task<[WebResult], Error>] = [:]
    private var prefetchOrder: [String] = []
    private var idleUnload: Task<Void, Never>?

    init(model: String, contextFile: URL?, contextExample: URL?, speculative: Bool = true) {
        llm = LocalLLM(modelID: model, useDraft: speculative)
        self.contextFile = contextFile
        self.contextExample = contextExample
    }

    // MARK: Model

    func setModel(_ newModel: String) async {
        await llm.setModel(newModel)
    }

    /// Speculative decoding with a small helper model (⚙).
    func setSpeculative(_ enabled: Bool) async {
        await llm.setDraftEnabled(enabled)
    }

    func warmup() async throws {
        defer { scheduleIdleUnload() }
        _ = try await llm.load()
    }

    func unload() async -> Bool {
        idleUnload?.cancel()
        return await llm.unload()
    }

    private func scheduleIdleUnload() {
        idleUnload?.cancel()
        idleUnload = Task { [weak self] in
            try? await Task.sleep(for: .seconds(Self.idleUnloadSeconds))
            guard !Task.isCancelled else { return }
            _ = await self?.unloadWhenIdle()
        }
    }

    private func unloadWhenIdle() async -> Bool {
        await llm.unload()
    }

    /// Loads the model first, so token counts use the real tokenizer.
    private func ready() async throws {
        idleUnload?.cancel()
        _ = try await llm.load()
    }

    private func systemPrompt(_ prompts: AIPrompts, open: Bool = false) -> String {
        AIText.systemPrompt(prompts, context: AIText.loadContext(from: contextFile, example: contextExample), open: open)
    }

    private func head(_ text: String, _ tokens: Int) async -> String {
        await llm.fitTokens(text, tokens, keepTail: false)
    }

    private func tail(_ text: String, _ tokens: Int) async -> String {
        await llm.fitTokens(text, tokens, keepTail: true)
    }

    private func detailKey(_ detail: String) -> String {
        Self.answerMaxTokens[detail] == nil ? "long" : detail
    }

    // MARK: Live summary

    func liveInsights(text: String, newText: String, previous: String, lang: String) async throws -> LiveInsights {
        defer { scheduleIdleUnload() }
        let prompts = AIPrompts.forLanguage(lang)
        let markdown: String
        if AIText.trimmed(text).isEmpty, AIText.trimmed(newText).isEmpty {
            guard !AIText.trimmed(previous).isEmpty else { throw Failure(message: "no transcript to summarize yet") }
            markdown = AIText.dedupeLines(previous)
        } else {
            try await ready()
            let raw = try await liveNotes(text: text, newText: newText, previous: previous, prompts: prompts)
            markdown = try AIText.cleanMinutes(raw)
        }
        let (rest, questions) = AIText.splitKeyQuestions(markdown)
        let (summary, actionItems) = AIText.splitActionItems(rest)
        return LiveInsights(markdown: markdown, summary: summary, questions: questions, actionItems: actionItems)
    }

    private func liveNotes(text: String, newText: String, previous: String, prompts: AIPrompts) async throws -> String {
        if !AIText.trimmed(text).isEmpty {
            var text = text
            if text.count <= AIText.singlePassChars { text = await tail(text, Self.liveTextTokens) }
            return try await summarize(text, prompts: prompts)
        }
        if !AIText.trimmed(previous).isEmpty, !AIText.trimmed(newText).isEmpty {
            // Incremental: previous notes + only the new part, both with a fixed
            // budget, so the cost does not grow with the meeting. The notes are
            // cleaned of repeats so a loop is not carried over.
            let notes = await head(AIText.dedupeLines(previous), Self.liveNotesTokens)
            let new = await tail(newText, Self.liveNewTokens)
            let user = AIText.fill(prompts.update, ["previous": notes, "new": new]) + prompts.sections
            return try await llm.generate(system: systemPrompt(prompts), user: user, maxTokens: 700, priority: .background)
        }
        let recent = await tail(newText, Self.liveTextTokens)
        return try await summarize(recent, prompts: prompts)
    }

    private func minutesPrompt(_ text: String, _ prompts: AIPrompts) -> String {
        "\(prompts.minutes)\(prompts.sections)\n\(prompts.transcript)\n\(text)"
    }

    private func summarize(_ text: String, prompts: AIPrompts) async throws -> String {
        let system = systemPrompt(prompts)
        if text.count <= AIText.singlePassChars {
            return try await llm.generate(
                system: system, user: minutesPrompt(text, prompts), maxTokens: 700, priority: .background
            )
        }
        // Long meeting: notes per chunk and final notes from them, so the end
        // of the meeting (decisions, to-dos) is never lost.
        var notes: [String] = []
        for (offset, chunk) in AIText.splitText(text).enumerated() {
            let index = String(offset + 1)
            let note = try await llm.generate(
                system: system, user: AIText.fill(prompts.chunk, ["index": index, "chunk": chunk]), maxTokens: 400,
                priority: .background
            )
            notes.append("\(prompts.part) \(index):\n\(note)")
        }
        var joined = notes.joined(separator: "\n\n")
        if joined.count > AIText.maxNotesChars { joined = String(joined.suffix(AIText.maxNotesChars)) }
        return try await llm.generate(
            system: system, user: minutesPrompt(prompts.consolidated + "\n" + joined, prompts), maxTokens: 700,
            priority: .background
        )
    }

    // MARK: Questions panel

    func ask(
        question: String, text: String, notes: String, session: String, memoryDir: String,
        lang: String, detail: String, mode: String, web: Bool, sites: String, readPages: Bool,
        prefetchWeb: Bool, partial: PartialText?
    ) async throws -> (answer: String, sources: [WebSource]) {
        defer { scheduleIdleUnload() }
        let question = AIText.trimmed(question)
        guard !question.isEmpty else { throw Failure(message: "empty question") }
        guard !AIText.trimmed(text).isEmpty else { throw Failure(message: "no transcript to answer from yet") }
        let prompts = AIPrompts.forLanguage(lang)
        let memory = recall(memoryDir: memoryDir, session: session, question: question, recentText: text)
        let read = readPages ? Self.webReadPages : 0
        if mode == "open" {
            if prefetchWeb { prefetch(question, lang: lang, sites: sites, read: read) }
            return try await openAnswer(
                question: question, text: text, notes: notes, memory: memory, prompts: prompts, lang: lang,
                detail: detail, web: web, sites: sites, read: read, partial: partial
            )
        }
        let answer = try await meetingAnswer(
            question: question, text: text, notes: notes, memory: memory, prompts: prompts, detail: detail,
            partial: partial
        )
        return (answer, [])
    }

    /// "Reunión" mode: only what was said in the meeting.
    private func meetingAnswer(
        question: String, text: String, notes: String, memory: String, prompts: AIPrompts, detail: String,
        partial: PartialText?
    ) async throws -> String {
        let text = AIText.tailText(AIText.trimmed(text), maxChars: AIText.qaMaxChars, prompts: prompts)
        try await ready()
        let notes = await head(notes, Self.qaNotesTokens)
        let memory = await head(memory, Self.qaMemoryTokens)
        let recent = await tail(text, Self.qaRecentTokens)
        var blocks: [String] = []
        if !notes.isEmpty { blocks.append("\(prompts.notes)\n\(notes)") }
        if !memory.isEmpty { blocks.append("\(prompts.memory)\n\(memory)") }
        blocks.append("\(notes.isEmpty && memory.isEmpty ? prompts.meetingTranscript : prompts.recent)\n\(recent)")
        let detail = detailKey(detail)
        blocks.append("\(prompts.question) \(question)\n\(detail == "short" ? prompts.instructionShort : prompts.instructionLong)")
        let raw = try await llm.generate(
            system: systemPrompt(prompts), user: blocks.joined(separator: "\n\n"),
            maxTokens: Self.answerMaxTokens[detail] ?? 700, temperature: 0.2, partial: partial
        )
        return try AIText.cleanAnswer(raw)
    }

    /// "Ideas" mode: the meeting as context + the model's knowledge + the web.
    private func openAnswer(
        question: String, text: String, notes: String, memory: String, prompts: AIPrompts, lang: String,
        detail: String, web: Bool, sites: String, read: Int, partial: PartialText?
    ) async throws -> (answer: String, sources: [WebSource]) {
        var results: [WebResult] = []
        if web {
            // Without internet it still answers with the meeting and what it knows.
            results = (try? await search(question, lang: lang, sites: sites, read: read)) ?? []
        }
        let text = AIText.tailText(AIText.trimmed(text), maxChars: AIText.qaMaxChars, prompts: prompts)
        try await ready()
        let notes = await head(notes, Self.qaNotesTokens)
        let memory = await head(memory, Self.qaMemoryTokens)
        let recent = await tail(text, Self.qaRecentTokens)
        var blocks: [String] = []
        if !notes.isEmpty { blocks.append("\(prompts.notes)\n\(notes)") }
        if !memory.isEmpty { blocks.append("\(prompts.memory)\n\(memory)") }
        if !recent.isEmpty {
            blocks.append("\(notes.isEmpty && memory.isEmpty ? prompts.meetingTranscript : prompts.recent)\n\(recent)")
        }
        if !results.isEmpty {
            let webText = await head(AIText.webBlock(results, prompts: prompts), Self.webTokens)
            blocks.append("\(prompts.web)\n\(webText)")
        }
        let detail = detailKey(detail)
        let length = detail == "short" ? prompts.openShort : prompts.openLong
        let instruction = results.isEmpty ? prompts.openInstructionNoWeb : prompts.openInstruction
        blocks.append("\(prompts.question) \(question)\n\(instruction) \(length)")
        let raw = try await llm.generate(
            system: systemPrompt(prompts, open: true), user: blocks.joined(separator: "\n\n"),
            maxTokens: (Self.answerMaxTokens[detail] ?? 700) + 100, temperature: 0.4, partial: partial
        )
        return (try AIText.cleanAnswer(raw), AIText.sources(results))
    }

    /// Old parts of the meeting (on disk) relevant to the question.
    private func recall(memoryDir: String, session: String, question: String, recentText: String) -> String {
        guard !memoryDir.isEmpty, !session.isEmpty,
              let store = try? MeetingMemoryStore(directory: memoryDir, session: session) else { return "" }
        let hits = store.search(question, k: Self.qaMemoryHits) { recentText.contains($0.text) }
        return hits.map { "[\($0.start) – \($0.end)]\n\($0.text)" }.joined(separator: "\n\n")
    }

    // MARK: Internet

    private func search(_ question: String, lang: String, sites: String, read: Int) async throws -> [WebResult] {
        try await Self.loggedSearch(question, lang: lang, sites: sites, read: read)
    }

    /// The search plus one log line: results, reliable ones, pages read, time
    /// (not the question text).
    private static func loggedSearch(_ question: String, lang: String, sites: String, read: Int) async throws -> [WebResult] {
        let started = Date()
        do {
            let results = try await WebSearch.search(
                question, language: lang, limit: Self.webResults, sites: sites, read: read,
                tavilyKey: TavilyKey.shared.current
            )
            SessionLog.shared.log(
                "WEB", "\(results.count) resultados (\(results.filter(\.trusted).count) confiables, "
                    + "\(results.filter { $0.content != nil }.count) páginas leídas) en "
                    + "\(SessionLog.seconds(Date().timeIntervalSince(started))) · dominios: "
                    + results.map { WebSearch.hostOf($0.url) }.joined(separator: ", ")
            )
            return results
        } catch {
            SessionLog.shared.log("ERROR", "búsqueda web: \(error.localizedDescription)")
            throw error
        }
    }

    private func prefetchKey(_ question: String, _ lang: String, _ sites: String, _ read: Int) -> String {
        "\(AIText.trimmed(question))\u{1F}\(lang)\u{1F}\(sites)\u{1F}\(read)"
    }

    /// Starts the web search in the background (network only, the model is
    /// free to write the quick answer meanwhile).
    private func prefetch(_ question: String, lang: String, sites: String, read: Int) {
        let key = prefetchKey(question, lang, sites, read)
        guard prefetched[key] == nil else { return }
        while prefetchOrder.count >= Self.maxPrefetched {  // never used (e.g. the answer failed)
            let old = prefetchOrder.removeFirst()
            prefetched.removeValue(forKey: old)?.cancel()
        }
        prefetched[key] = Task {
            try await Self.loggedSearch(question, lang: lang, sites: sites, read: read)
        }
        prefetchOrder.append(key)
    }

    /// The prefetched search, or a new one if there is none.
    private func webResults(_ question: String, lang: String, sites: String, read: Int) async throws -> [WebResult] {
        let key = prefetchKey(question, lang, sites, read)
        if let task = prefetched.removeValue(forKey: key) {
            prefetchOrder.removeAll { $0 == key }
            return try await task.value
        }
        return try await search(question, lang: lang, sites: sites, read: read)
    }

    /// "Ampliar con internet": the whole answer again, extended with the web.
    func extend(
        question: String, previous: String, notes: String, lang: String, detail: String,
        sites: String, readPages: Bool, partial: PartialText?
    ) async throws -> (answer: String, sources: [WebSource]) {
        defer { scheduleIdleUnload() }
        let question = AIText.trimmed(question)
        guard !question.isEmpty else { throw Failure(message: "empty question") }
        let results = try await search(question, lang: lang, sites: sites, read: readPages ? Self.webReadPages : 0)
        guard !results.isEmpty else {
            return (lang == "es" ? "No encontré resultados en internet para esa pregunta."
                : "I couldn't find web results for that question.", [])
        }
        let prompts = AIPrompts.forLanguage(lang)
        try await ready()
        var blocks = ["\(prompts.question) \(question)"]
        let previous = await head(previous, Self.extendPreviousTokens)
        let notes = await head(notes, Self.extendNotesTokens)
        if !previous.isEmpty { blocks.append("\(prompts.previous)\n\(previous)") }
        if !notes.isEmpty { blocks.append("\(prompts.notes)\n\(notes)") }
        let webText = await head(AIText.webBlock(results, prompts: prompts), Self.webTokens)
        blocks.append("\(prompts.web)\n\(webText)")
        blocks.append(prompts.extend)
        let detail = detailKey(detail)
        let raw = try await llm.generate(
            system: systemPrompt(prompts), user: blocks.joined(separator: "\n\n"),
            maxTokens: (Self.answerMaxTokens[detail] ?? 700) + 100, temperature: 0.2, partial: partial
        )
        return (try AIText.cleanAnswer(raw), AIText.sources(results))
    }

    /// Ideas mode, second step: only what the web adds to the answer already
    /// given ("" if nothing new).
    func complement(
        question: String, previous: String, notes: String, lang: String, detail: String,
        sites: String, readPages: Bool, partial: PartialText?
    ) async throws -> (answer: String, sources: [WebSource]) {
        defer { scheduleIdleUnload() }
        let question = AIText.trimmed(question)
        guard !question.isEmpty else { throw Failure(message: "empty question") }
        let results = try await webResults(question, lang: lang, sites: sites, read: readPages ? Self.webReadPages : 0)
        guard !results.isEmpty else { return ("", []) }
        let prompts = AIPrompts.forLanguage(lang)
        try await ready()
        let detail = Self.complementMaxTokens[detail] == nil ? "long" : detail
        var blocks = ["\(prompts.question) \(question)"]
        let notes = await head(notes, Self.extendNotesTokens)
        let previous = await head(previous, Self.extendPreviousTokens)
        if !notes.isEmpty { blocks.append("\(prompts.notes)\n\(notes)") }
        if !previous.isEmpty { blocks.append("\(prompts.given)\n\(previous)") }
        let webText = await head(AIText.webBlock(results, prompts: prompts), Self.webTokens)
        blocks.append("\(prompts.web)\n\(webText)")
        let length = detail == "short" ? prompts.complementShort : prompts.complementLong
        blocks.append("\(prompts.complement) \(length)")
        let raw = try await llm.generate(
            system: systemPrompt(prompts, open: true), user: blocks.joined(separator: "\n\n"),
            maxTokens: (Self.complementMaxTokens[detail] ?? 650) + 60, temperature: 0.2, partial: partial
        )
        var answer = try AIText.cleanAnswer(raw)
        if AIText.saysNothingNew(answer) { answer = "" }
        return (answer, AIText.sources(results))
    }

    // MARK: Live questions, tasks, memory

    func questions(text: String, known: [String], lang: String) async throws -> [String] {
        defer { scheduleIdleUnload() }
        let text = AIText.trimmed(text)
        guard !text.isEmpty else { return [] }
        let prompts = AIPrompts.forLanguage(lang)
        try await ready()
        var blocks: [String] = []
        let known = known.map(AIText.trimmed).filter { !$0.isEmpty }
        if !known.isEmpty {
            let listed = await tail(known.map { "- \($0)" }.joined(separator: "\n"), Self.questionsKnownTokens)
            blocks.append("\(prompts.known)\n\(listed)")
        }
        let latest = await tail(text, Self.questionsTextTokens)
        blocks.append("\(prompts.latest)\n\(latest)")
        blocks.append(AIText.fill(prompts.questions, ["limit": String(AIText.questionsMax)]))
        let raw = try await llm.generate(
            system: systemPrompt(prompts), user: blocks.joined(separator: "\n\n"),
            maxTokens: Self.questionsMaxTokens, temperature: 0.2, priority: .background
        )
        return AIText.parseQuestions(raw)
    }

    func tasks(notes: String, text: String, lang: String, today: String) async throws -> [MeetingTask] {
        defer { scheduleIdleUnload() }
        guard !(AIText.trimmed(notes).isEmpty && AIText.trimmed(text).isEmpty) else {
            throw Failure(message: "no transcript to extract tasks from yet")
        }
        let prompts = AIPrompts.forLanguage(lang)
        let text = AIText.tailText(AIText.trimmed(text), maxChars: AIText.qaMaxChars, prompts: prompts)
        try await ready()
        var blocks = [AIText.fill(prompts.tasks, ["today": today])]
        let notes = await head(notes, Self.tasksNotesTokens)
        let recent = await tail(text, Self.tasksRecentTokens)
        if !notes.isEmpty { blocks.append("\(prompts.notes)\n\(notes)") }
        if !recent.isEmpty { blocks.append("\(notes.isEmpty ? prompts.meetingTranscript : prompts.recent)\n\(recent)") }
        // No repetition penalty: the "task | owner | due" format repeats on purpose.
        let raw = try await llm.generate(
            system: systemPrompt(prompts), user: blocks.joined(separator: "\n\n"),
            maxTokens: 500, temperature: 0.1, penalizeRepeats: false
        )
        return AIText.parseTasks(raw)
    }

    /// Stores a closed transcript fragment on disk (idempotent by `index`). No model.
    func remember(session: String, memoryDir: String, index: Int, start: String, end: String, text: String) async throws {
        let store = try MeetingMemoryStore(directory: memoryDir, session: session)
        try store.add(index: index, start: start, end: end, text: text)
    }
}
