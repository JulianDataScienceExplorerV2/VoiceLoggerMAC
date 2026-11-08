import Foundation

/// The model answered with its reasoning (or nothing useful) instead of the result.
struct AIOutputError: LocalizedError {
    var errorDescription: String? {
        "El modelo respondió con su razonamiento en vez del resultado (pasa con modelos que «piensan»). "
            + "Vuelve a intentarlo o usa Qwen3 4B 2507."
    }
}

/// Text handling of the native AI engine: a port of the cleaning, parsing and
/// prompt-building helpers of `summarizer.py`. Pure functions (no model), so
/// they are unit-tested on their own.
enum AIText {
    // MARK: Limits (same values as summarizer.py)

    static let singlePassChars = 12000
    static let chunkChars = 8000
    static let maxNotesChars = 9000
    static let qaMaxChars = 12000
    static let maxKeyQuestions = 6
    static let maxTasks = 10
    static let maxLinesPerSection = 25
    static let answerMaxLines = 60
    static let maxContextChars = 4000
    static let questionsMax = 4

    // MARK: Regular expressions

    private static func regex(_ pattern: String, _ options: NSRegularExpression.Options = []) -> NSRegularExpression {
        // swiftlint:disable:next force_try
        try! NSRegularExpression(pattern: pattern, options: options)
    }

    nonisolated(unsafe) private static let thinkBlock = regex("<think>.*?</think>", [.dotMatchesLineSeparators])
    nonisolated(unsafe) private static let summaryHeading = regex("^#{1,3}\\s*(summary|resumen)\\b", [.caseInsensitive])
    nonisolated(unsafe) private static let finalAnswer = regex(
        "^\\**\\s*(final answer|answer|respuesta final|respuesta)\\s*:\\**\\s*", [.caseInsensitive]
    )
    nonisolated(unsafe) private static let numbered = regex("^\\d+[.)]\\s+")
    nonisolated(unsafe) private static let listPrefix = regex("^(?:[-*•]|\\d+[.)])\\s*(?:\\[[ xX]\\]\\s*)?")
    nonisolated(unsafe) private static let bullet = regex("^(?:[-*•+]|\\d+[.)])\\s*")
    nonisolated(unsafe) private static let word = regex("\\w+")
    nonisolated(unsafe) private static let nothingNewPattern = regex("^\\W*(nothing new|nada nuevo)\\b", [.caseInsensitive])
    nonisolated(unsafe) private static let isoDate = regex("^\\d{4}-\\d{2}-\\d{2}$")

    private static func matches(_ regex: NSRegularExpression, _ text: String) -> Bool {
        regex.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)) != nil
    }

    private static func removing(_ regex: NSRegularExpression, from text: String) -> String {
        regex.stringByReplacingMatches(in: text, range: NSRange(text.startIndex..., in: text), withTemplate: "")
    }

    private static func words(_ text: String) -> [String] {
        word.matches(in: text, range: NSRange(text.startIndex..., in: text)).compactMap {
            Range($0.range, in: text).map { String(text[$0]) }
        }
    }

    /// Lines like Python's `splitlines()` (without the trailing empty line).
    static func lines(_ text: String) -> [String] {
        let normalized = text.replacingOccurrences(of: "\r\n", with: "\n").replacingOccurrences(of: "\r", with: "\n")
        var result = normalized.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
        if result.last == "" { result.removeLast() }
        return result
    }

    static func trimmed(_ text: String) -> String {
        text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // MARK: Model output

    /// Removes `<think>…</think>` reasoning (Qwen3, LFM2.5).
    static func stripThinking(_ text: String) -> String {
        var result = removing(thinkBlock, from: text)
        if let close = result.range(of: "</think>") {  // the template opened <think> in the prompt
            result = String(result[close.upperBound...])
        }
        if let open = result.range(of: "<think>") {  // reasoning cut by max tokens
            result = String(result[..<open.lowerBound])
        }
        return trimmed(result)
    }

    static func stripCodeFence(_ text: String) -> String {
        var result = trimmed(text)
        guard result.hasPrefix("```") else { return result }
        var all = lines(result)
        if let first = all.first, first.hasPrefix("```") { all.removeFirst() }
        if let last = all.last, trimmed(last) == "```" { all.removeLast() }
        result = trimmed(all.joined(separator: "\n"))
        return result
    }

    private static let reasoningOpeners = [
        "the user", "okay, so", "okay, let", "ok, so", "alright, so", "let me", "let's see", "i need to",
        "we need to", "hmm", "wait,", "el usuario", "necesito", "vale, ", "bueno, el usuario", "veamos",
    ]
    private static let reasoningPhrases = [
        "the user wants", "the user is asking", "the user asked", "user wants me to",
        "el usuario quiere", "el usuario pregunta", "me pide que",
    ]

    /// True if the text looks like the model's reasoning, not an answer.
    static func looksLikeReasoning(_ text: String) -> Bool {
        let head = String(trimmed(text).lowercased().prefix(240))
        if reasoningOpeners.contains(where: { head.hasPrefix($0) }) { return true }
        return reasoningPhrases.contains { head.contains($0) }
    }

    /// Only the notes: from the last summary heading (or the first section
    /// heading) on. Reasoning instead of notes throws `AIOutputError`.
    static func cleanMinutes(_ markdown: String) throws -> String {
        let all = lines(trimmed(markdown))
        let summaryStarts = all.indices.filter { matches(summaryHeading, trimmed(all[$0])) }
        let start: Int
        if let last = summaryStarts.last {
            start = last  // the final version usually comes after the draft
        } else if let first = all.firstIndex(where: { $0.drop(while: { $0 == " " || $0 == "\t" }).hasPrefix("#") }) {
            start = first
        } else {
            throw AIOutputError()
        }
        let cleaned = trimmed(all[start...].joined(separator: "\n"))
        let body = trimmed(all[(start + 1)...].joined(separator: "\n"))
        if looksLikeReasoning(cleaned) || looksLikeReasoning(body) { throw AIOutputError() }
        return dedupeLines(cleaned)
    }

    /// Answer without reasoning. If the model thought aloud but left an
    /// "Answer:" line, what follows it is used; otherwise `AIOutputError`.
    static func cleanAnswer(_ text: String) throws -> String {
        let text = dedupeLines(text, maxPerSection: answerMaxLines)
        if !looksLikeReasoning(text) { return text }
        let all = lines(text)
        if let mark = all.indices.last(where: { matches(finalAnswer, trimmed(all[$0])) }) {
            let tail = [removing(finalAnswer, from: trimmed(all[mark]))] + all[(mark + 1)...]
            let answer = trimmed(tail.joined(separator: "\n"))
            if !answer.isEmpty, !looksLikeReasoning(answer) { return answer }
        }
        throw AIOutputError()
    }

    /// Normalized line, to spot repeats ("- **Budget** ok." == "1. budget OK").
    static func lineKey(_ line: String) -> String {
        let text = removing(bullet, from: trimmed(line))
        return words(text.replacingOccurrences(of: "*", with: "").lowercased()).joined(separator: " ")
    }

    /// Drops repeated lines of each section (even with another bullet, bold or
    /// punctuation), empty bullets and lines past `maxPerSection`. A short line
    /// (under 3 words, e.g. "Pros:") is only dropped if it repeats the previous
    /// one. Headings, blank lines and separators stay. Stops model loops
    /// ("X" ×80) from filling the notes and from being fed back.
    static func dedupeLines(_ text: String, maxPerSection: Int = maxLinesPerSection) -> String {
        var kept: [String] = []
        var seen = Set<String>()
        var previousKey: String?
        var inSection = 0
        for line in lines(text) {
            let stripped = trimmed(line)
            if stripped.hasPrefix("#") {
                seen = []
                previousKey = nil
                inSection = 0
                kept.append(line)
                continue
            }
            if stripped.isEmpty {
                if !(kept.last.map { trimmed($0).isEmpty } ?? false) { kept.append(line) }
                continue
            }
            let key = lineKey(stripped)
            if key.isEmpty {
                if !removing(bullet, from: stripped).isEmpty { kept.append(line) }  // separator or symbol
                continue  // empty bullet ("-" cut by max tokens)
            }
            let repeated = key == previousKey || (seen.contains(key) && key.split(separator: " ").count >= 3)
            previousKey = key
            if repeated || inSection >= maxPerSection { continue }
            seen.insert(key)
            inSection += 1
            kept.append(line)
        }
        return trimmed(kept.joined(separator: "\n"))
    }

    // MARK: Sections of the notes

    private static let noQuestions: Set<String> = [
        "none", "n/a", "na", "none yet", "none so far", "no questions", "no key questions",
        "ninguna", "ninguno", "sin preguntas", "no hay preguntas", "ninguna por ahora",
    ]

    static func isEmptyItem(_ item: String) -> Bool {
        let lowered = item.trimmingCharacters(in: CharacterSet(charactersIn: " .")).lowercased()
        if noQuestions.contains(lowered) { return true }
        if lowered.hasPrefix("("), lowered.hasSuffix(")") { return true }  // the model repeated "(bullets)"
        return (lowered.hasPrefix("no ") && lowered.contains("question")) || lowered.hasPrefix("no hay pregunta")
    }

    /// Takes out the sections whose title matches: (rest of the notes, their bullets).
    static func extractSection(
        _ markdown: String, limit: Int, matching matchesTitle: (String) -> Bool
    ) -> (rest: String, items: [String]) {
        var kept: [String] = []
        var items: [String] = []
        var inside = false
        for line in lines(markdown) {
            let stripped = trimmed(line)
            if stripped.hasPrefix("#") {
                inside = matchesTitle(trimmed(String(stripped.drop(while: { $0 == "#" }))).lowercased())
                if inside { continue }
            } else if inside {
                var item = stripped
                if let bulletPrefix = ["- ", "* ", "• "].first(where: { item.hasPrefix($0) }) {
                    item = trimmed(String(item.dropFirst(bulletPrefix.count)))
                } else {
                    item = trimmed(removing(numbered, from: item))
                }
                if !item.isEmpty, !isEmptyItem(item), items.count < limit { items.append(item) }
                continue
            }
            kept.append(line)
        }
        return (trimmed(kept.joined(separator: "\n")), items)
    }

    static func splitKeyQuestions(_ markdown: String) -> (rest: String, items: [String]) {
        extractSection(markdown, limit: maxKeyQuestions) { $0.contains("question") || $0.contains("pregunta") }
    }

    static func splitActionItems(_ markdown: String) -> (rest: String, items: [String]) {
        extractSection(markdown, limit: maxTasks) { title in
            ["action", "task", "to-do", "todo", "tarea", "pendiente"].contains { title.contains($0) }
        }
    }

    // MARK: Parsing lists from the model

    /// "- question" lines → clean list (no "none", reasoning or repeats).
    static func parseQuestions(_ raw: String, limit: Int = questionsMax) -> [String] {
        var questions: [String] = []
        var seen = Set<String>()
        for line in lines(raw) {
            let stripped = trimmed(line)
            if stripped.isEmpty || stripped.hasPrefix("#") { continue }
            var item = trimmed(removing(listPrefix, from: stripped))
            item = trimmed(item.trimmingCharacters(in: CharacterSet(charactersIn: "\"“”«»*")))
            if item.isEmpty || isEmptyItem(item) || looksLikeReasoning(item) { continue }
            if item.split(whereSeparator: \.isWhitespace).count < 3 { continue }
            if !item.hasSuffix("?") {
                item = item.trimmingCharacters(in: CharacterSet(charactersIn: ".:;")) + "?"
            }
            let key = item.lowercased().filter { $0.isLetter || $0.isNumber || $0 == "_" }
            if seen.contains(key) { continue }
            seen.insert(key)
            questions.append(String(item.prefix(300)))
            if questions.count >= limit { break }
        }
        return questions
    }

    private static let noValue: Set<String> = [
        "", "-", "none", "n/a", "na", "unknown", "tbd", "ninguno", "ninguna", "nadie", "sin fecha",
    ]

    /// "TASK | OWNER | DUE" lines → tasks (due as YYYY-MM-DD or "").
    static func parseTasks(_ raw: String, limit: Int = maxTasks) -> [MeetingTask] {
        var tasks: [MeetingTask] = []
        var seen = Set<String>()
        for line in lines(raw) {
            let cleaned = trimmed(removing(listPrefix, from: trimmed(line)))
            if cleaned.isEmpty { continue }
            let parts = cleaned.components(separatedBy: "|").map(trimmed)
            let title = trimmed(parts[0].trimmingCharacters(in: CharacterSet(charactersIn: " .")))
            if title.isEmpty || isEmptyItem(title) || ["TASK", "TAREA"].contains(title.uppercased()) { continue }
            var owner = parts.count > 1 ? parts[1] : ""
            if noValue.contains(owner.trimmingCharacters(in: CharacterSet(charactersIn: " .")).lowercased()) { owner = "" }
            var due = parts.count > 2 ? parts[2] : ""
            if !(matches(isoDate, due) && isValidDate(due)) { due = "" }
            let key = title.lowercased()
            if seen.contains(key) { continue }
            seen.insert(key)
            tasks.append(MeetingTask(title: String(title.prefix(200)), owner: String(owner.prefix(80)), due: due))
            if tasks.count >= limit { break }
        }
        return tasks
    }

    private static func isValidDate(_ text: String) -> Bool {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.calendar = Calendar(identifier: .gregorian)
        formatter.dateFormat = "yyyy-MM-dd"
        formatter.isLenient = false
        return formatter.date(from: text) != nil
    }

    /// True if a web complement only says "NOTHING NEW" / "NADA NUEVO".
    static func saysNothingNew(_ text: String) -> Bool {
        text.count < 80 && matches(nothingNewPattern, text)
    }

    // MARK: Transcript helpers

    /// Splits the text into ~`maxChars` chunks at line breaks.
    static func splitText(_ text: String, maxChars: Int = chunkChars) -> [String] {
        var chunks: [String] = []
        var current: [String] = []
        var size = 0
        for original in lines(text) {
            var line = original
            while line.count > maxChars {  // a huge line without breaks is cut
                if !current.isEmpty {
                    chunks.append(current.joined(separator: "\n"))
                    current = []
                    size = 0
                }
                chunks.append(String(line.prefix(maxChars)))
                line = String(line.dropFirst(maxChars))
            }
            if size + line.count + 1 > maxChars, !current.isEmpty {
                chunks.append(current.joined(separator: "\n"))
                current = []
                size = 0
            }
            current.append(line)
            size += line.count + 1
        }
        if !current.isEmpty { chunks.append(current.joined(separator: "\n")) }
        return chunks
    }

    /// The most recent `maxChars` without a cut first line, marked as trimmed.
    static func tailText(_ text: String, maxChars: Int, prompts: AIPrompts) -> String {
        guard text.count > maxChars else { return text }
        var tail = String(text.suffix(maxChars))
        if let newline = tail.firstIndex(of: "\n"), tail.index(after: newline) < tail.endIndex {
            tail = String(tail[tail.index(after: newline)...])
        }
        return prompts.trimmed + "\n" + tail
    }

    /// qwen_context.txt without comments (#) or unfilled XXXX lines, created
    /// from the example the first time.
    static func loadContext(from url: URL?, example: URL?) -> String {
        guard let url else { return "" }
        let fm = FileManager.default
        var path = url
        if !fm.fileExists(atPath: url.path), let example, fm.fileExists(atPath: example.path) {
            if (try? fm.copyItem(at: example, to: url)) == nil { path = example }
        }
        guard let raw = try? String(contentsOf: path, encoding: .utf8) else { return "" }
        let kept = lines(raw).filter { line in
            let stripped = trimmed(line)
            return !stripped.hasPrefix("#") && !stripped.uppercased().contains("XXXX")
        }.map { $0.replacingOccurrences(of: "\\s+$", with: "", options: .regularExpression) }
        return String(trimmed(kept.joined(separator: "\n")).prefix(maxContextChars))
    }

    /// Replaces `{key}` placeholders in one pass (inserted text is never
    /// re-scanned, like Python's `str.format`).
    static func fill(_ template: String, _ values: [String: String]) -> String {
        var result = ""
        var rest = Substring(template)
        while let open = rest.firstIndex(of: "{") {
            result += rest[..<open]
            let afterOpen = rest.index(after: open)
            if let close = rest[afterOpen...].firstIndex(of: "}"), let value = values[String(rest[afterOpen..<close])] {
                result += value
                rest = rest[rest.index(after: close)...]
            } else {
                result.append("{")
                rest = rest[afterOpen...]
            }
        }
        result += rest
        return result
    }

    static func systemPrompt(_ prompts: AIPrompts, context: String, open: Bool = false) -> String {
        let job = open ? prompts.jobOpen : prompts.job
        return context.isEmpty ? "\(prompts.role) \(job)" : "\(context)\n\n\(job)"
    }

    // MARK: Web results in the prompt

    /// Numbered results [1], [2]… : the page text if it was read, otherwise
    /// the search snippet; reliable sources are marked.
    static func webBlock(_ results: [WebResult], prompts: AIPrompts) -> String {
        results.enumerated().map { index, result in
            let mark = result.trusted ? " (\(prompts.reliable))" : ""
            let body = result.content ?? result.snippet
            return "[\(index + 1)] \(result.title)\(mark) — \(result.url)\n\(body)"
        }.joined(separator: "\n\n")
    }

    static func sources(_ results: [WebResult]) -> [WebSource] {
        results.map { WebSource(title: $0.title, url: $0.url, trusted: $0.trusted, read: $0.content != nil) }
    }
}
