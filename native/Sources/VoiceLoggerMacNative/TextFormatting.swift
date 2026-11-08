import Foundation

/// Cleans WhisperKit output before it reaches the UI and the logs.
/// Port of `is_bad_transcript` in `transcriber.py`, plus removal of
/// Whisper special tokens (`<|en|>`, `<|0.00|>`, `[BLANK_AUDIO]`).
enum TranscriptCleaner {
    /// Phrases Whisper "hallucinates" on silence or noise (EN / ES / PT),
    /// compared after `normalize`.
    static let hallucinations: Set<String> = [
        "thank you", "thanks", "thanks for watching", "thank you for watching",
        "please subscribe", "subscribe to my channel", "like and subscribe", "you", "bye",
        "gracias", "gracias por ver el video", "gracias por ver el vídeo", "gracias por vernos",
        "suscribete", "suscríbete", "suscribete al canal", "suscríbete al canal",
        "hasta la proxima", "hasta la próxima",
        "obrigado", "obrigada", "obrigado por assistir", "inscreva-se no canal", "inscreva se no canal",
    ]

    /// Fragments that discard the phrase even with more text around them.
    static let badFragments = [
        "amara.org", "subtitles by", "subtítulos realizados por", "subtitulos realizados por",
        "subtítulos por la comunidad", "legendas pela comunidade",
    ]

    /// Sound descriptions Whisper writes instead of (or next to) speech:
    /// [Music], (upbeat music), [Applause], [BLANK_AUDIO], ♪…
    static let soundTagPattern =
        "[\\[(][^\\])]*(music|música|musica|applause|aplausos|laugh|risa|noise|ruido|silence|silencio|blank_audio|no_speech|inaudible|singing|cheering)[^\\])]*[\\])]"

    /// Removes Whisper special tokens and sound tags; keeps the spoken text.
    static func stripTags(_ raw: String) -> String {
        var text = raw.replacingOccurrences(of: "<\\|[^|>]*\\|>", with: " ", options: .regularExpression)
        text = text.replacingOccurrences(of: soundTagPattern, with: " ", options: [.regularExpression, .caseInsensitive])
        text = text.replacingOccurrences(of: "[♪♫🎵🎶]", with: " ", options: .regularExpression)
        return text.split(whereSeparator: \.isWhitespace).joined(separator: " ")
    }

    /// True when Whisper returned only sound tags / nothing (e.g. "[Music]").
    static func isNonSpeechOnly(_ raw: String) -> Bool {
        stripTags(raw).trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    /// Returns the cleaned text, or nil when nothing useful remains.
    static func clean(_ raw: String) -> String? {
        let text = removeRepeatedClauses(stripTags(raw))
        return isBad(text) ? nil : text
    }

    /// Whisper sometimes loops ("we reveal this big pot… we reveal this big
    /// pot…"). A clause of 4+ words that matches one of the last 3 long clauses
    /// is dropped; short ones ("of course, of course") are real speech and stay.
    static func removeRepeatedClauses(_ text: String) -> String {
        let chars = Array(text)
        var clauses: [(text: String, separator: String)] = []
        var buffer = ""
        for (index, char) in chars.enumerated() {
            let dialogueDash = char == "-" && index > 0 && chars[index - 1] == " "
                && (index + 1 == chars.count || chars[index + 1] == " ")
            if ".!?,;".contains(char) || dialogueDash {
                clauses.append((buffer, String(char)))
                buffer = ""
            } else {
                buffer.append(char)
            }
        }
        if !buffer.isEmpty { clauses.append((buffer, "")) }

        var recent: [String] = []
        var output = ""
        for clause in clauses {
            let key = clauseKey(clause.text)
            let words = key.split(separator: " ").count
            if words >= 4 && recent.contains(key) { continue }
            output += clause.text + clause.separator
            if words >= 4 {
                recent.append(key)
                if recent.count > 3 { recent.removeFirst() }
            }
        }
        return output.split(whereSeparator: \.isWhitespace).joined(separator: " ")
    }

    private static func clauseKey(_ clause: String) -> String {
        var key = clause.lowercased()
            .replacingOccurrences(of: "[^\\p{L}\\p{N}\\s']", with: " ", options: .regularExpression)
            .split(whereSeparator: \.isWhitespace).joined(separator: " ")
        for word in ["and ", "so ", "but ", "y ", "pero ", "entonces "] where key.hasPrefix(word) {
            key.removeFirst(word.count)
            break
        }
        return key
    }

    static func isBad(_ text: String) -> Bool {
        let raw = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if raw.count < 2 { return true }
        let lowered = raw.lowercased()
        if badFragments.contains(where: { lowered.contains($0) }) { return true }
        return hallucinations.contains(normalize(raw))
    }

    /// Lowercase, punctuation to spaces, collapsed whitespace (like Python's `_normalize`).
    static func normalize(_ text: String) -> String {
        var scalars = String.UnicodeScalarView()
        for scalar in text.lowercased().unicodeScalars {
            if CharacterSet.alphanumerics.contains(scalar) || scalar == "_" || scalar == "-" {
                scalars.append(scalar)
            } else {
                scalars.append(" ")
            }
        }
        return String(scalars).split(separator: " ").joined(separator: " ")
    }
}

/// Minimal block-level Markdown for the AI panels: headings, bullets and
/// paragraphs. Inline styles (**bold**, *italic*) are rendered by
/// `AttributedString(markdown:)` in the view.
enum MarkdownBlock: Equatable {
    case heading(String)
    case bullet(String)
    case paragraph(String)

    static func parse(_ markdown: String) -> [MarkdownBlock] {
        var blocks: [MarkdownBlock] = []
        for rawLine in markdown.components(separatedBy: .newlines) {
            let line = rawLine.trimmingCharacters(in: .whitespaces)
            if line.isEmpty { continue }
            if line.hasPrefix("#") {
                let title = line.drop(while: { $0 == "#" }).trimmingCharacters(in: .whitespaces)
                if !title.isEmpty { blocks.append(.heading(title)) }
            } else if line.hasPrefix("- ") || line.hasPrefix("* ") || line.hasPrefix("• ") {
                let item = String(line.dropFirst(2)).trimmingCharacters(in: .whitespaces)
                if !item.isEmpty { blocks.append(.bullet(item)) }
            } else {
                blocks.append(.paragraph(line))
            }
        }
        return blocks
    }
}

/// Tells the questions said word for word in the meeting apart from the ones
/// the AI phrased from what it heard (the live summary's key questions can be
/// either).
enum LiteralQuestions {
    /// Words of every question in `text`: the sentence before each "?" (from
    /// the previous . ! ? or the start). Join the transcript blocks first, so a
    /// question split across two blocks is found whole.
    static func questions(in text: String) -> [[String]] {
        var result: [[String]] = []
        var sentence = ""
        for character in text {
            if character == "." || character == "!" || character == "?" {
                if character == "?" {
                    let said = words(sentence)
                    if said.count >= 3 { result.append(said) }
                }
                sentence = ""
            } else {
                sentence.append(character)
            }
        }
        return result
    }

    /// Lowercase words without accents or punctuation ("¿Qué?" → ["que"]).
    static func words(_ text: String) -> [String] {
        text.folding(options: [.caseInsensitive, .diacriticInsensitive], locale: nil)
            .lowercased()
            .split(whereSeparator: { !($0.isLetter || $0.isNumber) })
            .map(String.init)
    }

    /// True if `question` matches the end of a question said in the meeting:
    /// at least 75 % of its words are there, and they are at least half of
    /// that stretch (fillers like "so" or "bueno" before it don't matter).
    static func isLiteral(_ question: String, in said: [[String]]) -> Bool {
        let ordered = words(question)
        let asked = Set(ordered)
        guard asked.count >= 3 else { return false }
        for spoken in said {
            let window = Set(spoken.suffix(ordered.count + 4))
            let common = asked.intersection(window).count
            if Double(common) >= 0.75 * Double(asked.count), Double(common) >= 0.5 * Double(window.count) {
                return true
            }
        }
        return false
    }
}
