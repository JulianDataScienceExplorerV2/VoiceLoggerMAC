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

    /// Returns the cleaned text, or nil when nothing useful remains.
    static func clean(_ raw: String) -> String? {
        var text = raw.replacingOccurrences(of: "<\\|[^|>]*\\|>", with: " ", options: .regularExpression)
        text = text.replacingOccurrences(
            of: "\\[(BLANK_AUDIO|MUSIC|NO_SPEECH|SILENCE)\\]",
            with: " ",
            options: [.regularExpression, .caseInsensitive]
        )
        text = text.split(whereSeparator: \.isWhitespace).joined(separator: " ")
        return isBad(text) ? nil : text
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
