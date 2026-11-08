import Foundation

/// Writes transcripts in the same TXT/JSONL format as the Python
/// `TranscriptLogger` (`logs/transcripcion_YYYY-MM-DD.{txt,jsonl}`),
/// so the native app and the Python app can share a logs folder.
public final class TranscriptLogger: @unchecked Sendable {
    public let logsDirectory: URL
    private let lock = NSLock()
    private var currentDate: String?
    private var txtURL: URL?
    private var jsonlURL: URL?

    public init(logsDirectory: URL? = nil) {
        if let logsDirectory {
            self.logsDirectory = logsDirectory
        } else {
            let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
            self.logsDirectory = base.appendingPathComponent("VoiceLoggerMac/logs", isDirectory: true)
        }
        try? FileManager.default.createDirectory(at: self.logsDirectory, withIntermediateDirectories: true)
    }

    @discardableResult
    public func log(text: String, model: String = "native", lang: String = "en", timestamp: String? = nil) -> [String: String]? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        let ts = timestamp ?? PhraseSegmenter.nowString()
        let entry: [String: String] = [
            "timestamp": ts,
            "iso": ISO8601DateFormatter().string(from: Date()),
            "lang": lang.uppercased(),
            "model": model,
            "text": trimmed,
        ]
        lock.lock()
        defer { lock.unlock() }
        let (txt, jsonl) = fileURLs()
        Self.append("[\(ts)] [\(entry["lang"]!)] \(trimmed)\n", to: txt)
        if let data = try? JSONSerialization.data(withJSONObject: entry),
           var s = String(data: data, encoding: .utf8) {
            s += "\n"
            Self.append(s, to: jsonl)
        }
        return entry
    }

    private func fileURLs() -> (URL, URL) {
        let today = {
            let f = DateFormatter()
            f.dateFormat = "yyyy-MM-dd"
            return f.string(from: Date())
        }()
        if today != currentDate || txtURL == nil {
            currentDate = today
            txtURL = logsDirectory.appendingPathComponent("transcripcion_\(today).txt")
            jsonlURL = logsDirectory.appendingPathComponent("transcripcion_\(today).jsonl")
            if let txtURL, !FileManager.default.fileExists(atPath: txtURL.path) {
                try? "=== Meeting Log - \(today) ===\n\n".write(to: txtURL, atomically: false, encoding: .utf8)
            }
        }
        return (txtURL!, jsonlURL!)
    }

    private static func append(_ text: String, to url: URL) {
        if !FileManager.default.fileExists(atPath: url.path) {
            try? text.write(to: url, atomically: false, encoding: .utf8)
            return
        }
        guard let handle = try? FileHandle(forWritingTo: url) else { return }
        defer { try? handle.close() }
        try? handle.seekToEnd()
        if let data = text.data(using: .utf8) {
            try? handle.write(contentsOf: data)
        }
    }
}
