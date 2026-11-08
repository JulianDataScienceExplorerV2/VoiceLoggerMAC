import Foundation

/// Runs the repo's Python helpers (`summarizer.py`, Qwen via mlx-lm) from the
/// native app. The summary / Q&A model still lives in Python.
///
/// The repo root is located at runtime (no hardcoded user paths):
/// 1. `VOICELOGGER_ROOT` environment variable
/// 2. `python_root` in UserDefaults (`defaults write VoiceLoggerMacNative python_root /ruta`)
/// 3. walking up from the working directory and from the executable
///    (`swift run` inside `native/` finds the repo root this way)
/// 4. a few usual clone locations in the home folder
struct PythonBridge: Sendable {
    let root: URL
    let python: URL

    struct Failure: LocalizedError {
        let message: String
        var errorDescription: String? { message }
    }

    static let missingMessage =
        "No encuentro summarizer.py con su .venv. Ejecuta ./setup.sh en la raíz del repo o define VOICELOGGER_ROOT."

    static func locate() -> PythonBridge? {
        let fm = FileManager.default
        var candidates: [URL] = []

        if let env = ProcessInfo.processInfo.environment["VOICELOGGER_ROOT"], !env.isEmpty {
            candidates.append(URL(fileURLWithPath: env, isDirectory: true))
        }
        if let saved = UserDefaults.standard.string(forKey: "python_root"), !saved.isEmpty {
            candidates.append(URL(fileURLWithPath: saved, isDirectory: true))
        }
        let starts = [
            URL(fileURLWithPath: fm.currentDirectoryPath, isDirectory: true),
            Bundle.main.executableURL?.resolvingSymlinksInPath().deletingLastPathComponent(),
        ].compactMap { $0 }
        for start in starts {
            var dir = start.standardizedFileURL
            for _ in 0..<8 {
                candidates.append(dir)
                let parent = dir.deletingLastPathComponent()
                if parent.path == dir.path { break }
                dir = parent
            }
        }
        let home = fm.homeDirectoryForCurrentUser
        for name in ["Desktop/VoiceLoggerMac", "Desktop/VoiceLoggerMAC", "VoiceLoggerMac", "VoiceLoggerMAC"] {
            candidates.append(home.appendingPathComponent(name, isDirectory: true))
        }

        for root in candidates {
            let script = root.appendingPathComponent("summarizer.py")
            let python = root.appendingPathComponent(".venv/bin/python")
            if fm.fileExists(atPath: script.path), fm.isExecutableFile(atPath: python.path) {
                return PythonBridge(root: root, python: python)
            }
        }
        return nil
    }

    /// Runs `summarizer.py <arguments>` feeding `input` through stdin.
    ///
    /// stdin/stdout/stderr go through temp files instead of pipes: the first
    /// model download prints long progress bars to stderr, and a full pipe
    /// that nobody reads would hang the child process.
    func run(_ arguments: [String], input: String) async throws -> String {
        let fm = FileManager.default
        let workDir = fm.temporaryDirectory
            .appendingPathComponent("voicelogger-\(UUID().uuidString)", isDirectory: true)
        try fm.createDirectory(at: workDir, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: workDir) }

        let inURL = workDir.appendingPathComponent("stdin.txt")
        let outURL = workDir.appendingPathComponent("stdout.txt")
        let errURL = workDir.appendingPathComponent("stderr.txt")
        try Data(input.utf8).write(to: inURL)
        fm.createFile(atPath: outURL.path, contents: nil)
        fm.createFile(atPath: errURL.path, contents: nil)

        let process = Process()
        process.executableURL = python
        process.arguments = [root.appendingPathComponent("summarizer.py").path] + arguments
        process.currentDirectoryURL = root
        var environment = ProcessInfo.processInfo.environment
        environment["PYTHONIOENCODING"] = "utf-8"
        environment["TOKENIZERS_PARALLELISM"] = "false"
        process.environment = environment
        process.standardInput = try FileHandle(forReadingFrom: inURL)
        process.standardOutput = try FileHandle(forWritingTo: outURL)
        process.standardError = try FileHandle(forWritingTo: errURL)

        let status: Int32 = try await withCheckedThrowingContinuation { continuation in
            process.terminationHandler = { finished in
                continuation.resume(returning: finished.terminationStatus)
            }
            do {
                try process.run()
            } catch {
                process.terminationHandler = nil
                continuation.resume(throwing: Failure(
                    message: "No se pudo lanzar Python: \(error.localizedDescription)"
                ))
            }
        }

        let output = ((try? String(contentsOf: outURL, encoding: .utf8)) ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard status == 0 else {
            let log = (try? String(contentsOf: errURL, encoding: .utf8)) ?? ""
            throw Failure(message: Self.lastMeaningfulLine(of: log) ?? "Python terminó con código \(status)")
        }
        guard !output.isEmpty else {
            throw Failure(message: "El modelo no devolvió texto.")
        }
        return output
    }

    static func lastMeaningfulLine(of log: String) -> String? {
        let lines = log.split(whereSeparator: \.isNewline)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
        guard let last = lines.last else { return nil }
        return String(last.prefix(300))
    }
}
