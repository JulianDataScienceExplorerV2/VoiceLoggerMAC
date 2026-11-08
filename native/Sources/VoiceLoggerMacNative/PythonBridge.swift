import Foundation

/// Locates the repo's Python side (`summarizer.py` + `.venv`, Qwen via mlx-lm).
/// `QwenServer` uses it to start the persistent model process.
///
/// The repo root is located at runtime (no hardcoded user paths):
/// 1. `VOICELOGGER_ROOT` environment variable
/// 2. `python_root` in UserDefaults (`defaults write VoiceLoggerMacNative python_root /ruta`)
///    and `VoiceLoggerRepoRoot` in the packaged app's Info.plist (native/package.sh)
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
        // The packaged app (native/package.sh) records where the repo is.
        if let packaged = Bundle.main.object(forInfoDictionaryKey: "VoiceLoggerRepoRoot") as? String, !packaged.isEmpty {
            candidates.append(URL(fileURLWithPath: packaged, isDirectory: true))
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

    var script: URL { root.appendingPathComponent("summarizer.py") }
    /// Persona / context prepended to every Qwen prompt (read on each request).
    var contextFile: URL { root.appendingPathComponent("qwen_context.txt") }
    var contextExample: URL { root.appendingPathComponent("qwen_context.example.txt") }

    static func lastMeaningfulLine(of log: String) -> String? {
        let lines = log.split(whereSeparator: \.isNewline)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
        guard let last = lines.last else { return nil }
        return String(last.prefix(300))
    }
}
