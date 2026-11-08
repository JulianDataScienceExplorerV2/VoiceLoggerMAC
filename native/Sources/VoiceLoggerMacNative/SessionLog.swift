import Foundation

/// General log of each session, to see what happens: Whisper and AI timings,
/// model loads, optimizations, web searches, memory and errors.
///
/// One plain-text file per session in `logs/sesion_AAAA-MM-DD_HH-mm-ss.log`:
/// a new one when the app opens and every time "Iniciar" is pressed. It never
/// stores what is said in the meeting (no transcript, questions or answers),
/// only times, sizes and counts, so it can be shared to debug.
final class SessionLog: @unchecked Sendable {
    static let shared = SessionLog()

    private let queue = DispatchQueue(label: "voicelogger.session-log", qos: .utility)
    private var handle: FileHandle?
    private var currentURL: URL?

    let directory: URL = {
        let support = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        return support.appendingPathComponent("VoiceLoggerMac/logs", isDirectory: true)
    }()

    private static func stamp(_ format: String) -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = format
        return formatter.string(from: Date())
    }

    /// The file being written, for "Abrir log de la sesión".
    var url: URL? { queue.sync { currentURL } }

    /// Starts a new file with a header (machine and settings).
    func begin(_ title: String, settings: [String: String] = [:]) {
        let name = "sesion_\(Self.stamp("yyyy-MM-dd_HH-mm-ss")).log"
        queue.async {
            try? self.handle?.close()
            self.handle = nil
            try? FileManager.default.createDirectory(at: self.directory, withIntermediateDirectories: true)
            let url = self.directory.appendingPathComponent(name)
            FileManager.default.createFile(atPath: url.path, contents: nil)
            self.handle = try? FileHandle(forWritingTo: url)
            self.currentURL = url
            var header = [
                "=== VoiceLogger Mac · \(title) · \(Self.stamp("yyyy-MM-dd HH:mm:ss")) ===",
                "macOS \(ProcessInfo.processInfo.operatingSystemVersionString) · \(Self.chip()) · "
                    + "RAM \(ProcessInfo.processInfo.physicalMemory / 1_073_741_824) GB · "
                    + "app \(Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "dev")",
            ]
            for key in settings.keys.sorted() { header.append("  \(key): \(settings[key] ?? "")") }
            self.write(header.joined(separator: "\n") + "\n")
        }
    }

    /// One line: "12:03:41.123 [IA] message".
    func log(_ category: String, _ message: String) {
        let line = "\(Self.stamp("HH:mm:ss.SSS")) [\(category)] \(message)\n"
        queue.async {
            if self.handle == nil { self.openDefault() }
            self.write(line)
        }
    }

    private func openDefault() {
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let url = directory.appendingPathComponent("sesion_\(Self.stamp("yyyy-MM-dd_HH-mm-ss")).log")
        FileManager.default.createFile(atPath: url.path, contents: nil)
        handle = try? FileHandle(forWritingTo: url)
        currentURL = url
    }

    private func write(_ text: String) {
        guard let handle, let data = text.data(using: .utf8) else { return }
        _ = try? handle.seekToEnd()
        try? handle.write(contentsOf: data)
    }

    // MARK: Helpers

    /// Memory the app uses now (what Activity Monitor shows), in MB.
    static func memoryMB() -> Int {
        var info = rusage_info_v4()
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) {
                proc_pid_rusage(getpid(), RUSAGE_INFO_V4, $0)
            }
        }
        return result == 0 ? Int(info.ri_phys_footprint / 1_048_576) : 0
    }

    private static func chip() -> String {
        var size = 0
        sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0)
        guard size > 0 else { return "Mac" }
        var buffer = [CChar](repeating: 0, count: size)
        sysctlbyname("machdep.cpu.brand_string", &buffer, &size, nil, 0)
        return String(cString: buffer)
    }

    /// "1.23 s" / "450 ms".
    static func seconds(_ value: Double) -> String {
        value < 1 ? "\(Int((value * 1000).rounded())) ms" : String(format: "%.2f s", value)
    }
}
