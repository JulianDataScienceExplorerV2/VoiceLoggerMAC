import Foundation
import WhisperKit

/// Transcription backend. The Python app's MLX model
/// `mlx-community/whisper-large-v3-turbo-q4` is the quality reference;
/// the native equivalent is WhisperKit's CoreML `large-v3` turbo model.
public protocol Transcriber: Sendable {
    func transcribe(_ audio: [Float], language: String) async throws -> (text: String, language: String)
}

/// WhisperKit (CoreML on Apple Neural Engine) transcriber.
/// The model downloads from Hugging Face on first use and is cached locally;
/// override with the `VOICELOGGER_MODEL` environment variable.
/// Minimal async semaphore: waits happen off the cooperative pool.
final class AsyncSemaphore: @unchecked Sendable {
    private let semaphore = DispatchSemaphore(value: 1)
    func wait() async {
        await withCheckedContinuation { cont in
            DispatchQueue.global(qos: .userInitiated).async {
                self.semaphore.wait()
                cont.resume()
            }
        }
    }
    func signal() { semaphore.signal() }
}

public final class WhisperKitTranscriber: Transcriber, @unchecked Sendable {
    /// Recommended: the full large-v3 turbo (best quality); the 626MB
    /// compressed variant stays as the lighter option.
    public static let defaultModel = "large-v3-v20240930_turbo"

    /// Same idea as Python's `MODEL_CHOICES` in `transcriber.py`:
    /// first option is the recommended turbo model.
    public static let modelChoices: [(label: String, id: String)] = [
        ("Large v3 Turbo — máxima calidad (recomendado) — ~90s", "large-v3-v20240930_turbo"),
        ("Large v3 Turbo 626MB — casi igual, más liviano — ~60s", "large-v3-v20240930_626MB"),
        ("Small (ligero) — ~20s", "small"),
        ("Small English — ~20s", "small.en"),
        ("Base (ultra ligero) — ~10s", "base"),
        ("Base English — ~10s", "base.en"),
        ("Tiny (solo pruebas) — ~2s", "tiny"),
    ]

    private let pipe: WhisperKit
    private let gate = AsyncSemaphore()

    public init(model: String? = nil) async throws {
        let name = model ?? ProcessInfo.processInfo.environment["VOICELOGGER_MODEL"] ?? Self.defaultModel
        self.pipe = try await WhisperKit(WhisperKitConfig(model: name, verbose: false))
    }

    public func transcribe(_ audio: [Float], language: String) async throws -> (text: String, language: String) {
        await gate.wait()
        defer { gate.signal() }
        let lang: String? = (language == "auto" || language.isEmpty) ? nil : language
        let options = DecodingOptions(language: lang, detectLanguage: lang == nil)
        let results = try await pipe.transcribe(audioArray: audio, decodeOptions: options)
        let text = results.map(\.text).joined(separator: " ").trimmingCharacters(in: .whitespacesAndNewlines)
        let detected = results.first?.language ?? (lang ?? "en")
        return (text, detected)
    }
}

/// Placeholder kept for tests / offline UI work.
public struct StubTranscriber: Transcriber {
    public init() {}
    public func transcribe(_ audio: [Float], language: String) async throws -> (text: String, language: String) {
        ("[stub: \(audio.count) samples, WhisperKit pendiente]", language)
    }
}
