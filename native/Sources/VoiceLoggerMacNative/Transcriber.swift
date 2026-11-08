import CoreML
import Foundation
import WhisperKit

/// Transcription backend. The Python app's MLX model
/// `mlx-community/whisper-large-v3-turbo-q4` is the quality reference;
/// the native equivalent is WhisperKit's CoreML `large-v3` turbo model.
public protocol Transcriber: Sendable {
    /// - prompt: context for Whisper (vocabulary + the end of what was just said),
    ///   so names stay consistent and phrases cut mid-sentence keep their meaning.
    /// - relaxed: rescue attempt for a block that came back empty: no quality
    ///   gates and no timestamp tokens.
    func transcribe(_ audio: [Float], language: String, prompt: String?, relaxed: Bool) async throws -> (text: String, language: String)
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

    /// Where each part runs: the encoder and decoder on the Neural Engine
    /// (WhisperKit's default) and the small mel spectrogram on the CPU instead
    /// of the GPU, so the GPU stays free for the AI (MLX) and live transcription
    /// does not slow down while the AI writes.
    static let computeOptions = ModelComputeOptions(
        melCompute: .cpuOnly, audioEncoderCompute: .cpuAndNeuralEngine, textDecoderCompute: .cpuAndNeuralEngine
    )
    /// Retries at higher temperature when a block fails Whisper's quality checks
    /// (default 5): each one is a whole new decode, so 5 could add seconds on
    /// music or noise. Empty blocks have their own rescue in the app.
    static let temperatureFallbackCount = 2

    public init(model: String? = nil) async throws {
        let name = model ?? ProcessInfo.processInfo.environment["VOICELOGGER_MODEL"] ?? Self.defaultModel
        let started = Date()
        SessionLog.shared.log("WHISPER", "cargando \(name)…")
        self.pipe = try await WhisperKit(WhisperKitConfig(model: name, computeOptions: Self.computeOptions, verbose: false))
        SessionLog.shared.log(
            "WHISPER", "\(name) listo en \(SessionLog.seconds(Date().timeIntervalSince(started))) · "
                + "mel CPU, codificador y decodificador en Neural Engine · memoria app \(SessionLog.memoryMB()) MB"
        )
    }

    public func transcribe(_ audio: [Float], language: String, prompt: String?, relaxed: Bool) async throws -> (text: String, language: String) {
        await gate.wait()
        defer { gate.signal() }
        let lang: String? = (language == "auto" || language.isEmpty) ? nil : language
        var options = DecodingOptions(language: lang, detectLanguage: lang == nil)
        // Blocks are short (≤ 12 s) and only their text is used: no timestamp
        // tokens to decode (also the cause of some empty blocks in the
        // diagnostics: Whisper predicted a timestamp and stopped).
        options.withoutTimestamps = true
        options.skipSpecialTokens = true
        options.temperatureFallbackCount = Self.temperatureFallbackCount
        if let prompt, !prompt.isEmpty, let tokenizer = pipe.tokenizer {
            // Same recipe as WhisperKit's CLI: text tokens only, no special tokens.
            options.promptTokens = tokenizer.encode(text: " " + prompt)
                .filter { $0 < tokenizer.specialTokens.specialTokenBegin }
        }
        if relaxed {
            // Rescue for a block that came back empty (seen in the diagnostics:
            // clear speech, empty text, even with no-speech disabled). No quality
            // gates and no timestamp tokens: Whisper sometimes predicts a
            // timestamp and ends the block without writing any text.
            options.noSpeechThreshold = 1.0
            options.logProbThreshold = -100
            options.firstTokenLogProbThreshold = -100
            options.compressionRatioThreshold = 100
            options.withoutTimestamps = true
        }
        let results = try await pipe.transcribe(audioArray: audio, decodeOptions: options)
        let text = results.map(\.text).joined(separator: " ").trimmingCharacters(in: .whitespacesAndNewlines)
        if let timings = results.first?.timings {
            let audioSeconds = Double(audio.count) / 16000
            SessionLog.shared.log(
                "WHISPER", String(format: "%.1f s de audio en %@ (×%.0f tiempo real) · codificar %@ · decodificar %@ · "
                    + "reintentos %.0f · %@%@",
                    audioSeconds, SessionLog.seconds(timings.fullPipeline),
                    timings.fullPipeline > 0 ? audioSeconds / timings.fullPipeline : 0,
                    SessionLog.seconds(timings.encoding), SessionLog.seconds(timings.decodingLoop),
                    timings.totalDecodingFallbacks, relaxed ? "rescate" : "normal", text.isEmpty ? " · VACÍO" : "")
            )
        }
        let detected = results.first?.language ?? (lang ?? "en")
        return (text, detected)
    }
}

/// Placeholder kept for tests / offline UI work.
public struct StubTranscriber: Transcriber {
    public init() {}
    public func transcribe(_ audio: [Float], language: String, prompt: String?, relaxed: Bool) async throws -> (text: String, language: String) {
        ("[stub: \(audio.count) samples, WhisperKit pendiente]", language)
    }
}
