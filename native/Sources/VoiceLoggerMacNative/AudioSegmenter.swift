import Foundation

/// Energy-based voice activity segmenter ported from the Python
/// `PhraseSegmenter` in `audio_capture.py`.
///
/// Samples arrive as 16 kHz mono Float32 chunks. Completed phrases are
/// delivered through `onPhrase` as a single Float32 buffer plus timestamp.
public final class PhraseSegmenter: @unchecked Sendable {
    public var energyThreshold: Float
    let silenceTimeoutSec: Double
    let minSpeechSec: Double
    let maxSpeechSec: Double
    let chunkSec: Double
    let onPhrase: ([Float], String) -> Void
    let onLevel: (Float) -> Void

    private var speaking = false
    private var streak = 0
    private var current: [Float] = []
    private var preRoll: [[Float]] = []
    private var voiced = 0
    private var silence = 0
    private var timestamp: String?

    public init(
        energyThreshold: Float = 0.0035,
        silenceTimeoutSec: Double = 0.65,
        minSpeechSec: Double = 0.5,
        maxSpeechSec: Double = 10.0,
        chunkSec: Double = 0.03,
        onPhrase: @escaping ([Float], String) -> Void = { _, _ in },
        onLevel: @escaping (Float) -> Void = { _ in }
    ) {
        self.energyThreshold = energyThreshold
        self.silenceTimeoutSec = silenceTimeoutSec
        self.minSpeechSec = minSpeechSec
        self.maxSpeechSec = maxSpeechSec
        self.chunkSec = chunkSec
        self.onPhrase = onPhrase
        self.onLevel = onLevel
    }

    public func setEnergyThreshold(_ threshold: Float) {
        energyThreshold = max(0.0005, min(0.05, threshold))
    }

    public func reset() {
        speaking = false
        streak = 0
        current = []
        preRoll = []
        voiced = 0
        silence = 0
        timestamp = nil
    }

    public func consume(_ chunk: [Float]) {
        guard !chunk.isEmpty else { return }
        let rms = Self.rms(chunk)
        onLevel(min(1.0, rms * 40.0))
        let above = rms >= energyThreshold

        if !speaking {
            preRoll.append(chunk)
            if preRoll.count > 6 { preRoll.removeFirst() }
            if above {
                streak += 1
                if streak >= 2 {
                    speaking = true
                    streak = 0
                    silence = 0
                    voiced = preRoll.count
                    current = preRoll.flatMap { $0 }
                    timestamp = Self.nowString()
                }
            } else {
                streak = 0
            }
            return
        }

        current.append(contentsOf: chunk)
        if above {
            voiced += 1
            silence = 0
        } else {
            silence += 1
        }

        let maxSilenceChunks = max(3, Int(silenceTimeoutSec / chunkSec))
        let maxPhraseChunks = max(10, Int(maxSpeechSec / chunkSec))
        let minVoicedChunks = max(2, Int(minSpeechSec / chunkSec))
        let chunkSamples = chunk.count

        if silence >= maxSilenceChunks || current.count / max(chunkSamples, 1) >= maxPhraseChunks {
            if voiced >= minVoicedChunks, let ts = timestamp {
                onPhrase(Self.prepare(current), ts)
            }
            speaking = false
            current = []
            voiced = 0
            silence = 0
            preRoll = []
            timestamp = nil
        }
    }

    public func flush() {
        let minVoicedChunks = max(2, Int(minSpeechSec / chunkSec))
        if speaking, !current.isEmpty, voiced >= minVoicedChunks, let ts = timestamp {
            onPhrase(Self.prepare(current), ts)
        }
        reset()
    }

    static func rms(_ samples: [Float]) -> Float {
        guard !samples.isEmpty else { return 0 }
        let sum = samples.reduce(0.0) { $0 + Double($1 * $1) }
        return Float(sqrt(sum / Double(samples.count)))
    }

    static func prepare(_ audio: [Float]) -> [Float] {
        guard !audio.isEmpty else { return audio }
        let mean = audio.reduce(0.0) { $0 + Double($1) } / Double(audio.count)
        var out = audio.map { $0 - Float(mean) }
        let r = rms(out)
        if r > 1e-4 {
            let gain = min(0.07 / r, 10.0)
            out = out.map { $0 * gain }
        }
        return out.map { max(-0.95, min(0.95, $0)) }
    }

    static func nowString() -> String {
        let f = DateFormatter()
        f.dateFormat = "HH:mm:ss"
        return f.string(from: Date())
    }
}
