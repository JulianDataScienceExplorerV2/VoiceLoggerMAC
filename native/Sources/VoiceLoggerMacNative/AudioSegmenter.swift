import Foundation

/// Energy-based voice activity segmenter ported from the Python
/// `PhraseSegmenter` in `audio_capture.py`.
///
/// Samples arrive as 16 kHz mono Float32 buffers of any size and are re-cut
/// into exact 30 ms chunks (like the Python app does) before the VAD runs.
/// Completed phrases are delivered through `onPhrase` as a single Float32
/// buffer plus timestamp. A phrase ends after a pause, or when it reaches the
/// max length: then it is cut at the quietest moment of its last 1.5 s (a gap
/// between words) instead of mid-word, and the rest starts the next phrase, so
/// no audio is lost.
public final class PhraseSegmenter: @unchecked Sendable {
    public var energyThreshold: Float
    public private(set) var silenceTimeoutSec: Double
    let minSpeechSec: Double
    public private(set) var maxSpeechSec: Double
    let chunkSec: Double
    let onPhrase: ([Float], String) -> Void
    let onLevel: (Float) -> Void

    private var speaking = false
    private var streak = 0
    private var current: [Float] = []
    private var preRoll: [[Float]] = []
    private var preRollLevels: [Float] = []
    /// RMS of each 30 ms chunk in `current` (to find a quiet place to cut).
    private var currentLevels: [Float] = []
    /// How far back from the max length to look for that quiet place.
    let cutLookbackSec = 1.5
    private var voiced = 0
    private var silence = 0
    private var timestamp: String?
    /// Samples waiting to complete the next 30 ms chunk.
    private var pending: [Float] = []

    /// Samples per analysis chunk (480 = 30 ms at 16 kHz).
    private var chunkSize: Int { max(1, Int((16000 * chunkSec).rounded())) }

    public init(
        energyThreshold: Float = 0.0035,
        silenceTimeoutSec: Double = 0.65,
        minSpeechSec: Double = 0.5,
        maxSpeechSec: Double = 8.0,
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

    /// Changes how soon phrases are cut (pause length / max phrase length),
    /// even while listening.
    public func setTiming(silence: Double, maxSpeech: Double) {
        silenceTimeoutSec = max(0.2, min(2.0, silence))
        maxSpeechSec = max(2.0, min(20.0, maxSpeech))
    }

    public func setEnergyThreshold(_ threshold: Float) {
        energyThreshold = max(0.0005, min(0.05, threshold))
    }

    public func reset() {
        speaking = false
        streak = 0
        current = []
        currentLevels = []
        preRoll = []
        preRollLevels = []
        voiced = 0
        silence = 0
        timestamp = nil
        pending = []
    }

    /// Feeds audio of any buffer size.
    ///
    /// ScreenCaptureKit buffers have no fixed length, but every VAD timing
    /// (0.65 s of silence to cut, 8 s max phrase) counts 30 ms chunks. Feeding
    /// raw buffers stretched all of them in proportion to the buffer size, so
    /// phrases were cut much later than intended and the live transcript lagged.
    public func consume(_ samples: [Float]) {
        guard !samples.isEmpty else { return }
        pending.append(contentsOf: samples)
        let size = chunkSize
        var offset = 0
        var peak: Float = -1
        while pending.count - offset >= size {
            peak = max(peak, process(Array(pending[offset..<offset + size])))
            offset += size
        }
        if offset > 0 { pending.removeFirst(offset) }
        // One level update per buffer, not per chunk: fewer UI hops.
        if peak >= 0 { onLevel(min(1.0, peak * 40.0)) }
    }

    /// Runs the VAD on one exact 30 ms chunk; returns its RMS level.
    private func process(_ chunk: [Float]) -> Float {
        let rms = Self.rms(chunk)
        let above = rms >= energyThreshold

        if !speaking {
            preRoll.append(chunk)
            preRollLevels.append(rms)
            if preRoll.count > 6 {
                preRoll.removeFirst()
                preRollLevels.removeFirst()
            }
            if above {
                streak += 1
                if streak >= 2 {
                    speaking = true
                    streak = 0
                    silence = 0
                    voiced = preRoll.count
                    current = preRoll.flatMap { $0 }
                    currentLevels = preRollLevels
                    timestamp = Self.nowString()
                }
            } else {
                streak = 0
            }
            return rms
        }

        current.append(contentsOf: chunk)
        currentLevels.append(rms)
        if above {
            voiced += 1
            silence = 0
        } else {
            silence += 1
        }

        let maxSilenceChunks = max(3, Int(silenceTimeoutSec / chunkSec))
        let maxPhraseChunks = max(10, Int(maxSpeechSec / chunkSec))
        let minVoicedChunks = max(2, Int(minSpeechSec / chunkSec))

        if silence >= maxSilenceChunks {
            if voiced >= minVoicedChunks, let ts = timestamp {
                onPhrase(Self.prepare(current), ts)
            }
            endPhrase()
        } else if currentLevels.count >= maxPhraseChunks {
            cutAtQuietestPoint(chunkSize: chunk.count, minVoicedChunks: minVoicedChunks)
        }
        return rms
    }

    /// Max length reached in continuous speech: cut after the quietest chunk of
    /// the last `cutLookbackSec` and keep the rest as the start of the next phrase.
    private func cutAtQuietestPoint(chunkSize: Int, minVoicedChunks: Int) {
        let window = min(currentLevels.count - 1, Int(cutLookbackSec / chunkSec))
        let start = currentLevels.count - window
        var cutChunk = currentLevels.count
        var quietest = Float.greatestFiniteMagnitude
        for index in start..<currentLevels.count where currentLevels[index] < quietest {
            quietest = currentLevels[index]
            cutChunk = index + 1
        }
        let cutSample = min(current.count, cutChunk * chunkSize)
        let head = Array(current[..<cutSample])
        let tail = Array(current[cutSample...])
        let tailLevels = Array(currentLevels[min(cutChunk, currentLevels.count)...])
        if voiced >= minVoicedChunks, let ts = timestamp {
            onPhrase(Self.prepare(head), ts)
        }
        guard !tail.isEmpty else {
            endPhrase()
            return
        }
        current = tail
        currentLevels = tailLevels
        voiced = tailLevels.filter { $0 >= energyThreshold }.count
        silence = tailLevels.reversed().prefix { $0 < energyThreshold }.count
        timestamp = Self.nowString()
    }

    private func endPhrase() {
        speaking = false
        current = []
        currentLevels = []
        voiced = 0
        silence = 0
        preRoll = []
        preRollLevels = []
        timestamp = nil
    }

    public func flush() {
        let minVoicedChunks = max(2, Int(minSpeechSec / chunkSec))
        if speaking { current.append(contentsOf: pending) }
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
