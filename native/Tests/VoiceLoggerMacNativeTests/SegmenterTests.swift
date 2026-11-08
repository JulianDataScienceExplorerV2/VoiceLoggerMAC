@testable import VoiceLoggerMacNative
import XCTest

final class SegmenterTests: XCTestCase {
    func testSilenceProducesNothing() {
        var phrases = 0
        let seg = PhraseSegmenter(onPhrase: { _, _ in phrases += 1 })
        for _ in 0..<100 { seg.consume([Float](repeating: 0, count: 480)) }
        seg.flush()
        XCTAssertEqual(phrases, 0)
    }

    func testToneProducesOnePhrase() {
        var phrases: [[Float]] = []
        let seg = PhraseSegmenter(onPhrase: { audio, _ in phrases.append(audio) })
        for _ in 0..<30 { seg.consume([Float](repeating: 0, count: 480)) }
        for i in 0..<50 {
            let v = Float(sin(Double(i) * 0.3)) * 0.2
            seg.consume([Float](repeating: v, count: 480))
        }
        for _ in 0..<30 { seg.consume([Float](repeating: 0, count: 480)) }
        seg.flush()
        XCTAssertEqual(phrases.count, 1)
        XCTAssertGreaterThan(phrases.first?.count ?? 0, 16000)
    }

    func testPrepareIsBounded() {
        let out = PhraseSegmenter.prepare([Float](repeating: 0.9, count: 16000))
        XCTAssertLessThanOrEqual(out.map(abs).max() ?? 0, 0.95)
    }

    /// ScreenCaptureKit buffers have no fixed size: the timings must not depend
    /// on it. 20 s of non-stop speech in 4096-sample buffers must be cut at the
    /// 8 s max phrase (3 phrases), not come out as one late phrase.
    func testBigBuffersKeepRealTiming() {
        var phrases: [[Float]] = []
        let seg = PhraseSegmenter(onPhrase: { audio, _ in phrases.append(audio) })
        let speech = (0..<(16000 * 20)).map { Float(sin(Double($0) * 0.3)) * 0.2 }
        var index = 0
        while index < speech.count {
            let end = min(index + 4096, speech.count)
            seg.consume(Array(speech[index..<end]))
            index = end
        }
        seg.flush()
        XCTAssertGreaterThanOrEqual(phrases.count, 3)
        XCTAssertLessThanOrEqual(phrases.map(\.count).max() ?? 0, 16000 * 8)
        XCTAssertEqual(phrases.map(\.count).reduce(0, +), speech.count) // no audio lost at the cuts
    }

    /// Tiny buffers (smaller than 30 ms) must still detect silence after speech.
    func testTinyBuffersStillCutOnSilence() {
        var phrases = 0
        let seg = PhraseSegmenter(onPhrase: { _, _ in phrases += 1 })
        let speech = (0..<(16000 * 2)).map { Float(sin(Double($0) * 0.3)) * 0.2 }
        for start in stride(from: 0, to: speech.count, by: 100) {
            seg.consume(Array(speech[start..<min(start + 100, speech.count)]))
        }
        for _ in 0..<(16000 / 100) { seg.consume([Float](repeating: 0, count: 100)) }
        XCTAssertEqual(phrases, 1)  // cut by 1 s of silence, before flush
    }

    /// "Rápida" mode: 5 s max phrase → 12 s of non-stop speech in 3 phrases.
    func testFastTimingCutsSooner() {
        var phrases: [[Float]] = []
        let seg = PhraseSegmenter(onPhrase: { audio, _ in phrases.append(audio) })
        seg.setTiming(silence: 0.4, maxSpeech: 5)
        let speech = (0..<(16000 * 12)).map { Float(sin(Double($0) * 0.3)) * 0.2 }
        for start in stride(from: 0, to: speech.count, by: 1024) {
            seg.consume(Array(speech[start..<min(start + 1024, speech.count)]))
        }
        seg.flush()
        XCTAssertGreaterThanOrEqual(phrases.count, 3)
        XCTAssertLessThanOrEqual(phrases.map(\.count).max() ?? 0, 16000 * 5)
        XCTAssertEqual(phrases.map(\.count).reduce(0, +), speech.count)
    }

    /// In continuous speech the max-length cut lands on the short pause between
    /// words (here at 7.2 s), not in the middle of a word at exactly 8 s.
    func testMaxLengthCutsAtThePause() {
        var phrases: [[Float]] = []
        let seg = PhraseSegmenter(onPhrase: { audio, _ in phrases.append(audio) })
        func tone(_ count: Int) -> [Float] { (0..<count).map { Float(sin(Double($0) * 0.3)) * 0.2 } }
        let audio = tone(Int(16000 * 7.2)) + [Float](repeating: 0.0005, count: Int(16000 * 0.12)) + tone(Int(16000 * 4.68))
        for start in stride(from: 0, to: audio.count, by: 1024) {
            seg.consume(Array(audio[start..<min(start + 1024, audio.count)]))
        }
        seg.flush()
        let first = Double(phrases.first?.count ?? 0) / 16000
        XCTAssertGreaterThan(first, 7.1)
        XCTAssertLessThan(first, 7.4)
        XCTAssertEqual(phrases.map(\.count).reduce(0, +), audio.count)
    }
}

