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
}
