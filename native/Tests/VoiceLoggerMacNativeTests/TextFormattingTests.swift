@testable import VoiceLoggerMacNative
import XCTest

final class TextFormattingTests: XCTestCase {
    func testCleanStripsWhisperSpecialTokens() {
        XCTAssertEqual(
            TranscriptCleaner.clean("<|startoftranscript|><|es|><|transcribe|><|0.00|> Hola equipo <|2.40|>"),
            "Hola equipo"
        )
    }

    func testCleanDropsHallucinations() {
        for text in ["", ".", "Thank you.", "Thanks for watching!", "¡Suscríbete!", "[BLANK_AUDIO]",
                     "Subtítulos realizados por la comunidad de Amara.org"] {
            XCTAssertNil(TranscriptCleaner.clean(text), text)
        }
    }

    func testCleanKeepsRealSpeech() {
        XCTAssertEqual(TranscriptCleaner.clean("  Vamos a revisar   el sprint de UPTC "), "Vamos a revisar el sprint de UPTC")
        XCTAssertEqual(TranscriptCleaner.clean("Thank you for the update on the API"), "Thank you for the update on the API")
    }

    func testMarkdownBlocks() {
        let blocks = MarkdownBlock.parse("## Resumen\nTexto **clave**\n- uno\n* dos\n\n### Tareas")
        XCTAssertEqual(blocks, [
            .heading("Resumen"),
            .paragraph("Texto **clave**"),
            .bullet("uno"),
            .bullet("dos"),
            .heading("Tareas"),
        ])
    }

    func testLastMeaningfulLineOfPythonError() {
        let log = "Traceback (most recent call last):\n  File \"x\"\nModuleNotFoundError: No module named 'mlx_lm'\n\n"
        XCTAssertEqual(PythonBridge.lastMeaningfulLine(of: log), "ModuleNotFoundError: No module named 'mlx_lm'")
        XCTAssertNil(PythonBridge.lastMeaningfulLine(of: "\n  \n"))
    }

    func testLineBufferJoinsChunksIntoLines() {
        let buffer = LineBuffer()
        XCTAssertEqual(buffer.append(Data("{\"id\":\"a\"}\n{\"id\"".utf8)), [Data("{\"id\":\"a\"}".utf8)])
        XCTAssertEqual(buffer.append(Data(":\"b\"}\n\n".utf8)), [Data("{\"id\":\"b\"}".utf8)])
        XCTAssertEqual(buffer.append(Data()), [])
    }
}
