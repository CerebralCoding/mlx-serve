import XCTest
@testable import MLXCore

/// The tray transcript: where a long recording is cut, the mix, the file it writes.
final class TrayTranscriptTests: XCTestCase {
    func testALongRecordingIsCutInItsQuietestStretchBeforeEachLimit() {
        let rate = 100
        var samples = [Float](repeating: 0.5, count: 2500 * rate)
        for i in 1196 * rate ..< 1196 * rate + rate / 2 { samples[i] = 0 }
        let ends = TrayTranscript.pieceEnds(samples, rate: rate)
        XCTAssertEqual(ends.count, 3)
        XCTAssertTrue((1196 * rate ..< 1196 * rate + rate / 2).contains(ends[0]), "\(ends[0])")
        XCTAssertLessThanOrEqual(ends[1] - ends[0], 1200 * rate)
        XCTAssertEqual(ends.last, samples.count)
        XCTAssertEqual(TrayTranscript.pieceEnds([Float](repeating: 0, count: 10 * rate), rate: rate), [10 * rate])
    }

    /// Desktop audio joins where it was ticked and stops with the microphone.
    func testDesktopAudioIsAddedOnTopOfTheMicrophone() {
        XCTAssertEqual(TrayTranscript.mix([0.1, 0.2, 0.3, 0.4], [0, 0.5]), [0.1, 0.7, 0.3, 0.4])
        XCTAssertEqual(TrayTranscript.mix([0.1, 0.2], [0, 0.5, 0.9]), [0.1, 0.7])
        XCTAssertEqual(TrayTranscript.mix([0.1], []), [0.1])
    }

    func testTheFileIsItsStartThenTimedParagraphs() {
        XCTAssertEqual(TrayTranscript.markdown(title: "2026-10-13 18:40:05", segments: [(0, "Hello there."), (65.5, "Bye."), (3725, "Late.")]),
                       "# 2026-10-13 18:40:05\n\n**00:00** Hello there.\n\n**01:05** Bye.\n\n**1:02:05** Late.\n")
        XCTAssertEqual(TrayTranscript.markdown(title: "t", segments: [(0, "Only one.")]), "# t\n\nOnly one.\n")
    }

    /// Live: TRANSCRIPT_SERVER=http://127.0.0.1:<port> TRANSCRIPT_MODEL=<Qwen3-ASR dir> TRANSCRIPT_WAV=<wav>.
    func testARecordingRoundTripsThroughTheServer() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let server = env["TRANSCRIPT_SERVER"], let model = env["TRANSCRIPT_MODEL"], let wav = env["TRANSCRIPT_WAV"] else {
            throw XCTSkip("set TRANSCRIPT_SERVER, TRANSCRIPT_MODEL and TRANSCRIPT_WAV")
        }
        let segments = try await TrayTranscript.transcribe(try Data(contentsOf: URL(fileURLWithPath: wav)), model: model, server: server)
        XCTAssertFalse(segments.isEmpty)
        XCTAssertTrue(segments.map(\.text).joined(separator: " ").lowercased().contains("fox"), "\(segments)")
    }
}
