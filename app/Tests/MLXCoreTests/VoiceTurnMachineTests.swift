import XCTest
@testable import MLXCore

/// Pins the hands-free voice turn lifecycle: the happy path
/// listening → recognizing → thinking → speaking → listening, plus barge-in,
/// empty answers, errors and closing the mode.
final class VoiceTurnMachineTests: XCTestCase {

    private func reduce(_ s: VoiceTurnState, _ e: VoiceTurnEvent) -> VoiceTurnState {
        VoiceTurnMachine.reduce(s, on: e)
    }

    func testStartOpensListening() {
        XCTAssertEqual(reduce(.idle, .start), .listening)
    }

    func testSpeechStartedMovesToRecognizing() {
        XCTAssertEqual(reduce(.listening, .speechStarted), .recognizing)
    }

    func testFinalTranscriptMovesToThinking() {
        XCTAssertEqual(reduce(.recognizing, .transcriptFinalized), .thinking)
        // A short utterance may finalize before speechStarted ever fired.
        XCTAssertEqual(reduce(.listening, .transcriptFinalized), .thinking)
    }

    func testResponseStartedMovesToSpeaking() {
        XCTAssertEqual(reduce(.thinking, .responseStarted), .speaking)
    }

    func testTurnFinishedReopensListening() {
        XCTAssertEqual(reduce(.speaking, .turnFinished), .listening)
    }

    func testEmptyAnswerFinishesStraightFromThinking() {
        // Model produced nothing speakable; we still loop back to listening.
        XCTAssertEqual(reduce(.thinking, .turnFinished), .listening)
    }

    func testBargeInWhileSpeakingReturnsToListening() {
        XCTAssertEqual(reduce(.speaking, .bargeIn), .listening)
    }

    func testBargeInWhileThinkingReturnsToListening() {
        // Hitting "Stop" while the model is still generating (nothing spoken yet)
        // must drop back to listening, not leave the UI stuck on "Thinking…".
        XCTAssertEqual(reduce(.thinking, .bargeIn), .listening)
    }

    func testDismissedUtteranceKeepsListening() {
        // No wake word in the utterance → drop it and keep the mic open, whether
        // we'd advanced to recognizing or never left listening.
        XCTAssertEqual(reduce(.recognizing, .utteranceDismissed), .listening)
        XCTAssertEqual(reduce(.listening, .utteranceDismissed), .listening)
    }

    func testFailureMovesToError() {
        XCTAssertEqual(reduce(.thinking, .failed("boom")), .error("boom"))
        XCTAssertEqual(reduce(.speaking, .failed("x")), .error("x"))
    }

    func testStopAlwaysReturnsToIdle() {
        XCTAssertEqual(reduce(.speaking, .stop), .idle)
        XCTAssertEqual(reduce(.listening, .stop), .idle)
        XCTAssertEqual(reduce(.error("e"), .stop), .idle)
    }

    func testIrrelevantEventLeavesStateUnchanged() {
        // A late responseStarted while already listening must not corrupt state.
        XCTAssertEqual(reduce(.listening, .responseStarted), .listening)
        XCTAssertEqual(reduce(.idle, .turnFinished), .idle)
    }
}
