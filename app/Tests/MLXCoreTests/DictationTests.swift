import XCTest
@testable import MLXCore

/// Join rule for dictated utterances landing in the Voice tab's text editor:
/// utterances arrive one per silence gap and must concatenate into natural
/// prose — single-space joins, no leading space in an empty editor, no double
/// spacing after existing whitespace, and blank utterances are dropped.
final class DictationTests: XCTestCase {

    func testFirstUtteranceIntoEmptyTextHasNoLeadingSpace() {
        XCTAssertEqual(Dictation.appending("Hello there.", to: ""), "Hello there.")
    }

    func testUtterancesJoinWithASingleSpace() {
        XCTAssertEqual(Dictation.appending("Second thought.", to: "First thought."),
                       "First thought. Second thought.")
    }

    func testNoExtraSpaceAfterTrailingWhitespaceOrNewline() {
        XCTAssertEqual(Dictation.appending("continues", to: "Line one\n"), "Line one\ncontinues")
        XCTAssertEqual(Dictation.appending("continues", to: "Trailing space "), "Trailing space continues")
    }

    func testBlankUtterancesLeaveTextUntouched() {
        XCTAssertEqual(Dictation.appending("", to: "Keep me."), "Keep me.")
        XCTAssertEqual(Dictation.appending("   \n", to: "Keep me."), "Keep me.")
    }

    func testUtteranceWhitespaceIsTrimmedBeforeJoining() {
        XCTAssertEqual(Dictation.appending("  padded words  ", to: "Text."), "Text. padded words")
    }
}
