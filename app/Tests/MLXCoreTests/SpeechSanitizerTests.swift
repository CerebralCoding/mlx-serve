import XCTest
@testable import MLXCore

/// Pins the Markdown→spoken-text rules for voice mode: emphasis, headings, list
/// markers and links are reduced to plain words; fenced code is replaced by a
/// short spoken placeholder so the synthesizer never reads code character by
/// character; whitespace is tidied.
final class SpeechSanitizerTests: XCTestCase {

    func testStripsBoldAndItalic() {
        XCTAssertEqual(SpeechSanitizer.spokenText(from: "**Hello** _world_"), "Hello world")
        XCTAssertEqual(SpeechSanitizer.spokenText(from: "a *quick* __brown__ fox"), "a quick brown fox")
    }

    func testStripsStrikethroughKeepingText() {
        XCTAssertEqual(SpeechSanitizer.spokenText(from: "~~no~~ yes"), "no yes")
    }

    func testStripsHeadingMarkers() {
        XCTAssertEqual(SpeechSanitizer.spokenText(from: "# Title"), "Title")
        XCTAssertEqual(SpeechSanitizer.spokenText(from: "### Deep heading"), "Deep heading")
    }

    func testKeepsLinkLabelDropsURL() {
        XCTAssertEqual(SpeechSanitizer.spokenText(from: "see [Anthropic](https://anthropic.com) now"),
                       "see Anthropic now")
    }

    func testInlineCodeKeepsInnerText() {
        XCTAssertEqual(SpeechSanitizer.spokenText(from: "run `ls -la` here"), "run ls -la here")
    }

    func testFencedCodeBecomesPlaceholder() {
        let md = "before\n```swift\nlet x = 1\nprint(x)\n```\nafter"
        let out = SpeechSanitizer.spokenText(from: md)
        XCTAssertTrue(out.contains("before"), out)
        XCTAssertTrue(out.contains("after"), out)
        XCTAssertTrue(out.contains(SpeechSanitizer.codePlaceholder), out)
        XCTAssertFalse(out.contains("let x = 1"), out)
        XCTAssertFalse(out.contains("```"), out)
    }

    func testStripsLeadingListMarkers() {
        XCTAssertEqual(SpeechSanitizer.spokenText(from: "- item one\n- item two"),
                       "item one\nitem two")
        XCTAssertEqual(SpeechSanitizer.spokenText(from: "1. first\n2. second"),
                       "first\nsecond")
    }

    func testStripsBlockquoteMarker() {
        XCTAssertEqual(SpeechSanitizer.spokenText(from: "> quoted line"), "quoted line")
    }

    func testCollapsesRunsOfSpacesAndTrims() {
        XCTAssertEqual(SpeechSanitizer.spokenText(from: "  a     b   "), "a b")
    }

    func testPreservesNewlinesAsSoftBoundaries() {
        // Headings/list items become separate utterances downstream, so newlines
        // between blocks must survive sanitization.
        XCTAssertEqual(SpeechSanitizer.spokenText(from: "# Title\nBody text"), "Title\nBody text")
    }

    func testReplacesBareURLWithSpokenWord() {
        XCTAssertEqual(SpeechSanitizer.spokenText(from: "see https://example.com/x?y=1 now"),
                       "see the link now")
        XCTAssertEqual(SpeechSanitizer.spokenText(from: "go to www.apple.com today"),
                       "go to the link today")
    }

    func testMarkdownLinkStillBecomesLabelNotTheLink() {
        // Link label handling runs first, so a Markdown link keeps its label and
        // never falls through to the bare-URL replacement.
        XCTAssertEqual(SpeechSanitizer.spokenText(from: "see [Anthropic](https://anthropic.com) now"),
                       "see Anthropic now")
    }

    func testReplacesEmailWithSpokenWord() {
        XCTAssertEqual(SpeechSanitizer.spokenText(from: "email me at bob@example.com please"),
                       "email me at the email address please")
    }

    func testIsIdempotent() {
        let inputs = ["**Hello** _world_", "# Title\nBody", "see [x](http://y) now", "run `ls` ok"]
        for s in inputs {
            let once = SpeechSanitizer.spokenText(from: s)
            XCTAssertEqual(SpeechSanitizer.spokenText(from: once), once, "not idempotent for: \(s)")
        }
    }
}
