import XCTest
@testable import MLXCore

// Regression tests for the agent dead-loop where a model emits an unresolvable
// tool name (e.g. Gemma 4 12B's `shell:` with a trailing colon) and the loop
// grinds to the iteration cap on "Unknown tool 'shell:'".
//
// Two defenses:
//   1. `canonicalToolName` — normalize common name quirks before dispatch.
//   2. `StuckDetector` — bail the loop after consecutive no-progress rounds.
@MainActor
final class AgentToolResolutionTests: XCTestCase {

    // MARK: - canonicalToolName

    func testStripsTrailingColon() {
        XCTAssertEqual(AgentEngine.canonicalToolName("shell:"), "shell")
    }

    func testStripsWhitespaceAndRepeatedColon() {
        XCTAssertEqual(AgentEngine.canonicalToolName("  shell : "), "shell")
        XCTAssertEqual(AgentEngine.canonicalToolName("shell::"), "shell")
    }

    func testStripsNamespacePrefix() {
        XCTAssertEqual(AgentEngine.canonicalToolName("functions.shell"), "shell")
        XCTAssertEqual(AgentEngine.canonicalToolName("tool.writeFile"), "writeFile")
    }

    func testCaseInsensitiveMatchToKnownTool() {
        XCTAssertEqual(AgentEngine.canonicalToolName("Shell"), "shell")
        XCTAssertEqual(AgentEngine.canonicalToolName("WRITEFILE"), "writeFile")
    }

    func testCleanNameUnchanged() {
        XCTAssertEqual(AgentEngine.canonicalToolName("cwd"), "cwd")
        XCTAssertEqual(AgentEngine.canonicalToolName("searchFiles"), "searchFiles")
    }

    func testUnknownNameReturnedCleanedNotMatched() {
        // No known tool matches — return the cleaned name (colon stripped).
        XCTAssertEqual(AgentEngine.canonicalToolName("frobnicate:"), "frobnicate")
    }

    func testLeakedColonResolvesToRealToolKind() {
        // The crux of the bug: a `shell:` name must resolve to a real tool.
        XCTAssertEqual(AgentToolKind(rawValue: AgentEngine.canonicalToolName("shell:")), .shell)
        // And the old exact-match path it replaces would NOT have:
        XCTAssertNil(AgentToolKind(rawValue: "shell:"))
    }

    // MARK: - StuckDetector

    func testStuckAfterConsecutiveFailures() {
        var d = AgentEngine.StuckDetector()
        for _ in 0..<AgentEngine.StuckDetector.limit {
            XCTAssertFalse(d.isStuck)
            d.record(outputs: ["Error: Unknown tool 'shell:'"])
        }
        XCTAssertTrue(d.isStuck)
    }

    func testSuccessResetsCounter() {
        var d = AgentEngine.StuckDetector()
        d.record(outputs: ["Error: Unknown tool 'x'"])
        d.record(outputs: ["Error: Unknown tool 'x'"])
        d.record(outputs: ["Changed working directory to /tmp"]) // a success
        XCTAssertEqual(d.consecutiveNoProgress, 0)
        XCTAssertFalse(d.isStuck)
    }

    func testBlockedOutputCountsAsFailure() {
        var d = AgentEngine.StuckDetector()
        for _ in 0..<AgentEngine.StuckDetector.limit {
            d.record(outputs: ["BLOCKED: shell has been called too many times with the same arguments."])
        }
        XCTAssertTrue(d.isStuck)
    }

    func testWarningWrappedErrorCountsAsFailure() {
        // applyWarning prepends "WARNING:" to a repeated call's output, so the
        // failure marker is no longer at the start — must still be detected.
        var d = AgentEngine.StuckDetector()
        let wrapped = "WARNING: shell: called 5 times recently.\n\nError: Unknown tool 'shell:'"
        for _ in 0..<AgentEngine.StuckDetector.limit {
            d.record(outputs: [wrapped])
        }
        XCTAssertTrue(d.isStuck)
    }

    func testMixedRoundWithOneSuccessIsProgress() {
        var d = AgentEngine.StuckDetector()
        for _ in 0..<10 {
            d.record(outputs: ["Error: x", "Changed working directory to /tmp"])
        }
        XCTAssertFalse(d.isStuck)
    }

    func testEmptyRoundDoesNotChangeCounter() {
        var d = AgentEngine.StuckDetector()
        d.record(outputs: ["Error: x"])
        d.record(outputs: []) // no tools ran this round
        XCTAssertEqual(d.consecutiveNoProgress, 1)
    }
}
