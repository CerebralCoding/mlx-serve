import XCTest
@testable import MLXCore

// Regression tests for the agentic shell tool. The original handler had no
// stdin redirection and a kill path that blocked on `readDataToEndOfFile`, so
// an interactive scaffolder (`npx sv create`) could hang the agent for 100s+.
final class ShellHandlerTests: XCTestCase {

    func testEchoSucceeds() async throws {
        let out = try await ShellHandler().execute(parameters: ["command": "echo hello"], workingDirectory: nil)
        XCTAssertTrue(out.contains("hello"), out)
        XCTAssertFalse(out.contains("timed out"), out)
        XCTAssertFalse(out.contains("exit code"), out)
    }

    func testChildStdinIsEmpty() async throws {
        // Contract: the child's stdin is /dev/null, so a command that consumes
        // stdin sees zero bytes (and an interactive prompt would hit EOF).
        let out = try await ShellHandler().execute(parameters: ["command": "wc -c"], workingDirectory: nil)
        // `wc -c` over /dev/null prints 0.
        XCTAssertTrue(out.contains("0"), out)
        XCTAssertFalse(out.contains("timed out"), out)
    }

    func testTimeoutKillsHangingCommandQuickly() async throws {
        let start = Date()
        let out = try await ShellHandler(timeoutSeconds: 2).execute(parameters: ["command": "sleep 20"], workingDirectory: nil)
        let elapsed = Date().timeIntervalSince(start)
        XCTAssertTrue(out.contains("timed out"), out)
        // Must be bounded by the timeout, not the 20s sleep.
        XCTAssertLessThan(elapsed, 10, "shell did not honor the timeout (elapsed \(elapsed)s)")
    }

    func testNonZeroExitReported() async throws {
        let out = try await ShellHandler().execute(parameters: ["command": "exit 7"], workingDirectory: nil)
        XCTAssertTrue(out.contains("[exit code: 7]"), out)
    }

    func testRunsInWorkingDirectory() async throws {
        let tmp = NSTemporaryDirectory()
        let out = try await ShellHandler().execute(parameters: ["command": "pwd"], workingDirectory: tmp)
        let resolved = (tmp as NSString).standardizingPath
        XCTAssertTrue(out.contains(resolved) || out.contains(tmp), out)
    }

    func testMissingCommandThrows() async {
        do {
            _ = try await ShellHandler().execute(parameters: [:], workingDirectory: nil)
            XCTFail("expected missingParameter")
        } catch {
            // expected
        }
    }
}
