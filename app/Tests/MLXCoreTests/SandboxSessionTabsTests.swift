import XCTest
@testable import MLXCore

/// The Sandbox window's multi-session tab model — pure, no SwiftTerm, no VM.
/// Several pi/hermes/shell sessions run concurrently (each its own ssh into
/// the shared guest); tabs select which terminal is visible, and terminals
/// are NEVER unmounted by tab or Terminal/Activity switches (that killed the
/// live session — the original bug this model replaces).
final class SandboxSessionTabsTests: XCTestCase {

    func testAddPreparingSelectsTheNewTabAndTitlesTheWindow() {
        var m = SandboxSessionTabs()
        XCTAssertEqual(m.windowTitle, "MLX Sandbox")
        let id = m.addPreparing(label: "pi")
        XCTAssertEqual(m.selectedID, id)
        XCTAssertEqual(m.windowTitle, "pi — MLX Sandbox")
        m.markLive(id)
        XCTAssertEqual(m.selected?.phase, .live)
        XCTAssertEqual(m.windowTitle, "pi — MLX Sandbox")
    }

    func testConcurrentSameAgentTabsGetStableNumberedNames() {
        var m = SandboxSessionTabs()
        let a = m.addPreparing(label: "pi")
        let b = m.addPreparing(label: "pi")
        let c = m.addPreparing(label: "hermes")
        XCTAssertEqual(m.displayName(a), "pi")
        XCTAssertEqual(m.displayName(b), "pi 2")
        XCTAssertEqual(m.displayName(c), "hermes")
        // Names are assigned at creation and NEVER renumber — "pi 2" turning
        // into "pi" mid-session would gaslight the user about which is which.
        m.close(a)
        XCTAssertEqual(m.displayName(b), "pi 2")
        let d = m.addPreparing(label: "pi")
        XCTAssertEqual(m.displayName(d), "pi 3")
    }

    func testWindowTitleUsesTheSelectedTabsDisplayName() {
        var m = SandboxSessionTabs()
        _ = m.addPreparing(label: "pi")
        let b = m.addPreparing(label: "pi")
        m.markLive(b)
        m.select(b)
        XCTAssertEqual(m.windowTitle, "pi 2 — MLX Sandbox")
    }

    func testExitKeepsTheTabWithAnHonestNoticeAndReleasesTheTitle() {
        var m = SandboxSessionTabs()
        let id = m.addPreparing(label: "hermes")
        m.markLive(id)
        m.markExited(id, exitCode: 7)
        XCTAssertEqual(m.tabs.count, 1, "an exited tab stays until closed — its notice explains what happened")
        XCTAssertEqual(m.exitNotice(id), "hermes session ended (exit 7)")
        XCTAssertEqual(m.windowTitle, "MLX Sandbox", "a dead session no longer owns the window")
        // Clean exit / IO-layer death (nil) stay calm.
        let b = m.addPreparing(label: "pi")
        m.markExited(b, exitCode: 0)
        XCTAssertEqual(m.exitNotice(b), "pi session ended")
        let c = m.addPreparing(label: "pi")
        m.markExited(c, exitCode: nil)
        XCTAssertEqual(m.exitNotice(c), "pi 2 session ended")
    }

    func testCloseSelectsANeighborNeverNothingWhileTabsRemain() {
        var m = SandboxSessionTabs()
        let a = m.addPreparing(label: "pi")
        let b = m.addPreparing(label: "hermes")
        let c = m.addPreparing(label: "shell")
        m.select(b)
        m.close(b)
        XCTAssertNotNil(m.selectedID)
        XCTAssertTrue([a, c].contains(m.selectedID!), "selection must move to a surviving neighbor")
        m.close(a); m.close(c)
        XCTAssertNil(m.selectedID)
        XCTAssertTrue(m.tabs.isEmpty)
        XCTAssertEqual(m.windowTitle, "MLX Sandbox")
    }

    func testCloseConfirmationOnlyWhenASessionWouldActuallyDie() {
        // The chip's ✕ is a small target — an accidental click must not kill
        // a live TUI. But an exited tab holds nothing to lose, so closing it
        // stays one click (a confirm there would just be nagging).
        var m = SandboxSessionTabs()
        let a = m.addPreparing(label: "pi")
        XCTAssertTrue(m.closeNeedsConfirmation(a), "preparing: a session is being created — closing abandons it")
        m.markLive(a)
        XCTAssertTrue(m.closeNeedsConfirmation(a), "live: closing terminates the session")
        m.markExited(a, exitCode: 0)
        XCTAssertFalse(m.closeNeedsConfirmation(a), "exited: nothing to lose, no nag")
        XCTAssertFalse(m.closeNeedsConfirmation(UUID()))
    }

    func testMostRecentActiveFindsTheNewestLivingSessionOfAnAgent() {
        // The tray's "pi in Sandbox" shortcut FOCUSES a running pi session
        // instead of stacking a new one — and it must pick a session that is
        // actually alive, never an exited tab that merely kept its notice.
        var m = SandboxSessionTabs()
        XCTAssertNil(m.mostRecentActive(label: "pi"))
        let a = m.addPreparing(label: "pi")
        m.markLive(a)
        let b = m.addPreparing(label: "pi")
        m.markLive(b)
        _ = m.addPreparing(label: "hermes")
        XCTAssertEqual(m.mostRecentActive(label: "pi"), b, "newest living session wins")
        m.markExited(b, exitCode: 0)
        XCTAssertEqual(m.mostRecentActive(label: "pi"), a, "an exited tab is not a focus target")
        m.markExited(a, exitCode: 0)
        XCTAssertNil(m.mostRecentActive(label: "pi"))
        // preparing counts — clicking the shortcut twice while booting must
        // not start a second session.
        let c = m.addPreparing(label: "pi")
        XCTAssertEqual(m.mostRecentActive(label: "pi"), c)
    }

    func testCloseOfUnknownIdIsANoop() {
        var m = SandboxSessionTabs()
        let a = m.addPreparing(label: "pi")
        m.close(UUID())
        XCTAssertEqual(m.tabs.map(\.id), [a])
        XCTAssertEqual(m.selectedID, a)
    }

    // MARK: - Source audit

    /// The Sandbox window presents through exactly ONE `.alert` modifier.
    /// Legacy `.alert(item:)` shadowing is not just a same-node problem: a
    /// modifier on an ANCESTOR also swallows a descendant's presentation —
    /// the close-confirm alert on the session strip (a child of the VStack
    /// carrying the window alert) silently never appeared, so the tab ✕ did
    /// nothing while a session was live. Every dialog this window shows must
    /// ride the single `$alert` item.
    func testSandboxWindowHasExactlyOneAlertPresentationPath() throws {
        let source = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // MLXCoreTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // app
            .appendingPathComponent("Sources/MLXServe/Views/SandboxTerminalView.swift")
        let body = try String(contentsOf: source, encoding: .utf8)
        let alertCount = body.split(separator: "\n", omittingEmptySubsequences: false)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.hasPrefix("//") && $0.contains(".alert(") }
            .count
        XCTAssertEqual(alertCount, 1, """
            SandboxTerminalView must present every dialog through the single \
            window-level `.alert(item: $alert)` — a second legacy alert \
            modifier anywhere on the same ancestor chain is silently shadowed \
            (the ✕-confirm-that-never-showed bug). Found \(alertCount).
            """)
    }
}
