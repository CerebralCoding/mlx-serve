import XCTest
@testable import MLXCore

/// Pins the ghost-turn guard (live capture 2026-07-03): deleting a chat while
/// its agent turn was in flight left the turn running invisibly — every
/// append/update no-ops against the gone session, the empty-response check
/// reads "" and pad-retries with FULL multi-minute generations, and the
/// app-wide engine stays busy so every other surface reports "The model is
/// answering another chat" with no Stop control anywhere. Server stop/restart
/// can't clear it (the turn is app-side). The rule: a turn whose session no
/// longer exists is ORPHANED and must stop — `AppState.deleteSession` stops it
/// immediately, and the agent loop re-checks per iteration as defense in
/// depth for any other session-removal path.
final class OrphanedTurnTests: XCTestCase {

    func testOrphanedWhenActiveTurnSessionIsGone() {
        let turn = UUID()
        let other = UUID()
        XCTAssertTrue(ChatTurnEngine.turnOrphaned(
            isGenerating: true, activeTurnSessionId: turn, sessionIds: [other]))
        XCTAssertTrue(ChatTurnEngine.turnOrphaned(
            isGenerating: true, activeTurnSessionId: turn, sessionIds: []))
    }

    func testNotOrphanedWhileItsSessionStillExists() {
        let turn = UUID()
        let other = UUID()
        XCTAssertFalse(ChatTurnEngine.turnOrphaned(
            isGenerating: true, activeTurnSessionId: turn, sessionIds: [turn, other]))
    }

    /// An idle engine is never orphaned — `activeTurnSessionId` deliberately
    /// keeps its last value after a turn ends (composerState gates on
    /// `isGenerating`), so the check must too.
    func testIdleEngineIsNeverOrphaned() {
        let stale = UUID()
        XCTAssertFalse(ChatTurnEngine.turnOrphaned(
            isGenerating: false, activeTurnSessionId: stale, sessionIds: []))
        XCTAssertFalse(ChatTurnEngine.turnOrphaned(
            isGenerating: true, activeTurnSessionId: nil, sessionIds: []))
    }
}
