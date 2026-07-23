import XCTest
@testable import MLXCore

/// The Settings restart banner and the per-row dirty dots diff against
/// `liveLaunchedOptions`, not the raw last-launch snapshot: a STOPPED (or
/// errored) server has nothing to restart — edited launch flags simply apply
/// on the next start, so offering "Restart Now" there is a lie (live bite:
/// the banner showed on a stopped server).
@MainActor
final class RestartBannerGateTests: XCTestCase {
    func testRestartOnlyNeededWhileTheProcessIsLive() {
        let mgr = ServerManager()
        mgr.lastLaunchedOptions = ServerOptions()
        var edited = ServerOptions()
        edited.port = 12345

        mgr.status = .stopped
        XCTAssertNil(mgr.liveLaunchedOptions)
        XCTAssertFalse(mgr.needsRestartFor(edited))

        mgr.status = .error("boom")
        XCTAssertFalse(mgr.needsRestartFor(edited))

        mgr.status = .running
        XCTAssertNotNil(mgr.liveLaunchedOptions)
        XCTAssertTrue(mgr.needsRestartFor(edited))

        // A server mid-boot runs the OLD flags too — edits still warrant it.
        mgr.status = .starting
        XCTAssertTrue(mgr.needsRestartFor(edited))

        // Unchanged options never need a restart, live or not.
        mgr.status = .running
        XCTAssertFalse(mgr.needsRestartFor(ServerOptions()))
    }
}
