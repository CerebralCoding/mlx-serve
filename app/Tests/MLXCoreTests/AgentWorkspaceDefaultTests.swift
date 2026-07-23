import XCTest
@testable import MLXCore

/// The default agent workspace (`ChatSession.defaultWorkingDirectory`) is now a
/// SETTING (Settings → Agent Sandbox → workspace row), not a hardcoded path:
/// UserDefaults-backed, falling back to the historical `~/.mlx-serve/workspace`.
/// Everything that anchored on the old static — new sessions, the decode
/// backfill, `AgentSandbox.fallbackSharedRoot`, CLILauncher, MCPManager — must
/// follow the stored value, and changing it in Settings retargets sessions
/// still sitting on the OLD default (so the folder shown in the chat toolbar
/// stays in sync) while never touching a per-session custom pick.
final class AgentWorkspaceDefaultTests: XCTestCase {

    private var suite: UserDefaults!
    private let suiteName = "AgentWorkspaceDefaultTests"

    override func setUp() {
        super.setUp()
        suite = UserDefaults(suiteName: suiteName)
        suite.removePersistentDomain(forName: suiteName)
    }

    override func tearDown() {
        suite.removePersistentDomain(forName: suiteName)
        super.tearDown()
    }

    private func tempPath() -> String {
        (NSTemporaryDirectory() as NSString).appendingPathComponent("awd-\(UUID().uuidString)")
    }

    func testBuiltinDefaultIsTheMlxServeWorkspace() {
        XCTAssertTrue(ChatSession.builtinDefaultWorkingDirectory.hasSuffix("/.mlx-serve/workspace"),
                      "the historical anchor must not move: \(ChatSession.builtinDefaultWorkingDirectory)")
    }

    func testUnsetFallsBackToBuiltin() {
        XCTAssertEqual(ChatSession.defaultWorkingDirectory(defaults: suite),
                       ChatSession.builtinDefaultWorkingDirectory)
    }

    func testStoredDefaultOverridesBuiltinAndCreatesTheFolder() {
        let custom = tempPath()
        defer { try? FileManager.default.removeItem(atPath: custom) }
        ChatSession.setDefaultWorkingDirectory(custom, defaults: suite)
        XCTAssertEqual(ChatSession.defaultWorkingDirectory(defaults: suite), custom)
        XCTAssertTrue(FileManager.default.fileExists(atPath: custom),
                      "the workspace folder must exist so agent tools can use it immediately")
    }

    func testEmptyStoredValueFallsBackToBuiltin() {
        suite.set("", forKey: ChatSession.defaultWorkspaceDefaultsKey)
        XCTAssertEqual(ChatSession.defaultWorkingDirectory(defaults: suite),
                       ChatSession.builtinDefaultWorkingDirectory)
    }

    func testClearingRestoresBuiltin() {
        ChatSession.setDefaultWorkingDirectory(tempPath(), defaults: suite)
        ChatSession.setDefaultWorkingDirectory(nil, defaults: suite)
        XCTAssertEqual(ChatSession.defaultWorkingDirectory(defaults: suite),
                       ChatSession.builtinDefaultWorkingDirectory)
    }

    func testNewSessionAdoptsStoredDefault() {
        // ChatSession() reads the standard defaults — save/restore around it.
        let key = ChatSession.defaultWorkspaceDefaultsKey
        let previous = UserDefaults.standard.string(forKey: key)
        let custom = tempPath()
        defer {
            if let previous { UserDefaults.standard.set(previous, forKey: key) }
            else { UserDefaults.standard.removeObject(forKey: key) }
            try? FileManager.default.removeItem(atPath: custom)
        }
        ChatSession.setDefaultWorkingDirectory(custom)
        XCTAssertEqual(ChatSession().workingDirectory, custom)
    }

    func testRetargetMovesOnlySessionsOnTheOldDefault() {
        let old = "/old/default"
        let new = "/new/default"
        var onDefault = ChatSession(); onDefault.workingDirectory = old
        var customPick = ChatSession(); customPick.workingDirectory = "/my/project"
        var unset = ChatSession(); unset.workingDirectory = nil
        let out = ChatSession.retargeted([onDefault, customPick, unset], from: old, to: new)
        XCTAssertEqual(out[0].workingDirectory, new, "sessions on the old default follow the setting")
        XCTAssertEqual(out[1].workingDirectory, "/my/project", "a per-session pick is never overridden")
        XCTAssertNil(out[2].workingDirectory, "a nil (yolo) wd is not a default to migrate")
    }
}
