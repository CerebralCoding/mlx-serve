import XCTest
@testable import MLXCore

final class AgentPromptTests: XCTestCase {
    // The agent must avoid interactive scaffolders (`npx sv create`, etc.) — in
    // the agent's TTY-less shell they fail/loop. The base prompt must steer it
    // toward non-interactive flags or manual setup.
    func testSystemPromptHasScaffoldingGuidance() {
        let p = AgentPrompt.defaultPromptFile
        XCTAssertTrue(p.contains("Scaffolding"), "base prompt is missing a scaffolding section")
        XCTAssertTrue(p.lowercased().contains("interactive"),
                      "base prompt should warn about interactive commands")
        XCTAssertTrue(p.contains("npm init -y") || p.lowercased().contains("non-interactive"),
                      "base prompt should steer toward non-interactive setup")
    }
}
