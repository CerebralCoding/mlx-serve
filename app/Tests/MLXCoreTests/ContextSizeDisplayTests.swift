import XCTest
@testable import MLXCore

/// Settings → Context size shows THREE different token counts (the model's
/// architectural max, the GPU-safe memory ceiling, and what the running server
/// actually settled on), and it is easy to describe the wrong one.
///
/// Regression: the row read `"Auto" uses the model's declared maximum at load
/// time` while Auto in fact resolved to 78,848 on a model whose declared max is
/// 262,144 — Auto has never used the declared maximum, it fits the model to
/// available memory. The number was also invisible, so a CLI reporting its own
/// context looked inconsistent with the app.
final class ContextSizeDisplayTests: XCTestCase {

    func testFormatTokens() {
        XCTAssertEqual(ContextSizeDisplay.formatTokens(0), "Auto")
        XCTAssertEqual(ContextSizeDisplay.formatTokens(512), "512")
        XCTAssertEqual(ContextSizeDisplay.formatTokens(4096), "4K")
        XCTAssertEqual(ContextSizeDisplay.formatTokens(78848), "77K")   // the live pinned value
        XCTAssertEqual(ContextSizeDisplay.formatTokens(262_144), "256K")
        XCTAssertEqual(ContextSizeDisplay.formatTokens(1_048_576), "1M")
    }

    func testInUseValueOnlyShownOnceTheServerHasReported() {
        XCTAssertNil(ContextSizeDisplay.inUseValue(contextLength: nil))
        XCTAssertNil(ContextSizeDisplay.inUseValue(contextLength: 0))
        XCTAssertEqual(ContextSizeDisplay.inUseValue(contextLength: 78848), "77K")
    }

    func testHelpTextDoesNotClaimAutoUsesTheModelsDeclaredMaximum() {
        let help = ContextSizeDisplay.helpText
        // The false claim that shipped.
        XCTAssertFalse(help.contains("uses the model's declared maximum"),
                       "help text still describes Auto as the model's declared max:\n\(help)")
        // What Auto actually does: fits to memory, then holds still.
        XCTAssertTrue(help.lowercased().contains("memory"), help)
        XCTAssertTrue(help.lowercased().contains("restart"), help)
    }

    /// The same behaviour is described in two places (the Settings row and the
    /// `serverFlagFields` introspection map). They must not drift.
    func testSettingsExplainerMatchesTheRowHelpText() {
        let field = ServerOptions.serverFlagFields["ctxSize"]
        XCTAssertNotNil(field)
        XCTAssertEqual(field?.explainer, ContextSizeDisplay.helpText)
    }
}
