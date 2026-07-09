import XCTest
@testable import MLXCore

/// Unit tests for `ServerOptions.resetToDefaults(preserving:)` — the pure
/// decision behind the Settings screen's "Reset to Defaults" button.
///
/// The button used to assign a bare `ServerOptions()`, which wiped the Telegram
/// bot token. That token is a credential the user obtained from @BotFather and
/// cannot re-derive from anything on this machine, so a reset must carry it
/// across. Everything else is re-derivable (a chat id re-adopts on the next
/// message, a path can be re-picked) and is deliberately restored.
final class SettingsResetTests: XCTestCase {

    func testResetPreservesTheTelegramBotToken() {
        var current = ServerOptions()
        current.telegram.botToken = "123456:ABC-DEF_secret"

        let reset = ServerOptions.resetToDefaults(preserving: current)

        XCTAssertEqual(reset.telegram.botToken, "123456:ABC-DEF_secret")
    }

    func testResetPreservesTheTokenVerbatimIncludingWhitespace() {
        // The bridge trims on use (`trimmedToken`); reset must not silently
        // rewrite what the user pasted into the field.
        var current = ServerOptions()
        current.telegram.botToken = "  123456:ABC\n"

        let reset = ServerOptions.resetToDefaults(preserving: current)

        XCTAssertEqual(reset.telegram.botToken, "  123456:ABC\n")
    }

    func testResetRestoresEveryOtherFieldToTheShippedDefault() {
        var current = ServerOptions()
        current.telegram.botToken = "keep-me"
        current.host = "127.0.0.1"
        current.port = 9999
        current.ctxSize = 32768
        current.noVision = true
        current.enableMetrics = true
        current.apiKey = "hunter2"
        current.defaultTemperature = 1.9
        current.maxConcurrent = 8

        let reset = ServerOptions.resetToDefaults(preserving: current)

        let shipped = ServerOptions()
        XCTAssertEqual(reset.host, shipped.host)
        XCTAssertEqual(reset.port, shipped.port)
        XCTAssertEqual(reset.ctxSize, shipped.ctxSize)
        XCTAssertEqual(reset.noVision, shipped.noVision)
        XCTAssertEqual(reset.enableMetrics, shipped.enableMetrics)
        XCTAssertEqual(reset.apiKey, shipped.apiKey)
        XCTAssertEqual(reset.defaultTemperature, shipped.defaultTemperature)
        XCTAssertEqual(reset.maxConcurrent, shipped.maxConcurrent)
    }

    /// Only the token survives — the rest of the Telegram block is re-derivable
    /// (the bot re-adopts the first chat that messages it), so a reset should
    /// genuinely reset it. This pins the scope of the carve-out.
    func testResetClearsTheOtherTelegramFields() {
        var current = ServerOptions()
        current.telegram.botToken = "keep-me"
        current.telegram.enabled = true
        current.telegram.agentMode = true
        current.telegram.useMCP = true
        current.telegram.enableThinking = true
        current.telegram.allowedChatIds = [42, 43]

        let reset = ServerOptions.resetToDefaults(preserving: current)

        XCTAssertEqual(reset.telegram.botToken, "keep-me")
        XCTAssertFalse(reset.telegram.enabled)
        XCTAssertFalse(reset.telegram.agentMode)
        XCTAssertFalse(reset.telegram.useMCP)
        XCTAssertFalse(reset.telegram.enableThinking)
        XCTAssertEqual(reset.telegram.allowedChatIds, [])
    }

    /// With no token set, a reset is byte-for-byte the shipped defaults. Uses
    /// the whole-struct `Equatable` so a field added later is covered for free.
    func testResetWithNoTokenEqualsShippedDefaults() {
        var current = ServerOptions()
        current.host = "1.2.3.4"
        current.apiKey = "x"

        XCTAssertEqual(ServerOptions.resetToDefaults(preserving: current), ServerOptions())
    }

    /// Whole-struct guard: a reset differs from the shipped defaults in exactly
    /// one way — the token. Any future field that starts leaking through the
    /// carve-out fails here.
    func testResetDiffersFromDefaultsOnlyByTheToken() {
        var current = ServerOptions()
        current.telegram.botToken = "keep-me"
        current.host = "1.2.3.4"
        current.telegram.enabled = true

        var reset = ServerOptions.resetToDefaults(preserving: current)
        XCTAssertNotEqual(reset, ServerOptions())

        reset.telegram.botToken = ""
        XCTAssertEqual(reset, ServerOptions())
    }
}
