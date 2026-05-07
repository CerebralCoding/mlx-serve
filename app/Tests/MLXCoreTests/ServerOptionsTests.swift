import XCTest
@testable import MLXCore

/// Unit tests for `ServerOptions.toCLIArgs()` — the Settings UI relies on
/// this to translate user choices into the actual `mlx-serve` CLI invocation.
/// A wrong arg here would silently launch the server with the wrong config,
/// so we lock down the contract.
final class ServerOptionsTests: XCTestCase {
    func testDefaultsProduceCanonicalArgs() {
        let opts = ServerOptions()
        let args = opts.toCLIArgs()

        // Always present.
        XCTAssertEqual(args.first, "--serve")
        XCTAssertTrue(contains(args, flag: "--port", value: "11234"))
        XCTAssertTrue(contains(args, flag: "--host", value: "0.0.0.0"))
        XCTAssertTrue(contains(args, flag: "--log-level", value: "info"))
        // Default `ctxSize == 0` means "Auto" — server picks the memory-bounded
        // safe ceiling at startup. The CLI flag is omitted entirely in that
        // case (the server's own `getEffectiveContextLength` runs).
        XCTAssertFalse(contains(args, flag: "--ctx-size"))

        // Spec-decode: PLD default-on, MTP default-off — the v5 flip from CLAUDE.md.
        XCTAssertTrue(args.contains("--pld"))
        XCTAssertFalse(args.contains("--no-pld"))
        XCTAssertTrue(args.contains("--no-mtp"))
        XCTAssertFalse(args.contains("--mtp"))
        XCTAssertTrue(contains(args, flag: "--pld-draft-len", value: "5"))
        XCTAssertTrue(contains(args, flag: "--pld-key-len", value: "3"))

        // Off-by-default flags omit themselves.
        XCTAssertFalse(args.contains("--no-vision"))
        XCTAssertFalse(args.contains("--drafter"))
        XCTAssertFalse(contains(args, flag: "--timeout"))   // 300 = default, not emitted
    }

    func testPLDOffUsesNoPldFlag() {
        var opts = ServerOptions()
        opts.enablePLD = false
        let args = opts.toCLIArgs()
        XCTAssertTrue(args.contains("--no-pld"))
        XCTAssertFalse(args.contains("--pld"))
    }

    func testCustomPortAndCtxSizeAreEmitted() {
        var opts = ServerOptions()
        opts.port = 8080
        opts.ctxSize = 65536
        let args = opts.toCLIArgs()
        XCTAssertTrue(contains(args, flag: "--port", value: "8080"))
        XCTAssertTrue(contains(args, flag: "--ctx-size", value: "65536"))
    }

    func testCtxSizeZeroOmitsFlag() {
        var opts = ServerOptions()
        opts.ctxSize = 0
        let args = opts.toCLIArgs()
        XCTAssertFalse(contains(args, flag: "--ctx-size"))
    }

    func testDrafterPathPullsInBlockSize() {
        var opts = ServerOptions()
        opts.drafterPath = "/tmp/gemma-4-E4B-it-assistant-bf16"
        opts.draftBlockSize = 6
        let args = opts.toCLIArgs()
        XCTAssertTrue(contains(args, flag: "--drafter", value: "/tmp/gemma-4-E4B-it-assistant-bf16"))
        XCTAssertTrue(contains(args, flag: "--draft-block-size", value: "6"))
    }

    func testEmptyDrafterPathOmitsBothFlags() {
        let opts = ServerOptions()  // drafterPath = ""
        let args = opts.toCLIArgs()
        XCTAssertFalse(args.contains("--drafter"))
        XCTAssertFalse(args.contains("--draft-block-size"))
    }

    func testCustomTimeoutIsEmittedWhenNonDefault() {
        var opts = ServerOptions()
        opts.requestTimeout = 600
        XCTAssertTrue(contains(opts.toCLIArgs(), flag: "--timeout", value: "600"))

        opts.requestTimeout = 0  // unlimited
        XCTAssertTrue(contains(opts.toCLIArgs(), flag: "--timeout", value: "0"))
    }

    func testNoVisionFlag() {
        var opts = ServerOptions()
        opts.noVision = true
        XCTAssertTrue(opts.toCLIArgs().contains("--no-vision"))
    }

    func testServerLaunchEqualsIgnoresPerRequestFields() {
        var a = ServerOptions()
        var b = ServerOptions()
        b.defaultTemperature = 0.9
        b.defaultMaxTokens = 8192
        b.perRequestEnablePLD = .off
        XCTAssertTrue(a.serverLaunchEquals(b),
                     "Per-request defaults must NOT trigger restart")

        a.port = 9000
        XCTAssertFalse(a.serverLaunchEquals(b),
                      "Server-launch fields MUST trigger restart")
    }

    func testTriStateMaps() {
        XCTAssertNil(ServerOptions.TriState.auto.asOptionalBool)
        XCTAssertEqual(ServerOptions.TriState.on.asOptionalBool, true)
        XCTAssertEqual(ServerOptions.TriState.off.asOptionalBool, false)
    }

    func testRoundTripCodable() throws {
        var opts = ServerOptions()
        opts.port = 9999
        opts.enableMTP = true
        opts.drafterPath = "/x/y/z"
        opts.defaultTemperature = 0.42
        opts.perRequestEnableDrafter = .off

        let data = try JSONEncoder().encode(opts)
        let decoded = try JSONDecoder().decode(ServerOptions.self, from: data)
        XCTAssertEqual(opts, decoded)
    }

    // MARK: helpers

    private func contains(_ args: [String], flag: String, value: String? = nil) -> Bool {
        guard let i = args.firstIndex(of: flag) else { return false }
        guard let value else { return true }
        let next = i + 1
        return next < args.count && args[next] == value
    }
}
