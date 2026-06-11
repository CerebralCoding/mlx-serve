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

        // Spec-decode: PLD default-on.
        XCTAssertTrue(args.contains("--pld"))
        XCTAssertFalse(args.contains("--no-pld"))
        XCTAssertTrue(contains(args, flag: "--pld-draft-len", value: "5"))
        XCTAssertTrue(contains(args, flag: "--pld-key-len", value: "3"))

        // Off-by-default flags omit themselves.
        XCTAssertFalse(args.contains("--no-vision"))
        XCTAssertFalse(args.contains("--drafter"))
        XCTAssertFalse(contains(args, flag: "--timeout"))   // 300 = default, not emitted
    }

    /// 4096 was too low: a single thinking trace + agentic answer routinely
    /// tripped `finish_reason: "length"` and surfaced the truncation notice.
    /// The default per-turn output budget must be generous — the server still
    /// clamps it to the live context window, so a high value can't overflow.
    func testDefaultMaxTokensIsGenerous() {
        XCTAssertGreaterThanOrEqual(ServerOptions().defaultMaxTokens, 16384)
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

    // MARK: - Host / port (Settings UI fields)

    /// `parsePort` backs the Settings port text field — it must accept exactly
    /// what a TCP listen can bind and reject everything else, because an
    /// invalid value that slipped through would silently launch on a port the
    /// rest of the app (health checks, chat client) isn't watching.
    func testParsePortAcceptsValidPorts() {
        XCTAssertEqual(ServerOptions.parsePort("11234"), 11234)
        XCTAssertEqual(ServerOptions.parsePort(" 8080 "), 8080)   // trims whitespace
        XCTAssertEqual(ServerOptions.parsePort("1"), 1)
        XCTAssertEqual(ServerOptions.parsePort("65535"), 65535)
    }

    func testParsePortRejectsJunk() {
        XCTAssertNil(ServerOptions.parsePort(""))
        XCTAssertNil(ServerOptions.parsePort("0"))      // 0 = kernel-assigned ephemeral; client couldn't find the server
        XCTAssertNil(ServerOptions.parsePort("65536"))
        XCTAssertNil(ServerOptions.parsePort("-1"))
        XCTAssertNil(ServerOptions.parsePort("80x"))
        XCTAssertNil(ServerOptions.parsePort("abc"))
        XCTAssertNil(ServerOptions.parsePort("11 234"))
    }

    /// The host field is free text in Settings; a cleared field must not
    /// launch `--host ""` (the server would fail to bind).
    func testEmptyHostFallsBackToBindAll() {
        var opts = ServerOptions()
        opts.host = "   "
        XCTAssertTrue(contains(opts.toCLIArgs(), flag: "--host", value: "0.0.0.0"))
    }

    func testCustomHostIsEmitted() {
        var opts = ServerOptions()
        opts.host = "127.0.0.1"
        XCTAssertTrue(contains(opts.toCLIArgs(), flag: "--host", value: "127.0.0.1"))
    }

    func testNoVisionFlag() {
        var opts = ServerOptions()
        opts.noVision = true
        XCTAssertTrue(opts.toCLIArgs().contains("--no-vision"))
    }

    func testServerLaunchEqualsIgnoresPerRequestFields() {
        var a = ServerOptions()
        var b = ServerOptions()
        // Still purely per-request: max tokens, penalties, spec-decode
        // TriStates. (Sampling defaults — temperature/top-p/top-k — became
        // launch flags in 2026-06 so external clients like Claude Code
        // inherit them; covered by testSamplingDefaultsAffectRestartDetection.)
        b.defaultMaxTokens = 8192
        b.perRequestEnablePLD = .off
        b.defaultRepeatPenalty = 1.1
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
        opts.drafterPath = "/x/y/z"
        opts.defaultTemperature = 0.42
        opts.perRequestEnableDrafter = .off

        let data = try JSONEncoder().encode(opts)
        let decoded = try JSONDecoder().decode(ServerOptions.self, from: data)
        XCTAssertEqual(opts, decoded)
    }

    // MARK: - GGUF + common-engine flags

    func testLlamaKvQuantOmittedAtDefault() {
        let args = ServerOptions().toCLIArgs()
        XCTAssertFalse(args.contains("--llama-kv-quant"),
                      "default (.off) must NOT emit the flag so existing CLI invocations stay byte-identical")
    }

    func testLlamaKvQuantQ8EmitsFlag() {
        var opts = ServerOptions()
        opts.llamaKvQuant = .q8
        let args = opts.toCLIArgs()
        XCTAssertTrue(contains(args, flag: "--llama-kv-quant", value: "q8"))
    }

    func testLlamaKvQuantQ4EmitsFlag() {
        var opts = ServerOptions()
        opts.llamaKvQuant = .q4
        let args = opts.toCLIArgs()
        XCTAssertTrue(contains(args, flag: "--llama-kv-quant", value: "q4"))
    }

    func testLlamaCacheEntriesOmittedAtDefault() {
        let args = ServerOptions().toCLIArgs()
        XCTAssertFalse(args.contains("--llama-cache-entries"))
    }

    func testLlamaCacheEntriesEmitsWhenAboveOne() {
        var opts = ServerOptions()
        opts.llamaCacheEntries = 4
        let args = opts.toCLIArgs()
        XCTAssertTrue(contains(args, flag: "--llama-cache-entries", value: "4"))
    }

    func testTokenizeCacheEntriesOmittedAtDefault() {
        let args = ServerOptions().toCLIArgs()
        XCTAssertFalse(args.contains("--tokenize-cache-entries"),
                      "default (4) must NOT emit — matches server-side default")
    }

    func testTokenizeCacheEntriesEmitsWhenChanged() {
        var opts = ServerOptions()
        opts.tokenizeCacheEntries = 0
        var args = opts.toCLIArgs()
        XCTAssertTrue(contains(args, flag: "--tokenize-cache-entries", value: "0"))
        opts.tokenizeCacheEntries = 16
        args = opts.toCLIArgs()
        XCTAssertTrue(contains(args, flag: "--tokenize-cache-entries", value: "16"))
    }

    func testServerLaunchEqualsCoversNewFields() {
        var a = ServerOptions()
        var b = ServerOptions()
        // Each new field flipping must trigger a restart.
        b.llamaKvQuant = .q4
        XCTAssertFalse(a.serverLaunchEquals(b))
        b = ServerOptions()
        b.llamaCacheEntries = 4
        XCTAssertFalse(a.serverLaunchEquals(b))
        b = ServerOptions()
        b.tokenizeCacheEntries = 0
        XCTAssertFalse(a.serverLaunchEquals(b))
        // Sanity: untouched defaults are equal.
        a = ServerOptions(); b = ServerOptions()
        XCTAssertTrue(a.serverLaunchEquals(b))
    }

    // MARK: - Log level

    func testLogLevelDefaultIsInfo() {
        let args = ServerOptions().toCLIArgs()
        XCTAssertTrue(contains(args, flag: "--log-level", value: "info"))
    }

    func testCustomLogLevelEmitsCLIFlag() {
        for lvl in ServerOptions.LogLevel.allCases {
            var opts = ServerOptions()
            opts.logLevel = lvl
            XCTAssertTrue(
                contains(opts.toCLIArgs(), flag: "--log-level", value: lvl.rawValue),
                "logLevel=\(lvl.rawValue) must emit --log-level \(lvl.rawValue)"
            )
        }
    }

    func testLogLevelChangeTriggersRestart() {
        var a = ServerOptions()
        var b = ServerOptions()
        b.logLevel = .debug
        XCTAssertFalse(a.serverLaunchEquals(b),
                       "Switching log level must require a server restart")
        a.logLevel = .debug
        XCTAssertTrue(a.serverLaunchEquals(b))
    }

    func testLogLevelHasHumanReadableLabel() {
        // The Settings picker shows these — empty labels would render blank rows.
        for lvl in ServerOptions.LogLevel.allCases {
            XCTAssertFalse(lvl.label.isEmpty,
                           "\(lvl.rawValue) needs a label for the Settings picker")
        }
    }

    // MARK: - Engine inference

    func testEngineFromArchitecture() {
        // The Settings UI hides MLX-only sections when engine != .mlx and
        // surfaces the GGUF section when engine == .llama. The discriminator
        // is the `architecture` string the server reports for the active
        // model, derived from `model_type` in config.json (or the GGUF stub).
        var info = ModelInfo(name: "x", quantBits: 4, layers: 0,
                             hiddenSize: 0, vocabSize: 0,
                             contextLength: 0, modelMaxTokens: 0,
                             architecture: "gguf")
        XCTAssertEqual(info.engine, .llama)
        info.architecture = "deepseek_v4"
        XCTAssertEqual(info.engine, .dsv4)
        info.architecture = "gemma4"
        XCTAssertEqual(info.engine, .mlx)
        info.architecture = "qwen3_5_moe"
        XCTAssertEqual(info.engine, .mlx)
        info.architecture = ""  // older server build that omits the field
        XCTAssertEqual(info.engine, .mlx, "empty arch must default to .mlx (the most common path)")
    }

    // MARK: helpers

    private func contains(_ args: [String], flag: String, value: String? = nil) -> Bool {
        guard let i = args.firstIndex(of: flag) else { return false }
        guard let value else { return true }
        let next = i + 1
        return next < args.count && args[next] == value
    }
}

extension ServerOptionsTests {
    /// The Settings temperature must reach third-party clients (Claude Code
    /// omits sampling params entirely, so the server-launch default is the
    /// only channel). Top-p rides along; top-k 0 means "no opinion" and must
    /// be OMITTED so the model's generation_config.json recommendation
    /// (Qwen 3.6: top_k=20, Gemma 4: 64) stays in effect.
    func testSamplingDefaultsReachLaunchArgs() {
        var opts = ServerOptions()
        opts.defaultTemperature = 0.7
        opts.defaultTopP = 0.95
        opts.defaultTopK = 0
        let args = opts.toCLIArgs()
        XCTAssertTrue(contains(args, flag: "--temp", value: "0.7"))
        XCTAssertTrue(contains(args, flag: "--top-p", value: "0.95"))
        XCTAssertFalse(args.contains("--top-k"))

        opts.defaultTopK = 40
        XCTAssertTrue(contains(opts.toCLIArgs(), flag: "--top-k", value: "40"))
    }

    /// Changing a sampling default must trip the restart detector — these now
    /// affect the launched process, not just the app's own request bodies.
    func testSamplingDefaultsAffectRestartDetection() {
        let base = ServerOptions()
        var changed = base
        changed.defaultTemperature = 0.42
        XCTAssertFalse(base.serverLaunchEquals(changed))
    }
}

extension ServerOptionsTests {
    /// Slider arithmetic leaves float dirt (0.8 − 0.1 = 0.7000000000000001);
    /// argv must carry the clean decimal (seen verbatim in `ps` output live).
    func testSamplingFlagFormattingIsClean() {
        var opts = ServerOptions()
        opts.defaultTemperature = 0.8 - 0.1
        XCTAssertTrue(contains(opts.toCLIArgs(), flag: "--temp", value: "0.7"))
    }
}
