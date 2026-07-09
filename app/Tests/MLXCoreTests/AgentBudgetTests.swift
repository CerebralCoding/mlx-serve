import XCTest
@testable import MLXCore

/// The CLI launcher writes config files for third-party agents (pi, opencode)
/// and env vars for Claude Code. Those configs declare how much context the
/// model has — and the agents budget their own `max_tokens` against that number,
/// NOT against anything the server says.
///
/// Regression: the pi config hardcoded `contextWindow: 32768, maxTokens: 8192`
/// while the server was advertising ~94k. Late in a long session pi's remaining
/// budget collapsed and it started sending `max_tokens=1` (observed live,
/// 2026-07-08 — `prompt=30827 tokens, max_gen=1, ctx=92387` in the server log),
/// so every tool call truncated and the session died. The launcher must derive
/// these numbers from the context the RUNNING server advertises.
final class AgentBudgetTests: XCTestCase {

    // MARK: Budget derivation

    func testUnknownServerContextFallsBackToTheConservativeDefault() {
        // No server info yet (not running / pre-metadata build): never guess high.
        XCTAssertEqual(AgentBudget.forServerContext(nil).context, 32768)
        XCTAssertEqual(AgentBudget.forServerContext(nil).output, 8192)
        XCTAssertEqual(AgentBudget.forServerContext(0).context, 32768)
        XCTAssertEqual(AgentBudget.forServerContext(-1).context, 32768)
    }

    func testServerContextIsDeclaredEXACTLY_noSecondMargin() {
        // The live number a 128 GB Mac running Qwen3.6-27B pins at load.
        let b = AgentBudget.forServerContext(78848)

        // The server already reserved 15% of the memory ceiling before it
        // advertised this. Shaving a second margin here double-counted that
        // headroom AND made the CLI report a different context than Settings
        // showed (opencode said 75K where the server said 77K). Declare the
        // server's number verbatim: it IS the enforced limit, and our own
        // `clampMaxTokens` uses the same one.
        XCTAssertEqual(b.context, 78848)
        // Far above the old hardcoded 32768 — that was the original bug.
        XCTAssertGreaterThan(b.context, 60000)
        // Enough output budget for a one-shot whole-file write.
        XCTAssertGreaterThanOrEqual(b.output, 16384)
        // prompt + output must be expressible inside the window.
        XCTAssertLessThan(b.output, b.context)
    }

    func testContextNeverExceedsWhatTheServerAdvertises() {
        for advertised in [1024, 4096, 8192, 16384, 32768, 65536, 94729, 262144] {
            let b = AgentBudget.forServerContext(advertised)
            XCTAssertEqual(b.context, advertised,
                "declared context must equal the server's \(advertised)")
            XCTAssertLessThanOrEqual(b.output, b.context,
                "output \(b.output) > context \(b.context) at advertised=\(advertised)")
            XCTAssertGreaterThan(b.output, 0)
        }
    }

    func testBudgetGrowsMonotonicallyWithServerContext() {
        var last = 0
        for advertised in [8192, 16384, 32768, 65536, 94729] {
            let c = AgentBudget.forServerContext(advertised).context
            XCTAssertGreaterThanOrEqual(c, last)
            last = c
        }
    }

    // MARK: The configs we actually write

    func testPiConfigCarriesTheDerivedBudgetNotAHardcodedOne() throws {
        let b = AgentBudget.forServerContext(94729)
        let json = AgentConfigs.piModelsJSON(
            baseURL: "http://localhost:11234", model: "Qwen3.6-27B", budget: b)

        // Parse it — a broken config silently strands the user on defaults.
        let obj = try XCTUnwrap(try JSONSerialization.jsonObject(
            with: Data(json.utf8)) as? [String: Any])
        let providers = try XCTUnwrap(obj["providers"] as? [String: Any])
        let mlx = try XCTUnwrap(providers["mlx"] as? [String: Any])
        XCTAssertEqual(mlx["baseUrl"] as? String, "http://localhost:11234/v1")
        let models = try XCTUnwrap(mlx["models"] as? [[String: Any]])
        let m = try XCTUnwrap(models.first)

        XCTAssertEqual(m["contextWindow"] as? Int, b.context)
        XCTAssertEqual(m["maxTokens"] as? Int, b.output)
        XCTAssertNotEqual(m["contextWindow"] as? Int, 32768, "still hardcoded")
        XCTAssertEqual(m["id"] as? String, "Qwen3.6-27B")
    }

    func testOpencodeConfigDeclaresPerModelLimits() throws {
        let b = AgentBudget.forServerContext(94729)
        let json = AgentConfigs.opencodeJSON(
            baseURL: "http://localhost:11234", model: "Qwen3.6-27B", budget: b)

        let obj = try XCTUnwrap(try JSONSerialization.jsonObject(
            with: Data(json.utf8)) as? [String: Any])
        let provider = try XCTUnwrap(obj["provider"] as? [String: Any])
        let mlx = try XCTUnwrap(provider["mlx"] as? [String: Any])
        let models = try XCTUnwrap(mlx["models"] as? [String: Any])
        let model = try XCTUnwrap(models["Qwen3.6-27B"] as? [String: Any])
        // opencode's schema: models.<id>.limit.{context,output}
        let limit = try XCTUnwrap(model["limit"] as? [String: Any])
        XCTAssertEqual(limit["context"] as? Int, b.context)
        XCTAssertEqual(limit["output"] as? Int, b.output)
    }

    func testClaudeCodeExportsCapOutputTokens() {
        let b = AgentBudget.forServerContext(94729)
        let script = AgentConfigs.claudeCodeExports(
            baseURL: "http://localhost:11234", model: "mlx-serve", budget: b)

        XCTAssertTrue(script.contains("export ANTHROPIC_BASE_URL='http://localhost:11234'"))
        // Claude Code exposes CLAUDE_CODE_MAX_OUTPUT_TOKENS (verified present in
        // the 2.1.x binary); it has NO context-window override, so that is the
        // only budget lever we have on this CLI.
        XCTAssertTrue(script.contains("export CLAUDE_CODE_MAX_OUTPUT_TOKENS=\(b.output)"),
                      "missing output cap in:\n\(script)")
        XCTAssertTrue(script.contains("ANTHROPIC_DEFAULT_SONNET_MODEL=mlx-serve"))
    }
}
