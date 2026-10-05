import XCTest
@testable import MLXCore

/// DeepSeek-V4.1-Flash is served by the mlx-stream plugin (arch `deepseek_v41`, experts streamed from SSD). The app
/// offers it like its other models: a catalog row, the launch flags the plugin's served runs use, and its own
/// context-size choices. These pin each surface to the one definition in `DeepSeekV41`.
final class DeepSeekV41LaunchTests: XCTestCase {

    func testCatalogCarriesTheStreamedPackFor128GBMacs() {
        let row = gemmaModelOptions.first { $0.id == DeepSeekV41.catalogId }
        XCTAssertNotNil(row, "the V4.1 pack is missing from the download catalog")
        XCTAssertEqual(row?.repoId, DeepSeekV41.repoId)
        XCTAssertEqual(DeepSeekV41.repoId, "OpensourceWTF/DeepSeek-V4.1-Flash-streaming-repack-exl3-3.0bpw")
        XCTAssertFalse(DeepSeekV41.repoId.contains("MTPLX"))
        XCTAssertNil(row?.ggufFilename, "the pack is a model directory, not a GGUF file")
        XCTAssertEqual(row?.minHostRamBytes, 128 * (UInt64(1) << 30))
    }

    func testTheArchIsAServedModelType() {
        XCTAssertTrue(supportedModelTypes.contains(DeepSeekV41.modelType))
    }

    func testLaunchFlagsAreTheServedRunsFlagsAndOnlyForV41() {
        XCTAssertEqual(DeepSeekV41.launchArgs(modelType: "deepseek_v41"),
                       ["--memory-ceiling-gb", "120.259", "--wired-margin-gib", "2"])
        XCTAssertEqual(DeepSeekV41.launchArgs(modelType: "deepseek_v4"), [])
        XCTAssertEqual(DeepSeekV41.launchArgs(modelType: nil), [])
    }

    func testLaunchFlagsParseAsTheServerRequires() {
        // main.zig: --memory-ceiling-gb takes decimal GB above 4; --wired-margin-gib an integer 2..32.
        let args = DeepSeekV41.launchArgs(modelType: DeepSeekV41.modelType)
        let ceiling = Double(args[1])
        XCTAssertNotNil(ceiling)
        XCTAssertGreaterThan(ceiling ?? 0, 4)
        // 112 GiB in decimal GB, the box ceiling the served runs admit under.
        XCTAssertEqual(ceiling ?? 0, 112 * 1_073_741_824 / 1e9, accuracy: 0.001)
        XCTAssertTrue((2...32).contains(Int(args[3]) ?? 0))
    }

    func testModelTypeIsReadFromTheModelsConfig() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("dsv41-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertNil(DeepSeekV41.modelType(atModelPath: dir.path))
        try #"{"model_type": "deepseek_v41", "hidden_size": 5120}"#.write(
            to: dir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        XCTAssertEqual(DeepSeekV41.modelType(atModelPath: dir.path), "deepseek_v41")
        XCTAssertNil(DeepSeekV41.modelType(atModelPath: dir.appendingPathComponent("x.gguf").path))
    }

    func testContextChoicesAreTheSweepSizesPlusTheReplyPadAndTheModelLimit() {
        // ctx_size = prompt + 1,088 (max_tokens 1,024 + a 64-token pad), as the served sweep sets it; 1,048,576 is
        // the model's limit. No choice = the server's default bill (every prompt up to 16,384).
        let sweep = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512].map { $0 * 1024 + 1088 }
        XCTAssertEqual(DeepSeekV41.ctxSizeChoices, sweep + [1_048_576])
        XCTAssertEqual(DeepSeekV41.ctxSizeChoices, DeepSeekV41.ctxSizeChoices.sorted())
        XCTAssertEqual(DeepSeekV41.contextPresets(modelType: "deepseek_v41"), DeepSeekV41.ctxSizeChoices)
        XCTAssertEqual(DeepSeekV41.contextPresets(modelType: "qwen3"), ContextSizeDisplay.presets)
        XCTAssertEqual(DeepSeekV41.contextPresets(modelType: nil), ContextSizeDisplay.presets)
    }
}
