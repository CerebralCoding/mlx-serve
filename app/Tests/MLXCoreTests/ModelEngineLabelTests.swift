import XCTest
@testable import MLXCore

/// Engine identification for the model picker + status menu.
///
/// 1. `LocalModel.engine` — which embedded engine serves a selected model
///    (MLX safetensors, llama.cpp for generic GGUF, ds4 for DeepSeek-V4-Flash
///    GGUF). Mirrors the server's auto-routing (`model_discovery.isDs4GgufBasename`).
/// 2. `LocalModel.duplicateNames(in:)` — macOS `.menu` Pickers key the
///    checkmark state by item TITLE, so two models sharing a display name
///    (same repo as GGUF and MLX) both render selected. Rows whose name is
///    duplicated get an engine suffix to keep titles unique.
final class ModelEngineLabelTests: XCTestCase {
    private func model(name: String, path: String, modelType: String) -> LocalModel {
        LocalModel(
            id: "test:\(path)",
            name: name,
            path: path,
            sizeFormatted: "1 GB",
            modelType: modelType,
            source: .lmStudio,
            kind: .base
        )
    }

    func testSafetensorsDirIsMlxEngine() {
        let m = model(name: "mlx-community/gemma-4-e4b-it-4bit",
                      path: "/models/mlx-community/gemma-4-e4b-it-4bit",
                      modelType: "gemma4")
        XCTAssertEqual(m.engine, .mlx)
        XCTAssertEqual(m.engine.shortLabel, "MLX-Serve")
    }

    func testIndeterminateEngineDefaultsToMlxServe() {
        // No .gguf extension and an unknown model type — when the engine
        // can't be determined, MLX-Serve is the default.
        let m = model(name: "mystery/model", path: "/models/mystery", modelType: "unknown")
        XCTAssertEqual(m.engine, .mlx)
        XCTAssertEqual(m.engine.displayName, "MLX-Serve")
    }

    func testGenericGgufIsLlamaEngine() {
        let m = model(name: "lmstudio-community/gemma-4-E4B-it-GGUF",
                      path: "/models/gemma-4-E4B-it-Q4_K_M.gguf",
                      modelType: "gguf")
        XCTAssertEqual(m.engine, .llamaCpp)
        XCTAssertEqual(m.engine.shortLabel, "GGUF")
    }

    func testDeepseekGgufIsDs4Engine() {
        let m = model(name: "antirez/deepseek-v4-gguf",
                      path: "/models/DeepSeek-V4-Flash-IQ2XXS-chat-v2.gguf",
                      modelType: "deepseek_v4")
        XCTAssertEqual(m.engine, .ds4)
        XCTAssertEqual(m.engine.shortLabel, "DS4")
    }

    func testEngineDisplayNamesAreHumanReadable() {
        XCTAssertEqual(ModelEngine.mlx.displayName, "MLX-Serve")
        XCTAssertEqual(ModelEngine.llamaCpp.displayName, "GGUF · llama.cpp")
        XCTAssertEqual(ModelEngine.ds4.displayName, "GGUF · DS4")
    }

    func testDuplicateNamesDetectedAcrossEngines() {
        // Same display name as MLX dir and GGUF file — the live bug: both
        // rows showed the selection checkmark.
        let mlx = model(name: "qwen/Qwen3.6-27B",
                        path: "/models/qwen/Qwen3.6-27B",
                        modelType: "qwen3_5")
        let gguf = model(name: "qwen/Qwen3.6-27B",
                         path: "/models/qwen/Qwen3.6-27B/model-Q4_K_M.gguf",
                         modelType: "gguf")
        let unique = model(name: "mlx-community/gemma-4-e4b-it-4bit",
                           path: "/models/mlx-community/gemma-4-e4b-it-4bit",
                           modelType: "gemma4")
        let dups = LocalModel.duplicateNames(in: [mlx, gguf, unique])
        XCTAssertEqual(dups, ["qwen/Qwen3.6-27B"])
    }

    func testNoDuplicatesMeansEmptySet() {
        let a = model(name: "a", path: "/a", modelType: "gemma4")
        let b = model(name: "b", path: "/b", modelType: "gguf")
        XCTAssertTrue(LocalModel.duplicateNames(in: [a, b]).isEmpty)
    }
}
