import XCTest
@testable import MLXCore

/// A Qwen3-ASR download brings the forced aligner the server times words with.
final class AsrAlignerDownloadTests: XCTestCase {
    private func pack(_ config: String?) throws -> String {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString).path
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(atPath: dir) }
        if let config { try config.write(toFile: (dir as NSString).appendingPathComponent("config.json"), atomically: true, encoding: .utf8) }
        return dir
    }

    /// Both packs say `qwen3_asr` at the root; only the aligner's thinker tells them apart.
    func testOnlyAnAsrPackCallsForTheAligner() throws {
        XCTAssertTrue(DownloadManager.isAsrPack(try pack(#"{"model_type":"qwen3_asr","thinker_config":{"model_type":"qwen3_asr"}}"#)))
        XCTAssertFalse(DownloadManager.isAsrPack(try pack(#"{"model_type":"qwen3_asr","thinker_config":{"model_type":"qwen3_forced_aligner"}}"#)))
        XCTAssertFalse(DownloadManager.isAsrPack(try pack(#"{"model_type":"qwen3"}"#)))
        XCTAssertFalse(DownloadManager.isAsrPack(try pack(nil)))
        XCTAssertEqual(DownloadManager.alignerRepo, "mlx-community/Qwen3-ForcedAligner-0.6B-8bit")
    }
}
