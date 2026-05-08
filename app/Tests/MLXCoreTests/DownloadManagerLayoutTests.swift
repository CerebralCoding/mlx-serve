import XCTest
@testable import MLXCore

/// Tests for the LM-Studio-style `<author>/<repo>` on-disk layout in
/// DownloadManager. New downloads land in the 2-level layout; existing flat
/// dirs continue to resolve via the dual-scan fallback. No auto-migration —
/// users redownload or move dirs manually.
final class DownloadManagerLayoutTests: XCTestCase {
    private var tempRoot: String!

    override func setUpWithError() throws {
        tempRoot = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("mlx-serve-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(atPath: tempRoot, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(atPath: tempRoot)
    }

    // MARK: - Path resolution

    func testNewLayoutDirSplitsAuthorAndName() {
        let p = DownloadManager.newLayoutDir(rootDir: tempRoot, repoId: "mlx-community/Qwen3.6-27B-mtp")
        XCTAssertEqual(p, (tempRoot as NSString)
            .appendingPathComponent("mlx-community")
            .appending("/Qwen3.6-27B-mtp"))
    }

    func testNewLayoutDirBareNameFallsBackToTopLevel() {
        // No author component — caller passed a bare name. Land at top level so
        // we don't fabricate an author dir.
        let p = DownloadManager.newLayoutDir(rootDir: tempRoot, repoId: "Qwen3.6-27B-mtp")
        XCTAssertEqual(p, (tempRoot as NSString).appendingPathComponent("Qwen3.6-27B-mtp"))
    }

    func testExistingModelDirPrefersNewLayout() throws {
        // Set up both: legacy flat AND new <author>/<name>.
        let name = "demo"
        let legacy = (tempRoot as NSString).appendingPathComponent(name)
        let nested = ((tempRoot as NSString).appendingPathComponent("acme") as NSString)
            .appendingPathComponent(name)
        try makeFakeModel(at: legacy)
        try makeFakeModel(at: nested)

        let resolved = DownloadManager.existingModelDir(rootDir: tempRoot, repoId: "acme/\(name)")
        XCTAssertEqual(resolved, nested, "new layout should win over legacy when both exist")
    }

    func testExistingModelDirFallsBackToLegacy() throws {
        // Only legacy exists. With a 2-level repoId we still want it found.
        let legacy = (tempRoot as NSString).appendingPathComponent("legacy-only")
        try makeFakeModel(at: legacy)

        let resolved = DownloadManager.existingModelDir(rootDir: tempRoot, repoId: "mlx-community/legacy-only")
        XCTAssertEqual(resolved, legacy, "legacy flat layout must remain discoverable until migrated")
    }

    func testExistingModelDirReturnsNilWhenAbsent() {
        XCTAssertNil(DownloadManager.existingModelDir(rootDir: tempRoot, repoId: "nobody/missing"))
    }

    // MARK: - Helpers

    /// Minimal model dir layout: just `config.json`. The path-resolution and
    /// migration logic only checks for that file's presence.
    private func makeFakeModel(at path: String) throws {
        try FileManager.default.createDirectory(atPath: path, withIntermediateDirectories: true)
        let cfg = (path as NSString).appendingPathComponent("config.json")
        try "{}".write(toFile: cfg, atomically: true, encoding: .utf8)
    }
}
