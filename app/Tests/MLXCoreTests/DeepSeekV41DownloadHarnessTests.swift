import CryptoKit
import Foundation
import XCTest
@testable import MLXCore

/// The V4.1 pack through the app's REAL `DownloadManager` (the catalog row's path: `download(repoId:)`, no mock).
///
/// Offline: the catalog row's file selection over the pack's published layout must select every file the
/// mlx-stream served load reads.
///
/// Live (`MLX_SERVE_LIVE_V41_DOWNLOAD=1`), into `MLX_SERVE_V41_DOWNLOAD_ROOT`
/// (default /Users/davidtai/models/app-download-test):
/// - `MLX_SERVE_V41_DRY_RUN=1` fetches only the files under 64 MB, then proves resume by cutting one file back to a
///   half-written `.partial` and fetching again;
/// - otherwise the whole catalog selection.
/// Every file is then checked against the HF tree (size, and sha256 for LFS files / git blob sha1 otherwise) and
/// printed as one receipt line: `V41_DL_RECEIPT <ok|FAIL> <path> size=<n> want=<oid> got=<oid> secs=<t> rev=<sha>`.
/// The files are verified against the PINNED revision's tree (`MLX_SERVE_V41_REVISION`, default `pinnedRevision`); the
/// download reads `main`, so a selected file whose size or oid differs between `main` and the pin refuses the run.
@MainActor
final class DeepSeekV41DownloadHarnessTests: XCTestCase {

    /// What the served load opens (mlx-stream 8e8c937: the arch's config/tokenizer, the v2 expert manifest and bank,
    /// the Engram token map, residents, manifest and row tables, and the resident trunk shards).
    /// The published revision the receipts are checked against (sizes and shas verified at upload).
    static let pinnedRevision = "7c45f40f005ff4259dcfb3553d9c3b226f26d1e3"

    static let servedLoadReads: [String] = [
        "config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
        "expert-manifest-v2.json", "experts.bin", "engram-token-map.u32",
        "engram/engram-residents.safetensors", "engram/engram-manifest.json",
        "engram/engram-L1.bin", "engram/engram-L14.bin", "model-00001.safetensors",
    ]

    /// The selection `start(repoId:)` uses for the catalog row.
    static var catalogSelection: FileSelection { DeepSeekV41.selection(forRepo: DeepSeekV41.repoId) ?? .chatDefault }

    func testTheCatalogSelectionTakesEveryFileTheServedLoadReads() {
        let tree = Self.servedLoadReads.map { ["type": "file", "path": $0, "size": 1] as [String: Any] }
        let picked = Set(DownloadManager.selectNeededFiles(from: tree, selection: Self.catalogSelection).map(\.0))
        let missing = Self.servedLoadReads.filter { !picked.contains($0) }
        XCTAssertEqual(missing, [], "the catalog row's download leaves out files the served load reads")
    }

    func testTheCatalogSelectionLeavesOutTheReceiptsAndTheReferenceEncoder() {
        let tree = ["receipts/state.json", "encoding/README.md", "encoding/tests/test_output_1.txt", "README.md"]
            .map { ["type": "file", "path": $0, "size": 1] as [String: Any] }
        XCTAssertEqual(DownloadManager.selectNeededFiles(from: tree, selection: Self.catalogSelection).map(\.0), [])
        XCTAssertNil(DeepSeekV41.selection(forRepo: "mlx-community/gemma-4-e4b-it-4bit"))
    }

    func testLiveDownload() async throws {
        let env = ProcessInfo.processInfo.environment
        try XCTSkipUnless(env["MLX_SERVE_LIVE_V41_DOWNLOAD"] == "1", "set MLX_SERVE_LIVE_V41_DOWNLOAD=1")
        let dry = env["MLX_SERVE_V41_DRY_RUN"] == "1"
        let root = env["MLX_SERVE_V41_DOWNLOAD_ROOT"] ?? "/Users/davidtai/models/app-download-test"
        let repo = DeepSeekV41.repoId
        let rev = env["MLX_SERVE_V41_REVISION"] ?? Self.pinnedRevision
        let tree = try await Self.hfTree(repo, revision: rev)
        let files = tree.filter { ($0["type"] as? String) == "file" }
        let mainFiles = try await Self.hfTree(repo, revision: "main").filter { ($0["type"] as? String) == "file" }
        print("V41_DL_TREE repo=\(repo) rev=\(rev) entries=\(files.count) bytes=\(files.reduce(Int64(0)) { $0 + Self.size($1) })")
        for e in files { print("V41_DL_TREE_FILE \(e["path"] as? String ?? "?") size=\(Self.size(e)) lfs=\(Self.lfsOid(e) != nil)") }

        let catalog = DownloadManager.selectNeededFiles(from: tree, selection: Self.catalogSelection)
        print("V41_DL_PLAN catalog_selection=\(catalog.count) files bytes=\(catalog.reduce(Int64(0)) { $0 + $1.1 })")
        let skipped = files.compactMap { $0["path"] as? String }.filter { p in !catalog.contains { $0.0 == p } }
        for p in skipped { print("V41_DL_PLAN_SKIPPED \(p)") }

        let dm = DownloadManager(modelsRoot: root)
        let dest = DownloadManager.newLayoutDir(rootDir: root, repoId: repo)
        let big = Set(files.filter { Self.size($0) >= 64 << 20 }.compactMap { $0["path"] as? String })
        var selection = Self.catalogSelection
        if dry { selection.excludeSubstrings += Array(big) }
        let want = DownloadManager.selectNeededFiles(from: tree, selection: selection).map(\.0)
        XCTAssertFalse(want.isEmpty, "nothing to fetch yet")
        for p in want {
            let pinned = files.first { ($0["path"] as? String) == p }
            let head = mainFiles.first { ($0["path"] as? String) == p }
            let same = pinned.map(Self.size) == head.map(Self.size)
                && (pinned.flatMap(Self.lfsOid) ?? pinned?["oid"] as? String) == (head.flatMap(Self.lfsOid) ?? head?["oid"] as? String)
            guard same else { XCTFail("\(p) on main differs from revision \(rev): refusing to download"); return }
        }
        print("V41_DL_PIN main matches rev=\(rev) for all \(want.count) selected files")

        let t0 = Date()
        await dm.download(repoId: repo, selection: selection, alertOnFailure: false)
        XCTAssertEqual(dm.downloads[repo]?.status, .completed, dm.downloads[repo]?.error ?? "")
        print("V41_DL_PASS1 secs=\(String(format: "%.1f", Date().timeIntervalSince(t0)))")

        if dry, let victim = want.max(by: { Self.fileSize("\(dest)/\($0)") < Self.fileSize("\(dest)/\($1)") }) {
            // Resume: the file becomes a half-written partial, and the same call must finish it from there.
            let path = "\(dest)/\(victim)"
            let data = try Data(contentsOf: URL(fileURLWithPath: path))
            try FileManager.default.removeItem(atPath: path)
            try data.prefix(data.count / 2).write(to: URL(fileURLWithPath: path + ".partial"))
            print("V41_DL_RESUME cut \(victim) to \(data.count / 2) of \(data.count) B")
            await dm.download(repoId: repo, selection: selection, alertOnFailure: false)
            XCTAssertEqual(dm.downloads[repo]?.status, .completed, dm.downloads[repo]?.error ?? "")
            XCTAssertFalse(FileManager.default.fileExists(atPath: path + ".partial"))
        }

        var failed = 0
        for p in want {
            let e = files.first { ($0["path"] as? String) == p }!
            let path = "\(dest)/\(p)"
            let t = Date()
            let (wantOid, gotOid) = try Self.digests(entry: e, path: path)
            let ok = Self.fileSize(path) == Self.size(e) && wantOid == gotOid
            if !ok { failed += 1 }
            print("V41_DL_RECEIPT \(ok ? "ok" : "FAIL") \(p) size=\(Self.fileSize(path)) want=\(wantOid) got=\(gotOid) secs=\(String(format: "%.1f", Date().timeIntervalSince(t))) rev=\(rev)")
        }
        print("V41_DL_DONE files=\(want.count) failed=\(failed) rev=\(rev) dest=\(dest)")
        XCTAssertEqual(failed, 0)
    }

    // MARK: - HF tree and digests

    static func hfTree(_ repo: String, revision: String) async throws -> [[String: Any]] {
        let url = URL(string: "https://huggingface.co/api/models/\(repo)/tree/\(revision)?recursive=true")!
        let (data, _) = try await DownloadSession.shared.data(for: DownloadManager.hfApiRequest(url))
        return (try JSONSerialization.jsonObject(with: data) as? [[String: Any]]) ?? []
    }

    static func size(_ e: [String: Any]) -> Int64 {
        (e["size"] as? Int64) ?? (e["size"] as? Int).map(Int64.init) ?? 0
    }

    static func lfsOid(_ e: [String: Any]) -> String? { (e["lfs"] as? [String: Any])?["oid"] as? String }

    static func fileSize(_ path: String) -> Int64 {
        ((try? FileManager.default.attributesOfItem(atPath: path))?[.size] as? Int64) ?? -1
    }

    /// (want, got): sha256 for an LFS file, else git's blob sha1 (`blob <size>\0<bytes>`), streamed in 64 MB reads.
    static func digests(entry e: [String: Any], path: String) throws -> (String, String) {
        guard let h = FileHandle(forReadingAtPath: path) else { return (lfsOid(e) ?? (e["oid"] as? String ?? "?"), "missing") }
        defer { try? h.close() }
        if let lfs = lfsOid(e) {
            var sha = SHA256()
            while let chunk = try h.read(upToCount: 64 << 20), !chunk.isEmpty { sha.update(data: chunk) }
            return (lfs, sha.finalize().map { String(format: "%02x", $0) }.joined())
        }
        var sha = Insecure.SHA1()
        sha.update(data: Data("blob \(fileSize(path))\0".utf8))
        while let chunk = try h.read(upToCount: 64 << 20), !chunk.isEmpty { sha.update(data: chunk) }
        return (e["oid"] as? String ?? "?", sha.finalize().map { String(format: "%02x", $0) }.joined())
    }
}
