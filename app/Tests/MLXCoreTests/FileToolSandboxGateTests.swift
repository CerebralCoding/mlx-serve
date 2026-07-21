import XCTest
@testable import MLXCore

/// CLASS GUARD — "a file tool that ignores the Agent Sandbox" plus the
/// follow-up ruling: a working directory is MANDATORY for file tools in every
/// mode.
///
/// Live 2026-07-20: with the sandbox ON and the VM pinned to a different
/// workspace by a CLI session, `shell` was correctly declined — but `writeFile`
/// happily wrote to the host in the same turn. The sandbox only gated `shell`;
/// the five file tools ran host-side with `resolveAndConfine` as their only
/// containment, and that silently no-oped on a nil working directory (the old
/// yolo lever). Two rules now hold:
///  1. File tools ALWAYS require a working directory — every mode, sandbox on
///     or off. The nil-wd "unconfined" lever is gone; yolo tasks anchor at the
///     default agent workspace instead (approval stays unrestricted — see
///     ApprovalPolicy; confinement is this layer's job).
///  2. With the sandbox ON, a VM pinned to a workspace that doesn't cover this
///     session's folder declines file tools with shell's remount-block
///     semantics (`FileToolSandboxGate`).
final class FileToolSandboxGateTests: XCTestCase {

    private func makeTempDir() throws -> String {
        let dir = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("ftsg-\(UUID().uuidString)")
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        return dir
    }

    /// One entry per file tool, with valid params rooted in `dir`. A NEW file
    /// tool must be added here (and take the gate) — file access that skips
    /// the gate or the mandatory-workspace rule is the exact shipped class.
    private func fileToolCases(gate: FileToolSandboxGate, dir: String)
        -> [(String, any ToolHandler, [String: String])] {
        [
            ("readFile", ReadFileHandler(gate: gate), ["path": "f.txt"]),
            ("writeFile", WriteFileHandler(gate: gate), ["path": "f.txt", "content": "x"]),
            ("editFile", EditFileHandler(gate: gate), ["path": "f.txt", "find": "a", "replace": "b"]),
            ("searchFiles", SearchFilesHandler(gate: gate), ["pattern": "x", "path": dir]),
            ("listFiles", ListFilesHandler(gate: gate), ["path": dir]),
        ]
    }

    // MARK: Pure gate rule (sandbox pin agreement)

    func testSandboxOffNeverRejects() {
        XCTAssertNil(FileToolSandboxGate.rejectReason(
            sandboxEnabled: false, workingDirectory: "/w", pinnedRoot: "/other", pinnedLabel: "pi"))
    }

    func testGateHasNoOpinionOnNilWorkingDirectory() {
        // Mandatory-workspace is enforced universally at the confinement layer
        // (resolveAndConfine — pinned by the handler test below), not by the
        // sandbox gate: the gate only answers the PIN question.
        XCTAssertNil(FileToolSandboxGate.rejectReason(
            sandboxEnabled: true, workingDirectory: nil, pinnedRoot: nil, pinnedLabel: nil))
    }

    func testSandboxOnWithWorkspaceUnpinnedAllows() {
        XCTAssertNil(FileToolSandboxGate.rejectReason(
            sandboxEnabled: true, workingDirectory: "/w", pinnedRoot: nil, pinnedLabel: nil))
    }

    func testSandboxOnPinnedCoveringWorkspaceAllows() {
        XCTAssertNil(FileToolSandboxGate.rejectReason(
            sandboxEnabled: true, workingDirectory: "/w/sub", pinnedRoot: "/w", pinnedLabel: "pi"),
            "wd under the pinned share → same files the guest sees; allowed")
        XCTAssertNil(FileToolSandboxGate.rejectReason(
            sandboxEnabled: true, workingDirectory: "/w", pinnedRoot: "/w", pinnedLabel: "pi"))
    }

    func testSandboxOnPinnedElsewhereRejectsNamingBothPathsAndSession() {
        let reason = FileToolSandboxGate.rejectReason(
            sandboxEnabled: true, workingDirectory: "/projects/wksp",
            pinnedRoot: "/other/root", pinnedLabel: "pi")
        XCTAssertNotNil(reason, "file tools must agree with shell's remount-block")
        XCTAssertTrue(reason!.contains("pi"), "must NAME the pinning session: \(reason!)")
        XCTAssertTrue(reason!.contains("/other/root"), "must name the pinned share: \(reason!)")
        XCTAssertTrue(reason!.contains("/projects/wksp"), "must name this session's folder: \(reason!)")
        XCTAssertTrue(reason!.lowercased().contains("close"), "must tell the way out: \(reason!)")
    }

    func testPinnedWithoutLabelDoesNotReject() {
        // A stale sharedRoot with no live CLI pin is shell's silent-remount
        // case — file tools must not block where shell would proceed.
        XCTAssertNil(FileToolSandboxGate.rejectReason(
            sandboxEnabled: true, workingDirectory: "/w", pinnedRoot: "/other", pinnedLabel: nil))
    }

    // MARK: Mandatory working directory — every mode, every file tool

    func testEveryFileToolRequiresAWorkingDirectoryInEveryMode() async throws {
        // Sandbox OFF on purpose: the rule is universal, not a sandbox rule.
        // The old contract (nil wd = unconfined host access, the yolo lever)
        // is GONE — yolo runs now anchor at the default agent workspace.
        let off = FileToolSandboxGate(sandboxEnabled: { false },
                                      pinnedWorkspace: { (nil, nil) })
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(atPath: dir) }
        let absoluteTarget = (dir as NSString).appendingPathComponent("gate.txt")

        for (name, handler, params) in fileToolCases(gate: off, dir: dir) {
            // Absolute paths must not slip past the rule either.
            var p = params
            if p["path"] == "f.txt" { p["path"] = absoluteTarget }
            do {
                _ = try await handler.execute(parameters: p, workingDirectory: nil)
                XCTFail("\(name) must refuse to run without a working directory")
            } catch {
                XCTAssertTrue(error.localizedDescription.contains("working folder"),
                              "\(name) must throw the workspace-required error, got: \(error.localizedDescription)")
            }
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: absoluteTarget),
                       "a refused writeFile must not have touched the disk")
    }

    // MARK: Gate wiring — every file tool consults the pin gate

    func testEveryFileToolRejectsWhenPinnedElsewhere() async throws {
        let pinned = FileToolSandboxGate(sandboxEnabled: { true },
                                         pinnedWorkspace: { ("/pinned/elsewhere", "hermes") })
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(atPath: dir) }

        for (name, handler, params) in fileToolCases(gate: pinned, dir: dir) {
            do {
                _ = try await handler.execute(parameters: params, workingDirectory: dir)
                XCTFail("\(name) must consult the sandbox pin gate")
            } catch {
                XCTAssertTrue(error.localizedDescription.contains("hermes"),
                              "\(name) must throw the GATE's error, got: \(error.localizedDescription)")
            }
        }
        XCTAssertFalse(FileManager.default.fileExists(
            atPath: (dir as NSString).appendingPathComponent("f.txt")),
            "a declined writeFile must not have touched the disk")
    }

    // MARK: Allowed paths stay allowed

    func testWriteAllowedInsideWorkspaceWithSandboxOn() async throws {
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(atPath: dir) }
        let gate = FileToolSandboxGate(sandboxEnabled: { true },
                                       pinnedWorkspace: { (nil, nil) })
        _ = try await WriteFileHandler(gate: gate).execute(
            parameters: ["path": "ok.txt", "content": "hello"], workingDirectory: dir)
        XCTAssertTrue(FileManager.default.fileExists(
            atPath: (dir as NSString).appendingPathComponent("ok.txt")))
    }

    func testWriteAllowedUnderPinnedShare() async throws {
        let root = try makeTempDir()
        defer { try? FileManager.default.removeItem(atPath: root) }
        let sub = (root as NSString).appendingPathComponent("sub")
        try FileManager.default.createDirectory(atPath: sub, withIntermediateDirectories: true)
        let gate = FileToolSandboxGate(sandboxEnabled: { true },
                                       pinnedWorkspace: { (root, "pi") })
        _ = try await WriteFileHandler(gate: gate).execute(
            parameters: ["path": "ok.txt", "content": "hello"], workingDirectory: sub)
        XCTAssertTrue(FileManager.default.fileExists(
            atPath: (sub as NSString).appendingPathComponent("ok.txt")))
    }

    func testWriteAllowedWithSandboxOffAndWorkspaceSet() async throws {
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(atPath: dir) }
        let gate = FileToolSandboxGate(sandboxEnabled: { false },
                                       pinnedWorkspace: { (nil, nil) })
        _ = try await WriteFileHandler(gate: gate).execute(
            parameters: ["path": "ok.txt", "content": "hello"], workingDirectory: dir)
        XCTAssertTrue(FileManager.default.fileExists(
            atPath: (dir as NSString).appendingPathComponent("ok.txt")))
    }
}
