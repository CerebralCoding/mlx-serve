import XCTest
@testable import MLXCore

/// Pure-logic tests for the sandbox ssh plumbing: the `/usr/bin/ssh` argv
/// builder shared by the embedded terminal and the copyable "connect from your
/// terminal" row, ssh port allocation, and the guest-side file contents
/// (authorized_keys, the distinct sandbox prompt). No VM, no keygen — safe in CI.
final class SandboxSSHTests: XCTestCase {

    // MARK: ssh argv builder

    func testSshArgsCarryKeyPortAndIsolatedKnownHosts() {
        let args = SandboxSSH.sshArgs(port: 2222,
                                      keyPath: "/Users/d/.mlx-serve/sandbox/ssh/id_ed25519",
                                      knownHostsPath: "/Users/d/.mlx-serve/sandbox/ssh/known_hosts")
        XCTAssertEqual(args.last, "root@localhost", "the guest is reached via the loopback port mirror, never a LAN address")
        XCTAssertTrue(args.contains("-i"), "\(args)")
        XCTAssertTrue(args.contains("/Users/d/.mlx-serve/sandbox/ssh/id_ed25519"))
        XCTAssertEqual(args[args.firstIndex(of: "-p")!.advanced(by: 1)], "2222")
        // Our OWN known_hosts + accept-new: never a prompt, never a write into
        // the user's ~/.ssh (guest host keys would pollute it).
        XCTAssertTrue(args.contains("UserKnownHostsFile=/Users/d/.mlx-serve/sandbox/ssh/known_hosts"))
        XCTAssertTrue(args.contains("StrictHostKeyChecking=accept-new"))
        XCTAssertFalse(args.joined(separator: " ").contains("/.ssh/"),
                       "must never reference the user's ~/.ssh: \(args)")
    }

    func testSshArgsRemoteCommandRidesATtyAfterTheTarget() {
        let args = SandboxSSH.sshArgs(port: 2230, keyPath: "/k", knownHostsPath: "/kh",
                                      remoteCommand: "sh /.vz-bootstrap-pi")
        // Agent CLIs are full-screen TUIs — a remote command needs a forced tty.
        XCTAssertTrue(args.contains("-t"), "\(args)")
        let target = args.firstIndex(of: "root@localhost")!
        XCTAssertEqual(args[target.advanced(by: 1)], "sh /.vz-bootstrap-pi",
                       "the remote command must follow the target as ONE argv element")
        // A plain shell session must NOT force a command (login shell instead).
        XCTAssertFalse(SandboxSSH.sshArgs(port: 2230, keyPath: "/k", knownHostsPath: "/kh")
            .contains("-t"))
    }

    func testDisplayCommandIsOneCopyableLine() {
        let cmd = SandboxSSH.displayCommand(port: 2223)
        XCTAssertFalse(cmd.contains("\n"), "the copy row must be a single line")
        XCTAssertTrue(cmd.hasPrefix("ssh "), cmd)
        XCTAssertTrue(cmd.contains("-p 2223"), cmd)
        XCTAssertTrue(cmd.contains("root@localhost"), cmd)
        XCTAssertTrue(cmd.contains("StrictHostKeyChecking=accept-new"), cmd)
        // Paths live under $HOME (spaces possible) — they must be quoted.
        XCTAssertTrue(cmd.contains("\"") || !SandboxSSH.privateKeyPath.contains(" "),
                      "key/known_hosts paths must be quoted in the copyable command: \(cmd)")
    }

    func testDisplayCommandAndArgvAgreeOnTheOptionSet() {
        // The copy row and the embedded terminal must open equivalent sessions —
        // same key, same known_hosts, same host-key policy. Pin the two builders
        // against each other (the CLISetupInstructions cross-pin pattern).
        let argv = SandboxSSH.sshArgs(port: 2224,
                                      keyPath: SandboxSSH.privateKeyPath,
                                      knownHostsPath: SandboxSSH.knownHostsPath)
        let display = SandboxSSH.displayCommand(port: 2224)
        for needle in ["UserKnownHostsFile", "StrictHostKeyChecking=accept-new", "IdentitiesOnly=yes"] {
            XCTAssertTrue(argv.joined(separator: " ").contains(needle), "argv missing \(needle)")
            XCTAssertTrue(display.contains(needle), "display command missing \(needle)")
        }
    }

    // MARK: port allocation

    func testAllocateSshPortStartsAt2222AndSkipsTakenAndBusy() {
        // Explicit isFree everywhere: the DEFAULT probe binds real sockets,
        // and a live app instance holding its ssh mirror on 2222 made this
        // test see 2223 (caught live 2026-07-19 — the test ran beside a
        // running session). Unit tests never touch the host's port state.
        XCTAssertEqual(SandboxSSH.allocateSshPort(taken: [], isFree: { _ in true }), 2222)
        XCTAssertEqual(SandboxSSH.allocateSshPort(taken: [2222, 2223], isFree: { _ in true }), 2224)
        // A port nobody registered but the OS says is busy must be skipped too.
        XCTAssertEqual(SandboxSSH.allocateSshPort(taken: [2222], isFree: { $0 != 2223 }), 2224)
    }

    func testAllocateSshPortGivesUpAfterTheProbeWindow() {
        XCTAssertNil(SandboxSSH.allocateSshPort(taken: [], isFree: { _ in false }),
                     "all candidates busy → nil, never an infinite scan")
    }

    // MARK: guest-side file contents

    func testAuthorizedKeysContentIsTheTrimmedKeyPlusNewline() {
        let content = SandboxSSH.authorizedKeysContent(publicKey: "  ssh-ed25519 AAAA... mlx-serve\n\n")
        XCTAssertEqual(content, "ssh-ed25519 AAAA... mlx-serve\n")
    }

    func testGuestProfileSetsADistinctSandboxPrompt() {
        let profile = SandboxSSH.guestProfileContent()
        // The user will have host and sandbox terminals side by side — the
        // prompt must make the sandbox unmistakable.
        XCTAssertTrue(profile.contains("sandbox"), profile)
        XCTAssertTrue(profile.contains("PS1"), profile)
        // Installer-managed bins (~/.local/bin, npm globals) must be reachable
        // from a login shell — the bootstrap installs agents there.
        XCTAssertTrue(profile.contains("/root/.local/bin"), profile)
    }

    // MARK: known_hosts lifecycle (host-key churn across re-pulls)

    func testResetKnownHostsRemovesTheFileAndToleratesAbsence() throws {
        // dropbear's host key lives in the rootfs, so it CHANGES on every
        // re-pull while the endpoint stays [localhost]:<port> — and accept-new
        // only auto-accepts UNKNOWN hosts; a CHANGED key hard-fails ssh with
        // the MITM banner. Our known_hosts serves only our own guest, so it is
        // reset at each boot (TOFU-per-boot; the trust anchor is our loopback
        // bind, not key continuity).
        let path = NSTemporaryDirectory() + "sbx-kh-\(UUID().uuidString)"
        try "[localhost]:2222 ssh-ed25519 AAAA...stale\n".write(toFile: path, atomically: true, encoding: .utf8)
        SandboxSSH.resetKnownHosts(at: path)
        XCTAssertFalse(FileManager.default.fileExists(atPath: path))
        SandboxSSH.resetKnownHosts(at: path) // absent file → no-op, no throw
    }

    // MARK: image source cross-pin

    func testGuestImageSourceBakesDropbearIn() throws {
        // The ssh transport REQUIRES dropbear baked into the guest image; the
        // image source lives in-repo (containers/agent-shell-mlxserve) so the
        // two can't drift: an image rebuilt without dropbear would strand
        // every session on the stale-image error after the next pull.
        let dockerfile = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // MLXCoreTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // app
            .deletingLastPathComponent()  // repo root
            .appendingPathComponent("containers/agent-shell-mlxserve/Dockerfile")
        let text = try String(contentsOf: dockerfile, encoding: .utf8)
        XCTAssertTrue(text.contains("dropbear"),
                      "the guest image must ship dropbear — the Sandbox terminal's ssh transport depends on it")
    }

    // MARK: key material location

    func testKeyMaterialLivesUnderTheSandboxSshDirNeverUserSsh() {
        for path in [SandboxSSH.privateKeyPath, SandboxSSH.publicKeyPath, SandboxSSH.knownHostsPath] {
            XCTAssertTrue(path.contains(".mlx-serve/sandbox/ssh"), path)
            XCTAssertFalse(path.contains("/.ssh/"), path)
        }
        XCTAssertEqual(SandboxSSH.publicKeyPath, SandboxSSH.privateKeyPath + ".pub")
    }
}
