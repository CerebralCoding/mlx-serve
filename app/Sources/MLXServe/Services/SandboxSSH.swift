import Foundation

/// SSH plumbing for the Agent Sandbox guest (issue #89 follow-up: run agent
/// CLIs *inside* the sandbox).
///
/// One transport for everything: dropbear is baked into the guest image
/// (`ddalcu/agent-shell-mlxserve`), the host reaches it through a dedicated
/// `SandboxPortForwarder` mapping `localhost:<sshPort>` → guest `:22`, and
/// BOTH the embedded terminal (SwiftTerm spawning `/usr/bin/ssh` on a PTY) and
/// the user's own Terminal.app ride the exact same invocation. No
/// PTY-over-vsock, no vsock reverse tunnel — vz-agent is untouched.
///
/// Key material is app-owned and container-scoped: an ed25519 keypair plus OUR
/// known_hosts under `~/.mlx-serve/sandbox/ssh/`. The user's `~/.ssh` is never
/// read or written — guest host keys would otherwise pollute their
/// known_hosts, and `StrictHostKeyChecking=accept-new` means we never prompt.
/// Note accept-new only auto-accepts UNKNOWN hosts: a re-pull swaps the
/// dropbear host key at the SAME `[localhost]:<port>`, which would hard-fail
/// with the MITM banner — so `resetKnownHosts` wipes the file at each guest
/// boot (TOFU-per-boot; the trust anchor is our own loopback bind to our own
/// VM, not host-key continuity).
enum SandboxSSH {

    // MARK: Paths (host side)

    /// The client the embedded terminal spawns. Lives HERE (not in the view)
    /// so the host-escape audit tracks exactly one file. Note the inversion:
    /// this spawn is how a session ENTERS the sandbox guest, not an escape
    /// from it. MAS: system binaries are executable inside the App Sandbox
    /// and the key/known_hosts paths are container-scoped — live MAS behavior
    /// is a release-validation item (if blocked, gate Developer-ID-only with
    /// an explicit message, never silently broken).
    static let sshExecutablePath = "/usr/bin/ssh"

    static var sshDir: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".mlx-serve/sandbox/ssh", isDirectory: true)
    }
    static var privateKeyPath: String { sshDir.appendingPathComponent("id_ed25519").path }
    static var publicKeyPath: String { privateKeyPath + ".pub" }
    static var knownHostsPath: String { sshDir.appendingPathComponent("known_hosts").path }

    // MARK: Pure builders (unit tested)

    /// argv for `/usr/bin/ssh` — used verbatim by the embedded terminal.
    /// `remoteCommand` (a bootstrap script invocation) forces a tty (`-t`):
    /// agent CLIs are full-screen TUIs. Without one, ssh opens a login shell
    /// and allocates the tty itself.
    static func sshArgs(port: UInt16, keyPath: String, knownHostsPath: String,
                        remoteCommand: String? = nil) -> [String] {
        var args = [
            "-i", keyPath,
            "-p", String(port),
            "-o", "UserKnownHostsFile=\(knownHostsPath)",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "IdentitiesOnly=yes",
            "-o", "LogLevel=ERROR", // banners would garble the TUI handoff
        ]
        if remoteCommand != nil { args.append("-t") }
        args.append("root@localhost")
        if let remoteCommand { args.append(remoteCommand) }
        return args
    }

    /// One-line command for the "connect from your own terminal" copy row.
    /// Must open a session equivalent to the embedded terminal's (pinned
    /// against `sshArgs` by SandboxSSHTests). Paths are double-quoted — they
    /// live under $HOME, where spaces are possible.
    static func displayCommand(port: UInt16) -> String {
        "ssh -i \"\(privateKeyPath)\" -p \(port) "
            + "-o \"UserKnownHostsFile=\(knownHostsPath)\" "
            + "-o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes root@localhost"
    }

    /// First free ssh port from `base`, skipping `taken` and anything the
    /// `isFree` probe rejects (production passes a real TCP bind probe so a
    /// port owned by another app is skipped rather than discovered at connect
    /// time). Bounded scan — all-busy returns nil, never spins.
    static func allocateSshPort(taken: Set<UInt16>, base: UInt16 = 2222, tries: Int = 64,
                                isFree: (UInt16) -> Bool = portIsFree) -> UInt16? {
        for offset in 0..<tries {
            let candidate = base &+ UInt16(offset)
            if !taken.contains(candidate), isFree(candidate) { return candidate }
        }
        return nil
    }

    /// authorized_keys wants exactly one clean key line.
    static func authorizedKeysContent(publicKey: String) -> String {
        publicKey.trimmingCharacters(in: .whitespacesAndNewlines) + "\n"
    }

    /// `/root/.profile` injected into the rootfs before boot: a prompt that
    /// makes the sandbox unmistakable next to a host terminal, plus the PATH
    /// entries first-use agent installs land in. Login shells read `.profile`
    /// in both dash and bash (bash: when no `.bash_profile` exists — slim
    /// images don't ship one); dash renders `\w` literally, so it gets the
    /// plain variant.
    static func guestProfileContent() -> String {
        """
        # injected by mlx-serve (Agent Sandbox) — rewritten on every guest boot
        export PATH="$PATH:/root/.local/bin"
        if [ -n "$BASH_VERSION" ]; then
          export PS1='⬢ sandbox \\w # '
        else
          export PS1='⬢ sandbox # '
        fi
        """
    }

    // MARK: Effectful (host side)

    /// True when nothing on the host loopback owns `port` (both families —
    /// the forwarder binds 127.0.0.1 AND ::1, so either being taken makes the
    /// mapping fail).
    static func portIsFree(_ port: UInt16) -> Bool {
        func bindable(_ family: Int32, _ addr: UnsafeRawPointer, _ len: socklen_t) -> Bool {
            let fd = socket(family, SOCK_STREAM, 0)
            guard fd >= 0 else { return true } // can't probe → let the bind decide later
            defer { close(fd) }
            var one: Int32 = 1
            setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, socklen_t(MemoryLayout<Int32>.size))
            return bind(fd, addr.assumingMemoryBound(to: sockaddr.self), len) == 0
        }
        var v4 = sockaddr_in()
        v4.sin_family = sa_family_t(AF_INET)
        v4.sin_port = port.bigEndian
        v4.sin_addr.s_addr = inet_addr("127.0.0.1")
        v4.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        let v4ok = withUnsafePointer(to: &v4) { bindable(AF_INET, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) }
        guard v4ok else { return false }
        var v6 = sockaddr_in6()
        v6.sin6_family = sa_family_t(AF_INET6)
        v6.sin6_port = port.bigEndian
        v6.sin6_addr = in6addr_loopback
        v6.sin6_len = UInt8(MemoryLayout<sockaddr_in6>.size)
        return withUnsafePointer(to: &v6) { bindable(AF_INET6, $0, socklen_t(MemoryLayout<sockaddr_in6>.size)) }
    }

    /// Drop the app-owned known_hosts. Called at each guest boot: dropbear's
    /// host key lives IN the rootfs, so it changes on every re-pull while the
    /// endpoint stays `[localhost]:<port>` — and `accept-new` only accepts
    /// UNKNOWN hosts; a CHANGED key hard-fails with the MITM banner, which
    /// would strand every session right after the stale-image re-pull flow.
    /// TOFU-per-boot is the honest model here: the trust anchor is our own
    /// loopback bind to our own VM, not host-key continuity.
    static func resetKnownHosts(at path: String = knownHostsPath) {
        try? FileManager.default.removeItem(atPath: path)
    }

    /// Create the app-owned ed25519 keypair once (0700 dir, ssh-keygen with an
    /// empty passphrase). Idempotent — an existing key is left alone so the
    /// guest's authorized_keys stays valid across app updates.
    static func ensureKeypair() throws {
        let fm = FileManager.default
        if fm.fileExists(atPath: privateKeyPath), fm.fileExists(atPath: publicKeyPath) { return }
        try fm.createDirectory(at: sshDir, withIntermediateDirectories: true,
                               attributes: [.posixPermissions: 0o700])
        // A stale half-pair (key without pub or vice versa) can't be injected —
        // clear and regenerate.
        try? fm.removeItem(atPath: privateKeyPath)
        try? fm.removeItem(atPath: publicKeyPath)
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/usr/bin/ssh-keygen")
        proc.arguments = ["-t", "ed25519", "-N", "", "-q", "-C", "mlx-serve-sandbox", "-f", privateKeyPath]
        proc.standardOutput = Pipe(); proc.standardError = Pipe()
        try proc.run()
        proc.waitUntilExit()
        guard proc.terminationStatus == 0, fm.fileExists(atPath: publicKeyPath) else {
            throw AgentSandbox.SandboxError(message: "could not create the sandbox ssh key (ssh-keygen exit \(proc.terminationStatus))")
        }
    }

    /// Write the guest's `/root/.ssh/authorized_keys` (0700 dir / 0600 file)
    /// into the rootfs dir HOST-SIDE before boot — same precedent as
    /// `/.vz-init` and `/.vz-agent`. The guest runs as root, so ownership is
    /// already right (virtiofs presents host files to the guest's uid 0).
    static func injectAuthorizedKeys(rootfsDir: String) throws {
        let pub = try String(contentsOfFile: publicKeyPath, encoding: .utf8)
        let fm = FileManager.default
        let sshDirGuest = rootfsDir + "/root/.ssh"
        try fm.createDirectory(atPath: sshDirGuest, withIntermediateDirectories: true,
                               attributes: [.posixPermissions: 0o700])
        let dest = sshDirGuest + "/authorized_keys"
        try authorizedKeysContent(publicKey: pub).write(toFile: dest, atomically: true, encoding: .utf8)
        try fm.setAttributes([.posixPermissions: 0o600], ofItemAtPath: dest)
    }

    /// Write the distinct-prompt `/root/.profile` into the rootfs (host-side,
    /// pre-boot). Rewritten every boot — the sandbox prompt is a contract, not
    /// a default the image can drift away from.
    static func injectGuestProfile(rootfsDir: String) throws {
        try guestProfileContent().write(toFile: rootfsDir + "/root/.profile",
                                        atomically: true, encoding: .utf8)
    }
}
