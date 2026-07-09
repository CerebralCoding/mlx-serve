import Foundation
import SwiftUI

/// Detects third-party CLI coding agents installed on PATH and launches them
/// configured to talk to the local mlx-serve instance.
///
/// Detection uses a login zsh shell so nvm / asdf / pyenv / Homebrew paths are
/// resolved the same way the user's terminal sees them. Results are cached on
/// the main actor and refreshed on demand.
@MainActor
final class CLILauncher: ObservableObject {
    /// What the user has right now.
    @Published private(set) var available: [LauncherCLI] = []
    /// Keep rescanning off the initial launch path until we actually have a result.
    @Published private(set) var hasScanned = false

    private static let candidates: [LauncherCLI] = [
        .claudeCode,
        .pi,
        .opencode,
    ]

    init() {
        Task { await refresh() }
    }

    /// Re-scan PATH. Cheap — three `which` calls in a single shell invocation.
    func refresh() async {
        let found = await Self.detectInstalled()
        self.available = found
        self.hasScanned = true
    }

    /// Resolve installed binaries by running a single `command -v` sweep inside
    /// an **interactive** login zsh so user-specific PATH additions (nvm,
    /// Homebrew, ~/.local/bin, ~/.opencode/bin) are honored.
    ///
    /// Both `-i` and `-l` matter: a login shell only sources `.zprofile`/
    /// `.zlogin`, while most users put PATH mutations in `.zshrc` — which is
    /// sourced only by interactive shells. When the app is launched from
    /// Finder/LaunchServices the child process starts with a near-empty
    /// environment, so without `-i` we'd see none of the user's tools.
    ///
    /// Output is keyed (`name=path`) instead of positional so any stray
    /// stdout from `.zshrc` (e.g. `pyenv init`, version managers) can't
    /// misalign parsing.
    private static func detectInstalled() async -> [LauncherCLI] {
        let names = candidates.map { $0.binaryName }
        let script = names.map { "printf '%s=%s\\n' \($0) \"$(command -v \($0) 2>/dev/null)\"" }.joined(separator: "; ")

        let output = await Task.detached { () -> String in
            let proc = Process()
            proc.executableURL = URL(fileURLWithPath: "/bin/zsh")
            proc.arguments = ["-i", "-l", "-c", script]
            let pipe = Pipe()
            proc.standardOutput = pipe
            proc.standardError = Pipe()
            do {
                try proc.run()
                proc.waitUntilExit()
                let data = pipe.fileHandleForReading.readDataToEndOfFile()
                return String(data: data, encoding: .utf8) ?? ""
            } catch {
                return ""
            }
        }.value

        var resolvedByName: [String: String] = [:]
        for line in output.split(separator: "\n", omittingEmptySubsequences: true) {
            guard let eq = line.firstIndex(of: "=") else { continue }
            let key = String(line[..<eq])
            let value = String(line[line.index(after: eq)...]).trimmingCharacters(in: .whitespacesAndNewlines)
            guard !value.isEmpty else { continue }
            // Only keep keys we actually asked about — guards against any
            // stray `foo=bar` lines printed from user rc files.
            if names.contains(key) { resolvedByName[key] = value }
        }

        var result: [LauncherCLI] = []
        for cli in candidates {
            guard let path = resolvedByName[cli.binaryName],
                  FileManager.default.isExecutableFile(atPath: path) else { continue }
            var updated = cli
            updated.resolvedPath = path
            result.append(updated)
        }
        return result
    }

    /// Launch a CLI with a folder picker for its working directory.
    func launchWithPicker(_ cli: LauncherCLI, baseURL: String, servedModelId: String,
                          budget: AgentBudget.Budget) {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.canCreateDirectories = true // show the "New Folder" button
        panel.prompt = "Open"
        panel.message = "Select or create a working directory"
        let defaultWS = NSString(string: "~/.mlx-serve/workspace").expandingTildeInPath
        try? FileManager.default.createDirectory(atPath: defaultWS, withIntermediateDirectories: true)
        panel.directoryURL = URL(fileURLWithPath: defaultWS)
        guard panel.runModal() == .OK, let url = panel.url else { return }
        launch(cli, baseURL: baseURL, servedModelId: servedModelId,
               budget: budget, workingDirectory: url.path)
    }

    /// Write a shell script that sets the right env vars / config for the given
    /// CLI, then hand it to Terminal.app via NSWorkspace.
    func launch(_ cli: LauncherCLI, baseURL: String, servedModelId: String,
                budget: AgentBudget.Budget, workingDirectory: String?) {
        // pi and opencode both need their config files written before launch.
        // The budget travels with them: neither CLI reads `/v1/models`, so the
        // number we write here IS the context they believe the model has.
        cli.prepareConfig?(baseURL, servedModelId, budget)

        let cdLine = workingDirectory.map { "cd '\($0)'" } ?? ""
        let script = cli.scriptBody(baseURL, servedModelId, cdLine, budget)
        let fullScript = "#!/bin/zsh -l\n\(script)\n"

        let filename = "mlx-launch-\(cli.id).command"
        let path = NSTemporaryDirectory() + filename
        try? fullScript.write(toFile: path, atomically: true, encoding: String.Encoding.utf8)
        try? FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: path)
        NSWorkspace.shared.open(URL(fileURLWithPath: path))
    }
}

/// One row in the launcher dropdown. `resolvedPath` is filled in after detection.
struct LauncherCLI: Identifiable, Equatable {
    let id: String
    let displayName: String
    let binaryName: String
    let iconSystemName: String?
    let useClaudeIcon: Bool
    /// Optional side-effect invoked before the terminal script runs (e.g. write
    /// `~/.pi/agent/models.json`).
    let prepareConfig: (@Sendable (_ baseURL: String, _ servedModelId: String,
                                  _ budget: AgentBudget.Budget) -> Void)?
    /// Shell body that sets env vars and execs the CLI. Does NOT include the shebang.
    let scriptBody: (_ baseURL: String, _ servedModelId: String, _ cdLine: String,
                     _ budget: AgentBudget.Budget) -> String
    var resolvedPath: String = ""

    static func == (lhs: LauncherCLI, rhs: LauncherCLI) -> Bool { lhs.id == rhs.id }
}

extension LauncherCLI {

    /// Claude Code — Anthropic Messages API route. Uses env vars so the CLI
    /// talks to our `/v1/messages` endpoint with no code changes upstream.
    static let claudeCode = LauncherCLI(
        id: "claude",
        displayName: "Claude Code",
        binaryName: "claude",
        iconSystemName: nil,
        useClaudeIcon: true,
        prepareConfig: nil,
        scriptBody: { baseURL, model, cdLine, budget in
            """
            \(AgentConfigs.claudeCodeExports(baseURL: baseURL, model: model, budget: budget))
            \(cdLine)
            claude --model \(model)
            """
        }
    )

    /// pi (https://github.com/badlogic/pi-mono) — OpenAI-compatible. Needs a
    /// `~/.pi/agent/models.json` entry naming our provider.
    static let pi = LauncherCLI(
        id: "pi",
        displayName: "pi",
        binaryName: "pi",
        iconSystemName: "terminal",
        useClaudeIcon: false,
        prepareConfig: { baseURL, model, budget in
            let dir = NSString(string: "~/.pi/agent").expandingTildeInPath
            try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
            let config = AgentConfigs.piModelsJSON(baseURL: baseURL, model: model, budget: budget)
            let path = (dir as NSString).appendingPathComponent("models.json")
            try? config.write(toFile: path, atomically: true, encoding: .utf8)
        },
        scriptBody: { _, model, cdLine, _ in
            """
            \(cdLine)
            pi --provider mlx --model \(model)
            """
        }
    )

    /// opencode (https://opencode.ai) — registers a custom OpenAI-compatible
    /// provider via a dedicated `OPENCODE_CONFIG` file so the user's main
    /// `~/.config/opencode/opencode.json` is left untouched.
    static let opencode = LauncherCLI(
        id: "opencode",
        displayName: "OpenCode",
        binaryName: "opencode",
        iconSystemName: "chevron.left.forwardslash.chevron.right",
        useClaudeIcon: false,
        prepareConfig: { baseURL, model, budget in
            let config = AgentConfigs.opencodeJSON(baseURL: baseURL, model: model, budget: budget)
            let path = NSTemporaryDirectory() + "mlx-opencode-config.json"
            try? config.write(toFile: path, atomically: true, encoding: String.Encoding.utf8)
        },
        scriptBody: { _, model, cdLine, _ in
            let configPath = NSTemporaryDirectory() + "mlx-opencode-config.json"
            return """
            export OPENCODE_CONFIG='\(configPath)'
            \(cdLine)
            opencode --model mlx/\(model)
            """
        }
    )
}

// MARK: - UI

/// Launcher button for the menu bar. Renders either:
///   * nothing (when we've scanned and found no CLIs installed — the user
///     doesn't care about a feature they can't use),
///   * a single bordered button for one CLI (skips the extra click),
///   * a `Menu` dropdown for 2+ CLIs.
@MainActor
struct CLILauncherButton: View {
    let baseURL: String
    let servedModelId: String
    /// The running server's EFFECTIVE context (`/v1/models` meta.context_length).
    /// pi and opencode budget their own `max_tokens` against the number we write
    /// into their config, so passing nil here silently caps every agent session
    /// at the conservative fallback. See `AgentBudget`.
    let serverContextLength: Int?
    let isEnabled: Bool

    private var budget: AgentBudget.Budget { AgentBudget.forServerContext(serverContextLength) }

    @StateObject private var detector = CLILauncher()

    var body: some View {
        Group {
            if !detector.hasScanned {
                // Still scanning — reserve the space with a placeholder so the
                // footer doesn't reflow when scan finishes a moment later.
                Color.clear.frame(width: 0, height: 0)
            } else if detector.available.isEmpty {
                // None installed — hide entirely.
                EmptyView()
            } else if detector.available.count == 1, let only = detector.available.first {
                Button {
                    detector.launchWithPicker(only, baseURL: baseURL, servedModelId: servedModelId, budget: budget)
                } label: {
                    label(for: only).frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .disabled(!isEnabled)
                .help("Launch \(only.displayName)")
            } else {
                Menu {
                    ForEach(detector.available) { cli in
                        Button {
                            detector.launchWithPicker(cli, baseURL: baseURL, servedModelId: servedModelId, budget: budget)
                        } label: {
                            Label(cli.displayName, systemImage: cli.iconSystemName ?? "terminal")
                        }
                    }
                } label: {
                    HStack(spacing: 6) {
                        Image(systemName: "terminal")
                        Text("Code")
                    }
                    .frame(maxWidth: .infinity)
                }
                .menuStyle(.borderlessButton)
                .menuIndicator(.hidden)
                .padding(.horizontal, 6)
                .padding(.vertical, 3)
                .background(
                    RoundedRectangle(cornerRadius: 5)
                        .strokeBorder(.tertiary, lineWidth: 0.5)
                        .background(RoundedRectangle(cornerRadius: 5).fill(.regularMaterial))
                )
                .disabled(!isEnabled)
                .help("Launch coding agent (\(detector.available.map(\.displayName).joined(separator: ", ")))")
            }
        }
        .task { await detector.refresh() }
    }

    /// Tray launcher label — icon + "Code" so it sits alongside the Chat/Tasks buttons.
    @ViewBuilder
    private func label(for cli: LauncherCLI) -> some View {
        HStack(spacing: 6) {
            if cli.useClaudeIcon {
                ClaudeIcon(size: 12)
            } else {
                Image(systemName: cli.iconSystemName ?? "terminal")
            }
            Text("Code")
        }
    }
}
