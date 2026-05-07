import SwiftUI
import AppKit
import UniformTypeIdentifiers

/// Single-window form for the user-facing mlx-serve tunables. Bindings flow
/// through `appState.serverOptions`; AppState's `didSet` auto-saves to
/// UserDefaults.
///
/// Intentionally narrow surface: only the things end-users actually want to
/// tune. Host / port / request-timeout / log-level live in the CLI for
/// power users; per-request spec-decode overrides duplicate what the
/// Speculative Decoding toggles already express; "Enable thinking" lives on
/// the chat toolbar.
struct SettingsView: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var server: ServerManager

    var body: some View {
        VStack(spacing: 0) {
            if server.needsRestartFor(appState.serverOptions) {
                RestartBanner()
            }
            ScrollView {
                VStack(alignment: .leading, spacing: 24) {
                    SettingsSection(
                        title: "Server",
                        subtitle: "Server-launch flags. Restart the server to apply changes."
                    ) {
                        ServerSectionContent()
                    }
                    SettingsSection(
                        title: "Speculative Decoding",
                        subtitle: "Server-launch flags. Big throughput wins on echo-heavy work; gates auto-disable on novel content."
                    ) {
                        SpecDecodeSectionContent()
                    }
                    SettingsSection(
                        title: "Per-Request Defaults",
                        subtitle: "Apply on the next chat request — no restart needed."
                    ) {
                        RequestDefaultsSectionContent()
                    }
                }
                .padding(.horizontal, 24)
                .padding(.vertical, 20)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .navigationTitle("Settings")
    }
}

// MARK: - Restart banner

private struct RestartBanner: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var server: ServerManager

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: "arrow.clockwise.circle.fill")
                .font(.title2)
                .foregroundStyle(.orange)
            VStack(alignment: .leading, spacing: 2) {
                Text("Some changes require a server restart")
                    .font(.subheadline.weight(.semibold))
                Text("Click Restart Now to apply, or Discard to revert the unsaved server-launch fields.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Button("Restart Now") {
                let opts = appState.serverOptions
                let model = appState.selectedModelPath
                server.stop()
                if !model.isEmpty {
                    server.start(modelPath: model, options: opts)
                }
            }
            .buttonStyle(.borderedProminent)
            .disabled(appState.selectedModelPath.isEmpty)

            Button("Discard") {
                if let last = server.lastLaunchedOptions {
                    var current = appState.serverOptions
                    // Revert every server-launch field to the last-launched
                    // snapshot; per-request defaults are preserved.
                    current.host = last.host
                    current.port = last.port
                    current.ctxSize = last.ctxSize
                    current.noVision = last.noVision
                    current.logLevel = last.logLevel
                    current.requestTimeout = last.requestTimeout
                    current.enableMTP = last.enableMTP
                    current.enablePLD = last.enablePLD
                    current.pldDraftLen = last.pldDraftLen
                    current.pldKeyLen = last.pldKeyLen
                    current.drafterPath = last.drafterPath
                    current.draftBlockSize = last.draftBlockSize
                    appState.serverOptions = current
                }
            }
            .buttonStyle(.bordered)
            .disabled(server.lastLaunchedOptions == nil)
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 12)
        .background(Color.orange.opacity(0.10))
        .overlay(Divider(), alignment: .bottom)
    }
}

// MARK: - Section frame

private struct SettingsSection<Content: View>: View {
    let title: String
    let subtitle: String
    @ViewBuilder var content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.title3.weight(.semibold))
                Text(subtitle)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            VStack(alignment: .leading, spacing: 18) {
                content
            }
            .padding(16)
            .background(Color(NSColor.controlBackgroundColor))
            .clipShape(RoundedRectangle(cornerRadius: 10))
        }
    }
}

// MARK: - One row helper

private struct SettingsRow<Control: View>: View {
    let title: String
    let explainer: String
    /// True when this field has been changed since the running server was
    /// last launched — i.e. the user has edited it but not yet hit "Restart
    /// Now". Drives the orange restart icon. False (or always-false for
    /// per-request fields) hides the icon. We deliberately don't show it on
    /// every server-launch row by default — that's noisy when nothing has
    /// actually been changed yet.
    var isDirty: Bool = false
    @ViewBuilder var control: Control

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(alignment: .firstTextBaseline) {
                HStack(spacing: 6) {
                    Text(title)
                        .font(.body)
                    if isDirty {
                        Image(systemName: "arrow.clockwise.circle.fill")
                            .font(.caption)
                            .foregroundStyle(.orange)
                            .help("Restart the server to apply this change")
                    }
                }
                Spacer(minLength: 12)
                control
                    .frame(maxWidth: 280, alignment: .trailing)
            }
            Text(explainer)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}

/// Shared dirty-bit helper. Compares a single `ServerOptions` keypath against
/// the snapshot the server was last launched with. Returns false until the
/// server has been launched at least once (no baseline to compare against).
fileprivate struct ServerLaunchDirty {
    let current: ServerOptions
    let last: ServerOptions?

    func dirty<V: Equatable>(_ keyPath: KeyPath<ServerOptions, V>) -> Bool {
        guard let last else { return false }
        return current[keyPath: keyPath] != last[keyPath: keyPath]
    }
}

// MARK: - Server section

private struct ServerSectionContent: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var server: ServerManager

    private var meta: [String: ServerOptionField] { ServerOptions.serverFlagFields }
    private var dirty: ServerLaunchDirty {
        ServerLaunchDirty(current: appState.serverOptions, last: server.lastLaunchedOptions)
    }

    var body: some View {
        ContextSizeRow()
        if let m = meta["noVision"] {
            SettingsRow(
                title: m.title,
                explainer: m.explainer,
                isDirty: dirty.dirty(\.noVision)
            ) {
                Toggle("", isOn: $appState.serverOptions.noVision)
                    .labelsHidden()
                    .toggleStyle(.switch)
            }
        }
    }
}

/// Snapping slider over a fixed list of common context lengths, capped at the
/// model's declared maximum. The slider position 0 is "Auto" (= use model
/// default at load time). A secondary line shows the GPU-safe ceiling for
/// this Mac and warns when the chosen value exceeds it.
private struct ContextSizeRow: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var server: ServerManager

    private static let allPresets: [Int] = [
        0, 4_096, 8_192, 16_384, 32_768, 65_536,
        131_072, 262_144, 524_288, 1_048_576,
    ]

    /// Drop any preset larger than the model's `max_position_embeddings` so
    /// the slider can't pick a value the model would refuse. Auto (0) always
    /// stays. We deliberately use `modelMaxTokens` (the architectural cap from
    /// config.json) — NOT `contextLength` (which is the *running* server's
    /// effective context size and would change with this very setting).
    private var presets: [Int] {
        let modelMax = server.modelInfo?.modelMaxTokens ?? 0
        guard modelMax > 0 else { return Self.allPresets }
        return Self.allPresets.filter { $0 == 0 || $0 <= modelMax }
    }

    private var currentIndex: Int {
        let value = appState.serverOptions.ctxSize
        if let i = presets.firstIndex(of: value) { return i }
        // User has a value that doesn't match a preset (legacy data) — snap
        // visually to the closest non-Auto preset without mutating storage.
        guard value > 0 else { return 0 }
        var best = 1
        for i in 1..<presets.count where abs(presets[i] - value) < abs(presets[best] - value) {
            best = i
        }
        return best
    }

    private static func formatTokens(_ n: Int) -> String {
        if n == 0 { return "Auto" }
        if n >= 1_048_576 { return "\(n / 1_048_576)M" }
        if n >= 1024 { return "\(n / 1024)K" }
        return "\(n)"
    }

    private var isDirty: Bool {
        guard let last = server.lastLaunchedOptions else { return false }
        return appState.serverOptions.ctxSize != last.ctxSize
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(alignment: .firstTextBaseline) {
                HStack(spacing: 6) {
                    Text("Context size")
                        .font(.body)
                    if isDirty {
                        Image(systemName: "arrow.clockwise.circle.fill")
                            .font(.caption)
                            .foregroundStyle(.orange)
                            .help("Restart the server to apply this change")
                    }
                }
                Spacer(minLength: 12)
                HStack(spacing: 8) {
                    Slider(
                        value: Binding(
                            get: { Double(currentIndex) },
                            set: { raw in
                                let i = Int(raw.rounded())
                                let clamped = max(0, min(i, presets.count - 1))
                                appState.serverOptions.ctxSize = presets[clamped]
                            }
                        ),
                        in: 0...Double(max(1, presets.count - 1)),
                        step: 1
                    )
                    .frame(width: 200)
                    Text(Self.formatTokens(appState.serverOptions.ctxSize))
                        .font(.body.monospacedDigit())
                        .frame(minWidth: 56, alignment: .trailing)
                }
            }
            Text("Maximum prompt + completion tokens. \"Auto\" uses the model's declared maximum at load time. Higher values use more memory.")
                .font(.caption2)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            // Cap info: model max + GPU-safe max for this Mac. Visible only
            // when the server has reported them (after first model load).
            HStack(spacing: 12) {
                if let modelMax = server.modelInfo?.modelMaxTokens, modelMax > 0 {
                    capPill(
                        label: "Model max",
                        value: Self.formatTokens(modelMax),
                        warn: false
                    )
                }
                if let safeMax = server.memoryInfo?.maxSafeContext, safeMax > 0 {
                    let chosen = appState.serverOptions.ctxSize
                    let exceeds = chosen > 0 && chosen > safeMax
                    capPill(
                        label: "GPU-safe max",
                        value: Self.formatTokens(safeMax),
                        warn: exceeds
                    )
                }
            }
        }
    }

    @ViewBuilder
    private func capPill(label: String, value: String, warn: Bool) -> some View {
        let labelColor: Color = warn ? .orange : .secondary
        let valueColor: Color = warn ? .orange : .primary
        HStack(spacing: 4) {
            Text(label)
                .font(.caption2)
                .foregroundStyle(labelColor)
            Text(value)
                .font(.caption2.monospacedDigit().weight(.medium))
                .foregroundStyle(valueColor)
        }
        .padding(.horizontal, 6)
        .padding(.vertical, 2)
        .background((warn ? Color.orange : Color.secondary).opacity(0.10))
        .clipShape(Capsule())
    }
}

// MARK: - Spec-decode section

private struct SpecDecodeSectionContent: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var server: ServerManager

    private var meta: [String: ServerOptionField] { ServerOptions.serverFlagFields }
    private var dirty: ServerLaunchDirty {
        ServerLaunchDirty(current: appState.serverOptions, last: server.lastLaunchedOptions)
    }

    /// Drafter rows are intentionally hidden from Settings for now — PLD gives
    /// almost the same throughput on echo-heavy workloads without the drafter
    /// pairing constraints (Gemma-4-only, must match target architecture). The
    /// `drafterPath` / `draftBlockSize` fields stay in `ServerOptions` so users
    /// who set them via CLI keep working; we just don't surface them here.

    var body: some View {
        let opts = $appState.serverOptions
        // MTP availability gate: stays disabled while the server is stopped
        // (we don't know yet) and when the loaded model's config doesn't
        // declare MTP layers. We don't auto-clear `enableMTP` — we just keep
        // the toggle inert. That way nothing changes server-side until the
        // user explicitly toggles it after loading a supporting model.
        let mtpAvailable = server.modelInfo?.supportsMTP ?? false
        let mtpExplainerSuffix = mtpAvailable
            ? ""
            : (server.modelInfo == nil
                ? " (start the server to detect MTP support)"
                : " (loaded model has no MTP layers)")

        if let m = meta["enablePLD"] {
            SettingsRow(
                title: m.title,
                explainer: m.explainer,
                isDirty: dirty.dirty(\.enablePLD)
            ) {
                Toggle("", isOn: opts.enablePLD)
                    .labelsHidden()
                    .toggleStyle(.switch)
            }
        }
        if let m = meta["pldDraftLen"] {
            SettingsRow(
                title: m.title,
                explainer: m.explainer,
                isDirty: dirty.dirty(\.pldDraftLen)
            ) {
                Stepper(value: opts.pldDraftLen, in: 1...16) {
                    Text("\(appState.serverOptions.pldDraftLen)")
                        .font(.body.monospacedDigit())
                }
                .disabled(!appState.serverOptions.enablePLD)
            }
        }
        if let m = meta["pldKeyLen"] {
            SettingsRow(
                title: m.title,
                explainer: m.explainer,
                isDirty: dirty.dirty(\.pldKeyLen)
            ) {
                Stepper(value: opts.pldKeyLen, in: 1...8) {
                    Text("\(appState.serverOptions.pldKeyLen)")
                        .font(.body.monospacedDigit())
                }
                .disabled(!appState.serverOptions.enablePLD)
            }
        }
        if let m = meta["enableMTP"] {
            SettingsRow(
                title: m.title,
                explainer: m.explainer + mtpExplainerSuffix,
                isDirty: dirty.dirty(\.enableMTP)
            ) {
                Toggle("", isOn: opts.enableMTP)
                    .labelsHidden()
                    .toggleStyle(.switch)
                    .disabled(!mtpAvailable)
            }
        }
    }
}

// MARK: - Per-request defaults section

private struct RequestDefaultsSectionContent: View {
    @EnvironmentObject var appState: AppState

    private var meta: [String: ServerOptionField] { ServerOptions.requestDefaultFields }

    /// Snapping presets for Max Tokens. Powers of 2 from 256 up to 256K cover
    /// every realistic per-turn budget — short replies, agent loops, big code
    /// generations.
    private static let maxTokensPresets: [Int] = [
        256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144,
    ]

    /// Snapping presets for Reasoning Budget. Position 0 is the special
    /// "Unlimited" sentinel (-1); the rest are powers of 2 from 256 up to 32K.
    private static let reasoningPresets: [Int] = [
        -1, 0, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
    ]

    private static func formatTokens(_ n: Int) -> String {
        if n >= 1_048_576 { return "\(n / 1_048_576)M" }
        if n >= 1024 { return "\(n / 1024)K" }
        return "\(n)"
    }

    var body: some View {
        let opts = $appState.serverOptions

        // Max Tokens — snapping slider
        if let m = meta["defaultMaxTokens"] {
            SettingsRow(title: m.title, explainer: m.explainer) {
                snappingSlider(
                    presets: Self.maxTokensPresets,
                    current: appState.serverOptions.defaultMaxTokens,
                    set: { appState.serverOptions.defaultMaxTokens = $0 },
                    label: Self.formatTokens(appState.serverOptions.defaultMaxTokens)
                )
            }
        }
        if let m = meta["defaultTemperature"] {
            SettingsRow(title: m.title, explainer: m.explainer) {
                HStack(spacing: 8) {
                    Slider(value: opts.defaultTemperature, in: 0...2, step: 0.05)
                    Text(String(format: "%.2f", appState.serverOptions.defaultTemperature))
                        .font(.body.monospacedDigit())
                        .frame(minWidth: 36, alignment: .trailing)
                }
            }
        }
        if let m = meta["defaultTopP"] {
            SettingsRow(title: m.title, explainer: m.explainer) {
                HStack(spacing: 8) {
                    Slider(value: opts.defaultTopP, in: 0.1...1.0, step: 0.01)
                    Text(String(format: "%.2f", appState.serverOptions.defaultTopP))
                        .font(.body.monospacedDigit())
                        .frame(minWidth: 36, alignment: .trailing)
                }
            }
        }
        if let m = meta["defaultTopK"] {
            SettingsRow(title: m.title, explainer: m.explainer) {
                Stepper(value: opts.defaultTopK, in: 0...1000) {
                    Text(appState.serverOptions.defaultTopK == 0
                         ? "Disabled"
                         : "\(appState.serverOptions.defaultTopK)")
                        .font(.body.monospacedDigit())
                }
            }
        }
        if let m = meta["defaultRepeatPenalty"] {
            SettingsRow(title: m.title, explainer: m.explainer) {
                HStack(spacing: 8) {
                    Slider(value: opts.defaultRepeatPenalty, in: 1.0...2.0, step: 0.01)
                    Text(String(format: "%.2f", appState.serverOptions.defaultRepeatPenalty))
                        .font(.body.monospacedDigit())
                        .frame(minWidth: 40, alignment: .trailing)
                }
            }
        }
        if let m = meta["defaultPresencePenalty"] {
            SettingsRow(title: m.title, explainer: m.explainer) {
                HStack(spacing: 8) {
                    Slider(value: opts.defaultPresencePenalty, in: 0.0...2.0, step: 0.01)
                    Text(String(format: "%.2f", appState.serverOptions.defaultPresencePenalty))
                        .font(.body.monospacedDigit())
                        .frame(minWidth: 40, alignment: .trailing)
                }
            }
        }
        // Reasoning Budget — snapping slider; position 0 is the "Unlimited"
        // sentinel (-1).
        if let m = meta["defaultReasoningBudget"] {
            SettingsRow(title: m.title, explainer: m.explainer) {
                snappingSlider(
                    presets: Self.reasoningPresets,
                    current: appState.serverOptions.defaultReasoningBudget,
                    set: { appState.serverOptions.defaultReasoningBudget = $0 },
                    label: appState.serverOptions.defaultReasoningBudget < 0
                        ? "Unlimited"
                        : Self.formatTokens(appState.serverOptions.defaultReasoningBudget)
                )
            }
        }
    }

    /// Build a snapping slider over a discrete preset list. The slider's float
    /// value is the index into `presets`; rounding pins to the nearest entry.
    /// `label` is the textual readout shown next to the slider.
    @ViewBuilder
    private func snappingSlider(
        presets: [Int],
        current: Int,
        set: @escaping (Int) -> Void,
        label: String
    ) -> some View {
        let safePresets = presets.isEmpty ? [0] : presets
        let currentIdx = Self.closestIndex(in: safePresets, to: current)
        HStack(spacing: 8) {
            Slider(
                value: Binding(
                    get: { Double(currentIdx) },
                    set: { raw in
                        let i = Int(raw.rounded())
                        let clamped = max(0, min(i, safePresets.count - 1))
                        set(safePresets[clamped])
                    }
                ),
                in: 0...Double(max(1, safePresets.count - 1)),
                step: 1
            )
            .frame(width: 200)
            Text(label)
                .font(.body.monospacedDigit())
                .frame(minWidth: 70, alignment: .trailing)
        }
    }

    /// Find the index of the preset closest to `value`, so a stored value not
    /// on the snap grid still positions the slider sensibly.
    private static func closestIndex(in presets: [Int], to value: Int) -> Int {
        if let exact = presets.firstIndex(of: value) { return exact }
        var best = 0
        for i in 1..<presets.count where abs(presets[i] - value) < abs(presets[best] - value) {
            best = i
        }
        return best
    }
}
