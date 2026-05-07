import Foundation

/// All user-tunable mlx-serve options, persisted to UserDefaults as JSON.
///
/// Split into two groups:
/// 1. Server-launch flags: passed on the `mlx-serve --serve` CLI; require a
///    server restart to take effect.
/// 2. Per-request defaults: injected into the JSON body of every chat request
///    by APIClient; apply on the next request, no restart needed.
///
/// The Settings UI introspects these via the `serverFlagFields` /
/// `requestDefaultFields` metadata to render labels, captions and the
/// "needs restart" badge automatically — every option carries its own
/// human-readable explainer.
struct ServerOptions: Codable, Equatable {
    // MARK: Server-launch flags (require restart)
    var host: String = "0.0.0.0"
    var port: UInt16 = 11234
    var ctxSize: Int = 0                // 0 = Auto (memory-bounded safe ceiling, capped at model max)
    var noVision: Bool = false
    var logLevel: LogLevel = .info
    var requestTimeout: Int = 300       // seconds; 0 = unlimited

    // Speculative decoding (server-launch flags)
    var enableMTP: Bool = false         // --mtp (off by default; opt-in)
    var enablePLD: Bool = true          // --pld is default-on now (CLI flips with --no-pld)
    var pldDraftLen: Int = 5
    var pldKeyLen: Int = 3
    var drafterPath: String = ""        // empty = no drafter
    var draftBlockSize: Int = 4

    // MARK: Per-request defaults (apply immediately, no restart)
    var defaultMaxTokens: Int = 4096
    var defaultTemperature: Double = 0.8
    var defaultTopP: Double = 0.95
    var defaultTopK: Int = 0            // 0 = disabled
    var defaultRepeatPenalty: Double = 1.0
    var defaultPresencePenalty: Double = 0.0
    var defaultReasoningBudget: Int = -1    // -1 = unlimited
    var defaultEnableThinking: Bool = false

    // Per-request overrides for spec-decode (default = follow server default)
    var perRequestEnableMTP: TriState = .auto
    var perRequestEnablePLD: TriState = .auto
    var perRequestEnableDrafter: TriState = .auto

    enum LogLevel: String, Codable, CaseIterable, Identifiable {
        case error, warn, info, debug
        var id: String { rawValue }
    }

    enum TriState: String, Codable, CaseIterable, Identifiable {
        case auto, on, off
        var id: String { rawValue }
        var label: String {
            switch self {
            case .auto: return "Auto (server default)"
            case .on:   return "Force on"
            case .off:  return "Force off"
            }
        }
        /// `nil` means leave the request body alone — the server's startup
        /// flag governs. `true`/`false` overrides per-request.
        var asOptionalBool: Bool? {
            switch self {
            case .auto: return nil
            case .on:   return true
            case .off:  return false
            }
        }
    }

    // MARK: Restart-detection helpers

    /// Compares only fields that affect the launched mlx-serve process, ignoring
    /// per-request defaults.
    func serverLaunchEquals(_ other: ServerOptions) -> Bool {
        host == other.host &&
        port == other.port &&
        ctxSize == other.ctxSize &&
        noVision == other.noVision &&
        logLevel == other.logLevel &&
        requestTimeout == other.requestTimeout &&
        enableMTP == other.enableMTP &&
        enablePLD == other.enablePLD &&
        pldDraftLen == other.pldDraftLen &&
        pldKeyLen == other.pldKeyLen &&
        drafterPath == other.drafterPath &&
        draftBlockSize == other.draftBlockSize
    }

    // MARK: CLI args builder

    /// Translate to the `mlx-serve` CLI flags. The leading `--model <path>` is
    /// passed by ServerManager (since the model path comes from AppState).
    func toCLIArgs() -> [String] {
        var args: [String] = [
            "--serve",
            "--port", "\(port)",
            "--host", host,
            "--log-level", logLevel.rawValue,
        ]
        if ctxSize > 0 {
            args += ["--ctx-size", "\(ctxSize)"]
        }
        if noVision {
            args += ["--no-vision"]
        }
        if requestTimeout != 300 {
            args += ["--timeout", "\(requestTimeout)"]
        }
        // Spec-decode: explicit flags either way so the server's CLI defaults
        // can't drift out from under the UI.
        args += [enableMTP ? "--mtp" : "--no-mtp"]
        args += [enablePLD ? "--pld" : "--no-pld"]
        args += ["--pld-draft-len", "\(pldDraftLen)"]
        args += ["--pld-key-len", "\(pldKeyLen)"]
        if !drafterPath.isEmpty {
            args += ["--drafter", drafterPath,
                     "--draft-block-size", "\(draftBlockSize)"]
        }
        return args
    }

    // MARK: Persistence (UserDefaults JSON)

    private static let storageKey = "serverOptions"

    static func load() -> ServerOptions {
        guard let data = UserDefaults.standard.data(forKey: storageKey),
              let opts = try? JSONDecoder().decode(ServerOptions.self, from: data) else {
            return ServerOptions()
        }
        return opts
    }

    func save() {
        guard let data = try? JSONEncoder().encode(self) else { return }
        UserDefaults.standard.set(data, forKey: Self.storageKey)
    }
}

// MARK: - UI introspection metadata

/// Describes a single tunable for the Settings UI: a label, an explainer,
/// and (where relevant) a flag indicating whether changes need a server restart.
struct ServerOptionField {
    let title: String
    let explainer: String
    let needsRestart: Bool
}

extension ServerOptions {
    /// Human-readable metadata for the server-launch fields, in the order the
    /// Settings UI should render them.
    static let serverFlagFields: [String: ServerOptionField] = [
        "host": .init(
            title: "Host",
            explainer: "Bind address. 0.0.0.0 lets other devices on your network reach the server; 127.0.0.1 is local-only.",
            needsRestart: true),
        "port": .init(
            title: "Port",
            explainer: "HTTP port the server listens on (default 11234). Change if it conflicts with another local service.",
            needsRestart: true),
        "ctxSize": .init(
            title: "Context size",
            explainer: "Maximum prompt + completion tokens. 0 means use the model's declared maximum. Higher values use more memory.",
            needsRestart: true),
        "noVision": .init(
            title: "Disable vision",
            explainer: "Skip loading the SigLIP image encoder. Saves ~3 GB of memory on text-only workloads.",
            needsRestart: true),
        "logLevel": .init(
            title: "Log level",
            explainer: "Server log verbosity. Use 'debug' to capture Jinja errors, KV cache hits and per-request token counts.",
            needsRestart: true),
        "requestTimeout": .init(
            title: "Request timeout (s)",
            explainer: "Max seconds a single HTTP request is allowed to take. 0 = unlimited. Long agent loops may need 600+.",
            needsRestart: true),
        "enableMTP": .init(
            title: "Enable MTP",
            explainer: "Multi-Token Prediction speculative decoding. Available only when the loaded model declares MTP layers in its config (Qwen3.5+, Qwen3-Next).",
            needsRestart: true),
        "enablePLD": .init(
            title: "Enable PLD (recommended)",
            explainer: "Prompt Lookup Decoding. Big wins on echo-heavy workloads (code editing, RAG, agent loops). The adaptive prompt-time gate auto-disables it on novel content.",
            needsRestart: true),
        "pldDraftLen": .init(
            title: "PLD draft length",
            explainer: "Maximum draft tokens proposed per PLD step (default 5). Higher = bigger speedup when matches hit, more wasted work when they miss.",
            needsRestart: true),
        "pldKeyLen": .init(
            title: "PLD key length",
            explainer: "N-gram match key length for PLD lookup (default 3). Shorter keys = more matches, lower precision.",
            needsRestart: true),
        "drafterPath": .init(
            title: "Drafter checkpoint",
            explainer: "Path to a Gemma 4 assistant drafter directory (gemma-4-*-it-assistant-bf16). Must pair with a Gemma 4 target. Empty = no drafter.",
            needsRestart: true),
        "draftBlockSize": .init(
            title: "Drafter block size",
            explainer: "Tokens per drafter round (default 4 = 3 drafter steps + 1 verify token).",
            needsRestart: true),
    ]

    /// Human-readable metadata for the per-request defaults.
    static let requestDefaultFields: [String: ServerOptionField] = [
        "defaultMaxTokens": .init(
            title: "Max tokens",
            explainer: "Default max_tokens to request per chat turn. Per-message overrides win when set.",
            needsRestart: false),
        "defaultTemperature": .init(
            title: "Temperature",
            explainer: "0 = deterministic greedy. 0.6–1.0 typical chat. Above 1.0 gets erratic.",
            needsRestart: false),
        "defaultTopP": .init(
            title: "Top-p",
            explainer: "Nucleus sampling threshold. 0.95 keeps all but the long tail. 1.0 disables top-p filtering.",
            needsRestart: false),
        "defaultTopK": .init(
            title: "Top-k",
            explainer: "Cap on candidate tokens per step. 0 = disabled (use top-p only).",
            needsRestart: false),
        "defaultRepeatPenalty": .init(
            title: "Repetition penalty",
            explainer: "Penalty multiplier for tokens already in the context. 1.0 = none. 1.1 is a typical anti-repeat setting.",
            needsRestart: false),
        "defaultPresencePenalty": .init(
            title: "Presence penalty",
            explainer: "Additive penalty per token already present in the context. 0 = none.",
            needsRestart: false),
        "defaultReasoningBudget": .init(
            title: "Reasoning budget",
            explainer: "Max thinking tokens per request. -1 = unlimited. Only applies when thinking is enabled.",
            needsRestart: false),
        "defaultEnableThinking": .init(
            title: "Enable thinking",
            explainer: "Default the chat client to send `enable_thinking: true`. Only models with reasoning support honor this.",
            needsRestart: false),
        "perRequestEnableMTP": .init(
            title: "Per-request MTP",
            explainer: "Auto = follow the server's --mtp setting. On/Off forces it for every request from this app.",
            needsRestart: false),
        "perRequestEnablePLD": .init(
            title: "Per-request PLD",
            explainer: "Auto = follow the server's --pld setting (and the adaptive gate). On/Off forces it.",
            needsRestart: false),
        "perRequestEnableDrafter": .init(
            title: "Per-request drafter",
            explainer: "Auto = follow the server. On/Off forces it. Only meaningful when --drafter is loaded.",
            needsRestart: false),
    ]
}
