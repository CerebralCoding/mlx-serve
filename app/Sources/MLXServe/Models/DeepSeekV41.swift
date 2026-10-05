import Foundation

/// DeepSeek-V4.1-Flash, served by the mlx-stream plugin (`deepseek_v41`): the trunk loads once and the routed
/// experts stream from SSD (EXL3 3.0 bpw). One place for what the app needs to offer it.
enum DeepSeekV41 {
    static let modelType = "deepseek_v41"
    static let catalogId = "dsv41-flash-exl3"
    static let artifactName = "DeepSeek-V4.1-Flash-streaming-repack-exl3-3.0bpw"
    static let repoId = "OpensourceWTF/" + artifactName

    /// The plugin admits its expert rows against this box ceiling: 112 GiB in decimal GB, which leaves macOS 16 GiB
    /// of a 128 GiB Mac. Only meaningful for this arch, so it is emitted only when it is the launched model.
    static let memoryCeilingGB = "120.259"
    static let wiredMarginGiB = "2"

    /// `--memory-ceiling-gb` sets the server's static GPU ceiling for every bill, so it rides only a V4.1 launch.
    static func launchArgs(modelType: String?) -> [String] {
        guard modelType == Self.modelType else { return [] }
        return ["--memory-ceiling-gb", memoryCeilingGB, "--wired-margin-gib", wiredMarginGiB]
    }

    /// The reply allowance the plugin bills beside a prompt: max_tokens 1,024 plus a 64-token pad.
    static let replyPad = 1_088

    /// `ctx_size` choices: prompts of 1K..512K plus the reply allowance, then the model's 1M limit (each one measured).
    /// The plugin bills every prompt up to `ctx_size` at load, so a larger choice admits fewer expert rows; none = 16K.
    static let ctxSizeChoices: [Int] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512].map { $0 * 1024 + replyPad } + [1_048_576]

    static func contextPresets(modelType: String?) -> [Int] {
        modelType == Self.modelType ? ctxSizeChoices : ContextSizeDisplay.presets
    }

    /// The `model_type` in a model directory's config.json, or nil (no config, a GGUF file).
    static func modelType(atModelPath path: String) -> String? {
        let url = URL(fileURLWithPath: path).appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: url),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        return obj["model_type"] as? String
    }

    /// The pack's files: the Engram tables live in `engram/` and the token map is a `.u32`, both read by the served
    /// load; the conversion receipts and the reference encoder are not.
    static let fileSelection = FileSelection(recursive: true, excludeSubstrings: ["receipts/", "encoding/"],
                                             extraExtensions: ["u32"])

    static func selection(forRepo repo: String) -> FileSelection? {
        repo == repoId ? fileSelection : nil
    }

    static let catalogEntry = GemmaModelOption(
        id: catalogId,
        displayName: "DeepSeek-V4.1-Flash (streaming repack, EXL3 3.0 bpw)",
        repoId: repoId,
        sizeEstimate: "~410 GB on disk (experts stream from SSD), needs 128 GB RAM",
        minHostRamBytes: 128 * (UInt64(1) << 30)
    )
}
