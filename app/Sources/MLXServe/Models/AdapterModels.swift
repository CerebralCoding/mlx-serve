import Foundation

struct ChatAdapterSelection: Codable, Equatable {
    let model: String
    let name: String
    var scale: Double = 1

    func forModel(_ id: String?, local: Bool) -> Self? {
        local && id == model && scale.isFinite && (0...2).contains(scale)
            && FineTunePresentation.validName(name) ? self : nil
    }
}

struct CustomizeRequest: Identifiable {
    let id = UUID()
    let path: String
    let title: String
    let chatId: UUID?
    var prompt = ""
}

struct FineTuneJob: Decodable, Equatable {
    enum Status: String, Decodable {
        case queued, running, completed, cancelled, failed
        var terminal: Bool { self == .completed || self == .cancelled || self == .failed }
    }
    enum Stage: String, Decodable {
        case synth, train, verify
        var title: String {
            switch self {
            case .synth: return "SYNTH"
            case .train: return "TRAIN"
            case .verify: return "CHECK"
            }
        }
    }
    struct Result: Decodable, Equatable {
        let verified: Bool
        let heldoutPrompts, refusedBefore, refusedAfter: Int
        let meanKl, maxKl: Double
        let prompt, before, after, recommendation: String
        enum CodingKeys: String, CodingKey {
            case verified, prompt, before, after, recommendation
            case heldoutPrompts = "heldout_prompts", refusedBefore = "refused_before"
            case refusedAfter = "refused_after", meanKl = "mean_kl", maxKl = "max_kl"
        }
    }
    let id, model, name: String
    let status: Status
    let stage: Stage
    let step, total: Int
    let loss: Double?
    let etaSeconds: Int?
    let cancelRequested: Bool
    let preview: String
    let errorName: String?
    let adapterDiscarded: Bool
    let result: Result?
    enum CodingKeys: String, CodingKey {
        case id, model, name, status, stage, step, total, loss, preview, result
        case etaSeconds = "eta_seconds", cancelRequested = "cancel_requested"
        case errorName = "error_name", adapterDiscarded = "adapter_discarded"
    }
}

struct FineTuneState {
    private(set) var job: FineTuneJob?
    private(set) var starting = false
    private(set) var stopping = false
    var error: String?
    var busy: Bool { starting || job.map { !$0.status.terminal } == true }

    mutating func begin() { self = Self(); starting = true }
    mutating func requestStop() { if busy { stopping = true } }
    mutating func receive(_ next: FineTuneJob) {
        if let job {
            guard job.id == next.id else { return }
            if job.status.terminal && !next.status.terminal { return }
        }
        starting = false
        job = next
        if next.status.terminal { stopping = false }
        error = next.errorName
    }
    mutating func startFailed(_ message: String) { starting = false; stopping = false; error = message }
    mutating func stopFailed(_ message: String) { stopping = false; error = message }
    mutating func jobMissing() {
        self = Self()
        error = "The server no longer has this job. It may have restarted. Check the model picker for a completed adapter before training again."
    }
}

enum FineTunePresentation {
    static func supports(_ model: LocalModel) -> Bool {
        ["llama", "qwen3", "gemma4", "gemma4_text", "gemma4_unified", "gemma4_unified_text", "qwen3_5", "qwen3_5_text"].contains(model.modelType) && model.quantFile == nil
            && (model.quantBits.map { [4, 16].contains($0) } ?? true)
            && (model.numExperts ?? 0) == 0
    }

    static func eta(_ seconds: Int?) -> String {
        guard let seconds else { return "Estimating…" }
        let value = max(0, seconds)
        return String(format: "%d:%02d", value / 60, value % 60)
    }
    static func validName(_ name: String) -> Bool {
        !name.isEmpty && name.utf8.count <= 128 && !name.hasPrefix(".")
            && name.utf8.allSatisfy { (48...57).contains($0) || (65...90).contains($0)
                || (97...122).contains($0) || [45, 95, 46].contains($0) }
    }
    static func isRefusal(_ text: String) -> Bool {
        let prefix = text.prefix(240).lowercased().replacingOccurrences(of: "’", with: "'")
        return ["i can't", "i cannot", "i won't", "i'm unable", "i am unable", "i must decline"]
            .contains { prefix.contains($0) }
    }
}
