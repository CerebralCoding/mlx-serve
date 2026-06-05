import Foundation
import AVFoundation

/// One selectable text-to-speech voice. Kept free of AVFoundation types so the
/// filtering/sorting/default-pick logic is unit-testable; `systemVoices()` is the
/// only AV-touching seam.
struct VoiceOption: Identifiable, Equatable, Hashable {
    let id: String        // AVSpeechSynthesisVoice.identifier
    let name: String      // e.g. "Samantha"
    let language: String  // e.g. "en-US"
    let quality: Int      // 1 = default/compact, 2 = enhanced, 3 = premium

    var qualityLabel: String {
        switch quality {
        case 3: return "Premium"
        case 2: return "Enhanced"
        default: return ""
        }
    }

    /// "Samantha — Premium" / "Daniel (en-GB)".
    var displayName: String {
        qualityLabel.isEmpty ? "\(name) (\(language))" : "\(name) — \(qualityLabel)"
    }
}

/// Builds the picker's voice list and chooses a sensible default. The default
/// macOS voice is the robotic compact one; we prefer the highest-quality voice
/// for the user's language so voice mode sounds natural out of the box.
enum VoiceCatalog {
    /// Compact/"default" voices (quality 1) are the robotic ones; we hide them.
    static let minQuality = 2

    /// Voices for `preferredLanguagePrefix` (e.g. "en"), best quality first then
    /// alphabetical, with low-quality compact voices removed. Falls back to all
    /// languages when none match the prefix, and to the compact voices only when
    /// nothing better is installed (so the picker is never empty).
    static func options(from all: [VoiceOption], preferredLanguagePrefix prefix: String) -> [VoiceOption] {
        func sortedByQuality(_ voices: [VoiceOption]) -> [VoiceOption] {
            voices.sorted { a, b in
                if a.quality != b.quality { return a.quality > b.quality }
                return a.name.localizedCaseInsensitiveCompare(b.name) == .orderedAscending
            }
        }
        let matching = all.filter { $0.language.lowercased().hasPrefix(prefix.lowercased()) }
        let pool = matching.isEmpty ? all : matching
        let good = pool.filter { $0.quality >= minQuality }
        return sortedByQuality(good.isEmpty ? pool : good)
    }

    /// Identifier of the best default voice, or nil if none are available.
    static func defaultVoiceId(from all: [VoiceOption], preferredLanguagePrefix prefix: String) -> String? {
        options(from: all, preferredLanguagePrefix: prefix).first?.id
    }

    /// The installed system voices (AV seam — not unit-tested).
    static func systemVoices() -> [VoiceOption] {
        AVSpeechSynthesisVoice.speechVoices().map { v in
            let q: Int
            switch v.quality {
            case .premium: q = 3
            case .enhanced: q = 2
            default: q = 1
            }
            return VoiceOption(id: v.identifier, name: v.name, language: v.language, quality: q)
        }
    }
}
