import Foundation

/// Converts streamed Markdown answer text into something a speech synthesizer
/// should read aloud: emphasis/heading/list markers stripped, link labels kept
/// (URLs dropped), fenced code blocks collapsed to a short spoken placeholder,
/// and runs of whitespace tidied. Pure + stateless so it can be unit-tested and
/// re-run over the whole accumulated answer on every streaming delta.
enum SpeechSanitizer {
    /// A fenced code block is unpleasant to hear character-by-character; we read
    /// a short marker instead.
    static let codePlaceholder = "code block"

    static func spokenText(from markdown: String) -> String {
        var out: [String] = []
        var inFence = false

        for rawLine in markdown.components(separatedBy: "\n") {
            let trimmed = rawLine.trimmingCharacters(in: .whitespaces)

            // Fenced code blocks: toggle on ``` / ~~~ and emit one placeholder per block.
            if trimmed.hasPrefix("```") || trimmed.hasPrefix("~~~") {
                if !inFence { out.append(codePlaceholder) }
                inFence.toggle()
                continue
            }
            if inFence { continue }

            // Horizontal rules read as nothing.
            if isHorizontalRule(trimmed) { continue }

            let line = collapseSpaces(inlineClean(stripLeadingMarkers(rawLine)))
                .trimmingCharacters(in: .whitespaces)
            if !line.isEmpty { out.append(line) }
        }

        return out.joined(separator: "\n")
    }

    // MARK: - Pieces

    private static func stripLeadingMarkers(_ line: String) -> String {
        var s = String(line.drop(while: { $0 == " " || $0 == "\t" }))
        // Heading: one or more '#'
        s = s.replacingOccurrences(of: "^#{1,6}\\s+", with: "", options: .regularExpression)
        // Blockquote: one or more leading '>'
        s = s.replacingOccurrences(of: "^(>\\s?)+", with: "", options: .regularExpression)
        // Unordered list bullet
        s = s.replacingOccurrences(of: "^[-*+]\\s+", with: "", options: .regularExpression)
        // Ordered list "1." / "1)"
        s = s.replacingOccurrences(of: "^\\d+[.)]\\s+", with: "", options: .regularExpression)
        return s
    }

    private static func inlineClean(_ line: String) -> String {
        var s = line
        // Images first (drop), then links (keep label).
        s = regexReplace(s, "!\\[[^\\]]*\\]\\([^)]*\\)", "")
        s = regexReplace(s, "\\[([^\\]]*)\\]\\([^)]*\\)", "$1")
        // Bare URLs / emails are painful to hear character-by-character — reduce
        // them to a spoken word. Runs after Markdown links so labels survive.
        s = regexReplace(s, "(?i)\\b(?:https?://|www\\.)\\S+", "the link")
        s = regexReplace(s, "(?i)\\b[\\w.+-]+@[\\w.-]+\\.[a-z]{2,}\\b", "the email address")
        // Inline code: keep inner text, drop backticks.
        s = regexReplace(s, "`([^`]*)`", "$1")
        // Strikethrough, then bold, then italic (longest markers first).
        s = regexReplace(s, "~~([^~]+)~~", "$1")
        s = regexReplace(s, "\\*\\*([^*]+)\\*\\*", "$1")
        s = regexReplace(s, "__([^_]+)__", "$1")
        s = regexReplace(s, "\\*([^*]+)\\*", "$1")
        s = regexReplace(s, "(?<![A-Za-z0-9])_([^_]+)_(?![A-Za-z0-9])", "$1")
        return s
    }

    private static func collapseSpaces(_ s: String) -> String {
        regexReplace(s, "[ \\t]+", " ")
    }

    private static func isHorizontalRule(_ trimmed: String) -> Bool {
        guard trimmed.count >= 3 else { return false }
        let set = Set(trimmed)
        return set == ["-"] || set == ["*"] || set == ["_"]
    }

    private static func regexReplace(_ s: String, _ pattern: String, _ template: String) -> String {
        guard let re = try? NSRegularExpression(pattern: pattern) else { return s }
        let range = NSRange(s.startIndex..., in: s)
        return re.stringByReplacingMatches(in: s, range: range, withTemplate: template)
    }
}
