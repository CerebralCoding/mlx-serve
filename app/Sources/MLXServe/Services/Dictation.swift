import Foundation

/// Pure join rule for dictation: how a finalized utterance from the speech
/// recognizer lands in an editable text field. Kept out of the view so the
/// concatenation behavior is unit-tested.
enum Dictation {
    /// Append `utterance` to `text` as natural prose: blank utterances are
    /// dropped, joins use a single space, and text already ending in
    /// whitespace (space or newline) gets no extra separator.
    static func appending(_ utterance: String, to text: String) -> String {
        let trimmed = utterance.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return text }
        if text.isEmpty || text.last?.isWhitespace == true { return text + trimmed }
        return text + " " + trimmed
    }
}
