import Foundation

/// Facts the model can't know on its own and must be told fresh each turn —
/// today the wall-clock date/time. Injected into agent and voice system prompts
/// so the assistant answers "what time/day is it" from reality instead of
/// hallucinating (and so it reasons about recency correctly). Pure → testable.
enum SystemGrounding {
    /// One sentence stating the current local date and time, with an instruction
    /// to trust it over the model's own guess.
    static func dateTimeLine(now: Date = Date()) -> String {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US")
        f.dateFormat = "EEEE, MMMM d, yyyy 'at' h:mm a zzz"
        return "The current date and time is \(f.string(from: now)). " +
            "Treat this as the present moment — do not guess the date or time from memory; " +
            "answer any date or time question from this."
    }
}
