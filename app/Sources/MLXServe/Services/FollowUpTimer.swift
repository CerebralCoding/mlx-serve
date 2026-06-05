import Foundation

/// A one-shot, cancellable timer for the voice assistant's follow-up window —
/// the grace period after an answer (or a bare wake phrase) during which the
/// user can speak again without repeating the wake word. Abstracted behind a
/// protocol so tests can fire it synchronously instead of waiting wall-clock.
@MainActor
protocol FollowUpTimer: AnyObject {
    /// Run `action` after `seconds`, replacing any previously scheduled action.
    func schedule(after seconds: TimeInterval, _ action: @escaping () -> Void)
    /// Cancel a pending action (no-op when nothing is scheduled).
    func cancel()
}

/// Production `FollowUpTimer` backed by a one-shot main-run-loop `Timer`.
@MainActor
final class RealFollowUpTimer: FollowUpTimer {
    private var timer: Timer?

    /// `nonisolated` so it can serve as a default argument (which Swift evaluates
    /// in a nonisolated context); the body only nil-initializes the timer.
    nonisolated init() {}

    func schedule(after seconds: TimeInterval, _ action: @escaping () -> Void) {
        cancel()
        timer = Timer.scheduledTimer(withTimeInterval: seconds, repeats: false) { _ in
            Task { @MainActor in action() }
        }
    }

    func cancel() {
        timer?.invalidate()
        timer = nil
    }
}
