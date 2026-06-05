import Foundation

/// Time-driven "breathe" math for the voice UI's pulsing indicators (the tray
/// status dot and the full-screen orb).
///
/// These used to breathe via an imperative `withAnimation(...).repeatForever(...)`
/// started in `.onAppear`. Hosted inside the `MenuBarExtra(.window)` popover that
/// never-settling animation wedged SwiftUI's event handling: plain `Button`s
/// stopped responding to clicks while AppKit pop-up controls (the model `Picker`,
/// the voice `Menu`) kept working from their own event-tracking loop — the "tray
/// locks up but the dropdown still opens" report.
///
/// The fix drives the identical visual from a `TimelineView` that computes each
/// frame's value with these pure functions, so no repeating `Animation` is ever
/// retained in the popover. Keeping the curve here (instead of inline in the
/// view) also makes it unit-testable — see `VoicePulseTests`.
enum VoicePulse {
    /// Normalized breathe phase in `0...1` at time `t` (a smooth sine, one full
    /// cycle per `period` seconds). `t` is any monotonic seconds value, e.g.
    /// `timelineDate.timeIntervalSinceReferenceDate`.
    static func phase(at t: TimeInterval, period: TimeInterval = 2.2) -> Double {
        guard period > 0 else { return 1 }
        return (sin(2 * .pi * t / period) + 1) / 2
    }

    /// Opacity for the tray status dot: a steady `1.0` when not animating (so a
    /// paused `TimelineView` renders a solid dot), otherwise a breathe between
    /// `floor` and `1.0`.
    static func dotOpacity(animating: Bool, at t: TimeInterval,
                           period: TimeInterval = 2.2, floor: Double = 0.35) -> Double {
        guard animating else { return 1 }
        return floor + (1 - floor) * phase(at: t, period: period)
    }

    /// Extra scale added to the orb's mic-level scale: `0` when not animating,
    /// otherwise a breathe between `0` and `amplitude`.
    static func orbBreathe(animating: Bool, at t: TimeInterval,
                           period: TimeInterval = 3.0, amplitude: Double = 0.05) -> Double {
        guard animating else { return 0 }
        return amplitude * phase(at: t, period: period)
    }
}
