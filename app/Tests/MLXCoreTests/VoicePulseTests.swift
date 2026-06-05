import XCTest
@testable import MLXCore

/// Pins the time-driven pulse math that replaced the imperative
/// `withAnimation(...).repeatForever(...)` breathe in the voice tray panel and
/// the orb.
///
/// Regression context: that repeating animation, kicked off in `.onAppear`
/// inside the `MenuBarExtra(.window)` popover, never settled and wedged
/// SwiftUI's event handling — the tray's `Button`s stopped responding to clicks
/// while AppKit pop-up controls (the model `Picker`, the voice `Menu`) kept
/// working from their own event-tracking loop ("the dropdown is still clickable
/// but nothing else is"). Driving the same breathe from a `TimelineView` + these
/// pure functions removes the long-lived animation; this test locks the curve so
/// nobody reintroduces a constant (dead) or out-of-range pulse.
final class VoicePulseTests: XCTestCase {

    func testPhaseStaysInUnitRange() {
        for i in 0..<400 {
            let p = VoicePulse.phase(at: Double(i) * 0.05)
            XCTAssertGreaterThanOrEqual(p, 0)
            XCTAssertLessThanOrEqual(p, 1)
        }
    }

    func testPhaseIsPeriodic() {
        let period = 2.2
        XCTAssertEqual(VoicePulse.phase(at: 0.7, period: period),
                       VoicePulse.phase(at: 0.7 + period, period: period),
                       accuracy: 1e-9)
    }

    func testPhaseActuallyVaries() {
        // The whole point of the fix: a *live* breathe, not a constant. A quarter
        // period in, the phase must have moved a lot.
        let period = 2.2
        let a = VoicePulse.phase(at: 0, period: period)
        let b = VoicePulse.phase(at: period / 4, period: period)
        XCTAssertGreaterThan(abs(a - b), 0.3)
    }

    func testDotOpacitySteadyWhenNotAnimating() {
        for i in 0..<50 {
            XCTAssertEqual(VoicePulse.dotOpacity(animating: false, at: Double(i) * 0.1), 1.0)
        }
    }

    func testDotOpacityBreathesBetweenFloorAndOne() {
        let floor = 0.35
        var sawLow = false, sawHigh = false
        for i in 0..<400 {
            let o = VoicePulse.dotOpacity(animating: true, at: Double(i) * 0.03, floor: floor)
            XCTAssertGreaterThanOrEqual(o, floor - 1e-9)
            XCTAssertLessThanOrEqual(o, 1 + 1e-9)
            if o < floor + 0.05 { sawLow = true }
            if o > 0.95 { sawHigh = true }
        }
        XCTAssertTrue(sawLow, "pulse should reach near the dim floor")
        XCTAssertTrue(sawHigh, "pulse should reach near full brightness")
    }

    func testOrbBreatheZeroWhenNotAnimating() {
        XCTAssertEqual(VoicePulse.orbBreathe(animating: false, at: 1.23), 0)
    }

    func testOrbBreatheWithinAmplitude() {
        let amp = 0.05
        for i in 0..<400 {
            let s = VoicePulse.orbBreathe(animating: true, at: Double(i) * 0.04, amplitude: amp)
            XCTAssertGreaterThanOrEqual(s, -1e-9)
            XCTAssertLessThanOrEqual(s, amp + 1e-9)
        }
    }
}
