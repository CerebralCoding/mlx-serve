import XCTest

// =============================================================================
// MARK: - Replica of GenExperiment (Views/StatusMenuView.swift)
//
// MLXCore is an executable target — tests can't @testable-import it, so the
// pure model behind the menu's "Experiments" accordion is replicated verbatim
// here (same pattern as MediaGenTests). Keep in sync with the production enum.
// =============================================================================

private enum GenExperimentReplica: String, CaseIterable, Identifiable {
    case image, video, audio

    var id: String { rawValue }

    var icon: String {
        switch self {
        case .image: "photo.on.rectangle.angled"
        case .video: "film.stack"
        case .audio: "waveform"
        }
    }

    var title: String {
        switch self {
        case .image: "ImageGen"
        case .video: "VideoGen"
        case .audio: "AudioGen"
        }
    }

    func help(ready: Bool) -> String {
        switch self {
        case .image: ready ? "Image Generation (FLUX.2)"
                           : "Image Generation — click to install dependencies"
        case .video: ready ? "Video Generation (LTX-Video 2.3)"
                           : "Video Generation — click to install dependencies"
        case .audio: ready ? "Audio Generation — neural TTS & voice cloning"
                           : "Audio Generation — click to install dependencies"
        }
    }

    func ready(imagesReady: Bool, fullyReady: Bool) -> Bool {
        switch self {
        case .image, .audio: imagesReady
        case .video: fullyReady
        }
    }
}

final class GenExperimentTests: XCTestCase {

    /// The accordion must contain exactly the three media tools, in display
    /// order. Adding a fourth gen feature without surfacing it here is the
    /// regression this pins.
    func testAccordionHasExactlyTheThreeMediaToolsInOrder() {
        XCTAssertEqual(GenExperimentReplica.allCases.map(\.title),
                       ["ImageGen", "VideoGen", "AudioGen"])
    }

    /// Readiness wiring: ImageGen/AudioGen ride the image stack; VideoGen needs
    /// the full LTX install. Mis-wiring this lets a tile look ready (and open
    /// to a failure) before its venv exists.
    func testReadinessGatesOnTheCorrectStatusFlag() {
        // Only the image stack installed:
        XCTAssertTrue(GenExperimentReplica.image.ready(imagesReady: true, fullyReady: false))
        XCTAssertTrue(GenExperimentReplica.audio.ready(imagesReady: true, fullyReady: false))
        XCTAssertFalse(GenExperimentReplica.video.ready(imagesReady: true, fullyReady: false),
                       "VideoGen must wait for the full install, not just the image stack")
        // Full install:
        XCTAssertTrue(GenExperimentReplica.video.ready(imagesReady: true, fullyReady: true))
        // Nothing installed:
        for e in GenExperimentReplica.allCases {
            XCTAssertFalse(e.ready(imagesReady: false, fullyReady: false))
        }
    }

    /// The tooltip must change with install state, and the not-ready copy must
    /// point the user at installing dependencies.
    func testHelpTextReflectsReadiness() {
        for e in GenExperimentReplica.allCases {
            XCTAssertNotEqual(e.help(ready: true), e.help(ready: false))
            XCTAssertTrue(e.help(ready: false).localizedCaseInsensitiveContains("install"),
                          "not-ready help for \(e.title) should prompt installation: \(e.help(ready: false))")
        }
    }

    /// Icons are the picker's visual identity — distinct per tool.
    func testIconsAreUnique() {
        let icons = GenExperimentReplica.allCases.map(\.icon)
        XCTAssertEqual(Set(icons).count, icons.count, "Each experiment needs a distinct icon")
    }
}
