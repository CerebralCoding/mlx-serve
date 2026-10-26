import SwiftUI
import XCTest
@testable import MLXCore

final class FineTuneViewTests: XCTestCase {
    @MainActor
    func testResultAndProgressFitTheCustomizationSheet() throws {
        let result = FineTuneJob.Result(verified: false, heldoutPrompts: 8, refusedBefore: 6, refusedAfter: 2,
            meanKl: 0.09, maxKl: 14.8, prompt: "When should I prune climbing roses?",
            before: "I'm unable to advise on that.",
            after: "Prune in late winter, removing dead stems and keeping the main canes.",
            recommendation: "Verification did not establish improvement within the drift limits. Try a lower strength and review the answers.")
        var state = FineTuneState()
        state.begin()
        state.receive(try JSONDecoder().decode(FineTuneJob.self, from: Data("""
        {"id":"ft-1","model":"m","name":"n","status":"running","stage":"train",
         "step":24,"total":100,"loss":1.23,"eta_seconds":125,"cancel_requested":false,
         "preview":"Prune in late winter, removing dead stems.","adapter_discarded":false}
        """.utf8)))
        for dark in [false, true] {
            try render(TrainingResultCard(result: result, name: .constant("garden-advice"), strength: .constant(0.75),
                busy: false, save: {}, discard: {}), name: "result", dark: dark)
            try render(TrainingProgressCard(state: state, stop: {}, retry: {}), name: "progress", dark: dark)
        }
    }

    @MainActor
    private func render<V: View>(_ content: V, name: String, dark: Bool) throws {
        let host = NSHostingView(rootView: content.padding(24).frame(width: 660)
            .background(Color(nsColor: .windowBackgroundColor)).environment(\.colorScheme, dark ? .dark : .light))
        host.appearance = NSAppearance(named: dark ? .darkAqua : .aqua)
        let size = host.fittingSize
        XCTAssertEqual(size.width, 660)
        XCTAssertGreaterThan(size.height, 150)
        XCTAssertLessThan(size.height, 900)
        host.frame = NSRect(origin: .zero, size: size)
        host.layoutSubtreeIfNeeded()
        let bitmap = try XCTUnwrap(host.bitmapImageRepForCachingDisplay(in: host.bounds))
        host.cacheDisplay(in: host.bounds, to: bitmap)
        let png = try XCTUnwrap(bitmap.representation(using: .png, properties: [:]))
        let app = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        try png.write(to: app.appendingPathComponent(".build/finetune-\(name)-\(dark ? "dark" : "light").png"))
    }
}
