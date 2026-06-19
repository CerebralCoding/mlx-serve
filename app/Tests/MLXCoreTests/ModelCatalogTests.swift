import XCTest
@testable import MLXCore

/// Pins the curated download catalog (`gemmaModelOptions`) and the menu-bar tray
/// subset (`gemmaModelOptionsTrayMenu`). The tray filter keys on the literal
/// substring `"4bit"` in `id` — an entry written with `"4-bit"` would silently
/// vanish from the tray, so the surfacing is tested through the real filter.
final class ModelCatalogTests: XCTestCase {

    func testQwen36MtpIsCuratedAndSurfacesInTray() {
        let repo = "ddalcu/Qwen3.6-27B-4bit-MTP-MLX-Serve"
        XCTAssertTrue(
            gemmaModelOptions.contains { $0.repoId == repo },
            "Qwen 3.6 27B (4-bit, MTP) must be in the curated catalog"
        )
        XCTAssertTrue(
            gemmaModelOptionsTrayMenu.contains { $0.repoId == repo },
            "Qwen 3.6 27B (4-bit, MTP) must surface in the menu-bar tray (id needs the \"4bit\" token)"
        )
    }

    /// Class guard: ids are the dictionary key into download state, so collisions
    /// silently merge two models' progress.
    func testCatalogIdsAreUnique() {
        let ids = gemmaModelOptions.map(\.id)
        XCTAssertEqual(ids.count, Set(ids).count, "duplicate id in gemmaModelOptions")
    }

    /// Class guard: every tray entry must satisfy the tray filter's own predicate,
    /// so a new entry can't claim tray membership without the right id token.
    func testTrayMenuMembersMatchFilterPredicate() {
        for opt in gemmaModelOptionsTrayMenu {
            XCTAssertTrue(
                opt.id.contains("4bit") || opt.id.contains("dsv4"),
                "tray entry \(opt.id) does not match the 4bit/dsv4 filter predicate"
            )
        }
    }
}
