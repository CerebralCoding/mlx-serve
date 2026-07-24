import XCTest
@testable import MLXCore

/// Pins `reconciledModelSelection`, the pure selection repair `refreshModels`
/// runs after every rescan. Live bug: with the models directory deleted, the
/// old repair only swapped to another pickable model — with ZERO left it kept
/// the dead path in UserDefaults, so every start site (autostart, the LAN
/// share/discover boot, the tray Start button) launched `--model <gone>` and
/// the server died instantly with FileNotFound.
final class ModelSelectionReconcileTests: XCTestCase {

    func testValidSelectionIsKept() {
        XCTAssertEqual(
            reconciledModelSelection(current: "/m/b", pickablePaths: ["/m/a", "/m/b"]),
            "/m/b")
    }

    func testDanglingSelectionSwapsToFirstPickable() {
        XCTAssertEqual(
            reconciledModelSelection(current: "/m/deleted", pickablePaths: ["/m/a", "/m/b"]),
            "/m/a")
    }

    /// The regression case: dangling selection and nothing pickable left must
    /// CLEAR the selection, never keep the dead path.
    func testDanglingSelectionWithNothingLeftClears() {
        XCTAssertEqual(
            reconciledModelSelection(current: "/m/deleted", pickablePaths: []),
            "")
    }

    func testEmptySelectionAutoPicksFirst() {
        XCTAssertEqual(
            reconciledModelSelection(current: "", pickablePaths: ["/m/a"]),
            "/m/a")
    }

    func testEmptySelectionWithNoModelsStaysEmpty() {
        XCTAssertEqual(
            reconciledModelSelection(current: "", pickablePaths: []),
            "")
    }
}
