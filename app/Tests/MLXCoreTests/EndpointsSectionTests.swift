import XCTest
@testable import MLXCore

/// Pins the URL that the tray's Endpoints accordion "open in browser" root item
/// navigates to. The construction is a pure static helper so it's testable
/// without rendering the SwiftUI view.
final class EndpointsSectionTests: XCTestCase {

    func testRootURLPointsAtServerRoot() {
        XCTAssertEqual(
            EndpointsSection.rootURL("http://localhost:11234")?.absoluteString,
            "http://localhost:11234/"
        )
    }

    /// `baseURL` may or may not already carry a trailing slash; either way the
    /// root link must resolve to exactly one — never `…//`.
    func testRootURLNormalizesToSingleTrailingSlash() {
        XCTAssertEqual(
            EndpointsSection.rootURL("http://localhost:8080/")?.absoluteString,
            "http://localhost:8080/"
        )
    }
}
