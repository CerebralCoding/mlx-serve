import XCTest
@testable import MLXCore

/// The welcome screen's "star us on GitHub" link must point at the real
/// repository and stay derived from `UpdateChecker.repo` (the app's single
/// source of truth for the GitHub repo) so the two can never drift apart.
final class WelcomeGitHubLinkTests: XCTestCase {
    func testStarURLPointsAtTheRepo() {
        XCTAssertEqual(WelcomeView.gitHubStarURL.absoluteString,
                       "https://github.com/\(UpdateChecker.repo)")
        XCTAssertEqual(WelcomeView.gitHubStarURL.absoluteString,
                       "https://github.com/ddalcu/mlx-serve")
    }
}
