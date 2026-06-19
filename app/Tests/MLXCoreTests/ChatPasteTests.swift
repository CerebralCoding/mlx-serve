import XCTest
@testable import MLXCore

/// Pins the pure classifier that routes a pasted (or dropped) file URL to the
/// same destination as the attach button: image / PDF / audio attachments, or a
/// document folder for mini-RAG. The classification is a pure static helper so
/// the paste routing is testable without a pasteboard or a rendered view.
///
/// Assertions compare `.rawValue` strings rather than enum cases — XCTAssertEqual's
/// leading-dot generic inference is flaky for some case names in this module's
/// import set, and string comparison sidesteps it entirely.
final class ChatPasteTests: XCTestCase {

    private func kind(_ ext: String, dir: Bool = false, audio: Bool = false) -> String {
        PasteFileKind.classify(ext: ext, isDirectory: dir, audioSupported: audio).rawValue
    }

    func testDirectoryIsFolderEvenWithAFileExtension() {
        // A directory literally named "notes.pdf" is still a folder.
        XCTAssertEqual(kind("pdf", dir: true), "folder")
        XCTAssertEqual(kind("", dir: true, audio: true), "folder")
    }

    func testPDFIsClassifiedRegardlessOfCase() {
        XCTAssertEqual(kind("pdf"), "pdf")
        XCTAssertEqual(kind("PDF"), "pdf")
    }

    func testCommonImageExtensionsAreImages() {
        for ext in ["png", "jpg", "jpeg", "heic", "gif", "tiff"] {
            XCTAssertEqual(kind(ext), "image", "\(ext) should classify as an image")
        }
    }

    func testAudioIsGatedOnModelSupport() {
        XCTAssertEqual(kind("wav", audio: true), "audio")
        // Model can't hear audio → don't attach it as audio.
        XCTAssertEqual(kind("wav", audio: false), "unhandled")
    }

    func testUnknownExtensionIsUnhandled() {
        XCTAssertEqual(kind("docx", audio: true), "unhandled")
        XCTAssertEqual(kind("", audio: true), "unhandled")
    }
}
