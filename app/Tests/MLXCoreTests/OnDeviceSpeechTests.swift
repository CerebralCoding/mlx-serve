import XCTest
@testable import MLXCore

/// Pins Voice mode's on-device-only contract: when on-device recognition is
/// available we proceed (forcing it); when it isn't we surface an actionable
/// error instead of silently streaming audio to Apple's servers.
final class OnDeviceSpeechTests: XCTestCase {

    func testSupportedReturnsNilSoCallerProceeds() {
        XCTAssertNil(OnDeviceSpeech.unavailableMessage(supportsOnDevice: true, locale: "en_US"))
    }

    func testUnsupportedReturnsActionableMessageNamingTheLocale() {
        let msg = OnDeviceSpeech.unavailableMessage(supportsOnDevice: false, locale: "cy_GB")
        XCTAssertNotNil(msg)
        XCTAssertTrue(msg!.contains("cy_GB"))                 // tells the user which language
        XCTAssertTrue(msg!.contains("System Settings"))       // and where to fix it
        // Promises we won't fall back to the network.
        XCTAssertTrue(msg!.lowercased().contains("apple's servers") || msg!.lowercased().contains("on your mac"))
    }
}
