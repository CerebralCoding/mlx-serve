import XCTest
@testable import MLXCore

/// Pins voice-list filtering/sorting and the best-default pick used to avoid the
/// robotic compact voice.
final class VoiceCatalogTests: XCTestCase {

    private let voices = [
        VoiceOption(id: "en.samantha.compact",  name: "Samantha", language: "en-US", quality: 1),
        VoiceOption(id: "en.ava.premium",        name: "Ava",      language: "en-US", quality: 3),
        VoiceOption(id: "en.daniel.enhanced",    name: "Daniel",   language: "en-GB", quality: 2),
        VoiceOption(id: "fr.thomas.enhanced",    name: "Thomas",   language: "fr-FR", quality: 2),
        VoiceOption(id: "en.alex.enhanced",      name: "Alex",     language: "en-US", quality: 2),
    ]

    func testHidesLowQualityCompactVoices() {
        let out = VoiceCatalog.options(from: voices, preferredLanguagePrefix: "en")
        XCTAssertFalse(out.contains { $0.quality < 2 })             // no robotic compact voices
        XCTAssertFalse(out.contains { $0.id == "en.samantha.compact" })
        XCTAssertFalse(out.contains { $0.language.hasPrefix("fr") }) // wrong language excluded
        XCTAssertEqual(out.count, 3)                                 // Ava, Alex, Daniel
    }

    func testSortedByQualityThenName() {
        let out = VoiceCatalog.options(from: voices, preferredLanguagePrefix: "en")
        XCTAssertEqual(out.map(\.id), [
            "en.ava.premium",      // quality 3
            "en.alex.enhanced",    // quality 2, "Alex" < "Daniel"
            "en.daniel.enhanced",  // quality 2
        ])
    }

    func testFallsBackToAllLanguagesWhenNoLanguageMatch() {
        let out = VoiceCatalog.options(from: voices, preferredLanguagePrefix: "de")
        XCTAssertFalse(out.contains { $0.quality < 2 })  // still hides compact
        XCTAssertEqual(out.first?.id, "en.ava.premium")  // best quality first
        XCTAssertEqual(out.count, 4)                     // Ava, Alex, Daniel, Thomas (compact dropped)
    }

    func testKeepsCompactOnlyWhenNothingBetterExists() {
        // Default macOS installs ship only compact voices — never leave the picker
        // empty; show them rather than nothing.
        let onlyCompact = [
            VoiceOption(id: "en.fred",  name: "Fred",  language: "en-US", quality: 1),
            VoiceOption(id: "en.kathy", name: "Kathy", language: "en-US", quality: 1),
        ]
        let out = VoiceCatalog.options(from: onlyCompact, preferredLanguagePrefix: "en")
        XCTAssertEqual(out.map(\.id), ["en.fred", "en.kathy"])
        XCTAssertEqual(VoiceCatalog.defaultVoiceId(from: onlyCompact, preferredLanguagePrefix: "en"), "en.fred")
    }

    func testDefaultPicksHighestQualityForLanguage() {
        XCTAssertEqual(VoiceCatalog.defaultVoiceId(from: voices, preferredLanguagePrefix: "en"),
                       "en.ava.premium")
    }

    func testDefaultIsNilWhenNoVoices() {
        XCTAssertNil(VoiceCatalog.defaultVoiceId(from: [], preferredLanguagePrefix: "en"))
    }

    func testDisplayNameReflectsQuality() {
        XCTAssertEqual(VoiceOption(id: "x", name: "Ava", language: "en-US", quality: 3).displayName, "Ava — Premium")
        XCTAssertEqual(VoiceOption(id: "x", name: "Daniel", language: "en-GB", quality: 2).displayName, "Daniel — Enhanced")
        XCTAssertEqual(VoiceOption(id: "x", name: "Samantha", language: "en-US", quality: 1).displayName, "Samantha (en-US)")
    }
}
