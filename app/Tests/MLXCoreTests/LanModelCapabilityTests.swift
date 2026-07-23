import XCTest
@testable import MLXCore

/// Pins `ModelInfo.lanAdvertises` — the tray empty-state and the "On Your
/// Network" picker count LAN chat models through it. Live bug (2026-07-21):
/// a peer running pre-26.7.11 rendered its loaded GGUF (embedded ds4/llama
/// engine, no chat_template in the header) with capabilities:[] — the tray
/// said "No models yet" while the user was actively chatting on the peer's
/// DeepSeek. Empty capabilities on a LAN entry means exactly that old-peer
/// bug (media entries always advertise their modality), so it counts as chat.
final class LanModelCapabilityTests: XCTestCase {

    private func model(_ name: String, caps: [String], peer: String?) -> ModelInfo {
        ModelInfo(name: name, quantBits: 0, layers: 0, hiddenSize: 0, vocabSize: 0,
                  contextLength: 0, modelMaxTokens: 0, capabilities: caps, lanPeer: peer)
    }

    func testAdvertisedCapabilityMatches() {
        let m = model("gemma@Peer", caps: ["chat"], peer: "Peer")
        XCTAssertTrue(m.lanAdvertises("chat"))
        XCTAssertFalse(m.lanAdvertises("image"))
    }

    /// The regression case: old peer under-reporting a loaded GGUF.
    func testOldPeerEmptyCapabilitiesCountsAsChat() {
        let m = model("DeepSeek-V4-Flash@M4Max", caps: [], peer: "M4Max")
        XCTAssertTrue(m.lanAdvertises("chat"))
        XCTAssertFalse(m.lanAdvertises("image"), "the empty-caps tolerance is chat-only")
    }

    func testLocalEntriesNeverMatch() {
        let m = model("local-model", caps: ["chat"], peer: nil)
        XCTAssertFalse(m.lanAdvertises("chat"))
    }

    func testMediaLanEntryStaysMedia() {
        let m = model("flux@Peer", caps: ["image"], peer: "Peer")
        XCTAssertTrue(m.lanAdvertises("image"))
        XCTAssertFalse(m.lanAdvertises("chat"))
    }
}
