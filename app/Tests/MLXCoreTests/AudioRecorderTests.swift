import XCTest
@testable import MLXCore

/// Covers the pure, device-independent piece of AudioRecorder: serializing
/// captured float samples to the float32-LE wire format the server frames into
/// audio tokens. (Live mic capture itself can't run in a unit test.)
final class AudioRecorderTests: XCTestCase {

    func testPcmDataLengthIsFourBytesPerSample() {
        let samples: [Float] = [0, 0.5, -0.25, 1, -1]
        let data = AudioRecorder.pcmData(from: samples)
        XCTAssertEqual(data.count, samples.count * 4)
    }

    func testPcmDataIsLittleEndianRoundTrip() {
        let samples: [Float] = [0.0, 0.5, -0.5, 0.125, -0.875]
        let data = AudioRecorder.pcmData(from: samples)
        // Reinterpret the bytes the same way the server does (raw float32-LE).
        let decoded: [Float] = data.withUnsafeBytes { raw in
            Array(raw.bindMemory(to: Float.self))
        }
        XCTAssertEqual(decoded, samples)
    }

    func testEmptySamplesProduceEmptyData() {
        XCTAssertEqual(AudioRecorder.pcmData(from: []).count, 0)
    }

    func testRecordedPcmWrapsIntoChatAudioWithCorrectDuration() {
        // 16000 samples = exactly 1 s at the recorder's 16 kHz target rate.
        let samples = [Float](repeating: 0, count: 16_000)
        let clip = ChatAudio(name: "mic", pcm: AudioRecorder.pcmData(from: samples))
        XCTAssertEqual(clip.sampleCount, 16_000)
        XCTAssertEqual(clip.durationSeconds, 1.0, accuracy: 1e-6)
    }
}
