import XCTest
@testable import MLXCore

/// Validates that AudioPreprocessor decodes a real audio file into the exact PCM
/// shape the Gemma 4 12B unified audio embedder expects: float32-LE 16 kHz mono.
final class AudioPreprocessorTests: XCTestCase {

    /// Write a minimal 16-bit PCM WAV (mono) at `sampleRate` with `frames`
    /// samples of a quiet sine, return its temp URL.
    private func writeWav(frames: Int, sampleRate: Int) throws -> URL {
        let bytesPerSample = 2
        let dataBytes = frames * bytesPerSample
        var d = Data()
        func u32(_ v: UInt32) -> Data { withUnsafeBytes(of: v.littleEndian) { Data($0) } }
        func u16(_ v: UInt16) -> Data { withUnsafeBytes(of: v.littleEndian) { Data($0) } }
        d.append("RIFF".data(using: .ascii)!)
        d.append(u32(UInt32(36 + dataBytes)))
        d.append("WAVE".data(using: .ascii)!)
        d.append("fmt ".data(using: .ascii)!)
        d.append(u32(16))                              // PCM fmt chunk size
        d.append(u16(1))                               // PCM
        d.append(u16(1))                               // mono
        d.append(u32(UInt32(sampleRate)))
        d.append(u32(UInt32(sampleRate * bytesPerSample))) // byte rate
        d.append(u16(UInt16(bytesPerSample)))          // block align
        d.append(u16(16))                              // bits per sample
        d.append("data".data(using: .ascii)!)
        d.append(u32(UInt32(dataBytes)))
        for i in 0..<frames {
            let s = Int16(8000.0 * sin(2.0 * Double.pi * 440.0 * Double(i) / Double(sampleRate)))
            d.append(u16(UInt16(bitPattern: s)))
        }
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("g4pp_\(UUID().uuidString).wav")
        try d.write(to: url)
        return url
    }

    func testDecodes16kMonoPassThrough() throws {
        let frames = 8_000 // 0.5 s @ 16 kHz
        let url = try writeWav(frames: frames, sampleRate: 16_000)
        defer { try? FileManager.default.removeItem(at: url) }

        guard let pcm = AudioPreprocessor.preprocess(url: url) else {
            return XCTFail("preprocess returned nil for a valid 16 kHz mono WAV")
        }
        XCTAssertEqual(pcm.count % 4, 0, "output must be whole float32 samples")
        let n = pcm.count / 4
        // No resampling needed at 16 kHz; allow a tiny tolerance for converter priming.
        XCTAssertEqual(Double(n), Double(frames), accuracy: 256, "decoded sample count ≈ input frames")
    }

    func testResamples44kDownTo16k() throws {
        let url = try writeWav(frames: 44_100, sampleRate: 44_100) // 1.0 s @ 44.1 kHz
        defer { try? FileManager.default.removeItem(at: url) }

        guard let pcm = AudioPreprocessor.preprocess(url: url) else {
            return XCTFail("preprocess returned nil for a 44.1 kHz WAV")
        }
        let n = pcm.count / 4
        // 1 second resampled to 16 kHz → ~16000 samples.
        XCTAssertEqual(Double(n), 16_000, accuracy: 800, "≈1s of audio → ~16k samples at 16 kHz")
    }

    func testReturnsNilForNonAudioFile() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("g4pp_\(UUID().uuidString).txt")
        try "not audio".data(using: .utf8)!.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        XCTAssertNil(AudioPreprocessor.preprocess(url: url))
    }
}
