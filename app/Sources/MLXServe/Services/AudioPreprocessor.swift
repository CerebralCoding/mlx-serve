import Foundation
import AVFoundation

/// Decodes an audio file into the raw PCM the Gemma 4 12B unified audio embedder
/// expects: little-endian float32 mono samples at 16 kHz. Any format AVFoundation
/// can read (wav/mp3/m4a/aiff/caf/flac) is resampled + downmixed to that shape.
enum AudioPreprocessor {
    static let targetSampleRate: Double = 16_000

    /// Returns float32-LE 16 kHz mono PCM bytes, or nil if the file can't be
    /// read/decoded. Caps the clip at `maxSeconds` to bound prompt length.
    static func preprocess(url: URL, maxSeconds: Double = 120) -> Data? {
        guard let file = try? AVAudioFile(forReading: url) else { return nil }
        let srcFormat = file.processingFormat
        guard srcFormat.sampleRate > 0, file.length > 0 else { return nil }

        guard let dstFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: targetSampleRate,
            channels: 1,
            interleaved: false
        ) else { return nil }

        // Read the whole source file into one buffer.
        let srcFrames = AVAudioFrameCount(min(file.length, Int64(srcFormat.sampleRate * maxSeconds)))
        guard srcFrames > 0,
              let srcBuf = AVAudioPCMBuffer(pcmFormat: srcFormat, frameCapacity: srcFrames) else { return nil }
        do { try file.read(into: srcBuf, frameCount: srcFrames) } catch { return nil }
        guard srcBuf.frameLength > 0 else { return nil }

        guard let converter = AVAudioConverter(from: srcFormat, to: dstFormat) else { return nil }
        let ratio = targetSampleRate / srcFormat.sampleRate
        let dstCapacity = AVAudioFrameCount(Double(srcBuf.frameLength) * ratio) + 4096
        guard let dstBuf = AVAudioPCMBuffer(pcmFormat: dstFormat, frameCapacity: dstCapacity) else { return nil }

        var consumed = false
        var convError: NSError?
        let status = converter.convert(to: dstBuf, error: &convError) { _, outStatus in
            if consumed {
                outStatus.pointee = .endOfStream
                return nil
            }
            consumed = true
            outStatus.pointee = .haveData
            return srcBuf
        }
        if status == .error || convError != nil { return nil }

        let n = Int(dstBuf.frameLength)
        guard n > 0, let chan = dstBuf.floatChannelData else { return nil }
        return Data(bytes: chan[0], count: n * MemoryLayout<Float>.size)
    }
}
