import Foundation
import AVFoundation

/// Captures microphone audio and resamples it on the fly to the exact PCM the
/// Gemma 4 12B unified audio embedder expects: float32-LE 16 kHz mono. Stopping
/// yields a `Data` blob ready to wrap in a `ChatAudio` — the same format
/// `AudioPreprocessor` produces from files, so the rest of the pipeline is shared.
final class AudioRecorder: ObservableObject {
    @Published private(set) var isRecording = false
    @Published private(set) var duration: TimeInterval = 0   // seconds captured
    @Published private(set) var level: Float = 0             // 0…1 RMS, for a meter

    static let targetSampleRate: Double = 16_000

    private let engine = AVAudioEngine()
    private var converter: AVAudioConverter?
    private let targetFormat = AVAudioFormat(
        commonFormat: .pcmFormatFloat32, sampleRate: AudioRecorder.targetSampleRate,
        channels: 1, interleaved: false
    )!
    private let lock = NSLock()
    private var samples: [Float] = []   // accumulated 16 kHz mono float32

    /// Ask for mic permission (returns the granted state). Safe to call repeatedly.
    static func requestPermission() async -> Bool {
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized: return true
        case .notDetermined:
            return await withCheckedContinuation { cont in
                AVCaptureDevice.requestAccess(for: .audio) { cont.resume(returning: $0) }
            }
        default: return false
        }
    }

    /// Begin capturing. Throws if the engine can't start (no input device, etc.).
    func start() throws {
        guard !isRecording else { return }
        lock.lock(); samples.removeAll(keepingCapacity: true); lock.unlock()
        DispatchQueue.main.async { self.duration = 0; self.level = 0 }

        let input = engine.inputNode
        let inFormat = input.inputFormat(forBus: 0)
        guard inFormat.sampleRate > 0, inFormat.channelCount > 0 else {
            throw NSError(domain: "AudioRecorder", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "No usable audio input device."])
        }
        converter = AVAudioConverter(from: inFormat, to: targetFormat)

        input.installTap(onBus: 0, bufferSize: 4096, format: inFormat) { [weak self] buffer, _ in
            self?.appendBuffer(buffer, inFormat: inFormat)
        }
        engine.prepare()
        try engine.start()
        DispatchQueue.main.async { self.isRecording = true }
    }

    /// Stop capturing and return the recorded PCM (float32-LE 16 kHz mono), or
    /// nil if nothing was captured.
    func stop() -> Data? {
        guard isRecording || engine.isRunning else { return nil }
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        converter = nil
        DispatchQueue.main.async { self.isRecording = false; self.level = 0 }

        lock.lock(); let snapshot = samples; samples.removeAll(); lock.unlock()
        guard !snapshot.isEmpty else { return nil }
        return Self.pcmData(from: snapshot)
    }

    /// Discard an in-progress recording.
    func cancel() { _ = stop() }

    // Audio render thread: resample → accumulate → publish meter/duration.
    private func appendBuffer(_ buffer: AVAudioPCMBuffer, inFormat: AVAudioFormat) {
        guard let converter, buffer.frameLength > 0 else { return }
        let ratio = Self.targetSampleRate / inFormat.sampleRate
        let capacity = AVAudioFrameCount(Double(buffer.frameLength) * ratio) + 1024
        guard let out = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: capacity) else { return }

        // Streaming convert: feed this buffer once, then report no-data-now so the
        // stateful converter keeps its resampler continuity across taps.
        var fed = false
        var err: NSError?
        converter.convert(to: out, error: &err) { _, status in
            if fed { status.pointee = .noDataNow; return nil }
            fed = true; status.pointee = .haveData; return buffer
        }
        guard err == nil, out.frameLength > 0, let ch = out.floatChannelData else { return }
        let n = Int(out.frameLength)

        lock.lock()
        samples.append(contentsOf: UnsafeBufferPointer(start: ch[0], count: n))
        let total = samples.count
        lock.unlock()

        var sumSq: Float = 0
        for i in 0..<n { sumSq += ch[0][i] * ch[0][i] }
        let rms = (n > 0) ? (sumSq / Float(n)).squareRoot() : 0
        DispatchQueue.main.async {
            self.level = min(1, rms * 6)
            self.duration = Double(total) / Self.targetSampleRate
        }
    }

    /// Serialize float samples to little-endian float32 bytes (the wire format).
    /// Pure + testable; shared by `stop()`.
    static func pcmData(from samples: [Float]) -> Data {
        samples.withUnsafeBufferPointer { Data(buffer: $0) }
    }
}
