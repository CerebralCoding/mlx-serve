import Foundation
import AVFoundation

/// A subtle audible "I'm working" cue played while the model is busy but nothing
/// is being spoken (generating the first tokens, or waiting on a tool call mid
/// agent loop). Behind a protocol so the controller's start/stop logic is
/// unit-testable with a fake.
protocol LoadingCue: AnyObject {
    func start()
    func stop()
}

/// Soft sine "blip" with a smooth (click-free) envelope, low volume, repeated
/// every ~1.6 s. Uses its own `AVAudioEngine` for output — independent of the
/// mic capture engine, and the mic is off while we're loading anyway.
final class SystemLoadingCue: LoadingCue {
    private let engine = AVAudioEngine()
    private let player = AVAudioPlayerNode()
    private let buffer: AVAudioPCMBuffer
    private var timer: Timer?
    private var running = false

    init() {
        let sampleRate = 44_100.0
        let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1)!
        buffer = SystemLoadingCue.makeBlip(format: format, sampleRate: sampleRate)
        engine.attach(player)
        engine.connect(player, to: engine.mainMixerNode, format: format)
        engine.mainMixerNode.outputVolume = 0.16   // keep it subtle
    }

    func start() {
        guard !running else { return }
        running = true
        do {
            if !engine.isRunning { try engine.start() }
        } catch { running = false; return }
        player.play()
        playOnce()
        let t = Timer(timeInterval: 1.6, repeats: true) { [weak self] _ in self?.playOnce() }
        RunLoop.main.add(t, forMode: .common)
        timer = t
    }

    func stop() {
        guard running else { return }
        running = false
        timer?.invalidate(); timer = nil
        player.stop()
        engine.pause()
    }

    private func playOnce() {
        guard running else { return }
        player.scheduleBuffer(buffer, at: nil, options: [], completionHandler: nil)
    }

    /// ~160 ms, 396 Hz sine under a Hann window so it fades in/out without clicks.
    private static func makeBlip(format: AVAudioFormat, sampleRate: Double) -> AVAudioPCMBuffer {
        let duration = 0.16, freq = 396.0
        let n = AVAudioFrameCount(sampleRate * duration)
        let buf = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: n)!
        buf.frameLength = n
        let ch = buf.floatChannelData![0]
        let count = Int(n)
        for i in 0..<count {
            let t = Double(i) / sampleRate
            let env = 0.5 - 0.5 * cos(2 * .pi * Double(i) / Double(max(1, count - 1)))
            ch[i] = Float(sin(2 * .pi * freq * t) * env * 0.5)
        }
        return buf
    }
}
